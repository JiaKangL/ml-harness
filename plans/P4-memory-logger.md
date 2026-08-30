# P4 — `harness/memory.py` + `harness/logger.py` (L4, Memory)

**Purpose.** Remember what was tried, which attempt is currently best, and what has
been ruled out — and write the run's primary graded deliverable.

**Depends on:** L1, L2. **Must not import:** agent, critics, loop.

---

## `memory.py` — state tracker

**Tree with a greedy trunk.** Nodes are immutable once scored. The trunk is a
*pointer* to the best confirmed node; promotion moves the pointer and never mutates a
node. Rollback is therefore free, and a degraded node can never become a parent —
which is the whole reason not to use a linear lineage, where one subtly-wrong
non-crashing edit becomes the substrate for everything after it.

```python
class StateTree:
    def add(self, node: Node) -> str: ...          # append-only JSONL + fsync
    def update(self, node_id: str, **fields) -> Node: ...
    def trunk(self) -> Node: ...                   # current best confirmed
    def promote(self, node_id: str) -> None: ...   # moves the pointer only
    def select_parent(self, axis: Axis) -> Node: ...   # ~60% trunk / 40% explore
    def prune_subtree(self, node_id: str, reason: str) -> int: ...
    def resume(self) -> None: ...                  # exact replay from JSONL
```

Persistence is append-only JSONL with fsync per node: a multi-hour run **will** be
interrupted, and resume must be exact.

### `FeatureInsightsLedger`

Keyed by `(axis, technique)`, not by "feature". Seeded at construction with the
organizers' three published dead ends, each carrying its **mechanism** — mechanisms
generalise, bare prohibitions decay over 30 turns.

**`INCONCLUSIVE` is a first-class verdict** alongside KEEP and DISCARD. At σ≈0.001
most single-run deltas genuinely are inconclusive; forcing a binary verdict
manufactures false knowledge, which then propagates into every later prompt. That is
the worst failure a memory system can have.

`render(max_tokens)` produces the prompt's tier-B block: one line per experiment,
~30 tokens each, append-only so the cached prefix survives.

---

## `logger.py` — the deliverable

`logs/iteration_logs.json` is **graded output, not debug output**. Judges read
Innovation and Robustness directly from it.

**One entry per agent iteration** — one turn of the loop, not one training epoch. If
iteration 7 trains 40 epochs across 3 seeds, that is *one* entry reporting mean ± std.

Each entry carries:

- `hypothesis` — what and **why** (scored under Innovation)
- `axis`, `grounding` (the cited data-profile fact), `grounding_verified` (bool),
  `predicted_delta` vs realised
- **`change_summary`** — one line, e.g. *"Rewrote numpy FM as PyTorch DeepFM with
  pairwise BPR"* — plus **`diff`**, the raw unified diff against the parent

### Two diff consumers, two representations

When the agent switches from numpy FM to PyTorch DeepFM, `difflib` emits a ~300-line
diff. Store **both** forms, because they serve different readers:

| Consumer | Gets | Why |
|---|---|---|
| `iteration_logs.json` (judges, parsers) | `change_summary` **and** full `diff` | The summary keeps the log scannable; the raw diff is the evidence and must not be truncated in a graded deliverable |
| The next prompt (tier C) | `change_summary` only, plus the diff if under a line budget | Context is a budget; a 300-line diff of code the agent just wrote adds nothing it does not already know |

A rewrite is detected by diff size relative to the parent (say >60% of lines changed)
and labelled as a rewrite rather than an edit, so the ledger can distinguish
"modified the loss" from "replaced the model".
- `metrics` — GAUC and nDCG@5 **separately**, both splits, with seed count and std
- `errors` — every recovery event: class, signature, repair attempts, resolution
- `resources` — wall time per stage (generation / smoke / each seed), peak RSS
- `tokens` — **`prompt_tokens`, `completion_tokens`**, `cache_read`, `cache_write`,
  and derived `cost_usd`, per call and cumulative for the run

### Resource metering is scored

Feasibility & Practicality is 15% of the rubric and is graded on *total token
consumption and agent wall-clock to reach the converged result*. That makes metering a
deliverable, not telemetry. `llm.py` must extract usage off **every** API response —
including repair calls and critic calls, which are easy to forget and are exactly the
calls that inflate the total. `logger.py` records per-step and cumulative figures, and
the run summary reports the totals the rubric asks for.

Plus one run-level summary: **manual intervention count** (target 0) and the
predicted-vs-realised calibration correlation.

Write append-only JSONL during the run; render the JSON array at the end. Every write
atomic (temp + `os.replace`) so a kill at hour 5 never truncates the deliverable.

### Contract

```python
@dataclass(frozen=True)
class LogEntry:
    iteration: int
    node_id: str
    parent_id: str | None
    status: str                     # NodeStatus.value
    hypothesis: str
    axis: str
    grounding: str
    grounding_verified: bool
    predicted_delta: float
    realised_delta: float | None    # vs the parent, computed by the logger
    change_summary: str
    diff: str                       # full unified diff; never truncated here
    is_rewrite: bool
    metrics: dict | None            # gauc / ndcg5 / primary / std / seeds
    per_seed: list[dict]
    gate: dict | None               # promote, reason, quarantined, deltas
    errors: list[dict]              # every repair attempt, not just the last
    resources: dict                 # per-stage wall time, peak RSS
    tokens: dict                    # prompt/completion/cache, cost, latency
    source: str = "agent"           # "agent" | "critic:A" | "ensemble"

class IterationLogger:
    def __init__(self, jsonl: Path, json_out: Path): ...
    def log(self, entry: LogEntry) -> None: ...        # atomic append
    def finalize(self) -> RunSummary: ...              # renders the JSON array
    def resume(self) -> int: ...                       # last iteration on disk

@dataclass(frozen=True)
class RunSummary:
    iterations: int
    manual_interventions: int       # target 0 -- scored under Autonomy
    total_tokens: TokenUsage
    wall_clock_seconds: float
    best_valid: dict
    calibration_r: float | None     # predicted vs realised delta correlation
    converged: bool
    convergence_iteration: int | None
```

`source` exists so a critic-originated proposal is distinguishable in the log -- P6's
acceptance test needs it and `Proposal` has no origin field.

## Acceptance tests — `tests/test_memory_logger.py`

| Test | Passes when |
|---|---|
| Immutability | A scored node's metrics cannot be overwritten |
| Rollback | A regressed candidate leaves the trunk pointer unchanged |
| No degraded parent | `select_parent` never returns a FAILED or PRUNED node |
| Crash-safe resume | Truncating the JSONL mid-line still resumes to the last valid node |
| Ledger tri-state | A +0.001 single-seed result records INCONCLUSIVE, not KEEP |
| Dead ends seeded | The three published dead ends are present with mechanisms at init |
| Atomic log write | A kill during `log()` leaves the previous file intact |
| Log schema | Every entry has hypothesis, change_summary, diff, both metrics, errors, tokens |
| Rewrite labelled | A >60%-changed diff is recorded as a rewrite, not an edit |
| Token accounting | Repair and critic calls are included in the cumulative total |
| Cumulative metering | Run summary reports total prompt+completion tokens and wall-clock |
| Calibration computed | RunSummary.calibration_r is populated from >= 3 scored entries |
| Critic origin visible | A critic-originated entry is distinguishable via `source` |
| Realised delta | Computed against the parent node, not left to the caller |
