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
- `axis`, `grounding` (the cited data-profile fact), `predicted_delta` vs realised
- unified `diff` against the parent node
- `metrics` — GAUC and nDCG@5 **separately**, both splits, with seed count and std
- `errors` — every recovery event: class, signature, repair attempts, resolution
- `resources` — wall time, peak RSS; `tokens` — in/out/cached, cost

Plus one run-level summary: **manual intervention count** (target 0) and the
predicted-vs-realised calibration correlation.

Write append-only JSONL during the run; render the JSON array at the end. Every write
atomic (temp + `os.replace`) so a kill at hour 5 never truncates the deliverable.

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
| Log schema | Every entry has hypothesis, diff, both metrics, errors, tokens |
