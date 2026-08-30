# Architecture — Autonomous ML Research Agent

**TikTok TechJam 2026, Track 2.** Submission due 1 Sep, 12pm.

This file is the **generic architecture**: the parts, what each is responsible for,
and the contracts between them. Each part has its own implementation sub-plan under
[`plans/`](plans/), written so it can be picked up independently.

---

## Context

We are building **an agent that does ML research**, not an ML model. It reads the
problem, writes code, trains, evaluates, reads its own results, decides what to try
next, and stops when it stops improving. The score on KuaiRand-Pure is *evidence the
agent works*.

Two constraints shape everything:

- jk has never built a recommender model. The goal is a system jk can **defend**, not
  the highest score from a system jk can't explain.
- The starter kit is the *agent's* seed, not ours. Editing `data.py` or `baseline.py`
  by hand is a manual intervention — the exact thing scored against us.

### What the rubric pays for

| Criterion | Weight | Measures |
|---|---|---|
| Technical Execution | 35% | Hidden-test delta vs baseline **+ robustness** |
| Innovation & Problem Insight | 20% | *"What the agent identified as worth trying and why — not implementation"* |
| Impact & Relevance | 20% | **Autonomy** — counted as *number of manual interventions* |
| Feasibility & Practicality | 15% | Token + wall-clock cost. **Only scored if you beat baseline.** |

**40% is about the loop, not the number.** Scoring is `mean(ΔGAUC, ΔnDCG@5)` vs
baseline (0.6610 / 0.5282), evaluated **once** on hidden test, from the
**validation-best checkpoint at convergence**.

### Three measured facts that drive the design

1. **Ranking is within-user, so a feature is worth exactly what it varies inside one
   user's group.** `tab` spans 0.4%→48.9% watch rate globally yet is constant for 48%
   of users. This is the mechanism behind the organizers' "user-side features give
   zero" finding — arithmetic, not an experiment.
2. **Selection noise exceeds the target.** Seed std 0.0008; best-of-50 noise
   σ·√(2 ln 50) ≈ **+0.0033**. The competition ships the *validation-best* checkpoint,
   so naive promotion systematically ships a lucky seed.
3. **`long_view` is a threshold on watch time and `play_time_ms` is in the same row**
   (median play/duration 0.98 for positives, 0.03 for negatives). Must be prevented
   structurally, not by instruction.

---

## The system

```
                          ┌──────────────────────────────────┐
  L6  ORCHESTRATION       │  loop.py          console.py     │
                          └───────────────┬──────────────────┘
                          ┌───────────────▼──────────────────┐
  L5  GENERATION          │  agent.py   critics.py           │
                          │  prompts.py     llm.py           │
                          └───────────────┬──────────────────┘
              ┌───────────────────────────┼───────────────────────────┐
  L4  MEMORY  │  memory.py  logger.py     │                           │
              └───────────────────────────┤                           │
              ┌───────────────────────────┤              ┌────────────▼────────────┐
  L3  EXECUTION  │  executor.py           │              │  L2  MEASUREMENT        │
                 └────────────────────────┤              │  evaluator.py           │
                                          │              └────────────┬────────────┘
                          ┌───────────────▼───────────────────────────▼──────────┐
  L1  GROUND TRUTH        │ config.py  data_guard.py  preflight.py               │
                          │ profiler.py  eda.py                    tests/        │
                          └──────────────────────────────────────────────────────┘

  TERMINAL (outside the loop, runs once)   score_final.py
```

**Dependencies point strictly downward.** No L1 module imports from L2+; no L2 module
imports from L3+. This is what makes the parts independently implementable and
testable, and it is why the foundation was built first: a mistake in L1 silently
corrupts every number the higher layers report.

### The parts

| Layer | Module | Responsible for | Produces |
|---|---|---|---|
| L1 | `config.py` | Paths, run policy, **which columns are legal on which split** | — |
| L1 | `data_guard.py` | The only data surface the agent gets; both firewalls | `cache/*.npz` |
| L1 | `preflight.py` | Blocking gate: 9 checks incl. reproducing the official baselines | `logs/preflight.json` |
| L1 | `profiler.py` | Prompt-sized measured data facts | `logs/data_profile.json` |
| L1 | `eda.py` | Full human-readable EDA; superset of the profile | `logs/eda_report.md` |
| L1 | `tests/` | Assert every guard **fires**, not that it exists | — |
| L2 | `evaluator.py` | Score, 3-seed promotion gate, convergence, submission build | `outputs/submission.csv` |
| L3 | `executor.py` | Run agent-written code safely; validate its output | `runs/iter_NN/` |
| L4 | `memory.py` | State tree, trunk pointer, insight ledger | `logs/state.jsonl` |
| L4 | `logger.py` | The primary graded deliverable | `logs/iteration_logs.json` |
| L5 | `prompts.py` | System prompt + prior-knowledge pack, versioned | — |
| L5 | `llm.py` | Anthropic client, retries, token/cost accounting, cache placement | — |
| L5 | `agent.py` | Three-tier context assembly; parse + validate proposals | `outputs/candidate_iter_NN.py` |
| L5 | `critics.py` | Stall escalation: 3 isolated reviewers | — |
| L6 | `loop.py` | Orchestrates one iteration and the run | — |
| L6 | `console.py` | Live progress display | — |
| — | `score_final.py` | The **only** code permitted to read test labels | `logs/final_result.json` |

---

## Shared contracts

Every part codes against these. They live in `harness/types.py`, created first so the
sub-plans can be implemented in parallel without drifting.

```python
Split = Literal["train", "valid", "test"]
Axis  = Literal["loss", "sequence", "multitask", "watchtime",
                "architecture", "temporal", "debias", "ensemble"]

class Verdict(Enum):     KEEP; DISCARD; INCONCLUSIVE
class NodeStatus(Enum):  PENDING; SCORED; PROMOTED; FAILED; PRUNED; QUARANTINED
class FailureClass(Enum): SYNTAX; CONTRACT; SMOKE; RUNTIME; TIMEOUT; OOM;
                          INVALID_OUTPUT; NONDETERMINISTIC

@dataclass(frozen=True)
class Metrics:            # one scored evaluation, possibly averaged over seeds
    gauc: float; ndcg5: float; primary: float
    users: int; rows: int
    seeds: tuple[int, ...] = (42,)
    primary_std: float = 0.0

@dataclass(frozen=True)
class Proposal:           # the agent's structured output contract
    hypothesis: str       # what and WHY -- directly scored under Innovation
    axis: Axis
    grounding: str        # a named field from data_profile.json; resolved fuzzily
    predicted_delta: float    # required -- absent means there is no hypothesis
    code: str

@dataclass
class Node:               # one attempt; immutable once scored
    node_id: str; parent_id: str | None; iteration: int
    proposal: Proposal; status: NodeStatus
    code_path: str; code_sha256: str
    valid: Metrics | None = None
    grounding_verified: bool = True
    change_summary: str = ""   # one line; the raw diff lives beside it in the log
    failures: list[FailureRecord] = field(default_factory=list)
    resources: ResourceFacts | None = None
    tokens: TokenUsage | None = None

@dataclass(frozen=True)
class Insight:            # ledger entry, keyed by (axis, technique)
    axis: Axis; technique: str; verdict: Verdict
    delta_primary: float; delta_gauc: float; delta_ndcg5: float
    n_seeds: int; mechanism: str      # WHY -- mechanisms generalise, bans decay
```

**`grounding` and `predicted_delta` are the anti-score-chasing mechanism.** Each
proposal names a measured fact that motivates it and predicts its own effect, so an
iteration is a hypothesis test rather than a search step. Predicted-vs-realised
correlation is then tracked as a *metric about the agent* — reasoning versus guessing.

Grounding resolution is **advisory, not blocking**: exact match, then fuzzy match
(`difflib`, cutoff 0.6) to absorb paraphrase, and if still unresolvable we record
`grounding_verified=False`, warn, and run it anyway. Refusing to run costs a real
iteration to punish a spelling mistake; the flag makes the rate visible instead.
A missing `predicted_delta` **is** still rejected pre-execution — without it there is
no hypothesis to test.

---

## Cross-cutting rules

### Two firewalls (non-negotiable)

**Leakage.** Post-impression outcome columns are materialised for `train` only —
absent from evaluation-split arrays, not masked within them.

**Test.** Test labels never load during the run; `labels("test")` raises. They sit in
`cache/_holdout/`, which nothing in the agent's prompt or import path names. Only
`score_final.py` reads them, once, after convergence. Harness improvements afterward
go back to test-blind: a second draw makes the estimate optimistically biased.

Additionally `video_features_statistic_pure.csv` is quarantined — dataset-wide
aggregates of play/completion behaviour spanning the test window, disguised as an
ordinary item-features file.

### No framework

No LangChain. Prompt caching needs byte-exact request control (a frozen prefix
identical across ~30 calls on a 1h TTL, which framework layers silently break); a thin
loop is easier to defend under "coherent architecture, appropriate boundary"; and we
do none of what a framework is for — no RAG, no vector store, no tool-calling chain.

**The agent is code-emitting, not tool-using.** One Python script per iteration, not a
read/edit/bash loop. Hence no tool registry — and the graded deliverables
(`candidate_iter_NN.py`, the per-iteration diff) fall out for free.

### Guarding against score-chasing

- **Statistical:** the 3-seed gate, plus the random-exposure log (1.18M rows) as a
  promotion-time check with a *different* bias — a candidate that gains on logged
  traffic but not on random exposure learned the logging policy, not the ranking.
- **Epistemic:** the stored EDA is built *before* the loop and ships in every prompt;
  `grounding` must cite it; `predicted_delta` makes each iteration a hypothesis test;
  calibration is reported. GAUC and nDCG@5 are logged separately, since a change that
  moves exactly one has a mechanism while one that nudges both is likelier noise.

### Prior-knowledge pack

The organizers published three measured dead ends *and* seven ranked directions. The
agent gets both, as literature, then chooses. Dead ends carry their **mechanism**.

The seven directions are a **search space, not a menu** — the agent runs 10–15
iterations and they compose (`FM + listwise loss + multi-task heads + hour-of-day` is
three of seven, stacked). They are also the ledger's vocabulary.

We do **not** hard-code listwise softmax: it is ranked #1 in the pack, and Critic A's
entire job is to surface loss misalignment if the agent misses it. Forcing it costs
the sentence we most want — *"the agent chose it."*

---

## Build order and sub-plans

| Phase | Parts | Sub-plan | State |
|---|---|---|---|
| 0 | `preflight.py` | — | **done** — 9 checks/30s; FM 0.6015 vs 0.6016 |
| 1 | `config` `data_guard` `profiler` `tests` | — | **done** — 23 tests, 1,134-token profile |
| 1b | `eda.py` | [`plans/P1b-eda.md`](plans/P1b-eda.md) | |
| 2 | `evaluator.py` | [`plans/P2-evaluator.md`](plans/P2-evaluator.md) | next |
| 3 | `executor.py` | [`plans/P3-executor.md`](plans/P3-executor.md) | |
| 4 | `memory.py` `logger.py` | [`plans/P4-memory-logger.md`](plans/P4-memory-logger.md) | |
| 5 | `prompts` `llm` `agent` `loop` `console` | [`plans/P5-agent-loop.md`](plans/P5-agent-loop.md) | |
| 6 | `critics.py` + ensembling | [`plans/P6-critics.md`](plans/P6-critics.md) | |
| 7 | `score_final.py` + write-up | [`plans/P7-submission.md`](plans/P7-submission.md) | |

**Mock mode is built with Phase 5, not after it.** `--mock` swaps the LLM for four
pre-written candidates (good / crashes / constant scores / attempts leakage), so L2–L6
can be exercised end-to-end in seconds with zero tokens.

**If time runs short, cut in this order:** unbiased-validation check → seed ensembling
→ the third critic. **Never cut** `data_guard`, the 3-seed gate, or the iteration log.

---

## Definition of done

| Check | Passes when |
|---|---|
| Noise gate | Evaluator **refuses to promote** FM re-run with a different seed |
| Test firewall | `labels("test")` raises; cache holds no test label |
| Leakage firewall | A script reading `play_time_ms` on an eval split fails at load |
| Process hygiene | A hung script leaves **no orphan processes** |
| Output validity | Constant-score output is rejected (within-user variance = 0) |
| Crash recovery | A deliberate `NameError` is repaired without costing an iteration |
| End-to-end | `loop.py --mock --max-iters 5` → 5 log entries; forced stall fires critics |
| Submission | `submit.py --check --split test` passes; `--score --split valid` agrees to 1e-6 |

**Submission package:** public repo, README + architecture diagram,
`iteration_logs.json`, all generated solutions, `submission.csv`, results table with
absolute deltas per metric, and the manual-intervention count — target **0**.

### Git

**Commit:** `harness/`, unmodified starter kit, `logs/*.json`, every generated
`candidate_iter_NN.py`, `submission.csv`, `requirements.txt`, `scripts/get_data.sh`,
README + diagram.
**Never commit:** API keys (check *history*, not just the tree), the dataset (~200MB),
`.venv/`, `__pycache__/`, `cache/`, `runs/`.
