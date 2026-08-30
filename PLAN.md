# Autonomous ML Research Agent — KuaiRand-Pure

**TikTok TechJam 2026, Track 2.** Submission due 1 Sep, 12pm. Budget: ~1 day.

---

## Context

We are building **an agent that does ML research**, not an ML model. The agent reads
the problem, writes code, trains, evaluates, reads its own results, decides what to
try next, and stops when it stops improving. The score on KuaiRand-Pure is *evidence
the agent works*.

Two constraints shape every decision below:

- jk has never built a recommender model. The goal is a system jk can **defend**, not
  the highest score from a system jk can't explain.
- The starter kit is the *agent's* seed, not ours. Editing `data.py` or `baseline.py`
  by hand is a manual intervention — the exact thing scored against us.

---

## Why the design looks like this

### What the rubric actually pays for

| Criterion | Weight | Measures |
|---|---|---|
| Technical Execution | 35% | Hidden-test delta vs baseline **+ robustness** |
| Innovation & Problem Insight | 20% | *"What the agent identified as worth trying and why — not implementation"* |
| Impact & Relevance | 20% | **Autonomy** — counted as *number of manual interventions* |
| Feasibility & Practicality | 15% | Token + wall-clock cost. **Only scored if you beat baseline.** |

**40% (Innovation + Autonomy) is about the loop, not the number.** The hypothesis text
the agent writes into each log entry is a directly scored artifact.

Scoring: `mean(ΔGAUC, ΔnDCG@5)` against baseline (0.6610 / 0.5282), evaluated **once**
on hidden test, using the **validation-best checkpoint at convergence**.

### Three measured facts that drive the architecture

**1. You rank within a user, so a feature is worth exactly what it varies inside one
user's group.** `tab` has a 100× spread in long-view rate globally (0.4% → 48.9%) yet
is constant for 48% of users. This is the mechanism behind the organizers' "user-side
features contribute 0" finding — arithmetic, not an experiment. It also explains why
loss alignment is the top direction: pointwise logloss spends capacity on each user's
*absolute* level, which cancels out of the ranking entirely.

**2. Selection noise is larger than the target.** Seed std is 0.0008; best-of-50 noise
is σ·√(2 ln 50) ≈ **+0.0033**. Since the competition ships the *validation-best*
checkpoint, naive promotion systematically ships a lucky seed. This is the single
biggest threat to a real result.

**3. `long_view` is a threshold on watch time, and `play_time_ms` sits in the same
row.** For positives the median play/duration ratio is 0.98; for negatives, 0.03. The
multi-task direction points the agent straight at this cliff. It must be prevented
structurally, not by instruction.

---

## Architecture — the loop, in order

```
preflight  →  profile  →  [ propose → execute → score → gate → log ]×N  →  ensemble  →  submit
                               ↑                          │
                               └────── critics ←── stall ─┘
```

**1. Preflight (blocking).** Pin the interpreter by absolute path, checksum
`evaluate.py`, reproduce `--model random` → valid 0.4834 ±0.001 and `--model fm` →
0.6016. The loop refuses to start otherwise — an agent loop on a broken harness
produces 50 iterations of confidently-logged garbage.

**2. Profile once.** `data_profile.json` (<1200 tokens, ships in every prompt): group
sizes, within-user variance per feature, per-user history lengths, label drift, aux
signal correlations. Measured, not assumed — the README's "hundreds to thousands of
interactions per user" is actually a median of 31.

**3. Propose.** Claude Opus 5, adaptive thinking. Frozen cached prefix (task spec,
`evaluate.py` verbatim, data profile, prior-knowledge pack) + append-only ledger +
volatile working set. Output contract requires an `axis` and a **predicted delta
before execution** — which makes the verdict calibratable and catches proposals with
no hypothesis behind them.

**4. Execute.** `ast.parse` → smoke run on 1% / 30s → full run → validate output.
Process-group kill, stdout/stderr to files never PIPE, within-user-variance check.
Syntax and smoke failures are free; runtime failures cost an iteration and are
repaired once with the traceback.

**5. Gate.** 3 seeds on every scored candidate; promote on the mean; report mean ± std.
*We never promote on a single sample.* Promotion moves a trunk pointer — nodes are
immutable, so rollback is free and a degraded node can never become a parent.

**6. Escalate on stall.** At **2** consecutive non-improvements (one before the
organizers' N=3 convergence, so the result can still break the streak), fan out to 3
isolated critics in fresh context — objective alignment, validity, unexplored space.
They see the trunk, profile and ledger, **not** the agent's reasoning chain, because a
critic sharing that context just agrees with it. Their output re-enters as ordinary
proposals through the same gate. Max 2 rounds. **A critic is not a human, so this
keeps manual interventions at zero.**

**7. Endgame.** When critique is exhausted, average the per-row scores of the top-k
confirmed nodes. Their errors are partly independent, so noise falls as ~√k — turning
"several things that each worked a little" into one thing that works slightly more.
Logged as its own iteration, through the same gate.

**8. Submit.** Re-execute the promoted node's stored source from disk (never a pickled
model), then `submit.py --check`.

### Two firewalls that are not optional

**Leakage.** `features(split)` returns eval splits with behavioural columns *absent*,
not masked. `aux_targets("train")` exposes them for training only, behind a call whose
name says what they are. Enforced by the type system, not the prompt.

**Test.** Test labels never load during the run; the agent cannot compute a test score,
so it cannot select on one. A separate `score_final.py` is the only path that may read
them, refuses to run unless the run is converged, and runs **once**. Harness
improvements afterward go back to test-blind — a second draw makes the estimate
optimistically biased.

### Prior-knowledge pack

The organizers published both halves — three measured dead ends *and* seven ranked
directions. The agent gets both, as literature, then chooses. Dead ends carry their
**mechanism**, because mechanisms generalize and bare bans decay over 30 turns.

The seven directions are a **search space, not a menu**. The agent runs 10–15
iterations, each building on what already worked; they compose (`FM + listwise loss +
multi-task heads + hour-of-day` is three of seven, stacked). They are also the ledger's
vocabulary — without them, iteration 9 can't know iteration 2 settled the loss question.

We do **not** hard-code listwise softmax. It's ranked #1 by the organizers, and if the
agent misses it, Critic A's entire job is to surface loss misalignment. Forcing it
costs the sentence we most want: *"the agent chose it."*

---

## Not using a framework

No LangChain. The loop is ~200 lines of control flow we fully specify, and three
costs bite: prompt caching needs byte-exact control over the request (our frozen
prefix must be identical across ~30 calls on a 1h TTL, and framework layers silently
break that); "coherent architecture, appropriate boundary" is easier to defend in a
thin loop we wrote; and we do none of what a framework is for — no RAG, no vector
store, no multi-provider routing, no tool-calling chain. Single structured
completion calls against the `anthropic` SDK.

**The agent is code-emitting, not tool-using.** It returns one Python script per
iteration rather than driving a read/edit/bash loop. That is why there is no tool
registry. It is also why the deliverables fall out for free: `candidate_iter_NN.py`
and the per-iteration diff *are* the graded artifacts, which a free-form tool loop
makes awkward to produce cleanly.

---

## Guarding against score-chasing

Two different overfitting risks, with different fixes.

**Statistical — selection noise on valid.** The 3-seed gate, plus the random-exposure
log (`log_random_4_22_to_5_08_pure.csv`, 1.18M rows) as a promotion-time check with a
*different* bias: a candidate that gains on logged traffic but not on random exposure
learned the logging policy, not the ranking.

**Epistemic — hill-climbing the metric without understanding.** Four mechanisms, all
enforceable in the output contract:

1. **The stored EDA is the prior.** `logs/data_profile.json` is built before the loop
   starts and ships in every prompt, so proposals are anchored to measured properties
   rather than to score feedback alone. A fuller `logs/eda_report.md` (superset, for
   humans and the write-up) is written at the same time.
2. **Every proposal must cite a grounding fact** — a named field from the data profile
   that motivates the change. Proposals that cite nothing are rejected before
   execution, at zero compute cost. This is also exactly what Innovation (20%) scores:
   *what the agent identified as worth trying and why*.
3. **Predicted delta before execution**, which makes each iteration a hypothesis test
   rather than a search step.
4. **Calibration is tracked as a metric about the agent itself.** If predicted deltas
   correlate with realised ones, the agent is reasoning; if they don't, it is
   guessing. Reporting that correlation is a genuinely novel thing to put in the
   write-up, and it is nearly free to compute.

Per-iteration we also log **GAUC and nDCG@5 separately**: they weight users
differently (GAUC by positive count, nDCG equally with 36.3% of users fixed), so a
change that moves exactly one of them has a mechanism, while a change that nudges
both slightly is more likely noise.

---

## Build order

| # | Deliverable | Time | Why here |
|---|---|---|---|
| 0 | `preflight.py` | 30m | **done** — 9 checks in 30s; FM reproduces to 0.6015 vs 0.6016 |
| 1 | `data_guard.py` + `profiler.py` + `tests/` | 2h | **done** — firewalls, 1,134-token profile, 23 acceptance tests |
| 1b | `eda_report.md` | 30m | Fuller stored EDA for the write-up (superset of the prompt profile) |
| 2 | `evaluator.py` | 2h | The noise gate. Built before the generator, because a mistake here silently corrupts every result we'd report |
| 3 | `executor.py` | 2h | Sandbox, smoke run, output validation |
| 4 | `memory.py` + `logger.py` | 1.5h | Trunk pointer, insight ledger, `iteration_logs.json` |
| 5 | `prompts.py` + `llm.py` + `agent.py` + `loop.py` + `console.py` | 2.5h | The generator — **last**, because a mistake here is cheap and visible |
| 6 | `critics.py` | 1.5h | Stall escalation + seed ensembling |
| 7 | Run + write-up | 3h | Unattended run, then `score_final.py` once, README, diagram, Devpost |

`prompts.py` holds the system prompt as a versioned constant rather than a string
buried in `agent.py`: it shapes what the agent proposes, Innovation is scored on
exactly that, and the frozen cache prefix must be byte-identical across calls.

**Mock mode is built alongside the loop, not after it.** `--mock` swaps the LLM for a
set of pre-written candidate scripts (one good, one that crashes, one that emits
constant scores, one that attempts leakage), so the executor, evaluator, state
tracker, logger and stall escalation can be exercised end-to-end in seconds with zero
tokens. On a one-day build this is the difference between debugging the loop and
debugging the loop *while* waiting on API calls.

**If time runs short, cut in this order:** unbiased-validation check → seed ensembling
→ the third critic. **Never cut** `data_guard`, the 3-seed gate, or the iteration log.

---

## Git

**Commit:** `harness/`, the unmodified starter kit, `iteration_logs.json`, every
generated `runs/iter_*/solution.py` (evidence of what the agent actually wrote), the
submission CSV, results table, `requirements.txt`, `scripts/get_data.sh`, README +
architecture diagram.

**Never commit:** API keys (check git *history*, not just the tree), the KuaiRand
dataset (~200MB), `.venv/`, `__pycache__/`, `cache/*.npz`, per-iteration checkpoints.

---

## Logging granularity

**One entry per agent iteration** — one turn of the MLE loop, not one training epoch.
If iteration 7 trains 40 epochs across 3 seeds, that is **one** entry reporting mean ±
std. Each carries: hypothesis (*what and why*), code diff, GAUC / nDCG@5, and any error
or recovery events. Plus one run-level count of manual interventions. This log is how
judges read Innovation and Robustness.

---

## Definition of done

| Check | Passes when |
|---|---|
| Noise gate | Evaluator **refuses to promote** FM re-run with a different seed |
| Test firewall | `features("test")` carries no labels; reaching them raises |
| Leakage firewall | A script reading `play_time_ms` on an eval split fails at load |
| Process hygiene | A hung script leaves **no orphan processes** |
| Output validity | Constant-score output is rejected (within-user variance = 0) |
| Crash recovery | A deliberate `NameError` is repaired without costing an iteration |
| End-to-end | `loop.py --max-iters 3` → 3 complete log entries; forced stall fires the critics |
| Submission | `submit.py --check --split test` passes; `--score --split valid` agrees to 1e-6 |

**Submission package:** public repo, README + architecture diagram,
`iteration_logs.json`, all generated solutions, submission CSV, results table with
absolute deltas per metric, and the manual-intervention count — target **0**.
