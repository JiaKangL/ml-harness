# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026, Track 2.

An agent that does machine-learning research on its own. It reads the problem, writes
training code, runs it, reads the crash if it crashes, fixes it, checks whether the
score improved, decides what to try next, keeps notes, and stops when it stops
improving — with no human in the loop.

The task it works on is ranking: for each user, sort the videos they were shown so
that the ones they actually watched come first. The organizers' Factorization Machine
baseline scores **0.5946**. Beating that is how we show the agent works.

See [PLAN.md](PLAN.md) for the full design and the reasoning behind each decision.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./scripts/get_data.sh                                    # 47 MB from Zenodo
./.venv/bin/python -m harness.preflight                  # safety gate, ~30s
./.venv/bin/python -m harness.profiler                   # writes logs/data_profile.json
./.venv/bin/python -m harness.eda                        # writes logs/eda_report.md
./.venv/bin/python -m unittest discover -s tests -t .     # the full acceptance suite
```

Then run the agent:

```bash
export ANTHROPIC_API_KEY=...
./.venv/bin/python -m harness.loop --mock --max-iters 5   # no API calls, ~2 min
./.venv/bin/python -m harness.loop --max-iters 15         # the real run
./.venv/bin/python score_final.py                         # once, after it converges
```

---

## How it works

One **iteration** is one turn of the research loop:

```
          ┌──────────────────────────────────────────────────┐
          │                                                  │
  propose ──▶ execute ──▶ score ──▶ good enough? ──▶ log ─────┘
     ▲                                   │
     │                                   ▼ no, and stuck
     └────────────── critics ◀───────────┘
```

1. **Propose** — the agent reads what's been tried so far and writes a new training
   script, stating a hypothesis and predicting how much it will help.
2. **Execute** — we run that script in a separate process, with a timeout and a
   memory cap, and check the output isn't nonsense.
3. **Score** — we measure it on the validation set. **Three times, with three
   different random seeds.**
4. **Decide** — if the average genuinely beats the current best, it becomes the new
   best. Otherwise it's discarded and the next attempt starts from the old best.
5. **Log** — hypothesis, code change, scores, and any crashes-and-fixes get written
   to `logs/iteration_logs.json`. That file is the main thing judges read.
6. **When it gets stuck** — after two iterations with no improvement, three
   independent critic agents look at the code fresh and suggest what's being missed.
   Their ideas go back in as normal proposals. No human is asked.

### Two ideas the whole design rests on

**Why three seeds.** Training the same model twice with different random starts gives
slightly different answers. We measured that spread: **0.00035** on validation, so
picking the best of 50 attempts gains about **+0.0010** by luck alone. That is smaller
than we first assumed — but it is not why we run three seeds. We run three because
that spread was measured on *one* model; a different model can be far less stable, and
one run tells you nothing about how stable it is. Three runs give an average and a
spread, so we can tell "this genuinely helped" from "this got lucky once".

**Why a feature can be predictive and still worthless.** We rank *within* a user, so
only differences *inside one person's list* matter. Anything identical across all of
one user's videos cancels out completely. Example: `tab` (which part of the app the
video appeared in) ranges from a 0.4% to a 49% watch rate — hugely predictive
overall — but it's the same value for about half of users, so for those users it
changes nothing. This is arithmetic, not a guess, and it explains why the organizers
found several obvious-looking ideas gave exactly zero gain.

---

## What's built

Six layers, dependencies pointing strictly downward. A mistake in the bottom layer
does not crash — it silently corrupts every number the layers above it report — which
is why the foundation was built first and why each layer's tests assert that its
guards *fire*, not that they exist.

| Layer | Module | What it does |
|---|---|---|
| L1 | `harness/config.py` | Paths, run policy, and which data columns are legal on which split |
| L1 | `harness/data_guard.py` | The only data surface the agent gets; both firewalls |
| L1 | `harness/preflight.py` | Nine blocking checks, including reproducing the organizers' own baselines |
| L1 | `harness/profiler.py` | The prompt-sized measured data profile |
| L1 | `harness/eda.py` | The full human-readable EDA; superset of the profile |
| L1 | `harness/holdout.py` | The sole path to a test label, behind a seal |
| L2 | `harness/evaluator.py` | Scoring, the seed ladder, the promotion gate, convergence, the submission |
| L3 | `harness/executor.py` | Runs agent-written code under a timeout, an RSS cap and its own process group; validates the output |
| L4 | `harness/memory.py` | The state tree with a greedy trunk, and the insight ledger |
| L4 | `harness/logger.py` | `logs/iteration_logs.json` — the primary graded deliverable |
| L5 | `harness/prompts.py` | The versioned prompt, and the frozen cache prefix |
| L5 | `harness/llm.py` | The Anthropic client, token and cost accounting, and mock mode |
| L5 | `harness/agent.py` | Context assembly, proposal parsing, and everything rejected before execution |
| L5 | `harness/critics.py` | Three isolated reviewers, and the endgame ensemble |
| L6 | `harness/loop.py` | The orchestrator |
| L6 | `harness/console.py` | The live display |
| — | `score_final.py` | The only code permitted to read test labels. Nothing in `harness/` imports it. |

**Preflight** reproduces the organizers' own sanity checks: random guessing scores
0.4827 (expected 0.4834) and their Factorization Machine scores 0.6015 (expected
0.6016). It also verifies our data rows are in *exactly* the same order as theirs —
the submission file identifies rows by position, so a reordering would misalign every
number we submit while everything still looked fine.

The run's **root node** is that same FM baseline, ported to the data guard's API and
executed through the same executor, the same seeds and the same scorer as every
candidate (`harness/seeds/baseline_fm.py`, 0.6024 on validation). It is the incumbent
the agent has to beat, so it has to be measured the same way — otherwise the first
promotion is a comparison between two different measurement procedures.

### The two safety rules

**The agent must not see the answer.** Whether a video was watched is decided by how
long it was played, and play time sits in the same row of the log. If the agent used
play time as an input it would score near-perfectly and be worthless. So the columns
describing what the user *did* (play time, clicks, likes) only exist for the training
period. On the evaluation data they aren't hidden — they aren't loaded at all.

**The agent must not see the test set.** The final score comes from data the agent is
never allowed to look at — and we do not merely hide those labels, we **never write
them down**. Nothing in `cache/` contains a test answer, so there is no file for a
stray line of code to stumble onto.

They are read exactly once, at the very end, straight from the organizer's original
file, and only after the run has been *sealed* — a marker written when the loop
finishes that records which attempt won. Asking for them before that raises an error.
Asking a second time also raises, because scoring the held-out set twice quietly
flatters the result; you can override it, and the override is written into a log that
records every time test was ever touched.

Being precise about the limit: the organizer's original CSV still contains those
labels, and a program running as you could parse it. Truly preventing that needs a
separate user account or a container, which is beyond this build. What we can say
without qualification is that the harness never creates a copy, never holds one in a
process that makes decisions, and cannot reach one before the run is over.

One file, `video_features_statistic_pure.csv`, is excluded entirely. It looks like an
ordinary "facts about each video" file, but its columns are totals counted across the
*whole* time period, including the test window. Using it would leak both the answer
and the future.

---

## What gets rejected before anything runs

A rejection here costs no compute, so the bar is "this cannot teach us anything",
not "this looks unpromising".

| Rejected | Why |
|---|---|
| No `predicted_delta` | Without a prediction the iteration is a search step, not a hypothesis test |
| An axis outside the closed set of eight | The axis is also the ledger's key; one outside it cannot be looked up later |
| `architecture` before the four priority axes have a result | A model swap is the organizers' fifth-ranked direction and the most expensive move available |
| A diff touching only the feature list, or only an embedding dimension | Both are measured dead ends with published counter-evidence |
| A technique the ledger already settled as KEEP or DISCARD | `INCONCLUSIVE` deliberately does *not* block — most single-run deltas honestly are |
| Network access, a subprocess, the raw CSV tree, `harness.holdout` | Static AST lint, by call target rather than by argument |

Two things are deliberately **not** blocking. An unresolvable `grounding` citation is
recorded as `grounding_verified: false` and runs anyway — refusing would spend a real
iteration punishing a spelling mistake, and the flag reports the rate honestly instead.
And the random-exposure cross-check is advisory: it carries a *different* bias, not
less noise, so treating a null result there as disqualifying would discard real gains.

---

## The three things worth leading with

**1. The noise gate, and the fact that we measured it.** We assumed the seed spread was
σ≈0.0012, measured σ=0.00035 over five seeds, and found our own headline claim was 3.3×
overstated. The three-seed gate stayed, but for a corrected reason: that spread was
measured on *one* model, a torch model with random initialisation can be far less
stable, and one run gives no estimate of stability at all. Reporting the correction is
a stronger claim than the original would have been.

**2. The within-user variance lens.** A feature is worth exactly what it varies inside
one user's impression group. `tab` spans a 2%-to-46% watch rate globally and is
constant for 48% of users, so half its apparent predictive power cannot reach the
ranking at all. This is arithmetic, not an experiment, and it explains the organizers'
published dead ends mechanistically rather than as a list of things that did not work.

**3. Autonomous stall escalation.** At two consecutive non-improvements — one before
formal convergence, so the rescue can still land in time — three critics review the run
in *fresh* contexts. They see the code, the data profile and the ledger's verdicts, but
never the agent's own reasoning, because a reviewer who reads the reasoning agrees with
it. Their proposals re-enter through the same three-seed gate as any other. The
intervention count stays at zero.

We also report **calibration**: the correlation between predicted and realised deltas
across the run. Positive means the agent is reasoning; flat means it is guessing.
Reporting it honestly either way is stronger than not measuring it.

---

## Status

| Phase | State |
|---|---|
| 0 — preflight gate | done |
| 1 — data guard, profiler, tests | done |
| 1b — fuller EDA report | done |
| 2 — evaluator | done |
| 3 — executor | done |
| 4 — memory + logger | done |
| 5 — prompts, llm, agent, loop, console, mock mode | done |
| 6 — critics + ensembling | done |
| 7 — `score_final.py` | done |
| 7 — the run itself, and the results table | pending |
