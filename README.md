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
./.venv/bin/python -m unittest discover -s tests -t .     # 23 tests, ~2s
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

| Module | What it does | Why it exists |
|---|---|---|
| `harness/config.py` | All paths, settings, and the list of which data columns are legal to use when | One place to change anything; the column list *is* the safety rule |
| `harness/data_guard.py` | Loads the data and hands the agent only what it's allowed to see | Stops the agent cheating by accident — see below |
| `harness/preflight.py` | 9 checks that must pass before the loop may start | A broken harness doesn't fail loudly, it produces 50 confident wrong answers |
| `harness/profiler.py` | Measures the dataset once, writes `logs/data_profile.json` | The agent's background knowledge, so it reasons from facts about the data rather than guessing |
| `tests/` | 23 tests, run in 2 seconds | Each one is a specific way this project could produce a confident wrong answer |

**Preflight** reproduces the organizers' own sanity checks: random guessing scores
0.4827 (expected 0.4834) and their Factorization Machine scores 0.6015 (expected
0.6016). It also verifies our data rows are in *exactly* the same order as theirs —
the submission file identifies rows by position, so a reordering would misalign every
number we submit while everything still looked fine.

### The two safety rules

**The agent must not see the answer.** Whether a video was watched is decided by how
long it was played, and play time sits in the same row of the log. If the agent used
play time as an input it would score near-perfectly and be worthless. So the columns
describing what the user *did* (play time, clicks, likes) only exist for the training
period. On the evaluation data they aren't hidden — they aren't loaded at all.

**The agent must not see the test set.** The final score comes from data the agent is
never allowed to look at. Its labels are stored separately, outside everything the
agent is pointed at, and only one script may read them — once, at the very end.

One file, `video_features_statistic_pure.csv`, is excluded entirely. It looks like an
ordinary "facts about each video" file, but its columns are totals counted across the
*whole* time period, including the test window. Using it would leak both the answer
and the future.

---

## What's left to build

| Module | What it will do | Why it's needed |
|---|---|---|
| `harness/evaluator.py` | Score a candidate, decide whether it really beat the best, detect when we've stopped improving | Without it we can't tell a real improvement from luck |
| `harness/executor.py` | Run agent-written code safely: timeout, memory cap, kill stray processes, reject nonsense output | Generated code crashes and hangs; that has to be survivable, not fatal |
| `harness/memory.py` | Track every attempt, which is currently best, and what's been ruled out | Otherwise iteration 9 has no idea what iteration 2 already settled |
| `harness/logger.py` | Write `logs/iteration_logs.json` | This is the main graded deliverable |
| `harness/prompts.py` | The agent's instructions, kept in one versioned file | Judges score *what the agent chose to try and why*, which this shapes directly |
| `harness/llm.py` | Talk to the model; count tokens and cost | Keeps the network layer separate so everything else is testable |
| `harness/agent.py` | Assemble what the agent sees each turn | Its context must stay useful at turn 30, not just turn 3 |
| `harness/critics.py` | The three reviewers that fire when progress stalls | Lets the run rescue itself instead of asking a human — asking a human costs marks |
| `harness/console.py` | The live progress display | It's what we demo |
| `harness/loop.py` | Runs everything in order | The actual agent |
| `score_final.py` | The one script permitted to read test labels | Runs once, after we've finished |

### Also planned

- **Mock mode** — a fake model that returns four pre-written scripts (one good, one
  that crashes, one that outputs rubbish, one that tries to cheat). Lets us test the
  entire loop in seconds without spending money or waiting. Built alongside the loop,
  not after it.
- **`logs/eda_report.md`** — a fuller, human-readable version of the data profile, for
  the write-up.
- **Grounding requirement** — every proposal must point at a specific measured fact
  about the data that motivates it. Proposals that cite nothing get rejected before
  they run, which costs nothing and stops the agent from blindly chasing the score.

---

## Status

| Phase | State |
|---|---|
| 0 — preflight gate | done |
| 1 — data guard, profiler, tests | done |
| 1b — fuller EDA report | |
| 2 — evaluator | next |
| 3 — executor | |
| 4 — memory + logger | |
| 5 — prompts, llm, agent, loop, console | |
| 6 — critics + ensembling | |
| 7 — run + write-up | |
