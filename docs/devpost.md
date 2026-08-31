# Autonomous ML Research Agent — KuaiRand-Pure

## Inspiration

The brief asks for a better recommender. We read it as asking for something else: a
system that *finds* one. Anybody can hand-tune a factorization machine for a weekend;
the interesting artifact is the loop that does the tuning, states why it tried what it
tried, and knows when to stop.

That reframing is also the honest one. Neither of us had built a recommender before.
Given a choice between a high score from a system we couldn't explain and a defensible
system whose reasoning we could show, the second is worth more — and the rubric agrees:
40% of it is about the loop, not the number.

## What it does

One iteration is one turn of a research loop:

1. **Propose.** The agent reads the measured data profile, the trunk source, and a
   ledger of what has already been settled, then writes a complete training script —
   with a hypothesis, a named data fact that motivates it, and a *predicted* effect
   size.
2. **Reject, for free.** Syntax, a static contract lint, and known dead ends are caught
   before anything executes. A rejection costs no compute, so the bar for rejecting is
   "this cannot teach us anything", not "this looks unpromising".
3. **Smoke.** 1% of *users* — never rows, because the metric is computed inside a
   user's impression group and a row sample shreds those groups.
4. **Score, on a ladder.** Seed 42 first; a clear regression is pruned there. Survivors
   run two more seeds and are promoted only on the three-seed mean.
5. **Remember.** The result becomes a ledger entry keyed by `(axis, technique)` with
   its verdict — `KEEP`, `DISCARD`, or `INCONCLUSIVE` — and a full log entry.
6. **Escalate, without a human.** Two non-improving iterations trigger three critics in
   fresh contexts. Their proposals re-enter through the same gate.

Nobody edits the agent's code. Manual interventions: **0**.

## Results

**Hidden test set, read exactly once, after the run was sealed:**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| FM baseline (official) | 0.6610 | 0.5282 | 0.5946 |
| This agent | **0.6632** | **0.5306** | **0.5969** |
| **Δ** | **+0.0022** | **+0.0024** | **+0.0023** |

`score_dataset` = mean of per-metric deltas = **+0.0023**

8 iterations to convergence · 133K tokens (34K in / 99K out, 64% served from cache) ·
26 minutes wall clock · **$1.07** · **0 manual interventions** · 0 crashes

The winning candidate came from **Critic A**, not the main agent: a within-user pairwise
BPR loss, proposed after the run stalled twice. Validation +0.0012 ± 0.0001 across three
seeds; the test delta is *larger* than the validation delta, which is the direction you
want — it did not overfit the split it was selected on.

Predicted-vs-realised calibration: **r = −0.40**. Reported because we said we would.
The agent's *mechanisms* were consistently right — the listwise-softmax and BPR
arguments about within-group invariance are correct — but its *effect sizes* were not:
it predicted +0.006 and got −0.003. It reasons well and estimates badly, and over nine
iterations that is what the number says.

## The three things worth leading with

**1. We measured our own noise floor, and it made our headline claim smaller.**
We assumed the seed-to-seed spread was σ≈0.0012. Measured over five seeds it was
σ=0.00035 — our claim had been 3.3× overstated. The three-seed gate stayed, but for a
corrected reason: that spread was measured on *one* model, a torch model with random
initialisation can be far less stable, and a single run gives no estimate of stability
at all. Reporting the correction is a stronger claim than the original would have been.

**2. A feature is worth exactly what it varies inside one user's group.**
Ranking happens within a user, so any term constant across that user's impressions
leaves their ordering unchanged — exactly, not approximately. `tab` spans a 2%-to-46% watch
rate across the validation split yet varies inside only 48% of impression groups — so
for the other 52% of users it moves nothing, and half its apparent predictive power
cannot reach the ranking at all. This one piece of arithmetic explains the
organizers' published dead ends mechanistically instead of as a list of things that
didn't work, and it is the first thing the agent is told.

**3. The headroom is small, and we measured that too.**
Ranking validation by item popularity alone — no user model whatsoever — scores 0.5813
primary against FM's 0.6016. *All* of FM's personalisation is worth +0.028 GAUC. That
is the size of the prize, and it is why the promotion bar is +0.002 rather than
something that sounds more impressive.

## How we built it

Six layers, dependencies pointing strictly downward, foundation first. A mistake in the
bottom layer doesn't crash — it silently corrupts every number the layers above report,
so the ground truth was built and tested before anything that consumes it. Preflight is
ten blocking checks that reproduce the organizers' own baselines (random 0.4827 vs
0.4834; FM 0.6015 vs 0.6016) and verify our row order matches theirs byte for byte,
because the submission identifies rows by position.

**No framework.** Prompt caching needs byte-exact request control — a frozen prefix
identical across ~30 calls on a 1-hour TTL — and framework layers silently break that.
We also do none of what a framework is for: no RAG, no vector store, no tool-calling
chain. The agent is *code-emitting*, not tool-using: one complete script per iteration,
which means the graded deliverables fall out for free.

**Two firewalls, structural rather than instructed.** `long_view` is a threshold on
watch time and `play_time_ms` sits in the same row — a single hand-written threshold on
`play_time/duration` reproduces the label 88% of the time. So outcome columns are
materialised for `train` only: absent from the evaluation arrays, not masked within
them. And test labels are never written anywhere; they are parsed from the raw log once,
after the run is sealed, with every draw recorded permanently.

## Challenges we ran into

Every one of these was found by *running* the thing, not by reading it.

- **Three unbounded loops.** A turn whose proposals were all rejected spent no budget. A
  turn that failed at smoke spent no budget. Both leave the next turn's state identical,
  so each was an infinite loop that would have burned the entire token budget on a live
  run before anyone noticed.
- **Ctrl-C leaked the child process.** The banner promised the run was interruptible.
  It wasn't: the interrupt unwound past the supervision loop and the candidate — which
  leads its own session precisely so the harness can kill it as a group — kept running,
  stealing a core and corrupting every wall-clock number measured afterwards.
- **A clean clone failed preflight.** The quarantine check reads the cache to prove the
  excluded columns never reached it, and it ran *before* the cache was built. It only
  ever passed because a developer's cache already existed — invisible until somebody
  clones the repository.
- **A check that never checked anything.** The random-exposure validation cached only
  nodes that had themselves been promoted, so the parent was never present and the
  comparison — the entire point — was silently never computed. It cost a full extra
  execution per promotion to print one number and compare nothing.

## What we're proud of

The tests. There are 208 of them and each one corresponds to a specific way this project
could produce a confident wrong answer: that a constant-score model is rejected even
though it exits cleanly; that a hung script leaves no orphan; that a torn state file
still resumes; that `INCONCLUSIVE` is recorded rather than a fabricated verdict; that
the seal test can no longer write the real seal. A guard that exists is not the same
claim as a guard that fires.

## What we learned

Measure before you design. Our first architecture was written before we had downloaded
the data, and three of its premises collapsed within ninety seconds of actually looking:
a cold-item opportunity that didn't exist, a per-user history an order of magnitude
shorter than the brief claimed, and a ranking group half the size we had derived from
row counts. Cheap acquisition steps come before architecture, not after it.

## What's next

Hand the loop a harder benchmark and see whether the ledger's mechanisms — rather than
its prohibitions — transfer. The interesting question isn't whether the agent beats one
baseline; it's whether "the user_id × video_id cross already absorbs the signal"
generalises to a dataset where it doesn't.

## Built with

Python 3.14 · numpy · PyTorch · the Anthropic API (Claude Opus 5) · KuaiRand-Pure
