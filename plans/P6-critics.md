# P6 — `harness/critics.py` + seed ensembling (L5)

**Purpose.** When the run stalls, rescue it **without a human**. Autonomy is scored by
number of manual interventions, so an autonomous escalation path is worth more than
any single modelling idea.

**Depends on:** L1–L5.

## The trigger

Fire at **2 consecutive non-improving scored iterations** — deliberately one *before*
the organizers' N=3 convergence. If we waited for 3 the run would already be formally
converged before a critique could produce anything; firing at 2 lets the
critique-driven attempt land as the third iteration and break the streak.

## Three isolated critics

Each runs in a **fresh context**. They receive the trunk source, `data_profile.json`
and the insight ledger — **not** the main agent's reasoning chain. This is
load-bearing: a critic that sees the agent's reasoning agrees with it, because models
are sycophantic toward their own prior reasoning. Isolation is what makes the critique
worth anything.

| Critic | Question it is asked |
|---|---|
| **A — objective alignment** | Is the loss optimising something the metric discards? (e.g. per-user absolute level, which cancels out of within-user ranking) |
| **B — validity** | Leakage, unsound validation, is this gain real or noise? |
| **C — unexplored space** | Given the ledger, what has *not* been tried, and what does the published literature suggest? |

Critic A is also the safety net for the one idea we deliberately did not hard-code:
if the agent never proposes loss alignment, A surfaces it unprompted.

## Rules

- Critique output re-enters as **ordinary proposals through the same 3-seed gate**. A
  critic can talk the agent into a bad direction; the gate is the safety net.
- **Max 2 critique rounds.** After that the run is genuinely converged.
- Every critique is logged as its own iteration entry with its own hypothesis.

## Endgame — seed ensembling

When critique is exhausted, average the per-row scores of the top-k **confirmed**
nodes (those that already passed the 3-seed gate). Their errors are partly
independent, so noise falls as ~√k — turning "several things that each worked a
little" into one thing that works slightly more. It attacks the selection-noise
problem directly rather than gambling against it.

**The ensemble node writes a real script.** `build_submission` re-executes one
`node.code_path` from disk, so an ensemble cannot be an in-memory average of k score
arrays -- it would have no source. Instead the ensemble step *generates* a candidate
script that loads the k constituent nodes' score files and averages them, writes it to
`outputs/candidate_iter_NN.py` like any other candidate, and runs it through the
normal path. It is then one script with one `code_sha256`, reproducible from disk like
everything else.

Logged as its own iteration with its own hypothesis, and scored through the same gate.
Never applied post-hoc as untracked tuning.

## Acceptance tests — `tests/test_critics.py`

| Test | Passes when |
|---|---|
| Trigger timing | Critics fire at exactly 2 stalls, not 3 |
| Isolation | The critic prompt provably excludes the agent's reasoning chain |
| Same gate | A critic-originated proposal still requires 3 seeds to promote |
| Round cap | A third critique round never fires |
| Zero interventions | A full mock run with forced stalls records 0 manual interventions |
| Ensemble improves | Averaging k synthetic noisy scorers beats the mean single scorer |
| Ensemble logged | The ensemble appears as a normal iteration entry |
| Ensemble is a script | The ensemble node has a real code_path that re-executes standalone |
