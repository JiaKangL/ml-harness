# P2 — `harness/evaluator.py` (L2, Measurement)

**Purpose.** Decide whether a candidate genuinely beat the incumbent. This is the
module that separates a real improvement from luck, and it is built before the
generator because a mistake here silently corrupts every number we report.

**Depends on:** L1 only (`config`, `data_guard`, and the frozen `evaluate.py`).
**Must not import:** executor, memory, logger, agent, loop.

---

## Why this module exists

**Measured, not assumed.** 5 FM seeds on valid (`logs/sigma_valid.txt`):
**σ(primary) = 0.00035**, mean 0.60157. Expected best-of-K noise is σ·√(2·ln K):

| K | expected best-of-K noise |
|---|---|
| 10 | +0.00076 |
| 30 | +0.00092 |
| 50 | +0.00099 |

This is *below* the organizers' 0.0008 test std, and the reason matters: our valid
score is already an argmax over ~40 early-stopping epochs, and selecting the peak
suppresses seed variance. Since valid-best is exactly what we compare across
candidates, 0.00035 is the right scale for promotion decisions.

### So why three seeds?

Not because the noise floor demands it — at σ=0.00035, +0.002 is 5.7σ on one seed.
The real reasons:

1. **FM's variance is not every candidate's variance.** A torch model with random
   initialisation can spread 3–5× wider. We cannot know a candidate's variance until
   we measure it, and a single sample gives no estimate at all.
2. **The variance estimate is itself a signal.** A candidate whose seeds disagree is
   unstable, and that belongs in the ledger as INCONCLUSIVE rather than KEEP.
3. **It costs ~45s** on a benchmark where the brief says compute is deliberately not
   the binding constraint.

Honest framing for the write-up: we measured the noise floor rather than assuming it,
found it 3× smaller than our first estimate, and kept the gate anyway for a reason
that survives the correction.

## The two-tier seed ladder (wall-clock)

Running 3 seeds on everything triples execution time. The ladder prunes early but
**never promotes early**:

1. **Run seed 42 alone.** If Δ ≤ `PRUNE_AT_ONE_SEED_DELTA` (−0.005), prune the node
   immediately at one seed — a clear regression needs no confirmation.
2. **Otherwise run seeds 43 and 44.** Promote only if the 3-seed mean beats the parent
   by ≥ `PROMOTE_DELTA` (+0.002).

Asymmetry is the point: an early *prune* costs us only a candidate that was already
6σ below promotable, whereas an early *promote* is exactly the lucky-seed failure this
module exists to prevent.

> **Tuning note.** −0.005 is ~6σ and very safe, but it only catches
> badly-broken-yet-running candidates: a merely neutral candidate scores ≈0.000 and
> survives to 3 seeds anyway. −0.002 is still 3.6σ (loses a good candidate ~0.02% of
> the time) and prunes considerably more. One-line change in `config.py`.

---

## Contract

```python
class Evaluator:
    def __init__(self, data: DataAPI, evaluate_sha256: str,
                 confirm_seeds: tuple[int, ...] = C.CONFIRM_SEEDS): ...

    def score(self, scores: np.ndarray, split: Split) -> Metrics:
        """Wrap the frozen evaluate.py. Never modifies it; verifies its checksum."""

    def aggregate(self, per_seed: list[Metrics]) -> Metrics:
        """Mean across seeds, with primary_std populated."""

    def gate_first_seed(self, candidate: Metrics, incumbent: Metrics) -> LadderDecision:
        """Tier 1. Returns PRUNE (clear regression) or CONTINUE. Never PROMOTE."""

    def gate(self, candidate: Metrics, incumbent: Metrics) -> GateDecision:
        """Tier 2. Promote iff the 3-seed mean beats the incumbent by >= PROMOTE_DELTA
        and the candidate is not quarantined."""

    def convergence(self, history: list[Metrics]) -> ConvergenceState:
        """Organizers' rule, reported exactly: no improvement > 0.002 over the last
        3 SCORED iterations, measured against the running incumbent."""

    def unbiased_check(self, scores_fn, node_id: str) -> Metrics | None:
        """Optional promotion-time re-score on the random-exposure log."""

    def build_submission(self, node: Node, out_csv: Path, runner: Runner) -> Path:
        """Re-execute node.code_path FROM DISK via an INJECTED runner. Never a
        pickled model, and never our own subprocess launcher: running agent code
        under process-group and RSS supervision is the executor's contract (L3), and
        L2 may not import it. `runner` is `Executor.run` bound by the loop.

        Runner = Callable[[str, str, Split, int], RunResult]"""

    def verify_submission(self, csv_path: Path, split: Split) -> bool:
        """Shell out to the official submit.py --check (test) / --score (valid)."""
```

```python
@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str            # human-readable, goes straight into the iteration log
    quarantined: bool      # valid primary > 0.70 => presumed leakage
    delta_primary: float
    delta_gauc: float      # tracked separately -- see below
    delta_ndcg5: float

@dataclass(frozen=True)
class ConvergenceState:
    converged: bool
    stalled_iterations: int    # >= STALL_TRIGGER (2) fires the critics
    best_primary: float
```

---

## Design points that matter

**Report GAUC and nDCG@5 separately, always.** They weight users differently — GAUC by
positive count over only the 63.7% discriminative users; nDCG equally over everyone,
with 36.3% of users permanently fixed at 0 or 1. A change that moves exactly one of
them has a mechanism; one that nudges both slightly is more likely noise. The agent
needs to see which half moved.

**Quarantine above valid primary 0.70.** The valid oracle ceiling is 0.8484 and the
baseline is 0.6016. A jump to 0.70+ is leakage until proven otherwise: freeze the
node, do not promote, flag it. This is the backstop behind `data_guard`.

**Convergence counts SCORED runs only.** Three crashes in a row are not three
non-improvements — they are a broken branch. Measure against the running incumbent,
not run-to-run. And convergence is reported, not obeyed: at `STALL_TRIGGER = 2` the
loop escalates to critics rather than stopping.

**Checksum `evaluate.py` on construction.** If it has changed since preflight, refuse
to score. It is the sole source of truth and must be provably unmodified.

**`build_submission` re-executes from disk, through an injected runner.** Serialising
an in-memory model makes the submission unreproducible and unauditable -- the stored
source is the artifact. But *running* it is L3's job, and L2 must not import L3. So
the loop passes `Executor.run` in, exactly as `unbiased_check` already takes a
callable. Duplicating the launcher inside L2 would be worse than the import: the
unsupervised copy is the one that produces the file we actually submit.

---

## Acceptance tests — `tests/test_evaluator.py`

| Test | Passes when |
|---|---|
| **The headline test** | Evaluator **refuses to promote** the official FM re-run under a different seed. If it promotes that, it will promote noise for six hours. |
| Genuine gain promotes | A synthetic +0.01 shift across all three seeds is promoted |
| Borderline gain rejected | A +0.0015 mean (inside the noise band) is not promoted |
| Ladder prunes early | A −0.01 first seed prunes without running seeds 43/44 |
| Ladder never promotes early | A +0.05 first seed still runs all three seeds |
| Ladder escalates | A −0.001 first seed (above the prune bar) triggers seeds 43/44 |
| Std is populated | `aggregate()` reports non-zero `primary_std` for differing seeds |
| Quarantine fires | Feeding true labels as scores (primary ≈ 0.85) sets `quarantined=True`, `promote=False` |
| Convergence ignores crashes | 3 failed iterations do not count toward N=3 |
| Stall trigger | 2 consecutive non-improvements set `stalled_iterations >= 2` |
| Checksum guard | A modified `evaluate.py` makes construction raise |
| Metric decomposition | `GateDecision` carries distinct `delta_gauc` and `delta_ndcg5` |
| Submission round-trip | Written CSV passes the official `submit.py --check` |

---

## Gotchas found in Phase 1

- **Labels must be int64.** The frozen `evaluate.py` aggregates with builtin `sum()`;
  an int8 array wraps past 127 positives and returns a wrong metric with no error.
  `DataAPI.labels()` already widens — do not "optimise" it back.
- **`evaluate()` is cheap** (0.12s on a valid-shaped split, ~5s for 40 epochs). Do not
  build a faster reimplementation; correctness here is worth far more than speed.
- **Compute is not the binding constraint.** The brief: ~28 min of single-core CPU for
  100 baseline iterations. Three seeds costs minutes, not hours.
