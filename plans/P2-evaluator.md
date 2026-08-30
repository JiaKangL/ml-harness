# P2 — `harness/evaluator.py` (L2, Measurement)

**Purpose.** Decide whether a candidate genuinely beat the incumbent. This is the
module that separates a real improvement from luck, and it is built before the
generator because a mistake here silently corrupts every number we report.

**Depends on:** L1 only (`config`, `data_guard`, and the frozen `evaluate.py`).
**Must not import:** executor, memory, logger, agent, loop.

---

## Why this module exists

FM's seed-to-seed std is **0.0008**. Over K candidate evaluations the expected best is
σ·√(2·ln K) above the mean by luck alone:

| K | expected best-of-K noise |
|---|---|
| 10 | +0.0026 |
| 30 | +0.0031 |
| 50 | +0.0033 |

The competition ships the **validation-best checkpoint**, so an evaluator that
promotes on a single sample will hand the judges a lucky seed. The organizers' own
ε=0.002 sits *inside* that noise band — it is a fine convergence-reporting rule and an
unusable promotion criterion. Hence: three seeds, always, promote on the mean.

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

    def gate(self, candidate: Metrics, incumbent: Metrics) -> GateDecision:
        """Promote iff mean primary beats incumbent by > EPSILON across all seeds,
        and the candidate is not quarantined."""

    def convergence(self, history: list[Metrics]) -> ConvergenceState:
        """Organizers' rule, reported exactly: no improvement > 0.002 over the last
        3 SCORED iterations, measured against the running incumbent."""

    def unbiased_check(self, scores_fn, node_id: str) -> Metrics | None:
        """Optional promotion-time re-score on the random-exposure log."""

    def build_submission(self, node: Node, out_csv: Path) -> Path:
        """Re-execute node.code_path FROM DISK. Never a pickled model."""

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

**`build_submission` re-executes from disk.** Serialising an in-memory model makes the
submission unreproducible and unauditable. The stored source is the artifact.

---

## Acceptance tests — `tests/test_evaluator.py`

| Test | Passes when |
|---|---|
| **The headline test** | Evaluator **refuses to promote** the official FM re-run under a different seed. If it promotes that, it will promote noise for six hours. |
| Genuine gain promotes | A synthetic +0.01 shift across all three seeds is promoted |
| Borderline gain rejected | A +0.0015 mean (inside the noise band) is not promoted |
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
