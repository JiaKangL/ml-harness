"""L2 — the promotion gate. Decides whether a candidate genuinely beat the incumbent.

Built before the generator on purpose: a mistake here does not crash, it silently
corrupts every number we report, and the number we report is the submission.

Depends on L1 only. It never launches a subprocess -- running agent-written code
under process-group and RSS supervision is L3's contract, so anything that needs code
executed takes an injected callable instead.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from . import config as C
from . import scoring
from .data_guard import DataAPI
from .types import (
    ConvergenceState,
    GateDecision,
    LadderDecision,
    Metrics,
    Node,
    Split,
)


class Runner(Protocol):
    """`Executor.run`, bound by the loop. L2 declares the shape it needs without
    importing L3."""

    def __call__(self, code_path: str, node_id: str, split: Split, seed: int) -> object: ...


ScoresFn = Callable[[dict[str, np.ndarray]], np.ndarray]


class Evaluator:
    def __init__(
        self,
        data: DataAPI,
        evaluate_sha256: str,
        promote_delta: float = C.PROMOTE_DELTA,
        prune_delta: float = C.PRUNE_AT_ONE_SEED_DELTA,
        quarantine_above: float = C.LEAKAGE_QUARANTINE_PRIMARY,
    ):
        # If the frozen metric changed since preflight, refuse to score at all rather
        # than report numbers against a moved goalpost.
        scoring.assert_evaluate_unmodified(evaluate_sha256)
        self.data = data
        self.promote_delta = promote_delta
        self.prune_delta = prune_delta
        self.quarantine_above = quarantine_above

    # ---------------------------------------------------------------- scoring

    def score(self, scores: np.ndarray, split: Split = "valid", seed: int = 42) -> Metrics:
        if split == "test":
            # Not a policy choice -- DataAPI.labels("test") raises, so this is simply
            # unreachable. Named explicitly so the intent is legible.
            raise ValueError("test cannot be scored during the run; that is score_final.py")
        users = self.data.features(split)["user_id"]
        labels = self.data.labels(split)
        if scores.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{split}: {scores.shape[0]} scores for {labels.shape[0]} rows"
            )
        return scoring.score(users, labels, scores, seed=seed)

    @staticmethod
    def aggregate(per_seed: list[Metrics]) -> Metrics:
        return scoring.aggregate(per_seed)

    # ---------------------------------------------------------------- the ladder

    def gate_first_seed(self, candidate: Metrics, incumbent: Metrics) -> LadderDecision:
        """Tier 1. PRUNE a clear regression at one seed, else CONTINUE.

        There is deliberately no PROMOTE outcome. The asymmetry is the whole design:
        an early prune costs us a candidate already far below promotable, while an
        early promote is precisely the lucky-seed failure this module exists to stop.
        """
        if candidate.primary - incumbent.primary <= self.prune_delta:
            return LadderDecision.PRUNE
        return LadderDecision.CONTINUE

    def gate(self, candidate: Metrics, incumbent: Metrics) -> GateDecision:
        """Tier 2. Promote only on a multi-seed mean."""
        d_primary = candidate.primary - incumbent.primary
        d_gauc = candidate.gauc - incumbent.gauc
        d_ndcg5 = candidate.ndcg5 - incumbent.ndcg5
        common = dict(
            delta_primary=d_primary,
            delta_gauc=d_gauc,
            delta_ndcg5=d_ndcg5,
            seeds_run=candidate.n_seeds,
        )

        # The valid oracle ceiling is 0.8484 and the baseline 0.6016. A jump past 0.70
        # is leakage until proven otherwise -- the backstop behind data_guard.
        if candidate.primary > self.quarantine_above:
            return GateDecision(
                promote=False,
                quarantined=True,
                reason=(
                    f"quarantined: valid primary {candidate.primary:.4f} exceeds "
                    f"{self.quarantine_above:.2f}; the valid oracle ceiling is 0.8484 "
                    "and the baseline 0.6016, so this is presumed leakage"
                ),
                **common,
            )

        if candidate.n_seeds < len(C.CONFIRM_SEEDS):
            return GateDecision(
                promote=False,
                quarantined=False,
                reason=(
                    f"only {candidate.n_seeds} seed(s); promotion requires "
                    f"{len(C.CONFIRM_SEEDS)}. One sample gives no variance estimate, "
                    "and FM's variance is not every candidate's variance."
                ),
                **common,
            )

        if d_primary < self.promote_delta:
            return GateDecision(
                promote=False,
                quarantined=False,
                reason=(
                    f"Δprimary {d_primary:+.4f} (±{candidate.primary_std:.4f} over "
                    f"{candidate.n_seeds} seeds) below the {self.promote_delta:+.3f} bar"
                ),
                **common,
            )

        return GateDecision(
            promote=True,
            quarantined=False,
            reason=(
                f"Δprimary {d_primary:+.4f} (±{candidate.primary_std:.4f} over "
                f"{candidate.n_seeds} seeds) — ΔGAUC {d_gauc:+.4f}, "
                f"ΔnDCG@5 {d_ndcg5:+.4f}"
            ),
            **common,
        )

    # ---------------------------------------------------------------- convergence

    def convergence(
        self,
        history: list[Metrics | None],
        epsilon: float = C.EPSILON,
        n_converge: int = C.N_CONVERGE,
        stall_trigger: int = C.STALL_TRIGGER,
    ) -> ConvergenceState:
        """The organizers' rule, reported exactly -- but counting SCORED runs only.

        `None` entries are failed iterations. Three crashes in a row are not three
        non-improvements; they are a broken branch, and treating them as convergence
        would end the run at iteration 6 with hours of budget left.

        Convergence is reported, not obeyed. At `stall_trigger` (2, one before the
        organizers' N=3) the loop escalates to the critics, so a critique-driven
        attempt can still land as the third iteration and break the streak.
        """
        scored = [m for m in history if m is not None]
        if not scored:
            return ConvergenceState(False, 0, float("-inf"), "no scored runs yet")

        best = float("-inf")
        stalled = 0
        for m in scored:
            if m.primary > best + epsilon:
                best, stalled = m.primary, 0
            else:
                best, stalled = max(best, m.primary), stalled + 1

        converged = stalled >= n_converge
        reason = (
            f"{stalled} scored iteration(s) without a >{epsilon} improvement over the "
            f"running incumbent {best:.4f}"
        )
        return ConvergenceState(converged, stalled, best, reason)

    # ---------------------------------------------------------------- unbiased check

    def unbiased_check(self, scores_fn: ScoresFn) -> Metrics:
        """Re-score on randomly-exposed impressions -- a second validation set with a
        *different* bias. A candidate that gains on logged traffic but not here learned
        the logging policy rather than the ranking.

        The underlying log spans both the valid and test windows, and 75.7% of its rows
        are test-window. Those were dropped at cache-build time, so this cannot become a
        test-set selection signal.
        """
        feats, labels = self.data.random_exposure()
        return scoring.score(feats["user_id"], labels, scores_fn(feats))

    # ---------------------------------------------------------------- submission

    def build_submission(self, node: Node, out_csv: Path, runner: Runner) -> Path:
        """Re-execute the promoted node's stored source, through an injected runner.

        Never a pickled model: the stored source is the artifact, and a serialised
        model is neither reproducible nor auditable. Never our own launcher either --
        supervising agent code is L3's contract and L2 may not import it, so the loop
        passes `Executor.run` in.
        """
        result = runner(node.code_path, node.node_id, "test", C.CONFIRM_SEEDS[0])
        scores_path = getattr(result, "scores_path", None)
        if not getattr(result, "ok", False) or scores_path is None:
            raise RuntimeError(f"submission run failed for node {node.node_id}")
        return self.write_submission(np.load(scores_path), out_csv)

    def write_submission_for(
        self, scores: np.ndarray, out_csv: Path, split: Split = "test"
    ) -> Path:
        """The same writer, for any split. Exists so the valid cross-check can produce
        a file `submit.py --score --split valid` will accept -- which is the only way
        to check our scoring path against the organizers' on the same artifact."""
        feats = self.data.features(split)
        n = feats["user_id"].shape[0]
        if scores.shape[0] != n:
            raise ValueError(f"{scores.shape[0]} scores for {n} {split} rows")
        if not np.isfinite(scores).all():
            raise ValueError("submission contains NaN or Inf")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row_id", "user_id", "video_id", "score"])
            for i, (u, v, s) in enumerate(zip(feats["user_id"], feats["video_id"], scores)):
                w.writerow([i, int(u), int(v), f"{float(s):.6g}"])
        return out_csv

    def write_submission(self, scores: np.ndarray, out_csv: Path) -> Path:
        """CSV in the official schema: row_id,user_id,video_id,score.

        `row_id` is a positional index into data.load()'s ordering, which the cache
        reproduces byte for byte (asserted by preflight). `(user_id, video_id)` is
        redundant and exists only so the organizers can verify alignment -- it is not
        unique, since 3.06% of test rows are repeated pairs.
        """
        feats = self.data.features("test")
        n = feats["user_id"].shape[0]
        if scores.shape[0] != n:
            raise ValueError(f"{scores.shape[0]} scores for {n} test rows")
        if not np.isfinite(scores).all():
            raise ValueError("submission contains NaN or Inf")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row_id", "user_id", "video_id", "score"])
            for i, (u, v, s) in enumerate(zip(feats["user_id"], feats["video_id"], scores)):
                w.writerow([i, int(u), int(v), f"{float(s):.6g}"])
        return out_csv

    # -- Verification is split into two non-parameterised methods on purpose.
    #    `submit.py --score --split test` reads long_view straight from the raw CSV,
    #    so a single method taking `split` is one wrong argument away from printing
    #    the final hidden-test score mid-run.

    def verify_alignment(self, csv_path: Path, split: Split = "test") -> bool:
        """Format and alignment only. Never scores."""
        return self._submit(["--check", "--split", split, str(csv_path)])

    def verify_and_score_valid(self, csv_path: Path) -> bool:
        """Scores, and only ever against valid. `split` is not a parameter here."""
        return self._submit(["--score", "--split", "valid", str(csv_path)])

    @staticmethod
    def _submit(args: list[str]) -> bool:
        proc = subprocess.run(
            [C.PYTHON_BIN, "submit.py", "--data_dir", str(C.DATA_DIR), *args],
            cwd=C.STARTER_KIT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"submit.py failed:\n{proc.stdout}\n{proc.stderr}")
        return True
