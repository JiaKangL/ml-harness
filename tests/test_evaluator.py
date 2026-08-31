"""Acceptance tests for the promotion gate.

The headline test is `test_refuses_to_promote_a_reseeded_baseline`. If the evaluator
promotes the same model re-run under a different seed, it will promote noise for six
hours and hand the judges a lucky draw. Everything else here is secondary to that.
"""
from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from harness import config as C
from harness import scoring
from harness.data_guard import DataAPI
from harness.evaluator import Evaluator
from harness.scoring import evaluate_sha256
from harness.types import LadderDecision, Metrics


def _metrics(primary: float, seeds=(42, 43, 44), std: float = 0.0004) -> Metrics:
    """A Metrics at a chosen primary, with GAUC/nDCG split the way the real metric
    splits them (GAUC runs ~0.11 above nDCG@5 on this dataset)."""
    return Metrics(
        gauc=primary + 0.055,
        ndcg5=primary - 0.055,
        primary=primary,
        users=22_377,
        rows=124_909,
        seeds=seeds,
        primary_std=std if len(seeds) > 1 else 0.0,
    )


class TestNoiseGate(unittest.TestCase):
    """Does the gate tell a real improvement from a lucky one?"""

    @classmethod
    def setUpClass(cls):
        cls.api = DataAPI()
        cls.ev = Evaluator(cls.api, evaluate_sha256())
        cls.baseline = _metrics(0.60157)  # measured FM mean over 5 valid seeds

    def test_refuses_to_promote_a_reseeded_baseline(self):
        """THE headline test.

        Measured sigma over 5 FM seeds on valid is 0.00035. A re-run of the same model
        under a different seed lands within a couple of sigma of the incumbent. If that
        promotes, the trunk drifts onto whichever seed got lucky and the hidden-test
        delta comes back near zero with no way to tell why.
        """
        for offset in (-0.0007, -0.0003, 0.0, +0.0003, +0.0007):
            with self.subTest(offset=offset):
                candidate = _metrics(self.baseline.primary + offset)
                decision = self.ev.gate(candidate, self.baseline)
                self.assertFalse(
                    decision.promote,
                    f"promoted a re-seeded baseline at {offset:+.4f}: {decision.reason}",
                )

    def test_promotes_a_genuine_gain(self):
        decision = self.ev.gate(_metrics(self.baseline.primary + 0.01), self.baseline)
        self.assertTrue(decision.promote)
        self.assertFalse(decision.quarantined)

    def test_rejects_a_gain_the_search_could_have_manufactured(self):
        """A gain under the bar is refused however much it looks like progress.

        The bar is +0.001, set from the MEASURED noise: sigma(3-seed mean) ~ 0.0002 and
        a best-of-15 selection floor ~ 0.0005. +0.0005 is exactly that floor -- the
        amount of apparent improvement that searching produces for free -- so it must
        not promote.
        """
        decision = self.ev.gate(_metrics(self.baseline.primary + 0.0005), self.baseline)
        self.assertFalse(decision.promote)
        self.assertIn("below", decision.reason)

    def test_the_bar_stays_above_the_selection_noise_floor(self):
        """Guards the recalibration itself. The bar moved from 0.002 to 0.001 after the
        first run; it must never drift below what searching alone can manufacture, or
        the gate stops being a gate."""
        sigma_three_seed = 0.00035 / math.sqrt(3)
        selection_floor = sigma_three_seed * math.sqrt(2 * math.log(15))
        self.assertGreater(C.PROMOTE_DELTA, 2 * selection_floor)
        self.assertGreater(C.PROMOTE_DELTA, 4 * sigma_three_seed)

    def test_single_seed_never_promotes(self):
        """Even a large single-seed gain must go through the full ladder: one sample
        gives no variance estimate at all."""
        big = _metrics(self.baseline.primary + 0.05, seeds=(42,))
        decision = self.ev.gate(big, self.baseline)
        self.assertFalse(decision.promote)
        self.assertIn("seed", decision.reason)

    def test_gate_reports_both_metrics_separately(self):
        """GAUC and nDCG@5 weight users differently -- GAUC by positive count over the
        63.7% discriminative users, nDCG equally with ~36% of users permanently fixed.
        A change that moves exactly one of them has a mechanism; one that nudges both
        is likelier noise. So the gate must surface them independently.

        Modelled here as a pairwise loss: GAUC +0.02, nDCG@5 unchanged.
        """
        candidate = replace(
            self.baseline,
            gauc=self.baseline.gauc + 0.02,
            primary=self.baseline.primary + 0.01,
        )
        decision = self.ev.gate(candidate, self.baseline)
        self.assertAlmostEqual(decision.delta_gauc, 0.02)
        self.assertAlmostEqual(decision.delta_ndcg5, 0.0)
        self.assertAlmostEqual(decision.delta_primary, 0.01)
        self.assertTrue(decision.promote)
        self.assertIn("ΔGAUC", decision.reason)


class TestSeedLadder(unittest.TestCase):
    """Prune early, never promote early."""

    @classmethod
    def setUpClass(cls):
        cls.ev = Evaluator(DataAPI(), evaluate_sha256())
        cls.baseline = _metrics(0.60157)

    def test_prunes_a_clear_regression_at_one_seed(self):
        bad = _metrics(self.baseline.primary - 0.01, seeds=(42,))
        self.assertIs(self.ev.gate_first_seed(bad, self.baseline), LadderDecision.PRUNE)

    def test_escalates_a_near_neutral_first_seed(self):
        """-0.001 is above the -0.005 prune bar, so it earns seeds 43 and 44."""
        neutral = _metrics(self.baseline.primary - 0.001, seeds=(42,))
        self.assertIs(
            self.ev.gate_first_seed(neutral, self.baseline), LadderDecision.CONTINUE
        )

    def test_never_promotes_on_the_first_seed(self):
        """A spectacular first seed still only earns the right to run the other two."""
        great = _metrics(self.baseline.primary + 0.05, seeds=(42,))
        self.assertIs(self.ev.gate_first_seed(great, self.baseline), LadderDecision.CONTINUE)
        self.assertFalse(self.ev.gate(great, self.baseline).promote)


class TestLeakageQuarantine(unittest.TestCase):
    """The backstop behind data_guard: a score too good to be true is treated as
    leakage rather than as a breakthrough."""

    @classmethod
    def setUpClass(cls):
        cls.api = DataAPI()
        cls.ev = Evaluator(cls.api, evaluate_sha256())

    def test_oracle_scores_are_quarantined_not_promoted(self):
        """Feed the true labels as scores -- exactly what a leaking candidate produces."""
        labels = self.api.labels("valid")
        oracle = self.ev.score(labels.astype(float), "valid")
        self.assertGreater(oracle.primary, C.LEAKAGE_QUARANTINE_PRIMARY)

        decision = self.ev.gate(replace(oracle, seeds=(42, 43, 44)), _metrics(0.60157))
        self.assertTrue(decision.quarantined)
        self.assertFalse(decision.promote, "a leaking candidate must never be promoted")


class TestConvergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = Evaluator(DataAPI(), evaluate_sha256())

    def test_crashes_do_not_count_as_non_improvement(self):
        """Three crashes are a broken branch, not three stalled iterations. Counting
        them would end the run at iteration 6 with hours of budget left."""
        state = self.ev.convergence([_metrics(0.60), None, None, None])
        self.assertFalse(state.converged)
        self.assertEqual(state.stalled_iterations, 0)

    def test_three_scored_stalls_converge(self):
        state = self.ev.convergence(
            [_metrics(0.60), _metrics(0.6005), _metrics(0.6003), _metrics(0.6007)]
        )
        self.assertTrue(state.converged)

    def test_stall_trigger_fires_one_iteration_before_convergence(self):
        """Critics must fire at 2 so a critique-driven attempt can land as the third
        iteration and break the streak."""
        state = self.ev.convergence([_metrics(0.60), _metrics(0.6005), _metrics(0.6003)])
        self.assertFalse(state.converged)
        self.assertGreaterEqual(state.stalled_iterations, C.STALL_TRIGGER)

    def test_improvement_resets_the_streak(self):
        state = self.ev.convergence(
            [_metrics(0.60), _metrics(0.6005), _metrics(0.6003), _metrics(0.62)]
        )
        self.assertEqual(state.stalled_iterations, 0)
        self.assertAlmostEqual(state.best_primary, 0.62)


class TestScoringPathIntegrity(unittest.TestCase):
    def test_refuses_to_construct_against_a_modified_metric(self):
        with self.assertRaises(scoring.EvaluateModifiedError):
            Evaluator(DataAPI(), "0" * 64)

    def test_cannot_score_test(self):
        ev = Evaluator(DataAPI(), evaluate_sha256())
        with self.assertRaises(ValueError):
            ev.score(np.zeros(170_588), "test")

    def test_length_mismatch_is_caught(self):
        ev = Evaluator(DataAPI(), evaluate_sha256())
        with self.assertRaises(ValueError):
            ev.score(np.zeros(10), "valid")

    def test_aggregate_reports_spread(self):
        agg = scoring.aggregate([_metrics(0.60, (42,)), _metrics(0.61, (43,)), _metrics(0.62, (44,))])
        self.assertEqual(agg.seeds, (42, 43, 44))
        self.assertGreater(agg.primary_std, 0.0)
        self.assertAlmostEqual(agg.primary, 0.61)


class TestUnbiasedCheck(unittest.TestCase):
    def test_random_exposure_scores_and_holds_no_test_rows(self):
        ev = Evaluator(DataAPI(), evaluate_sha256())
        m = ev.unbiased_check(lambda f: np.random.default_rng(0).random(len(f["user_id"])))
        self.assertEqual(m.rows, 288_338)
        self.assertGreater(m.primary, 0.3)
        self.assertLess(m.primary, 0.7)


class TestSubmission(unittest.TestCase):
    def test_written_csv_passes_the_official_check(self):
        ev = Evaluator(DataAPI(), evaluate_sha256())
        scores = np.random.default_rng(0).random(ev.data.n_rows("test"))
        out = C.RUNS_DIR / "_test" / "submission.csv"
        ev.write_submission(scores, out)
        self.assertTrue(ev.verify_alignment(out, "test"))
        out.unlink()

    def test_non_finite_scores_are_refused(self):
        ev = Evaluator(DataAPI(), evaluate_sha256())
        scores = np.zeros(ev.data.n_rows("test"))
        scores[0] = np.nan
        with self.assertRaises(ValueError):
            ev.write_submission(scores, C.RUNS_DIR / "_test" / "bad.csv")

    def test_score_verification_cannot_target_test(self):
        """`submit.py --score --split test` reads long_view from the raw CSV, so the
        scoring verifier deliberately takes no split parameter."""
        import inspect

        sig = inspect.signature(Evaluator.verify_and_score_valid)
        self.assertNotIn("split", sig.parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
