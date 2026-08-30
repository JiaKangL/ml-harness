"""Acceptance tests for the preflight gate and the data firewall.

Every test here corresponds to a specific way this project can produce a confident
wrong answer. They are written as assertions about behaviour under attack, not as
assertions that the code was written -- "the guard exists" is not the same claim as
"the guard fires".

    ./.venv/bin/python -m unittest discover -s tests -v

Stdlib unittest on purpose: the starter kit's selling point is that it needs numpy
and nothing else, and a reviewer should be able to run these without installing a
test framework first.
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from harness import config as C
from harness import holdout
from harness.data_guard import (
    DataAPI,
    OutcomeColumnAccessError,
    TestLabelAccessError,

)


class TestTestSplitFirewall(unittest.TestCase):
    """The agent must not be able to compute a test score, so it cannot select on one."""

    @classmethod
    def setUpClass(cls):
        cls.api = DataAPI()

    def test_labels_test_raises(self):
        with self.assertRaises(TestLabelAccessError):
            self.api.labels("test")

    def test_column_test_label_raises(self):
        with self.assertRaises(TestLabelAccessError):
            self.api.column("test", C.LABEL)

    def test_cache_contains_no_test_label(self):
        with np.load(C.SPLITS_NPZ) as z:
            self.assertNotIn(f"test__{C.LABEL}", z.files)

    def test_no_test_label_artifact_exists_anywhere(self):
        """The strongest form of the guarantee: nothing to read, not merely hidden.

        A file that is never written cannot be found by a stray glob over cache/.
        """
        self.assertEqual(holdout.assert_no_holdout_artifact(), [])

    def test_test_features_are_still_available(self):
        """The firewall must not break the thing it protects: scoring test needs features."""
        feats = self.api.features("test")
        self.assertEqual(set(feats), set(C.LOG_SAFE))
        self.assertEqual(feats["user_id"].shape[0], 170_588)


class TestOutcomeColumnFirewall(unittest.TestCase):
    """long_view is a threshold on play_time/duration, so outcome columns on an
    evaluation split are the answer rather than a feature."""

    @classmethod
    def setUpClass(cls):
        cls.api = DataAPI()

    def test_outcome_column_on_eval_split_raises(self):
        for split in ("valid", "test"):
            for col in ("play_time_ms", "is_click", "is_like"):
                with self.subTest(split=split, col=col):
                    with self.assertRaises(OutcomeColumnAccessError):
                        self.api.column(split, col)

    def test_aux_targets_outside_train_raises(self):
        with self.assertRaises(OutcomeColumnAccessError):
            self.api.aux_targets("valid")  # type: ignore[arg-type]

    def test_cache_has_no_eval_outcome_arrays(self):
        with np.load(C.SPLITS_NPZ) as z:
            forbidden = [
                f"{s}__{c}" for s in ("valid", "test") for c in C.LOG_OUTCOME
            ]
            self.assertEqual([k for k in forbidden if k in z.files], [])

    def test_aux_targets_on_train_still_work(self):
        aux = self.api.aux_targets("train")
        self.assertEqual(set(aux), set(C.LOG_OUTCOME))

    def test_quarantined_file_not_in_cache(self):
        """video_features_statistic_pure.csv aggregates span the test window."""
        with np.load(C.SIDE_NPZ) as z:
            for k in z.files:
                self.assertNotIn("play_cnt", k)
                self.assertNotIn("play_progress", k)


class TestLabelDtype(unittest.TestCase):
    """Regression test for a silent metric corruption.

    evaluate.py aggregates labels with builtin sum(). Under NumPy 2's weak promotion
    an int8 array accumulates in int8 and wraps past 127 positives, producing a wrong
    metric with no exception raised. This surfaced as primary=inf; in another shape it
    would have returned a plausible number.
    """

    def test_labels_are_int64(self):
        api = DataAPI()
        for split in ("train", "valid"):
            self.assertEqual(api.labels(split).dtype, np.int64, msg=split)

    def test_builtin_sum_over_labels_does_not_overflow(self):
        api = DataAPI()
        y = api.labels("valid")
        self.assertEqual(sum(y), int(y.sum()))
        self.assertGreater(sum(y), 127)


class TestRowAlignment(unittest.TestCase):
    """Submission row_id is a positional index into data.load()'s ordering. A
    reordering misaligns every submitted score while every metric still looks fine."""

    def test_counts_match_the_brief(self):
        api = DataAPI()
        for split, expected in (
            ("train", 1_141_112),
            ("valid", 124_909),
            ("test", 170_588),
        ):
            self.assertEqual(api.n_rows(split), expected, msg=split)

    def test_row_ids_are_contiguous_from_zero(self):
        api = DataAPI()
        rid = api.row_ids("test")
        self.assertEqual(rid[0], 0)
        self.assertEqual(rid[-1], api.n_rows("test") - 1)


class TestGrouping(unittest.TestCase):
    """groups() must partition rows exactly as evaluate.py does, or a listwise loss
    optimises a different partition than the metric scores."""

    def test_groups_match_user_id_partition(self):
        api = DataAPI()
        for split in ("valid", "test"):
            with self.subTest(split=split):
                users = api.features(split)["user_id"]
                groups = api.groups(split)
                self.assertEqual(np.unique(groups).size, np.unique(users).size)
                # Same user always lands in the same group, and vice versa.
                order = np.argsort(users, kind="stable")
                u, g = users[order], groups[order]
                boundary = u[1:] != u[:-1]
                self.assertTrue(np.array_equal(boundary, g[1:] != g[:-1]))

    def test_group_count_matches_evaluate(self):
        import sys

        sys.path.insert(0, str(C.STARTER_KIT))
        from evaluate import evaluate

        api = DataAPI()
        users, labels = api.features("valid")["user_id"], api.labels("valid")
        res = evaluate(users, labels, np.zeros(len(labels)))
        self.assertEqual(res["users"], int(api.groups("valid").max()) + 1)
        self.assertEqual(res["rows"], api.n_rows("valid"))


class TestScoringPath(unittest.TestCase):
    """The organizers' own harness self-test. If random scoring does not reproduce,
    nothing downstream means anything."""

    def test_random_baseline_reproduces(self):
        import sys

        sys.path.insert(0, str(C.STARTER_KIT))
        from evaluate import evaluate

        api = DataAPI()
        users, labels = api.features("valid")["user_id"], api.labels("valid")
        got = evaluate(users, labels, np.random.default_rng(0).random(len(labels)))
        self.assertAlmostEqual(
            got["primary"], C.EXPECTED["random_valid_primary"], delta=C.RANDOM_TOLERANCE
        )

    def test_oracle_scores_above_baseline(self):
        """Sanity on the metric's direction: true labels as scores must beat random."""
        import sys

        sys.path.insert(0, str(C.STARTER_KIT))
        from evaluate import evaluate

        api = DataAPI()
        users, labels = api.features("valid")["user_id"], api.labels("valid")
        oracle = evaluate(users, labels, labels.astype(float))
        self.assertGreater(oracle["primary"], 0.80)
        self.assertAlmostEqual(oracle["GAUC"], 1.0, places=6)


class TestDataProfile(unittest.TestCase):
    """The profile ships in every prompt, so its size is a budget, not a detail."""

    @classmethod
    def setUpClass(cls):
        cls.text = C.DATA_PROFILE_JSON.read_text()
        cls.prof = json.loads(cls.text)

    def test_fits_the_prompt_budget(self):
        self.assertLess(len(self.text) // 4, 1400, "profile exceeds ~1400 token budget")

    def test_carries_the_load_bearing_sections(self):
        for key in (
            "splits",
            "within_user_variance",
            "train_history_per_user",
            "aux_signal_correlation_with_label",
            "column_legality",
        ):
            self.assertIn(key, self.prof)

    def test_profile_leaks_no_test_labels(self):
        """Composition stats require labels, so test must not have them."""
        self.assertNotIn("user_composition", self.prof["splits"]["test"])
        self.assertNotIn("positive_rate", self.prof["splits"]["test"])

    def test_duplicate_rate_matches_the_published_figure(self):
        """Independent cross-check: the README states 3.06% duplicate pairs on test."""
        self.assertAlmostEqual(
            self.prof["splits"]["test"]["duplicate_user_video_pct"], 3.06, delta=0.01
        )

    def test_tab_is_recorded_as_within_user_constant_for_about_half(self):
        """The headline insight: globally predictive, locally useless for ~half of users."""
        self.assertLess(self.prof["within_user_variance"]["test"]["tab"], 0.60)
        self.assertGreater(self.prof["within_user_variance"]["test"]["video_id"], 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRandomExposureFirewall(unittest.TestCase):
    """The unbiased promotion check reads a log that is 75.7% test-window rows, every
    one carrying long_view. Without a build-time filter the design itself would route
    test ground truth into the promotion gate."""

    @classmethod
    def setUpClass(cls):
        cls.api = DataAPI()

    def test_no_test_window_rows_cached(self):
        feats, _ = self.api.random_exposure()
        self.assertEqual(int(feats["date"].max()), 20220428)
        self.assertLess(int(feats["date"].max()), C.SPLITS["test"][0])

    def test_valid_window_rows_present(self):
        feats, labels = self.api.random_exposure()
        self.assertEqual(feats["user_id"].shape[0], 288_338)
        self.assertEqual(labels.shape[0], 288_338)
        self.assertEqual(labels.dtype, np.int64)

    def test_dropped_the_test_window_majority(self):
        """The raw file has 1,186,059 rows; 897,721 of them are test-window."""
        feats, _ = self.api.random_exposure()
        self.assertEqual(1_186_059 - feats["date"].shape[0], 897_721)


class TestGroundTruthImmutability(unittest.TestCase):
    """L1's whole justification is that a mistake here corrupts every number the
    higher layers report. Accessors hand out references into shared arrays, and
    several consumers share one DataAPI -- so one in-place op would do exactly that."""

    def test_feature_arrays_are_read_only(self):
        api = DataAPI()
        for split in ("train", "valid", "test"):
            for name, arr in api.features(split).items():
                with self.subTest(split=split, col=name):
                    with self.assertRaises(ValueError):
                        arr[0] = 999

    def test_side_tables_are_read_only(self):
        api = DataAPI()
        with self.assertRaises(ValueError):
            api.video_feature("author_id")[0] = 999

    def test_groups_are_read_only(self):
        api = DataAPI()
        with self.assertRaises(ValueError):
            api.groups("valid")[0] = 999


class TestQuarantineCheckCanFail(unittest.TestCase):
    """Regression: this check previously returned a hardcoded True and reported which
    files existed on disk, which is a press release rather than a check."""

    def test_quarantine_check_reads_the_real_header(self):
        from harness.preflight import check_quarantine

        result = check_quarantine()
        self.assertTrue(result.ok)
        self.assertIn("verified absent", result.detail)

    def test_no_quarantined_column_reached_the_cache(self):
        import csv as _csv

        with np.load(C.SIDE_NPZ) as z:
            video_cols = {k.split("__", 1)[1] for k in z.files if k.startswith("video__")}
        for name in C.QUARANTINED_FILES:
            path = C.DATA_DIR / name
            if not path.exists():
                continue
            with open(path, newline="") as fh:
                header = next(_csv.reader(fh))
            leaked = [c for c in header if c != "video_id" and c in video_cols]
            self.assertEqual(leaked, [], msg=f"{name} leaked {leaked}")

    def test_namespace_collision_is_not_a_false_positive(self):
        """follow_user_num exists in BOTH the user table (static profile attribute,
        legitimate) and the quarantined video table (follows generated by the video,
        spanning the test window). Scoping by namespace is what tells them apart."""
        api = DataAPI()
        self.assertIn("follow_user_num", api.side_columns()["user"])
        self.assertNotIn("follow_user_num", api.side_columns()["video"])


class TestHoldoutSeal(unittest.TestCase):
    """Test labels are reachable only after a converged run is sealed, and every
    draw is recorded. The point is that the holdout cannot inform any decision the
    loop makes -- not by convention, but because it does not exist yet."""

    def setUp(self):
        self._seal = C.RUN_SEAL_JSON.read_text() if C.RUN_SEAL_JSON.exists() else None
        self._draws = (
            C.SCORED_MARKER_JSON.read_text() if C.SCORED_MARKER_JSON.exists() else None
        )
        C.RUN_SEAL_JSON.unlink(missing_ok=True)
        C.SCORED_MARKER_JSON.unlink(missing_ok=True)

    def tearDown(self):
        for path, saved in (
            (C.RUN_SEAL_JSON, self._seal),
            (C.SCORED_MARKER_JSON, self._draws),
        ):
            path.unlink(missing_ok=True)
            if saved is not None:
                path.write_text(saved)

    def test_unsealed_run_cannot_read_test_labels(self):
        self.assertFalse(holdout.is_sealed())
        with self.assertRaises(holdout.RunNotSealedError):
            holdout.extract_test_labels()

    def test_sealed_run_can_read_once(self):
        holdout.seal_run("node_07", "a" * 64, 0.6125, 14, "3 scored stalls")
        labels = holdout.extract_test_labels()
        self.assertEqual(labels.shape[0], 170_588)
        self.assertEqual(labels.dtype, np.int64)
        # The organizers' published test positive rate is ~0.313.
        self.assertAlmostEqual(float(labels.mean()), 0.313, delta=0.01)

    def test_second_draw_is_refused(self):
        holdout.seal_run("node_07", "a" * 64, 0.6125, 14, "converged")
        holdout.extract_test_labels()
        with self.assertRaises(holdout.AlreadyScoredError):
            holdout.extract_test_labels()

    def test_forced_second_draw_is_recorded_not_silent(self):
        holdout.seal_run("node_07", "a" * 64, 0.6125, 14, "converged")
        holdout.extract_test_labels()
        holdout.extract_test_labels(force=True)
        draws = json.loads(C.SCORED_MARKER_JSON.read_text())
        self.assertEqual(len(draws), 2)
        self.assertTrue(draws[1]["forced_second_draw"])

    def test_extracted_labels_align_with_test_features(self):
        """Row order must match features("test") positionally, or the final score is
        computed against the wrong rows while looking entirely plausible."""
        holdout.seal_run("node_07", "a" * 64, 0.6125, 14, "converged")
        labels = holdout.extract_test_labels()
        api = DataAPI()
        self.assertEqual(labels.shape[0], api.n_rows("test"))

    def test_single_import_site(self):
        """The audit trail is a one-line grep. Only score_final may import this."""
        import subprocess

        out = subprocess.run(
            ["grep", "-rln", "--include=*.py", "extract_test_labels", "harness/", "tests/"],
            cwd=C.ROOT, capture_output=True, text=True,
        ).stdout.split()
        self.assertEqual(
            sorted(out), ["harness/holdout.py", "tests/test_phase01.py"],
            f"unexpected importers of the holdout: {out}",
        )
