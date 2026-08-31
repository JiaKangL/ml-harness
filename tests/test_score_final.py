"""Acceptance tests for P7 -- the one module permitted to read the test labels.

It had no tests, and it is the module where a mistake is unrecoverable: a test score
that leaks back into the loop invalidates every claim the project makes about its own
validation. These are the most important tests in the repository.

    ./.venv/bin/python -m unittest tests.test_score_final -v
"""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import score_final
from harness import config as C
from harness import holdout


class TestRefusals(unittest.TestCase):
    """The seal is the precondition. Without it a test score could still inform a
    decision, which is the single thing the holdout exists to prevent."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        d = Path(self.dir.name)
        patches = [
            mock.patch.object(C, "RUN_SEAL_JSON", d / "run_seal.json"),
            mock.patch.object(C, "SCORED_MARKER_JSON", d / "test_draws.json"),
            mock.patch.object(C, "DATA_DIR", d / "data"),
            mock.patch.object(C, "FINAL_RESULT_JSON", d / "final_result.json"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.data = d / "data"
        self.data.mkdir()
        self._write_tiny_log()

    def _write_tiny_log(self, n: int = 6):
        """Two files with the real names, so `extract_test_labels` parses something
        small instead of 1.3M rows of the actual log."""
        lo, hi = C.SPLITS["test"]
        for i, name in enumerate(C.LOG_FILES):
            with open(self.data / name, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["date", C.LABEL])
                if i == 1:
                    for j in range(n):
                        w.writerow([lo, j % 2])

    def seal(self):
        holdout.seal_run("n07", "abc123", 0.6100, 9, "converged")

    def test_extract_refuses_before_the_run_is_sealed(self):
        with self.assertRaises(holdout.RunNotSealedError):
            holdout.extract_test_labels()
        self.assertFalse(C.SCORED_MARKER_JSON.exists(), "a refusal must record no draw")

    def test_score_final_refuses_and_reads_nothing_when_unsealed(self):
        with mock.patch.object(
            holdout, "extract_test_labels", side_effect=AssertionError("must not read")
        ):
            self.assertEqual(score_final.main([]), 2)

    def test_a_second_draw_refuses_without_force(self):
        self.seal()
        holdout.extract_test_labels()
        with self.assertRaises(holdout.AlreadyScoredError):
            holdout.extract_test_labels()

    def test_force_is_permitted_and_recorded_as_a_second_draw(self):
        """The override is logged rather than merely allowed: a second draw makes the
        estimate optimistically biased, and the honest version of that is a record."""
        self.seal()
        holdout.extract_test_labels()
        holdout.extract_test_labels(force=True)
        draws = json.loads(C.SCORED_MARKER_JSON.read_text())
        self.assertEqual(len(draws), 2)
        self.assertFalse(draws[0]["forced_second_draw"])
        self.assertTrue(draws[1]["forced_second_draw"])
        self.assertEqual(draws[1]["seal"]["node_id"], "n07")


class TestNoFeedbackPath(unittest.TestCase):
    """Structural, not behavioural: the guarantee is about which files exist."""

    ROOT = Path(__file__).resolve().parent.parent

    def test_holdout_is_imported_by_exactly_one_module_outside_the_harness(self):
        hits = subprocess.run(
            ["grep", "-rnE", r"^\s*(from|import)\s+[.\w]*\bholdout\b|^\s*from\s+\S+\s+import\s+.*\bholdout\b",
             "--include=*.py", str(self.ROOT / "harness"), str(self.ROOT / "score_final.py")],
            capture_output=True, text=True,
        ).stdout.splitlines()
        importers = {line.split(":")[0] for line in hits}
        allowed = {
            str(self.ROOT / "score_final.py"),
            str(self.ROOT / "harness" / "preflight.py"),  # asserts the artifact is absent
            str(self.ROOT / "harness" / "loop.py"),       # writes the seal, never reads labels
            str(self.ROOT / "harness" / "regate.py"),    # re-writes the seal, never reads labels
        }
        self.assertTrue(importers <= allowed, f"unexpected holdout importers: {importers - allowed}")

    def test_the_seal_writers_never_read_a_label(self):
        """`loop.py` and `regate.py` are allowed to import holdout because they WRITE
        the seal. The distinction is the whole firewall, so it is asserted rather than
        trusted: neither may call the one function that returns labels."""
        for module in ("loop.py", "regate.py"):
            source = (self.ROOT / "harness" / module).read_text()
            self.assertIn("seal_run", source, f"{module} should write the seal")
            self.assertNotIn("extract_test_labels", source,
                             f"{module} must never read test labels")

    def test_no_module_under_harness_reads_test_labels(self):
        hits = subprocess.run(
            ["grep", "-rn", r"extract_test_labels(", "--include=*.py", str(self.ROOT / "harness")],
            capture_output=True, text=True,
        ).stdout.splitlines()
        # `holdout.py` defines it, and `executor.py` names it in the contract lint's ban
        # list -- which is the opposite of calling it.
        callers = [
            h for h in hits
            if "holdout.py" not in h.split(":")[0] and "_FORBIDDEN" not in h
        ]
        self.assertEqual(callers, [], f"test labels reached from inside harness/: {callers}")

    def test_nothing_in_the_harness_imports_score_final(self):
        """Its output must never re-enter the loop. If we improve the harness
        afterwards, we go back to test-blind."""
        hits = subprocess.run(
            ["grep", "-rnE", r"^\s*(import|from)\s+score_final", "--include=*.py",
             str(self.ROOT / "harness")],
            capture_output=True, text=True,
        ).stdout
        self.assertEqual(hits.strip(), "", "score_final must not be reachable from the loop")


class TestArithmeticAndAlignment(unittest.TestCase):
    def test_delta_is_per_metric_against_the_published_baseline(self):
        deltas = score_final.deltas_against_baseline(0.6700, 0.5400)
        self.assertAlmostEqual(deltas["GAUC"], 0.6700 - 0.6610, places=6)
        self.assertAlmostEqual(deltas["nDCG@5"], 0.5400 - 0.5282, places=6)
        self.assertAlmostEqual(
            sum(deltas.values()) / 2, ((0.6700 + 0.5400) / 2) - C.BASELINE_TEST["primary"],
            places=4,
        )

    def _csv(self, path: Path, rows: list[tuple]):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row_id", "user_id", "video_id", "score"])
            w.writerows(rows)

    def test_submission_must_have_one_contiguous_row_per_test_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.csv"
            self._csv(path, [(i, 1, 2, 0.5 + i) for i in range(4)])
            self.assertEqual(list(score_final.read_submission(path, 4)), [0.5, 1.5, 2.5, 3.5])
            with self.assertRaises(ValueError):
                score_final.read_submission(path, 5)  # wrong row count

            self._csv(path, [(0, 1, 2, 0.1), (2, 1, 3, 0.2)])
            with self.assertRaises(ValueError):
                score_final.read_submission(path, 2)  # row_id skips

            self._csv(path, [(0, 1, 2, "nan")])
            with self.assertRaises(ValueError):
                score_final.read_submission(path, 1)  # NaN


if __name__ == "__main__":
    unittest.main()
