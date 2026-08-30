"""Acceptance tests for supervised execution.

Every test here corresponds to a way agent-written code has actually taken a harness
down. The headline one is `test_spawned_children_are_all_dead_after_a_kill`: it really
spawns worker processes, really hangs, and then asks `ps` whether they are gone. If
they are not, orphaned workers steal cores from the next iteration and quietly corrupt
every timing measurement the run reports afterwards.

The second-most important is `test_constant_scores_are_rejected`. That candidate runs
perfectly, exits 0, and writes a correctly-shaped float64 array. Nothing except output
validation can tell it apart from a working model.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from harness import config as C
from harness import executor as ex
from harness.data_guard import DataAPI
from harness.executor import Executor, HarnessDependencyError, check_imports, failure_signature
from harness.types import FailureClass

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def pid_alive(pid: int) -> bool:
    """`ps -p` rather than `os.kill(pid, 0)`: a zombie still answers signal 0, and a
    reaped-but-unwaited child is exactly the state we must not count as alive."""
    out = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,stat="], capture_output=True, text=True
    ).stdout.strip()
    if not out:
        return False
    return not out.split()[1].startswith("Z")


class ExecutorTestCase(unittest.TestCase):
    """One DataAPI and one workspace per class; the cache load is the slow part."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="exec_test_"))
        cls.data = DataAPI()
        cls.ex = Executor(workspace=cls.tmp, data=cls.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


# ------------------------------------------------------------------ process control


class TestProcessSupervision(ExecutorTestCase):
    def test_spawned_children_are_all_dead_after_a_kill(self):
        """THE headline test.

        `proc.kill()` signals only the direct child. A DataLoader's workers and a BLAS
        thread pool survive it, keep running for the rest of the session, and steal
        cores from every subsequent iteration -- so the timings we report afterwards
        are wrong and nothing raises. The executor spawns into a new session and kills
        the whole group; this asserts the group is actually empty afterwards.
        """
        result = self.ex.run(
            fixture("spawner_hang.py"), "orphan_reaping", "valid", 42, timeout_s=6
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.resources.killed_by, "timeout")

        pids_file = Path(str(result.stderr_path.parent / "scores.npy") + ".pids.json")
        self.assertTrue(pids_file.exists(), "fixture never recorded its worker PIDs")
        recorded = json.loads(pids_file.read_text())
        self.assertEqual(len(recorded["children"]), 3)

        time.sleep(0.5)  # give the kernel a moment to finish tearing the group down
        survivors = [p for p in [recorded["parent"], *recorded["children"]] if pid_alive(p)]
        self.assertEqual(survivors, [], f"orphaned processes survived the kill: {survivors}")

    def test_a_hung_script_still_yields_its_stderr_tail(self):
        """Proves files-not-PIPE, twice over.

        The fixture writes ~200 KB to stderr -- far past the ~64 KB pipe buffer where a
        polling parent deadlocks -- and then hangs. With `stderr=PIPE` the kill would
        also discard everything still buffered, so the timeout case, the one where the
        traceback matters most, would be the case that comes back empty.
        """
        result = self.ex.run(
            fixture("noisy_hang.py"), "timeout_captured", "valid", 42, timeout_s=6
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.cls, FailureClass.TIMEOUT)
        self.assertIn("LAST_LINE_BEFORE_HANG", result.failure.traceback_tail)
        self.assertGreater(result.stderr_path.stat().st_size, 64 * 1024)

    def test_timeout_is_enforced_within_a_couple_of_seconds(self):
        started = time.monotonic()
        result = self.ex.run(fixture("slow_candidate.py"), "timeout_prompt", "valid", 42,
                             timeout_s=4)
        elapsed = time.monotonic() - started
        self.assertEqual(result.failure.cls, FailureClass.TIMEOUT)
        self.assertLess(elapsed, 20, "the watchdog overshot its own deadline badly")
        self.assertGreaterEqual(result.resources.wall_seconds, 4)

    def test_resource_facts_are_populated_on_a_successful_run(self):
        result = self.ex.run(fixture("good_candidate.py"), "resources", "valid", 42)
        self.assertTrue(result.ok, result.failure)
        self.assertEqual(result.resources.exit_code, 0)
        self.assertIsNone(result.resources.killed_by)
        self.assertGreater(result.resources.wall_seconds, 0)
        self.assertGreater(result.resources.peak_rss_bytes, 0)

    def test_child_env_pins_reproducibility_and_the_repo_root(self):
        """The BLAS caps and PYTHONHASHSEED are set by the harness, never by generated
        code, so a candidate cannot make its own run irreproducible or oversubscribe
        the machine and wreck the next iteration's timings."""
        env = self.ex._child_env()
        self.assertEqual(env["PYTHONPATH"], str(C.ROOT))
        for key, value in C.CHILD_ENV.items():
            self.assertEqual(env[key], value)

    def test_the_rss_watchdog_kills_a_candidate_over_its_cap(self):
        """Out-of-band and at 1 Hz, over the whole process group.

        Not RLIMIT_AS: it is unreliable under the macOS ARM allocator, which reserves
        address space far beyond what it commits, so the limit fires on candidates that
        were never going to touch the memory and misses ones that do. The cap is
        lowered to 200 MB here; the real one is 9 GiB of 18 GB physical.
        """
        capped = Executor(workspace=self.tmp, data=self.data, preflight_imports=False,
                          rss_cap_bytes=200 * 1024 ** 2)
        result = capped.run(fixture("memory_hog.py"), "rss_cap", "valid", 42, timeout_s=30)
        self.assertFalse(result.ok)
        self.assertEqual(result.resources.killed_by, "rss")
        self.assertEqual(result.failure.cls, FailureClass.OOM)
        self.assertGreater(result.resources.peak_rss_bytes, 200 * 1024 ** 2)
        # The resource fact is what the loop feeds back to the LLM, so it has to be in
        # the message the agent sees, not only in the dataclass.
        self.assertIn("GiB", result.failure.traceback_tail)

    def test_a_crash_is_classified_runtime_with_the_failing_frame(self):
        result = self.ex.run(fixture("crasher.py"), "crash", "valid", 42)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.cls, FailureClass.RUNTIME)
        # The candidate is written to `candidate.py` in its own run directory, so
        # that -- not the fixture's name -- is the frame the agent sees.
        self.assertIn("candidate.py", result.failure.frame_context or "")
        self.assertIn("inner", result.failure.frame_context)
        self.assertIn("ValueError", result.failure.signature)


# ------------------------------------------------------------------ stage ordering


class TestCheapStagesRunFirst(ExecutorTestCase):
    def test_bad_syntax_never_reaches_a_subprocess(self):
        """A missing colon must cost a parse, not an interpreter launch and a process
        group. The stages are ordered cheapest-first precisely so that the failures a
        model makes most often are the ones that cost least."""
        bad = "def f(x)\n    return x\n"
        rec = self.ex.check_syntax(bad)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.cls, FailureClass.SYNTAX)

        launched = []
        original = subprocess.Popen
        try:
            subprocess.Popen = lambda *a, **k: launched.append(a) or original(*a, **k)
            self.assertIsNotNone(self.ex.check_syntax(bad))
        finally:
            subprocess.Popen = original
        self.assertEqual(launched, [], "check_syntax launched a subprocess")

    def test_valid_syntax_passes(self):
        self.assertIsNone(self.ex.check_syntax(fixture("good_candidate.py")))

    def test_syntax_failure_costs_no_iteration(self):
        self.assertFalse(FailureClass.SYNTAX.costs_an_iteration)
        self.assertFalse(FailureClass.CONTRACT.costs_an_iteration)
        self.assertTrue(FailureClass.RUNTIME.costs_an_iteration)


# ------------------------------------------------------------------ the lint


class TestContractLint(ExecutorTestCase):
    def assert_rejected(self, code: str, rule: str):
        rec = self.ex.lint_contract(code)
        self.assertIsNotNone(rec, "lint accepted a contract violation")
        self.assertEqual(rec.cls, FailureClass.CONTRACT)
        self.assertIn(rule, rec.traceback_tail,
                      f"the rejection must name the rule that was broken:\n{rec.traceback_tail}")
        return rec

    def test_the_reference_candidate_passes(self):
        self.assertIsNone(self.ex.lint_contract(fixture("good_candidate.py")))

    def test_torch_is_accepted_because_it_is_on_the_optional_list(self):
        """torch is in OPTIONAL_IMPORTS and installed. Rejecting it would make the
        lint contradict the prompt, which states the list verbatim."""
        self.assertIn("torch", C.OPTIONAL_IMPORTS)
        code = "import torch\nimport numpy as np\nx = torch.zeros(3)\n"
        self.assertIsNone(self.ex.lint_contract(code))

    def test_sklearn_is_rejected(self):
        """Not installed, not on the list. Caught statically, before a subprocess turns
        it into an ImportError the agent would misread as a broken idea."""
        self.assert_rejected(
            "from sklearn.linear_model import LogisticRegression\n", "import-allowlist"
        )

    def test_holdout_import_is_rejected(self):
        self.assert_rejected(
            "from harness.holdout import extract_test_labels\nlabels = extract_test_labels()\n",
            "no-test-labels",
        )

    def test_test_label_names_are_rejected_anywhere(self):
        for snippet in (
            "import numpy as np\ntest_labels = np.load('x.npy')\n",
            "from pathlib import Path\np = Path('cache/_holdout')\n",
            "import json\nd = {}\ny = d.holdout\n",
        ):
            with self.subTest(snippet=snippet.splitlines()[-1]):
                self.assert_rejected(snippet, "no-test-labels")

    def test_opening_the_raw_data_dir_is_rejected(self):
        """The acceptance case from the plan: a script that imports csv (which IS
        allowed) and opens the raw log. The import is legal; the open is not."""
        code = (
            "import csv\n"
            "with open('kuairand-starter-kit/KuaiRand-Pure/data/"
            "log_standard_4_22_to_5_08_pure.csv') as fh:\n"
            "    rows = list(csv.DictReader(fh))\n"
        )
        self.assert_rejected(code, "no-raw-data")

    def test_opening_any_csv_is_rejected(self):
        self.assert_rejected("f = open('log_standard.csv')\n", "no-raw-data")

    def test_raw_data_config_attributes_are_rejected(self):
        for snippet in (
            "from harness import config as C\np = C.DATA_DIR\n",
            "from harness import config\np = config.DATA_DIR\n",
            "from harness.config import DATA_DIR\np = DATA_DIR\n",
            "from harness import config as C\nfor f in C.LOG_FILES: pass\n",
            "from harness import config as C\np = C.RANDOM_LOG_FILE\n",
        ):
            with self.subTest(snippet=snippet.splitlines()[-1]):
                self.assert_rejected(snippet, "no-raw-data")

    def test_network_imports_are_rejected_by_name(self):
        """These are already off the allowlist, but the message matters: 'no network
        access' is a repairable instruction, while 'not on the allowlist' invites the
        model to try `http.client` on the next attempt."""
        for mod in ("socket", "urllib.request", "requests", "http.client"):
            with self.subTest(mod=mod):
                self.assert_rejected(f"import {mod}\n", "no-network")

    def test_comments_are_not_violations(self):
        """AST, not substring matching. A model that writes `# we never touch
        test_labels` is following the rules, not breaking them."""
        code = (
            "# deliberately avoids the holdout and never reads test_labels\n"
            "import numpy as np\n"
            "x = np.zeros(3)\n"
        )
        self.assertIsNone(self.ex.lint_contract(code))

    def test_every_violation_is_reported_not_just_the_first(self):
        """One round trip should fix all of them. Reporting one at a time turns a
        three-line mistake into three repair attempts and a tripped breaker."""
        code = "import requests\nimport sklearn\nx = test_labels\n"
        rec = self.ex.lint_contract(code)
        for rule in ("no-network", "import-allowlist", "no-test-labels"):
            self.assertIn(rule, rec.traceback_tail)

    def test_lint_defers_to_check_syntax_on_unparseable_code(self):
        self.assertIsNone(self.ex.lint_contract("def f(:\n"))


# ------------------------------------------------------------------ output validation


class TestOutputValidation(ExecutorTestCase):
    """The stage that separates 'it ran' from 'it worked'."""

    def setUp(self):
        self.path = self.tmp / "scores.npy"

    def save(self, arr) -> Path:
        np.save(self.path, arr)
        return self.path

    def test_good_scores_validate(self):
        rng = np.random.default_rng(0)
        self.save(rng.random(self.data.n_rows("valid")))
        self.assertIsNone(self.ex.validate_output(self.path, "valid"))

    def test_constant_scores_are_rejected(self):
        """A constant model runs perfectly and is worthless. GAUC and nDCG@5 are
        computed within user, so a score that never varies inside a user expresses no
        ranking at all -- it exits 0 and scores like a coin flip."""
        self.save(np.full(self.data.n_rows("valid"), 0.5))
        rec = self.ex.validate_output(self.path, "valid")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.cls, FailureClass.INVALID_OUTPUT)
        self.assertIn("constant-within-user", rec.traceback_tail)

    def test_per_user_constant_scores_are_rejected(self):
        """Subtler than a global constant and just as useless: scores that vary across
        users but not within them. The metric never compares two users."""
        groups = self.data.groups("valid")
        self.save(groups.astype(np.float64) / groups.max())
        rec = self.ex.validate_output(self.path, "valid")
        self.assertIsNotNone(rec)
        self.assertIn("constant-within-user", rec.traceback_tail)

    def test_nan_is_rejected(self):
        scores = np.random.default_rng(0).random(self.data.n_rows("valid"))
        scores[1234] = np.nan
        self.save(scores)
        rec = self.ex.validate_output(self.path, "valid")
        self.assertIn("non-finite", rec.traceback_tail)
        self.assertIn("1234", rec.traceback_tail)

    def test_inf_is_rejected(self):
        scores = np.random.default_rng(0).random(self.data.n_rows("valid"))
        scores[0] = np.inf
        self.save(scores)
        self.assertIn("non-finite", self.ex.validate_output(self.path, "valid").traceback_tail)

    def test_wrong_row_count_is_rejected(self):
        """Scores are positional, so a length mismatch misaligns every row, not just
        the missing ones."""
        self.save(np.random.default_rng(0).random(self.data.n_rows("valid") - 1))
        rec = self.ex.validate_output(self.path, "valid")
        self.assertIn("row-count", rec.traceback_tail)

    def test_scores_for_the_wrong_split_are_rejected(self):
        """The commonest form of the alignment bug: a script that ignores --split."""
        self.save(np.random.default_rng(0).random(self.data.n_rows("test")))
        self.assertIn("row-count", self.ex.validate_output(self.path, "valid").traceback_tail)

    def test_two_dimensional_output_is_rejected(self):
        n = self.data.n_rows("valid")
        self.save(np.random.default_rng(0).random((n, 1)))
        self.assertIn("shape", self.ex.validate_output(self.path, "valid").traceback_tail)

    def test_a_missing_file_is_rejected(self):
        missing = self.tmp / "nope.npy"
        rec = self.ex.validate_output(missing, "valid")
        self.assertEqual(rec.cls, FailureClass.INVALID_OUTPUT)
        self.assertIn("missing", rec.traceback_tail)

    def test_run_rejects_a_constant_candidate_end_to_end(self):
        """Proves `run` actually invokes validation: this script exits 0."""
        result = self.ex.run(fixture("constant_candidate.py"), "silent_wrong", "valid", 42)
        self.assertEqual(result.resources.exit_code, 0)
        self.assertFalse(result.ok, "a constant-score candidate was accepted")
        self.assertEqual(result.failure.cls, FailureClass.INVALID_OUTPUT)
        self.assertIsNone(result.scores_path)


# ------------------------------------------------------------------ smoke


class TestSmoke(ExecutorTestCase):
    def test_smoke_kills_a_five_minute_script_in_about_thirty_seconds(self):
        """The plan's acceptance bar: under 35s. The full-run timeout is 900s, so
        without this stage a script like this costs fifteen minutes to learn nothing."""
        self.assertEqual(C.SMOKE_TIMEOUT_S, 30)
        started = time.monotonic()
        result = self.ex.smoke(fixture("slow_candidate.py"), "smoke_slow")
        elapsed = time.monotonic() - started
        self.assertFalse(result.ok)
        self.assertLess(elapsed, 35, f"smoke took {elapsed:.1f}s")
        self.assertGreaterEqual(elapsed, C.SMOKE_TIMEOUT_S - 1)
        self.assertEqual(result.resources.killed_by, "timeout")

    def test_a_smoke_failure_is_classified_smoke_and_costs_no_iteration(self):
        """The stage, not the symptom, decides whether the repair is free -- so a
        crash during smoke must not be billed as a runtime failure."""
        result = self.ex.smoke(fixture("crasher.py"), "smoke_crash")
        self.assertEqual(result.failure.cls, FailureClass.SMOKE)
        self.assertFalse(result.failure.cls.costs_an_iteration)
        self.assertIn("ValueError", result.failure.signature)

    def test_smoke_passes_frac_and_still_requires_every_row(self):
        """`--frac` selects users for training; the script must still score all rows.
        A candidate that emits only the sampled users' rows fails right here."""
        result = self.ex.smoke(fixture("good_candidate.py"), "smoke_ok")
        self.assertTrue(result.ok, result.failure)
        self.assertEqual(np.load(result.scores_path).shape[0], self.data.n_rows("valid"))

    def test_smoke_does_not_apply_the_within_user_variance_check(self):
        """Structural checks only at 1% of users: a legitimate model can collapse to a
        constant on a tiny sample for reasons that say nothing about the code, and a
        smoke failure would send the agent off repairing a bug the harness invented."""
        n = self.data.n_rows("valid")
        path = self.tmp / "flat.npy"
        np.save(path, np.full(n, 0.5))
        self.assertIsNotNone(self.ex.validate_output(path, "valid"))
        self.assertIsNone(self.ex.validate_output(path, "valid", structural_only=True))


# ------------------------------------------------------------------ determinism


class TestDeterminism(ExecutorTestCase):
    def test_same_seed_twice_gives_identical_scores(self):
        """Not 'close': identical. The promotion gate resolves differences of 0.002,
        so run-to-run drift at the same seed would be indistinguishable from a real
        effect and would make every recorded delta unfalsifiable."""
        a = self.ex.run(fixture("good_candidate.py"), "determinism_a", "valid", 42)
        b = self.ex.run(fixture("good_candidate.py"), "determinism_b", "valid", 42)
        self.assertTrue(a.ok and b.ok, (a.failure, b.failure))
        np.testing.assert_array_equal(np.load(a.scores_path), np.load(b.scores_path))

    def test_a_different_seed_actually_changes_the_scores(self):
        """The other half: a script that ignores --seed would pass the test above
        trivially, and then a three-seed confirmation would measure one seed."""
        a = self.ex.run(fixture("good_candidate.py"), "determinism_a", "valid", 42)
        c = self.ex.run(fixture("good_candidate.py"), "determinism_c", "valid", 43)
        self.assertFalse(np.array_equal(np.load(a.scores_path), np.load(c.scores_path)))


# ------------------------------------------------------------------ signatures


class TestFailureSignature(unittest.TestCase):
    """The dedupe key behind the circuit breaker. The loop uses it to notice that
    three repair attempts produced the same failure three times, which is the most
    common way an agent harness turns its whole budget into nothing."""

    def test_line_numbers_and_paths_do_not_change_the_signature(self):
        a = failure_signature(
            "ValueError", "shapes (128,64) and (32,10) not aligned",
            'File "/Users/x/runs/iter_03/candidate.py", line 91, in fit',
        )
        b = failure_signature(
            "ValueError", "shapes (256,64) and (32,10) not aligned",
            'File "/private/tmp/runs/iter_07/candidate.py", line 214, in fit',
        )
        self.assertEqual(a, b)

    def test_memory_addresses_do_not_change_the_signature(self):
        a = failure_signature("TypeError", "unsupported operand for <Tensor at 0x7f9a1c30>")
        b = failure_signature("TypeError", "unsupported operand for <Tensor at 0x10ab44f0>")
        self.assertEqual(a, b)

    def test_genuinely_different_failures_stay_distinct(self):
        """Over-normalising is the opposite failure: it would collapse two unrelated
        bugs into one and stop the repair loop on the first."""
        a = failure_signature("ValueError", "shapes not aligned", "fit")
        b = failure_signature("KeyError", "shapes not aligned", "fit")
        c = failure_signature("ValueError", "index out of bounds", "fit")
        d = failure_signature("ValueError", "shapes not aligned", "predict")
        self.assertEqual(len({a, b, c, d}), 4)

    def test_signature_is_bounded(self):
        self.assertLessEqual(len(failure_signature("E", "x" * 5000, "y" * 5000)), 400)

    def test_traceback_parsing_finds_the_deepest_frame(self):
        text = (
            "Traceback (most recent call last):\n"
            '  File "/a/candidate.py", line 10, in <module>\n'
            "    main()\n"
            '  File "/a/candidate.py", line 7, in main\n'
            "    fit(x)\n"
            '  File "/a/model.py", line 42, in fit\n'
            "    w = a @ b\n"
            "ValueError: matmul: mismatched shapes\n"
        )
        exc, msg, frame = ex.parse_traceback(text)
        self.assertEqual(exc, "ValueError")
        self.assertEqual(msg, "matmul: mismatched shapes")
        self.assertEqual(frame, "model.py:42:fit")

    def test_unparseable_stderr_still_produces_a_usable_signature(self):
        exc, msg, frame = ex.parse_traceback("zsh: killed  python candidate.py\n")
        self.assertEqual(exc, "Unknown")
        self.assertTrue(msg)

    def test_the_same_crash_twice_has_the_same_signature(self):
        """End to end, through two real subprocesses in two different directories."""
        ex_ = Executor(workspace=Path(tempfile.mkdtemp(prefix="sig_")), preflight_imports=False,
                       data=DataAPI())
        try:
            a = ex_.run(fixture("crasher.py"), "sig_a", "valid", 42)
            b = ex_.run(fixture("crasher.py"), "sig_b", "valid", 43)
            self.assertEqual(a.failure.signature, b.failure.signature)
        finally:
            shutil.rmtree(ex_.workspace, ignore_errors=True)


# ------------------------------------------------------------------ library preflight


class TestImportPreflight(unittest.TestCase):
    def test_the_declared_libraries_all_import(self):
        versions = check_imports()
        for name in C.ALLOWED_IMPORTS + C.OPTIONAL_IMPORTS:
            self.assertIn(name, versions)

    def test_torch_is_present_because_the_prompt_promises_it(self):
        self.assertIn("torch", check_imports())

    def test_a_missing_library_is_a_harness_fault_not_a_candidate_bug(self):
        """The distinction is the whole point. If torch were missing and the agent had
        written torch code, an ImportError would read as a broken idea: the agent would
        record DISCARD against the technique and a whole research axis would die to one
        absent dependency."""
        with self.assertRaises(HarnessDependencyError) as ctx:
            check_imports(("numpy", "a_library_that_does_not_exist"))
        self.assertIn("a_library_that_does_not_exist", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, ValueError)

    def test_the_preflight_result_is_cached(self):
        first = check_imports()
        self.assertIs(first, check_imports())

    def test_argparse_is_on_the_allowlist_because_the_contract_is_a_cli(self):
        """Every candidate must parse `--split/--seed/--out`. Leaving argparse off the
        list would make the lint reject the one shape the prompt asks for."""
        self.assertIn("argparse", C.ALLOWED_IMPORTS)
        self.assertIn("--split", ex.CANDIDATE_CONTRACT)
        self.assertIn("--frac", ex.CANDIDATE_CONTRACT)


# ------------------------------------------------------------------ layering


class TestLayering(unittest.TestCase):
    def test_executor_imports_l1_only(self):
        """L3 may import config, types, data_guard and scoring. Importing memory,
        logger, agent, critics, loop or evaluator would invert the dependency graph --
        and the executor is the module the loop calls, so the cycle would be real."""
        source = (Path(ex.__file__)).read_text()
        imported = set(re.findall(r"^from \.(\w+) import|^from \. import (\w+)",
                                  source, re.M))
        names = {a or b for a, b in imported}
        self.assertTrue(names <= {"config", "types", "data_guard", "scoring"}, names)
        for forbidden in ("memory", "logger", "agent", "critics", "loop", "evaluator"):
            self.assertNotIn(f"from .{forbidden}", source)

    def test_the_executor_classifies_but_never_decides(self):
        """Ownership check. Regenerating code needs L5 and reverting a node needs L4,
        so the retry ladder cannot live here -- the executor returns a FailureRecord
        and the loop decides what happens next."""
        source = (Path(ex.__file__)).read_text()
        for word in ("NodeStatus", "MAX_SELF_HEAL_ATTEMPTS", "def repair", "def retry"):
            self.assertNotIn(word, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSubprocessDoorIsClosed(unittest.TestCase):
    """`os` is on the allowlist because candidates need os.environ, and it carries the
    whole subprocess family with it.

    Argument inspection does not close this: a literal path is caught, but
    `os.system('cat ' + d)` with a computed `d` is not, and no string analysis fixes
    that in general. So the call target is banned outright -- a candidate loads data
    through DataAPI and writes a .npy, and never has cause to start a process.
    """

    def test_computed_paths_do_not_evade_the_lint(self):
        for name, code in (
            ("os.system", "import os\nd='kuai'+'rand'\nos.system('cat '+d)"),
            ("os.popen", "import os\nos.popen('ls '+'/'.join(['a','b']))"),
            ("os.execv", "import os\nos.execv('/bin/sh',['sh'])"),
            ("os.fork", "import os\nos.fork()"),
            ("posix_spawn", "import os\nos.posix_spawn('/bin/sh',[],{})"),
        ):
            with self.subTest(call=name):
                fr = executor.lint_contract(code)
                self.assertIsNotNone(fr, f"{name} evaded the lint")
                self.assertIn("no-subprocess", fr.traceback_tail)

    def test_legitimate_os_use_still_passes(self):
        """The rule must not break what candidates legitimately need."""
        self.assertIsNone(
            executor.lint_contract("import os\nprint(os.environ.get('HOME'))")
        )
        self.assertIsNone(
            executor.lint_contract(
                "import numpy as np\nfrom harness.data_guard import DataAPI\n"
                "api = DataAPI()\nnp.save('out.npy', np.zeros(3))"
            )
        )
