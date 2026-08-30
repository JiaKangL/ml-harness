"""Acceptance tests for P6 -- stall escalation and the seed ensemble.

Autonomy is scored as the number of manual interventions, so the escalation path is
worth more than any single modelling idea: these tests are about whether the run can
rescue itself with nobody watching.

    ./.venv/bin/python -m unittest tests.test_critics -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness import config as C
from harness import critics
from harness import llm as LL
from harness import prompts
from harness.console import Console
from harness.evaluator import Evaluator
from harness.loop import Loop, LoopConfig
from harness.memory import FeatureInsightsLedger
from harness.types import GateDecision, Metrics, NodeStatus, Verdict

from tests.test_agent_loop import FAST_ROOT, LoopTestCase, reply
from tests.test_memory_logger import metrics as make_metrics
from tests.test_memory_logger import node as make_node


class TestTriggerAndIsolation(unittest.TestCase):
    def test_the_trigger_is_one_iteration_before_formal_convergence(self):
        """Firing at 3 would mean the run is already converged before a critique could
        produce anything. Firing at 2 lets the critique-driven attempt land as the
        third iteration and break the streak."""
        self.assertEqual(C.STALL_TRIGGER, C.N_CONVERGE - 1)
        self.assertEqual(C.STALL_TRIGGER, 2)

    def test_stall_count_reaches_the_trigger_at_exactly_two_non_improvements(self):
        from harness.data_guard import DataAPI
        from harness import scoring

        ev = Evaluator(DataAPI(), scoring.evaluate_sha256())
        history = [make_metrics(0.6016), make_metrics(0.6010)]
        self.assertEqual(ev.convergence(history).stalled_iterations, 1)
        history.append(make_metrics(0.6012))
        state = ev.convergence(history)
        self.assertEqual(state.stalled_iterations, 2)
        self.assertFalse(state.converged, "2 stalls fires the critics; it is not convergence")
        history.append(make_metrics(0.6011))
        self.assertTrue(ev.convergence(history).converged)

    def test_a_crash_is_not_a_non_improvement(self):
        """Three crashes in a row are a broken branch, not three failures to improve.
        Counting them as convergence would end the run with hours of budget left."""
        from harness.data_guard import DataAPI
        from harness import scoring

        ev = Evaluator(DataAPI(), scoring.evaluate_sha256())
        state = ev.convergence([make_metrics(0.6016), None, None, None])
        self.assertEqual(state.stalled_iterations, 0)
        self.assertFalse(state.converged)

    def test_the_critic_prompt_excludes_the_agents_reasoning(self):
        """A reviewer who reads the reasoning agrees with it. The isolation is the only
        thing that makes the critique worth anything."""
        trunk = make_node("n03", "n00", 3, technique="listwise_softmax")
        trunk.valid = make_metrics(0.6050)
        trunk.proposal = type(trunk.proposal)(
            hypothesis="AGENT_REASONING_SENTINEL: the loss is misaligned with the metric",
            axis="loss",
            grounding="metric",
            predicted_delta=0.004,
            code="print('trunk')",
            technique="listwise_softmax",
        )
        ledger = FeatureInsightsLedger()
        ledger.record(trunk, GateDecision(True, "ok", False, 0.004, 0.004, 0.004, 3))

        for role in prompts.CRITIC_ROLES:
            messages = critics.critic_messages(trunk, ledger, role)
            self.assertEqual([m["role"] for m in messages], ["user"],
                             "a critic sees no assistant turn -- the context is fresh")
            body = messages[0]["content"]
            self.assertNotIn("AGENT_REASONING_SENTINEL", body)
            # It still sees what it needs: the code, the verdicts, and the organizers'
            # own published mechanisms.
            self.assertIn("print('trunk')", body)
            self.assertIn("loss/listwise_softmax", body)
            self.assertIn("user_id x video_id", body)

    def test_the_ledger_keeps_published_mechanisms_but_drops_the_runs_own(self):
        ledger = FeatureInsightsLedger()
        node = make_node("n01", "n00", 1, technique="my_idea")
        node.valid = make_metrics(0.6050)
        ledger.record(node, GateDecision(True, "ok", False, 0.004, 0.004, 0.004, 3),
                      mechanism="RUN_AUTHORED_MECHANISM")
        full = ledger.render()
        isolated = ledger.render(include_mechanisms=False)
        self.assertIn("RUN_AUTHORED_MECHANISM", full)
        self.assertNotIn("RUN_AUTHORED_MECHANISM", isolated)
        self.assertIn("1.14M rows", isolated)  # the organizers' capacity mechanism


class TestEscalationInTheLoop(LoopTestCase):
    def test_critics_fire_on_a_stall_and_never_exceed_the_round_cap(self):
        loop = self.loop(critics=True, max_iters=6)
        fired: list[int] = []

        def fake_escalate(iteration):
            fired.append(iteration)
            loop._critic_rounds += 1
            return []

        loop._escalate = fake_escalate
        loop.run()
        self.assertLessEqual(len(fired), C.MAX_CRITIQUE_ROUNDS)
        self.assertGreaterEqual(len(fired), 1, "a stalling mock run must escalate")

    def test_a_full_stalling_run_records_zero_manual_interventions(self):
        loop = self.loop(critics=True, max_iters=4)
        summary = loop.run()
        self.assertEqual(summary.manual_interventions, 0)

    def test_a_critic_proposal_is_logged_with_its_origin_and_uses_the_same_gate(self):
        # Four iterations: the trigger needs two scored non-improvements before it
        # fires, so the critique lands on the third at the earliest.
        loop = self.loop(critics=True, max_iters=4)
        generation = loop.agent.validate(
            LL.LLMResponse(text=reply(technique="critic_idea"), usage=LL.TokenUsage())
        )
        loop._escalate = lambda iteration: [("A", "loss", generation)]
        loop.run()
        entries = json.loads(Path(loop.cfg.logs_json).read_text())["iterations"]
        critic_entries = [e for e in entries if e["source"].startswith("critic:")]
        self.assertTrue(critic_entries, "the critic's proposal must appear in the log")
        entry = critic_entries[0]
        self.assertEqual(entry["source"], "critic:A")
        if entry["gate"] is not None and entry["gate"]["promote"]:
            self.assertEqual(entry["gate"]["seeds_run"], len(C.CONFIRM_SEEDS),
                             "a critic-originated proposal promotes on 3 seeds, like any other")


class TestEnsemble(unittest.TestCase):
    def test_rank_averaging_k_noisy_scorers_beats_the_average_single_scorer(self):
        """The premise, tested on synthetic data rather than asserted: partly
        independent errors cancel, so the average ranking is better than the average
        ranking-quality of its members."""
        rng = np.random.default_rng(0)
        n, k = 4000, 5
        truth = rng.normal(size=n)
        members = [truth + rng.normal(0, 1.5, size=n) for _ in range(k)]

        def spearman(a, b):
            ra = np.argsort(np.argsort(a)).astype(float)
            rb = np.argsort(np.argsort(b)).astype(float)
            return float(np.corrcoef(ra, rb)[0, 1])

        singles = [spearman(m, truth) for m in members]
        ranks = sum(np.argsort(np.argsort(m)).astype(float) / (n - 1) for m in members)
        self.assertGreater(spearman(ranks, truth), max(singles))

    def test_the_ensemble_is_a_real_script_that_re_executes_standalone(self):
        """`build_submission` re-executes one stored source, so an in-memory average of
        k score arrays would have nothing to re-execute."""
        d = Path(tempfile.mkdtemp())
        score_dir = d / "scores"
        score_dir.mkdir(parents=True)
        for name in ("nA", "nB"):
            for split in ("valid", "test"):
                np.save(score_dir / f"{name}_{split}.npy",
                        np.linspace(0, 1, 50) + (0.1 if name == "nB" else 0.0))

        code = critics.ENSEMBLE_TEMPLATE.format(
            k=2, names="nA, nB", members=["nA", "nB"], score_dir=str(score_dir)
        )
        script = d / "ensemble.py"
        script.write_text(code)

        import subprocess
        import sys

        out = d / "out.npy"
        proc = subprocess.run(
            [sys.executable, str(script), "--split", "valid", "--seed", "42", "--out", str(out)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        scores = np.load(out)
        self.assertEqual(scores.shape, (50,))
        self.assertTrue(np.all(np.diff(scores) > 0), "a monotone input must stay monotone")

    def test_the_ensemble_script_passes_the_contract_lint(self):
        from harness.executor import check_syntax, lint_contract

        code = critics.ENSEMBLE_TEMPLATE.format(
            k=2, names="nA, nB", members=["nA", "nB"], score_dir="/tmp/x"
        )
        self.assertIsNone(check_syntax(code))
        self.assertIsNone(lint_contract(code))

    def test_only_confirmed_nodes_are_eligible(self):
        """An ensemble of unconfirmed candidates is an average of noise, so the
        eligibility rule is load-bearing rather than tidy."""
        import inspect

        source = inspect.getsource(critics.build_ensemble)
        self.assertIn("confirmed", source)
        caller = inspect.getsource(Loop._endgame)
        self.assertIn("NodeStatus.PROMOTED", caller)


if __name__ == "__main__":
    unittest.main()
