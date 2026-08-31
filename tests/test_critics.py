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

    def test_convergence_waits_for_the_critique_it_generated(self):
        """The organizers' rule fires at three non-improving iterations; the critics
        fire at two. So the first critique-driven attempt *is* the third, and obeying
        convergence there ends the run with critiques generated, paid for, and never
        tested. The first live run did exactly that: two of three critic proposals died
        in the queue with eleven iterations of budget unspent."""
        loop = self.loop(critics=True, max_iters=6)

        # Converged, but a critique is still queued -> the run must not stop.
        loop._critique_queue = [("B", "multitask", object())]
        loop._critic_rounds = C.MAX_CRITIQUE_ROUNDS
        self.assertTrue(loop._critique_pending())

        # Nothing queued but a round still unspent -> still not finished.
        loop._critique_queue = []
        loop._critic_rounds = 0
        self.assertTrue(loop._critique_pending())

        # Queue drained and every round spent -> convergence may now be honoured.
        loop._critic_rounds = C.MAX_CRITIQUE_ROUNDS
        self.assertFalse(loop._critique_pending())

        # And with critics switched off it never blocks convergence.
        loop.cfg.critics = False
        loop._critique_queue = [("B", "multitask", object())]
        self.assertFalse(loop._critique_pending())

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


class TestEnsemble(LoopTestCase):
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

    def test_only_confirmed_nodes_reach_the_ensemble(self):
        """An ensemble of unconfirmed candidates is an average of noise, so the
        eligibility rule is load-bearing rather than tidy.

        Checked by watching what the endgame actually hands over, not by reading the
        source: a source-text assertion passes or fails on how the code is spelled.
        """
        loop = _EnsembleLoop().build(self)
        handed: list = []
        original = critics.build_ensemble

        def capture(lp, confirmed, iteration):
            handed.extend(confirmed)
            return None

        critics.build_ensemble = capture
        try:
            loop._endgame(9)
        finally:
            critics.build_ensemble = original

        self.assertTrue(handed, "the endgame must offer the confirmed nodes")
        for node in handed:
            self.assertIs(node.status, NodeStatus.PROMOTED)
            self.assertIsNotNone(node.valid)
        self.assertNotIn("scored_not_promoted", [n.node_id for n in handed])
        self.assertNotIn("failed_node", [n.node_id for n in handed])

    def test_fewer_than_two_members_produces_no_ensemble(self):
        """Averaging one thing is that thing. Returning None rather than a degenerate
        one-member 'ensemble' keeps the log honest about what was actually done."""
        loop = _EnsembleLoop().build(self, promoted=1)
        confirmed = [n for n in loop.tree.nodes if n.status is NodeStatus.PROMOTED]
        self.assertIsNone(critics.build_ensemble(loop, confirmed, 9))


class _EnsembleLoop:
    """Builds a loop whose tree already holds a mix of node statuses."""

    def build(self, case: LoopTestCase, promoted: int = 2):
        loop = case.loop(critics=True, max_iters=1)
        # The root is promoted by _seed_root; add the rest by hand so the tree carries
        # one of every status the endgame has to discriminate between.
        loop._seed_root()
        for i in range(promoted - 1):
            nid = f"promoted_{i}"
            loop.tree.add(_child(loop, nid, i + 1))
            loop.tree.record_result(nid, make_metrics(0.61 + 0.001 * i),
                                    [make_metrics(0.61 + 0.001 * i)])
            loop.tree.promote(nid)
        loop.tree.add(_child(loop, "scored_not_promoted", 8))
        loop.tree.record_result("scored_not_promoted", make_metrics(0.605),
                                [make_metrics(0.605)])
        loop.tree.add(_child(loop, "failed_node", 9))
        loop.tree.update("failed_node", status=NodeStatus.FAILED)
        return loop


def _child(loop, node_id: str, iteration: int):
    from harness.types import Node, Proposal

    trunk = loop.tree.trunk()
    return Node(
        node_id=node_id,
        parent_id=trunk.node_id,
        iteration=iteration,
        proposal=Proposal("a hypothesis of sufficient length to be a real one here",
                          "loss", "metric", 0.003, trunk.proposal.code, f"t_{node_id}"),
        status=NodeStatus.PENDING,
        code_path=trunk.code_path,
        code_sha256="0" * 64,
    )


if __name__ == "__main__":
    unittest.main()
