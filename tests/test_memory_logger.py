"""Acceptance tests for P4 -- the state tree, the insight ledger, and the log.

The log is the primary graded deliverable and the tree is the run's resume path, so
these are tests about what survives an interruption and what cannot be rewritten
after the fact -- not tests that the classes exist.

    ./.venv/bin/python -m unittest tests.test_memory_logger -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from harness import config as C
from harness import logger as L
from harness.memory import (
    FeatureInsightsLedger,
    ImmutableNodeError,
    PUBLISHED_DEAD_ENDS,
    StateTree,
)
from harness.types import (
    FailureClass,
    FailureRecord,
    GateDecision,
    Metrics,
    Node,
    NodeStatus,
    Proposal,
    TokenUsage,
    Verdict,
)


def metrics(primary: float, seeds=(42,), std: float = 0.0) -> Metrics:
    return Metrics(
        gauc=primary + 0.03,
        ndcg5=primary - 0.03,
        primary=primary,
        users=22377,
        rows=124909,
        seeds=tuple(seeds),
        primary_std=std,
    )


def node(
    node_id: str,
    parent_id: str | None,
    iteration: int,
    axis: str = "loss",
    predicted: float = 0.003,
    technique: str = "listwise_softmax",
    status: NodeStatus = NodeStatus.PENDING,
) -> Node:
    return Node(
        node_id=node_id,
        parent_id=parent_id,
        iteration=iteration,
        proposal=Proposal(
            hypothesis=f"{technique} aligns the objective with the within-user metric",
            axis=axis,
            grounding="within_user_variance.tab",
            predicted_delta=predicted,
            code="print('x')",
            technique=technique,
        ),
        status=status,
        code_path=f"outputs/candidate_iter_{iteration:02d}.py",
        code_sha256="0" * 64,
    )


def gate(delta: float, seeds: int = 3, promote: bool | None = None, quarantined: bool = False):
    if promote is None:
        promote = seeds >= len(C.CONFIRM_SEEDS) and delta >= C.PROMOTE_DELTA
    return GateDecision(
        promote=promote,
        reason="test",
        quarantined=quarantined,
        delta_primary=delta,
        delta_gauc=delta,
        delta_ndcg5=delta,
        seeds_run=seeds,
    )


class TreeTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "state.jsonl"
        self.addCleanup(self.dir.cleanup)

    def seeded_tree(self, **kw) -> StateTree:
        tree = StateTree(self.path, resume=False, **kw)
        root = node("baseline", None, 0, axis="architecture", technique="fm_baseline")
        tree.add(root)
        tree.record_result("baseline", metrics(0.6016), [metrics(0.6016)])
        tree.promote("baseline")
        return tree


class TestStateTree(TreeTestCase):
    def test_scored_node_result_cannot_be_overwritten(self):
        tree = self.seeded_tree()
        with self.assertRaises(ImmutableNodeError):
            tree.update("baseline", valid=metrics(0.9))
        with self.assertRaises(ImmutableNodeError):
            tree.record_result("baseline", metrics(0.9), [metrics(0.9)])
        self.assertAlmostEqual(tree.get("baseline").valid.primary, 0.6016)

    def test_regression_leaves_the_trunk_pointer_unchanged(self):
        tree = self.seeded_tree()
        tree.add(node("n1", "baseline", 1))
        tree.record_result("n1", metrics(0.5990), [metrics(0.5990)])
        self.assertEqual(tree.trunk().node_id, "baseline")
        self.assertAlmostEqual(tree.trunk().valid.primary, 0.6016)

    def test_promotion_moves_the_pointer_without_mutating_the_old_trunk(self):
        tree = self.seeded_tree()
        before = tree.get("baseline")
        tree.add(node("n1", "baseline", 1))
        tree.record_result("n1", metrics(0.6050), [metrics(0.6050)])
        tree.promote("n1")
        self.assertEqual(tree.trunk().node_id, "n1")
        self.assertAlmostEqual(tree.get("baseline").valid.primary, before.valid.primary)

    def test_select_parent_never_returns_a_degraded_node(self):
        """Exploration must not be able to build on a corpse. Run with exploit
        probability 0 so every draw takes the exploration arm."""
        tree = self.seeded_tree(exploit_p=0.0)
        tree.add(node("failed", "baseline", 1, axis="sequence"))
        tree.update("failed", status=NodeStatus.FAILED)
        tree.add(node("pruned", "baseline", 2, axis="multitask"))
        tree.record_result("pruned", metrics(0.55), [metrics(0.55)])
        tree.prune_subtree("pruned", "clear regression")
        tree.add(node("quarantined", "baseline", 3, axis="debias"))
        tree.record_result("quarantined", metrics(0.95), [metrics(0.95)],
                           status=NodeStatus.QUARANTINED)
        for _ in range(50):
            self.assertNotIn(
                tree.select_parent().status,
                (NodeStatus.FAILED, NodeStatus.PRUNED, NodeStatus.QUARANTINED),
            )

    def test_prune_never_reaches_the_trunk_lineage(self):
        tree = self.seeded_tree()
        tree.add(node("n1", "baseline", 1))
        tree.record_result("n1", metrics(0.6050), [metrics(0.6050)])
        tree.promote("n1")
        pruned = tree.prune_subtree("baseline", "attempted prune of an ancestor")
        self.assertEqual(pruned, 0)
        self.assertEqual(tree.get("baseline").status, NodeStatus.PROMOTED)
        self.assertEqual(tree.trunk().node_id, "n1")

    def test_resume_is_exact(self):
        tree = self.seeded_tree()
        tree.add(node("n1", "baseline", 1))
        tree.record_result("n1", metrics(0.6050), [metrics(0.6050), metrics(0.6060)])
        tree.promote("n1")
        again = StateTree(self.path)
        self.assertEqual(len(again), 2)
        self.assertEqual(again.trunk().node_id, "n1")
        self.assertAlmostEqual(again.get("n1").valid.primary, 0.6050)
        self.assertEqual(len(again.get("n1").per_seed), 2)
        self.assertEqual(again.get("n1").proposal.technique, "listwise_softmax")

    def test_resume_survives_a_torn_final_line(self):
        """A run killed mid-fsync leaves half a JSON object. Half an object is not
        evidence of anything, so it is dropped rather than repaired."""
        tree = self.seeded_tree()
        tree.add(node("n1", "baseline", 1))
        tree.record_result("n1", metrics(0.6050), [metrics(0.6050)])
        raw = self.path.read_text()
        self.path.write_text(raw + '{"event":"add","node":{"node_i')
        again = StateTree(self.path)
        self.assertEqual(len(again), 2)
        self.assertAlmostEqual(again.get("n1").valid.primary, 0.6050)


class TestLedger(unittest.TestCase):
    def test_published_dead_ends_are_seeded_with_mechanisms(self):
        ledger = FeatureInsightsLedger()
        self.assertEqual(len(ledger), len(PUBLISHED_DEAD_ENDS))
        self.assertEqual(len(PUBLISHED_DEAD_ENDS), 3)
        for ins in ledger.insights:
            self.assertIs(ins.verdict, Verdict.DISCARD)
            self.assertGreater(len(ins.mechanism), 60, "a bare prohibition decays; the "
                                                       "mechanism is the point")
        rendered = ledger.render()
        self.assertIn("within", rendered)
        self.assertIn("DISCARD", rendered)

    def test_small_single_seed_gain_is_inconclusive_not_keep(self):
        """At sigma ~= 0.001 a +0.001 single-seed delta is 1 sigma. Recording it as
        KEEP would ship a fabricated fact into every later prompt."""
        ledger = FeatureInsightsLedger()
        n = node("n1", "baseline", 1)
        n.valid = metrics(0.6026, seeds=(42,))
        ins = ledger.record(n, gate(0.001, seeds=1))
        self.assertIs(ins.verdict, Verdict.INCONCLUSIVE)

    def test_confirmed_gain_is_keep_and_confirmed_loss_is_discard(self):
        ledger = FeatureInsightsLedger()
        good = node("n1", "baseline", 1, technique="listwise_softmax")
        good.valid = metrics(0.6050, seeds=(42, 43, 44))
        self.assertIs(ledger.record(good, gate(0.0034)).verdict, Verdict.KEEP)
        bad = node("n2", "baseline", 2, technique="pairwise_bpr")
        bad.valid = metrics(0.5980, seeds=(42, 43, 44))
        self.assertIs(ledger.record(bad, gate(-0.0036)).verdict, Verdict.DISCARD)

    def test_quarantine_is_always_discard(self):
        ledger = FeatureInsightsLedger()
        n = node("n1", "baseline", 1)
        n.valid = metrics(0.95, seeds=(42, 43, 44))
        ins = ledger.record(n, gate(0.35, quarantined=True))
        self.assertIs(ins.verdict, Verdict.DISCARD)

    def test_render_respects_a_token_budget(self):
        ledger = FeatureInsightsLedger()
        for i in range(40):
            n = node(f"n{i}", None, i, technique=f"technique_{i}")
            n.valid = metrics(0.60)
            ledger.record(n, gate(0.0))
        short = ledger.render(max_tokens=200)
        self.assertLess(len(short), 200 * 4 + 200)
        self.assertIn("elided", short)


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        d = Path(self.dir.name)
        self.jsonl, self.json_out = d / "it.jsonl", d / "it.json"
        self.log = L.IterationLogger(self.jsonl, self.json_out, resume=False)

    def entry(self, i: int, primary: float, predicted: float, parent_primary: float,
              source: str = "agent", tokens: TokenUsage | None = None,
              failures=(), diff: str = "") -> L.LogEntry:
        parent = node(f"p{i}", None, i - 1)
        parent.valid = metrics(parent_primary)
        parent.status = NodeStatus.PROMOTED
        child = node(f"n{i}", parent.node_id, i, predicted=predicted)
        child.valid = metrics(primary, seeds=(42, 43, 44), std=0.0004)
        child.per_seed = [metrics(primary), metrics(primary + 0.0002)]
        child.status = NodeStatus.SCORED
        child.change_summary = "Replaced pointwise logloss with a within-user softmax"
        child.failures = list(failures)
        child.tokens = tokens or TokenUsage(1000, 400, 6000, 0, 12.0, 0.02)
        return L.make_entry(child, parent, gate=gate(primary - parent_primary),
                            diff=diff, source=source)

    def test_realised_delta_is_computed_against_the_parent(self):
        e = self.entry(1, primary=0.6050, predicted=0.004, parent_primary=0.6016)
        self.assertAlmostEqual(e.realised_delta, 0.0034, places=6)

    def test_every_entry_carries_the_graded_fields(self):
        self.log.log(self.entry(1, 0.6050, 0.004, 0.6016, diff="--- a\n+++ b\n"))
        written = json.loads(self.json_out.read_text())["iterations"][0]
        for key in (
            "hypothesis", "axis", "grounding", "grounding_verified", "predicted_delta",
            "realised_delta", "change_summary", "diff", "is_rewrite", "metrics",
            "per_seed", "gate", "errors", "resources", "tokens", "source",
        ):
            self.assertIn(key, written)
        self.assertIn("gauc", written["metrics"])
        self.assertIn("ndcg5", written["metrics"])
        self.assertEqual(written["metrics"]["n_seeds"], 3)

    def test_diff_is_never_truncated_in_the_deliverable(self):
        big = "".join(f"+line {i}\n" for i in range(500))
        self.log.log(self.entry(1, 0.6050, 0.004, 0.6016, diff=big))
        written = json.loads(self.json_out.read_text())["iterations"][0]
        self.assertEqual(written["diff"], big)

    def test_rewrite_is_labelled(self):
        a = "".join(f"line {i}\n" for i in range(200))
        b = "".join(f"other {i}\n" for i in range(150))
        _, is_rewrite = L.compute_diff(a, b)
        self.assertTrue(is_rewrite)
        _, is_edit = L.compute_diff(a, a.replace("line 7", "line seven"))
        self.assertFalse(is_edit)

    def test_repair_and_critic_tokens_are_in_the_cumulative_total(self):
        """The two call classes easiest to forget, and exactly the two that inflate
        the bill the Feasibility criterion is scored on."""
        self.log.log(self.entry(1, 0.6050, 0.004, 0.6016,
                                tokens=TokenUsage(1000, 400, 0, 0, 10.0, 0.02)))
        self.log.log(self.entry(2, 0.6030, 0.002, 0.6050, source="repair",
                                tokens=TokenUsage(500, 200, 0, 0, 5.0, 0.01),
                                failures=[FailureRecord(FailureClass.RUNTIME, "sig",
                                                        "NameError: x", None, 1)]))
        self.log.log(self.entry(3, 0.6060, 0.003, 0.6050, source="critic:A",
                                tokens=TokenUsage(800, 300, 0, 0, 8.0, 0.015)))
        total = self.log.total_tokens()
        self.assertEqual(total.prompt_tokens, 2300)
        self.assertEqual(total.completion_tokens, 900)
        self.assertAlmostEqual(total.cost_usd, 0.045)

    def test_run_summary_reports_what_the_rubric_asks_for(self):
        for i in range(1, 4):
            self.log.log(self.entry(i, 0.60 + 0.001 * i, 0.002 * i, 0.6016))
        summary = self.log.finalize(converged=True, convergence_iteration=3)
        self.assertEqual(summary.iterations, 3)
        self.assertEqual(summary.manual_interventions, 0)
        self.assertGreater(summary.total_tokens.prompt_tokens, 0)
        self.assertGreaterEqual(summary.wall_clock_seconds, 0.0)
        self.assertTrue(summary.converged)
        on_disk = json.loads(self.json_out.read_text())["run"]["summary"]
        self.assertIn("total_tokens", on_disk)
        self.assertIn("cache_hit_rate", on_disk)

    def test_calibration_is_computed_from_three_or_more_scored_entries(self):
        self.log.log(self.entry(1, 0.6020, 0.001, 0.6016))
        self.assertIsNone(self.log.calibration())
        self.log.log(self.entry(2, 0.6050, 0.004, 0.6016))
        self.log.log(self.entry(3, 0.6080, 0.007, 0.6016))
        r = self.log.calibration()
        self.assertIsNotNone(r)
        self.assertGreater(r, 0.9, "monotone predictions must correlate positively")

    def test_critic_origin_is_visible(self):
        self.log.log(self.entry(1, 0.6050, 0.004, 0.6016, source="critic:A"))
        self.assertEqual(self.log.entries[0]["source"], "critic:A")

    def test_manual_interventions_are_recorded_with_a_reason(self):
        self.log.record_intervention("operator restarted the run by hand")
        run = json.loads(self.json_out.read_text())["run"]
        self.assertEqual(run["manual_interventions"], 1)
        self.assertIn("reason", run["manual_intervention_records"][0])

    def test_a_kill_during_the_render_leaves_the_previous_file_intact(self):
        """`os.replace` onto the deliverable is the only write that touches it, so an
        interrupted render leaves the last complete array in place."""
        self.log.log(self.entry(1, 0.6050, 0.004, 0.6016))
        before = self.json_out.read_text()
        real_replace = os.replace

        def die(*a, **kw):
            raise KeyboardInterrupt("killed mid-render")

        os.replace = die
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.log.log(self.entry(2, 0.6060, 0.004, 0.6050))
        finally:
            os.replace = real_replace
        self.assertEqual(self.json_out.read_text(), before)

    def test_resume_reloads_entries_and_survives_a_torn_line(self):
        self.log.log(self.entry(1, 0.6050, 0.004, 0.6016))
        self.log.log(self.entry(2, 0.6060, 0.004, 0.6050))
        self.jsonl.write_text(self.jsonl.read_text() + '{"iteration":3,"nod')
        again = L.IterationLogger(self.jsonl, self.json_out)
        self.assertEqual(len(again.entries), 2)
        self.assertEqual(again.resume(), 2)


if __name__ == "__main__":
    unittest.main()
