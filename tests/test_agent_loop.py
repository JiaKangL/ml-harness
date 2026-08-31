"""Acceptance tests for P5 -- prompts, the client, proposal validation, the loop.

Two families. The cheap ones assert the rules that cost nothing to enforce and
everything to get wrong: the frozen prefix, the cache TTL, what is rejected before
execution and what is deliberately not. The end-to-end ones drive the real loop in mock
mode over a two-second root, which exercises L2-L6 in seconds with zero tokens.

    ./.venv/bin/python -m unittest tests.test_agent_loop -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import agent as A
from harness import config as C
from harness import llm as LL
from harness import prompts
from harness.console import Console
from harness.loop import Loop, LoopConfig
from harness.types import PRIORITY_AXES, NodeStatus, TokenUsage

FIXTURES = Path(__file__).resolve().parent / "fixtures"
#: A two-second root for the loop tests. The FM baseline is the root of a real run --
#: it is the incumbent we are scored against -- but paying 50s of FM per test buys no
#: extra coverage of the orchestration, which is what these tests are about.
FAST_ROOT = FIXTURES / "fast_root.py"


def reply(**kw) -> str:
    fields = dict(
        hypothesis="A hypothesis long enough to pass the minimum-length check, stating "
                   "what changes and through what mechanism it acts on the ordering.",
        axis="loss",
        technique="listwise_softmax",
        grounding="within_user_variance.valid.tab",
        predicted=0.004,
        summary="Replaced pointwise logloss with a within-user softmax",
        code=LL.MOCK_SCRIPTS["popularity"],
    )
    fields.update(kw)
    return LL._mock_reply(**fields)


class TestPrompt(unittest.TestCase):
    def test_tier_a_is_byte_identical_across_assemblies(self):
        """One byte invalidates everything after it, and the miss raises nothing -- it
        shows up only on the bill. So this is asserted, not assumed."""
        prompts.static_prefix.cache_clear()
        first = prompts.static_prefix()
        for _ in range(4):
            prompts.static_prefix.cache_clear()
            self.assertEqual(prompts.static_prefix(), first)

    def test_tier_a_carries_no_per_turn_state(self):
        """The prefix must not name the turn it belongs to. A single interpolated
        iteration number or timestamp invalidates the whole cached prefix every call --
        the exact failure that shows up nowhere except the bill."""
        prefix = prompts.static_prefix()
        self.assertNotRegex(prefix, r"# Iteration \d")
        self.assertNotRegex(prefix, r"\b20\d\d-\d\d-\d\d\b")
        self.assertNotIn("## Your parent node", prefix)
        self.assertNotIn("Current best", prefix)

    def test_tier_a_includes_the_metric_verbatim_and_the_dead_ends(self):
        prefix = prompts.static_prefix()
        self.assertIn("def evaluate(user_ids, labels, scores, k=5)", prefix)
        self.assertIn("0.5940", prefix)  # the static-feature dead end
        self.assertIn("0.5895", prefix)  # the capacity dead end
        self.assertIn("exactly zero", prefix)  # the user-side dead end
        self.assertIn("within_user_variance", prefix)


class TestCacheRequestShape(unittest.TestCase):
    """The one thing that cannot be checked without a request: what we actually send."""

    def test_request_carries_a_one_hour_ttl_not_bare_ephemeral(self):
        captured = {}

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get_final_message(self):
                class U:
                    input_tokens, output_tokens = 100, 50
                    cache_read_input_tokens, cache_creation_input_tokens = 900, 0

                class M:
                    content = [type("B", (), {"type": "text", "text": reply()})()]
                    usage = U()
                    stop_reason = "end_turn"
                    model = C.MODEL

                return M()

        class FakeMessages:
            def stream(self, **kwargs):
                captured.update(kwargs)
                return FakeStream()

        client = LL.AnthropicLLM.__new__(LL.AnthropicLLM)
        client._anthropic = __import__("anthropic")
        client.client = type("C", (), {"messages": FakeMessages()})()
        client.model, client.max_tokens, client.max_retries = C.MODEL, 4096, 0

        response = client.complete("PREFIX", [{"role": "user", "content": "go"}])
        cache_control = captured["system"][0]["cache_control"]
        self.assertEqual(cache_control, {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(captured["thinking"], {"type": "adaptive"})
        self.assertEqual(response.usage.cache_read_tokens, 900)
        self.assertGreater(response.usage.cost_usd, 0)

    def test_operator_instruction_is_a_message_not_a_prefix_edit(self):
        """Editing the top-level system field to add a per-turn instruction would
        invalidate the cached prefix for every call after it."""
        captured = {}

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get_final_message(self):
                class M:
                    content = []
                    usage = type("U", (), {"input_tokens": 1, "output_tokens": 1,
                                           "cache_read_input_tokens": 0,
                                           "cache_creation_input_tokens": 0})()
                    stop_reason = "end_turn"
                    model = C.MODEL

                return M()

        client = LL.AnthropicLLM.__new__(LL.AnthropicLLM)
        client._anthropic = __import__("anthropic")
        client.client = type("C", (), {"messages": type("M", (), {
            "stream": lambda self, **kw: (captured.update(kw), FakeStream())[1]})()})()
        client.model, client.max_tokens, client.max_retries = C.MODEL, 4096, 0

        client.complete("PREFIX", [{"role": "user", "content": "go"}], operator="do X")
        self.assertEqual(captured["system"][0]["text"], "PREFIX")
        self.assertEqual(captured["messages"][-1], {"role": "system", "content": "do X"})


class TestBackendConfiguration(unittest.TestCase):
    """The credential and endpoint are environment-driven, because the key that runs
    this may be issued by a gateway rather than by Anthropic."""

    def build(self, **env):
        """Construct the client with `config` patched, without touching the network."""
        patches = [mock.patch.object(C, k, v) for k, v in env.items()]
        for p_ in patches:
            p_.start()
            self.addCleanup(p_.stop)
        captured = {}
        real = LL.AnthropicLLM.__init__

        import anthropic

        orig = anthropic.Anthropic

        def spy(**kwargs):
            captured.update(kwargs)
            return orig(api_key="sk-test-placeholder")

        anthropic.Anthropic = spy
        try:
            llm = LL.AnthropicLLM()
        finally:
            anthropic.Anthropic = orig
        return llm, captured

    def test_a_generic_key_is_used_without_touching_anthropic_api_key(self):
        _, kwargs = self.build(
            LLM_API_KEY="gateway-key", LLM_AUTH_TOKEN=None, LLM_BASE_URL=None
        )
        self.assertEqual(kwargs.get("api_key"), "gateway-key")
        self.assertNotIn("base_url", kwargs)

    def test_a_bearer_token_gateway_is_supported(self):
        """Most gateway credentials arrive on Authorization, not x-api-key."""
        _, kwargs = self.build(
            LLM_API_KEY=None, LLM_AUTH_TOKEN="bearer-abc", LLM_BASE_URL=None
        )
        self.assertEqual(kwargs.get("auth_token"), "bearer-abc")
        self.assertNotIn("api_key", kwargs)

    def test_an_alternate_endpoint_is_passed_through(self):
        _, kwargs = self.build(
            LLM_API_KEY="k", LLM_AUTH_TOKEN=None, LLM_BASE_URL="https://gateway.example/v1"
        )
        self.assertEqual(kwargs.get("base_url"), "https://gateway.example/v1")

    def test_an_api_key_wins_over_a_token_when_both_are_set(self):
        _, kwargs = self.build(
            LLM_API_KEY="k", LLM_AUTH_TOKEN="t", LLM_BASE_URL=None
        )
        self.assertEqual(kwargs.get("api_key"), "k")
        self.assertNotIn("auth_token", kwargs)

    def test_a_missing_credential_names_every_variable_that_would_fix_it(self):
        for attr in ("LLM_API_KEY", "LLM_AUTH_TOKEN", "LLM_BASE_URL"):
            p_ = mock.patch.object(C, attr, None)
            p_.start()
            self.addCleanup(p_.stop)
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            p_ = mock.patch.dict(os.environ, {}, clear=False)
            p_.start()
            self.addCleanup(p_.stop)
            os.environ.pop(var, None)
        with self.assertRaises(LL.LLMError) as ctx:
            LL.AnthropicLLM()
        message = str(ctx.exception)
        self.assertIn("HARNESS_LLM_API_KEY", message)
        self.assertIn("HARNESS_LLM_BASE_URL", message)
        self.assertIn("--mock", message)

    def test_the_model_id_is_overridable_for_a_gateway(self):
        """A gateway may expose the same model under a different id."""
        self.assertEqual(C.MODEL, os.environ.get("HARNESS_LLM_MODEL", "claude-opus-5"))
        self.assertEqual(C.CRITIC_MODEL, C.MODEL)


class TestProposalValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = A.Agent(LL.MockLLM())

    def parse(self, text: str):
        return self.agent.validate(LL.LLMResponse(text=text, usage=TokenUsage()))

    def test_missing_predicted_delta_is_rejected_before_execution(self):
        text = reply().replace("<predicted_delta>0.004</predicted_delta>", "")
        with self.assertRaises(A.ProposalRejected) as ctx:
            self.parse(text)
        self.assertEqual(ctx.exception.rule, "no_predicted_delta")

    def test_axis_outside_the_closed_set_is_rejected(self):
        with self.assertRaises(A.ProposalRejected) as ctx:
            self.parse(reply(axis="vibes"))
        self.assertEqual(ctx.exception.rule, "bad_axis")

    def test_a_paraphrased_grounding_key_resolves_and_still_runs(self):
        generation = self.parse(reply(grounding="tab_variance"))
        self.assertTrue(generation.grounding_verified)
        self.assertIn("variance", generation.proposal.grounding)

    def test_an_unresolvable_grounding_key_is_advisory_not_blocking(self):
        """Refusing to run would spend a real iteration punishing a spelling mistake.
        The flag makes the rate visible instead, which is the honest way to report it."""
        generation = self.parse(reply(grounding="qqq_not_a_field_zzz"))
        self.assertFalse(generation.grounding_verified)
        self.assertTrue(generation.proposal.code)

    def test_a_fields_only_diff_is_rejected_without_running(self):
        parent = "FIELDS = ['user_id', 'video_id']\nx = 1\n"
        child = "FIELDS = ['user_id', 'video_id', 'author_id', 'music_id']\nx = 1\n"
        with self.assertRaises(A.ProposalRejected) as ctx:
            self.agent.check_dead_end(parent, child)
        self.assertEqual(ctx.exception.rule, "known_dead_end")
        self.assertIn("0.5940", ctx.exception.message)

    def test_a_capacity_only_diff_is_rejected_without_running(self):
        with self.assertRaises(A.ProposalRejected):
            self.agent.check_dead_end("k = 16\ny = 2\n", "k = 32\ny = 2\n")

    def test_an_identical_script_is_rejected(self):
        with self.assertRaises(A.ProposalRejected):
            self.agent.check_dead_end("a = 1\n", "a = 1\n")

    def test_a_real_change_is_not_rejected(self):
        self.agent.check_dead_end(
            "loss = logloss(z, y)\nk = 16\n",
            "loss = listwise_softmax(z, y, groups)\nk = 16\n",
        )

    def test_architecture_is_locked_until_the_priority_axes_are_scored(self):
        with self.assertRaises(A.ProposalRejected) as ctx:
            self.agent.check_axis_lock("architecture", {"loss", "sequence"})
        self.assertEqual(ctx.exception.rule, "axis_locked")
        self.assertIn("multitask", ctx.exception.message)
        # ...and unlocked once every one of them has an observation.
        self.agent.check_axis_lock(
            "architecture", {"loss", "sequence", "multitask", "watchtime"}
        )

    def test_an_inconclusive_ledger_entry_does_not_block_a_retry(self):
        """A KEEP or DISCARD is settled knowledge; INCONCLUSIVE is the absence of it,
        and refusing to look again would freeze the run at its noisiest point."""
        from harness.memory import FeatureInsightsLedger
        from harness.types import Insight, Verdict

        ledger = FeatureInsightsLedger(seed_dead_ends=False)
        ledger._put(Insight("loss", "listwise_softmax", Verdict.INCONCLUSIVE,
                            0.001, 0.001, 0.001, 1, "noise", ("n1",)))
        proposal = self.parse(reply()).proposal
        self.agent.check_ledger(proposal, ledger.get)  # does not raise

        ledger._put(Insight("loss", "listwise_softmax", Verdict.DISCARD,
                            -0.004, 0.0, 0.0, 3, "measured worse", ("n1",)))
        with self.assertRaises(A.ProposalRejected) as ctx:
            self.agent.check_ledger(proposal, ledger.get)
        self.assertEqual(ctx.exception.rule, "already_settled")

    def test_usage_is_captured_on_every_call_including_repairs(self):
        llm = LL.MockLLM()
        first = llm.complete("prefix", [{"role": "user", "content": "go"}])
        repair = llm.complete(
            "prefix", [{"role": "user", "content": "## Your script failed. Repair attempt 1"}]
        )
        for response in (first, repair):
            self.assertGreater(response.usage.prompt_tokens, 0)
            self.assertGreater(response.usage.completion_tokens, 0)


class LoopTestCase(unittest.TestCase):
    #: One DataAPI for every loop test in the process. See Loop.__init__: a fresh one
    #: per test accumulates hundreds of megabytes that unittest never releases.
    _data = None

    @classmethod
    def shared_data(cls):
        if LoopTestCase._data is None:
            from harness.data_guard import DataAPI

            LoopTestCase._data = DataAPI()
        return LoopTestCase._data

    def loop(self, llm=None, **overrides) -> Loop:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        cfg = LoopConfig(
            mock=True,
            skip_preflight=True,
            resume=False,
            seal=False,
            critics=False,
            unbiased_check=False,
            enforce_axis_lock=False,
            cross_check_valid=False,
            build_submission=False,
            root_seeds=(42,),
            confirm_seeds=(42, 43, 44),
            seed_script=FAST_ROOT,
            workspace=d / "runs",
            outputs_dir=d / "outputs",
            state_path=d / "state.jsonl",
            logs_jsonl=d / "it.jsonl",
            logs_json=d / "it.json",
            submission_csv=d / "submission.csv",
            best_model_py=d / "best.py",
            max_iters=3,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return Loop(cfg, llm=llm, console=Console(quiet=True), data=self.shared_data())


class TestAxisSelection(LoopTestCase):
    """The search order over research directions."""

    def chooser(self, attempts: dict[str, int], realised: dict[str, float]):
        loop = self.loop(max_iters=1)
        loop.tree.axis_attempts = lambda: dict(attempts)
        loop.ledger.realised_by_axis = lambda: dict(realised)
        return loop

    def test_the_first_four_iterations_probe_each_priority_axis(self):
        """Without forced seeding the run commits to whichever axis happened to go
        first, and the ledger never gets an observation on the other three."""
        seen, attempts = [], {}
        loop = self.chooser(attempts, {})
        for _ in range(len(PRIORITY_AXES)):
            axis = loop._choose_axis()
            seen.append(axis)
            attempts[axis] = attempts.get(axis, 0) + 1
        self.assertEqual(sorted(seen), sorted(PRIORITY_AXES))

    def test_a_sub_threshold_gain_does_not_buy_an_axis_a_monopoly(self):
        """The whole point of the exploration bonus, and why its weight is the
        promotion bar. `loss` here has been tried six times for a best result of
        +0.001 -- real-looking, but half the bar and well inside the noise. Plain
        argmax over realised delta would ride it for the rest of the run; the bonus
        sends the next turn to an axis with one observation instead."""
        attempts = {a: 1 for a in Loop.SEARCH_AXES}
        attempts["loss"] = 6
        loop = self.chooser(attempts, {"loss": 0.001})
        chosen = {loop._choose_axis() for _ in range(40)}
        self.assertNotIn("loss", chosen, "a sub-threshold gain must not monopolise")

    def test_a_gain_worth_more_than_the_bar_is_exploited(self):
        """The other half of the trade. +0.004 is twice the promotion bar, so
        continuing to spend turns elsewhere would be exploration for its own sake."""
        attempts = {a: 1 for a in Loop.SEARCH_AXES}
        attempts["loss"] = 6
        loop = self.chooser(attempts, {"loss": 0.004})
        self.assertEqual({loop._choose_axis() for _ in range(40)}, {"loss"})

    def test_a_clearly_better_axis_is_still_preferred_once_evidence_accumulates(self):
        """Exploration must decay, or the run never exploits what it learned."""
        attempts = {a: 4 for a in Loop.SEARCH_AXES}
        loop = self.chooser(attempts, {"loss": 0.02, "temporal": 0.0})
        picks = [loop._choose_axis() for _ in range(40)]
        self.assertEqual(set(picks), {"loss"})

    def test_the_endgame_and_locked_axes_are_not_in_the_search_space(self):
        self.assertNotIn("ensemble", Loop.SEARCH_AXES)
        self.assertNotIn("architecture", Loop.SEARCH_AXES)


class TestLoopEndToEnd(LoopTestCase):
    def test_mock_run_produces_one_log_entry_per_iteration_plus_the_baseline(self):
        loop = self.loop(max_iters=3)
        summary = loop.run()
        entries = json.loads(Path(loop.cfg.logs_json).read_text())["iterations"]
        self.assertGreaterEqual(len(entries), 4)
        self.assertEqual(entries[0]["source"], "baseline")
        self.assertEqual(summary.manual_interventions, 0)
        for entry in entries:
            self.assertTrue(entry["hypothesis"])
            self.assertIn("predicted_delta", entry)
            self.assertIn("tokens", entry)

    def test_a_crashing_candidate_is_repaired_without_costing_an_iteration(self):
        """The crasher fails at smoke -- a defect in the writing, not in the idea -- so
        the repair happens and the turn is not charged for the missing name."""
        llm = LL.MockLLM([LL.MOCK_SEQUENCE[1]])  # the NameError candidate, on repeat
        loop = self.loop(llm=llm, max_iters=1)
        loop.run()
        node = [n for n in loop.tree.nodes if n.iteration == 1][0]
        self.assertTrue(node.failures)
        self.assertEqual(node.failures[0].cls.value, "smoke")
        self.assertFalse(node.failures[0].cls.costs_an_iteration)
        self.assertIsNotNone(node.valid, "the repaired script should have scored")

    def test_the_circuit_breaker_marks_failed_and_reverts_to_the_parent(self):
        """A repair loop that keeps regenerating the same fix is the most common way an
        agent harness turns its whole budget into nothing."""
        broken = LL._mock_reply(
            "A candidate that cannot be repaired, to exercise the circuit breaker.",
            "loss", "always_broken", "metric", 0.003, "broken",
            "import argparse\nraise RuntimeError('always')\n",
        )

        class AlwaysBroken(LL.MockLLM):
            def complete(self, prefix, messages, operator=None):
                response = super().complete(prefix, messages, operator=operator)
                return LL.LLMResponse(broken, response.usage)

        loop = self.loop(llm=AlwaysBroken([broken]), max_iters=1)
        trunk_before = None
        loop.run()
        node = [n for n in loop.tree.nodes if n.iteration == 1][0]
        self.assertIs(node.status, NodeStatus.FAILED)
        self.assertEqual(loop.tree.trunk().node_id, "n00", "the trunk must not move")
        self.assertLessEqual(len(node.failures), C.MAX_SELF_HEAL_ATTEMPTS + 1)
        self.assertGreaterEqual(len(node.failures), 2)

    def test_the_leakage_candidate_is_caught_and_logged_not_promoted(self):
        llm = LL.MockLLM([LL.MOCK_SEQUENCE[3]])  # reads play_time_ms on an eval split
        loop = self.loop(llm=llm, max_iters=1)

        # The repair is what would ordinarily rescue it; deny it so the firewall's own
        # verdict is what the test observes.
        loop._repair = lambda *a, **kw: None
        loop.run()
        node = [n for n in loop.tree.nodes if n.iteration == 1][0]
        self.assertIs(node.status, NodeStatus.FAILED)
        self.assertTrue(node.failures)
        self.assertIn("play_time_ms", node.failures[0].traceback_tail)
        self.assertEqual(loop.tree.trunk().node_id, "n00")
        entries = json.loads(Path(loop.cfg.logs_json).read_text())["iterations"]
        self.assertTrue(any(e["errors"] for e in entries))

    def test_an_interrupt_still_finalises_the_graded_deliverable(self):
        """The banner promises Ctrl-C is safe. `iteration_logs.json` is re-rendered
        after every entry, so it is already whole -- but the run *summary* is not, and
        it carries the two figures Feasibility is graded on by name."""
        loop = self.loop(max_iters=3)
        real_iteration = loop._iteration
        calls = {"n": 0}

        def interrupt_on_second(*a, **kw):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise KeyboardInterrupt("operator pressed ctrl-C")
            return real_iteration(*a, **kw)

        loop._iteration = interrupt_on_second
        summary = loop.run()

        self.assertGreaterEqual(summary.iterations, 1)
        self.assertGreater(summary.total_tokens.prompt_tokens, 0)
        self.assertGreaterEqual(summary.wall_clock_seconds, 0.0)
        rendered = json.loads(Path(loop.cfg.logs_json).read_text())
        self.assertIn("summary", rendered["run"])
        self.assertEqual(len(rendered["iterations"]), summary.iterations)

    def test_a_run_resumes_from_the_last_node(self):
        first = self.loop(max_iters=2)
        first.run()
        nodes_before = len(first.tree)

        resumed = Loop(
            LoopConfig(**{**first.cfg.__dict__, "resume": True, "max_iters": 3}),
            llm=LL.MockLLM(),
            console=Console(quiet=True),
            data=self.shared_data(),
        )
        self.assertEqual(len(resumed.tree), nodes_before)
        self.assertEqual(resumed.tree.trunk().node_id, first.tree.trunk().node_id)
        self.assertGreater(len(resumed.log.entries), 0)


if __name__ == "__main__":
    unittest.main()
