"""L6 -- the orchestrator.

    preflight -> profile -> seed the root with the FM baseline -> for each iteration:
        choose an axis; pick a parent; ask for a proposal
        reject-or-execute: syntax -> lint -> smoke -> seed 42 -> seeds 43,44
        gate -> promote? -> ledger -> log -> console
        if stalled: escalate to the critics
    until converged, out of iterations, or out of wall clock
    -> ensemble -> build the submission -> verify

Built last on purpose: a mistake here is cheap and visible, whereas a mistake in L1-L4
silently corrupts every number we would report.

    python -m harness.loop --mock --max-iters 5
    python -m harness.loop --max-iters 15
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C
from . import logger as L
from . import scoring
from .agent import Agent, ProposalRejected
from .console import Console
from .data_guard import DataAPI
from .evaluator import Evaluator
from .executor import Executor
from .llm import LLM, AnthropicLLM, MockLLM
from .memory import FeatureInsightsLedger, StateTree
from .types import (
    ConvergenceState,
    FailureClass,
    FailureRecord,
    GateDecision,
    LadderDecision,
    Metrics,
    Node,
    NodeStatus,
    PRIORITY_AXES,
    Proposal,
    TokenUsage,
)

SEED_SCRIPT = Path(__file__).resolve().parent / "seeds" / "baseline_fm.py"

#: How many times a rejected proposal is re-asked before the turn is abandoned. Two,
#: because a rejection is free in compute but not in tokens, and a model that has been
#: told the rule twice and still breaks it is not going to be told a third time.
MAX_REJECTION_RETRIES = 2

#: Consecutive turns that produce nothing runnable before the run gives up. Without a
#: cap this is an unbounded loop: a rejected turn changes no state, so the next turn
#: sees the same context and rejects again.
MAX_CONSECUTIVE_SKIPS = 3

#: Turns a run may take beyond `max_iters` on failures that do not spend an iteration.
#: Syntax, contract and smoke failures are repaired for free -- correctly, since they
#: are defects in the writing rather than in the idea -- but "free" means free of the
#: iteration budget, not free of tokens. Without a ceiling, a candidate that always
#: dies at smoke never charges the budget and the loop runs until it is killed.
MAX_FREE_TURNS = 5


@dataclass
class LoopConfig:
    mock: bool = False
    max_iters: int = C.MAX_ITERATIONS
    skip_preflight: bool = False
    confirm_seeds: tuple[int, ...] = C.CONFIRM_SEEDS
    root_seeds: tuple[int, ...] = C.CONFIRM_SEEDS
    wall_clock_ceiling_s: float = C.WALL_CLOCK_CEILING_S
    unbiased_check: bool = True
    critics: bool = True
    #: Build and verify the submission at the end of the run. A real run's whole point;
    #: ruinous inside a test, because it re-executes the trunk on the test split and
    #: then shells out to the organizers' `submit.py`, which re-parses the raw CSVs and
    #: scores in pure Python. Paying that in every orchestration test turned a
    #: three-minute suite into one that never finished.
    build_submission: bool = True
    #: Additionally score a valid-split submission through `submit.py` and compare it to
    #: the harness's own number. Worth a minute at the end of a real run: it is the only
    #: check that our scoring path and theirs agree on the same file.
    cross_check_valid: bool = True
    #: The research-ordering rule that refuses `architecture` until every priority axis
    #: has a scored attempt. Off in mock mode, where the point is to exercise the
    #: execution plumbing rather than the search order -- the rule has its own test.
    enforce_axis_lock: bool = True
    rng_seed: int = 0
    workspace: Path | None = None
    outputs_dir: Path | None = None
    state_path: Path | None = None
    logs_jsonl: Path | None = None
    logs_json: Path | None = None
    resume: bool = True
    run_timeout_s: float = C.RUN_TIMEOUT_S
    #: The root node's source. The FM baseline in every real run -- it is the incumbent
    #: we are scored against, so it has to be the thing candidates are measured
    #: against. Overridable so the acceptance tests can seed a two-second root and
    #: exercise the loop without paying 50 seconds of FM per test.
    seed_script: Path = SEED_SCRIPT
    submission_csv: Path | None = None
    best_model_py: Path | None = None
    #: Writing the seal is what unlocks the test labels for `score_final.py`, so it is
    #: a real run's last act -- and must never happen from a test or a rehearsal.
    seal: bool = True


@dataclass
class IterationOutcome:
    """What one turn produced. `costs_an_iteration` is what the budget actually spends:
    a syntax or smoke failure is repaired for free, so it must not consume a turn."""

    node: Node
    metrics: Metrics | None = None
    gate: GateDecision | None = None
    costs_an_iteration: bool = True
    stage_seconds: dict[str, float] = field(default_factory=dict)
    source: str = "agent"


class Loop:
    def __init__(
        self,
        cfg: LoopConfig | None = None,
        llm: LLM | None = None,
        console: Console | None = None,
        data: DataAPI | None = None,
    ):
        self.cfg = cfg or LoopConfig()
        self.console = console or Console()
        self.outputs = Path(self.cfg.outputs_dir or C.OUTPUTS_DIR)
        self.outputs.mkdir(parents=True, exist_ok=True)

        # Injectable, and shared with the executor and the evaluator below. A DataAPI
        # materialises the whole cache -- a few hundred megabytes -- so one per Loop is
        # right for a real run and ruinous for a test suite that builds a dozen of
        # them: unittest holds every test case alive, so the arrays accumulate until
        # the machine starts swapping and the suite appears to hang.
        self.data = data if data is not None else DataAPI()
        self.evaluator = Evaluator(self.data, scoring.evaluate_sha256())
        self.executor = Executor(
            workspace=self.cfg.workspace,
            data=self.data,
            run_timeout_s=self.cfg.run_timeout_s,
        )
        self.tree = StateTree(
            self.cfg.state_path, seed=self.cfg.rng_seed, resume=self.cfg.resume
        )
        self.ledger = FeatureInsightsLedger()
        self.log = L.IterationLogger(
            self.cfg.logs_jsonl, self.cfg.logs_json, resume=self.cfg.resume
        )
        self.llm = llm or (MockLLM() if self.cfg.mock else AnthropicLLM())
        self.agent = Agent(self.llm)
        self.prune_delta = C.PRUNE_AT_ONE_SEED_DELTA
        self.rng = random.Random(self.cfg.rng_seed)
        self.started = time.monotonic()
        self._critic_rounds = 0
        # Three critics produce three proposals but a turn consumes one. They queue and
        # are spent one per iteration, so each still passes through the ordinary gate.
        self._critique_queue: list = []
        self._skipped = 0
        self._unbiased_cache: dict[str, float] = {}

    # ------------------------------------------------------------ the run

    def run(self) -> L.RunSummary:
        """Drive the run. Interruptible at any point, without losing the deliverable.

        `iteration_logs.json` is re-rendered after every entry, so it is already
        complete when a Ctrl-C arrives -- but the run *summary* is not, and the summary
        carries the two figures the rubric asks for by name: total tokens and
        wall-clock. Finalising on the way out means an interrupted run still reports
        what it cost, and the state tree resumes from the last node.
        """
        try:
            return self._run()
        except KeyboardInterrupt:
            self.console.warn("interrupted — finalising the log; rerun to resume")
            convergence = self.evaluator.convergence(self.tree.history())
            iteration = max((n.iteration for n in self.tree.nodes), default=0)
            return self._finalize(convergence, iteration)

    def _run(self) -> L.RunSummary:
        self.console.banner(
            "mock" if self.cfg.mock else "live",
            self.cfg.max_iters,
            backend=self.llm.describe() if hasattr(self.llm, "describe") else "",
        )
        if not self.cfg.skip_preflight:
            self._preflight()

        root = self._seed_root()
        iteration = max((n.iteration for n in self.tree.nodes), default=0)
        spent = sum(1 for n in self.tree.nodes if n.iteration > 0)
        convergence = ConvergenceState(False, 0, root.valid.primary if root.valid else 0.0)

        turns = 0
        while spent < self.cfg.max_iters and turns < self.cfg.max_iters + MAX_FREE_TURNS:
            turns += 1
            if time.monotonic() - self.started > self.cfg.wall_clock_ceiling_s:
                self.console.warn("wall-clock ceiling reached; stopping")
                break

            iteration += 1
            source = "agent"
            if (
                self.cfg.critics
                and not self._critique_queue
                and convergence.stalled_iterations >= C.STALL_TRIGGER
                and self._critic_rounds < C.MAX_CRITIQUE_ROUNDS
            ):
                self._critique_queue = self._escalate(iteration)
            critique = self._critique_queue.pop(0) if self._critique_queue else None
            if critique is not None:
                source = f"critic:{critique[0]}"

            outcome = self._iteration(iteration, source=source, critique=critique)
            if outcome is None:
                # A turn whose proposals were all rejected still spends the budget. It
                # cost no compute, but it cost three generations, and not charging it
                # is an unbounded loop: nothing about the next turn's state differs, so
                # it would reject again, forever, until the token budget is gone.
                spent += 1
                self._skipped += 1
                if self._skipped >= MAX_CONSECUTIVE_SKIPS:
                    self.console.fail(
                        f"{self._skipped} turns in a row produced nothing runnable; "
                        "stopping rather than spending the budget on rejections"
                    )
                    break
                continue
            self._skipped = 0
            if outcome.costs_an_iteration:
                spent += 1

            convergence = self.evaluator.convergence(self.tree.history())
            if convergence.converged:
                self.console.ok(f"converged: {convergence.reason}")
                break
        else:
            if turns >= self.cfg.max_iters + MAX_FREE_TURNS:
                self.console.warn(
                    f"stopping after {turns} turns: too many failed without spending an "
                    f"iteration. Free repairs are free of the budget, not of tokens."
                )

        self._endgame(iteration)
        return self._finalize(convergence, iteration)

    def _finalize(self, convergence: ConvergenceState, iteration: int) -> L.RunSummary:
        summary = self.log.finalize(
            converged=convergence.converged,
            convergence_iteration=iteration if convergence.converged else None,
        )
        self.console.summary(summary)
        return summary

    # ------------------------------------------------------------ setup

    def _preflight(self) -> None:
        from . import preflight

        self.console.stage("preflight")
        report = preflight.run(skip_fm=self.cfg.mock)
        if not report.ok:
            raise RuntimeError(
                "preflight failed; refusing to start a run whose ground truth is "
                f"unverified:\n{report.summary() if hasattr(report, 'summary') else report}"
            )
        self.console.ok(f"preflight passed ({len(report.checks)} checks)")

    def _seed_root(self) -> Node:
        """The root is the FM baseline, executed through the same path as any candidate.

        Not a hard-coded number: the incumbent every candidate is measured against has
        to come out of the same executor, the same seeds and the same scorer, or the
        first promotion is a comparison between two different measurement procedures.
        """
        if self.tree.trunk_id is not None:
            root = self.tree.trunk()
            self.console.stage(f"resuming from node {root.node_id} "
                               f"(primary {root.valid.primary:.4f})")
            return root

        code_path = C.candidate_path(0, self.outputs)
        shutil.copyfile(self.cfg.seed_script, code_path)
        code = code_path.read_text()
        root = Node(
            node_id="n00",
            parent_id=None,
            iteration=0,
            proposal=Proposal(
                hypothesis="The organizers' FM baseline, ported to the DataAPI. The "
                           "incumbent every candidate is measured against.",
                axis="architecture",
                grounding="metric",
                predicted_delta=0.0,
                code=code,
                technique="fm_baseline",
            ),
            status=NodeStatus.PENDING,
            code_path=str(code_path),
            code_sha256=_sha256(code),
            change_summary="the FM baseline",
            created_at=time.time(),
        )
        self.tree.add(root)

        self.console.stage(f"scoring the baseline on seeds {list(self.cfg.root_seeds)}")
        per_seed, resources = self._run_seeds(root, code, self.cfg.root_seeds)
        if not per_seed:
            raise RuntimeError("the FM baseline itself failed to run; fix the harness")
        aggregate = self.evaluator.aggregate(per_seed)
        self.tree.update(root.node_id, resources=resources)
        root = self.tree.record_result(root.node_id, aggregate, per_seed)
        self.tree.promote(root.node_id)
        self.console.ok(
            f"baseline primary {aggregate.primary:.4f} "
            f"(GAUC {aggregate.gauc:.4f}, nDCG@5 {aggregate.ndcg5:.4f})"
        )
        self.log.log(
            L.make_entry(self.tree.get(root.node_id), None, source="baseline")
        )
        return self.tree.get(root.node_id)

    # ------------------------------------------------------------ one iteration

    def _iteration(
        self, iteration: int, source: str = "agent", critique=None
    ) -> IterationOutcome | None:
        axis = critique[1] if critique else self._choose_axis()
        parent = self.tree.select_parent(axis)
        best = self.tree.trunk().valid
        self.console.iteration(iteration, axis, parent.node_id, best.primary if best else 0.0)

        generation = self._generate(iteration, parent, axis, critique)
        if generation is None:
            return None
        usage = generation.usage
        self.console.hypothesis(
            generation.proposal.hypothesis,
            generation.proposal.predicted_delta,
            generation.proposal.grounding,
            generation.grounding_verified,
        )
        self.console.tokens(usage)

        node_id = f"n{iteration:02d}"
        code_path = C.candidate_path(iteration, self.outputs)
        code_path.write_text(generation.proposal.code)
        diff, is_rewrite = L.compute_diff(
            parent.proposal.code or Path(parent.code_path).read_text(),
            generation.proposal.code,
            parent_name=parent.node_id,
        )
        node = Node(
            node_id=node_id,
            parent_id=parent.node_id,
            iteration=iteration,
            proposal=generation.proposal,
            status=NodeStatus.PENDING,
            code_path=str(code_path),
            code_sha256=_sha256(generation.proposal.code),
            grounding_verified=generation.grounding_verified,
            change_summary=generation.change_summary,
            is_rewrite=is_rewrite,
            tokens=usage,
            created_at=time.time(),
        )
        self.tree.add(node)

        outcome = self._execute(node, generation.proposal.code, iteration, parent, usage)
        outcome.source = source
        self._record(outcome, parent, diff)
        return outcome

    def _generate(self, iteration: int, parent: Node, axis: str, critique):
        """Ask, validate, and re-ask on a rejection. Rejections cost no compute."""
        parent_code = parent.proposal.code or Path(parent.code_path).read_text()
        scored_axes = {n.proposal.axis for n in self.tree.nodes if n.valid is not None}

        if critique is not None:
            # A critic's proposal goes through the same pre-execution gates as the
            # agent's. It has already been through the same parser; exempting it from
            # the dead-end and ledger checks would make "rescue the run" a way to buy a
            # measured dead end back, and there is no re-ask to fall back on.
            generation = critique[2]
            try:
                self.agent.check_dead_end(parent_code, generation.proposal.code)
                self.agent.check_ledger(generation.proposal, self.ledger.get)
            except ProposalRejected as rejection:
                self.console.warn(
                    f"critic {critique[0]}'s proposal rejected [{rejection.rule}]: "
                    f"{rejection.message.splitlines()[0]}"
                )
                return None
            return generation

        messages = self.agent.build_messages(
            iteration=iteration,
            parent=parent,
            parent_code=parent_code,
            ledger_block=self.ledger.render(),
            best=self.tree.trunk().valid,
            assigned_axis=axis,
            axis_reason=self._axis_reason(axis),
            recent_changes=self._recent_changes(),
        )
        operator = None
        for attempt in range(MAX_REJECTION_RETRIES + 1):
            try:
                generation = self.agent.propose(messages, operator=operator)
                if self.cfg.enforce_axis_lock:
                    self.agent.check_axis_lock(generation.proposal.axis, scored_axes)
                self.agent.check_dead_end(parent_code, generation.proposal.code)
                self.agent.check_ledger(generation.proposal, self.ledger.get)
                return generation
            except ProposalRejected as rejection:
                self.console.warn(f"rejected pre-execution [{rejection.rule}]: "
                                  f"{rejection.message.splitlines()[0]}")
                operator = (
                    f"## Your last proposal was rejected before it ran\n\n"
                    f"[{rejection.rule}] {rejection.message}\n\n"
                    f"This cost no compute but it did cost a turn. Reply again in the "
                    f"required format with a proposal that does not hit this rule."
                )
        self.console.fail("three rejected proposals; skipping this turn")
        return None

    # ------------------------------------------------------------ execution

    def _execute(
        self,
        node: Node,
        code: str,
        iteration: int,
        parent: Node,
        usage: TokenUsage,
    ) -> IterationOutcome:
        """Stages 1-5 with the self-heal loop wrapped around all of them.

        The repair loop covers the full run, not only the cheap stages. A candidate
        that passes smoke on 1% of users and then dies at full scale -- an OOM, a
        timeout, a constant-score collapse -- is exactly the case where the traceback
        is most informative and the idea is most likely still sound. Repairing only the
        stages before it would throw those away.

        What the stages differ on is *cost*: a syntax, contract or smoke failure is a
        defect in the writing rather than in the idea, so an unrecoverable one spends no
        iteration. Charging the idea for a missing colon is how a budget disappears.
        """
        failures: list[FailureRecord] = []
        repairs: list[str] = []
        stage_seconds: dict[str, float] = {}
        current = code
        first: list[Metrics] = []
        resources = None
        incumbent = parent.valid or self.tree.trunk().valid

        for attempt in range(C.MAX_SELF_HEAL_ATTEMPTS + 1):
            failure = self.executor.check_syntax(current) or self.executor.lint_contract(current)

            if failure is None:
                t0 = time.monotonic()
                smoke = self.executor.smoke(current, node.node_id)
                stage_seconds["smoke_seconds"] = round(time.monotonic() - t0, 1)
                failure = smoke.failure
                if failure is None:
                    self.console.ok(f"smoke passed in {stage_seconds['smoke_seconds']:.0f}s")

            if failure is None:
                t0 = time.monotonic()
                first, resources, failure = self._run_seed(
                    node, current, self.cfg.confirm_seeds[0]
                )
                stage_seconds["seed_seconds"] = round(time.monotonic() - t0, 1)

            if failure is None:
                break

            failure = FailureRecord(
                cls=failure.cls,
                signature=failure.signature,
                traceback_tail=failure.traceback_tail,
                frame_context=failure.frame_context,
                repair_attempt=attempt,
            )
            failures.append(failure)
            self.console.fail(
                f"{failure.cls.value}: {failure.traceback_tail.splitlines()[0][:140]}"
            )
            if attempt >= C.MAX_SELF_HEAL_ATTEMPTS:
                break
            if _identical_failures(failures) >= C.MAX_IDENTICAL_FAILURES:
                self.console.warn(
                    "the same failure twice — the diagnosis is wrong, not the patch; "
                    "stopping repairs"
                )
                break
            repaired = self._repair(node, failure, repairs, attempt + 1, current, parent)
            if repaired is None:
                break
            current, repair_usage = repaired
            usage = usage + repair_usage
            repairs.append(
                f"attempt {attempt + 1}: {failure.traceback_tail.splitlines()[0][:100]}"
            )
            Path(node.code_path).write_text(current)

        self.tree.update(node.node_id, failures=failures, tokens=usage, resources=resources)
        if not first:
            return self._fail(node, failures, stage_seconds)
        if failures:
            self.console.ok(f"repaired after {len(failures)} failed attempt(s)")

        # -- tier 2 of the ladder: the remaining seeds
        if self.evaluator.gate_first_seed(first[0], incumbent) is LadderDecision.PRUNE:
            node = self.tree.record_result(node.node_id, first[0], first)
            base = self.evaluator.gate(first[0], incumbent)
            # The ladder's own reason, not the promotion gate's. `gate()` would say
            # "only 1 seed", which is true and useless: nothing was pruned for lack of
            # seeds, it was pruned for being far below the incumbent at the first one.
            gate = GateDecision(
                promote=False,
                reason=(
                    f"pruned at seed {self.cfg.confirm_seeds[0]}: Δprimary "
                    f"{base.delta_primary:+.4f} is below the {self.prune_delta:+.3f} "
                    f"bar, ~6σ under a promotable candidate — the remaining seeds "
                    f"cannot rescue it and would cost real wall clock"
                ),
                quarantined=base.quarantined,
                delta_primary=base.delta_primary,
                delta_gauc=base.delta_gauc,
                delta_ndcg5=base.delta_ndcg5,
                seeds_run=1,
            )
            self.console.gate(gate)
            self.tree.prune_subtree(node.node_id, "clear regression at one seed")
            return IterationOutcome(self.tree.get(node.node_id), first[0], gate,
                                    stage_seconds=stage_seconds)

        rest, resources = self._run_seeds(node, current, self.cfg.confirm_seeds[1:])
        per_seed = first + rest
        aggregate = self.evaluator.aggregate(per_seed)
        self.tree.update(node.node_id, resources=resources)
        node = self.tree.record_result(node.node_id, aggregate, per_seed)

        gate = self.evaluator.gate(aggregate, incumbent)
        gate = self._maybe_unbiased(node, current, gate)
        self.console.gate(gate)
        if gate.quarantined:
            self.tree.update(node.node_id, status=NodeStatus.QUARANTINED)
        elif gate.promote and aggregate.primary > self.tree.trunk().valid.primary:
            self.tree.promote(node.node_id)
        elif gate.promote:
            self.console.warn("beat its parent but not the trunk; the trunk stands")
        return IterationOutcome(self.tree.get(node.node_id), aggregate, gate,
                                stage_seconds=stage_seconds)

    def _fail(self, node: Node, failures, stage_seconds) -> IterationOutcome:
        self.tree.update(node.node_id, status=NodeStatus.FAILED, failures=failures)
        costs = bool(failures) and failures[-1].cls.costs_an_iteration
        self.console.fail(
            f"node {node.node_id} FAILED after {len(failures)} attempt(s); "
            f"reverting to the parent"
            + ("" if costs else " (no iteration spent -- a writing defect, not an idea)")
        )
        return IterationOutcome(self.tree.get(node.node_id), None, None,
                                costs_an_iteration=costs, stage_seconds=stage_seconds)

    def _repair(self, node, failure, repairs, attempt, code, parent):
        from . import prompts

        instruction = prompts.repair_instruction(
            failure.traceback_tail, failure.frame_context, repairs, attempt
        )
        messages = self.agent.build_messages(
            iteration=node.iteration,
            parent=parent,
            parent_code=code,
            ledger_block=self.ledger.render(),
            best=self.tree.trunk().valid,
            assigned_axis=node.proposal.axis,
            failure_context=instruction,
        )
        try:
            generation = self.agent.propose(messages)
        except ProposalRejected as exc:
            self.console.warn(f"repair rejected: {exc.message.splitlines()[0]}")
            return None
        self.console.stage(f"repair {attempt}: {generation.change_summary or 'regenerated'}")
        return generation.proposal.code, generation.usage

    def _run_seed(self, node: Node, code: str, seed: int):
        """One seed. Returns `([metrics] or [], resources, failure or None)`."""
        t0 = time.monotonic()
        result = self.executor.run(code, node.node_id, "valid", seed)
        if not result.ok:
            return [], result.resources, result.failure
        metrics = self.evaluator.score(np.load(result.scores_path), "valid", seed)
        self.console.seed_result(seed, metrics, time.monotonic() - t0)
        return [metrics], result.resources, None

    def _run_seeds(self, node: Node, code: str, seeds) -> tuple[list[Metrics], object]:
        out: list[Metrics] = []
        resources = None
        for seed in seeds:
            metrics, resources, failure = self._run_seed(node, code, seed)
            if failure is not None:
                self.console.fail(
                    f"seed {seed}: {failure.cls.value} — "
                    f"{failure.traceback_tail.splitlines()[0][:140]}"
                )
                return [], resources
            out += metrics
        return out, resources

    def _unbiased_score(self, node_id: str, code: str) -> float | None:
        """Primary on the randomly-exposed impressions, memoised per node.

        `None` when the script will not run there -- advisory checks never fail a run.
        """
        if node_id in self._unbiased_cache:
            return self._unbiased_cache[node_id]
        result = self.executor.run(code, f"{node_id}_rand", "rand", self.cfg.confirm_seeds[0])
        if not result.ok:
            return None
        feats, labels = self.data.random_exposure()
        primary = scoring.score(
            feats["user_id"], labels, np.load(result.scores_path)
        ).primary
        self._unbiased_cache[node_id] = primary
        return primary

    def _maybe_unbiased(self, node: Node, code: str, gate: GateDecision) -> GateDecision:
        """Re-score a would-be promotion on randomly-exposed impressions.

        Advisory, not blocking. It carries a different bias, not less noise: it is a
        quarter the size of valid, so treating a null result here as disqualifying
        would throw away real gains for a reason we cannot measure. It is recorded and
        reported instead.
        """
        if not (self.cfg.unbiased_check and gate.promote):
            return gate

        candidate = self._unbiased_score(node.node_id, code)
        if candidate is None:
            self.console.warn("unbiased check could not run; promotion stands")
            return gate

        note = f"; random-exposure primary {candidate:.4f}"
        parent = self.tree.get(node.parent_id) if node.parent_id else None
        if parent is not None:
            # Measure the parent on demand when it is not already cached. The earlier
            # version only ever cached nodes that had themselves been promoted, so the
            # parent was almost never present -- the root never is -- and the delta,
            # which is the entire point of this check, silently never computed. It
            # printed one absolute number and compared nothing, at the price of a full
            # extra execution per promotion.
            base = self._unbiased_score(
                parent.node_id, parent.proposal.code or Path(parent.code_path).read_text()
            )
            if base is not None:
                note += f" (Δ{candidate - base:+.4f} vs parent on unbiased traffic)"
                if candidate < base:
                    self.console.warn(
                        "gained on logged traffic but not on random exposure — this may "
                        "be the logging policy rather than the ranking"
                    )
        return GateDecision(
            promote=gate.promote,
            reason=gate.reason + note,
            quarantined=gate.quarantined,
            delta_primary=gate.delta_primary,
            delta_gauc=gate.delta_gauc,
            delta_ndcg5=gate.delta_ndcg5,
            seeds_run=gate.seeds_run,
        )

    # ------------------------------------------------------------ bookkeeping

    def _record(self, outcome: IterationOutcome, parent: Node, diff: str) -> None:
        node = outcome.node
        if outcome.gate is not None:
            self.ledger.record(node, outcome.gate)
        self.log.log(
            L.make_entry(
                node,
                parent,
                gate=outcome.gate,
                diff=diff,
                stage_seconds=outcome.stage_seconds,
                source=outcome.source,
            )
        )

    #: Axes the loop will choose between. `architecture` is reachable only once the
    #: priority axes have results (the agent may still propose it, subject to the axis
    #: lock), and `ensemble` belongs to the endgame rather than to the search.
    SEARCH_AXES: tuple[str, ...] = PRIORITY_AXES + ("temporal", "debias")

    #: UCB's exploration weight, in units of primary. The promotion bar is the natural
    #: scale: an axis is worth another look when its uncertainty is worth about one
    #: promotion. A constant tuned in the abstract would be a number with no meaning on
    #: this metric.
    UCB_C = C.PROMOTE_DELTA

    def _choose_axis(self) -> str:
        """Forced seeding, then UCB over axes weighted by realised Δ.

        Iterations 1-4 are one forced probe per priority axis so the ledger has a real
        observation on each before exploitation starts. Without that the run commits to
        whichever axis happened to go first.

        After that, each axis scores `best realised Δ + c·sqrt(2·ln N / n)`. Plain
        argmax over realised Δ is the wrong rule for a 15-iteration budget: it would
        ride whichever axis got lucky first and never revisit an axis tried once for a
        mediocre result, even though one observation says almost nothing at σ≈0.001.
        The bonus is what makes "tried once" and "tried three times" different states,
        and it decays on its own as evidence accumulates -- so no axis starves and none
        is explored past the point of being informative.
        """
        attempts = self.tree.axis_attempts()
        for axis in PRIORITY_AXES:
            if attempts.get(axis, 0) == 0:
                return axis

        realised = self.ledger.realised_by_axis()
        total = sum(max(attempts.get(a, 0), 0) for a in self.SEARCH_AXES) or 1

        def ucb(axis: str) -> float:
            n = attempts.get(axis, 0)
            if n == 0:
                return float("inf")  # never chosen: no estimate exists to exploit
            exploit = realised.get(axis, 0.0)
            return exploit + self.UCB_C * math.sqrt(2.0 * math.log(total) / n)

        best = max(ucb(a) for a in self.SEARCH_AXES)
        # Ties are common early, when every axis has one attempt and the same bonus.
        # Breaking them at random rather than by list order keeps the run from always
        # exploring the axes in the order they happen to be declared in.
        return self.rng.choice([a for a in self.SEARCH_AXES if ucb(a) == best])

    def _axis_reason(self, axis: str) -> str:
        attempts = self.tree.axis_attempts().get(axis, 0)
        if attempts == 0:
            return (
                "This axis has no observation yet. The first four iterations probe one "
                "priority axis each so the ledger has a real result on every one before "
                "exploitation begins."
            )
        realised = self.ledger.realised_by_axis().get(axis)
        return (
            f"{attempts} attempt(s) so far on this axis"
            + (f"; best realised Δprimary {realised:+.4f}." if realised is not None else ".")
        )

    def _recent_changes(self, k: int = 3) -> str:
        """The last few one-line summaries. Summaries, not diffs: a 300-line diff of
        code the agent wrote two turns ago adds nothing it does not already know, and
        context is a budget."""
        recent = [n for n in self.tree.nodes if n.iteration > 0][-k:]
        if not recent:
            return ""
        return "\n".join(
            f"- iter {n.iteration:02d} [{n.status.value}] {n.proposal.axis}/"
            f"{n.proposal.technique}: {n.change_summary or n.proposal.hypothesis[:80]}"
            + (f" → primary {n.valid.primary:.4f}" if n.valid else "")
            for n in recent
        )

    # ------------------------------------------------------------ escalation

    def _escalate(self, iteration: int):
        from . import critics

        self._critic_rounds += 1
        self.console.warn(
            f"stalled — escalating to {C.N_CRITICS} isolated critics "
            f"(round {self._critic_rounds} of {C.MAX_CRITIQUE_ROUNDS})"
        )
        return critics.escalate(
            llm=self.llm,
            agent=self.agent,
            trunk=self.tree.trunk(),
            ledger=self.ledger,
            console=self.console,
        )

    def _cross_check_valid(self, trunk: Node) -> None:
        """Score a valid-split submission through the organizers' own `submit.py` and
        compare it to the number we recorded.

        The test submission cannot be scored locally, so this is the only way to check
        that our scoring path and theirs agree on the same file. A disagreement means
        something is misaligned, and the time to find that out is before submitting,
        not after.
        """
        result = self.executor.run(
            trunk.code_path, f"{trunk.node_id}_xcheck", "valid", self.cfg.confirm_seeds[0]
        )
        if not result.ok:
            self.console.warn("valid cross-check could not run")
            return
        scores = np.load(result.scores_path)
        valid_csv = Path(self.cfg.submission_csv or C.SUBMISSION_CSV).with_name(
            "submission_valid.csv"
        )
        self.evaluator.write_submission_for(scores, valid_csv, "valid")
        ours = self.evaluator.score(scores, "valid", self.cfg.confirm_seeds[0])
        try:
            self.evaluator.verify_and_score_valid(valid_csv)
            self.console.ok(
                f"submit.py agrees on the valid file (our primary {ours.primary:.4f})"
            )
        except RuntimeError as exc:
            self.console.fail(f"submit.py disagrees with the harness on valid: {exc}")

    def _endgame(self, iteration: int) -> None:
        from . import critics

        trunk = self.tree.trunk()
        if self.cfg.critics:
            confirmed = [
                n for n in self.tree.nodes
                if n.status is NodeStatus.PROMOTED and n.valid is not None
            ]
            if len(confirmed) >= 2:
                self.console.stage(
                    f"ensembling the top {min(C.ENSEMBLE_TOP_K, len(confirmed))} confirmed nodes"
                )
                try:
                    node = critics.build_ensemble(self, confirmed, iteration + 1)
                    if node is not None:
                        trunk = self.tree.trunk()
                except Exception as exc:  # pragma: no cover - never lose a good run to this
                    self.console.warn(f"ensembling failed, keeping the trunk: {exc}")

        if not self.cfg.build_submission:
            return
        self.console.stage(f"building the submission from node {trunk.node_id}")
        try:
            csv_path = self.evaluator.build_submission(
                trunk, Path(self.cfg.submission_csv or C.SUBMISSION_CSV), self.executor.run
            )
            shutil.copyfile(trunk.code_path, Path(self.cfg.best_model_py or C.BEST_MODEL_PY))
            self.evaluator.verify_alignment(csv_path)
            self.console.ok(f"submission verified: {csv_path}")
            if self.cfg.cross_check_valid:
                self._cross_check_valid(trunk)
            from . import holdout

            if not self.cfg.seal:
                self.console.warn("not sealing: this run was configured not to")
                return
            convergence = self.evaluator.convergence(self.tree.history())
            holdout.seal_run(
                node_id=trunk.node_id,
                submission_sha256=_sha256(Path(csv_path).read_text()),
                valid_primary=trunk.valid.primary if trunk.valid else 0.0,
                iterations=iteration,
                converged_reason=convergence.reason,
            )
            self.console.ok("run sealed; score_final.py may now read the test labels")
        except Exception as exc:
            # Caught, because a run that produced a good model must not be lost to a
            # packaging failure -- the tree and the log are already on disk. But it is
            # recorded, not just printed: the submission is the deliverable, and a run
            # that ends without one has to say so in the artifact a judge reads.
            self.console.fail(f"submission build failed: {type(exc).__name__}: {exc}")
            self.log.record_intervention(
                f"submission build failed and needs a human: {type(exc).__name__}: {exc}"
            )


# ---------------------------------------------------------------- helpers


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _identical_failures(failures: list[FailureRecord]) -> int:
    if not failures:
        return 0
    last = failures[-1].signature
    return sum(1 for f in failures if f.signature == last)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock", action="store_true", help="pre-written candidates, no API calls")
    ap.add_argument("--max-iters", type=int, default=C.MAX_ITERATIONS)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--no-critics", action="store_true")
    ap.add_argument("--no-unbiased", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore any resumable state")
    ap.add_argument("--root-seeds", type=int, default=len(C.CONFIRM_SEEDS))
    args = ap.parse_args(argv)

    # Mock mode never seals and never writes into the real deliverable directory. A
    # mock run's submission is the FM baseline with a rehearsal attached; sealing on it
    # would unlock the test labels for a run that proved nothing, and its candidates
    # would sit in `outputs/` looking like the agent's work.
    cfg = LoopConfig(
        mock=args.mock,
        enforce_axis_lock=not args.mock,
        seal=not args.mock,
        cross_check_valid=not args.mock,
        build_submission=True,
        outputs_dir=(C.OUTPUTS_DIR / "mock") if args.mock else None,
        state_path=(C.LOGS_DIR / "mock_state.jsonl") if args.mock else None,
        logs_jsonl=(C.LOGS_DIR / "mock_iteration_logs.jsonl") if args.mock else None,
        logs_json=(C.LOGS_DIR / "mock_iteration_logs.json") if args.mock else None,
        submission_csv=(C.OUTPUTS_DIR / "mock" / "submission.csv") if args.mock else None,
        best_model_py=(C.OUTPUTS_DIR / "mock" / "best_model.py") if args.mock else None,
        max_iters=args.max_iters,
        skip_preflight=args.skip_preflight,
        critics=not args.no_critics,
        unbiased_check=not args.no_unbiased,
        resume=not args.fresh,
        root_seeds=C.CONFIRM_SEEDS[: max(1, args.root_seeds)],
    )
    Loop(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
