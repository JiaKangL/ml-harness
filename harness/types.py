"""Shared contracts. Every layer codes against these; nobody redefines them locally.

This module exists so the phases can be implemented independently without drifting.
It deliberately imports nothing from the harness except `config`, so it sits at the
bottom of the dependency graph and every layer may import it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

Split = Literal["train", "valid", "test"]

# The closed set of research directions. Doubles as the insight ledger's vocabulary:
# without it, iteration 9 has no way to know iteration 2 already settled the loss
# question. The first four are the organizers' priority directions.
Axis = Literal[
    "loss",
    "sequence",
    "multitask",
    "watchtime",
    "architecture",
    "temporal",
    "debias",
    "ensemble",
]

PRIORITY_AXES: tuple[Axis, ...] = ("loss", "sequence", "multitask", "watchtime")


class Verdict(Enum):
    KEEP = "KEEP"
    DISCARD = "DISCARD"
    # First-class, not a fallback. At sigma ~= 0.001 most single-run deltas genuinely
    # are inconclusive; forcing a binary verdict manufactures false knowledge that
    # then propagates into every later prompt.
    INCONCLUSIVE = "INCONCLUSIVE"


class NodeStatus(Enum):
    PENDING = "pending"
    SCORED = "scored"
    PROMOTED = "promoted"
    FAILED = "failed"
    PRUNED = "pruned"
    QUARANTINED = "quarantined"  # presumed leakage; frozen, never promoted


class FailureClass(Enum):
    SYNTAX = "syntax"
    CONTRACT = "contract"
    SMOKE = "smoke"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    OOM = "oom"
    INVALID_OUTPUT = "invalid_output"
    NONDETERMINISTIC = "nondeterministic"

    @property
    def costs_an_iteration(self) -> bool:
        """Syntax and smoke failures are repaired for free; the rest spend a turn."""
        return self not in (FailureClass.SYNTAX, FailureClass.CONTRACT, FailureClass.SMOKE)


class LadderDecision(Enum):
    """Tier-1 outcome of the seed ladder. Note there is no PROMOTE member: we prune
    early on one seed but never promote on one."""

    PRUNE = "prune"
    CONTINUE = "continue"


@dataclass(frozen=True)
class Metrics:
    """One scored evaluation, possibly averaged across seeds.

    GAUC and nDCG@5 are kept separate deliberately: they weight users differently
    (GAUC by positive count over the discriminative users only; nDCG equally, with
    ~36% of users permanently fixed), so a change that moves exactly one of them has
    a mechanism, while one that nudges both slightly is likelier noise.
    """

    gauc: float
    ndcg5: float
    primary: float
    users: int
    rows: int
    seeds: tuple[int, ...] = (42,)
    primary_std: float = 0.0

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)


@dataclass(frozen=True)
class ResourceFacts:
    wall_seconds: float
    peak_rss_bytes: int
    exit_code: int | None = None
    killed_by: str | None = None  # "timeout" | "rss" | None


@dataclass(frozen=True)
class FailureRecord:
    cls: FailureClass
    signature: str  # (exc type, normalised message, frame) -- the dedupe key
    traceback_tail: str
    frame_context: str | None = None
    repair_attempt: int = 0


@dataclass(frozen=True)
class TokenUsage:
    """Feasibility & Practicality is 15% of the rubric and is graded on total tokens
    and wall-clock, so this is a deliverable rather than telemetry. Captured on every
    call -- including repair and critic calls, which are the easiest to forget and
    exactly the ones that inflate the total."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.latency_seconds + other.latency_seconds,
            self.cost_usd + other.cost_usd,
        )

    @property
    def cache_hit_rate(self) -> float:
        """Share of prompt tokens served from cache. If this is 0 after iteration 2,
        a silent invalidator is at work -- the failure shows up only on the bill."""
        total = self.prompt_tokens + self.cache_read_tokens
        return self.cache_read_tokens / total if total else 0.0


@dataclass(frozen=True)
class Proposal:
    """The agent's structured output contract."""

    hypothesis: str  # what and WHY -- scored directly under Innovation
    axis: Axis
    grounding: str  # a named field from data_profile.json; resolved fuzzily
    predicted_delta: float  # required: absent means there is no hypothesis to test
    code: str
    technique: str = ""  # short label; ledger key is (axis, technique)


@dataclass
class Node:
    """One attempt. Immutable once scored -- promotion moves a pointer, never mutates,
    so rollback is free and a degraded node can never become a parent."""

    node_id: str
    parent_id: str | None
    iteration: int
    proposal: Proposal
    status: NodeStatus
    code_path: str
    code_sha256: str
    valid: Metrics | None = None
    per_seed: list[Metrics] = field(default_factory=list)
    grounding_verified: bool = True
    change_summary: str = ""  # one line; the raw diff lives beside it in the log
    is_rewrite: bool = False  # >60% of lines changed vs parent
    failures: list[FailureRecord] = field(default_factory=list)
    resources: ResourceFacts | None = None
    tokens: TokenUsage | None = None
    created_at: float = 0.0


@dataclass(frozen=True)
class Insight:
    """Ledger entry, keyed by (axis, technique). `mechanism` carries the WHY, because
    mechanisms generalise across contexts while bare prohibitions decay over turns."""

    axis: Axis
    technique: str
    verdict: Verdict
    delta_primary: float
    delta_gauc: float
    delta_ndcg5: float
    n_seeds: int
    mechanism: str
    node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str  # human-readable; goes straight into the iteration log
    quarantined: bool
    delta_primary: float
    delta_gauc: float
    delta_ndcg5: float
    seeds_run: int


@dataclass(frozen=True)
class ConvergenceState:
    converged: bool
    stalled_iterations: int  # >= STALL_TRIGGER fires the critics
    best_primary: float
    reason: str = ""


@dataclass(frozen=True)
class RunResult:
    """One supervised execution of agent-written code.

    Lives here rather than in `executor.py` because L2 already codes against it:
    `Evaluator.build_submission` takes an injected runner and reads `.ok` and
    `.scores_path` off whatever it returns. A dataclass defined inside L3 would force
    L2 to duck-type its way around a layer it may not import.

    `failure` is populated for every non-ok result; `resources` is populated always,
    including on failure -- the peak RSS and wall time of a run that died are exactly
    the facts the loop feeds back to the LLM.
    """

    ok: bool
    scores_path: "Path | None"
    resources: ResourceFacts
    stdout_path: "Path | None" = None
    stderr_path: "Path | None" = None
    failure: FailureRecord | None = None
