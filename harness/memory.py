"""L4 -- what has been tried, what is currently best, and what has been ruled out.

Two structures, one file, because they answer the same question from opposite ends:
`StateTree` remembers the attempts, `FeatureInsightsLedger` remembers the conclusions.

The tree is a tree with a greedy trunk rather than a linear lineage. That is the whole
design: with a linear lineage one subtly-wrong non-crashing edit becomes the substrate
for everything after it, and there is no way back. Here the trunk is a *pointer* to the
best confirmed node; promotion moves the pointer and never mutates a node, so rollback
is free and a degraded node can never become a parent.

Depends on L1-L2. Must not import agent, critics or loop.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

from . import config as C
from .types import (
    Axis,
    FailureClass,
    FailureRecord,
    GateDecision,
    Insight,
    Metrics,
    Node,
    NodeStatus,
    Proposal,
    ResourceFacts,
    TokenUsage,
    Verdict,
)


class ImmutableNodeError(RuntimeError):
    """An attempt to rewrite a scored node's result. The tree is append-only."""


class UnknownNodeError(KeyError):
    pass


# ---------------------------------------------------------------- serialisation
#
# Written by hand rather than derived from the dataclass fields. The tree is the
# resume path for a multi-hour run, and a reflection-driven codec fails silently on a
# type it does not recognise -- it round-trips a tuple as a list, an Enum as its repr,
# and the run comes back subtly different from the one that was interrupted.

_STATUS_ELIGIBLE = (NodeStatus.SCORED, NodeStatus.PROMOTED)


def encode_node(node: Node) -> dict:
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "iteration": node.iteration,
        "status": node.status.value,
        "code_path": node.code_path,
        "code_sha256": node.code_sha256,
        "grounding_verified": node.grounding_verified,
        "change_summary": node.change_summary,
        "is_rewrite": node.is_rewrite,
        "created_at": node.created_at,
        "proposal": {
            "hypothesis": node.proposal.hypothesis,
            "axis": node.proposal.axis,
            "grounding": node.proposal.grounding,
            "predicted_delta": node.proposal.predicted_delta,
            "technique": node.proposal.technique,
            # The source lives in `code_path` on disk and in the iteration log's diff.
            # Duplicating it into every state line would make the resume file tens of
            # megabytes of text we already have two copies of.
            "code": "",
        },
        "valid": encode_metrics(node.valid),
        "per_seed": [encode_metrics(m) for m in node.per_seed],
        "failures": [
            {
                "cls": f.cls.value,
                "signature": f.signature,
                "traceback_tail": f.traceback_tail,
                "frame_context": f.frame_context,
                "repair_attempt": f.repair_attempt,
            }
            for f in node.failures
        ],
        "resources": asdict(node.resources) if node.resources else None,
        "tokens": asdict(node.tokens) if node.tokens else None,
    }


def encode_metrics(m: Metrics | None) -> dict | None:
    if m is None:
        return None
    d = asdict(m)
    d["seeds"] = list(m.seeds)
    return d


def decode_metrics(d: dict | None) -> Metrics | None:
    if d is None:
        return None
    d = dict(d)
    d["seeds"] = tuple(d["seeds"])
    return Metrics(**d)


def decode_node(d: dict, code: str = "") -> Node:
    p = d["proposal"]
    return Node(
        node_id=d["node_id"],
        parent_id=d["parent_id"],
        iteration=d["iteration"],
        proposal=Proposal(
            hypothesis=p["hypothesis"],
            axis=p["axis"],
            grounding=p["grounding"],
            predicted_delta=p["predicted_delta"],
            code=code or p.get("code", ""),
            technique=p.get("technique", ""),
        ),
        status=NodeStatus(d["status"]),
        code_path=d["code_path"],
        code_sha256=d["code_sha256"],
        valid=decode_metrics(d["valid"]),
        per_seed=[decode_metrics(m) for m in d["per_seed"]],
        grounding_verified=d["grounding_verified"],
        change_summary=d["change_summary"],
        is_rewrite=d["is_rewrite"],
        failures=[
            FailureRecord(
                cls=FailureClass(f["cls"]),
                signature=f["signature"],
                traceback_tail=f["traceback_tail"],
                frame_context=f["frame_context"],
                repair_attempt=f["repair_attempt"],
            )
            for f in d["failures"]
        ],
        resources=ResourceFacts(**d["resources"]) if d["resources"] else None,
        tokens=TokenUsage(**d["tokens"]) if d["tokens"] else None,
        created_at=d["created_at"],
    )


# ---------------------------------------------------------------- the tree


class StateTree:
    """Append-only JSONL of events, replayed on resume.

    Events rather than snapshots: a snapshot file rewritten every iteration has a
    window in which it is neither the old state nor the new one, and a multi-hour run
    *will* be interrupted. An append plus fsync has no such window -- the worst case
    is a torn final line, which the replay drops.
    """

    def __init__(
        self,
        jsonl: Path | None = None,
        seed: int = 0,
        resume: bool = True,
        exploit_p: float = C.EXPLOIT_TRUNK_PROBABILITY,
    ):
        self.path = Path(jsonl) if jsonl is not None else C.STATE_JSONL
        self.exploit_p = exploit_p
        self._nodes: dict[str, Node] = {}
        self._order: list[str] = []
        self._children: dict[str, list[str]] = {}
        self._trunk_id: str | None = None
        self._rng = random.Random(seed)
        self._replaying = False
        if resume and self.path.exists():
            self.resume()

    # -- persistence

    def _append(self, event: dict) -> None:
        if self._replaying:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def resume(self) -> int:
        """Exact replay. A torn final line is dropped, not repaired.

        Returns the number of events replayed. Half a JSON object is not evidence of
        anything, and guessing at its contents is how a resume quietly invents a node
        that never ran.
        """
        self._nodes.clear()
        self._order.clear()
        self._children.clear()
        self._trunk_id = None
        n = 0
        self._replaying = True
        try:
            with open(self.path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        break  # torn tail: everything after it is unreadable too
                    self._apply(event)
                    n += 1
        finally:
            self._replaying = False
        return n

    def _apply(self, event: dict) -> None:
        kind = event["event"]
        if kind == "add":
            self.add(decode_node(event["node"]))
        elif kind == "update":
            self.update(event["node_id"], _replay=True, **_decode_fields(event["fields"]))
        elif kind == "promote":
            self.promote(event["node_id"])
        elif kind == "prune":
            self.prune_subtree(event["node_id"], event["reason"])

    # -- mutation

    def add(self, node: Node) -> str:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node_id {node.node_id!r}")
        if node.parent_id is not None and node.parent_id not in self._nodes:
            raise UnknownNodeError(f"parent {node.parent_id!r} is not in the tree")
        self._nodes[node.node_id] = node
        self._order.append(node.node_id)
        if node.parent_id is not None:
            self._children.setdefault(node.parent_id, []).append(node.node_id)
        self._append({"event": "add", "node": encode_node(node)})
        return node.node_id

    #: Fields that describe the *result* of an attempt. Once the attempt has a result,
    #: they are the record of what happened and may not be rewritten. Status is not on
    #: the list: promotion and pruning are pointer and bookkeeping operations, not
    #: revisions of the measurement.
    FROZEN_ONCE_SCORED = ("valid", "per_seed", "proposal", "code_sha256", "code_path")

    def update(self, node_id: str, _replay: bool = False, **fields) -> Node:
        node = self.get(node_id)
        scored = node.status in _STATUS_ELIGIBLE or node.valid is not None
        if scored:
            frozen = [f for f in fields if f in self.FROZEN_ONCE_SCORED]
            if frozen:
                raise ImmutableNodeError(
                    f"node {node_id} is already scored; {', '.join(frozen)} cannot be "
                    "rewritten. A result that can be edited after the fact is not a "
                    "result. Add a child node instead."
                )
        if "status" in fields and not isinstance(fields["status"], NodeStatus):
            fields["status"] = NodeStatus(fields["status"])
        updated = replace(node, **fields)
        self._nodes[node_id] = updated
        if not _replay:
            self._append(
                {
                    "event": "update",
                    "node_id": node_id,
                    "fields": {
                        k: (v.value if isinstance(v, NodeStatus) else _jsonable(v))
                        for k, v in fields.items()
                    },
                }
            )
        return updated

    def record_result(
        self,
        node_id: str,
        valid: Metrics,
        per_seed: Iterable[Metrics],
        status: NodeStatus = NodeStatus.SCORED,
    ) -> Node:
        """The one legitimate write of a result, allowed exactly once."""
        node = self.get(node_id)
        if node.valid is not None:
            raise ImmutableNodeError(f"node {node_id} already has a result")
        updated = replace(node, valid=valid, per_seed=list(per_seed), status=status)
        self._nodes[node_id] = updated
        self._append(
            {
                "event": "update",
                "node_id": node_id,
                "fields": {
                    "valid": encode_metrics(valid),
                    "per_seed": [encode_metrics(m) for m in updated.per_seed],
                    "status": status.value,
                },
            }
        )
        return updated

    def promote(self, node_id: str) -> None:
        """Move the trunk pointer. Never mutates the previous trunk."""
        node = self.get(node_id)
        if node.valid is None:
            raise ValueError(f"cannot promote unscored node {node_id}")
        if node.status in (NodeStatus.FAILED, NodeStatus.PRUNED, NodeStatus.QUARANTINED):
            raise ValueError(f"cannot promote a {node.status.value} node ({node_id})")
        self._nodes[node_id] = replace(node, status=NodeStatus.PROMOTED)
        self._trunk_id = node_id
        self._append({"event": "promote", "node_id": node_id})

    def prune_subtree(self, node_id: str, reason: str) -> int:
        """Mark a node and everything descended from it PRUNED.

        The trunk and its ancestors are never pruned: the current best result is what
        the submission is built from, and a pruning rule that can reach it turns a
        bookkeeping mistake into a lost run.
        """
        protected = set(self.lineage(self._trunk_id)) if self._trunk_id else set()
        stack, pruned = [node_id], 0
        while stack:
            nid = stack.pop()
            if nid in protected:
                continue
            node = self._nodes.get(nid)
            if node is None or node.status is NodeStatus.PRUNED:
                continue
            self._nodes[nid] = replace(node, status=NodeStatus.PRUNED)
            pruned += 1
            stack.extend(self._children.get(nid, ()))
        self._append({"event": "prune", "node_id": node_id, "reason": reason})
        return pruned

    # -- queries

    def get(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise UnknownNodeError(node_id) from None

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list[Node]:
        return [self._nodes[i] for i in self._order]

    def lineage(self, node_id: str | None) -> list[str]:
        out = []
        while node_id is not None:
            out.append(node_id)
            node_id = self._nodes[node_id].parent_id
        return out

    def trunk(self) -> Node:
        if self._trunk_id is None:
            raise RuntimeError("no trunk yet; seed the tree with the baseline node")
        return self._nodes[self._trunk_id]

    @property
    def trunk_id(self) -> str | None:
        return self._trunk_id

    def eligible_parents(self) -> list[Node]:
        """Scored, not degraded. FAILED, PRUNED and QUARANTINED nodes are excluded --
        building on a quarantined node would launder presumed leakage into the trunk
        through a child that never triggers the check itself."""
        return [n for n in self.nodes if n.status in _STATUS_ELIGIBLE and n.valid is not None]

    def axis_attempts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.proposal.axis] = counts.get(n.proposal.axis, 0) + 1
        return counts

    def select_parent(self, axis: Axis | None = None) -> Node:
        """Exploit the trunk most of the time; otherwise branch from an
        under-explored, still-healthy node.

        Pure greed on the trunk collapses the tree into the linear lineage this
        structure exists to avoid, and pure exploration wastes a short run. The split
        is `EXPLOIT_TRUNK_PROBABILITY`; the exploration arm prefers a parent on an axis
        the tree has touched least, so no axis starves.
        """
        candidates = self.eligible_parents()
        if not candidates:
            return self.trunk()
        if self._rng.random() < self.exploit_p:
            return self.trunk()

        attempts = self.axis_attempts()
        if axis is not None:
            same = [n for n in candidates if n.proposal.axis == axis]
            if same:
                return max(same, key=lambda n: n.valid.primary)
        best = min(
            candidates,
            key=lambda n: (attempts.get(n.proposal.axis, 0), -n.valid.primary),
        )
        return best

    def history(self) -> list[Metrics | None]:
        """Per-iteration metrics in creation order, `None` for a failed iteration.

        This is what `Evaluator.convergence` consumes: it counts scored runs only, so
        three crashes in a row are a broken branch rather than three non-improvements.
        """
        return [n.valid for n in self.nodes if n.iteration > 0]


def _decode_fields(fields: dict) -> dict:
    """Turn a replayed `update` event back into dataclass values.

    Only the fields that are not plain JSON need it. Getting this wrong is how a
    resumed run comes back with `valid` as a dict and every later comparison against
    `.primary` raising an AttributeError three iterations in.
    """
    out = dict(fields)
    if "valid" in out:
        out["valid"] = decode_metrics(out["valid"])
    if "per_seed" in out:
        out["per_seed"] = [decode_metrics(m) for m in out["per_seed"]]
    if "resources" in out and out["resources"] is not None:
        out["resources"] = ResourceFacts(**out["resources"])
    if "tokens" in out and out["tokens"] is not None:
        out["tokens"] = TokenUsage(**out["tokens"])
    if "failures" in out:
        out["failures"] = [
            FailureRecord(
                cls=FailureClass(f["cls"]),
                signature=f["signature"],
                traceback_tail=f["traceback_tail"],
                frame_context=f["frame_context"],
                repair_attempt=f["repair_attempt"],
            )
            for f in out["failures"]
        ]
    return out


def _jsonable(value):
    if isinstance(value, Metrics):
        return encode_metrics(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (FailureRecord, ResourceFacts, TokenUsage)):
        d = asdict(value)
        for k, v in d.items():
            if isinstance(v, FailureClass):
                d[k] = v.value
        return d
    return value


# ---------------------------------------------------------------- the ledger


#: The organizers' three published dead ends, each with its mechanism. Mechanisms
#: rather than prohibitions on purpose: "do not add static features" decays into
#: noise over thirty turns, while "the user_id x video_id cross already absorbs the
#: learnable signal" is a fact the agent can reason *from* -- including reasoning its
#: way to a case where it does not apply.
PUBLISHED_DEAD_ENDS: tuple[Insight, ...] = (
    Insight(
        axis="architecture",
        technique="static_feature_expansion",
        verdict=Verdict.DISCARD,
        delta_primary=-0.0010,
        delta_gauc=0.0,
        delta_ndcg5=0.0,
        n_seeds=1,
        mechanism=(
            "organizers measured: all 13 CWM feature fields gives primary 0.5940 vs "
            "0.5950 for the 5-field baseline. The user_id x video_id cross already "
            "absorbs most of the learnable signal, and coarse buckets like "
            "follow_user_num_range are redundant in the presence of user_id"
        ),
    ),
    Insight(
        axis="architecture",
        technique="embedding_capacity",
        verdict=Verdict.DISCARD,
        delta_primary=-0.0005,
        delta_gauc=0.0,
        delta_ndcg5=0.0,
        n_seeds=1,
        mechanism=(
            "organizers measured: embedding dim k = 8 / 16 / 32 gives 0.5895 / 0.5902 "
            "/ 0.5887. 1.14M rows cannot support more capacity; the bottleneck is not "
            "capacity, so tuning k is a way to spend iterations on noise"
        ),
    ),
    Insight(
        axis="architecture",
        technique="user_side_first_order_terms",
        verdict=Verdict.DISCARD,
        delta_primary=0.0,
        delta_gauc=0.0,
        delta_ndcg5=0.0,
        n_seeds=1,
        mechanism=(
            "exactly zero, by arithmetic rather than by experiment: ranking is within "
            "user, so any term constant inside a user's group leaves that group's "
            "ordering unchanged. User-side features can only act through crosses with "
            "the item side. Measured confirmation: item_pop x user_bias scores "
            "identically to item_pop alone, to the last digit"
        ),
    ),
)


class FeatureInsightsLedger:
    """What has been ruled in, ruled out, and genuinely left open.

    Keyed by `(axis, technique)` rather than by feature, because the question the next
    iteration asks is "has this *direction* been settled", and a technique is the unit
    at which that is answerable.
    """

    def __init__(self, seed_dead_ends: bool = True):
        self._insights: dict[tuple[str, str], Insight] = {}
        self._order: list[tuple[str, str]] = []
        if seed_dead_ends:
            for ins in PUBLISHED_DEAD_ENDS:
                self._put(ins)

    def _put(self, ins: Insight) -> None:
        key = (ins.axis, ins.technique)
        if key not in self._insights:
            self._order.append(key)
        self._insights[key] = ins

    def get(self, axis: str, technique: str) -> Insight | None:
        return self._insights.get((axis, technique))

    def __len__(self) -> int:
        return len(self._insights)

    @property
    def insights(self) -> list[Insight]:
        return [self._insights[k] for k in self._order]

    @staticmethod
    def verdict_for(gate: GateDecision, metrics: Metrics | None) -> Verdict:
        """Tri-state, and INCONCLUSIVE is the honest default.

        At sigma ~= 0.001 most single-run deltas genuinely are inconclusive. Forcing a
        binary verdict manufactures knowledge that did not exist, and that fabricated
        knowledge then ships in every subsequent prompt -- the worst failure mode a
        memory system has, because it is invisible and compounding.
        """
        if gate.quarantined:
            return Verdict.DISCARD
        if gate.promote:
            return Verdict.KEEP
        seeds = metrics.n_seeds if metrics else gate.seeds_run
        if seeds >= len(C.CONFIRM_SEEDS) and gate.delta_primary <= -C.PROMOTE_DELTA:
            return Verdict.DISCARD
        if seeds < len(C.CONFIRM_SEEDS) and gate.delta_primary <= C.PRUNE_AT_ONE_SEED_DELTA:
            # A single-seed result this far below the incumbent is ~6 sigma; it is a
            # broken candidate, not an unlucky one.
            return Verdict.DISCARD
        return Verdict.INCONCLUSIVE

    def record(self, node: Node, gate: GateDecision, mechanism: str = "") -> Insight:
        technique = node.proposal.technique or node.change_summary[:48] or node.node_id
        key = (node.proposal.axis, technique)
        prior = self._insights.get(key)
        ins = Insight(
            axis=node.proposal.axis,
            technique=technique,
            verdict=self.verdict_for(gate, node.valid),
            delta_primary=round(gate.delta_primary, 5),
            delta_gauc=round(gate.delta_gauc, 5),
            delta_ndcg5=round(gate.delta_ndcg5, 5),
            n_seeds=gate.seeds_run,
            mechanism=mechanism or node.proposal.hypothesis,
            node_ids=(prior.node_ids if prior else ()) + (node.node_id,),
        )
        self._put(ins)
        return ins

    def render(self, max_tokens: int = 1400) -> str:
        """The prompt's tier-B block: one line per experiment, ~30 tokens each.

        Append-only ordering so that the block only ever grows at the end. Rewriting
        earlier lines would be a needless invalidation of everything after it in the
        prompt, and would also hide the run's own history from the agent mid-run.
        """
        lines = ["## What is already known (ledger, keyed by axis/technique)"]
        budget = max_tokens * 4  # ~4 chars per token
        used = len(lines[0])
        for ins in self.insights:
            sign = f"{ins.delta_primary:+.4f}" if ins.n_seeds else "n/a"
            line = (
                f"- [{ins.verdict.value}] {ins.axis}/{ins.technique}: "
                f"Δprimary {sign} over {ins.n_seeds} seed(s) — {ins.mechanism}"
            )
            if used + len(line) > budget:
                lines.append(f"- ... {len(self.insights) - (len(lines) - 1)} older entries elided")
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def realised_by_axis(self) -> dict[str, float]:
        """Best realised Δprimary per axis -- the weight the loop's axis choice uses."""
        out: dict[str, float] = {}
        for ins in self.insights:
            if ins.n_seeds:
                out[ins.axis] = max(out.get(ins.axis, float("-inf")), ins.delta_primary)
        return out
