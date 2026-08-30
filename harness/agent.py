"""L5 -- context assembly, and the validation that happens before anything runs.

Two jobs. Assemble the three prompt tiers so the frozen one stays byte-identical, and
turn the reply into a `Proposal` that is either runnable or rejected for a stated
reason. A rejection here costs no compute, so the bar for rejecting is "this cannot
teach us anything", not "this looks unpromising".
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Iterable

from . import config as C
from . import prompts
from .llm import LLM, LLMResponse
from .types import Insight, Metrics, Node, PRIORITY_AXES, Proposal, TokenUsage

VALID_AXES: tuple[str, ...] = (
    "loss", "sequence", "multitask", "watchtime",
    "architecture", "temporal", "debias", "ensemble",
)


class ProposalRejected(Exception):
    """Rejected before execution. `rule` is the ledger key; `message` goes to the LLM."""

    def __init__(self, rule: str, message: str):
        super().__init__(message)
        self.rule = rule
        self.message = message


@dataclass(frozen=True)
class Generation:
    proposal: Proposal
    change_summary: str
    grounding_verified: bool
    usage: TokenUsage
    raw_text: str


# ---------------------------------------------------------------- parsing


_TAG = r"<{t}>\s*(.*?)\s*</{t}>"


def _tag(name: str, text: str) -> str | None:
    m = re.search(_TAG.format(t=name), text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def extract_code(text: str) -> str | None:
    """The script, from inside `<code>` and its fence.

    Falls back to a bare fenced block when the tag is missing. Being generous here is
    correct: the whole script is present and runnable, and refusing it over a dropped
    closing tag spends a real iteration on a formatting slip.
    """
    block = _tag("code", text)
    haystack = block if block is not None else text
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", haystack, re.DOTALL)
    if fence:
        return fence.group(1)
    if block and "import" in block:
        return block
    return None


def parse_reply(text: str) -> tuple[Proposal, str]:
    """`(proposal, change_summary)`; raises `ProposalRejected` on anything unrunnable."""
    code = extract_code(text)
    if not code or not code.strip():
        raise ProposalRejected(
            "no_code",
            "No script found. Put the complete script inside <code> and a ```python "
            "fence, as the output format specifies.",
        )

    axis = (_tag("axis", text) or "").strip().lower()
    if axis not in VALID_AXES:
        raise ProposalRejected(
            "bad_axis",
            f"axis {axis!r} is not one of the eight: {', '.join(VALID_AXES)}. The axis "
            "set is closed because it is also the ledger's key -- an axis outside it "
            "cannot be looked up later.",
        )

    raw_delta = (_tag("predicted_delta", text) or "").strip()
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_delta)
    if not match:
        # The one hard rejection on content. Without a prediction the iteration is a
        # search step, not a hypothesis test, and the calibration statistic -- the
        # measure of whether the agent reasons or guesses -- has nothing to record.
        raise ProposalRejected(
            "no_predicted_delta",
            "predicted_delta is missing or unparseable. It is required: without a "
            "prediction there is no hypothesis to test, only a search step. Give a "
            "signed float, e.g. <predicted_delta>0.004</predicted_delta>.",
        )

    hypothesis = (_tag("hypothesis", text) or "").strip()
    if len(hypothesis) < 40:
        raise ProposalRejected(
            "no_hypothesis",
            "The hypothesis is missing or too short. State what you are changing, why "
            "you expect it to help, and through what mechanism it acts on within-user "
            "ordering. This text is read and scored directly.",
        )

    proposal = Proposal(
        hypothesis=hypothesis,
        axis=axis,  # type: ignore[arg-type]
        grounding=(_tag("grounding", text) or "").strip(),
        predicted_delta=float(match.group(0)),
        code=code,
        technique=(_tag("technique", text) or "").strip().lower().replace(" ", "_"),
    )
    return proposal, (_tag("change_summary", text) or "").strip()


# ---------------------------------------------------------------- grounding


def profile_keys(profile: dict) -> list[str]:
    """Dotted paths through the profile. Leading underscores (the `_why` notes) are
    skipped -- they are prose for the reader, not citable facts."""
    out: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k.startswith("_"):
                    continue
                p = f"{path}.{k}" if path else k
                out.append(p)
                walk(v, p)

    walk(profile, "")
    return out


def resolve_grounding(grounding: str, keys: Iterable[str]) -> tuple[str, bool]:
    """Exact, then fuzzy, then run it anyway.

    Advisory rather than blocking on purpose. The field exists to make the agent reason
    from measured data rather than from score feedback, and a paraphrased key still
    demonstrates that. Refusing to run would spend a real iteration punishing a
    spelling mistake; `grounding_verified=False` makes the rate visible instead, which
    is the honest way to report it.
    """
    keys = list(keys)
    cleaned = grounding.strip().strip("`").strip()
    if not cleaned:
        return grounding, False
    if cleaned in keys:
        return cleaned, True
    lowered = {k.lower(): k for k in keys}
    if cleaned.lower() in lowered:
        return lowered[cleaned.lower()], True
    close = difflib.get_close_matches(
        cleaned, keys, n=1, cutoff=C.GROUNDING_FUZZY_CUTOFF
    )
    if close:
        return close[0], True

    # Separator-insensitive: a citation of "tab_variance" or "tab variance" is the same
    # claim as "within_user_variance.tab", and punishing the difference measures the
    # agent's memory for our key layout rather than its reasoning.
    normalised = {_norm(k): k for k in keys}
    close = difflib.get_close_matches(
        _norm(cleaned), list(normalised), n=1, cutoff=C.GROUNDING_FUZZY_CUTOFF
    )
    if close:
        return normalised[close[0]], True

    # Token containment, shortest key first: every word of the citation appears in the
    # key. This is what actually catches paraphrase ("tab variance" -> the within-user
    # variance of tab) once the character-level measures have run out.
    words = set(_tokens(cleaned))
    if words:
        containing = [k for k in keys if words <= set(_tokens(k))]
        if containing:
            return min(containing, key=len), True
    return grounding, False


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


# ---------------------------------------------------------------- dead-end lint


#: A changed line matching one of these, and nothing else changed, means the candidate
#: is one of the organizers' two measured dead ends wearing a new hypothesis.
_FIELDS_LINE = re.compile(r"\bFIELDS\b|\bfields\s*=\s*[\[\(]", re.IGNORECASE)
_DIM_LINE = re.compile(
    r"\b(k|dim|emb(edding)?_?dim|n_factors|latent_dim)\s*=\s*\d+", re.IGNORECASE
)


def _significant_changes(parent_code: str, code: str) -> list[str]:
    a = [l for l in parent_code.splitlines() if l.strip() and not l.strip().startswith("#")]
    b = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    return [
        line[1:].strip()
        for line in difflib.unified_diff(a, b, n=0, lineterm="")
        if line[:1] in "+-" and not line.startswith(("+++", "---")) and line[1:].strip()
    ]


def reject_known_dead_end(parent_code: str, code: str) -> str | None:
    """A diff touching only the field list or a capacity constant. Both were measured
    by the organizers to do nothing, and both are published with counter-evidence, so
    running one buys an already-known answer at the price of a real iteration."""
    changed = _significant_changes(parent_code, code)
    if not changed:
        return (
            "The script is identical to its parent. Re-running the parent measures seed "
            "noise, not an idea."
        )
    if all(_FIELDS_LINE.search(l) for l in changed):
        return (
            "This changes only the feature-field list. The organizers measured that: "
            "all 13 CWM fields gives 0.5940 against 0.5950 for 5 fields. The mechanism "
            "is that the user_id x video_id cross already absorbs the learnable signal, "
            "so adding fields cannot help. Propose something that changes what is "
            "optimised or how, not which columns are indexed."
        )
    if all(_DIM_LINE.search(l) for l in changed):
        return (
            "This changes only a capacity constant. The organizers measured k = 8/16/32 "
            "at 0.5895/0.5902/0.5887 -- 1.14M rows cannot support more capacity, and "
            "the bottleneck is not capacity."
        )
    return None


# ---------------------------------------------------------------- the agent


class Agent:
    def __init__(
        self,
        llm: LLM,
        profile: dict | None = None,
        priority_axes: tuple[str, ...] = PRIORITY_AXES,
    ):
        self.llm = llm
        self.profile = profile if profile is not None else json.loads(
            C.DATA_PROFILE_JSON.read_text()
        )
        self._keys = profile_keys(self.profile)
        self.priority_axes = priority_axes

    # -- context

    @property
    def prefix(self) -> str:
        return prompts.static_prefix()

    def build_messages(
        self,
        *,
        iteration: int,
        parent: Node,
        parent_code: str,
        ledger_block: str,
        best: Metrics | None,
        assigned_axis: str | None,
        axis_reason: str = "",
        recent_changes: str = "",
        failure_context: str = "",
        resource_note: str = "",
    ) -> list[dict]:
        working = prompts.working_set(
            iteration=iteration,
            parent_code=parent_code,
            parent_summary=parent.change_summary or "the FM baseline",
            parent_metrics=_fmt(parent.valid),
            best_metrics=_fmt(best),
            assigned_axis=assigned_axis,
            axis_reason=axis_reason,
            recent_changes=recent_changes,
            failure_context=failure_context,
            resource_note=resource_note,
        )
        return [{"role": "user", "content": f"{ledger_block}\n\n{working}"}]

    # -- generation

    def propose(self, messages: list[dict], operator: str | None = None) -> Generation:
        response = self.llm.complete(self.prefix, messages, operator=operator)
        return self.validate(response)

    def validate(self, response: LLMResponse) -> Generation:
        proposal, summary = parse_reply(response.text)
        grounding, verified = resolve_grounding(proposal.grounding, self._keys)
        return Generation(
            proposal=Proposal(
                hypothesis=proposal.hypothesis,
                axis=proposal.axis,
                grounding=grounding,
                predicted_delta=proposal.predicted_delta,
                code=proposal.code,
                technique=proposal.technique,
            ),
            change_summary=summary,
            grounding_verified=verified,
            usage=response.usage,
            raw_text=response.text,
        )

    # -- pre-execution gates

    def check_axis_lock(self, axis: str, scored_axes: set[str]) -> None:
        """`architecture` is refused until each priority axis has a scored attempt.

        Not a hunch: switching the model is the organizers' own fifth-ranked direction
        precisely because capacity was measured not to be the bottleneck, and it is the
        most expensive thing the agent can do. Ordering the search so the cheap,
        high-prior axes are observed first is worth more than the freedom to start with
        a rewrite.
        """
        if axis != "architecture":
            return
        missing = [a for a in self.priority_axes if a not in scored_axes]
        if missing:
            raise ProposalRejected(
                "axis_locked",
                f"`architecture` is locked until each priority axis has a scored "
                f"attempt. Still unobserved: {', '.join(missing)}. The organizers rank "
                f"a model swap fifth precisely because capacity was measured not to be "
                f"the bottleneck -- take one of those axes first.",
            )

    def check_dead_end(self, parent_code: str, code: str) -> None:
        reason = reject_known_dead_end(parent_code, code)
        if reason:
            raise ProposalRejected("known_dead_end", reason)

    def check_ledger(self, proposal: Proposal, ledger_lookup) -> None:
        """A technique already settled on this axis is not re-run for free.

        Only KEEP and DISCARD block. INCONCLUSIVE deliberately does not: at this scale
        most single-run deltas are inconclusive, and refusing to look again at an
        undecided question would freeze the run's knowledge at its noisiest point.
        """
        prior: Insight | None = ledger_lookup(proposal.axis, proposal.technique)
        if prior is None or prior.verdict.value == "INCONCLUSIVE":
            return
        raise ProposalRejected(
            "already_settled",
            f"{proposal.axis}/{proposal.technique} is already {prior.verdict.value} in "
            f"the ledger (Δprimary {prior.delta_primary:+.4f} over {prior.n_seeds} "
            f"seed(s)): {prior.mechanism}. Propose a different technique, or say in the "
            f"hypothesis what you would do differently and why that changes the result.",
        )


def _fmt(m: Metrics | None) -> str:
    if m is None:
        return "not yet scored"
    return (
        f"primary {m.primary:.4f} (GAUC {m.gauc:.4f}, nDCG@5 {m.ndcg5:.4f})"
        + (f" ±{m.primary_std:.4f} over {m.n_seeds} seeds" if m.n_seeds > 1 else "")
    )
