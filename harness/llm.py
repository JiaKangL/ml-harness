"""L5 -- the Anthropic client. Thin on purpose.

No framework. Prompt caching needs byte-exact request control -- a frozen prefix
identical across ~30 calls on a 1h TTL -- and framework layers silently break that;
the miss raises nothing and shows up only on the bill. We also do none of what a
framework is for: no RAG, no vector store, no tool-calling chain. The agent is
code-emitting, not tool-using.

Token accounting is a deliverable, not telemetry: Feasibility is 15% of the rubric and
is graded on total tokens and wall-clock. Usage is therefore extracted off **every**
response -- repair calls and critic calls included, which are the easiest to forget and
exactly the ones that inflate the total.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Protocol

from . import config as C
from .types import TokenUsage


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: TokenUsage
    stop_reason: str = "end_turn"
    model: str = C.MODEL


class LLM(Protocol):
    """What `agent.py` and `critics.py` code against. `MockLLM` satisfies it too, which
    is what lets the whole loop run end-to-end in seconds with zero tokens."""

    def complete(
        self, prefix: str, messages: list[dict], operator: str | None = None
    ) -> LLMResponse: ...


def price(usage_obj, model: str = C.MODEL) -> float:
    """USD from a raw SDK usage object, at the rates for `model`.

    Per model, not a constant: cost is a graded deliverable, and pricing a Sonnet run
    at Opus rates would overstate it 2.5x in the artifact a judge reads.
    """
    p_in, p_out, p_read, p_write = C.prices_for(model)
    return (
        getattr(usage_obj, "input_tokens", 0) * p_in
        + getattr(usage_obj, "output_tokens", 0) * p_out
        + (getattr(usage_obj, "cache_read_input_tokens", 0) or 0) * p_read
        + (getattr(usage_obj, "cache_creation_input_tokens", 0) or 0) * p_write
    ) / 1_000_000


class AnthropicLLM:
    """One call shape, used by the agent, the repair turn and the critics alike."""

    def __init__(
        self,
        model: str = C.MODEL,
        max_tokens: int = C.MAX_TOKENS,
        api_key: str | None = None,
        max_retries: int = C.LLM_MAX_RETRIES,
        base_url: str | None = None,
        auth_token: str | None = None,
    ):
        """Credentials come from `config`, which reads them from the environment.

        The endpoint is configurable because the credential running this may be issued
        by a gateway rather than by Anthropic directly. What is *not* negotiable is the
        wire protocol: this client sends `cache_control` with a 1h TTL and adaptive
        thinking, and reads the token accounting -- a graded deliverable -- straight off
        Anthropic's `usage` fields. A gateway that merely proxies an OpenAI-shaped API
        will not fail loudly here; it will drop the cache directives, and the only
        symptom is the bill. So point this at an Anthropic-compatible endpoint or at
        Anthropic itself, and nothing in between.
        """
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "the anthropic SDK is not installed; `pip install anthropic`, or run "
                "the loop with --mock, which needs no network and no key"
            ) from exc
        self._anthropic = anthropic

        kwargs: dict[str, str] = {}
        key = api_key or C.LLM_API_KEY
        token = auth_token or C.LLM_AUTH_TOKEN
        url = base_url or C.LLM_BASE_URL
        if key:
            kwargs["api_key"] = key
        elif token:
            # Bearer rather than x-api-key: OAuth tokens and most gateway credentials
            # arrive this way, and the SDK sends them on Authorization instead.
            kwargs["auth_token"] = token
        if url:
            kwargs["base_url"] = url

        if C.PLACEHOLDER_VARS and not (key or token):
            # Before constructing: the SDK reads these variables itself, so leaving the
            # filler in place would hand it a credential we deliberately rejected.
            raise LLMError(
                f"{', '.join(C.PLACEHOLDER_VARS)} still holds the placeholder from "
                f"`.env.example`. Edit `.env` and put a real credential there — or "
                f"delete the line. `--mock` needs no credential at all."
            )

        self.client = anthropic.Anthropic(**kwargs)

        # Fail here, not at the first request. The SDK resolves credentials lazily and
        # raises only when a call is made, which on this harness means after preflight,
        # after the FM baseline has been scored on three seeds, and roughly a minute
        # into a run the operator expected to leave unattended.
        if not (self.client.api_key or self.client.auth_token):
            raise LLMError(
                "no LLM credential resolved. Set one of:\n"
                "  HARNESS_LLM_API_KEY      (or ANTHROPIC_API_KEY)     -- sent as x-api-key\n"
                "  HARNESS_LLM_AUTH_TOKEN   (or ANTHROPIC_AUTH_TOKEN)  -- sent as a bearer token\n"
                "and, for a gateway rather than Anthropic directly:\n"
                "  HARNESS_LLM_BASE_URL     (or ANTHROPIC_BASE_URL)\n"
                "  HARNESS_LLM_MODEL        if the gateway names the model differently\n"
                "`--mock` runs the whole loop with none of them."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def describe(self) -> str:
        """What this client will actually talk to. Printed at the top of a live run so
        an operator can see which endpoint and model a run's cost was spent on."""
        endpoint = C.LLM_BASE_URL or "api.anthropic.com (default)"
        credential = (
            "HARNESS_LLM_API_KEY/ANTHROPIC_API_KEY" if C.LLM_API_KEY
            else "HARNESS_LLM_AUTH_TOKEN/ANTHROPIC_AUTH_TOKEN" if C.LLM_AUTH_TOKEN
            else "none"
        )
        return f"model {self.model} via {endpoint} (credential: {credential})"

    def complete(
        self, prefix: str, messages: list[dict], operator: str | None = None
    ) -> LLMResponse:
        """`prefix` is tier A and is the cache breakpoint. Everything else follows it.

        The `ttl` is not optional. A bare `{"type": "ephemeral"}` is the 5-minute TTL,
        and one iteration is 3-5 minutes with numpy and longer with torch -- it
        straddles the boundary, so we would get erratic hits and never notice. The 1h
        TTL costs 2x on write and breaks even after three reads; we expect ~30.
        """
        payload = list(messages)
        if operator:
            # A per-turn operator instruction as a *message*, not by editing the
            # top-level system field: editing the prefix invalidates the cache for every
            # call after it, which is the whole thing we are protecting.
            #
            # `role: "system"` mid-conversation carries operator authority, but only
            # some models accept it -- Sonnet 5 returns a 400. Falling back to a user
            # message keeps the cached prefix intact either way, which is what actually
            # matters here; the authority distinction does not, because there is no
            # untrusted party in this conversation to distinguish it from.
            role = "system" if self.model in C.MID_CONVERSATION_SYSTEM else "user"
            payload.append({"role": role, "content": operator})

        started = time.monotonic()
        message = self._call(prefix, payload)
        latency = time.monotonic() - started

        text = "".join(b.text for b in message.content if b.type == "text")
        u = message.usage
        usage = TokenUsage(
            prompt_tokens=getattr(u, "input_tokens", 0),
            completion_tokens=getattr(u, "output_tokens", 0),
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            latency_seconds=round(latency, 2),
            cost_usd=price(u, self.model),
        )
        return LLMResponse(
            text=text,
            usage=usage,
            stop_reason=message.stop_reason or "end_turn",
            model=getattr(message, "model", self.model),
        )

    def _call(self, prefix: str, messages: list[dict]):
        """Streamed, because a turn that writes a 400-line script at 32K max_tokens
        will otherwise sit past the HTTP timeout and fail as a network error."""
        anthropic = self._anthropic
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": prefix,
                            "cache_control": {"type": "ephemeral", "ttl": C.CACHE_TTL},
                        }
                    ],
                    thinking={"type": "adaptive"},
                    output_config={"effort": "high"},
                    messages=messages,
                ) as stream:
                    return stream.get_final_message()
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
                last = exc
            except anthropic.APIStatusError as exc:
                if exc.status_code < 500:
                    raise  # a 400 is our bug; retrying it just spends money slower
                last = exc
            if attempt < self.max_retries:
                time.sleep(min(2.0 * 2 ** attempt + random.uniform(0, 0.5), 30.0))
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last}")


# ---------------------------------------------------------------- mock mode


#: The four mock candidates, in the order `MockLLM` serves them. Built in this phase
#: rather than after it: on a one-day build, this is the difference between debugging
#: the loop and debugging the loop *while waiting on API calls*. Each one exercises a
#: distinct branch of L2-L6 that is otherwise reachable only by getting unlucky.
MOCK_SCRIPTS: dict[str, str] = {}

MOCK_SCRIPTS["popularity"] = '''\
"""Smoothed item popularity. Cheap, deterministic, and a real ranking."""
import argparse

import numpy as np

from harness.data_guard import DataAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    api = DataAPI()
    vid = api.features("train")["video_id"].astype(np.int64)
    y = api.labels("train").astype(np.float64)
    n = max(int(vid.max()), int(api.features(args.split)["video_id"].max())) + 1
    imp = np.bincount(vid, minlength=n)
    pos = np.bincount(vid, weights=y, minlength=n)
    prior = float(y.mean())
    rate = (pos + 20.0 * prior) / (imp + 20.0)
    np.save(args.out, rate[api.features(args.split)["video_id"].astype(np.int64)])


if __name__ == "__main__":
    main()
'''

MOCK_SCRIPTS["popularity_plus_duration"] = '''\
"""Popularity crossed with the duration bucket -- a within-user-varying interaction."""
import argparse

import numpy as np

from harness.data_guard import DataAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    api = DataAPI()
    tr = api.features("train")
    y = api.labels("train").astype(np.float64)
    prior = float(y.mean())

    def rates(keys, n):
        imp = np.bincount(keys, minlength=n)
        pos = np.bincount(keys, weights=y, minlength=n)
        return (pos + 20.0 * prior) / (imp + 20.0)

    ev = api.features(args.split)
    n_vid = max(int(tr["video_id"].max()), int(ev["video_id"].max())) + 1
    vid_rate = rates(tr["video_id"].astype(np.int64), n_vid)

    edges = np.quantile(tr["duration_ms"].astype(np.float64), np.linspace(0, 1, 11)[1:-1])
    tr_bucket = np.searchsorted(edges, tr["duration_ms"].astype(np.float64))
    ev_bucket = np.searchsorted(edges, ev["duration_ms"].astype(np.float64))
    bucket_rate = rates(tr_bucket.astype(np.int64), len(edges) + 1)

    score = np.log(vid_rate[ev["video_id"].astype(np.int64)] + 1e-6) + 0.3 * np.log(
        bucket_rate[ev_bucket] + 1e-6
    )
    np.save(args.out, score.astype(np.float64))


if __name__ == "__main__":
    main()
'''

MOCK_SCRIPTS["crasher"] = '''\
"""Exercises the self-healing path: a NameError that only fires at runtime."""
import argparse

import numpy as np

from harness.data_guard import DataAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    api = DataAPI()
    feats = api.features(args.split)
    scores = np.zeros(feats["user_id"].shape[0], dtype=np.float64)
    scores += smoothed_rate  # noqa: F821 -- deliberately undefined
    np.save(args.out, scores)


if __name__ == "__main__":
    main()
'''

MOCK_SCRIPTS["constant"] = '''\
"""Exercises output validation: runs perfectly, expresses no ranking at all."""
import argparse

import numpy as np

from harness.data_guard import DataAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    api = DataAPI()
    n = api.features(args.split)["user_id"].shape[0]
    np.save(args.out, np.full(n, 0.42, dtype=np.float64))


if __name__ == "__main__":
    main()
'''

MOCK_SCRIPTS["leaker"] = '''\
"""Exercises the leakage firewall: asks for play_time_ms on an evaluation split."""
import argparse

import numpy as np

from harness.data_guard import DataAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    api = DataAPI()
    play = api.column(args.split, "play_time_ms").astype(np.float64)
    dur = api.features(args.split)["duration_ms"].astype(np.float64)
    np.save(args.out, np.divide(play, dur, out=np.zeros_like(play), where=dur > 0))


if __name__ == "__main__":
    main()
'''


def _mock_reply(
    hypothesis: str,
    axis: str,
    technique: str,
    grounding: str,
    predicted: float,
    summary: str,
    code: str,
) -> str:
    return (
        f"<hypothesis>\n{hypothesis}\n</hypothesis>\n"
        f"<axis>{axis}</axis>\n"
        f"<technique>{technique}</technique>\n"
        f"<grounding>{grounding}</grounding>\n"
        f"<predicted_delta>{predicted}</predicted_delta>\n"
        f"<change_summary>{summary}</change_summary>\n"
        f"<code>\n```python\n{code}```\n</code>\n"
    )


MOCK_SEQUENCE: list[str] = [
    _mock_reply(
        "Item quality dominates and varies inside 99% of user groups, so a smoothed "
        "per-video long_view rate should already rank most groups correctly at a "
        "fraction of FM's cost.",
        "architecture",
        "smoothed_item_popularity",
        "item_level_signal.long_view_rate",
        0.004,
        "Scored by smoothed per-video long_view rate",
        MOCK_SCRIPTS["popularity"],
    ),
    _mock_reply(
        "A NameError candidate, to exercise the repair path.",
        "loss",
        "broken_candidate",
        "within_user_variance.video_id",
        0.003,
        "Deliberately references an undefined name",
        MOCK_SCRIPTS["crasher"],
    ),
    _mock_reply(
        "A constant scorer, to exercise the silent-wrong check.",
        "architecture",
        "constant_scorer",
        "splits.valid.rows",
        0.002,
        "Emits one constant score per row",
        MOCK_SCRIPTS["constant"],
    ),
    _mock_reply(
        "Reading play_time_ms on the evaluation split, to exercise the firewall.",
        "watchtime",
        "play_ratio_feature",
        "aux_signal_correlation_with_label.play_time_ms",
        0.05,
        "Uses play_time_ms / duration_ms as an inference feature",
        MOCK_SCRIPTS["leaker"],
    ),
    _mock_reply(
        "Duration varies within 98% of groups while item popularity does not capture "
        "it, so crossing the two should add ordering information popularity alone "
        "cannot express.",
        "architecture",
        "popularity_x_duration",
        "within_user_variance.duration_bucket_10s",
        0.002,
        "Crossed item popularity with the duration-bucket rate",
        MOCK_SCRIPTS["popularity_plus_duration"],
    ),
]


class MockLLM:
    """Serves the pre-written replies in order, then cycles. Zero tokens, no network.

    The repair turn is answered with the *last successful* script rather than another
    broken one, so the self-heal path terminates and the loop's circuit breaker is
    exercised by the dedicated test instead of by every mock run.
    """

    def __init__(self, sequence: list[str] | None = None):
        self.sequence = list(sequence if sequence is not None else MOCK_SEQUENCE)
        self.calls: list[list[dict]] = []
        self._i = 0

    def complete(
        self, prefix: str, messages: list[dict], operator: str | None = None
    ) -> LLMResponse:
        self.calls.append(list(messages))
        is_repair = any(
            "Repair attempt" in str(m.get("content", "")) for m in messages
        ) or (operator or "").startswith("## Your script failed")
        if is_repair:
            text = _mock_reply(
                "Repaired: the undefined name is replaced with the smoothed per-video "
                "rate it was meant to be.",
                "loss",
                "broken_candidate",
                "item_level_signal.long_view_rate",
                0.003,
                "Repaired the undefined name",
                MOCK_SCRIPTS["popularity"],
            )
        else:
            text = self.sequence[self._i % len(self.sequence)]
            self._i += 1
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=len(prefix) // 4,
                completion_tokens=len(text) // 4,
                cache_read_tokens=0 if len(self.calls) == 1 else len(prefix) // 4,
                latency_seconds=0.0,
                cost_usd=0.0,
            ),
        )
