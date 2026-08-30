"""The single scoring path. L1, so every layer may use it.

Extracted because preflight and the evaluator would otherwise each wrap the frozen
`evaluate.py` themselves, and two implementations of the scoring path drift -- at
which point the FM reproduction number and the promotion numbers stop being
commensurable and nobody notices.

It also owns the starter-kit `sys.path` insertion. That used to happen as an import
side effect of `harness.preflight`, so anything calling `from evaluate import
evaluate` worked only if preflight had been imported first somewhere -- an invisible
runtime coupling from higher layers onto an L1 module's import order.
"""
from __future__ import annotations

import hashlib
import statistics
import sys

import numpy as np

from . import config as C
from .types import Metrics, Split

if str(C.STARTER_KIT) not in sys.path:
    sys.path.insert(0, str(C.STARTER_KIT))

from evaluate import evaluate as _official_evaluate  # noqa: E402


class EvaluateModifiedError(RuntimeError):
    """The frozen metric changed. Refuse to score rather than report against it."""


def evaluate_sha256() -> str:
    h = hashlib.sha256()
    with open(C.STARTER_KIT / "evaluate.py", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_evaluate_unmodified(expected: str) -> None:
    actual = evaluate_sha256()
    if actual != expected:
        raise EvaluateModifiedError(
            f"evaluate.py changed since preflight ({expected[:12]} -> {actual[:12]}). "
            "It is the sole source of truth for scoring and must not be modified."
        )


def score(
    user_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int = 42,
) -> Metrics:
    """Wrap the frozen evaluate.py. Never reimplements it.

    `evaluate()` costs ~0.12s on a valid-shaped split, so there is no reason to build
    a faster version -- correctness here is worth far more than speed. Labels arrive
    as int64 from DataAPI: evaluate.py aggregates with builtin sum(), and an int8
    array wraps past 127 positives under NumPy 2's weak promotion.
    """
    r = _official_evaluate(user_ids, labels, scores)
    return Metrics(
        gauc=float(r["GAUC"]),
        ndcg5=float(r["nDCG@5"]),
        primary=float(r["primary"]),
        users=int(r["users"]),
        rows=int(r["rows"]),
        seeds=(seed,),
    )


def aggregate(per_seed: list[Metrics]) -> Metrics:
    """Mean across seeds, carrying the spread.

    The spread is not decoration: a candidate whose seeds disagree is unstable, and
    that belongs in the ledger as INCONCLUSIVE rather than KEEP.
    """
    if not per_seed:
        raise ValueError("aggregate() needs at least one Metrics")
    if len(per_seed) == 1:
        return per_seed[0]
    primaries = [m.primary for m in per_seed]
    return Metrics(
        gauc=statistics.fmean(m.gauc for m in per_seed),
        ndcg5=statistics.fmean(m.ndcg5 for m in per_seed),
        primary=statistics.fmean(primaries),
        users=per_seed[0].users,
        rows=per_seed[0].rows,
        seeds=tuple(s for m in per_seed for s in m.seeds),
        primary_std=statistics.stdev(primaries),
    )
