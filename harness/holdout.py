"""Test labels: extracted on demand, at scoring time, behind a seal.

The earlier design wrote `cache/_holdout/test_labels.npy` during the cache build, so
the file existed for the whole run and "not read until testing" was a convention. Any
stray glob over `cache/` would have found it.

This version makes the claim structural instead: **no test-label artifact is ever
written.** Labels are parsed straight out of the organizer's raw log at the moment
score_final needs them, and only after the run has been *sealed* -- a marker written
when the loop converges, recording which node won and the sha256 of the submission
being scored.

So during a run there is nothing to read, and after the run reading requires an
explicit seal plus a single, greppable import site. What remains, honestly stated: the
raw CSV still contains those labels and a process running as this user can parse it.
Closing that needs a separate uid or a container. What we can now claim without
qualification is that the harness never materialises them, never holds them in a
process that also makes decisions, and cannot reach them before a converged run.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config as C


class RunNotSealedError(RuntimeError):
    """Test labels were requested before the run converged and was sealed."""


class AlreadyScoredError(RuntimeError):
    """The test set has already been scored once. A second draw biases the estimate."""


@dataclass(frozen=True)
class RunSeal:
    """Written by the loop on convergence. The precondition for touching test."""

    node_id: str
    submission_sha256: str
    valid_primary: float
    iterations: int
    converged_reason: str
    sealed_at: float


def seal_run(
    node_id: str,
    submission_sha256: str,
    valid_primary: float,
    iterations: int,
    converged_reason: str,
) -> RunSeal:
    seal = RunSeal(
        node_id=node_id,
        submission_sha256=submission_sha256,
        valid_primary=valid_primary,
        iterations=iterations,
        converged_reason=converged_reason,
        sealed_at=time.time(),
    )
    C.RUN_SEAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.RUN_SEAL_JSON.write_text(json.dumps(asdict(seal), indent=2))
    return seal


def read_seal() -> RunSeal | None:
    if not C.RUN_SEAL_JSON.exists():
        return None
    return RunSeal(**json.loads(C.RUN_SEAL_JSON.read_text()))


def is_sealed() -> bool:
    return C.RUN_SEAL_JSON.exists()


def has_been_scored() -> bool:
    return C.SCORED_MARKER_JSON.exists()


def extract_test_labels(force: bool = False) -> np.ndarray:
    """Parse test labels from the raw log. The only path to them in the repository.

    Refuses unless the run is sealed, and refuses a second time unless forced --
    scoring test twice makes the estimate optimistically biased, so the override is
    recorded rather than merely permitted.

    Row order matches `data.load()` exactly, which the cache reproduces byte for byte
    and preflight asserts, so these labels align positionally with `features("test")`.
    """
    if not is_sealed():
        raise RunNotSealedError(
            "The run is not sealed. Test labels are only reachable after the loop has "
            "converged and recorded which node won. This is the whole point of the "
            "holdout: it cannot inform any decision the loop makes."
        )
    if has_been_scored() and not force:
        raise AlreadyScoredError(
            "Test has already been scored once. A second draw on the held-out set "
            "makes the estimate optimistically biased -- if the harness changed, the "
            "honest move is to report the first number. Pass force=True to override; "
            "the override is logged as a second draw."
        )

    lo, hi = C.SPLITS["test"]
    labels: list[int] = []
    for fname in C.LOG_FILES:
        with open(C.DATA_DIR / fname, newline="") as fh:
            for row in csv.DictReader(fh):
                if lo <= int(row["date"]) <= hi:
                    labels.append(1 if row[C.LABEL] != "0" else 0)

    _record_draw(force=force, n=len(labels))
    return np.asarray(labels, dtype=np.int64)


def _record_draw(force: bool, n: int) -> None:
    """Append to the scoring ledger. Every draw on test is recorded, forever."""
    draws = []
    if C.SCORED_MARKER_JSON.exists():
        draws = json.loads(C.SCORED_MARKER_JSON.read_text())
    draws.append(
        {
            "at": time.time(),
            "rows": n,
            "forced_second_draw": bool(force and draws),
            "seal": asdict(read_seal()) if is_sealed() else None,
        }
    )
    C.SCORED_MARKER_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.SCORED_MARKER_JSON.write_text(json.dumps(draws, indent=2))


def assert_no_holdout_artifact() -> list[Path]:
    """Preflight assertion: nothing anywhere is a materialised test-label file.

    Inverted from the old design, where preflight checked that the holdout *existed*.
    """
    offenders = [p for p in C.CACHE_DIR.rglob("*") if p.is_file() and "holdout" in p.name]
    if C.HOLDOUT_DIR.exists():
        offenders += [p for p in C.HOLDOUT_DIR.rglob("*") if p.is_file()]
    return offenders
