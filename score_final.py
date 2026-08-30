#!/usr/bin/env python3
"""The one permitted read of the test labels. Nothing in `harness/` imports this.

Rules, enforced rather than intended:

1. Refuses unless the run is sealed -- the loop writes the seal on convergence, naming
   the winning node and the sha256 of the submission being scored.
2. Runs once. A second invocation needs an explicit `--force`, and the override is
   written into `logs/test_draws.json` as a second draw, because a second draw on a
   held-out set makes the estimate optimistically biased.
3. Its output never re-enters the loop. If we improve the harness afterwards, we go
   back to test-blind: no module under `harness/` may import this file, and a test
   asserts it.

    python3 score_final.py
    python3 score_final.py --submission outputs/submission.csv

Writes `logs/final_result.json`: GAUC, nDCG@5, primary, and the absolute delta per
metric against the published FM baseline -- which is the competition's own scoring
formula, `score_dataset = mean over metrics of delta(metric)`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from harness import config as C
from harness import holdout, scoring
from harness.data_guard import DataAPI


def read_submission(path: Path, expected_rows: int) -> np.ndarray:
    """Read the CSV we are actually submitting, not the in-memory array we think we
    submitted. The file is the artifact; anything else scores a different object."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"row_id", "user_id", "video_id", "score"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError(f"{path} has {len(rows)} rows; test has {expected_rows}")
    scores = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        if int(row["row_id"]) != i:
            raise ValueError(f"row_id is not contiguous at line {i + 2}: {row['row_id']}")
        scores[i] = float(row["score"])
    if not np.isfinite(scores).all():
        raise ValueError("submission contains NaN or Inf")
    return scores


def deltas_against_baseline(gauc: float, ndcg5: float) -> dict[str, float]:
    """`delta(m) = score_agent(m) - score_baseline(m)`, per metric.

    Per metric rather than on the primary, because that is the formula the organizers
    published -- `score_dataset = mean over m of delta(m)`. The two coincide
    arithmetically here, and writing it their way is what makes that checkable rather
    than assumed.
    """
    return {
        "GAUC": gauc - C.BASELINE_TEST["GAUC"],
        "nDCG@5": ndcg5 - C.BASELINE_TEST["nDCG@5"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default=str(C.SUBMISSION_CSV))
    ap.add_argument(
        "--force",
        action="store_true",
        help="score test a second time; recorded as a second draw, which biases the estimate",
    )
    args = ap.parse_args(argv)

    seal = holdout.read_seal()
    if seal is None:
        print(
            "refusing to score: the run is not sealed. The seal is written when the "
            "loop converges and records which node won -- without it, a test score "
            "could still inform a decision, which is the one thing the holdout exists "
            "to prevent.",
            file=sys.stderr,
        )
        return 2

    api = DataAPI()
    submission = Path(args.submission)
    scores = read_submission(submission, api.n_rows("test"))

    labels = holdout.extract_test_labels(force=args.force)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"{labels.shape[0]} test labels for {scores.shape[0]} submitted scores"
        )

    metrics = scoring.score(api.features("test")["user_id"], labels, scores)
    base = C.BASELINE_TEST
    deltas = deltas_against_baseline(metrics.gauc, metrics.ndcg5)
    result = {
        "node_id": seal.node_id,
        "submission": str(submission),
        "submission_sha256_at_seal": seal.submission_sha256,
        "valid_primary_at_seal": seal.valid_primary,
        "iterations": seal.iterations,
        "test": {
            "GAUC": round(metrics.gauc, 6),
            "nDCG@5": round(metrics.ndcg5, 6),
            "primary": round(metrics.primary, 6),
            "users": metrics.users,
            "rows": metrics.rows,
        },
        "baseline": dict(base),
        "delta": {k: round(v, 6) for k, v in deltas.items()},
        # The competition's formula: the mean of the per-metric absolute deltas. Not
        # the delta of the primary -- they coincide arithmetically here, and stating
        # the formula the organizers wrote is what makes that checkable.
        "score_dataset": round(sum(deltas.values()) / len(deltas), 6),
        "forced_second_draw": bool(args.force),
    }
    C.FINAL_RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.FINAL_RESULT_JSON.write_text(json.dumps(result, indent=2))

    print(f"node          {seal.node_id}")
    print(f"valid primary {seal.valid_primary:.4f} (at seal)")
    print()
    print(f"{'':14s}{'agent':>10s}{'baseline':>10s}{'delta':>10s}")
    print(f"{'GAUC':14s}{metrics.gauc:>10.4f}{base['GAUC']:>10.4f}{deltas['GAUC']:>+10.4f}")
    print(f"{'nDCG@5':14s}{metrics.ndcg5:>10.4f}{base['nDCG@5']:>10.4f}{deltas['nDCG@5']:>+10.4f}")
    print(f"{'primary':14s}{metrics.primary:>10.4f}{base['primary']:>10.4f}"
          f"{metrics.primary - base['primary']:>+10.4f}")
    print()
    print(f"score_dataset = mean of per-metric deltas = {result['score_dataset']:+.4f}")
    print(f"\n-> {C.FINAL_RESULT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
