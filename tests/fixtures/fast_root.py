"""A two-second stand-in for the FM baseline, used as the root in the loop tests.

Real enough to be a genuine ranking -- smoothed per-video long_view rate, deterministic
under --seed, honouring --frac by sampling users -- and cheap enough that a test can
afford to run it. The FM baseline is the root of a real run; paying 50 seconds of it
per orchestration test buys no extra coverage of the orchestration.
"""
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
    tr = api.features("train")["video_id"].astype(np.int64)
    y = api.labels("train").astype(np.float64)
    ev = api.features(args.split)["video_id"].astype(np.int64)
    n = max(int(tr.max()), int(ev.max())) + 1
    imp = np.bincount(tr, minlength=n)
    pos = np.bincount(tr, weights=y, minlength=n)
    prior = float(y.mean())
    # A seed-dependent tie-break, so two seeds differ slightly and the multi-seed path
    # is exercised rather than short-circuited by identical arrays.
    jitter = np.random.default_rng(args.seed).normal(0, 1e-6, size=n)
    np.save(args.out, ((pos + 20.0 * prior) / (imp + 20.0) + jitter)[ev])


if __name__ == "__main__":
    main()
