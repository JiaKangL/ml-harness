"""A candidate that satisfies the script contract exactly. Not a model.

It exists so the executor's tests have something that parses, lints, runs in under a
second on a full split, is deterministic under `--seed`, and honours `--frac` the way
the contract requires: the fraction selects USERS, and scores are still emitted for
every row of the requested split.
"""
import argparse

import numpy as np

from harness.data_guard import DataAPI

MIX_A = np.uint64(0x9E3779B97F4A7C15)
MIX_B = np.uint64(0xBF58476D1CE4E5B9)
MIX_C = np.uint64(0x94D049BB133111EB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    api = DataAPI()
    feats = api.features(args.split)
    users = feats["user_id"].astype(np.uint64)
    videos = feats["video_id"].astype(np.uint64)

    # Deterministic in (user, video, seed) and in nothing else: two runs at the same
    # seed must agree bit for bit, which is what the determinism test asserts.
    seed_term = np.uint64((args.seed * int(MIX_C)) % (1 << 64))
    h = users * MIX_A + videos * MIX_B + seed_term
    h ^= h >> np.uint64(31)
    h = h * MIX_B
    h ^= h >> np.uint64(29)
    scores = (h % np.uint64(1_000_003)).astype(np.float64) / 1_000_003.0

    # --frac samples USERS, never rows. A row sample would shred the impression groups
    # the metric is computed over. Every row still receives a score.
    if args.frac < 1.0:
        unique_users = np.unique(users)
        rng = np.random.default_rng(args.seed)
        k = max(1, int(round(len(unique_users) * args.frac)))
        fitted = set(rng.choice(unique_users, size=k, replace=False).tolist())
        seen = np.fromiter((int(u) in fitted for u in users), dtype=bool, count=len(users))
        scores = np.where(seen, scores, scores * 0.5 + 0.25)

    np.save(args.out, scores.astype(np.float64))


if __name__ == "__main__":
    main()
