"""Runs perfectly, exits 0, produces a correctly-shaped float64 array of the right
length -- and expresses no ranking whatsoever. The silent-wrong case."""
import argparse

import numpy as np

from harness.data_guard import DataAPI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()
    n = DataAPI().n_rows(args.split)
    np.save(args.out, np.full(n, 0.5, dtype=np.float64))


if __name__ == "__main__":
    main()
