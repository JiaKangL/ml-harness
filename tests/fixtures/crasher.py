"""Raises from a nested frame, so the traceback has a deepest frame worth parsing."""
import argparse

import numpy as np


def inner(x):
    return np.zeros((3, 4)) @ np.zeros((5, 6))


def outer(x):
    return inner(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.parse_args()
    outer(1)


if __name__ == "__main__":
    main()
