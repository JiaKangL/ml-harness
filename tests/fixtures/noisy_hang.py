"""Writes far more than one OS pipe buffer to stderr, then hangs forever.

Under `stderr=PIPE` this is the deadlock: the child blocks writing into a full ~64 KB
buffer while the parent blocks waiting for it to exit. Under files it just works, and
the tail survives the kill -- which is the whole reason the executor uses files.
"""
import argparse
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.parse_args()
    for i in range(4000):
        print(f"epoch {i:04d} loss=0.693 grad_norm=1.0 padding----------------", file=sys.stderr)
    print("LAST_LINE_BEFORE_HANG", file=sys.stderr, flush=True)
    time.sleep(600)


if __name__ == "__main__":
    main()
