"""Would take five minutes. Smoke must kill it in ~30s, before the full-run timeout
of 900s has a chance to matter."""
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
    print("training epoch 1/200", file=sys.stderr, flush=True)
    time.sleep(300)


if __name__ == "__main__":
    main()
