"""Allocates far more than its cap allows and then holds it.

The watchdog is out-of-band and samples at 1 Hz, so the fixture holds the allocation
rather than freeing it immediately -- a spike between two samples is invisible, which
is a documented limit of RSS sampling rather than a bug to hide.

RLIMIT_AS is deliberately not how this is enforced: it is unreliable under the macOS
ARM allocator, which reserves address space far beyond what it commits.
"""
import argparse
import sys
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.parse_args()

    blocks = []
    for i in range(12):
        blocks.append(np.ones(8_000_000, dtype=np.float64))  # 64 MB, touched
        print(f"allocated {(i + 1) * 64} MB", file=sys.stderr, flush=True)
        time.sleep(0.4)
    time.sleep(60)


if __name__ == "__main__":
    main()
