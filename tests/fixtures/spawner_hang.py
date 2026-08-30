"""Spawns worker children -- a DataLoader with num_workers, in miniature -- records
their PIDs, then hangs.

`proc.kill()` would reap the parent and leave the workers running. They then compete
for cores with the next iteration and corrupt every timing measurement taken after,
which is why the executor kills the whole process group.
"""
import argparse
import json
import os
import subprocess
import sys
import time

N_WORKERS = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    kids = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        for _ in range(N_WORKERS)
    ]
    with open(args.out + ".pids.json", "w") as fh:
        json.dump({"parent": os.getpid(), "children": [k.pid for k in kids]}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"spawned {N_WORKERS} workers", file=sys.stderr, flush=True)
    time.sleep(600)


if __name__ == "__main__":
    main()
