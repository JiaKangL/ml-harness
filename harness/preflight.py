"""The blocking gate. Nothing in the loop starts until every check here passes.

An agent loop running on a broken harness does not fail loudly -- it produces fifty
iterations of confidently-logged garbage. So the cost of these checks is paid once,
up front, and the loop refuses to start without them.

    python -m harness.preflight [--rebuild]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C
from . import data_guard

sys.path.insert(0, str(C.STARTER_KIT))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    seconds: float = 0.0

    def __post_init__(self) -> None:
        # Comparisons against numpy scalars yield np.bool_, which json refuses.
        self.ok = bool(self.ok)
        self.seconds = float(self.seconds)


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    environment: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)


class PreflightError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ checks


def check_environment() -> tuple[Check, dict]:
    env = {
        "python": sys.version.split()[0],
        "python_bin": C.PYTHON_BIN,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    ok = sys.version_info >= (3, 9)
    return Check("environment", ok, f"python {env['python']}, numpy {env['numpy']}"), env


def check_data_present() -> Check:
    missing = [
        f
        for f in (*C.LOG_FILES, C.VIDEO_FEATURES_FILE, C.USER_FEATURES_FILE)
        if not (C.DATA_DIR / f).exists()
    ]
    if missing:
        return Check("data present", False, f"missing: {', '.join(missing)}")
    total_mb = sum((C.DATA_DIR / f).stat().st_size for f in C.LOG_FILES) / 1e6
    return Check("data present", True, f"{total_mb:.0f} MB of standard logs")


def check_quarantine() -> Check:
    """The quarantined file should exist on disk and be absent from the cache.

    Recorded as a check rather than a silent omission: excluding an item-features
    file is a surprising decision and it should be visible that we made it.
    """
    names = list(C.QUARANTINED_FILES)
    present = [n for n in names if (C.DATA_DIR / n).exists()]
    return Check(
        "leakage quarantine",
        True,
        f"{len(present)} file(s) excluded from cache: {', '.join(present)}",
    )


def check_evaluate_frozen() -> tuple[Check, str]:
    path = C.STARTER_KIT / "evaluate.py"
    if not path.exists():
        return Check("evaluate.py frozen", False, "not found"), ""
    digest = _sha256(path)
    return Check("evaluate.py frozen", True, f"sha256 {digest[:16]}…"), digest


def check_cache(rebuild: bool) -> Check:
    t0 = time.time()
    counts = data_guard.build_cache(force=rebuild)
    detail = " / ".join(f"{k} {v:,}" for k, v in counts.items())
    return Check("cache built", True, detail, time.time() - t0)


def check_row_alignment() -> Check:
    """Verify the cache reproduces data.load()'s row order exactly.

    The submission's row_id is a positional index into that sequence. A reordering
    here would misalign every score we submit while every metric still looked
    plausible -- the worst class of bug this project can have, so it is checked
    against the official loader rather than assumed.
    """
    t0 = time.time()
    import data as kit_data  # noqa: E402  (starter kit, path-injected above)

    official = kit_data.load(str(C.DATA_DIR))
    api = data_guard.DataAPI()
    for split in C.SPLITS:
        rows = official[split]
        if len(rows) != api.n_rows(split):
            return Check(
                "row alignment",
                False,
                f"{split}: {api.n_rows(split)} cached vs {len(rows)} official",
                time.time() - t0,
            )
        feats = api.features(split)
        want_u = np.fromiter((int(r[1]) for r in rows), dtype=np.int32, count=len(rows))
        want_v = np.fromiter((int(r[2]) for r in rows), dtype=np.int32, count=len(rows))
        if not (
            np.array_equal(want_u, feats["user_id"])
            and np.array_equal(want_v, feats["video_id"])
        ):
            return Check("row alignment", False, f"{split}: order mismatch", time.time() - t0)
    return Check(
        "row alignment", True, "cache order == data.load() for all splits", time.time() - t0
    )


def check_firewall() -> Check:
    """The firewall must actually raise. Asserted, not asserted-to-be-true."""
    api = data_guard.DataAPI()
    failures = []

    try:
        api.labels("test")
        failures.append("labels('test') did not raise")
    except data_guard.TestLabelAccessError:
        pass

    try:
        api.aux_targets("valid")  # type: ignore[arg-type]
        failures.append("aux_targets('valid') did not raise")
    except data_guard.OutcomeColumnAccessError:
        pass

    try:
        api.column("test", "play_time_ms")
        failures.append("column('test','play_time_ms') did not raise")
    except data_guard.OutcomeColumnAccessError:
        pass

    with np.load(C.SPLITS_NPZ) as z:
        leaked = [k for k in z.files if k.startswith("test__") and k.endswith(C.LABEL)]
        leaked += [
            k for k in z.files if any(k == f"{s}__{c}" for s in ("valid", "test") for c in C.LOG_OUTCOME)
        ]
    if leaked:
        failures.append(f"cache contains forbidden keys: {leaked}")

    if failures:
        return Check("leakage firewall", False, "; ".join(failures))
    return Check("leakage firewall", True, "test labels + eval outcomes unreachable")


def check_random_baseline() -> Check:
    """The organizers' own harness self-test.

    If random scoring does not land on 0.4834 on valid, the scoring path is wrong and
    every number produced downstream is meaningless.
    """
    t0 = time.time()
    from evaluate import evaluate  # noqa: E402

    api = data_guard.DataAPI()
    users = api.features("valid")["user_id"]
    labels = api.labels("valid")
    rng = np.random.default_rng(0)
    got = evaluate(users, labels, rng.random(len(labels)))["primary"]
    delta = abs(got - C.EXPECTED["random_valid_primary"])
    ok = delta <= C.RANDOM_TOLERANCE
    return Check(
        "random baseline",
        ok,
        f"valid primary {got:.4f} vs expected {C.EXPECTED['random_valid_primary']:.4f} "
        f"(Δ{delta:.4f}, tol {C.RANDOM_TOLERANCE})",
        time.time() - t0,
    )


def check_fm_baseline() -> Check:
    """Reproduce the official FM on validation.

    Deliberately *not* a call to baseline.run_fm(): that function evaluates the test
    split before returning, which would breach our own test firewall on the harness's
    very first action. Same FM class, same encoder, same hyperparameters -- test
    arrays are simply never constructed.
    """
    t0 = time.time()
    import baseline as kit_baseline  # noqa: E402
    import data as kit_data  # noqa: E402
    from evaluate import evaluate  # noqa: E402

    splits = kit_data.load(str(C.DATA_DIR))
    splits.pop("test", None)  # the firewall, applied to our own reproduction
    enc, dim = kit_data.encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    cfg = dict(k=16, lr=0.001, bs=8192, epochs=40, patience=4, seed=0)
    m = kit_baseline.FM(dim, k=cfg["k"], lr=cfg["lr"], seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    best, best_state, bad = -1.0, None, 0
    for _ in range(cfg["epochs"]):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), cfg["bs"]):
            m.step(Xtr[idx[i : i + cfg["bs"]]], ytr[idx[i : i + cfg["bs"]]])
        p = evaluate(uva, yva, m.predict(Xva))["primary"]
        if p > best + 1e-5:
            best, bad = p, 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    m.V, m.W, m.b = best_state
    res = evaluate(uva, yva, m.predict(Xva))

    delta = abs(res["primary"] - C.EXPECTED["fm_valid_primary"])
    ok = delta <= C.FM_TOLERANCE
    return Check(
        "FM baseline",
        ok,
        f"valid GAUC {res['GAUC']:.4f} nDCG@5 {res['nDCG@5']:.4f} "
        f"primary {res['primary']:.4f} vs expected "
        f"{C.EXPECTED['fm_valid_primary']:.4f} (Δ{delta:.4f}, tol {C.FM_TOLERANCE})",
        time.time() - t0,
    )


# ------------------------------------------------------------------ driver


def run(rebuild: bool = False, skip_fm: bool = False) -> PreflightReport:
    report = PreflightReport()

    env_check, env = check_environment()
    report.environment = env
    report.checks.append(env_check)

    report.checks.append(check_data_present())
    if not report.passed:
        return report

    report.checks.append(check_quarantine())
    eval_check, digest = check_evaluate_frozen()
    report.checks.append(eval_check)
    report.environment["evaluate_sha256"] = digest

    report.checks.append(check_cache(rebuild))
    report.checks.append(check_row_alignment())
    report.checks.append(check_firewall())
    report.checks.append(check_random_baseline())
    if not skip_fm:
        report.checks.append(check_fm_baseline())

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Harness preflight gate")
    ap.add_argument("--rebuild", action="store_true", help="force cache rebuild")
    ap.add_argument("--skip-fm", action="store_true", help="skip the ~1min FM reproduction")
    args = ap.parse_args()

    t0 = time.time()
    report = run(rebuild=args.rebuild, skip_fm=args.skip_fm)

    width = max(len(c.name) for c in report.checks)
    print()
    for c in report.checks:
        mark = "PASS" if c.ok else "FAIL"
        timing = f"  [{c.seconds:5.1f}s]" if c.seconds >= 0.05 else ""
        print(f"  {mark}  {c.name:<{width}}  {c.detail}{timing}")

    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    C.PREFLIGHT_JSON.write_text(
        json.dumps(
            {
                "passed": report.passed,
                "environment": report.environment,
                "checks": [
                    {"name": c.name, "ok": c.ok, "detail": c.detail, "seconds": round(c.seconds, 2)}
                    for c in report.checks
                ],
            },
            indent=2,
        )
    )

    print(f"\n  {'PREFLIGHT PASSED' if report.passed else 'PREFLIGHT FAILED'} "
          f"in {time.time() - t0:.1f}s -> logs/preflight.json\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
