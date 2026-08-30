# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026, Track 2. An agent that runs the MLE loop on its own: reads the
problem, writes code, trains, evaluates, reads its own results, decides what to try
next, and stops when it stops improving.

Target: beat the organizers' Factorization Machine baseline (primary **0.5946** on
the hidden test split) with zero manual interventions.

See [PLAN.md](PLAN.md) for the architecture and the reasoning behind it.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./scripts/get_data.sh                     # 47 MB from Zenodo
./.venv/bin/python -m harness.preflight   # blocking gate, ~30s
./.venv/bin/python -m harness.profiler    # writes logs/data_profile.json
./.venv/bin/python -m unittest discover -s tests -t .   # 23 acceptance tests
```

`preflight` refuses to let the loop start until it has reproduced the organizers'
own self-checks: random scoring at valid primary 0.4834 and the official FM at
0.6016. An agent loop running on a broken harness does not fail loudly — it produces
fifty iterations of confidently-logged garbage.

## Two firewalls

**Leakage.** `long_view` is a threshold on watch time and `play_time_ms` sits in the
same log row (measured median play/duration ratio: 0.98 for positives, 0.03 for
negatives). Post-impression outcome columns are materialised for `train` only —
absent from evaluation-split arrays, not masked within them.

**Test.** Test labels never enter the agent's process; `labels("test")` raises. They
are written once to `cache/_holdout/`, which nothing in the agent's prompt or import
path names, and only `score_final.py` reads them — once, after convergence.

`video_features_statistic_pure.csv` is quarantined: its `long_time_play_cnt` /
`complete_play_cnt` / `play_progress` columns are dataset-wide aggregates spanning
the test window, so joining them leaks both the label and the future while looking
like an ordinary item-features file.

## Status

| Phase | State |
|---|---|
| 0 — preflight gate | done |
| 1 — data guard + profiler + tests | done |
| 2 — evaluator (noise gate) | next |
| 3 — executor | |
| 4 — memory + logger | |
| 5 — agent + loop | |
| 6 — critics + ensembling | |
| 7 — run + write-up | |
