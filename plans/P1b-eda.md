# P1b — `harness/eda.py` (L1, Ground truth)

**Purpose.** The full exploratory analysis, stored before the loop runs. Two audiences:
the write-up (and jk, who is new to recommender systems), and the agent indirectly —
`data_profile.json` is the compressed prompt-sized view of what this produces.

**Depends on:** L1 only. **Produces:** `logs/eda_report.md` + `logs/eda.json`.

## Why it exists

The requirement is *EDA first, result stored* — so that proposals are anchored to
measured properties of the data rather than to score feedback alone. `profiler.py`
already does the prompt-sized half (1,134 tokens). This is the human-readable
superset, and it is where anything too large for the prompt lives.

## What it must report

Everything in `data_profile.json`, plus:

- **long_view rate by video duration bucket** — measured 0.27→0.37, much flatter than
  intuition suggests; documents that duration is a weak feature.
- **long_view rate by `tab`** — 0.4% (tab 3) to 48.9% (tab 4), a 100x spread. Paired
  with the within-user variance figure (constant for 48% of users) this is the single
  most instructive table in the report.
- **Per-video long_view rate distribution** — p10 0.081 → p90 0.523 for videos with
  ≥50 train impressions. Shows item quality is the dominant signal.
- **Item-popularity-only within-group AUC** ≈ 0.6298 vs FM's GAUC 0.6610, i.e. all of
  FM's personalisation is worth **+0.031**. Sets realistic expectations for headroom.
- **Label mechanics** — median play/duration 0.98 for positives, 0.03 for negatives.
  The visual proof of why `play_time_ms` is firewalled.
- **Group-size cumulative table** — 13.6% of test users have 1 impression (unrankable),
  54.6% have ≤5 (the "@5" cutoff never binds for them).
- **Temporal drift** — label rate by date across train→valid; item churn per window.

## Constraints

- **Test labels are off-limits.** Anything label-derived is train/valid only. Test may
  contribute feature-side statistics (group sizes, cold rates) and nothing else.
- Reuse `profiler.build_profile()` rather than recomputing shared statistics.
- Markdown output, no plotting dependency — tables only. numpy is the sole dependency.

## Acceptance tests — `tests/test_eda.py`

| Test | Passes when |
|---|---|
| Report generated | `logs/eda_report.md` exists and is non-trivial (> 2 KB) |
| No test labels | No label-derived statistic is reported for the test split |
| Consistency | Figures shared with `data_profile.json` match it exactly |
| Known values | Duplicate-pair rate on test reproduces the published 3.06% |
