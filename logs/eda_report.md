# EDA — KuaiRand-Pure, within-user ranking of `long_view`

Computed once, before the agent loop runs, from the cache built by `harness/data_guard.py`. `logs/data_profile.json` is the prompt-sized compression of this report and every figure the two share is identical. **No label-derived statistic is reported for the test split** — the cache contains no test labels, so this is structural, not a promise.

## 1. Shape of the problem

| split | rows | users | videos | median impr/user | positive rate |
|---|---|---|---|---|---|
| train | 1,141,112 | 26,210 | 7,538 | 31 | 0.337 |
| valid | 124,909 | 22,377 | 5,951 | 4 | 0.313 |
| test | 170,588 | 23,875 | 5,982 | 5 | — (held out) |

The metric is computed **inside** each user's impression group. 30.3% of valid users have no positive at all and 11.9% have nothing but positives; neither can be ranked better or worse. Only the remaining 57.8% respond to anything the model does.

## 2. The within-user variance lens

A feature is worth exactly what it varies **inside one user's group**. A per-user constant cancels out of that user's ordering exactly, no matter how predictive it looks globally. This is arithmetic, not an experiment, and it is the mechanism behind the organizers' published finding that user-side features give zero.

| feature | varies in % of valid groups | % of test groups |
|---|---|---|
| video_id | 99.1% | 99.2% |
| author_id | 99.1% | 99.1% |
| duration_bucket_10s | 98.0% | 98.1% |
| hourmin | 95.9% | 97.2% |
| tab | 48.3% | 51.8% |
| date | 90.9% | 95.4% |

## 3. `tab` — 100× globally, constant for half of users

The single most instructive table in the report. Valid split:

| tab | impressions | long_view rate |
|---|---|---|
| 8 | 547 | 2.2% |
| 0 | 13,726 | 3.5% |
| 6 | 5,170 | 9.6% |
| 5 | 291 | 10.3% |
| 12 | 226 | 13.3% |
| 1 | 92,672 | 35.6% |
| 2 | 3,834 | 35.7% |
| 4 | 7,877 | 45.9% |

The spread across tabs is enormous, and `tab` still varies inside only 48.3% of valid groups — so for the other half of users it contributes nothing to their ordering. A feature's global predictiveness and its ranking value are different quantities, and this table is the proof.

## 4. Duration is a weak feature

| video length (s) | share of impressions | long_view rate |
|---|---|---|
| 0-10 | 5.8% | 25.9% |
| 10-20 | 14.6% | 27.5% |
| 20-30 | 8.2% | 31.9% |
| 30-60 | 15.9% | 35.0% |
| 60-120 | 23.8% | 34.8% |
| 120-300 | 26.6% | 30.0% |
| 300+ | 5.2% | 26.3% |

Much flatter than intuition suggests. `duration_ms` is legal at inference time and varies inside 98.0% of groups, so it *could* have carried the ranking — it simply does not carry much.

## 5. Item quality is the dominant signal

Per-video long_view rate over the 3,552 videos with ≥50 train impressions (93.1% of train rows):

| percentile | rate |
|---|---|
| p10 | 8.1% |
| p25 | 17.1% |
| p50 | 29.8% |
| p75 | 42.5% |
| p90 | 52.3% |

p10 → p90 spans 44.2% of absolute rate. Ranking by this number alone — no user model whatsoever — scores:

| scorer | GAUC | nDCG@5 | primary |
|---|---|---|---|
| item popularity only | 0.6395 | 0.5231 | 0.5813 |
| FM baseline (valid) | 0.6674 | 0.5357 | 0.6016 |

**All of FM's personalisation is worth +0.0279 GAUC** over knowing nothing but which video it is. That sets the realistic size of the headroom the agent is competing for, and it is why the promotion bar is +0.002 rather than something that sounds more impressive.

## 6. Label mechanics — why `play_time_ms` is firewalled

`play_time_ms / duration_ms`, train only:

| percentile | positives | negatives |
|---|---|---|
| p10 | 0.233 | 0.0 |
| p25 | 0.493 | 0.006 |
| p50 | 0.977 | 0.031 |
| p75 | 1.057 | 0.103 |
| p90 | 1.301 | 0.272 |

a single hand-written threshold on play/duration reproduces the label 88.2% of the time. That is not a feature, it is the answer, which is why outcome columns are materialised for train only and are absent from -- not masked within -- the evaluation arrays.

Post-impression outcome columns are therefore materialised for `train` only. They are absent from the evaluation arrays rather than masked within them: there is no column to accidentally select. `video_features_statistic_pure.csv` is excluded entirely — dataset-wide aggregates of play/completion behaviour spanning the test window; joining them leaks both the label and the future.

## 7. Group sizes cap what `@5` can measure

**valid**

| users with ≤ n impressions | users | cumulative % | % of rows |
|---|---|---|---|
| 1 | 3,917 | 17.5% | 3.1% |
| 2 | 7,248 | 32.4% | 8.5% |
| 3 | 10,135 | 45.3% | 15.4% |
| 5 | 14,254 | 63.7% | 30.1% |
| 10 | 19,479 | 87.0% | 61.8% |
| 20 | 21,825 | 97.5% | 87.9% |
| 50 | 22,362 | 99.9% | 99.3% |

**test**

| users with ≤ n impressions | users | cumulative % | % of rows |
|---|---|---|---|
| 1 | 3,247 | 13.6% | 1.9% |
| 2 | 6,212 | 26.0% | 5.4% |
| 3 | 8,792 | 36.8% | 9.9% |
| 5 | 13,026 | 54.6% | 21.0% |
| 10 | 18,951 | 79.4% | 47.6% |
| 20 | 22,657 | 94.9% | 78.7% |
| 50 | 23,807 | 99.7% | 97.4% |

A user with one impression is unrankable; a user with ≤5 has no cutoff at @5. These are feature-side counts and need no labels, which is how they can be reported for test at all.

## 8. Temporal drift and item churn

Daily long_view rate across train and valid stays within [29.0%, 37.7%].

| window | videos | unseen in any earlier window | % of catalogue |
|---|---|---|---|
| valid | 5,951 | 7 | 0.12% |
| test | 5,982 | 6 | 0.1% |

the label rate is stable across the train->valid boundary and the catalogue barely turns over, so temporal recency weighting and cold-start handling both have very little to work with. Measured, because the opposite assumption is the natural one.

## 9. What this implies for the agent

- The signal is mostly item quality; personalisation is a small, hard-won +0.0279 GAUC on top of it.
- Features that do not vary within a user's group cannot help, whatever their global correlation.
- Half the users are unrankable or already perfectly ranked, so realised deltas are diluted by construction and a +0.002 improvement is a real one.
- The auxiliary outcome columns are the richest untouched resource, and they are usable **only** as training targets.
