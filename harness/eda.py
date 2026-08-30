"""L1 -- the full exploratory analysis, stored before the loop runs.

Two audiences. The write-up (and jk, who is new to recommender systems) reads
`logs/eda_report.md`; the agent reads the compressed prompt-sized view that
`profiler.py` produces. This module is the superset, and it is where anything too
large for a prompt lives.

The requirement is *EDA first, result stored*: proposals must be anchored to measured
properties of the data rather than to score feedback alone. Everything here is
computed from the cache, which physically contains no test labels, so the constraint
"nothing label-derived is reported for test" is structural rather than a promise.

    python -m harness.eda
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from . import config as C
from . import profiler, scoring
from .data_guard import DataAPI

# Buckets in seconds. Chosen to straddle the short-form/long-form boundary rather
# than to be equal-width: the interesting question is whether long_view is really a
# duration threshold in disguise, and the answer lives at the extremes.
DURATION_BUCKETS_S = (0, 10, 20, 30, 60, 120, 300, 10**6)

PCTL = (10, 25, 50, 75, 90)


def _rate_table(keys: np.ndarray, labels: np.ndarray, min_n: int = 200) -> list[dict]:
    """long_view rate per key value, ordered by rate. Sparse keys are dropped: a
    100% rate over 7 impressions is noise wearing the costume of a finding."""
    keys = keys.astype(np.int64)
    n = int(keys.max()) + 1
    imp = np.bincount(keys, minlength=n)
    pos = np.bincount(keys, weights=labels.astype(np.float64), minlength=n)
    rows = [
        {
            "value": int(v),
            "impressions": int(imp[v]),
            "long_view_rate": round(float(pos[v] / imp[v]), 4),
        }
        for v in np.flatnonzero(imp >= min_n)
    ]
    return sorted(rows, key=lambda r: r["long_view_rate"])


def _duration_table(api: DataAPI, split: str) -> list[dict]:
    """long_view rate by video-length bucket.

    Measured much flatter than intuition suggests, which is the point: `duration_ms`
    is a legal impression-time feature and varies inside 98% of user groups, so it
    *could* carry the ranking -- and it does not.
    """
    dur_s = api.features(split)["duration_ms"].astype(np.float64) / 1000.0
    y = api.labels(split).astype(np.float64)
    out = []
    for lo, hi in zip(DURATION_BUCKETS_S, DURATION_BUCKETS_S[1:]):
        m = (dur_s >= lo) & (dur_s < hi)
        n = int(m.sum())
        if n < 200:
            continue
        out.append(
            {
                "bucket_s": f"{lo}-{hi}" if hi < 10**6 else f"{lo}+",
                "impressions": n,
                "share_pct": round(100.0 * n / dur_s.size, 1),
                "long_view_rate": round(float(y[m].mean()), 4),
            }
        )
    return out


def _per_video_rates(api: DataAPI, min_impressions: int = 50) -> dict:
    vid = api.features("train")["video_id"].astype(np.int64)
    y = api.labels("train").astype(np.float64)
    n = int(vid.max()) + 1
    imp = np.bincount(vid, minlength=n)
    pos = np.bincount(vid, weights=y, minlength=n)
    keep = imp >= min_impressions
    rate = pos[keep] / imp[keep]
    return {
        "min_impressions": min_impressions,
        "n_videos": int(keep.sum()),
        "coverage_of_train_rows_pct": round(100.0 * float(imp[keep].sum()) / vid.size, 1),
        "percentiles": {f"p{p}": round(float(np.percentile(rate, p)), 3) for p in PCTL},
        "spread_p90_minus_p10": round(
            float(np.percentile(rate, 90) - np.percentile(rate, 10)), 3
        ),
    }


def _popularity_only_gauc(api: DataAPI, min_impressions: int = 5) -> dict:
    """Score valid using nothing but each video's train long_view rate.

    This is the headroom measurement. FM's GAUC minus this number is the entire value
    of personalisation on this dataset -- everything the agent can win by modelling
    the user rather than the item. Smoothed toward the global rate so a video seen
    three times in train does not get a 0.0 or 1.0 prior.
    """
    tr_vid = api.features("train")["video_id"].astype(np.int64)
    tr_y = api.labels("train").astype(np.float64)
    n_vid = max(int(tr_vid.max()), int(api.features("valid")["video_id"].max())) + 1
    imp = np.bincount(tr_vid, minlength=n_vid)
    pos = np.bincount(tr_vid, weights=tr_y, minlength=n_vid)
    prior = float(tr_y.mean())
    smoothed = (pos + min_impressions * prior) / (imp + min_impressions)

    va = api.features("valid")
    scores = smoothed[va["video_id"].astype(np.int64)]
    m = scoring.score(va["user_id"], api.labels("valid"), scores)
    fm = C.EXPECTED
    return {
        "_what": "valid scored by train item popularity alone -- no user model at all",
        "smoothing_pseudocounts": min_impressions,
        "gauc": round(m.gauc, 4),
        "ndcg5": round(m.ndcg5, 4),
        "primary": round(m.primary, 4),
        "fm_baseline_gauc": fm["fm_valid_gauc"],
        "personalisation_worth_gauc": round(fm["fm_valid_gauc"] - m.gauc, 4),
        "personalisation_worth_primary": round(fm["fm_valid_primary"] - m.primary, 4),
    }


def _label_mechanics(api: DataAPI) -> dict:
    """The visual proof of why `play_time_ms` is firewalled off the eval splits.

    `long_view` is a threshold on watch time and `play_time_ms` sits in the same row.
    Train-only by construction: `column("play_time_ms", "valid")` raises.
    """
    play = api.column("train", "play_time_ms").astype(np.float64)
    dur = api.features("train")["duration_ms"].astype(np.float64)
    y = api.labels("train").astype(bool)
    ratio = np.divide(play, dur, out=np.zeros_like(play), where=dur > 0)
    ok = dur > 0

    def pct(mask: np.ndarray) -> dict[str, float]:
        a = ratio[mask & ok]
        return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in PCTL}

    sep = float(((ratio > 0.5) == y)[ok].mean())
    return {
        "_what": "play_time_ms / duration_ms, train only",
        "positives": pct(y),
        "negatives": pct(~y),
        "threshold_rule_agreement": round(sep, 3),
        "_why": (
            "a single hand-written threshold on play/duration reproduces the label "
            f"{sep:.1%} of the time. That is not a feature, it is the answer, which "
            "is why outcome columns are materialised for train only and are absent "
            "from -- not masked within -- the evaluation arrays."
        ),
    }


def _group_size_table(api: DataAPI, split: str) -> list[dict]:
    """Cumulative share of users by impression count.

    13.6% of test users have a single impression and are unrankable at any skill
    level; 54.6% have <=5, so the "@5" cutoff never binds for them. Both facts cap how
    much any modelling change can move nDCG@5, and neither is visible from row counts.
    Feature-side only, so it is computable on test.
    """
    sizes = np.bincount(api.groups(split))
    total = sizes.size
    out = []
    for k in (1, 2, 3, 5, 10, 20, 50):
        n = int((sizes <= k).sum())
        out.append(
            {
                "at_most": k,
                "users": n,
                "cumulative_pct": round(100.0 * n / total, 1),
                "rows_pct": round(100.0 * float(sizes[sizes <= k].sum()) / int(sizes.sum()), 1),
            }
        )
    return out


def _temporal_drift(api: DataAPI) -> dict:
    """Label rate by date across train->valid, plus item churn per window.

    Test contributes feature-side statistics only: the churn figures use video ids,
    which are impression-time facts, and no label-derived number is reported for it.
    """
    by_date = []
    for split in ("train", "valid"):
        dates = api.features(split)["date"].astype(np.int64)
        y = api.labels(split).astype(np.float64)
        for d in np.unique(dates):
            m = dates == d
            by_date.append(
                {
                    "date": int(d),
                    "split": split,
                    "rows": int(m.sum()),
                    "long_view_rate": round(float(y[m].mean()), 4),
                }
            )
    rates = np.array([r["long_view_rate"] for r in by_date])

    vids = {s: set(np.unique(api.features(s)["video_id"]).tolist()) for s in ("train", "valid", "test")}
    # Measured against *every* earlier window, not just the immediately preceding
    # one. A video shown in train and skipped in valid is not a new item in test, and
    # counting it as one inflates the churn figure roughly a hundredfold -- which
    # would then argue for cold-start work that the data does not support.
    churn = {}
    for cur, earlier in (("valid", ("train",)), ("test", ("train", "valid"))):
        seen = set().union(*(vids[e] for e in earlier))
        new = vids[cur] - seen
        churn[cur] = {
            "videos_in_window": len(vids[cur]),
            "unseen_in_any_earlier_window": len(new),
            "unseen_pct_of_catalogue": round(100.0 * len(new) / len(vids[cur]), 2),
        }
    return {
        "by_date": by_date,
        "label_rate_range": [round(float(rates.min()), 4), round(float(rates.max()), 4)],
        "item_churn": churn,
        "_why": (
            "the label rate is stable across the train->valid boundary and the "
            "catalogue barely turns over, so temporal recency weighting and cold-start "
            "handling both have very little to work with. Measured, because the "
            "opposite assumption is the natural one."
        ),
    }


def build_eda(api: DataAPI | None = None) -> dict:
    api = api or DataAPI()
    return {
        "profile": profiler.build_profile(api),
        "long_view_by_tab": {
            "train": _rate_table(api.features("train")["tab"], api.labels("train")),
            "valid": _rate_table(api.features("valid")["tab"], api.labels("valid")),
        },
        "long_view_by_duration": {
            "train": _duration_table(api, "train"),
            "valid": _duration_table(api, "valid"),
        },
        "per_video_long_view_rate": _per_video_rates(api),
        "popularity_only_ceiling": _popularity_only_gauc(api),
        "label_mechanics": _label_mechanics(api),
        "group_sizes": {s: _group_size_table(api, s) for s in ("valid", "test")},
        "temporal_drift": _temporal_drift(api),
    }


# ---------------------------------------------------------------- markdown


def _table(headers: Iterable[str], rows: Iterable[Iterable]) -> str:
    headers = list(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(eda: dict) -> str:
    p = eda["profile"]
    wuv = p["within_user_variance"]
    pop = eda["popularity_only_ceiling"]
    lab = eda["label_mechanics"]
    L: list[str] = []
    add = L.append

    add("# EDA — KuaiRand-Pure, within-user ranking of `long_view`")
    add("")
    add(
        "Computed once, before the agent loop runs, from the cache built by "
        "`harness/data_guard.py`. `logs/data_profile.json` is the prompt-sized "
        "compression of this report and every figure the two share is identical. "
        "**No label-derived statistic is reported for the test split** — the cache "
        "contains no test labels, so this is structural, not a promise."
    )
    add("")
    add("## 1. Shape of the problem")
    add("")
    add(
        _table(
            ["split", "rows", "users", "videos", "median impr/user", "positive rate"],
            [
                [
                    s,
                    f"{d['rows']:,}",
                    f"{d['users']:,}",
                    f"{d['videos']:,}",
                    d["impressions_per_user"]["p50"],
                    d.get("positive_rate", "— (held out)"),
                ]
                for s, d in p["splits"].items()
            ],
        )
    )
    add("")
    add(
        f"The metric is computed **inside** each user's impression group. "
        f"{p['splits']['valid']['user_composition']['all_negative_pct']}% of valid users "
        f"have no positive at all and "
        f"{p['splits']['valid']['user_composition']['all_positive_pct']}% have nothing "
        "but positives; neither can be ranked better or worse. Only the remaining "
        f"{p['splits']['valid']['user_composition']['discriminative_pct']}% respond to "
        "anything the model does."
    )
    add("")

    add("## 2. The within-user variance lens")
    add("")
    add(
        "A feature is worth exactly what it varies **inside one user's group**. A "
        "per-user constant cancels out of that user's ordering exactly, no matter how "
        "predictive it looks globally. This is arithmetic, not an experiment, and it "
        "is the mechanism behind the organizers' published finding that user-side "
        "features give zero."
    )
    add("")
    add(
        _table(
            ["feature", "varies in % of valid groups", "% of test groups"],
            [
                [k, f"{wuv['valid'][k]:.1%}", f"{wuv['test'][k]:.1%}"]
                for k in wuv["valid"]
            ],
        )
    )
    add("")

    add("## 3. `tab` — 100× globally, constant for half of users")
    add("")
    add("The single most instructive table in the report. Valid split:")
    add("")
    add(
        _table(
            ["tab", "impressions", "long_view rate"],
            [[r["value"], f"{r['impressions']:,}", f"{r['long_view_rate']:.1%}"]
             for r in eda["long_view_by_tab"]["valid"]],
        )
    )
    add("")
    add(
        f"The spread across tabs is enormous, and `tab` still varies inside only "
        f"{wuv['valid']['tab']:.1%} of valid groups — so for the other half of users "
        "it contributes nothing to their ordering. A feature's global predictiveness "
        "and its ranking value are different quantities, and this table is the proof."
    )
    add("")

    add("## 4. Duration is a weak feature")
    add("")
    add(
        _table(
            ["video length (s)", "share of impressions", "long_view rate"],
            [[r["bucket_s"], f"{r['share_pct']}%", f"{r['long_view_rate']:.1%}"]
             for r in eda["long_view_by_duration"]["valid"]],
        )
    )
    add("")
    add(
        "Much flatter than intuition suggests. `duration_ms` is legal at inference "
        f"time and varies inside {wuv['valid']['duration_bucket_10s']:.1%} of groups, so "
        "it *could* have carried the ranking — it simply does not carry much."
    )
    add("")

    pv = eda["per_video_long_view_rate"]
    add("## 5. Item quality is the dominant signal")
    add("")
    add(
        f"Per-video long_view rate over the {pv['n_videos']:,} videos with "
        f"≥{pv['min_impressions']} train impressions "
        f"({pv['coverage_of_train_rows_pct']}% of train rows):"
    )
    add("")
    add(_table(["percentile", "rate"], [[k, f"{v:.1%}"] for k, v in pv["percentiles"].items()]))
    add("")
    add(
        f"p10 → p90 spans {pv['spread_p90_minus_p10']:.1%} of absolute rate. Ranking by "
        "this number alone — no user model whatsoever — scores:"
    )
    add("")
    add(
        _table(
            ["scorer", "GAUC", "nDCG@5", "primary"],
            [
                ["item popularity only", pop["gauc"], pop["ndcg5"], pop["primary"]],
                ["FM baseline (valid)", C.EXPECTED["fm_valid_gauc"],
                 C.EXPECTED["fm_valid_ndcg5"], C.EXPECTED["fm_valid_primary"]],
            ],
        )
    )
    add("")
    add(
        f"**All of FM's personalisation is worth +{pop['personalisation_worth_gauc']:.4f} "
        f"GAUC** over knowing nothing but which video it is. That sets the realistic "
        "size of the headroom the agent is competing for, and it is why the promotion "
        "bar is +0.002 rather than something that sounds more impressive."
    )
    add("")

    add("## 6. Label mechanics — why `play_time_ms` is firewalled")
    add("")
    add("`play_time_ms / duration_ms`, train only:")
    add("")
    add(
        _table(
            ["percentile", "positives", "negatives"],
            [[k, lab["positives"][k], lab["negatives"][k]] for k in lab["positives"]],
        )
    )
    add("")
    add(lab["_why"])
    add("")
    add(
        "Post-impression outcome columns are therefore materialised for `train` only. "
        "They are absent from the evaluation arrays rather than masked within them: "
        "there is no column to accidentally select. "
        f"`{list(C.QUARANTINED_FILES)[0]}` is excluded entirely — "
        f"{list(C.QUARANTINED_FILES.values())[0]}."
    )
    add("")

    add("## 7. Group sizes cap what `@5` can measure")
    add("")
    for split in ("valid", "test"):
        add(f"**{split}**")
        add("")
        add(
            _table(
                ["users with ≤ n impressions", "users", "cumulative %", "% of rows"],
                [[r["at_most"], f"{r['users']:,}", f"{r['cumulative_pct']}%", f"{r['rows_pct']}%"]
                 for r in eda["group_sizes"][split]],
            )
        )
        add("")
    add(
        "A user with one impression is unrankable; a user with ≤5 has no cutoff at "
        "@5. These are feature-side counts and need no labels, which is how they can "
        "be reported for test at all."
    )
    add("")

    td = eda["temporal_drift"]
    add("## 8. Temporal drift and item churn")
    add("")
    add(
        f"Daily long_view rate across train and valid stays within "
        f"[{td['label_rate_range'][0]:.1%}, {td['label_rate_range'][1]:.1%}]."
    )
    add("")
    add(
        _table(
            ["window", "videos", "unseen in any earlier window", "% of catalogue"],
            [[k, f"{v['videos_in_window']:,}", f"{v['unseen_in_any_earlier_window']:,}",
              f"{v['unseen_pct_of_catalogue']}%"] for k, v in td["item_churn"].items()],
        )
    )
    add("")
    add(td["_why"])
    add("")

    add("## 9. What this implies for the agent")
    add("")
    add(
        "- The signal is mostly item quality; personalisation is a small, hard-won "
        f"+{pop['personalisation_worth_gauc']:.4f} GAUC on top of it.\n"
        "- Features that do not vary within a user's group cannot help, whatever "
        "their global correlation.\n"
        "- Half the users are unrankable or already perfectly ranked, so realised "
        "deltas are diluted by construction and a +0.002 improvement is a real one.\n"
        "- The auxiliary outcome columns are the richest untouched resource, and they "
        "are usable **only** as training targets."
    )
    add("")
    return "\n".join(L)


def main() -> int:
    api = DataAPI()
    eda = build_eda(api)
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    C.EDA_JSON.write_text(json.dumps(eda, indent=1))
    md = render_markdown(eda)
    C.EDA_REPORT_MD.write_text(md)
    print(md)
    print(
        f"\n-> {C.EDA_REPORT_MD.relative_to(C.ROOT)} ({len(md):,} chars)"
        f"\n-> {C.EDA_JSON.relative_to(C.ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
