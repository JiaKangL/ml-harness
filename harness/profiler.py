"""Pre-run data profiling -> logs/data_profile.json.

This file is written once and then ships inside every prompt the agent receives, so
it is kept small deliberately (target <1200 tokens). Everything in it is *measured*,
because the starter kit's own prose is wrong in at least one load-bearing place: it
describes "hundreds to thousands" of interactions per user, and the median is 31.

The centrepiece is `within_user_variance`. Ranking is done inside a user, so a
feature is worth exactly what it varies inside one user's impression group -- a
feature that is constant for a user cancels out of that user's ordering entirely, no
matter how predictive it looks globally. That single fact explains the organizers'
published dead ends, and it is the most useful thing we can hand the agent.

    python -m harness.profiler
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from . import config as C
from .data_guard import DataAPI

PCTL = (10, 25, 50, 75, 90)


def _pct(a: np.ndarray, pctl: Iterable[int] = PCTL) -> dict[str, int]:
    return {f"p{p}": int(np.percentile(a, p)) for p in pctl}


def _group_sizes(groups: np.ndarray) -> np.ndarray:
    return np.bincount(groups)


def _distinct_per_group(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Number of distinct `values` within each group, via a single lexsort."""
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order]
    starts = np.empty(len(g), dtype=bool)
    starts[0] = True
    np.logical_or(g[1:] != g[:-1], v[1:] != v[:-1], out=starts[1:])
    return np.bincount(g[starts], minlength=int(groups.max()) + 1)


def _within_user_variance(api: DataAPI, split: str) -> dict[str, float]:
    """Fraction of multi-impression groups in which each feature actually varies.

    Needs no labels, so it is computable on test without touching the firewall.
    """
    groups = api.groups(split)
    feats = dict(api.features(split))
    feats["author_id"] = api.video_feature("author_id")[feats["video_id"]]
    feats["duration_bucket_10s"] = feats["duration_ms"] // 10_000

    sizes = _group_sizes(groups)
    multi = sizes >= 2
    n_multi = int(multi.sum())

    out: dict[str, float] = {}
    for name in (
        "video_id",
        "author_id",
        "duration_bucket_10s",
        "hourmin",
        "tab",
        "date",
    ):
        counts = _distinct_per_group(groups, feats[name].astype(np.int64))
        out[name] = round(float((counts[multi] > 1).mean()), 3)
    return out


def _composition(labels: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """all-negative / all-positive / discriminative user split.

    Only the discriminative users enter GAUC at all, and the all-negative users
    contribute a permanent 0 to the nDCG average -- which is why the metric ceiling
    is 0.8645 rather than 1.0.
    """
    n = int(groups.max()) + 1
    pos = np.bincount(groups, weights=labels, minlength=n)
    tot = np.bincount(groups, minlength=n).astype(float)
    allneg = float((pos == 0).mean())
    allpos = float((pos == tot).mean())
    return {
        "all_negative_pct": round(allneg * 100, 1),
        "all_positive_pct": round(allpos * 100, 1),
        "discriminative_pct": round((1 - allneg - allpos) * 100, 1),
    }


def _aux_correlations(api: DataAPI) -> dict[str, float]:
    """Pearson r of each auxiliary feedback signal against long_view, on train.

    Tells the agent whether a multi-task head would carry information -- and shows
    exactly why play_time_ms must never be an inference-time feature.
    """
    y = api.labels("train").astype(np.float64)
    out = {}
    for name, arr in api.aux_targets("train").items():
        a = arr.astype(np.float64)
        out[name] = round(float(np.corrcoef(a, y)[0, 1]), 3) if a.std() > 0 else 0.0
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def _item_signal(api: DataAPI, min_impressions: int = 50) -> dict:
    """Spread of per-video long_view rate -- how much of the signal is item quality."""
    vid = api.features("train")["video_id"].astype(np.int64)
    y = api.labels("train")
    n = int(vid.max()) + 1
    imp = np.bincount(vid, minlength=n)
    pos = np.bincount(vid, weights=y, minlength=n)
    keep = imp >= min_impressions
    rate = pos[keep] / imp[keep]
    return {"n_videos": int(keep.sum()), "long_view_rate": _pct(rate * 1000)}


def _cold_rates(api: DataAPI, split: str, train_v: set, train_u: set) -> dict[str, float]:
    f = api.features(split)
    return {
        "unseen_video_pct": round(float(np.isin(f["video_id"], list(train_v), invert=True).mean()) * 100, 2),
        "unseen_user_pct": round(float(np.isin(f["user_id"], list(train_u), invert=True).mean()) * 100, 2),
    }


def build_profile(api: DataAPI | None = None) -> dict:
    api = api or DataAPI()
    prof: dict = {
        "task": "within-user ranking of logged impressions; label long_view (0/1)",
        "metric": "primary = mean(GAUC, nDCG@5); baseline to beat = 0.5946 on hidden test",
        "splits": {},
    }

    train_v = set(np.unique(api.features("train")["video_id"]).tolist())
    train_u = set(np.unique(api.features("train")["user_id"]).tolist())

    for split in ("train", "valid", "test"):
        f = api.features(split)
        g = api.groups(split)
        sizes = _group_sizes(g)
        entry: dict = {
            "rows": api.n_rows(split),
            "users": int(sizes.size),
            "videos": int(np.unique(f["video_id"]).size),
            "impressions_per_user": _pct(sizes),
        }
        if split != "train":
            entry["users_with_at_most_n_impressions_pct"] = {
                f"<={k}": round(float((sizes <= k).mean()) * 100, 1) for k in (1, 3, 5, 10)
            }
            entry.update(_cold_rates(api, split, train_v, train_u))
            uv = f["user_id"].astype(np.int64) * (int(f["video_id"].max()) + 1) + f["video_id"]
            entry["duplicate_user_video_pct"] = round(
                float(1 - np.unique(uv).size / uv.size) * 100, 2
            )
        if split != "test":
            y = api.labels(split)
            entry["positive_rate"] = round(float(y.mean()), 3)
            entry["user_composition"] = _composition(y, g)
        prof["splits"][split] = entry

    prof["within_user_variance"] = {
        "_why": (
            "ranking is within-user, so a feature contributes ONLY through the "
            "fraction of groups in which it varies; a per-user constant cancels out "
            "of that user's ordering exactly. This is why pure user-side features "
            "score 0, and it is arithmetic, not an experiment."
        ),
        "valid": _within_user_variance(api, "valid"),
        "test": _within_user_variance(api, "test"),
    }

    hist = _group_sizes(api.groups("train"))
    prof["train_history_per_user"] = {
        "_why": "feasibility gate for sequence models (DIN/SIM); measured, not quoted",
        **_pct(hist),
        "max": int(hist.max()),
        "mean": round(float(hist.mean()), 1),
    }

    prof["item_level_signal"] = _item_signal(api)
    prof["item_level_signal"]["_units"] = "long_view rate x1000, videos with >=50 train impressions"

    prof["aux_signal_correlation_with_label"] = {
        "_why": (
            "train-only auxiliary targets for multi-task learning. play_time_ms is "
            "near-deterministic of the label (long_view is a threshold on "
            "play_time/duration) -- legitimate as a training target, catastrophic as "
            "an inference feature. The DataAPI makes the latter impossible."
        ),
        **_aux_correlations(api),
    }

    prof["column_legality"] = {
        "safe_any_split": list(C.LOG_SAFE),
        "train_only_outcomes": list(C.LOG_OUTCOME),
        "quarantined_files": C.QUARANTINED_FILES,
        "side_tables": api.side_columns(),
    }
    return prof


def main() -> int:
    api = DataAPI()
    prof = build_profile(api)
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prof, indent=1)
    C.DATA_PROFILE_JSON.write_text(text)
    print(text)
    print(
        f"\n-> {C.DATA_PROFILE_JSON.relative_to(C.ROOT)}  "
        f"({len(text):,} chars, ~{len(text)//4:,} tokens)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
