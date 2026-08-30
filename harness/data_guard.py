"""The leakage firewall, and the only data path the agent is given.

Two rules are enforced structurally rather than by instruction:

1. **Outcome columns never reach an evaluation split.** `is_click`, `play_time_ms`
   and friends describe what the user did *after* the video was shown. They are
   legitimate auxiliary training targets and catastrophic inference-time features,
   so they are materialised for `train` only -- absent from the arrays returned for
   `valid`/`test`, not masked within them.

2. **Test labels are not in the cache the agent reads.** They are written once to
   `cache/_holdout/`, which nothing in the agent's prompt or import path mentions.
   `labels("test")` raises.

This is defence in depth, not a sandbox: a generated script running on this machine
could in principle re-parse the raw CSVs. The raw directory is never named to the
agent, the executor lints for references to it, and the cache it *is* pointed at
physically lacks the labels. That combination is what makes the guarantee real
enough to state honestly in the write-up.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from . import config as C

Split = Literal["train", "valid", "test"]

# int64 for the epoch-millisecond timestamp; everything else fits comfortably in 32.
_LOG_DTYPES: dict[str, type] = {
    "user_id": np.int32,
    "video_id": np.int32,
    "date": np.int32,
    "hourmin": np.int32,
    "time_ms": np.int64,
    "tab": np.int32,
    "is_rand": np.int8,
    "duration_ms": np.int32,
    "long_view": np.int8,
    "is_click": np.int8,
    "is_like": np.int8,
    "is_follow": np.int8,
    "is_comment": np.int8,
    "is_forward": np.int8,
    "is_hate": np.int8,
    "is_profile_enter": np.int8,
    "play_time_ms": np.int32,
    "profile_stay_time": np.int32,
    "comment_stay_time": np.int32,
}


class TestLabelAccessError(RuntimeError):
    """Raised when anything tries to read a test label through the agent's data path."""


class OutcomeColumnAccessError(RuntimeError):
    """Raised when anything asks for post-impression columns on an evaluation split."""


# ------------------------------------------------------------------ cache build


def _split_of(date: int) -> str | None:
    for name, (lo, hi) in C.SPLITS.items():
        if lo <= date <= hi:
            return name
    return None


def _iter_log_rows(data_dir: Path) -> Iterator[dict[str, str]]:
    """Yield log rows in exactly data.load()'s order.

    Order is load-bearing: the submission's `row_id` is a positional index into this
    sequence, so any reordering here silently misaligns every score we submit.
    """
    for fname in C.LOG_FILES:
        with open(data_dir / fname, newline="") as fh:
            yield from csv.DictReader(fh)


def _to_int(value: str, default: int = -1) -> int:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _build_side_tables(data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Video and user side features as arrays indexed by id, plus string vocabularies.

    Missing ids are filled with -1 rather than dropped, so a lookup on an unseen id
    is a defined value instead of an exception inside somebody's training loop.
    """
    arrays: dict[str, np.ndarray] = {}
    vocabs: dict[str, list[str]] = {}

    def encode_column(rows: list[dict[str, str]], col: str) -> tuple[np.ndarray, list[str]]:
        vocab: dict[str, int] = {}
        out = np.empty(len(rows), dtype=np.int32)
        for i, r in enumerate(rows):
            v = r.get(col, "")
            if v not in vocab:
                vocab[v] = len(vocab)
            out[i] = vocab[v]
        return out, list(vocab)

    # --- videos
    with open(data_dir / C.VIDEO_FEATURES_FILE, newline="") as fh:
        vrows = list(csv.DictReader(fh))
    vids = np.array([_to_int(r["video_id"]) for r in vrows], dtype=np.int32)
    n_video = int(vids.max()) + 1
    for col in C.VIDEO_NUMERIC_COLS:
        table = np.full(n_video, -1, dtype=np.int64)
        table[vids] = [_to_int(r[col]) for r in vrows]
        arrays[f"video__{col}"] = table
    for col in C.VIDEO_STRING_COLS:
        codes, vocab = encode_column(vrows, col)
        table = np.full(n_video, -1, dtype=np.int32)
        table[vids] = codes
        arrays[f"video__{col}"] = table
        vocabs[f"video__{col}"] = vocab

    # --- users
    with open(data_dir / C.USER_FEATURES_FILE, newline="") as fh:
        urows = list(csv.DictReader(fh))
    uids = np.array([_to_int(r["user_id"]) for r in urows], dtype=np.int32)
    n_user = int(uids.max()) + 1
    numeric_user_cols = [
        c
        for c in urows[0].keys()
        if c not in C.USER_STRING_COLS and c != "user_id"
    ]
    for col in numeric_user_cols:
        table = np.full(n_user, -1, dtype=np.int64)
        table[uids] = [_to_int(r[col]) for r in urows]
        arrays[f"user__{col}"] = table
    for col in C.USER_STRING_COLS:
        codes, vocab = encode_column(urows, col)
        table = np.full(n_user, -1, dtype=np.int32)
        table[uids] = codes
        arrays[f"user__{col}"] = table
        vocabs[f"user__{col}"] = vocab

    return arrays, vocabs


def build_cache(data_dir: Path | None = None, force: bool = False) -> dict[str, int]:
    """Parse the raw CSVs once into the arrays the agent will actually read.

    Returns per-split row counts. Test labels are diverted to the holdout directory
    and never enter `splits.npz`.
    """
    data_dir = data_dir or C.DATA_DIR
    if C.SPLITS_NPZ.exists() and not force:
        with np.load(C.SPLITS_NPZ) as z:
            return {s: int(z[f"{s}__user_id"].shape[0]) for s in C.SPLITS}

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    C.HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    wanted = list(C.LOG_SAFE) + [C.LABEL] + list(C.LOG_OUTCOME)
    buckets: dict[str, dict[str, list[int]]] = {
        s: {c: [] for c in wanted} for s in C.SPLITS
    }

    for row in _iter_log_rows(data_dir):
        split = _split_of(int(row["date"]))
        if split is None:
            continue
        b = buckets[split]
        for col in wanted:
            b[col].append(_to_int(row[col]))

    out: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for split, cols in buckets.items():
        counts[split] = len(cols["user_id"])
        for col in C.LOG_SAFE:
            out[f"{split}__{col}"] = np.asarray(cols[col], dtype=_LOG_DTYPES[col])

        if split == "test":
            # The one place test labels are written, outside the agent's cache.
            np.save(C.TEST_LABELS_NPY, np.asarray(cols[C.LABEL], dtype=np.int8))
            continue

        out[f"{split}__{C.LABEL}"] = np.asarray(cols[C.LABEL], dtype=np.int8)
        if split == "train":
            # Outcome columns exist for train only -- see module docstring rule 1.
            for col in C.LOG_OUTCOME:
                out[f"{split}__{col}"] = np.asarray(cols[col], dtype=_LOG_DTYPES[col])

    np.savez_compressed(C.SPLITS_NPZ, **out)

    side, vocabs = _build_side_tables(data_dir)
    np.savez_compressed(C.SIDE_NPZ, **side)
    C.VOCAB_JSON.write_text(json.dumps(vocabs, indent=1))

    return counts


# ------------------------------------------------------------------ the API


class DataAPI:
    """The only data surface the agent's generated code is given.

    Every method that could return a test label either refuses or does not exist.
    """

    def __init__(self, splits_npz: Path | None = None, side_npz: Path | None = None):
        self._splits = dict(np.load(splits_npz or C.SPLITS_NPZ))
        self._side = dict(np.load(side_npz or C.SIDE_NPZ))
        self._group_cache: dict[str, np.ndarray] = {}

    # -- shape

    def n_rows(self, split: Split) -> int:
        return int(self._splits[f"{split}__user_id"].shape[0])

    def row_ids(self, split: Split) -> np.ndarray:
        """0-based positional index -- the submission's primary key."""
        return np.arange(self.n_rows(split), dtype=np.int64)

    # -- features

    def safe_columns(self) -> tuple[str, ...]:
        return C.LOG_SAFE

    def features(self, split: Split) -> dict[str, np.ndarray]:
        """Impression-time columns. Identical column set for every split, by design:
        a feature you cannot compute at serving time is not a feature."""
        return {c: self._splits[f"{split}__{c}"] for c in C.LOG_SAFE}

    def column(self, split: Split, name: str) -> np.ndarray:
        if name in C.LOG_OUTCOME and split != "train":
            raise OutcomeColumnAccessError(
                f"{name!r} is a post-impression outcome and is only available on "
                f"train (requested {split!r}). Use it as an auxiliary target, not a "
                f"feature -- long_view is a threshold on play_time_ms/duration_ms."
            )
        if name == C.LABEL and split == "test":
            raise TestLabelAccessError("test labels are not reachable from DataAPI")
        return self._splits[f"{split}__{name}"]

    # -- targets

    def labels(self, split: Split) -> np.ndarray:
        """long_view. Available for train and valid; never for test.

        Valid labels are deliberately available: the brief says the agent develops on
        "the training split and the public validation feedback", and early stopping
        needs them.
        """
        if split == "test":
            raise TestLabelAccessError(
                "The agent never sees the hidden test set. Produce scores for test "
                "and let the harness evaluate them."
            )
        # int64, not the int8 we store. The frozen evaluate.py aggregates labels with
        # builtin sum(); under NumPy 2's weak promotion an int8 array accumulates in
        # int8 and wraps past 127 positives, yielding a wrong metric with no error.
        # Widening here removes the footgun for every caller, agent code included.
        return self._splits[f"{split}__{C.LABEL}"].astype(np.int64)

    def aux_targets(self, split: Literal["train"] = "train") -> dict[str, np.ndarray]:
        """The 10 auxiliary feedback signals, train only.

        Named for what they are so that using them as features is a deliberate act
        rather than an accident.
        """
        if split != "train":
            raise OutcomeColumnAccessError(
                "Auxiliary feedback signals exist for train only; on an evaluation "
                "split they are the answer, not a feature."
            )
        return {c: self._splits[f"train__{c}"] for c in C.LOG_OUTCOME}

    # -- grouping

    def groups(self, split: Split) -> np.ndarray:
        """Contiguous group id per row, grouped by user -- the unit the metric scores.

        This is what a pairwise or listwise loss needs. For finer training-time
        groups (per user-day, per session) build them from `date` / `hourmin`.
        """
        if split not in self._group_cache:
            users = self._splits[f"{split}__user_id"]
            _, inverse = np.unique(users, return_inverse=True)
            self._group_cache[split] = inverse.astype(np.int32)
        return self._group_cache[split]

    # -- side tables, indexed by id

    def video_feature(self, name: str) -> np.ndarray:
        return self._side[f"video__{name}"]

    def user_feature(self, name: str) -> np.ndarray:
        return self._side[f"user__{name}"]

    def side_columns(self) -> dict[str, list[str]]:
        vids = sorted(k[7:] for k in self._side if k.startswith("video__"))
        uids = sorted(k[6:] for k in self._side if k.startswith("user__"))
        return {"video": vids, "user": uids}


def load_test_labels() -> np.ndarray:
    """Read the held-out test labels. Only `score_final.py` may call this.

    Kept out of DataAPI on purpose: the import site is the audit trail.
    """
    if not C.TEST_LABELS_NPY.exists():
        raise FileNotFoundError("test labels not materialised; run build_cache() first")
    return np.load(C.TEST_LABELS_NPY)
