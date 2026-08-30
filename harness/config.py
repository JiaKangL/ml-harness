"""Paths, column legality, and the constants the preflight gate checks against.

Everything the harness treats as ground truth lives here so that no other module
hard-codes a path, a threshold, or a column name.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parent.parent
STARTER_KIT = ROOT / "kuairand-starter-kit"
DATA_DIR = STARTER_KIT / "KuaiRand-Pure" / "data"
CACHE_DIR = ROOT / "cache"
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"

# The cache the agent is allowed to see. Contains no test labels.
SPLITS_NPZ = CACHE_DIR / "splits.npz"
SIDE_NPZ = CACHE_DIR / "side.npz"
VOCAB_JSON = CACHE_DIR / "vocab.json"

# Deliverables, named as the submission expects them.
DATA_PROFILE_JSON = LOGS_DIR / "data_profile.json"
ITERATION_LOGS_JSON = LOGS_DIR / "iteration_logs.json"
ITERATION_LOGS_JSONL = LOGS_DIR / "iteration_logs.jsonl"  # crash-safe write path
PREFLIGHT_JSON = LOGS_DIR / "preflight.json"
BEST_MODEL_PY = OUTPUTS_DIR / "best_model.py"
SUBMISSION_CSV = OUTPUTS_DIR / "submission.csv"


def candidate_path(iteration: int) -> Path:
    """outputs/candidate_iter_04.py -- the script the agent wrote for one iteration."""
    return OUTPUTS_DIR / f"candidate_iter_{iteration:02d}.py"


def run_dir(iteration: int) -> Path:
    """Per-iteration working directory: stdout/stderr, scores, checkpoints."""
    return ROOT / "runs" / f"iter_{iteration:02d}"

# Test labels live off the agent's path entirely. Only score_final.py reads this.
HOLDOUT_DIR = CACHE_DIR / "_holdout"
TEST_LABELS_NPY = HOLDOUT_DIR / "test_labels.npy"

# Absolute interpreter path. Resolved by PATH exactly once, here, then pinned:
# three Pythons are installed on this machine and only one has numpy.
PYTHON_BIN = os.environ.get("HARNESS_PYTHON", sys.executable)

# ---------------------------------------------------------------- splits

SPLITS: dict[str, tuple[int, int]] = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}

# Read order is load-bearing: submission row_id is the index into this exact
# sequence, so the cache must reproduce data.load()'s ordering byte for byte.
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
RANDOM_LOG_FILE = "log_random_4_22_to_5_08_pure.csv"

# ---------------------------------------------------------------- column legality
#
# The split between these three lists IS the leakage firewall. A column is SAFE
# only if its value is knowable at the moment the video is shown, before the user
# reacts to it.

LABEL = "long_view"

LOG_SAFE = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "tab",  # which feed surface the impression came from
    "is_rand",  # randomly-inserted exposure flag (a property of the logging policy)
    "duration_ms",  # video length: a property of the item, not of the reaction
)

# Everything the user did *after* being shown the video. Legitimate as auxiliary
# training targets; catastrophic as inference-time features. `long_view` is a
# threshold on play_time_ms / duration_ms, so play_time_ms literally is the answer:
# measured median play/duration ratio is 0.98 for positives and 0.03 for negatives.
LOG_OUTCOME = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
)

VIDEO_FEATURES_FILE = "video_features_basic_pure.csv"
USER_FEATURES_FILE = "user_features_pure.csv"

# Columns of video_features_basic that are strings and need vocab encoding.
VIDEO_STRING_COLS = ("video_type", "upload_dt", "upload_type")
VIDEO_NUMERIC_COLS = (
    "author_id",
    "visible_status",
    "video_duration",
    "server_width",
    "server_height",
    "music_id",
    "music_type",
    "tag",
)

USER_STRING_COLS = (
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
)

# QUARANTINED. video_features_statistic_pure.csv holds dataset-wide aggregates
# (long_time_play_cnt, complete_play_cnt, valid_play_cnt, play_progress, ...)
# computed over the entire logging period, test window included. Those columns are
# aggregates of the very behaviour long_view measures, so joining them is both label
# leakage and temporal leakage -- while looking like an ordinary item-features file.
# Excluded from the cache. Not silently: the profile records the exclusion.
QUARANTINED_FILES = {
    "video_features_statistic_pure.csv": (
        "dataset-wide aggregates of play/completion behaviour spanning the test "
        "window; joining them leaks both the label and the future"
    ),
}

# ---------------------------------------------------------------- preflight targets
#
# From baseline_scores.json. The random check is the organizers' own harness
# self-test: if it does not reproduce, the scoring path is broken and nothing
# downstream means anything.

EXPECTED = {
    "random_valid_primary": 0.4834,
    "fm_valid_primary": 0.6016,
    "fm_valid_gauc": 0.6674,
    "fm_valid_ndcg5": 0.5357,
}
RANDOM_TOLERANCE = 0.001
FM_TOLERANCE = 0.004  # ~5 sigma of the 0.0008 seed std

# The baseline we are scored against, on the hidden test set.
BASELINE_TEST = {"GAUC": 0.6610, "nDCG@5": 0.5282, "primary": 0.5946}

# ---------------------------------------------------------------- run policy

CONFIRM_SEEDS = (42, 43, 44)  # every scored candidate runs all three
EPSILON = 0.002  # organizers' convergence threshold
N_CONVERGE = 3  # organizers' consecutive-iteration count
STALL_TRIGGER = 2  # fire critics one iteration BEFORE formal convergence
MAX_ITERATIONS = 50
WALL_CLOCK_CEILING_S = 6 * 3600

# Execution limits. Compute is deliberately not the binding constraint on this
# benchmark (the brief: ~28 min of single-core CPU for 100 baseline iterations), so
# the timeout exists to catch hangs, not to ration work.
RUN_TIMEOUT_S = 900
SMOKE_TIMEOUT_S = 30
SMOKE_FRACTION = 0.01
RSS_CAP_BYTES = 9 * 1024**3  # of 18 GB physical

# Self-healing. Syntax and smoke failures are repaired without spending an
# iteration; a full-run traceback gets one repair before the branch is abandoned.
MAX_SELF_HEAL_ATTEMPTS = 3
MAX_IDENTICAL_FAILURES = 2  # same (exc type, message, frame) twice -> prune

# LLM backend.
MODEL = "claude-opus-5"
CRITIC_MODEL = "claude-opus-5"
N_CRITICS = 3
MAX_CRITIQUE_ROUNDS = 2

# A candidate scoring above this on valid is presumed to be leakage, not skill:
# the valid oracle ceiling is 0.8484 and the baseline is 0.6016.
LEAKAGE_QUARANTINE_PRIMARY = 0.70

# Child-process environment. Pinned here rather than in generated code so the agent
# cannot accidentally (or deliberately) make a run non-reproducible.
CHILD_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}
