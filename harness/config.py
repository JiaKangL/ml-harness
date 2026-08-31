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
STATE_JSONL = LOGS_DIR / "state.jsonl"
EDA_REPORT_MD = LOGS_DIR / "eda_report.md"
EDA_JSON = LOGS_DIR / "eda.json"
FINAL_RESULT_JSON = LOGS_DIR / "final_result.json"
RUNS_DIR = ROOT / "runs"


def candidate_path(iteration: int) -> Path:
    """outputs/candidate_iter_04.py -- the script the agent wrote for one iteration."""
    return OUTPUTS_DIR / f"candidate_iter_{iteration:02d}.py"


def run_dir(iteration: int) -> Path:
    """Per-iteration working directory: stdout/stderr, scores, checkpoints."""
    return ROOT / "runs" / f"iter_{iteration:02d}"

# Test labels are NEVER materialised. They are parsed from the organizer's raw log at
# scoring time, and only after the run is sealed -- so during a run there is no
# test-label artifact anywhere for a stray glob to find. See harness/holdout.py.
HOLDOUT_DIR = CACHE_DIR / "_holdout"          # asserted empty by preflight
RUN_SEAL_JSON = LOGS_DIR / "run_seal.json"    # written on convergence; gates scoring
SCORED_MARKER_JSON = LOGS_DIR / "test_draws.json"  # every draw on test, recorded

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

# The random-exposure log spans 20220422-20220508, i.e. BOTH the valid and test
# windows: 1,186,059 rows of which 897,721 (75.7%) are test-window and every one
# carries long_view. We cache only rows on or before this date, so the unbiased
# promotion check can never see test ground truth.
RANDOM_EXPOSURE_MAX_DATE = 20220428

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

# Two-tier seed ladder. Seed 42 runs first; a clear regression is pruned there, and
# survivors run the remaining seeds. We prune early but NEVER promote early.
#
# sigma(single seed) ~ 0.0011, so sigma(3-seed mean) ~ 0.00064:
#   +0.002 at 1 seed  = 1.8 sigma  -> noise
#   +0.002 at 3 seeds = 3.1 sigma  -> signal
# Three seeds is precisely what makes the organizers' own epsilon usable as a
# promotion threshold rather than merely as a convergence-reporting rule.
CONFIRM_SEEDS = (42, 43, 44)

# Prune after seed 42 alone. At -0.005 this is ~6 sigma below a promotable candidate,
# so it is extremely safe -- but it only catches badly-broken-yet-running candidates,
# since a merely neutral one scores ~0.000 and survives to 3 seeds anyway. Tightening
# to -0.002 is still 3.6 sigma (loses a good candidate ~0.02% of the time) and prunes
# considerably more wall-clock. One-line change if we want the extra speed.
PRUNE_AT_ONE_SEED_DELTA = -0.005
PROMOTE_DELTA = 0.002  # required 3-seed mean improvement over the parent
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

# Self-healing circuit breaker. Syntax and smoke failures are repaired without
# spending an iteration. After 3 consecutive failed repairs the node is marked FAILED,
# the traceback is logged, and the loop reverts to the parent node -- a repair loop
# that keeps regenerating the same fix is the most common way an agent harness
# converts its entire budget into nothing.
MAX_SELF_HEAL_ATTEMPTS = 3
MAX_IDENTICAL_FAILURES = 2  # same (exc type, message, frame) twice -> prune early

# Grounding resolution is advisory, not blocking. A proposal cites a data_profile
# field; we resolve it exactly, then by fuzzy match, and if neither works we record
# grounding_verified=False and run it anyway. Never drop runnable code over a typo.
GROUNDING_FUZZY_CUTOFF = 0.6

# Libraries the agent may import. Verified at preflight so that a missing dependency
# surfaces as a harness fault rather than being misread as a broken candidate.
# The prompt states this list verbatim; anything outside it is a contract violation.
# `argparse` is on the list because the candidate script contract is a CLI --
# `--split/--seed/--out/--frac` -- so every candidate must parse arguments. Without it
# the lint would reject the one shape we ask the agent to produce.
ALLOWED_IMPORTS = ("numpy", "math", "csv", "json", "collections", "itertools",
                   "random", "time", "os", "sys", "pathlib", "dataclasses", "typing",
                   "functools", "heapq", "statistics", "warnings", "abc", "enum",
                   "argparse")

# Import-checked at preflight and stated verbatim in the prompt. Without the check, a
# missing package reads as a broken candidate: the agent concludes its *idea* failed,
# writes DISCARD into the ledger, and a whole research axis dies to one absent
# dependency.
OPTIONAL_IMPORTS = ("torch",)

# ---------------------------------------------------------------- LLM backend
#
# Read through the environment rather than hard-coded, because the key that runs this
# may not be a first-party Anthropic key: a hackathon gateway issues its own
# credential and its own endpoint. Every one of these falls back to the SDK's own
# variable, so an ordinary `export ANTHROPIC_API_KEY=...` still works untouched.
#
#   HARNESS_LLM_API_KEY     the credential, sent as x-api-key   (-> ANTHROPIC_API_KEY)
#   HARNESS_LLM_AUTH_TOKEN  a bearer token instead of a key     (-> ANTHROPIC_AUTH_TOKEN)
#   HARNESS_LLM_BASE_URL    an alternate endpoint               (-> ANTHROPIC_BASE_URL)
#   HARNESS_LLM_MODEL       the model id the gateway expects
#
# A gateway is usable here only if it speaks the Anthropic Messages API. That is not a
# preference: the run depends on `cache_control` with a 1h TTL and on adaptive
# thinking, neither of which survives translation through an OpenAI-shaped endpoint,
# and the whole token-accounting deliverable is read off Anthropic's `usage` fields.
def _load_dotenv(path: Path) -> None:
    """Read `.env` into the environment, without overriding anything already set.

    Hand-rolled rather than a dependency: the starter kit's selling point is that it
    needs numpy and nothing else, and this is fifteen lines. It exists so a credential
    can live in a gitignored file instead of a shell profile or a chat transcript --
    and so the same key is visible to the loop, to a subprocess, and to whoever picks
    the run up next, without anyone having to remember to export it.

    A real environment variable always wins, so `.env` is a convenience and never a
    surprise: if a run behaves oddly, what is exported is what is in effect.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

LLM_API_KEY = os.environ.get("HARNESS_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
LLM_AUTH_TOKEN = (
    os.environ.get("HARNESS_LLM_AUTH_TOKEN") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
)
LLM_BASE_URL = os.environ.get("HARNESS_LLM_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")

MODEL = os.environ.get("HARNESS_LLM_MODEL", "claude-opus-5")
CRITIC_MODEL = MODEL
N_CRITICS = 3
MAX_CRITIQUE_ROUNDS = 2
MAX_TOKENS = 32_000
LLM_MAX_RETRIES = 3

# 1h, not the bare-ephemeral 5-minute default: an iteration straddles 5 minutes, so
# the default would give erratic hits and the miss is silent (visible only on cost).
CACHE_TTL = "1h"

# Claude Opus 5 list price, USD per million tokens. Cache reads ~0.1x input,
# 1h-TTL writes 2x. Feasibility is graded on tokens, so this is a deliverable.
PRICE_IN_PER_MTOK = 5.00
PRICE_OUT_PER_MTOK = 25.00
PRICE_CACHE_READ_PER_MTOK = 0.50
PRICE_CACHE_WRITE_PER_MTOK = 10.00

KILL_GRACE_S = 5
ENSEMBLE_TOP_K = 5
N_SEEDING_ITERATIONS = 4        # one forced probe per priority axis
EXPLOIT_TRUNK_PROBABILITY = 0.6  # vs branching from an under-explored axis
REWRITE_LINE_FRACTION = 0.6      # >60% of lines changed => labelled a rewrite

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
