"""The held-out test labels. Deliberately its own module.

Nothing in the agent's data path imports this. It is not in `data_guard`, which is
the module whose contract ships verbatim in the prompt, so `from harness.data_guard
import ...` cannot reach it however the generated code is written.

Be honest about the strength of this: it is defence in depth, not a sandbox. A
subprocess running as this user can read `cache/_holdout/` directly. What we can
truthfully claim is that the cache the agent is pointed at contains no test label,
that no module it is told about exposes one, and that a single grep for this import
audits the whole repository. Real isolation would need a separate uid or a container;
neither is in scope, and the write-up says so rather than overclaiming.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def load_test_labels() -> np.ndarray:
    """Only `score_final.py` may call this, once, after convergence."""
    if not C.TEST_LABELS_NPY.exists():
        raise FileNotFoundError(
            "test labels not materialised; run `python -m harness.preflight --rebuild`"
        )
    return np.load(C.TEST_LABELS_NPY)
