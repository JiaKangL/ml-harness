# P7 — `score_final.py` + write-up

**Purpose.** Produce the submission, read the test set exactly once, and write the
deliverables judges actually read.

**Depends on:** everything.

## `score_final.py` — the one permitted test read

The **only** code in the repository that may call `data_guard.load_test_labels()`. The
import site is the audit trail.

Rules, enforced not intended:

1. Refuses to run unless the run is marked converged.
2. Runs **once**. A second invocation requires an explicit `--force` and is logged as
   an override, because a second draw on the test set makes the estimate
   optimistically biased.
3. Its output never re-enters the loop. If we improve the harness afterwards, we go
   back to test-blind.

Writes `logs/final_result.json`: GAUC, nDCG@5, primary, and the absolute delta per
metric against the baseline (0.6610 / 0.5282), which is the competition's scoring
formula `score_dataset = mean over m of delta(m)`.

## Submission

- Built by re-executing the promoted node's stored source from disk.
- `python3 submit.py --check --split test outputs/submission.csv` as a hard gate.
- `--score --split valid` as a consistency check against the harness-recorded number;
  disagreement means something is misaligned and we want to know before submitting.

## Deliverables

| Artifact | Notes |
|---|---|
| Public GitHub repo | Currently private — flip before submission |
| README | Overview, setup, reproduction steps, limitations reflection |
| Architecture diagram | One page: the six layers and the loop |
| `logs/iteration_logs.json` | The primary proof deliverable |
| `outputs/candidate_iter_NN.py` | Every script the agent wrote — the evidence |
| `outputs/best_model.py` | The winning standalone script |
| `outputs/submission.csv` | Verified with `submit.py --check` |
| `logs/data_profile.json`, `logs/eda_report.md` | The stored EDA |
| Results table | Validation-best + absolute delta per metric |
| **Manual intervention count** | Target **0** |
| Devpost description | How it addresses the problem, tools, libraries, datasets |

## Write-up: the three things worth leading with

1. **The noise gate, and the fact that we measured it.** We assumed σ≈0.0012, measured
   σ=0.00035 over 5 seeds, and found our headline claim was 3.3× overstated. The gate
   stayed — but for a corrected reason: FM's variance is not a torch model's variance,
   and a single sample gives no variance estimate at all. Reporting the correction is
   stronger than reporting the original claim would have been.
2. **The within-user variance lens.** A feature is worth what it varies inside a
   user's group; `tab` is 100x predictive globally and constant for 48% of users.
   This explains the organizers' published dead ends mechanistically.
3. **Autonomous stall escalation.** Critics instead of humans, so the intervention
   count stays at zero.

Also report **calibration**: the correlation between predicted and realised deltas. If
it is positive the agent is reasoning; if it is flat it is guessing. Reporting it
honestly either way is stronger than not measuring it.

## Final checklist

- [ ] No API key in the tree **or in git history**
- [ ] `unittest discover` fully green
- [ ] `submit.py --check` passes on test
- [ ] Repo flipped to public
- [ ] Dataset not committed; `scripts/get_data.sh` works from a clean clone
- [ ] A clean clone reproduces preflight
