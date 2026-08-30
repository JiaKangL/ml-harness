# P3 — `harness/executor.py` (L3, Execution)

**Purpose.** Run agent-written code without letting it take the harness down, and
decide whether its output is trustworthy. Generated code crashes, hangs, and
occasionally succeeds while producing nonsense; all three must be survivable.

**Depends on:** L1 only. **Must not import:** memory, logger, agent, loop.

## The candidate script contract

Decided here, once. The executor builds the argv, the lint decides what a candidate may
import in order to satisfy it, and L5 pastes it into the prompt -- three consumers, so
it lives in exactly one place: `executor.CANDIDATE_CONTRACT`.

    python <candidate.py> --split {train|valid|test} --seed <int> --out <path.npy> [--frac <float>]

- Writes a **float64 `.npy` array of scores, one per row of the requested split**, in
  the split's canonical row order: row *i* of the array scores row *i* of the split.
- Obtains data via `from harness.data_guard import DataAPI`. The repo root is on
  `PYTHONPATH`; the harness sets it, the script must not.
- `--frac` is used only by the smoke run and means *use this fraction of USERS for
  training*. **Sample users, not rows.** Ranking is scored within user, so a 1% row
  sample shreds the impression groups and produces a meaningless run that may fail for
  reasons unrelated to the candidate's logic -- and the agent then spends a repair
  attempt on a bug the harness invented. The script must still emit scores for **all**
  rows of the requested split.
- Exit 0 on success.

`argparse` is therefore on `ALLOWED_IMPORTS`: the contract is a CLI, so every candidate
must parse arguments, and a lint that rejected the one shape the prompt asks for would
fail every candidate in the run.

## Five stages, cheapest first

1. `ast.parse` — syntax.
2. **Static contract lint** — reject network imports, writes outside the workspace,
   and any reference to the raw data directory or `cache/_holdout`.
3. **Smoke run** — 1% stratified slice, 30s cap. *The highest-ROI mechanism in the
   build*: shape errors, bad indexing, wrong dtypes and misaligned groups surface in
   10 seconds instead of at full cost.
4. **Full run** under process-group and RSS supervision.
5. **Output validation** — see below.

## Runtime requirements

### Process-tree termination

`proc.kill()` signals only the direct child. PyTorch DataLoader workers, and any
BLAS thread pool, survive it — they then steal cores from the next iteration and
corrupt every timing measurement we take afterwards.

Spawn into a **new process group** (`start_new_session=True`, equivalently
`preexec_fn=os.setsid` on Unix) and terminate the whole tree:

```python
os.killpg(os.getpgid(proc.pid), signal.SIGTERM)   # 5s grace
os.killpg(os.getpgid(proc.pid), signal.SIGKILL)   # then force
```

Verify the group is empty before the next launch; a surviving PID is a test failure,
not a warning.

### Library preflight

The starter kit is pure numpy, but the agent will generate torch or LightGBM code. At
runner startup, import-check everything in `config.ALLOWED_IMPORTS` +
`OPTIONAL_IMPORTS` **once**, and fail loudly as a *harness* fault. Otherwise a missing
dependency reads as a broken candidate, the agent burns repair attempts on code that
was correct, and the ledger records a false DISCARD that poisons every later prompt.

The prompt states the available library list verbatim; anything outside it is a
contract violation caught at lint, before execution.
- **stdout/stderr to files, never `PIPE`.** Killing a piped process discards its
  buffered output — the timeout case is exactly when the traceback matters most.
- Out-of-band RSS watchdog at 1 Hz, kill at `RSS_CAP_BYTES`. `RLIMIT_AS` is unreliable
  under the macOS ARM allocator; do not rely on it.
- Child env from `config.CHILD_ENV` (`PYTHONHASHSEED`, BLAS thread pinning) — set by
  the harness, never by generated code, so the agent cannot make a run irreproducible.

## Output validation — what separates "it ran" from "it worked"

- Scores are finite (no NaN/Inf), correct row count, `row_id` alignment.
- **Within-user score variance > 0.** A constant-score model "runs" perfectly.
- Seed determinism: same seed twice → identical to 1e-6.

## The retry ladder

**Ownership: the loop (L6) drives this ladder; the executor only classifies.**
`Executor` returns a `FailureRecord` with a `FailureClass` and never decides what
happens next -- regenerating needs `llm.py` (L5), marking a node FAILED and reverting
needs `memory.py` (L4), and L3 may import neither. The table below is the loop's
policy, stated here because it is the executor's classification that drives it.

Not every failure costs an iteration. This distinction is the Robustness story.

| Class | Response | Costs an iteration? |
|---|---|---|
| Syntax | Regenerate with the error, max 2 | no |
| Contract violation | Reject citing the rule, regenerate | no |
| Smoke failure | Repair with traceback, max 2 | no |
| Runtime traceback | One repair attempt | yes |
| Timeout / OOM | Feed back the resource fact (peak RSS, wall time) | yes |
| Silent-wrong | Abandon branch | yes |

### The circuit breaker

Hard limit of `MAX_SELF_HEAL_ATTEMPTS = 3` repairs per iteration. On each failure the
`stderr` tail plus the failing frame goes back to the LLM for a patch. After the third
consecutive failure:

1. mark the node `FAILED`,
2. log the traceback and every repair attempt into `iteration_logs.json`,
3. revert to the parent node,
4. move to the next iteration.

Additionally dedupe by exception signature `(type, normalised message, frame)`: the
same failure twice short-circuits the breaker early. Repair loops that regenerate the
same fix are the most common way an agent harness converts its whole budget into
nothing — and the recovery record is itself graded under Robustness, which is scored
on *how* a failure is handled, not on whether one occurred.

## Contract

`RunResult` lives in `harness/types.py`, not here: L2 already codes against it
(`Evaluator.build_submission` reads `.ok` and `.scores_path` off whatever its injected
runner returns), and a dataclass defined inside L3 would force L2 to duck-type its way
around a layer it may not import.

```python
class HarnessDependencyError(RuntimeError): ...   # a declared library will not import
class ProcessGroupEscapeError(RuntimeError): ...  # a PID survived SIGTERM and SIGKILL

def check_imports(modules: tuple[str, ...] | None = None) -> dict[str, str]: ...
def failure_signature(exc_type: str, message: str, frame: str = "") -> str: ...
def parse_traceback(text: str) -> tuple[str, str, str]: ...   # (type, message, frame)
def group_pids(pgid: int) -> list[int]: ...                   # live, non-zombie

class Executor:
    def check_syntax(self, code: str) -> FailureRecord | None: ...
    def lint_contract(self, code: str) -> FailureRecord | None: ...
    def smoke(self, code: str | Path, node_id: str, split: Split = "valid",
              seed: int = 42) -> RunResult: ...
    def run(self, code: str | Path, node_id: str, split: Split = "valid",
            seed: int = 42, timeout_s: float | None = None) -> RunResult: ...
    def validate_output(self, scores_path: Path, split: Split,
                        structural_only: bool = False) -> FailureRecord | None: ...
```

`code` accepts source text *or* the path of a stored candidate. P2's `Runner` protocol
is `Callable[[str, str, Split, int], RunResult]` and passes `node.code_path`, while this
plan named the same parameter `code: str` and meant source; rather than have the loop
remember which is which, a single-line string ending in `.py` that exists on disk is
read as a path and anything else is treated as source.

`structural_only` is what smoke passes. Smoke validates that the file exists, is finite
and has the right length, but **not** within-user variance: on 1% of users a legitimate
model can collapse to a constant for reasons that say nothing about the code, and a
smoke failure would send the agent off repairing a bug the harness invented. Variance is
a silent-wrong check and belongs at full scale.

A smoke-stage failure is classified `SMOKE` whatever the symptom -- the *stage*, not the
symptom, decides whether the repair is free. The symptom is not lost: `killed_by` on the
returned `ResourceFacts` still says whether it was a timeout.

## Acceptance tests — `tests/test_executor.py`

| Test | Passes when |
|---|---|
| Orphan reaping | A script spawning children then hanging leaves **no** live PIDs after kill |
| Timeout captured | A hung script still yields its stderr tail (proves files-not-PIPE) |
| Constant scores rejected | Uniform output fails `validate_output` |
| NaN rejected | Non-finite scores fail |
| Misalignment rejected | Wrong row count fails |
| Syntax caught pre-run | Bad syntax never reaches a subprocess |
| Lint catches leakage | A script importing `csv` and opening the raw data dir is rejected |
| Smoke is fast | A script that would take 5 min fails smoke in < 35s |
| Determinism | Same seed twice → identical scores |
| Signature dedupe | The same traceback twice stops the repair loop |
| Circuit breaker | 3 failed repairs mark the node FAILED and revert to parent |
| Breaker is logged | All 3 attempts appear in the iteration log, not just the last |
| Import preflight | A missing optional library fails as a harness fault, not a candidate bug |
| RSS watchdog | A candidate over its cap is killed and classified OOM, with the peak in the message |
| Per-user constant | Scores that vary across users but not within them are rejected |
| Comments are not code | `# never read test_labels` lints clean; the lint is AST, not substring |

Circuit-breaker and iteration-log rows belong to the loop (L6) and L4 and are tested
there; L3's half of them is the `FailureClass` and the signature, both covered above.
