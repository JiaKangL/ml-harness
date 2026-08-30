# P3 — `harness/executor.py` (L3, Execution)

**Purpose.** Run agent-written code without letting it take the harness down, and
decide whether its output is trustworthy. Generated code crashes, hangs, and
occasionally succeeds while producing nonsense; all three must be survivable.

**Depends on:** L1 only. **Must not import:** memory, logger, agent, loop.

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

```python
@dataclass(frozen=True)
class RunResult:
    ok: bool
    scores_path: Path | None
    resources: ResourceFacts          # wall_seconds, peak_rss_bytes, killed_by
    stdout_path: Path; stderr_path: Path
    failure: FailureRecord | None

class Executor:
    def check_syntax(self, code: str) -> FailureRecord | None: ...
    def lint_contract(self, code: str) -> FailureRecord | None: ...
    def smoke(self, code: str, node_id: str) -> RunResult: ...
    def run(self, code: str, node_id: str, split: Split, seed: int) -> RunResult: ...
    def validate_output(self, scores_path: Path, split: Split) -> FailureRecord | None: ...
```

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
