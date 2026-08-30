# P5 — `prompts.py` `llm.py` `agent.py` `loop.py` `console.py` (L5–L6)

**Purpose.** The generator and the orchestrator. Built **last**, deliberately: a
mistake here is cheap and visible, whereas a mistake in L1–L4 silently corrupts every
result we would report.

**Depends on:** L1–L4. **Build mock mode in this phase, not after it.**

---

## `prompts.py` — versioned, one place

The system prompt shapes what the agent proposes, and Innovation (20%) is scored on
exactly that. It is an artifact to point at, not a string buried in `agent.py`. It is
also the frozen cache prefix, which must be **byte-identical** across ~30 calls.

Three tiers, ordered by mutation rate:

| Tier | Contents | Size | Mutates |
|---|---|---|---|
| **A — frozen** (cache breakpoint) | Role, task spec, metric semantics, **`evaluate.py` verbatim**, `DataAPI` contract, prior-knowledge pack (3 dead ends *with mechanisms* + 7 ranked directions), `data_profile.json`, output schema | ~6–8K | never |
| **B — ledger** | One line per experiment, append-only | ~1K at n=30 | appends |
| **C — working set** | Trunk source, current diff, failure context, assigned axis, last resource facts | ~4–10K | every turn |

Include `evaluate.py` verbatim: 61 lines, ~900 cached tokens, and it prevents a whole
class of hallucination about the zero-positive-user convention and GAUC's
positive-count weighting.

**Never include:** full stdout, full failed sources, prior turns' reasoning. Those are
what actually degrade a long context.

## `llm.py` — thin client

- Claude Opus 5, adaptive thinking, streaming (long turns must not hit HTTP timeouts).
### The cache breakpoint

Place **one explicit `cache_control` breakpoint immediately after the static prefix**
— system prompt + task rules + starter-kit constraints + dead ends +
`data_profile.json`. Everything dynamic (trunk code, active ledger, recent diff, error
tracebacks) is appended strictly *after* it. Target: 80%+ of prompt tokens served from
cache across iterations.

```python
system=[{"type": "text", "text": STATIC_PREFIX,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
```

**The `ttl` is not optional.** Bare `{"type": "ephemeral"}` is the **5-minute** TTL.
An iteration is roughly 3–5 minutes with numpy FM and longer with torch, so it
straddles that boundary — we would get erratic or zero cache hits and never notice,
because a cache miss raises nothing and shows up only on the bill. The 1h TTL costs 2×
on write and breaks even after three reads; we will have ~30.

Tier A must be **byte-identical** across calls: no timestamp, no iteration number, no
conditionally-included section. A single byte invalidates everything after it.
- Per-turn operator instructions go in as `{"role": "system"}` *messages*, preserving
  the cached prefix rather than invalidating it.
- Token/cost accounting per call, including cache-read/write fields.
- **Assert `cache_read_input_tokens > 0` in a test.** It is the only ground truth that
  caching works, and a prefix regression is otherwise invisible except on the bill.

Expect ~$0.25–0.30/iteration, ~$20–30 for a full run. Cost is not a constraint here.

## `agent.py` — context assembly + proposal validation

Parses the structured output into a `Proposal`.

### Grounding resolution is advisory, not blocking

A proposal cites a field from `data_profile.json` as its motivation. Resolve it in
three tiers:

1. **Exact match** against the profile's key set.
2. **Fuzzy match** — `difflib.get_close_matches(key, profile_keys, n=1, cutoff=0.6)`
   — which absorbs paraphrase (`"tab_variance"` → `"within_user_variance.tab"`).
3. **Unresolvable** → set `grounding_verified=False` on the node, log a warning, and
   **execute anyway**.

Never drop runnable code over a grounding typo. The purpose of the field is to make
the agent reason from measured data rather than from score feedback; a paraphrased key
still demonstrates that, and refusing to run costs a real iteration to punish a
spelling mistake. The `grounding_verified` flag makes the rate visible in the log,
which is the honest way to report it.

### Still rejected before execution (zero compute cost)

- missing `predicted_delta` — without it the iteration is a search step, not a test
- an `axis` outside the closed set
- `axis == "architecture"` before all four priority axes have a scored attempt
- a diff touching **only** the `FIELDS` list or an embedding-dim constant — a known
  dead end with published counter-evidence
- an import outside `ALLOWED_IMPORTS` + `OPTIONAL_IMPORTS`

### Self-healing loop

Failure context sent back to the model: `stderr` tail (~40 lines), the failing frame
±15 lines, and the list of repairs already attempted on this node so it does not loop
the same fix.

`MAX_SELF_HEAL_ATTEMPTS = 3`. After the third consecutive failure the node is marked
`FAILED`, all three attempts are written to `iteration_logs.json`, the loop reverts to
the parent node and moves on. Robustness is scored on *how* a failure is handled, so
the full recovery record is the deliverable — not just the final outcome.

## `loop.py` — the orchestrator

```
preflight → profile → for each iteration:
    parent = tree.select_parent(axis)
    proposal = agent.propose(parent, ledger, assigned_axis, failure_ctx)
    reject-or-execute → smoke → 3 seeds → evaluator.gate → tree.promote?
    ledger.record(insight) ; logger.log(entry) ; console.render()
    if stalled >= 2: critics.escalate()      # P6
until converged / 50 iterations / 6h
→ ensemble → build_submission → verify
```

Iterations 1–4 are a forced seeding round, one per priority axis, so the ledger has a
real observation on each before exploitation begins. Then UCB over axes weighted by
realised Δ, with a floor so no axis starves.

## `console.py`

The live display, and what gets demoed. Per iteration: header with parent node and
best score; hypothesis; generation with token counts; AST result; subprocess status
per seed; any self-heal attempts; the gate decision; state update.

## Mock mode — build it here

`--mock` swaps `llm.py` for four pre-written candidates:

1. a genuine improvement (listwise softmax)
2. one that crashes with a `NameError` (exercises self-healing)
3. one that emits constant scores (exercises output validation)
4. one attempting to read `play_time_ms` on valid (exercises the firewall)

This exercises L2–L6 end-to-end in seconds with zero tokens. On a one-day build it is
the difference between debugging the loop and debugging the loop *while waiting on
API calls*.

## Acceptance tests — `tests/test_agent_loop.py`

| Test | Passes when |
|---|---|
| Cache holds | Two consecutive calls report `cache_read_input_tokens > 0` |
| Cache TTL | The request carries `ttl: "1h"`, not bare ephemeral |
| Cache share | Cached tokens are >= 60% of prompt tokens by iteration 3 |
| Grounding fuzzy | A paraphrased key resolves via difflib and still runs |
| Missing prediction rejected | A proposal with no predicted_delta IS rejected pre-execution |
| Grounding advisory | An unresolvable key sets grounding_verified=False and still runs |
| Circuit breaker | 3 failed repairs mark FAILED and revert to parent |
| Usage captured | prompt_tokens/completion_tokens recorded on every call incl. repairs |
| Prefix frozen | Tier A is byte-identical across 5 assemblies |
| Dead-end rejection | A FIELDS-only diff is rejected without running |
| Axis lock | `architecture` is refused until the four priority axes are scored |
| Mock end-to-end | `loop.py --mock --max-iters 5` → 5 valid log entries |
| Self-heal | The crashing mock candidate is repaired without costing an iteration |
| Firewall in the loop | The leakage mock candidate is caught and logged, not promoted |
| Resume | Killing mid-run and restarting continues from the last node |
