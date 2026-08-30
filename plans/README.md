# Implementation sub-plans

One file per phase of [../PLAN.md](../PLAN.md). Each is written to be picked up
independently: it states the contract the module must satisfy, what it may and may
not depend on, and the acceptance tests that decide when it is done.

**Rules that apply to every sub-plan:**

1. **Dependencies point downward only.** An L3 module may import L1 and L2; never L4+.
   If you find yourself needing an upward import, the boundary is wrong — say so
   rather than working around it.
2. **Code against `harness/types.py`.** Do not redefine shared dataclasses locally.
3. **Never touch `kuairand-starter-kit/evaluate.py`.** It is checksummed by preflight.
4. **Never read test labels.** Only `score_final.py` may, and only once.
5. **Every module ships its tests in the same change.** `tests/test_<module>.py`,
   stdlib `unittest`, asserting behaviour under attack rather than that code exists.
6. **Run `./.venv/bin/python -m unittest discover -s tests -t .` before committing.**

| Phase | Sub-plan | Depends on |
|---|---|---|
| 1b | [P1b-eda.md](P1b-eda.md) | L1 |
| 2 | [P2-evaluator.md](P2-evaluator.md) | L1 |
| 3 | [P3-executor.md](P3-executor.md) | L1 |
| 4 | [P4-memory-logger.md](P4-memory-logger.md) | L1, L2 |
| 5 | [P5-agent-loop.md](P5-agent-loop.md) | L1–L4 |
| 6 | [P6-critics.md](P6-critics.md) | L1–L5 |
| 7 | [P7-submission.md](P7-submission.md) | all |
