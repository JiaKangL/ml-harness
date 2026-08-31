"""L6 -- the live display. What the run looks like while it is running, and what gets
demoed.

Deliberately plain text on stdout with no dependency and no cursor control: the run is
also piped to a file, watched over ssh, and read after the fact, and a redraw-based
display is unreadable in all three. Everything printed here is also in
`logs/iteration_logs.json`; this is the human view, not the record.
"""
from __future__ import annotations

import sys
import time

_C = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}


class Console:
    def __init__(self, stream=None, colour: bool | None = None, quiet: bool = False):
        self.stream = stream or sys.stdout
        self.quiet = quiet
        self.colour = self.stream.isatty() if colour is None else colour
        self.started = time.monotonic()

    def _c(self, name: str, text: str) -> str:
        return f"{_C[name]}{text}{_C['reset']}" if self.colour else text

    def write(self, line: str = "") -> None:
        if not self.quiet:
            print(line, file=self.stream, flush=True)

    # -- run level

    def banner(self, mode: str, max_iters: int, backend: str = "") -> None:
        self.write()
        self.write(self._c("bold", f"═══ autonomous ML research agent — {mode} mode ═══"))
        self.write(self._c("dim", f"    up to {max_iters} iterations; ctrl-C is safe, the run resumes"))
        if backend:
            self.write(self._c("dim", f"    {backend}"))

    def stage(self, text: str) -> None:
        self.write(self._c("dim", f"  · {text}"))

    def iteration(self, n: int, axis: str, parent: str, best: float) -> None:
        self.write()
        elapsed = time.monotonic() - self.started
        self.write(
            self._c("bold", f"── iteration {n:02d} ")
            + self._c("dim", f"axis={axis} parent={parent} best={best:.4f} "
                             f"t+{elapsed / 60:.0f}m")
        )

    def hypothesis(self, text: str, predicted: float, grounding: str, verified: bool) -> None:
        wrapped = text if len(text) < 300 else text[:297] + "..."
        self.write(f"  {self._c('cyan', 'hypothesis')} {wrapped}")
        mark = "" if verified else self._c("yellow", " (unresolved)")
        self.write(
            self._c("dim", f"  predicts Δ{predicted:+.4f}  grounding={grounding}") + mark
        )

    def tokens(self, usage) -> None:
        self.write(
            self._c(
                "dim",
                f"  generated in {usage.latency_seconds:.0f}s — "
                f"{usage.prompt_tokens:,} in / {usage.completion_tokens:,} out / "
                f"{usage.cache_read_tokens:,} cached (${usage.cost_usd:.3f})",
            )
        )

    def ok(self, text: str) -> None:
        self.write(f"  {self._c('green', '✓')} {text}")

    def warn(self, text: str) -> None:
        self.write(f"  {self._c('yellow', '!')} {text}")

    def fail(self, text: str) -> None:
        self.write(f"  {self._c('red', '✗')} {text}")

    def seed_result(self, seed: int, metrics, seconds: float) -> None:
        self.write(
            f"    seed {seed}: primary {metrics.primary:.4f} "
            f"(GAUC {metrics.gauc:.4f}, nDCG@5 {metrics.ndcg5:.4f})"
            + self._c("dim", f"  {seconds:.0f}s")
        )

    def gate(self, decision) -> None:
        if decision.promote:
            self.write(f"  {self._c('green', 'PROMOTED')} {decision.reason}")
        elif decision.quarantined:
            self.write(f"  {self._c('red', 'QUARANTINED')} {decision.reason}")
        else:
            self.write(f"  {self._c('dim', 'held')} {decision.reason}")

    def summary(self, summary) -> None:
        self.write()
        self.write(self._c("bold", "═══ run summary ═══"))
        best = summary.best_valid or {}
        self.write(f"  iterations              {summary.iterations}")
        self.write(f"  converged               {summary.converged} "
                   f"(at iteration {summary.convergence_iteration})")
        if best:
            self.write(
                f"  best valid              primary {best.get('primary', 0):.4f} "
                f"(GAUC {best.get('gauc', 0):.4f}, nDCG@5 {best.get('ndcg5', 0):.4f}) "
                f"— node {best.get('node_id')}"
            )
        self.write(f"  manual interventions    {summary.manual_interventions}")
        self.write(
            f"  tokens                  {summary.total_tokens.prompt_tokens:,} in / "
            f"{summary.total_tokens.completion_tokens:,} out / "
            f"{summary.total_tokens.cache_read_tokens:,} cached "
            f"({summary.total_tokens.cache_hit_rate:.0%} hit rate)"
        )
        self.write(f"  cost                    ${summary.cost_usd:.2f}")
        self.write(f"  wall clock              {summary.wall_clock_seconds / 60:.0f} min")
        cal = summary.calibration_r
        self.write(
            f"  prediction calibration  "
            + (f"r = {cal:+.2f}" if cal is not None else "not enough scored iterations")
        )
        if summary.failures_by_class:
            self.write(f"  failures                {summary.failures_by_class}")
