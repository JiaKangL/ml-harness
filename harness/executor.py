"""L3 -- run agent-written code without letting it take the harness down, and decide
whether what it produced is trustworthy.

Generated code crashes, hangs, and occasionally succeeds while producing nonsense.
All three have to be survivable, and the third is the dangerous one: a candidate that
emits a constant score runs perfectly, exits 0, and reports a plausible-looking metric.

Five stages, cheapest first: `ast.parse`, static contract lint, a 1%-of-users smoke
run, the full supervised run, and output validation. The first three cost no
iteration; that is the whole point of ordering them this way.

**Ownership.** The executor *classifies* failures and returns them. It never decides
what happens next. Regenerating code needs `llm.py` (L5) and marking a node FAILED and
reverting to its parent needs `memory.py` (L4), and L3 may import neither -- so `run`
returns a `RunResult` carrying a `FailureRecord` with a `FailureClass`, and the loop
(L6) owns the retry ladder and the circuit breaker.

Depends on L1 only: `config`, `types`, `data_guard`, `scoring`.
"""
from __future__ import annotations

import ast
import importlib
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C
from .data_guard import DataAPI
from .types import FailureClass, FailureRecord, ResourceFacts, RunResult, Split

# ---------------------------------------------------------------------------------
# The candidate script contract.
#
# Stated here, once, because three layers depend on it agreeing: the executor builds
# the argv, the lint decides what a candidate is allowed to import in order to satisfy
# it, and L5 pastes this text into the prompt. Two copies of a CLI contract drift, and
# the failure mode is every candidate failing at argument parsing for a whole run.

CANDIDATE_CONTRACT = """\
A candidate script is invoked as:

    python <candidate.py> --split {train|valid|test} --seed <int> --out <path.npy> \
[--frac <float>]

- It writes a float64 .npy array of scores, one per row of the requested split, in
  the split's canonical row order (row i of the array scores row i of the split).
- It obtains data via `from harness.data_guard import DataAPI`. The repo root is on
  PYTHONPATH; the harness sets it, the script must not.
- `--frac` is used only by the smoke run and means "use this fraction of USERS for
  training". Sample users, not rows: ranking is scored within user, so a row sample
  shreds the impression groups and produces a meaningless run. The script must still
  emit scores for ALL rows of the requested split.
- Exit 0 on success.
"""

# stderr tail handed back to the LLM as a repair hint. 8 KB is roughly a full Python
# traceback with a few frames of context and still small enough to paste into a prompt
# every repair attempt without dominating the token bill.
TAIL_BYTES = 8192

_POLL_INTERVAL_S = 0.1  # how often we check whether the child has exited
_RSS_SAMPLE_INTERVAL_S = 1.0  # the watchdog's 1 Hz sampling rate


# ---------------------------------------------------------------- harness faults


class HarnessDependencyError(RuntimeError):
    """A library the agent is *told* it may use does not import.

    Deliberately not a `FailureRecord`. If torch is missing and the agent wrote torch
    code, the ImportError reads as a broken idea: the agent concludes its hypothesis
    failed, records DISCARD against that technique in the insight ledger, and every
    later prompt carries the false claim -- so a whole research axis dies to one absent
    dependency. Failing loudly as a harness fault at startup, before any candidate
    runs, is the only way that mistake is unavailable to us.
    """


class ProcessGroupEscapeError(RuntimeError):
    """A process survived SIGTERM *and* SIGKILL of its whole group.

    A surviving PID is not a warning. Orphaned DataLoader workers and BLAS pools steal
    cores from the next iteration and corrupt every timing measurement taken after.
    """


_IMPORT_CHECK_CACHE: dict[tuple[str, ...], dict[str, str]] = {}


def check_imports(modules: tuple[str, ...] | None = None) -> dict[str, str]:
    """Import every library the agent is allowed to use, once, at startup.

    Returns `{module: version}` for the report. Raises `HarnessDependencyError` -- a
    harness fault, never a candidate failure -- listing everything that failed. See
    `HarnessDependencyError` for why the distinction is load-bearing.

    Cached: the loop, the executor's constructor and the tests all want this check,
    and importing torch costs a couple of seconds.
    """
    names = tuple(modules) if modules is not None else tuple(C.ALLOWED_IMPORTS) + tuple(
        C.OPTIONAL_IMPORTS
    )
    if names in _IMPORT_CHECK_CACHE:
        return _IMPORT_CHECK_CACHE[names]

    versions: dict[str, str] = {}
    broken: list[str] = []
    for name in names:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # ImportError, but a broken install raises anything
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        versions[name] = str(getattr(mod, "__version__", "stdlib"))

    if broken:
        raise HarnessDependencyError(
            "library preflight failed -- these are stated verbatim in the agent's "
            "prompt as available, so a candidate importing one is CORRECT code that "
            "would be misread as a broken idea:\n  " + "\n  ".join(broken)
        )
    _IMPORT_CHECK_CACHE[names] = versions
    return versions


# ---------------------------------------------------------------- signatures


_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_PATH_RE = re.compile(r"(?:/[^/\s'\"<>()]+)+")
_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Strip everything that varies between two occurrences of the same bug.

    Order matters: addresses before digits (`0x7f8a` must not become `0xN` piecewise),
    and paths before digits (so `/Users/.../iter_07/candidate.py` collapses to
    `candidate.py` rather than to `/N/N/N`).
    """
    text = _PATH_RE.sub(lambda m: m.group(0).rsplit("/", 1)[-1], text)
    text = _HEX_RE.sub("0xADDR", text)
    text = _DIGITS_RE.sub("N", text)
    return _WS_RE.sub(" ", text).strip()


def failure_signature(exc_type: str, message: str, frame: str = "") -> str:
    """The dedupe key: `(type, normalised message, frame)`.

    Line numbers, memory addresses, tensor shapes and absolute paths all move between
    two runs of the same broken idea. Without normalisation the loop sees three
    distinct failures, spends all three self-heal attempts, and converts an iteration
    into nothing -- which is the single most common way an agent harness burns its
    whole budget.
    """
    return f"{exc_type}|{_normalise(message)}|{_normalise(frame)}"[:400]


_TB_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<fn>.+)$')
_TB_EXC_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Warning|Exit))"
                        r"(?::\s*(?P<msg>.*))?$")


def parse_traceback(text: str) -> tuple[str, str, str]:
    """Pull `(exception type, message, deepest frame)` out of a stderr tail.

    Best-effort by design: generated code prints all sorts of things to stderr, and a
    signature built from a slightly odd traceback is still a usable dedupe key. Returns
    `("Unknown", <last non-empty line>, "")` when nothing parses.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ("Unknown", "", "")

    frame = ""
    for ln in lines:
        m = _TB_FRAME_RE.match(ln)
        if m:  # keep scanning: the last one is the deepest frame, where the bug is
            frame = f"{Path(m.group('file')).name}:{m.group('line')}:{m.group('fn')}"

    for ln in reversed(lines):
        m = _TB_EXC_RE.match(ln.strip())
        if m:
            return (m.group("type"), (m.group("msg") or "").strip(), frame)
    return ("Unknown", lines[-1].strip(), frame)


def _tail(path: Path, n_bytes: int = TAIL_BYTES) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if size > n_bytes:
            fh.seek(size - n_bytes)
        raw = fh.read()
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- stage 1: syntax


def check_syntax(code: str) -> FailureRecord | None:
    """`ast.parse` only. The cheapest stage, and it must run before any subprocess:
    launching an interpreter to be told about a missing colon costs a second and a
    process group for information a parse gives us for free."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        where = f"line {exc.lineno}, col {exc.offset}"
        detail = f"{exc.msg} ({where})"
        text = (exc.text or "").rstrip()
        return FailureRecord(
            cls=FailureClass.SYNTAX,
            signature=failure_signature("SyntaxError", exc.msg or "", f"line {exc.lineno}"),
            traceback_tail=f"SyntaxError: {detail}\n    {text}",
            frame_context=text or None,
        )
    return None


# ---------------------------------------------------------------- stage 2: lint


# Network egress. Every one of these is already outside ALLOWED_IMPORTS, so the
# allowlist rule would catch them -- but the message is what goes back to the LLM, and
# "no network access" is a repairable instruction while "not on the allowlist" invites
# the model to try `http.client` next.
_NETWORK_MODULES = frozenset(
    {"socket", "urllib", "urllib3", "requests", "http", "httpx", "aiohttp", "ftplib",
     "smtplib", "telnetlib", "xmlrpc", "webbrowser", "ssl", "asyncio"}
)

# The leakage vocabulary. `harness.holdout` is the only module that can produce a test
# label, and these are the names by which it and its output are reachable.
_FORBIDDEN_NAMES = frozenset({"holdout", "extract_test_labels", "test_labels", "_holdout"})

# Attributes that resolve to the raw CSV tree, which the cache exists to keep the
# agent away from. The cache physically lacks the test labels; the raw log does not.
_FORBIDDEN_ATTRS = frozenset({"DATA_DIR", "LOG_FILES", "RANDOM_LOG_FILE", "STARTER_KIT"})

# Shelling out is banned outright, by call target rather than by argument.
#
# `os` is on the allowlist (candidates need os.environ, os.path), and it carries a
# whole subprocess family with it. Inspecting the arguments does not work: a literal
# path is caught, but `os.system('cat ' + d + '/x')` with a computed `d` sails
# through, and no amount of string analysis fixes that in general. Closing the door
# costs nothing, because a candidate loads data through DataAPI and writes a .npy --
# it has no legitimate reason to start a process at all.
_FORBIDDEN_CALLS = frozenset({
    "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe", "execl", "execle",
    "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "fork", "forkpty", "posix_spawn", "posix_spawnp",
})

# `harness` is not a third-party import, it is the data path we hand the agent. The
# submodule is restricted: data_guard is the API, holdout is the thing it exists to
# make unreachable (and is caught by _FORBIDDEN_NAMES anyway).
_ALLOWED_HARNESS_SUBMODULES = frozenset({"data_guard", "config", "types", "scoring"})

_RAW_DIR_TOKEN = C.DATA_DIR.parent.name.lower()  # "kuairand-pure"


class _ContractVisitor(ast.NodeVisitor):
    """Walks the AST once and records every contract violation it can see.

    AST rather than substring matching, deliberately: `# never read test_labels` in a
    comment is not a violation, `import requests  # unused` is, and a substring check
    gets both backwards. Comments are not in the tree at all, which is the property we
    want.
    """

    def __init__(self) -> None:
        self.violations: list[tuple[str, str]] = []  # (rule, detail)

    def _flag(self, rule: str, detail: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", 0)
        self.violations.append((rule, f"line {line}: {detail}"))

    # -- imports

    def _check_module(self, dotted: str, node: ast.AST) -> None:
        top = dotted.split(".")[0]
        if top == "harness":
            sub = dotted.split(".")[1] if "." in dotted else ""
            if sub and sub not in _ALLOWED_HARNESS_SUBMODULES:
                self._flag(
                    "harness-submodule",
                    f"`{dotted}` is not part of the agent-facing API; the only data "
                    f"path is `from harness.data_guard import DataAPI`",
                    node,
                )
            return
        if top in _NETWORK_MODULES:
            self._flag(
                "no-network",
                f"`{dotted}` is network access; the run is offline and reproducible, "
                f"so no candidate may open a socket",
                node,
            )
            return
        if top not in C.ALLOWED_IMPORTS and top not in C.OPTIONAL_IMPORTS:
            self._flag(
                "import-allowlist",
                f"`{dotted}` is not on the allowed import list. Available: "
                f"{', '.join(C.ALLOWED_IMPORTS + C.OPTIONAL_IMPORTS)}",
                node,
            )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:  # `from . import x` -- relative to nothing we control
            self._flag("import-allowlist", "relative imports have no meaning in a "
                                           "standalone candidate script", node)
        elif node.module:
            self._check_module(node.module, node)
            for alias in node.names:
                if alias.name in _FORBIDDEN_NAMES:
                    self._flag(
                        "no-test-labels",
                        f"`{alias.name}` reaches the hidden test labels; the agent "
                        f"never sees them. Produce scores and let the harness score.",
                        node,
                    )
        self.generic_visit(node)

    # -- names and attributes

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            self._flag("no-test-labels", f"reference to `{node.id}`", node)
        elif node.id in _FORBIDDEN_ATTRS:
            self._flag(
                "no-raw-data",
                f"`{node.id}` resolves to the raw CSV tree. Read data through "
                f"`DataAPI`; the cache it reads is the copy without test labels.",
                node,
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _FORBIDDEN_CALLS:
            self._flag(
                "no-subprocess",
                f"`{node.attr}` starts a process. Candidates load data through "
                f"DataAPI and write a .npy; they never shell out, and allowing it "
                f"would make every other lint rule bypassable with a computed string",
                node,
            )
        elif node.attr in _FORBIDDEN_NAMES:
            self._flag("no-test-labels", f"attribute access `.{node.attr}`", node)
        elif node.attr in _FORBIDDEN_ATTRS:
            self._flag(
                "no-raw-data",
                f"`.{node.attr}` resolves to the raw CSV tree. Read data through "
                f"`DataAPI`; the cache it reads is the copy without test labels.",
                node,
            )
        self.generic_visit(node)

    # -- string literals and open()

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            low = node.value.lower()
            if "holdout" in low:
                self._flag("no-test-labels", f"path literal {node.value!r}", node)
            elif _RAW_DIR_TOKEN in low:
                self._flag(
                    "no-raw-data",
                    f"literal {node.value!r} names the raw data directory",
                    node,
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name == "open":
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    low = arg.value.lower()
                    if _RAW_DIR_TOKEN in low or low.endswith(".csv"):
                        self._flag(
                            "no-raw-data",
                            f"open({arg.value!r}): the raw CSVs carry the test labels "
                            f"and the outcome columns. Use `DataAPI`.",
                            node,
                        )
        self.generic_visit(node)


def lint_contract(code: str) -> FailureRecord | None:
    """Static contract lint. Returns `None` when the candidate is clean.

    A rejection is free -- it costs no iteration -- so the message is written as a
    repair hint: it names the rule, the line, and what to do instead, because the whole
    text goes straight back to the LLM.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # check_syntax owns this and runs first. Refusing to guess here keeps one
        # failure class per stage.
        return None

    visitor = _ContractVisitor()
    visitor.visit(tree)
    if not visitor.violations:
        return None

    rules = sorted({rule for rule, _ in visitor.violations})
    body = "\n".join(f"  [{rule}] {detail}" for rule, detail in visitor.violations)
    return FailureRecord(
        cls=FailureClass.CONTRACT,
        signature=failure_signature("ContractViolation", ",".join(rules)),
        traceback_tail=(
            f"contract violation ({len(visitor.violations)} in "
            f"{len(rules)} rule(s): {', '.join(rules)}):\n{body}"
        ),
        frame_context=visitor.violations[0][1],
    )


# ---------------------------------------------------------------- process groups


def _group_snapshot(pgid: int) -> list[tuple[int, int, str]]:
    """`[(pid, rss_bytes, stat)]` for every live process in the group.

    `ps` rather than psutil: one fewer dependency in the layer that has to keep working
    when everything else is on fire, and `ps -A -o pgid=,...` is stable on both macOS
    and Linux. Filtering the full table by pgid rather than using `ps -g` avoids the
    BSD/Linux disagreement about what `-g` selects (process group vs. effective group).

    Zombies are excluded: a killed direct child stays in the table as `<defunct>` until
    `Popen.wait()` reaps it, and counting that as a survivor would make every
    termination look like an escape.
    """
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pgid=,pid=,rss=,stat="],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            g, pid, rss_kb = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if g != pgid or parts[3].startswith("Z"):
            continue
        rows.append((pid, rss_kb * 1024, parts[3]))
    return rows


def group_pids(pgid: int) -> list[int]:
    """Live, non-zombie PIDs in the group. Public so the tests can assert emptiness."""
    return [pid for pid, _, _ in _group_snapshot(pgid)]


def _group_rss_bytes(pgid: int) -> int:
    return sum(rss for _, rss, _ in _group_snapshot(pgid))


def _terminate_group(pgid: int, proc: subprocess.Popen, grace: float = C.KILL_GRACE_S) -> None:
    """SIGTERM the whole group, wait `grace`, SIGKILL, then VERIFY the group is empty.

    `proc.kill()` signals only the direct child. PyTorch DataLoader workers and BLAS
    thread pools survive it, and then steal cores from the next iteration -- which
    corrupts every wall-clock number the harness reports afterwards, silently. Killing
    the group is the only version of this that actually works.
    """
    if pgid in (0, os.getpgrp()):
        # Would signal the harness itself. Should be impossible (start_new_session
        # gives the child its own group) but the consequence is total, so it is checked.
        raise ProcessGroupEscapeError(f"refusing to signal the harness's own group {pgid}")

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            proc.poll()  # reap the direct child so it stops showing up as a zombie
            if not group_pids(pgid):
                break
            time.sleep(0.1)
        if not group_pids(pgid):
            break

    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass

    survivors = group_pids(pgid)
    if survivors:
        raise ProcessGroupEscapeError(
            f"process group {pgid} still holds {survivors} after SIGTERM+SIGKILL; "
            f"orphans steal cores and corrupt every later timing measurement"
        )


# ---------------------------------------------------------------- the executor


class Executor:
    """Stages 1-5. Classifies; never decides. See the module docstring."""

    def __init__(
        self,
        workspace: Path | None = None,
        data: DataAPI | None = None,
        python_bin: str = C.PYTHON_BIN,
        run_timeout_s: float = C.RUN_TIMEOUT_S,
        smoke_timeout_s: float = C.SMOKE_TIMEOUT_S,
        rss_cap_bytes: int = C.RSS_CAP_BYTES,
        preflight_imports: bool = True,
    ):
        if preflight_imports:
            # Before any candidate runs, so a missing dependency can never be attributed
            # to a candidate. Cached, so paying it in the constructor is nearly free.
            check_imports()
        self.workspace = Path(workspace) if workspace else C.RUNS_DIR
        self.python_bin = python_bin
        self.run_timeout_s = run_timeout_s
        self.smoke_timeout_s = smoke_timeout_s
        self.rss_cap_bytes = rss_cap_bytes
        self._data = data

    @property
    def data(self) -> DataAPI:
        """Lazy: linting and syntax checking must not require a built cache."""
        if self._data is None:
            self._data = DataAPI()
        return self._data

    # -- stages 1 and 2 are module functions; bound here so callers hold one object

    check_syntax = staticmethod(check_syntax)
    lint_contract = staticmethod(lint_contract)

    # ------------------------------------------------------------ stages 3 and 4

    def smoke(
        self,
        code: str | Path,
        node_id: str,
        split: Split = "valid",
        seed: int = C.CONFIRM_SEEDS[0],
    ) -> RunResult:
        """1% of USERS, `SMOKE_TIMEOUT_S`. The highest-ROI stage in the build.

        Sampled by user, not by row. Ranking is scored within user, so a 1% row sample
        shreds the impression groups: the run then fails, or scores meaninglessly, for
        reasons that have nothing to do with the candidate's logic -- and the agent
        spends a repair attempt on a bug the harness invented.

        Validation is structural only here (file present, finite, right length).
        Within-user variance is a silent-wrong check and belongs at full scale: on 1%
        of users a legitimate model can collapse for reasons that say nothing about
        the code.
        """
        return self._run(
            code, node_id, split, seed,
            frac=C.SMOKE_FRACTION,
            timeout_s=self.smoke_timeout_s,
            tag="smoke",
            failure_class=FailureClass.SMOKE,
            structural_only=True,
        )

    def run(
        self,
        code: str | Path,
        node_id: str,
        split: Split = "valid",
        seed: int = C.CONFIRM_SEEDS[0],
        timeout_s: float | None = None,
    ) -> RunResult:
        """The full supervised run, with output validation."""
        return self._run(
            code, node_id, split, seed,
            frac=None,
            timeout_s=self.run_timeout_s if timeout_s is None else timeout_s,
            tag=f"{split}_seed{seed}",
            failure_class=None,
            structural_only=False,
        )

    # ------------------------------------------------------------ internals

    @staticmethod
    def _source(code: str | Path) -> str:
        """Accept either source text or the path of a stored candidate.

        P2's `Runner` protocol is `Callable[[str, str, Split, int], RunResult]` with a
        *code_path* first argument (`Evaluator.build_submission` passes
        `node.code_path`), while P3's contract names the same parameter `code: str` and
        means source. Rather than have the loop remember which is which, both are
        accepted: a single-line string naming an existing `.py` file is a path,
        anything else is source.
        """
        if isinstance(code, Path):
            return code.read_text()
        if "\n" not in code and code.endswith(".py"):
            p = Path(code)
            if p.exists():
                return p.read_text()
        return code

    def _child_env(self) -> dict[str, str]:
        """`CHILD_ENV` plus `PYTHONPATH`, layered over the inherited environment.

        Inherited rather than empty: torch wants HOME for its cache and the interpreter
        wants a sane locale, and a candidate dying on a missing HOME is a harness bug
        that reads as a candidate bug. What matters is that the reproducibility keys --
        PYTHONHASHSEED and the four BLAS thread caps -- are written *after* the copy, so
        generated code cannot make its own run irreproducible or oversubscribe the box.
        """
        env = dict(os.environ)
        env.update(C.CHILD_ENV)
        env["PYTHONPATH"] = str(C.ROOT)
        return env

    def _run(
        self,
        code: str | Path,
        node_id: str,
        split: Split,
        seed: int,
        frac: float | None,
        timeout_s: float,
        tag: str,
        failure_class: FailureClass | None,
        structural_only: bool,
    ) -> RunResult:
        workdir = self.workspace / node_id / tag
        workdir.mkdir(parents=True, exist_ok=True)
        script = workdir / "candidate.py"
        script.write_text(self._source(code))
        scores_path = workdir / "scores.npy"
        stdout_path = workdir / "stdout.log"
        stderr_path = workdir / "stderr.log"
        if scores_path.exists():
            scores_path.unlink()  # a stale file from a previous attempt would validate

        argv = [self.python_bin, str(script), "--split", split, "--seed", str(seed),
                "--out", str(scores_path)]
        if frac is not None:
            argv += ["--frac", str(frac)]

        # stdout/stderr to FILES, never PIPE. Two independent reasons, both fatal:
        # (1) polling a pipe while the child keeps writing deadlocks once the ~64 KB
        #     OS buffer fills -- the child blocks on write, we block on wait;
        # (2) killing a piped process discards whatever is still buffered, so the
        #     timeout case -- exactly when the traceback matters most -- is the case
        #     that comes back empty.
        killed_by: str | None = None
        peak_rss = 0
        started = time.monotonic()
        with open(stdout_path, "wb") as out_fh, open(stderr_path, "wb") as err_fh:
            proc = subprocess.Popen(
                argv,
                cwd=str(workdir),
                env=self._child_env(),
                stdout=out_fh,
                stderr=err_fh,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # the child leads its own process group
            )
            try:
                pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pgid = proc.pid  # exited already; start_new_session made pgid == pid

            # BaseException, not Exception: the case this exists for is Ctrl-C. The
            # child leads its own session, so an interrupt that unwinds past here
            # leaves it running -- burning a core, holding the cache open, and
            # corrupting every wall-clock number the harness reports afterwards. The
            # loop advertises that Ctrl-C is safe and resumable; this is what makes
            # that true rather than aspirational.
            try:
                next_sample = 0.0
                while proc.poll() is None:
                    now = time.monotonic()
                    if now >= next_sample:
                        next_sample = now + _RSS_SAMPLE_INTERVAL_S
                        rss = _group_rss_bytes(pgid)
                        peak_rss = max(peak_rss, rss)
                        if rss > self.rss_cap_bytes:
                            killed_by = "rss"
                            break
                    if now - started > timeout_s:
                        killed_by = "timeout"
                        break
                    time.sleep(_POLL_INTERVAL_S)
            except BaseException:
                killed_by = "interrupt"
                _terminate_group(pgid, proc)
                raise

            if killed_by:
                _terminate_group(pgid, proc)
            else:
                proc.wait()

        wall = time.monotonic() - started
        resources = ResourceFacts(
            wall_seconds=wall,
            peak_rss_bytes=peak_rss,
            exit_code=proc.returncode,
            killed_by=killed_by,
        )

        failure = self._classify(
            killed_by, proc.returncode, stderr_path, scores_path, split,
            resources, failure_class, structural_only,
        )
        return RunResult(
            ok=failure is None,
            scores_path=scores_path if failure is None else None,
            resources=resources,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            failure=failure,
        )

    def _classify(
        self,
        killed_by: str | None,
        exit_code: int | None,
        stderr_path: Path,
        scores_path: Path,
        split: Split,
        resources: ResourceFacts,
        failure_class: FailureClass | None,
        structural_only: bool,
    ) -> FailureRecord | None:
        """Turn "what happened" into a `FailureClass`. Decides nothing else.

        `failure_class` overrides the classification for the smoke stage: any smoke
        failure is a SMOKE failure, because the stage -- not the symptom -- is what
        determines whether the loop may repair for free. The underlying symptom is not
        lost: `resources.killed_by` still says whether it was a timeout.
        """
        tail = _tail(stderr_path)

        def record(cls: FailureClass, exc: str, msg: str, frame: str = "", body: str = "") -> FailureRecord:
            return FailureRecord(
                cls=failure_class or cls,
                signature=failure_signature(exc, msg, frame),
                traceback_tail=body or f"{exc}: {msg}\n{tail}",
                frame_context=frame or None,
            )

        if killed_by == "timeout":
            return record(
                FailureClass.TIMEOUT, "Timeout",
                f"killed after {resources.wall_seconds:.1f}s "
                f"(peak RSS {resources.peak_rss_bytes / 1024**3:.2f} GiB)",
                body=(f"TIMEOUT after {resources.wall_seconds:.1f}s; peak RSS "
                      f"{resources.peak_rss_bytes / 1024**3:.2f} GiB\n--- stderr tail ---\n{tail}"),
            )
        if killed_by == "rss":
            return record(
                FailureClass.OOM, "MemoryCap",
                f"peak RSS {resources.peak_rss_bytes / 1024**3:.2f} GiB exceeded the "
                f"{self.rss_cap_bytes / 1024**3:.1f} GiB cap",
                body=(f"OOM: peak RSS {resources.peak_rss_bytes / 1024**3:.2f} GiB over the "
                      f"{self.rss_cap_bytes / 1024**3:.1f} GiB cap after "
                      f"{resources.wall_seconds:.1f}s\n--- stderr tail ---\n{tail}"),
            )
        if exit_code != 0:
            exc, msg, frame = parse_traceback(tail)
            return record(FailureClass.RUNTIME, exc, msg or f"exit code {exit_code}", frame,
                          body=f"exit code {exit_code}\n--- stderr tail ---\n{tail}")

        invalid = self.validate_output(scores_path, split, structural_only=structural_only)
        if invalid is not None and failure_class is not None:
            # Smoke: keep the diagnosis, reclassify the stage.
            return FailureRecord(
                cls=failure_class,
                signature=invalid.signature,
                traceback_tail=invalid.traceback_tail,
                frame_context=invalid.frame_context,
            )
        return invalid

    # ------------------------------------------------------------ stage 5

    def validate_output(
        self,
        scores_path: Path,
        split: Split,
        structural_only: bool = False,
    ) -> FailureRecord | None:
        """What separates "it ran" from "it worked".

        The last check is the one that matters. A candidate emitting a constant score
        exits 0, produces a correctly-shaped array of the right dtype, and is worthless:
        the metric is computed *within* user, so a model that is constant inside each
        user has no ranking at all. That failure is invisible to every other stage.
        """
        scores_path = Path(scores_path)

        def bad(rule: str, detail: str) -> FailureRecord:
            return FailureRecord(
                cls=FailureClass.INVALID_OUTPUT,
                signature=failure_signature("InvalidOutput", rule),
                traceback_tail=f"invalid output [{rule}]: {detail}",
                frame_context=rule,
            )

        if not scores_path.exists():
            return bad("missing", f"no scores written to {scores_path.name}; the script "
                                  f"must np.save() its scores to the --out path")
        try:
            scores = np.load(scores_path, allow_pickle=False)
        except Exception as exc:
            return bad("unreadable", f"{type(exc).__name__}: {exc}")

        if scores.ndim != 1:
            return bad("shape", f"expected a 1-D array, got shape {scores.shape}")
        if not (np.issubdtype(scores.dtype, np.floating)
                or np.issubdtype(scores.dtype, np.integer)):
            return bad("dtype", f"expected float64 scores, got dtype {scores.dtype}")

        expected = self.data.n_rows(split)
        if scores.shape[0] != expected:
            return bad(
                "row-count",
                f"{scores.shape[0]} scores for {expected} rows of {split!r}. Scores are "
                f"positional: row i of the array must score row i of the split, so a "
                f"length mismatch means every score is misaligned, not just the extras.",
            )

        scores = scores.astype(np.float64, copy=False)
        finite = np.isfinite(scores)
        if not finite.all():
            n_bad = int((~finite).sum())
            first = int(np.flatnonzero(~finite)[0])
            return bad(
                "non-finite",
                f"{n_bad} of {scores.shape[0]} scores are NaN or Inf (first at row "
                f"{first}). The metric sorts within user; NaN sorts unpredictably and "
                f"silently changes the ranking rather than raising.",
            )

        if structural_only:
            return None

        groups = self.data.groups(split)
        n_multi, n_varying = _within_group_variation(groups, scores)
        if n_multi and not n_varying:
            return bad(
                "constant-within-user",
                f"scores are constant within every one of the {n_multi} users that have "
                f"more than one row. GAUC and nDCG@5 are computed within user, so this "
                f"model expresses no ranking at all -- it 'runs' perfectly and scores "
                f"like a coin flip.",
            )
        return None


def _within_group_variation(groups: np.ndarray, scores: np.ndarray) -> tuple[int, int]:
    """`(multi-row groups, of those, groups whose scores are not all equal)`.

    Exact min/max per group via a stable sort and `reduceat`, rather than a
    sum-of-squares variance: a float variance of a constant array is ~1e-16 rather than
    0, which would need a tolerance, and a tolerance here is a way to accidentally
    accept a nearly-constant model.
    """
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    s = scores[order]
    starts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
    sizes = np.diff(np.concatenate((starts, [g.shape[0]])))
    hi = np.maximum.reduceat(s, starts)
    lo = np.minimum.reduceat(s, starts)
    multi = sizes > 1
    return int(multi.sum()), int(((hi > lo) & multi).sum())


if __name__ == "__main__":  # pragma: no cover
    versions = check_imports()
    print(f"library preflight OK: {len(versions)} modules", file=sys.stderr)
    for name, version in versions.items():
        print(f"  {name:<14} {version}")
