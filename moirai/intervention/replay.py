"""Replay a trajectory prefix inside the task's container and decide whether the
resulting state can be trusted as an intervention checkpoint.

This is the instrument for the "verified full replay" policy. Every recorded
tool call before the anchor is re-executed in a fresh container from the
published per-task image, after the repository has been pinned to the
instance's base commit (and optionally the hidden test patch). Each replayed
observation is compared with the recorded one and judged by an explicit
acceptance rule; an anchor is accepted only when the baseline probe passed and
every executed step was accepted. Mismatches are recorded, never coerced.

Emulated tools follow the OpenHands scaffold: ``execute_bash`` in a persistent
shell (cwd and exported variables persist across calls) and ``str_replace_editor``
(view / create / str_replace / insert, with OpenHands' LF-normalising reads).
``think`` has no environment effect.
"""
from __future__ import annotations

import difflib
import json
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from moirai.intervention.eligibility import classify_action
from moirai.intervention.raw import index_events
from moirai.intervention.schema import RawEvent, RawTrajectory

OBSERVATION_CAP = 200_000


class ContainerError(RuntimeError):
    pass


def instrument_hash() -> str:
    import hashlib
    import inspect
    import sys

    return hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode("utf-8")).hexdigest()[:16]


# ── Recorded observation parsing ──────────────────────────────────

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_EXIT_RE = re.compile(r"\[(?:The command completed with exit code|Command finished with exit code) (-?\d+)\.?(?: CTRL\+C was sent\.)?\]")
_TIMEOUT_NOTICE = "[The command has no new output after"
_NOT_EXECUTED = "is NOT executed. The previous command is still running"
_PREVIOUS_OUTPUT = "[Below is the output of the previous command.]"
_TRUNCATED = "[... Observation truncated due to length ...]"
_CLIPPED = "<response clipped>"
_PYTEST_SUMMARY = re.compile(r"(?:=+ (.*?) in [\d.]+s(?: \([^)]*\))? =+|^((?:\d+ \w+(?:, )?)+) in [\d.]+s$)", re.M)   # banner or `-q` bare line
_PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)")
_PYTEST_LINE = re.compile(r"^(PASSED|FAILED|ERROR) (\S+)|^(\S+::\S+) (PASSED|FAILED|ERROR)\b", re.M)
_SUGAR_RESULTS = re.compile(r"^Results \([\d.]+s\):\s*$((?:\n\s+\d+ \w+.*)+)", re.M)
_SUGAR_LINE = re.compile(r"^\s*(\S+::\S+?)\s+([✓⨯xs])\s*(?:\d+% )?", re.M)

_NOISE = [
    (_ANSI, ""),
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "0xADDR"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "TIMESTAMP"),
    (re.compile(r"\b\d+\.\d+s\b"), "DURATION"),
    (re.compile(r"\bin \d+\.\d+ seconds\b"), "in DURATION"),
    (re.compile(r"'elapsed': \d+\.\d+"), "'elapsed': DURATION"),
    (re.compile(r"/tmp/[A-Za-z0-9_./-]+"), "/tmp/PATH"),
    (re.compile(r"\b[A-Z][a-z]{2} +\d{1,2} +(?:\d{2}:\d{2}|\d{4})\b"), "DATE"),
    (re.compile(r"^(d[rwx-]{9} +\d+ +\S+ +\S+) +\d+ +(DATE +\.\.?)$", re.M), r"\g<1> SIZE \g<2>"),
    (re.compile(r"^[0-9a-f]{7,12}(?= )", re.M), "HASH"),
    (re.compile(r"^(.*?)\s*\d+%\|.*$", re.M), r"\g<1>PROGRESS"),   # tqdm bars: width and captured percentage are render artefacts
    (re.compile(r"^([-dlcbps][rwxsStT-]{9}[+@.]?) +(\d+) +(\S+) +(\S+) +(\S+) +", re.M), r"\1 \2 \3 \4 \5 "),   # ls -l pads columns to the widest entry
    (re.compile(r"\bpid \d+\b", re.I), "pid N"),
    (re.compile(r"\[Command finished with exit code -?\d+\]"), ""),
    (re.compile(r"\[The command completed with exit code -?\d+\.?( CTRL\+C was sent\.)?\]"), ""),
    (re.compile(r"\[Current working directory: [^\]]*\]"), ""),
    (re.compile(r"\[The command has no new output after \d+ seconds\.[^\]]*\]"), ""),
    (re.compile(r"\[Your command \"[^\n]*?\" is NOT executed\.[^\]]*\]"), ""),
    (re.compile(r"\[Below is the output of the previous command\.\]"), ""),
    (re.compile(r"\[Python interpreter: [^\]]*\]"), ""),
    (re.compile(r"={5,}"), "===="),
    (re.compile(r"-{5,}"), "----"),
    (re.compile(r"\r"), ""),
    (re.compile(r"[ \t]+\n"), "\n"),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text or "")


def render_terminal(text: str) -> str:
    """Resolve carriage-return overwrites the way a captured tmux pane shows them.

    OpenHands records the rendered screen (``tmux capture-pane -J``), so progress
    lines such as ``collecting ...`` are overwritten by their final form. The pty
    stream keeps every intermediate segment; this collapses them.
    """
    out_lines = []
    for line in strip_ansi(text or "").split("\n"):
        if "\r" not in line:
            out_lines.append(line)
            continue
        buf = ""
        for seg in line.split("\r"):
            if not seg:
                continue
            buf = seg + buf[len(seg):] if len(seg) < len(buf) else seg
        out_lines.append(buf)
    return "\n".join(out_lines)


def parse_exit_code(observation: str) -> int | None:
    """Exit code from the last completion footer in the observation."""
    codes = _EXIT_RE.findall(observation or "")
    return int(codes[-1]) if codes else None


def hidden_footer(path: str, hidden_count: int) -> str:
    if hidden_count <= 0:
        return ""
    return f"\n{hidden_count} hidden files/directories in this directory are excluded. You can use 'ls -la {path}' to see them.\n"


_OH_EXEC_PIPE = re.compile(r"\\;\s*\|")


def openhands_bash_rejection(command: str) -> str | None:
    """Observation OpenHands' bash session produced without running the command, if any.

    The recording harness split commands with bashlex; ``find ... -exec ... {} \\; | ...``
    was reconstructed into a form bash rejected. Observed identically on three tasks.
    """
    if _OH_EXEC_PIPE.search(command):
        return "bash: syntax error near unexpected token `|'"
    return None


def group_timeout_steps(prefix: list[RawEvent]) -> dict[int, list[int]]:
    """Map a soft-timed-out command's index -> indices of its recorded follow-ups.

    OpenHands records a command that exceeds the soft timeout as a notice, then
    the agent's waits (``is_input`` with an empty command), interrupts (``C-c``)
    and any commands it tried that were NOT executed, until a completion footer
    appears. Those follow-ups carry no independent environment effect; the
    replay runs the head command once and compares against the merged output.
    """
    groups: dict[int, list[int]] = {}
    i = 0
    while i < len(prefix):
        ev = prefix[i]
        obs = ev.observation or ""
        args = ev.call.arguments or {}
        if ev.call.tool_name == "execute_bash" and not args.get("is_input") and recorded_flags(obs)["timeout_notice"] and parse_exit_code(obs) is None:
            members: list[int] = []
            j = i + 1
            while j < len(prefix):
                nxt = prefix[j]
                nobs = nxt.observation or ""
                nargs = nxt.call.arguments or {}
                flags = recorded_flags(nobs)
                if nxt.call.tool_name == "execute_bash" and (nargs.get("is_input") or flags["not_executed"] or flags["previous_output"]):
                    members.append(j)
                    j += 1
                    if parse_exit_code(nobs) is not None and not flags["not_executed"]:
                        break
                    continue
                break
            groups[i] = members
            i = j
            continue
        i += 1
    return groups


def recorded_flags(observation: str) -> dict[str, bool]:
    o = observation or ""
    return {
        "timeout_notice": _TIMEOUT_NOTICE in o,
        "not_executed": _NOT_EXECUTED in o,
        "previous_output": _PREVIOUS_OUTPUT in o,
        "truncated": _TRUNCATED in o or _CLIPPED in o,
    }


def _truncation_cut(text: str) -> int:
    """Index where the recorded observation was cut by the harness, or -1."""
    cuts = [i for i in (text.find(_TRUNCATED), text.find(_CLIPPED)) if i >= 0]
    return min(cuts) if cuts else -1


def strip_command_echo(recorded: str, command: str) -> str:
    """OpenHands echoes the command (possibly multi-line) at the top of a bash observation."""
    cmd_lines = [l.strip() for l in command.strip().split("\n") if l.strip()]
    rec_lines = recorded.split("\n")
    i = k = 0
    while k < len(cmd_lines) and i < len(rec_lines):
        if not rec_lines[i].strip():
            i += 1
            continue
        if rec_lines[i].strip() != cmd_lines[k]:
            break
        i += 1
        k += 1
    if cmd_lines and k == len(cmd_lines):
        return "\n".join(rec_lines[i:])
    return recorded


def pytest_outcome(observation: str) -> dict | None:
    """Counts from the pytest summary line plus per-test verdicts when present."""
    o = strip_ansi(observation or "")
    m = None
    for m in _PYTEST_SUMMARY.finditer(o):
        pass
    if m is None:
        sm = None
        for sm in _SUGAR_RESULTS.finditer(o):
            pass
        if sm is None:
            return None
        counts = {k: int(n) for n, k in _PYTEST_COUNT.findall(sm.group(1))}
        counts = {("error" if k.startswith("error") else k): v for k, v in counts.items() if not k.startswith("warning")}
        sym = {"✓": "PASSED", "⨯": "FAILED", "x": "FAILED", "s": "SKIPPED"}
        verdicts = {name: sym[v] for name, v in _SUGAR_LINE.findall(o)}
        return {"counts": counts, "verdicts": verdicts}
    counts = {k: int(n) for n, k in _PYTEST_COUNT.findall(m.group(1) or m.group(2) or "")}
    counts = {("error" if k.startswith("error") else ("warning" if k.startswith("warning") else k)): v for k, v in counts.items()}
    counts.pop("warning", None)
    verdicts = {}
    for v1, n1, n2, v2 in _PYTEST_LINE.findall(o):
        if v1:
            verdicts[n1] = v1
        else:
            verdicts[n2] = v2
    return {"counts": counts, "verdicts": verdicts}


_NUMBERED = re.compile(r"^\s*(\d+)\t(.*)$", re.M)


def numbered_lines(observation: str) -> dict[int, str]:
    """``cat -n`` style lines of an editor observation, keyed by line number."""
    return {int(n): text for n, text in _NUMBERED.findall(render_terminal(observation or ""))}


def similarity_status(recorded: str, replayed: str, order_free: bool,
                      match_threshold: float = 0.95, near_threshold: float = 0.80) -> str:
    """match / near / mismatch from normalised similarity; recomputed on re-judge."""
    sim = similarity(recorded, replayed, order_insensitive=order_free)
    return "match" if sim >= match_threshold else ("near" if sim >= near_threshold else "mismatch")


def failing_tests(outcome: dict | None) -> frozenset[str]:
    """Names of tests that did not pass; the state-relevant part of a pytest run."""
    if not outcome:
        return frozenset()
    return frozenset(k for k, v in outcome["verdicts"].items() if v != "PASSED")


def normalize_observation(text: str, limit: int = OBSERVATION_CAP) -> str:
    # tmux renders tabs to spaces at 8-column stops; the pty stream keeps the tab bytes
    t = "\n".join(l.expandtabs(8) for l in render_terminal(text or "").split("\n"))
    for rx, rep in _NOISE:
        t = rx.sub(rep, t)
    return t.strip()[:limit]


def similarity(a: str, b: str, order_insensitive: bool = False) -> float:
    na, nb = normalize_observation(a), normalize_observation(b)
    cut = _truncation_cut(na)
    if cut > 0:
        na, nb = na[:cut], nb[:cut]
    if order_insensitive:
        na, nb = "\n".join(sorted(na.splitlines())), "\n".join(sorted(nb.splitlines()))
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def truncated_parts_match(recorded: str, replayed: str, order_insensitive: bool = False) -> bool:
    """A recording cut by the harness keeps a head and a tail around the marker.

    The head (minus its partial last line) must be a line-prefix of the replay and
    the tail (minus its partial first line) a line-suffix; order-free classes use
    set containment. Offsets are not compared: normalisation shifts them.
    """
    cut = _truncation_cut(recorded)
    if cut < 0:
        return False
    marker = _TRUNCATED if recorded.find(_TRUNCATED) == cut else _CLIPPED
    def lines(t: str) -> list[str]:
        return [l.rstrip() for l in normalize_observation(t).splitlines() if l.strip()]
    head, tail = lines(recorded[:cut])[:-1], lines(recorded[cut + len(marker):])[1:]
    rep_lines = lines(replayed)
    if not head and not tail:
        return False
    if order_insensitive:
        return set(head) <= set(rep_lines) and set(tail) <= set(rep_lines)
    head_ok = rep_lines[:len(head)] == head
    tail_ok = not tail or rep_lines[-len(tail):] == tail
    return head_ok and tail_ok


def line_diff_empty(a: str, b: str, order_insensitive: bool = False) -> bool:
    la = [l.rstrip() for l in normalize_observation(a).splitlines() if l.strip()]
    lb = [l.rstrip() for l in normalize_observation(b).splitlines() if l.strip()]
    if order_insensitive:
        la, lb = sorted(la), sorted(lb)
    return la == lb


# ── Persistent shell inside a container ──────────────────────────


class PersistentShell:
    """One long-lived ``bash`` inside the container; cwd and env persist."""

    def __init__(self, container: str, init_commands: list[str] | None = None, pty: bool = True) -> None:
        self.container = container
        self.init_commands = init_commands or []
        self.cwd: str | None = None
        self.restarts = 0
        self.pty = pty and self._has_script()
        self._start()

    def _has_script(self) -> bool:
        r = subprocess.run(["docker", "exec", self.container, "sh", "-c", "command -v script >/dev/null && script --version >/dev/null 2>&1"],
                           capture_output=True, timeout=60)
        return r.returncode == 0

    def _start(self) -> None:
        # OpenHands runs commands in a tmux pane, so TTY-sensitive programs (pytest-sugar,
        # tqdm, colour) behave as if interactive. util-linux `script` gives the same pty.
        shell = ["script", "-q", "-f", "-c", "bash --noprofile --norc", "/dev/null"] if self.pty else ["bash", "--noprofile", "--norc"]
        self.proc = subprocess.Popen(
            ["docker", "exec", "-i", "-e", "TERM=xterm", self.container, *shell],   # no COLUMNS/LINES: piped programs must fall back to 80 as in the recording
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._lines: queue.Queue[bytes | None] = queue.Queue()

        def pump() -> None:
            assert self.proc.stdout is not None
            for line in iter(self.proc.stdout.readline, b""):
                self._lines.put(line)
            self._lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        if self.pty:
            # Disable echo so commands are not repeated into the output stream.
            assert self.proc.stdin is not None
            self.proc.stdin.write(b"stty -echo 2>/dev/null; stty cols 1000 rows 1000 2>/dev/null; export PS1=''; bind 'set enable-bracketed-paste off' 2>/dev/null\n")
            self.proc.stdin.flush()
            self.run("true", timeout=30)   # consume the echoed init line before echo is off
        for c in self.init_commands:
            self.run(c, timeout=60)
        if self.pty:
            # Activation scripts reset the prompt; an interactive pty shell would print it into every output.
            # recordings show git output inline (no pager artefacts in any of 25 git calls)
            self.run("export PS1='' PS2='' PROMPT_COMMAND='' GIT_PAGER=cat PAGER=cat", timeout=30)
        if self.cwd:
            self.run(f"cd {shlex.quote(self.cwd)}", timeout=10)

    def run(self, command: str, timeout: float = 120.0) -> tuple[str, int | None, bool]:
        """Returns (output, exit_code, timed_out)."""
        marker = f"__MOIRAI_{uuid.uuid4().hex}__"
        script = f"{command}\nprintf '\\n{marker} %s %s\\n' \"$?\" \"$PWD\"\n"
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(script.encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ContainerError(f"shell died: {e}") from e
        out: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._restart()
                return "".join(out), None, True
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                self._restart()
                return "".join(out), None, True
            if line is None:
                raise ContainerError("shell exited")
            text = line.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "")
            if text.startswith(marker):
                parts = text.strip().split(" ", 2)
                code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
                if len(parts) > 2:
                    self.cwd = parts[2]
                joined = "".join(out)
                return joined[:-1] if joined.endswith("\n") else joined, code, False
            out.append(text)

    def _restart(self) -> None:
        self.restarts += 1
        try:
            self.proc.kill()
        except OSError:
            pass
        subprocess.run(["docker", "exec", self.container, "bash", "-c", "pkill -P 1 -f bash || true"],
                       capture_output=True, timeout=30)
        self._start()

    def close(self) -> None:
        try:
            self.proc.stdin.close()  # type: ignore[union-attr]
            self.proc.kill()
        except OSError:
            pass


# ── Container lifecycle ───────────────────────────────────────────


@dataclass
class ContainerWorkspace:
    image: str
    platform: str = "linux/amd64"
    name: str = ""
    network: str = "none"     # replay must not depend on external services: a recorded step that did fails visibly
    init_commands: list[str] = field(default_factory=lambda: [
        "source /opt/miniconda3/bin/activate testbed 2>/dev/null || source /opt/conda/bin/activate testbed 2>/dev/null || true",
    ])
    shell: PersistentShell | None = None

    def start(self, workdir: str | None = None) -> None:
        self.name = self.name or f"moirai-replay-{uuid.uuid4().hex[:10]}"
        cmd = ["docker", "run", "-d", "--platform", self.platform, "--name", self.name, "--entrypoint", "/bin/sh"]
        if self.network:
            cmd += ["--network", self.network]
        if workdir:
            cmd += ["-w", workdir]
        cmd += [self.image, "-c", "sleep infinity"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise ContainerError(f"docker run failed: {r.stderr.strip()}")
        self.shell = PersistentShell(self.name, self.init_commands)

    def exec_capture(self, argv: list[str], stdin: bytes | None = None, timeout: float = 120) -> tuple[bytes, int]:
        try:
            r = subprocess.run(["docker", "exec", "-i", self.name, *argv], input=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise ContainerError(f"docker exec timed out: {argv[:2]}") from e
        if r.returncode == 126 or (r.returncode == 1 and b"Error response from daemon" in r.stderr):
            raise ContainerError(f"docker exec failed: {r.stderr.decode(errors='replace').strip()[:200]}")
        return r.stdout + r.stderr, r.returncode

    def read_file(self, path: str) -> str | None:
        out, code = self.exec_capture(["cat", path])
        return None if code != 0 else out.decode("utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> None:
        parent = str(Path(path).parent)
        self.exec_capture(["mkdir", "-p", parent])
        _, code = self.exec_capture(["sh", "-c", 'cat > "$1"', "sh", path], stdin=content.encode("utf-8"))
        if code != 0:
            raise ContainerError(f"write failed: {path}")

    def path_kind(self, path: str) -> str:
        out, code = self.exec_capture(["sh", "-c", 'if [ -d "$1" ]; then echo dir; elif [ -f "$1" ]; then echo file; else echo none; fi', "sh", path])
        kind = out.decode().strip()
        if kind not in ("dir", "file", "none"):
            raise ContainerError(f"path probe failed for {path}: {kind[:120]}")
        return kind

    def diff_summary(self) -> dict[str, int]:
        r = subprocess.run(["docker", "diff", self.name], capture_output=True, text=True, timeout=300)
        counts = {"A": 0, "C": 0, "D": 0}
        for line in r.stdout.splitlines():
            if line[:1] in counts:
                counts[line[:1]] += 1
        return counts

    def commit(self, tag: str) -> str:
        r = subprocess.run(["docker", "commit", self.name, tag], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise ContainerError(f"docker commit failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def stop(self, remove: bool = True) -> None:
        if self.shell:
            self.shell.close()
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=300)


# ── Baseline pinning ──────────────────────────────────────────────


@dataclass(frozen=True)
class BaselineSpec:
    base_commit: str | None
    test_patch: str | None = None
    apply_test_patch: bool = False
    repo_dir: str = "/testbed"
    squash_history: bool = True     # CoderForge recordings see a single "Initial commit" on master


@dataclass
class BaselineProbe:
    repo_dir: str
    head_before: str | None
    head_after: str | None
    base_commit: str | None
    checkout_ok: bool
    dirty_tracked_before: int | None
    dirty_tracked_after: int | None
    test_patch_applied: bool
    ok: bool
    errors: list[str] = field(default_factory=list)


def pin_baseline(ws: ContainerWorkspace, spec: BaselineSpec) -> BaselineProbe:
    """Check out the base commit (and optionally the test patch); report what happened."""
    assert ws.shell is not None
    sh = ws.shell
    errors: list[str] = []
    q = shlex.quote(spec.repo_dir)

    def git(cmd: str, timeout: float = 300) -> tuple[str, int | None]:
        out, code, timed_out = sh.run(f"cd {q} && git -c advice.detachedHead=false {cmd}", timeout=timeout)
        if timed_out:
            errors.append(f"git {cmd.split()[0]} timed out")
        return out.strip(), code

    head_before, code = git("rev-parse HEAD")
    if code != 0:
        errors.append(f"not a git repository at {spec.repo_dir}: {head_before[:120]}")
        return BaselineProbe(spec.repo_dir, None, None, spec.base_commit, False, None, None, False, False, errors)
    dirty_before_s, _ = git("status --porcelain --untracked-files=no | wc -l")
    dirty_before = int(dirty_before_s) if dirty_before_s.isdigit() else None

    checkout_ok = False
    head_after = head_before
    if spec.base_commit:
        out, code = git(f"cat-file -e {shlex.quote(spec.base_commit)}^{{commit}}")
        if code != 0:
            errors.append(f"base commit {spec.base_commit[:12]} not present in image history")
        else:
            out, code = git(f"checkout -f {shlex.quote(spec.base_commit)}")
            if code != 0:
                errors.append(f"checkout failed: {out[:200]}")
            else:
                head_after, _ = git("rev-parse HEAD")
                checkout_ok = head_after.startswith(spec.base_commit[:12]) or spec.base_commit.startswith(head_after[:12])
                if not checkout_ok:
                    errors.append(f"HEAD after checkout {head_after[:12]} != base {spec.base_commit[:12]}")
    else:
        errors.append("no base_commit available for this instance")

    patched = False
    if spec.apply_test_patch and spec.test_patch and checkout_ok:
        ws.write_file("/tmp/moirai_test.patch", spec.test_patch if spec.test_patch.endswith("\n") else spec.test_patch + "\n")
        out, code = git("apply --check /tmp/moirai_test.patch")
        if code != 0:
            errors.append(f"test patch does not apply cleanly: {out[:200]}")
        else:
            out, code = git("apply /tmp/moirai_test.patch")
            patched = code == 0
            if not patched:
                errors.append(f"test patch apply failed: {out[:200]}")

    squashed = False
    if spec.squash_history and checkout_ok:
        out, code, _ = sh.run(
            f"cd {q} && mv .git /tmp/moirai_git_orig_$RANDOM && git init -q -b master . && git add -A . && "
            f"git -c user.name=openhands -c user.email=openhands@all-hands.dev -c commit.gpgsign=false commit -qm 'Initial commit'",
            timeout=600)
        squashed = code == 0
        if not squashed:
            errors.append(f"history squash failed: {out[-200:]}")

    dirty_after_s, _ = git("status --porcelain --untracked-files=no | wc -l")
    dirty_after = int(dirty_after_s) if dirty_after_s.isdigit() else None
    expected_dirty = dirty_after == 0 or (patched and not squashed and dirty_after is not None)
    ok = checkout_ok and not any(e.startswith(("checkout failed", "test patch")) for e in errors) and expected_dirty
    if spec.apply_test_patch and not patched:
        ok = False
    if spec.squash_history and not squashed:
        ok = False
    sh.run(f"cd {q}", timeout=10)
    return BaselineProbe(spec.repo_dir, head_before, head_after, spec.base_commit, checkout_ok,
                         dirty_before, dirty_after, patched, ok, errors)


# ── OpenHands tool emulation ──────────────────────────────────────


def _cat_n(content: str, start: int = 1) -> str:
    """OpenHands numbers every element of ``content.split("\\n")``, including the empty tail."""
    lines = content.split("\n")
    return "\n".join(f"{i + start:6}\t{line}" for i, line in enumerate(lines))


def _read_text(ws: ContainerWorkspace, path: str) -> str | None:
    """OpenHands reads with universal newlines (CRLF becomes LF) and drops a leading UTF-8 BOM.

    Evidence: AzureAD run3 msg 10 records ``1  #!/usr/bin/env python`` for a file whose
    first bytes are EF BB BF.
    """
    raw = ws.read_file(path)
    if raw is None:
        return None
    return raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def editor_view(ws: ContainerWorkspace, path: str, view_range: list[int] | None) -> tuple[str, bool]:
    kind = ws.path_kind(path)
    if kind == "dir":
        out, _ = ws.exec_capture(["sh", "-c", 'find "$1" -maxdepth 2 -not -path "*/.*" -type d', "sh", path])
        dirs = set(out.decode("utf-8", errors="replace").split("\n"))
        out, _ = ws.exec_capture(["sh", "-c", 'find "$1" -maxdepth 2 -not -path "*/.*"', "sh", path])
        entries = sorted(e for e in out.decode("utf-8", errors="replace").split("\n") if e)
        listing = "\n".join(e + "/" if e in dirs else e for e in entries)
        out, _ = ws.exec_capture(["sh", "-c", 'find "$1" -maxdepth 2 -name ".*" | wc -l', "sh", path])
        hidden = int(out.decode().strip() or 0)
        return (f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{listing}\n"
                + hidden_footer(path, hidden)), True
    if kind != "file":
        return f"ERROR:\nThe path {path} does not exist. Please provide a valid path.", False
    content = _read_text(ws, path) or ""
    if view_range and len(view_range) == 2:
        lines = content.split("\n")
        n = len(lines) - 1 if content.endswith("\n") else len(lines)
        a, b = view_range
        if not (1 <= a <= n):
            return (f"ERROR:\nInvalid `view_range` parameter: {view_range}. Its first element `{a}` should be within the range of lines of the file: [1, {n}]."), False
        if b != -1 and b > n:
            return (f"ERROR:\nInvalid `view_range` parameter: {view_range}. Its second element `{b}` should be smaller than the number of lines in the file: `{n}`."), False
        if b != -1 and b < a:
            return (f"ERROR:\nInvalid `view_range` parameter: {view_range}. Its second element `{b}` should be larger or equal than its first element `{a}`."), False
        b = n if b == -1 else b
        snippet = "\n".join(lines[a - 1:b])
        return f"Here's the result of running `cat -n` on {path}:\n{_cat_n(snippet, a)}\n", True
    return f"Here's the result of running `cat -n` on {path}:\n{_cat_n(content)}\n", True


def editor_edit(ws: ContainerWorkspace, args: dict) -> tuple[str, bool]:
    command = args.get("command")
    path = str(args.get("path", ""))
    if command == "create":
        if ws.path_kind(path) != "none":
            return f"ERROR:\nFile already exists at: {path}. Cannot overwrite files using command `create`.", False
        ws.write_file(path, str(args.get("file_text", "")))
        return f"File created successfully at: {path}", True
    content = _read_text(ws, path)
    if content is None:
        return f"ERROR:\nThe path {path} does not exist. Please provide a valid path.", False
    if command == "str_replace":
        old, new = str(args.get("old_str", "")), str(args.get("new_str", "") or "")
        if old == new:
            # openhands-aci refuses a no-op replacement (serpent-tools run2 msg 74)
            return (f"ERROR:\nInvalid `new_str` parameter: {new}. No replacement was performed. "
                    "`new_str` and `old_str` must be different."), False
        n = content.count(old)
        if n == 0:
            return f"No replacement was performed, old_str `{old[:200]}` did not appear verbatim in {path}.", False
        if n > 1:
            return f"No replacement was performed. Multiple occurrences of old_str `{old[:200]}` in {path}.", False
        updated = content.replace(old, new, 1)
        ws.write_file(path, updated)
        # OpenHands' snippet window sits one line later than a naive count; calibrated on recordings.
        line = content[:content.index(old)].count("\n") + 1
        lines = updated.split("\n")
        a, b = max(0, line - 4), min(len(lines), line + new.count("\n") + 5)
        snippet = "\n".join(lines[a:b])
        return (f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of {path}:\n"
                f"{_cat_n(snippet, a + 1)}\nReview the changes and make sure they are as expected. Edit the file again if necessary."), True
    if command == "insert":
        try:
            at = int(args.get("insert_line", 0))
        except (TypeError, ValueError):
            return "ERROR: insert_line must be an integer", False
        lines = content.split("\n")
        new_lines = str(args.get("new_str", "")).split("\n")
        lines[at:at] = new_lines
        updated = "\n".join(lines)
        ws.write_file(path, updated)
        a, b = max(0, at - 4), min(len(lines), at + len(new_lines) + 4)
        return (f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of the edited file:\n"
                f"{_cat_n(chr(10).join(lines[a:b]), a + 1)}\nReview the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."), True
    return f"ERROR: unsupported editor command {command}", False


# ── Acceptance rule ───────────────────────────────────────────────

ORDER_INSENSITIVE_CLASSES = frozenset({"search", "list"})


def judge_step(
    action_class: str,
    tool_name: str,
    args: dict,
    recorded: str,
    replayed: str,
    status: str,
    replayed_exit: int | None,
    match_threshold: float,
) -> tuple[bool, str]:
    """(accepted, reason). Strict: the recorded state must be shown, not approximated.

    Order matters: hard failures, then recorded-process anomalies, then outcome
    rules (exit code, pytest pass/fail set) which override text similarity for
    test runs, then text comparison for everything else.
    """
    if status in ("edit_failed", "timeout", "error"):
        return False, status
    flags = recorded_flags(recorded)
    if status == "merged":
        return True, "merged_into_timed_out_command"
    if tool_name == "execute_bash" and args.get("is_input"):
        return False, "undeliverable:is_input"
    rec_exit_last = parse_exit_code(recorded)
    if flags["timeout_notice"] and rec_exit_last is None:
        return False, "recorded_process_still_running"
    if flags["timeout_notice"] and rec_exit_last == 130:
        ro, po = pytest_outcome(recorded), pytest_outcome(replayed)
        if ro is not None and po is not None and ro["counts"] == po["counts"] and failing_tests(ro) == failing_tests(po):
            return True, "pytest_outcome_equal_before_interrupt"
        return False, "recorded_process_interrupted"
    if (flags["not_executed"] or flags["previous_output"]) and not flags["timeout_notice"]:
        return False, "recorded_process_still_running"
    if status == "skipped":
        return True, "no_environment_effect"
    order_free = action_class in ORDER_INSENSITIVE_CLASSES or (tool_name == "str_replace_editor" and args.get("command") == "view" and recorded.startswith("Here's the files"))
    if tool_name == "execute_bash":
        if rec_exit_last is not None and replayed_exit is not None and rec_exit_last != replayed_exit:
            return False, f"exit_code:{rec_exit_last}!={replayed_exit}"
        if action_class == "test" or "pytest" in str(args.get("command", "")):
            ro, po = pytest_outcome(recorded), pytest_outcome(replayed)
            if (ro is None) != (po is None):
                return False, "pytest_summary_missing_on_one_side"
            if ro is not None and po is not None:
                if ro["counts"] != po["counts"]:
                    return False, f"pytest_counts:{ro['counts']}!={po['counts']}"
                if failing_tests(ro) != failing_tests(po):
                    return False, "pytest_failing_set_differs"
                return True, "pytest_outcome_equal"
            # no summary on either side (collect-only, piped): fall through to content comparison
    if tool_name == "str_replace_editor" and args.get("command") in ("str_replace", "insert"):
        rec_lines, rep_lines = numbered_lines(recorded), numbered_lines(replayed)
        # openhands-aci numbers the trailing newline of its snippet as an empty line
        # one past the window; it is not file content.
        if rec_lines and rec_lines[max(rec_lines)] == "":
            rec_lines.pop(max(rec_lines))
        common = set(rec_lines) & set(rep_lines)
        if rec_lines and rep_lines and common:
            bad = [n for n in common if rec_lines[n].rstrip() != rep_lines[n].rstrip()]
            if not bad:
                return True, "edit_snippet_consistent"
            return False, f"edit_snippet_line_differs:{min(bad)}"
    if status == "mismatch":
        return False, "mismatch"
    if similarity(recorded, replayed, order_insensitive=order_free) >= match_threshold and line_diff_empty(recorded, replayed, order_free):
        return True, "exact_after_normalisation"
    if line_diff_empty(recorded, replayed, order_free):
        return True, "line_diff_empty"
    if flags["truncated"] and truncated_parts_match(recorded, replayed, order_free):
        return True, "equal_outside_recorded_truncation"
    return False, "content_differs"


# ── Replay ────────────────────────────────────────────────────────


@dataclass
class StepReplay:
    message_idx: int
    tool_call_id: str
    tool_name: str
    action_class: str
    status: str                 # match | near | mismatch | edit_failed | skipped | timeout | error
    similarity: float | None
    accepted: bool
    reason: str
    recorded_exit: int | None
    replayed_exit: int | None
    recorded_len: int
    replayed_len: int
    detail: str = ""
    recorded: str = ""
    replayed: str = ""


@dataclass
class ReplayReport:
    trajectory_id: str
    instance_id: str
    image: str
    anchor_message_idx: int
    baseline: dict
    anchor_accepted: bool
    n_steps: int
    n_accepted: int
    n_match: int
    n_near: int
    n_mismatch: int
    n_edit_failed: int
    n_skipped: int
    n_timeout: int
    first_divergence_message_idx: int | None
    first_rejected_message_idx: int | None
    fidelity: float                    # fraction of executed steps with status match or near (descriptive)
    diff_summary: dict[str, int]
    shell_restarts: int
    snapshot: str | None
    wall_seconds: float
    steps: list[StepReplay]
    init_commands: list[str]
    notes: list[str] = field(default_factory=list)
    instrument_hash: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)


def choose_post_test_anchor(events: list[RawEvent]) -> int | None:
    """First eligible non-mutating action whose prefix already contains a test run."""
    seen_test = False
    for ev in events:
        v = classify_action(ev.call.tool_name, ev.call.arguments_json)
        if seen_test and v.eligible and ev.observation is not None:
            return ev.call.message_idx
        if v.action_class == "test":
            seen_test = True
    return None


def choose_deep_anchor(events: list[RawEvent], max_prefix_calls: int = 40) -> int | None:
    """Last eligible action, capped so at most ``max_prefix_calls`` calls precede it."""
    eligible = [e.call.message_idx for e in events
                if classify_action(e.call.tool_name, e.call.arguments_json).eligible and e.observation is not None]
    if not eligible:
        return None
    cap_idx = events[max_prefix_calls].call.message_idx if len(events) > max_prefix_calls else None
    last = eligible[-1]
    return min(last, cap_idx) if cap_idx is not None else last


def choose_full_anchor(events: list[RawEvent]) -> int | None:
    """Boundary after the last tool call: replaying to it certifies every earlier anchor."""
    if not events:
        return None
    return events[-1].call.message_idx + 1


def replay_prefix(
    traj: RawTrajectory,
    anchor_message_idx: int,
    image: str | None = None,
    baseline: BaselineSpec | None = None,
    snapshot_tag: str | None = None,
    command_timeout: float = 180.0,
    match_threshold: float = 0.95,
    near_threshold: float = 0.80,
    workspace: ContainerWorkspace | None = None,
    keep_container: bool = False,
    stop_on_reject: bool = False,
    network: str = "none",
) -> ReplayReport:
    events, _ = index_events(traj)
    prefix = [e for e in events if e.call.message_idx < anchor_message_idx]
    image = image or traj.image
    if not image:
        raise ContainerError("no container image known for this trajectory")
    ws = workspace or ContainerWorkspace(image=image, network=network)
    t0 = time.perf_counter()
    notes: list[str] = []
    steps: list[StepReplay] = []
    if workspace is None:
        ws.start()
    assert ws.shell is not None
    try:
        probe = pin_baseline(ws, baseline) if baseline else BaselineProbe("/testbed", None, None, None, False, None, None, False, False, ["baseline pinning disabled"])
        groups = group_timeout_steps(prefix)
        merged_members = {m: head for head, ms in groups.items() for m in ms}
        for pos, ev in enumerate(prefix):
            call = ev.call
            v = classify_action(call.tool_name, call.arguments_json)
            args = call.arguments or {}
            recorded = ev.observation or ""
            if pos in merged_members:
                steps.append(StepReplay(
                    message_idx=call.message_idx, tool_call_id=call.tool_call_id, tool_name=call.tool_name,
                    action_class=v.action_class, status="merged", similarity=None, accepted=True,
                    reason="merged_into_timed_out_command", recorded_exit=parse_exit_code(recorded), replayed_exit=None,
                    recorded_len=len(recorded), replayed_len=0, detail=f"follow-up of msg {prefix[merged_members[pos]].call.message_idx}",
                    recorded=strip_ansi(recorded)[:OBSERVATION_CAP], replayed="",
                ))
                continue
            if pos in groups:
                recorded = "\n".join((prefix[k].observation or "") for k in [pos] + groups[pos])
            replayed = ""
            status = "error"
            detail = ""
            sim: float | None = None
            rep_exit: int | None = None
            order_free = v.action_class in ORDER_INSENSITIVE_CLASSES
            for attempt in range(2):
                try:
                    if call.tool_name == "think":
                        status, replayed = "skipped", "Your thought has been logged."
                    elif call.tool_name == "finish":
                        status, detail = "skipped", "terminal action inside prefix"
                    elif call.tool_name == "execute_bash":
                        if args.get("is_input"):
                            status, detail = "skipped", "stdin to a running process is not deliverable"
                        else:
                            command = str(args.get("command", ""))
                            rejected_obs = openhands_bash_rejection(command)
                            if rejected_obs is not None:
                                out, rep_exit, timed_out = rejected_obs, 2, False
                            else:
                                out, rep_exit, timed_out = ws.shell.run(command, timeout=float(args.get("timeout") or command_timeout))
                            recorded = strip_command_echo(recorded, command)
                            replayed = out
                            if timed_out:
                                status, detail = "timeout", f"no completion within {command_timeout}s"
                            else:
                                status = similarity_status(recorded, replayed, order_free, match_threshold, near_threshold)
                                detail = f"exit={rep_exit}"
                    elif call.tool_name == "str_replace_editor":
                        if args.get("command") == "view":
                            replayed, ok = editor_view(ws, str(args.get("path", "")), args.get("view_range"))
                            order_free = recorded.startswith("Here's the files")
                        else:
                            replayed, ok = editor_edit(ws, args)
                        if not ok and not recorded.startswith(("ERROR", "No replacement")):
                            status, detail = "edit_failed", replayed[:200]
                        else:
                            status = similarity_status(recorded, replayed, order_free, match_threshold, near_threshold)
                    else:
                        status, detail = "skipped", f"tool {call.tool_name} not emulated"
                    break
                except ContainerError as e:
                    status, detail = "error", str(e)
                    if attempt == 0:
                        notes.append(f"msg {call.message_idx}: container error, restarting shell once: {e}")
                        try:
                            ws.shell._restart()
                        except Exception as e2:  # noqa: BLE001
                            notes.append(f"restart failed: {e2}")
                            break
            accepted, reason = judge_step(v.action_class, call.tool_name, args, recorded, replayed, status, rep_exit, match_threshold)
            steps.append(StepReplay(
                message_idx=call.message_idx, tool_call_id=call.tool_call_id, tool_name=call.tool_name,
                action_class=v.action_class, status=status, similarity=sim, accepted=accepted, reason=reason,
                recorded_exit=parse_exit_code(recorded), replayed_exit=rep_exit,
                recorded_len=len(recorded), replayed_len=len(replayed), detail=detail,
                recorded=strip_ansi(recorded)[:OBSERVATION_CAP], replayed=strip_ansi(replayed)[:OBSERVATION_CAP],
            ))
            if stop_on_reject and not accepted:
                notes.append(f"stopped at first rejected step (msg {call.message_idx}); {len(prefix) - pos - 1} later calls not replayed")
                break
        diff = ws.diff_summary()
        snap = ws.commit(snapshot_tag) if snapshot_tag else None
        restarts = ws.shell.restarts
    finally:
        if not keep_container and workspace is None:
            ws.stop()
    executed = [s for s in steps if s.status not in ("skipped", "merged")]
    good = sum(1 for s in executed if s.status in ("match", "near"))
    first_div = next((s.message_idx for s in steps if s.status in ("mismatch", "edit_failed", "timeout", "error")), None)
    first_rej = next((s.message_idx for s in steps if not s.accepted), None)
    return ReplayReport(
        trajectory_id=traj.trajectory_id, instance_id=traj.instance_id, image=image,
        anchor_message_idx=anchor_message_idx, baseline=asdict(probe),
        anchor_accepted=bool(probe.ok and all(s.accepted for s in steps) and len(steps) == len(prefix)),
        n_steps=len(steps), n_accepted=sum(s.accepted for s in steps),
        n_match=sum(s.status == "match" for s in steps), n_near=sum(s.status == "near" for s in steps),
        n_mismatch=sum(s.status == "mismatch" for s in steps), n_edit_failed=sum(s.status == "edit_failed" for s in steps),
        n_skipped=sum(s.status == "skipped" for s in steps), n_timeout=sum(s.status == "timeout" for s in steps),
        first_divergence_message_idx=first_div, first_rejected_message_idx=first_rej,
        fidelity=(good / len(executed)) if executed else 1.0,
        diff_summary=diff, shell_restarts=restarts, snapshot=snap, wall_seconds=time.perf_counter() - t0,
        steps=steps, init_commands=list(ws.init_commands), notes=notes, instrument_hash=instrument_hash(),
    )
