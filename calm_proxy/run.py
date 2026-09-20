"""I-5 `calm-run` — content-addressed memoization for commands.

    calm-run [--root .] -- pytest -q

The key is the hash of everything the command can read that we can see: the
(path, sha256) pairs of the git-tracked and untracked-but-not-ignored files
under `--root`, the argv, the cwd relative to the root, and a volatile-free
subset of the environment. A hit replays the recorded stdout/stderr bytes and
exits with the recorded code, so a test suite re-run on an unchanged tree costs
nothing. Timeouts and kills (137) are never cached: they are facts about the
machine, not about the tree.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from .hashing import canonical_json, sha256_hex

STORE_DIRNAME = ".calm-run"
UNCACHEABLE_EXITS = frozenset({137})

# Volatile or machine-specific: two runs that differ only in these are the same
# run as far as the tree is concerned.
VOLATILE_ENV = frozenset(
    {
        "PATH",
        "PYTHONHASHSEED",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "TERM",
        "COLUMNS",
        "LINES",
        "HOSTNAME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "RANDOM",
        "SSH_AUTH_SOCK",
        "SSH_CLIENT",
        "SSH_CONNECTION",
        "SSH_TTY",
        "DISPLAY",
        "XAUTHORITY",
        "PYTEST_CURRENT_TEST",
        "CALM_RUN_DISABLE",
    }
)

SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", STORE_DIRNAME})


def tracked_files(root: Path) -> list[Path]:
    """git-tracked plus untracked-but-not-ignored, falling back to a walk."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return _walk(root)
    names = [n for n in out.decode("utf-8", "surrogateescape").split("\0") if n]
    return sorted({root / n for n in names})


def _walk(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            found.append(Path(dirpath) / name)
    return found


def tree_digest(root: Path, paths: Iterable[Path] | None = None) -> str:
    entries: list[list[str]] = []
    for path in paths if paths is not None else tracked_files(root):
        try:
            data = path.read_bytes()
        except (OSError, IsADirectoryError):
            continue
        entries.append([path.relative_to(root).as_posix(), sha256_hex(data)])
    entries.sort()
    return sha256_hex("tree|" + canonical_json(entries))


def env_subset(env: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if env is None else env
    return {k: v for k, v in sorted(source.items()) if k not in VOLATILE_ENV}


def run_key(
    root: Path,
    cmd: Sequence[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    tree: str | None = None,
) -> str:
    material = {
        "tree": tree if tree is not None else tree_digest(root),
        "cmd": list(cmd),
        "cwd": _relative(cwd, root),
        "env": env_subset(env),
    }
    return sha256_hex("calm-run|" + canonical_json(material))


def _relative(cwd: Path, root: Path) -> str:
    try:
        return cwd.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return cwd.resolve().as_posix()


class RunStore:
    """Grow-only: one write-once file per key."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str, cmd: Sequence[str], exit_code: int, out: bytes, err: bytes, ms: int) -> None:
        path = self.path_for(key)
        if path.exists():
            return
        entry = {
            "key": key,
            "ts": time.time(),
            "cmd": list(cmd),
            "exit": exit_code,
            "stdout_b64": base64.b64encode(out).decode("ascii"),
            "stderr_b64": base64.b64encode(err).decode("ascii"),
            "wall_ms": ms,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry), encoding="utf-8")
        os.replace(tmp, path)


def _emit(entry: dict) -> int:
    sys.stdout.buffer.write(base64.b64decode(entry["stdout_b64"]))
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(base64.b64decode(entry["stderr_b64"]))
    sys.stderr.buffer.flush()
    return int(entry["exit"])


def execute(
    cmd: Sequence[str],
    root: Path,
    store: RunStore,
    timeout: float | None = None,
    verbose: bool = False,
) -> int:
    key = run_key(root, cmd, Path.cwd())
    hit = store.get(key)
    if hit is not None:
        code = _emit(hit)
        if verbose:
            print(f"calm-run: replayed {key[:12]}", file=sys.stderr)
        return code

    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except FileNotFoundError:
        print(f"calm-run: command not found: {cmd[0]}", file=sys.stderr)
        return 127
    except subprocess.TimeoutExpired as exc:
        sys.stdout.buffer.write(exc.stdout or b"")
        sys.stderr.buffer.write(exc.stderr or b"")
        print(f"calm-run: timeout after {timeout}s (not cached)", file=sys.stderr)
        return 124
    ms = int((time.monotonic() - started) * 1000)
    sys.stdout.buffer.write(proc.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(proc.stderr)
    sys.stderr.buffer.flush()
    if proc.returncode not in UNCACHEABLE_EXITS and proc.returncode >= 0:
        store.put(key, cmd, proc.returncode, proc.stdout, proc.stderr, ms)
    elif verbose:
        print(f"calm-run: exit {proc.returncode} not cached", file=sys.stderr)
    return proc.returncode


SHIM_TEMPLATE = """#!/bin/sh
# written by `calm-run shim`
exec {calm_run} --root {root} -- {target} "$@"
"""


def write_shims(directory: Path, root: Path, targets: Sequence[str] = ("pytest", "python")) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    calm_run = Path(sys.argv[0]).resolve()
    if not calm_run.exists():
        calm_run = Path(sys.executable).with_name("calm-run")
    written = []
    for target in targets:
        resolved = _which_outside(target, directory)
        path = directory / target
        path.write_text(
            SHIM_TEMPLATE.format(
                calm_run=str(calm_run), root=str(root.resolve()), target=resolved
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)
        written.append(path)
    return written


def _which_outside(target: str, exclude: Path) -> str:
    from shutil import which

    for entry in (os.environ.get("PATH") or "").split(os.pathsep):
        if not entry or Path(entry).resolve() == exclude.resolve():
            continue
        candidate = which(target, path=entry)
        if candidate:
            return candidate
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calm-run",
        description="Content-addressed memoization for commands: replay stdout/stderr/exit "
        "when the tree, the command, the cwd and the environment are unchanged.",
    )
    parser.add_argument("--root", default=".", help="tree whose files form the key (default: .)")
    parser.add_argument("--store", default=None, help=f"store directory (default: <root>/{STORE_DIRNAME})")
    parser.add_argument("--timeout", type=float, default=None, help="seconds; a timeout is never cached")
    parser.add_argument("--verbose", action="store_true", help="say when a result was replayed")
    parser.add_argument("--no-cache", action="store_true", help="run the command, ignoring the store")
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        metavar="-- CMD...",
        help="the command to run, after --; or `shim <dir>`",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    root = Path(args.root)

    if rest and rest[0] == "shim":
        if len(rest) < 2:
            print("calm-run: shim needs a directory", file=sys.stderr)
            return 2
        for path in write_shims(Path(rest[1]), root):
            print(path)
        return 0

    if not rest:
        build_parser().print_help()
        return 2

    if args.no_cache or os.environ.get("CALM_RUN_DISABLE"):
        return subprocess.run(rest).returncode
    store = RunStore(Path(args.store) if args.store else root / STORE_DIRNAME)
    return execute(rest, root, store, timeout=args.timeout, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
