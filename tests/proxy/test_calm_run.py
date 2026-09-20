"""I-5 calm-run: replay a command's output when nothing it can read changed."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from calm_proxy import run as calmrun


def tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.txt").write_text("hello\n")
    return root


def store(root: Path) -> calmrun.RunStore:
    return calmrun.RunStore(root / calmrun.STORE_DIRNAME)


ECHO = [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]


def test_key_changes_with_the_tree(tmp_path: Path):
    root = tree(tmp_path)
    first = calmrun.run_key(root, ["pytest"], root)
    assert first == calmrun.run_key(root, ["pytest"], root)
    (root / "a.py").write_text("x = 2\n")
    assert first != calmrun.run_key(root, ["pytest"], root)


def test_key_changes_with_cmd_and_env(tmp_path: Path):
    root = tree(tmp_path)
    base = calmrun.run_key(root, ["pytest"], root, env={"A": "1"})
    assert base != calmrun.run_key(root, ["pytest", "-q"], root, env={"A": "1"})
    assert base != calmrun.run_key(root, ["pytest"], root, env={"A": "2"})


def test_volatile_env_is_ignored(tmp_path: Path):
    root = tree(tmp_path)
    a = calmrun.run_key(root, ["pytest"], root, env={"A": "1", "PATH": "/x", "PYTHONHASHSEED": "0"})
    b = calmrun.run_key(root, ["pytest"], root, env={"A": "1", "PATH": "/y", "PYTHONHASHSEED": "7"})
    assert a == b


def test_hit_replays_bytes_and_exit_code(tmp_path: Path, capfdbinary):
    root = tree(tmp_path)
    st = store(root)
    code = calmrun.execute(ECHO, root, st)
    first = capfdbinary.readouterr()
    assert code == 0
    assert first.out == b"out\n"

    code = calmrun.execute(ECHO, root, st, verbose=True)
    second = capfdbinary.readouterr()
    assert code == 0
    assert second.out == first.out
    assert second.err.startswith(first.err)
    assert b"calm-run: replayed " in second.err


def test_quiet_by_default(tmp_path: Path, capfdbinary):
    root = tree(tmp_path)
    st = store(root)
    calmrun.execute(ECHO, root, st)
    capfdbinary.readouterr()
    calmrun.execute(ECHO, root, st)
    assert b"calm-run" not in capfdbinary.readouterr().err


def test_a_changed_file_misses(tmp_path: Path, capfdbinary):
    root = tree(tmp_path)
    st = store(root)
    calmrun.execute(ECHO, root, st)
    capfdbinary.readouterr()
    (root / "a.py").write_text("x = 99\n")
    calmrun.execute(ECHO, root, st, verbose=True)
    assert b"replayed" not in capfdbinary.readouterr().err


def test_failing_command_is_cached_with_its_code(tmp_path: Path, capfdbinary):
    root = tree(tmp_path)
    st = store(root)
    cmd = [sys.executable, "-c", "import sys; sys.exit(3)"]
    assert calmrun.execute(cmd, root, st) == 3
    capfdbinary.readouterr()
    assert calmrun.execute(cmd, root, st, verbose=True) == 3
    assert b"replayed" in capfdbinary.readouterr().err


def test_timeouts_are_not_cached(tmp_path: Path, capfdbinary):
    root = tree(tmp_path)
    st = store(root)
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
    assert calmrun.execute(cmd, root, st, timeout=0.2) == 124
    capfdbinary.readouterr()
    assert not list(st.dir.glob("*.json"))


def test_kill_137_is_not_cached(tmp_path: Path, capfdbinary):
    root = tree(tmp_path)
    st = store(root)
    cmd = [sys.executable, "-c", "import sys; sys.exit(137)"]
    assert calmrun.execute(cmd, root, st) == 137
    capfdbinary.readouterr()
    assert not list(st.dir.glob("*.json"))


def test_store_is_write_once(tmp_path: Path):
    root = tree(tmp_path)
    st = store(root)
    st.put("k", ["cmd"], 0, b"one", b"", 1)
    st.put("k", ["cmd"], 1, b"two", b"", 1)
    assert st.get("k")["exit"] == 0


def test_store_dir_is_not_part_of_the_key(tmp_path: Path):
    root = tree(tmp_path)
    st = store(root)
    before = calmrun.run_key(root, ["pytest"], root)
    calmrun.execute(ECHO, root, st)
    assert calmrun.run_key(root, ["pytest"], root) == before


def test_git_ignored_files_do_not_change_the_key(tmp_path: Path):
    root = tree(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("junk/\n")
    before = calmrun.run_key(root, ["pytest"], root)
    (root / "junk").mkdir()
    (root / "junk" / "big.log").write_text("noise\n")
    assert calmrun.run_key(root, ["pytest"], root) == before
    (root / "new.py").write_text("y = 1\n")
    assert calmrun.run_key(root, ["pytest"], root) != before


def test_cli_roundtrip(tmp_path: Path, capfdbinary, monkeypatch):
    root = tree(tmp_path)
    argv = ["--root", str(root), "--verbose", "--", *ECHO]
    assert calmrun.main(argv) == 0
    capfdbinary.readouterr()
    assert calmrun.main(argv) == 0
    assert b"replayed" in capfdbinary.readouterr().err


def test_cli_no_cache_bypasses_the_store(tmp_path: Path, capfdbinary):
    root = tree(tmp_path)
    argv = ["--root", str(root), "--no-cache", "--", *ECHO]
    assert calmrun.main(argv) == 0
    capfdbinary.readouterr()
    assert not (root / calmrun.STORE_DIRNAME).exists()


def test_shim_writes_executables(tmp_path: Path, capfdbinary):
    root = tree(tmp_path)
    bindir = tmp_path / "bin"
    assert calmrun.main(["--root", str(root), "shim", str(bindir)]) == 0
    capfdbinary.readouterr()
    for name in ("pytest", "python"):
        path = bindir / name
        assert path.exists()
        assert os.access(path, os.X_OK)
        assert "calm-run" in path.read_text()
        assert str(root.resolve()) in path.read_text()
