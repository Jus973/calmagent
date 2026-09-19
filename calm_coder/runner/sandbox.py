"""Run a test module against generated code in a fresh process and temp cwd (§3.9).

A guard against runaway loops and stray file writes, not against adversarial code.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUNNER = str(Path(__file__).with_name("_run_tests.py"))
_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


def run_tests(module_src: str, test_src: str, test_classes: list[str], *,
              per_class_timeout_s: float = 5, wall_timeout_s: float = 20) -> dict[str, dict]:
    """-> {test_class: {"result", "detail", "wall_ms", "tests_run"}}"""
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="calm_") as d:
        Path(d, "mod.py").write_text(module_src + "\n\n" + test_src)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", RUNNER, "mod.py", str(per_class_timeout_s), *test_classes],
                cwd=d, timeout=wall_timeout_s, capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            ms = int((time.monotonic() - t0) * 1000)
            return {c: {"result": "timeout", "detail": "wall timeout", "wall_ms": ms, "tests_run": 0}
                    for c in test_classes}
        res = Path(d, "result.json")
        out = json.loads(res.read_text()) if res.exists() else {}
    for c in test_classes:
        if c not in out:
            tail = (proc.stderr or "")[-2048:]
            kind = "timeout" if proc.returncode in (-9, -24, 152, 137) else "error"
            out[c] = {"result": kind, "detail": f"runner exit {proc.returncode}: {tail}",
                      "wall_ms": int((time.monotonic() - t0) * 1000), "tests_run": 0}
    return out


async def run_tests_async(*args, **kwargs) -> dict[str, dict]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_POOL, lambda: run_tests(*args, **kwargs))
