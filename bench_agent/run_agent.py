"""Run one coding agent on one packaged task, through a base URL, and grade it honestly.

    python -m bench_agent.run_agent --agent mini-swe --task ClassEval_7 \
        --base-url http://127.0.0.1:8999/v1 --out runs/<ts>_probe/

The agent works in a *copy* of the task, so a run cannot contaminate the next one. Grading
re-copies the pristine `tests/` over whatever the agent left behind before running pytest, so an
agent that edits or deletes the tests to make them pass is graded as if it had not.

`--agent`:
  mini-swe   mini-swe-agent's headless `DefaultAgent`, driven through its Python API rather than
             its CLI: the CLI insists on a terminal and dies under a pipe, and the API also lets
             the wall-clock cap be enforced from the outside.
  aider      `aider --message` one-shot, no git.
  calm       this repo's own `calm solve`. Always available, weakest story.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
DEFAULT_CAP_S = 480  # 8 minutes, per the overnight contract


def grade(workdir: pathlib.Path, task_dir: pathlib.Path, timeout: int = 300) -> dict:
    """Run the hidden tests from a pristine copy. The agent's edits to tests/ are discarded."""
    shutil.rmtree(workdir / "tests", ignore_errors=True)
    shutil.copytree(task_dir / "tests", workdir / "tests")
    shutil.copy(task_dir / "conftest.py", workdir / "conftest.py")
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--timeout=60"],
                           cwd=workdir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"solved": False, "reason": "grading_timeout", "tail": ""}
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"grading_error:{exc}", "tail": ""}
    tail = (p.stdout or p.stderr).strip().splitlines()
    return {"solved": p.returncode == 0, "reason": "graded",
            "tail": tail[-1] if tail else "", "returncode": p.returncode}


def run_mini_swe(workdir: pathlib.Path, base_url: str, model: str, cap_s: int, out: pathlib.Path,
                 task_id: str) -> dict:
    """Drive mini-swe-agent in a subprocess so the wall-clock cap can actually kill it."""
    driver = ROOT / "_mini_driver.py"
    env = dict(os.environ,
               OPENAI_API_BASE=base_url, OPENAI_BASE_URL=base_url, OPENAI_API_KEY="dummy",
               MSWEA_COST_TRACKING="ignore_errors", LITELLM_LOG="ERROR")
    cmd = [sys.executable, str(driver), "--cwd", str(workdir), "--model", model,
           "--task-file", str(workdir / "TASK.md"),
           "--traj", str(out / f"traj_{task_id}.json"), "--step-limit", "25"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cap_s, env=env)
        reason = "finished"
        detail = (p.stdout or "")[-4000:]
        if p.returncode != 0:
            reason = "agent_error"
            detail = ((p.stdout or "") + "\n" + (p.stderr or ""))[-4000:]
    except subprocess.TimeoutExpired as exc:
        reason, detail = "timeout", (exc.stdout or b"").decode("utf-8", "replace")[-2000:] if isinstance(exc.stdout, bytes) else str(exc.stdout)[-2000:]
    return {"agent_wall_s": round(time.time() - t0, 1), "agent_reason": reason, "agent_tail": detail}


def run_aider(workdir: pathlib.Path, base_url: str, model: str, cap_s: int) -> dict:
    env = dict(os.environ, OPENAI_API_BASE=base_url, OPENAI_API_KEY="dummy")
    cmd = ["aider", "--model", f"openai/{model}", "--openai-api-base", base_url,
           "--yes", "--no-git", "--no-auto-commits", "--no-stream",
           "--message", (workdir / "TASK.md").read_text(), "solution.py"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=cap_s, env=env)
        reason = "finished" if p.returncode == 0 else "agent_error"
        detail = ((p.stdout or "") + (p.stderr or ""))[-4000:]
    except subprocess.TimeoutExpired:
        reason, detail = "timeout", ""
    return {"agent_wall_s": round(time.time() - t0, 1), "agent_reason": reason, "agent_tail": detail}


def run_calm(workdir: pathlib.Path, base_url: str, model: str, cap_s: int) -> dict:
    env = dict(os.environ, CALM_BASE_URL=base_url, CALM_MODEL=model, CALM_NO_N="1")
    cmd = [sys.executable, "-m", "calm_coder.cli", "solve", "solution.py",
           "--tests", "tests/test_solution.py", "--N", "4", "--out", "solution.py"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=cap_s, env=env)
        reason = "finished" if p.returncode == 0 else "budget_exhausted_or_error"
        detail = ((p.stdout or "") + (p.stderr or ""))[-4000:]
    except subprocess.TimeoutExpired:
        reason, detail = "timeout", ""
    return {"agent_wall_s": round(time.time() - t0, 1), "agent_reason": reason, "agent_tail": detail}


def run_one(agent: str, task_id: str, base_url: str, model: str, out: pathlib.Path,
            cap_s: int = DEFAULT_CAP_S, keep: bool = False) -> dict:
    task_dir = TASKS / task_id
    if not task_dir.exists():
        raise SystemExit(f"no such task: {task_dir}")
    out.mkdir(parents=True, exist_ok=True)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"bench_{task_id}_"))
    workdir = tmp / task_id
    shutil.copytree(task_dir, workdir)

    t0 = time.time()
    if agent == "mini-swe":
        info = run_mini_swe(workdir, base_url, model, cap_s, out, task_id)
    elif agent == "aider":
        info = run_aider(workdir, base_url, model, cap_s)
    elif agent == "calm":
        info = run_calm(workdir, base_url, model, cap_s)
    else:
        raise SystemExit(f"unknown agent {agent}")

    result = {"task_id": task_id, "agent": agent, "model": model, "base_url": base_url,
              "cap_s": cap_s, "wall_s": round(time.time() - t0, 1), **info}
    result.update(grade(workdir, task_dir))
    (out / f"result_{task_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "results.jsonl").open("a") as fh:
        fh.write(json.dumps(result) + "\n")
    if keep:
        result["workdir"] = str(workdir)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="mini-swe", choices=["mini-swe", "aider", "calm"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--cap-s", type=int, default=DEFAULT_CAP_S)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = run_one(args.agent, args.task, args.base_url, args.model, pathlib.Path(args.out),
                args.cap_s, args.keep)
    print(json.dumps({k: v for k, v in r.items() if k != "agent_tail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
