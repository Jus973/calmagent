"""Left terminal of the demo (§7): one git worktree per agent, sequential last-writer-wins merges.

Every agent writes the whole class into kv.py on its own branch. Merging with `-X theirs` resolves
same-line conflicts to whichever branch merges later, so the result depends on merge order.
We merge in both orders, diff the two results, and run the tests on each.

python -m calm_coder.demo.git_baseline [--regen]
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

from calm_coder.runner.sandbox import run_tests
from calm_coder.task import task_from_files

HERE = Path(__file__).parent
RECORDED = HERE / "recorded"
N_AGENTS = 4
console = Console()


async def _generate(task, temperature: float) -> list[str]:
    from calm_coder.serve.client import Client
    from calm_coder.serve.extract import ExtractError, extract_class
    from calm_coder.serve.prompts import whole_class_messages
    out = []
    async with Client() as c:
        samples = await asyncio.gather(*(c.sample(whole_class_messages(task), n=1, temperature=temperature, top_p=0.95,
                                                  max_tokens=1024, seed=100 + i) for i in range(N_AGENTS)))
    for (s,) in samples:
        src = extract_class(s.text, task)
        out.append(s.text if isinstance(src, ExtractError) else src)
    return out


def agent_sources(task, regen: bool, temperature: float = 0.7) -> list[str]:
    files = [RECORDED / f"agent_{i}.py" for i in range(N_AGENTS)]
    if regen or not all(f.exists() for f in files):
        RECORDED.mkdir(exist_ok=True)
        for f, src in zip(files, asyncio.run(_generate(task, temperature))):
            f.write_text(src)
    return [f.read_text() for f in files]


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "user.name=agent", "-c", "user.email=agent@demo.local",
                           "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True, check=check)


def merged(repo: Path, order: list[int], name: str) -> str:
    _git(repo, "checkout", "-q", "-B", name, "main")
    for i in order:
        before = (repo / "kv.py").read_text()
        r = _git(repo, "merge", "-q", "--no-edit", "-X", "theirs", f"agent_{i}", check=False)
        after = (repo / "kv.py").read_text()
        changed = sum(1 for l in difflib.ndiff(before.splitlines(), after.splitlines()) if l[:1] in "+-")
        console.print(f"  [dim]merge agent_{i} → {name}:[/dim] "
                      + (f"[red]failed[/red] {r.stderr.strip()[:80]}" if r.returncode != 0 else
                         f"[yellow]{changed} lines overwritten (last writer wins)[/yellow]" if changed else "[dim]no change[/dim]"))
        if r.returncode != 0:
            _git(repo, "add", "-A")
            _git(repo, "commit", "-q", "--no-edit", "-m", f"force-merge agent_{i}", check=False)
    return (repo / "kv.py").read_text()


def run(regen: bool = False, temperature: float = 0.7) -> dict:
    task = task_from_files(HERE / "task.py", HERE / "test_task.py")
    agents = agent_sources(task, regen, temperature)
    with tempfile.TemporaryDirectory(prefix="calm_git_demo_") as d:
        repo = Path(d)
        _git(repo, "init", "-q", "-b", "main")
        (repo / "kv.py").write_text((HERE / "task.py").read_text())
        _git(repo, "add", "kv.py")
        _git(repo, "commit", "-q", "-m", "skeleton")
        for i, src in enumerate(agents):
            _git(repo, "checkout", "-q", "-b", f"agent_{i}", "main")
            (repo / "kv.py").write_text(src)
            _git(repo, "commit", "-q", "-am", f"agent {i}: full class")
        fwd_order, rev_order = list(range(N_AGENTS)), list(reversed(range(N_AGENTS)))
        console.rule("[bold]git: last writer wins")
        fwd = merged(repo, fwd_order, "merge_forward")
        rev = merged(repo, rev_order, "merge_reversed")
    diff = list(difflib.unified_diff(fwd.splitlines(), rev.splitlines(), "kv.py (order 0,1,2,3)",
                                     "kv.py (order 3,2,1,0)", lineterm=""))
    results = {}
    for name, src in (("forward", fwd), ("reversed", rev)):
        res = run_tests(src, task.test_src, list(task.test_classes))
        results[name] = {k: v["result"] for k, v in res.items()}
    console.print(f"diff between merge orders: [bold]{len([l for l in diff if l[:1] in '+-']) - 2} changed lines[/bold]")
    if diff:
        console.print(Syntax("\n".join(diff[:60]), "diff", theme="ansi_dark"))
    for name, r in results.items():
        passed = sum(v == "pass" for v in r.values())
        console.print(f"tests on {name:<8}: {passed}/{len(r)} classes pass  "
                      + " ".join(f"[{'green' if v == 'pass' else 'red'}]{k.replace('KVStoreTest', '') or 'Main'}[/]"
                                 for k, v in r.items()))
    console.print("[bold]Output depends on merge order.[/bold]" if diff else
                  "[yellow]The two orders happened to agree this time; rerun with --regen for new agent outputs.[/yellow]")
    return {"diff_lines": len(diff), "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="resample the 4 agents' classes from the model")
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    run(a.regen, a.temperature)
