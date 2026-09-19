"""ClassEval loading and subset selection (§3.11).

python -m calm_coder.bench.classeval --select   # freeze data/subset.json once
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from calm_coder.task import Task, TaskError

_TEST_CLASS_RE = re.compile(r"class (\w+)\(unittest\.TestCase\)")
DATA = Path(__file__).parents[2] / "data"
SUBSET = DATA / "subset.json"


def load_rows() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("FudanSELab/ClassEval")["test"]
    return [dict(r) for r in ds]


def task_from_row(row: dict) -> Task:
    slot_tests = {}
    for m in row["methods_info"]:
        t = m.get("test_class") or next(iter(_TEST_CLASS_RE.findall(m.get("test_code") or "")), None)
        if t:
            slot_tests[m["method_name"]] = t.strip()
    task = Task.from_skeleton(
        task_id=row["task_id"], skeleton=row["skeleton"], test_src=row["test"],
        test_classes=[c.strip() for c in row["test_classes"]], slot_tests=slot_tests,
        import_statement=list(row["import_statement"]),
    )
    declared = {m["method_name"] for m in row["methods_info"]}
    if set(task.slot_ids) != declared:
        raise TaskError(f"slot/methods_info mismatch: {sorted(task.slot_ids ^ declared)}")
    return task


def reference_emissions(row: dict, task: Task) -> dict[str, list[ast.FunctionDef]]:
    """Per slot: [reference method] + every non-slot method of the reference class as helpers,
    i.e. what an agent that happened to write the reference would have emitted."""
    mod = ast.parse(row["solution_code"])
    cdef = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == task.class_name)
    fns = {n.name: n for n in cdef.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    extra = [f for n, f in fns.items() if n not in task.slot_ids and n != "__init__"]
    return {s.id: [fns[s.id], *extra] for s in task.slots if s.id in fns}


# ---------------------------------------------------------------- subset selection

def _top_modules(src: str) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out.add(n.module.split(".")[0])
    return out


def _reference_module(row: dict, task: Task) -> str:
    from calm_coder.store.defs import Composition, emission_to_defs
    from calm_coder.store.materialize import materialize
    from calm_coder.store.store import Store
    s, bind = Store(), {}
    for slot_id, fns in reference_emissions(row, task).items():
        for d in emission_to_defs(task, task.slot(slot_id), fns, {"ref": True}):
            s.add_def(d)
            if d.kind == "fill":
                bind[slot_id] = d.hash
    if set(bind) != set(task.slot_ids):
        raise TaskError(f"reference missing slots: {sorted(set(task.slot_ids) - set(bind))}")
    return materialize(task, s, Composition.make(bind))


def check_task(row: dict, runs: int = 3) -> tuple[Task | None, str | None]:
    """-> (task, None) if eligible, else (None, reason). Criteria 1–4 of §3.11."""
    from calm_coder.runner.sandbox import run_tests
    try:
        mod = ast.parse(row["skeleton"])
    except SyntaxError as e:
        return None, f"skeleton_syntax: {e}"
    stray = [type(n).__name__ for n in mod.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))
             and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]  # stray string literals are inert
    if stray:
        return None, f"module_level_code: {stray}"
    try:
        task = task_from_row(row)
    except TaskError as e:
        return None, f"skeleton: {e}"
    try:
        mods = _top_modules("\n".join(row["import_statement"])) | _top_modules(row["test"])
    except SyntaxError as e:
        return None, f"import_syntax: {e}"
    nonstd = sorted(mods - set(sys.stdlib_module_names))
    if nonstd:
        return None, f"non_stdlib_imports: {nonstd}"
    if not 2 <= len(task.slots) <= 8:
        return None, f"slot_count: {len(task.slots)}"
    missing = [s.id for s in task.slots if task.slot_tests.get(s.id) not in task.test_classes]
    if missing:
        return None, f"no_slot_test_class: {missing}"
    try:
        src = _reference_module(row, task)
    except (TaskError, StopIteration, SyntaxError) as e:
        return None, f"reference_materialize: {e!r}"
    for i in range(runs):
        res = run_tests(src, task.test_src, list(task.test_classes))
        bad = {k: v["result"] for k, v in res.items() if v["result"] != "pass"}
        if bad:
            return None, f"reference_fails_run_{i}: {bad}"
    return task, None


def select_subset(n: int = 50, seed: int = 0, rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else load_rows()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(check_task, rows))
    eligible = [r["task_id"] for r, (t, _) in zip(rows, results) if t is not None]
    excluded = [{"task_id": r["task_id"], "reason": why} for r, (t, why) in zip(rows, results) if t is None]
    chosen = sorted(random.Random(seed).sample(eligible, min(n, len(eligible))),
                    key=lambda t: int(t.split("_")[1]))
    return {"n": len(chosen), "seed": seed, "eligible": len(eligible), "task_ids": chosen,
            "excluded": excluded}


def load_subset(path: Path = SUBSET, rows: list[dict] | None = None) -> list[tuple[dict, Task]]:
    ids = json.loads(path.read_text())["task_ids"]
    by_id = {r["task_id"]: r for r in (rows if rows is not None else load_rows())}
    return [(by_id[i], task_from_row(by_id[i])) for i in ids]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen subset")
    a = ap.parse_args()
    if a.select:
        if SUBSET.exists() and not a.force:
            sys.exit(f"{SUBSET} is frozen; pass --force to reselect")
        out = select_subset(a.n, a.seed)
        DATA.mkdir(exist_ok=True)
        SUBSET.write_text(json.dumps({k: v for k, v in out.items() if k != "excluded"}, indent=1) + "\n")
        with open(DATA / "subset_excluded.jsonl", "w") as f:
            for e in out["excluded"]:
                f.write(json.dumps(e) + "\n")
        print(f"eligible {out['eligible']}/100, chose {out['n']}, excluded {len(out['excluded'])}")
