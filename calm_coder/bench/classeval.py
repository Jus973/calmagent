"""ClassEval loading (§3.11). Subset selection is added in hours 4–6."""
from __future__ import annotations

import ast
import re

from calm_coder.task import Task, TaskError

_TEST_CLASS_RE = re.compile(r"class (\w+)\(unittest\.TestCase\)")


def load_rows() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("FudanSELab/ClassEval")["test"]
    return [dict(r) for r in ds]


def task_from_row(row: dict) -> Task:
    slot_tests = {}
    for m in row["methods_info"]:
        t = m.get("test_class") or next(iter(_TEST_CLASS_RE.findall(m.get("test_code") or "")), None)
        if t:
            slot_tests[m["method_name"]] = t
    task = Task.from_skeleton(
        task_id=row["task_id"], skeleton=row["skeleton"], test_src=row["test"],
        test_classes=list(row["test_classes"]), slot_tests=slot_tests,
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
