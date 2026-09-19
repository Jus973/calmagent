"""Prompts (§3.7). Slot prompts share one prefix per task: everything slot-specific is the last line."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from calm_coder.task import Task

SLOT_SYSTEM = (
    "You are implementing exactly one method of a Python class. The method to implement is named "
    "in the last line of the user message. Output a single Python code block containing that complete "
    "method with its signature exactly as in the skeleton. You may also define private helper methods "
    "you need (names starting with `_`). Do not include the class header, `__init__`, imports, or other "
    "declared methods. You may call other declared methods via `self.<name>()` and rely on their "
    "docstrings. No prose."
)

CLASS_SYSTEM = "You are an expert Python programmer. Output only code."

SAMPLING = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens_slot": 512,
    "max_tokens_class": 2048,
}

REPAIR_MAX_LINES = 40


def _skeleton(task: "Task") -> str:
    return task.skeleton.strip("\n")


def slot_messages(task: "Task", slot_id: str) -> list[dict]:
    return [
        {"role": "system", "content": SLOT_SYSTEM},
        {"role": "user", "content": f"{_skeleton(task)}\n\nImplement `{slot_id}` now."},
    ]


def whole_class_messages(task: "Task") -> list[dict]:
    return [
        {"role": "system", "content": CLASS_SYSTEM},
        {"role": "user", "content": f"{_skeleton(task)}\n\n"
                                    "Complete the class. Output one Python code block containing the full class."},
    ]


def incremental_messages(task: "Task", filled: Mapping[str, str], slot_id: str) -> list[dict]:
    """Baseline B: skeleton with previously generated methods substituted in."""
    import ast
    import textwrap
    mod = ast.parse(task.skeleton)
    cdef = next(n for n in mod.body if isinstance(n, ast.ClassDef))
    for i, node in enumerate(cdef.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in filled:
            try:
                cdef.body[i] = ast.parse(textwrap.dedent(filled[node.name])).body[0]
            except (SyntaxError, IndexError):
                pass
    return [
        {"role": "system", "content": SLOT_SYSTEM},
        {"role": "user", "content": f"{ast.unparse(mod)}\n\nImplement `{slot_id}` now."},
    ]


def shared_prefix(task: "Task") -> str:
    """Everything every request for a task has in common, up to the end of the skeleton block.

    Nothing variable may appear before this: no method name, sample index, round, or feedback.
    Slot, repair and whole-class-repair prompts all start with it (§5.1 of the v2 plan), so a
    server with prefix caching computes it once per task.
    """
    return _skeleton(task)


def slot_repair_messages(task: "Task", slot_id: str, current_class: str, feedback: str) -> list[dict]:
    """Targeted repair: skeleton, then the current best class, then the one slot to rewrite.

    Ordering matters: within a round every dead slot of a task shares everything up to the end of
    CURRENT_CLASS_BLOCK, so only the last block differs.
    """
    body = [shared_prefix(task)]
    if current_class:
        body.append("The current implementation of the class is:\n"
                    f"```python\n{current_class.strip()}\n```")
    if feedback:
        body.append(feedback)
    body.append(f"Rewrite only `{slot_id}`, keeping its signature. Output that method alone.")
    return [{"role": "system", "content": SLOT_SYSTEM}, {"role": "user", "content": "\n\n".join(body)}]


def whole_class_repair_messages(task: "Task", current_class: str, feedback: str) -> list[dict]:
    """The fair baseline for repair: same information, whole class regenerated."""
    body = [shared_prefix(task)]
    if current_class:
        body.append("The current implementation of the class is:\n"
                    f"```python\n{current_class.strip()}\n```")
    if feedback:
        body.append(feedback)
    body.append("Rewrite the whole class so that it is correct. "
                "Output one Python code block containing the full class.")
    return [{"role": "system", "content": CLASS_SYSTEM}, {"role": "user", "content": "\n\n".join(body)}]


def repair_messages(task: "Task", slot_id: str, failure: str) -> list[dict]:
    """Still emits a fresh definition; nothing is edited."""
    tb = "\n".join(failure.splitlines()[-REPAIR_MAX_LINES:])
    return [
        {"role": "system", "content": SLOT_SYSTEM},
        {"role": "user", "content": f"{_skeleton(task)}\n\nA previous implementation failed with:\n"
                                    f"```\n{tb}\n```\n\nImplement `{slot_id}` now."},
    ]
