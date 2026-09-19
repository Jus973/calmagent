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


def repair_messages(task: "Task", slot_id: str, failure: str) -> list[dict]:
    """Still emits a fresh definition; nothing is edited."""
    tb = "\n".join(failure.splitlines()[-REPAIR_MAX_LINES:])
    return [
        {"role": "system", "content": SLOT_SYSTEM},
        {"role": "user", "content": f"{_skeleton(task)}\n\nA previous implementation failed with:\n"
                                    f"```\n{tb}\n```\n\nImplement `{slot_id}` now."},
    ]
