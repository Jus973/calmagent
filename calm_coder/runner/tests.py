"""Which test classes belong to which slot (§3.10). Naming is inconsistent: never infer by suffix."""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calm_coder.task import Task

_TEST_CLASS_RE = re.compile(r"class (\w+)\(unittest\.TestCase\)")


def slot_test_class(task: "Task", slot_id: str) -> str | None:
    return task.slot_tests.get(slot_id)


def class_level_tests(task: "Task") -> list[str]:
    own = set(task.slot_tests.values())
    return [t for t in task.test_classes if t not in own]


@lru_cache(maxsize=512)
def _test_class_nodes(test_src: str) -> dict[str, ast.ClassDef]:
    try:
        mod = ast.parse(test_src)
    except SyntaxError:
        return {}
    return {n.name: n for n in ast.walk(mod) if isinstance(n, ast.ClassDef)}


def test_class_source(task: "Task", test_class: str) -> str | None:
    node = _test_class_nodes(task.test_src).get(test_class)
    return ast.unparse(node) if node is not None else None


def test_slot_deps(task: "Task", test_class: str) -> set[str]:
    """Over-approximation: every `.<slot>(` call and `.<slot>` reference in the test class body."""
    node = _test_class_nodes(task.test_src).get(test_class)
    if node is None:
        return set(task.slot_ids)
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and n.attr in task.slot_ids}


def predicted_inconclusive(task: "Task", test_class: str, bound: set[str]) -> set[str]:
    """Slots the test needs that the composition leaves stubbed. Non-empty => record `inconclusive`."""
    return test_slot_deps(task, test_class) - bound


def slots_in_traceback(task: "Task", detail: str) -> set[str]:
    """Slot names appearing as frames (`in <name>`) or stub hits in a traceback."""
    frames = set(re.findall(r", in (\w+)", detail)) | set(re.findall(r"CalmStubHit: '(\w+)'", detail))
    return frames & task.slot_ids
