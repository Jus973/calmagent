"""Task = a class skeleton plus its hidden test module. Built by bench/classeval.py and cli.py."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Mapping

from calm_coder.store.defs import Slot
from calm_coder.store.normalize import has_decorator

_TEST_CLASS_RE = re.compile(r"class (\w+)\(unittest\.TestCase\)")


class TaskError(ValueError):
    """The skeleton doesn't fit the spec; the caller excludes the task and logs why."""


@dataclass(frozen=True)
class Task:
    task_id: str
    class_name: str
    import_statement: tuple[str, ...]
    class_docstring: str | None
    bases_src: str                   # "" or "(Base, Other)"
    init_src: str | None             # constructor, verbatim from the skeleton (unparsed)
    slots: tuple[Slot, ...]
    test_src: str
    test_classes: tuple[str, ...]
    slot_tests: Mapping[str, str] = field(default_factory=dict)  # slot id -> its own test class
    fields: frozenset[str] = frozenset()                           # self.<x> assigned in __init__
    skeleton: str = ""

    @property
    def slot_ids(self) -> frozenset[str]:
        return frozenset(s.id for s in self.slots)

    def slot(self, slot_id: str) -> Slot:
        return next(s for s in self.slots if s.id == slot_id)

    @classmethod
    def from_skeleton(cls, *, task_id: str, skeleton: str, test_src: str,
                      test_classes: list[str] | None = None,
                      slot_tests: Mapping[str, str] | None = None,
                      import_statement: list[str] | None = None) -> "Task":
        try:
            mod = ast.parse(skeleton)
        except SyntaxError as e:
            raise TaskError(f"skeleton_syntax: {e}") from e
        classes = [n for n in mod.body if isinstance(n, ast.ClassDef)]
        if len(classes) != 1:
            raise TaskError(f"expected one class, found {len(classes)}")
        cdef = classes[0]
        if import_statement is None:
            import_statement = [ast.unparse(n) for n in mod.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        doc = ast.get_docstring(cdef, clean=False)
        init_src = None
        init_fields: set[str] = set()
        slots: list[Slot] = []
        body = cdef.body[1:] if doc is not None else cdef.body
        for node in body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise TaskError(f"non-method class member: {type(node).__name__}")
            if node.name == "__init__":
                init_src = ast.unparse(node)
                for n in ast.walk(node):
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" \
                            and isinstance(n.ctx, ast.Store):
                        init_fields.add(n.attr)
                continue
            is_static = has_decorator(node, "staticmethod")
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            slots.append(Slot(
                id=node.name,
                signature=f"{prefix} {node.name}({ast.unparse(node.args)}){ret}:",
                docstring=ast.get_docstring(node, clean=False) or "",
                is_static=is_static,
                order=len(slots),
            ))
        if not slots:
            raise TaskError("no slots")
        if test_classes is None:
            test_classes = _TEST_CLASS_RE.findall(test_src)
        bases = ", ".join(ast.unparse(b) for b in cdef.bases)
        return cls(
            task_id=task_id,
            class_name=cdef.name,
            import_statement=tuple(import_statement),
            class_docstring=doc,
            bases_src=f"({bases})" if bases else "",
            init_src=init_src,
            slots=tuple(slots),
            test_src=test_src,
            test_classes=tuple(test_classes),
            slot_tests=dict(slot_tests or {}),
            fields=frozenset(init_fields),
            skeleton=skeleton,
        )


def infer_slot_tests(task: Task) -> dict[str, str]:
    """For tasks without methods_info: a test class is slot s's own test iff s is the only slot it calls.
    Structural, never by name (§3.10). Slots without such a class get none (Phase 1 is then inconclusive)."""
    from calm_coder.runner.tests import test_slot_deps
    out: dict[str, str] = {}
    for t in task.test_classes:
        deps = test_slot_deps(task, t)
        if len(deps) == 1:
            out.setdefault(next(iter(deps)), t)
    return out


def task_from_files(skeleton_path, tests_path) -> Task:
    from pathlib import Path
    sk, tests = Path(skeleton_path), Path(tests_path)
    t = Task.from_skeleton(task_id=sk.stem, skeleton=sk.read_text(), test_src=tests.read_text())
    return Task(**{**t.__dict__, "slot_tests": infer_slot_tests(t)})
