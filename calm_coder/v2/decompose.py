"""Whole-class sample -> slot candidates in the same grow-only store.

This is what makes the store producer-agnostic: a whole-class sample and a per-method sample land
as the same kind of fact. The sample's own composition is returned so the search can try it first,
which is the no-loss property (a pooled run can never do worse than the samples it pooled).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from calm_coder.serve.extract import ExtractError, extract_class
from calm_coder.store.defs import emission_to_defs
from calm_coder.store.store import Store

if TYPE_CHECKING:
    from calm_coder.task import Task

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class ClassIngest:
    bindings: dict[str, str] = field(default_factory=dict)   # slot -> fill hash, this sample's own class
    def_hashes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.error is None and bool(self.bindings)


def ingest_class_sample(task: "Task", store: Store, text: str, meta: Mapping[str, object]) -> ClassIngest:
    src = extract_class(text, task)
    if isinstance(src, ExtractError):
        return ClassIngest(error=src.reason)
    try:
        mod = ast.parse(src)
    except SyntaxError as e:
        return ClassIngest(error=f"syntax: {e}")
    cdef = next((n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == task.class_name), None)
    if cdef is None:
        return ClassIngest(error=f"no class {task.class_name}")
    fns = {n.name: n for n in cdef.body if isinstance(n, FUNC)}
    extra = [f for n, f in fns.items() if n not in task.slot_ids and n != "__init__"]
    out = ClassIngest()
    for slot in task.slots:
        if slot.id not in fns:
            continue
        defs = emission_to_defs(task, slot, [fns[slot.id], *extra], dict(meta))
        if not defs:
            continue
        for d in defs:
            store.add_def(d)
            out.def_hashes.append(d.hash)
        out.bindings[slot.id] = next(d.hash for d in defs if d.kind == "fill")
    if len(out.bindings) < len(task.slots):
        out.error = f"missing slots: {sorted(set(s.id for s in task.slots) - set(out.bindings))}"
    return out
