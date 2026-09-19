"""Whole-class sample -> slot candidates in the same grow-only store.

This is what makes the store producer-agnostic: a whole-class sample and a per-method sample land
as the same kind of fact. The sample's own composition is returned so the search can try it first,
which is the no-loss property (a pooled run can never do worse than the samples it pooled).

A method is not self-contained, so lifting it out of its sample is only faithful if what the
sample put *around* it comes too: its imports, its module-level constants and its module-level
helpers. Imports and constants are inlined into the bodies that reference them (an import inside
a function binds the same name), module-level functions become ordinary helpers, and the fill's
hash covers the result — a method carrying `import re` is a different definition from one that
does not. What cannot follow the method: the sample's own `__init__` and its class attributes,
because the constructor is the skeleton's shared contract (§3.5). `ClassIngest.dropped` names them
so a sample whose composition is not the sample can be counted rather than silently scored.
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
PRELUDE = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)


@dataclass
class ClassIngest:
    bindings: dict[str, str] = field(default_factory=dict)   # slot -> fill hash, this sample's own class
    def_hashes: list[str] = field(default_factory=list)
    error: str | None = None
    dropped: list[str] = field(default_factory=list)         # sample context a composition cannot carry

    @property
    def complete(self) -> bool:
        return self.error is None and bool(self.bindings)


def _binds(stmt: ast.stmt) -> set[str]:
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in stmt.names}
    if isinstance(stmt, FUNC + (ast.ClassDef,)):
        return {stmt.name}
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    return {t.id for t in ast.walk(ast.Module(body=list(targets), type_ignores=[]))
            if isinstance(t, ast.Name)}


def _reads(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _with_prelude(fn: ast.AST, prelude: list[ast.stmt]) -> ast.AST:
    """`fn` with the module-level statements it reads (transitively) inlined at the top of its body.

    A function that reads a name its module bound is only that function together with the binding;
    copying the binding in is what keeps the lifted method equal to the method as sampled.
    """
    if not prelude:
        return fn
    need, keep = _reads(fn), []
    for _ in range(len(prelude)):
        picked = [s for s in prelude if s not in keep and _binds(s) & need]
        if not picked:
            break
        keep += picked
        need |= {n for s in picked for n in _reads(s)}
    if not keep:
        return fn
    out = ast.parse(ast.unparse(fn)).body[0]      # a copy; the original stays in the sample's AST
    doc = 1 if (out.body and isinstance(out.body[0], ast.Expr)
                and isinstance(getattr(out.body[0], "value", None), ast.Constant)) else 0
    ordered = [s for s in prelude if s in keep]
    out.body = out.body[:doc] + [ast.parse(ast.unparse(s)).body[0] for s in ordered] + out.body[doc:]
    return ast.fix_missing_locations(out)


def _dropped(cdef: ast.ClassDef, task: "Task") -> list[str]:
    """Sample context no composition can carry: its own constructor and its class attributes."""
    out = []
    if any(isinstance(n, FUNC) and n.name == "__init__" for n in cdef.body):
        skel = ast.parse(task.skeleton)
        skel_init = next((ast.unparse(n) for c in skel.body if isinstance(c, ast.ClassDef)
                          for n in c.body if isinstance(n, FUNC) and n.name == "__init__"), "")
        own = next(ast.unparse(n) for n in cdef.body if isinstance(n, FUNC) and n.name == "__init__")
        if own != skel_init:
            out.append("__init__")
    out += sorted(n for s in cdef.body if isinstance(s, (ast.Assign, ast.AnnAssign)) for n in _binds(s))
    return out


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
    # The skeleton's own imports are already in every materialized class, so inlining them would
    # only make a lifted fill hash differently from the same method sampled per-method.
    skel_imports = {ln.strip() for ln in task.import_statement}
    prelude = [n for n in mod.body if n is not cdef and isinstance(n, PRELUDE)
               and ast.unparse(n) not in skel_imports]
    mod_fns = [n for n in mod.body if isinstance(n, FUNC)]
    extra = [f for n, f in fns.items() if n not in task.slot_ids and n != "__init__"] + mod_fns
    out = ClassIngest(dropped=_dropped(cdef, task))
    for slot in task.slots:
        if slot.id not in fns:
            continue
        group = [_with_prelude(f, prelude) for f in (fns[slot.id], *extra)]
        defs = emission_to_defs(task, slot, group, dict(meta))
        if not defs:
            continue
        for d in defs:
            store.add_def(d)
            out.def_hashes.append(d.hash)
        out.bindings[slot.id] = next(d.hash for d in defs if d.kind == "fill")
    if len(out.bindings) < len(task.slots):
        out.error = f"missing slots: {sorted(set(s.id for s in task.slots) - set(out.bindings))}"
    return out
