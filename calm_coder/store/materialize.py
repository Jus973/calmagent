"""Composition -> Python module source (§3.5). A pure function of (task, defs reachable from comp).

Built from canonical_src, not raw_src: canonical_src is a function of the hash alone, so the
materialized module is independent of which emission of a def arrived first (confluence).
"""
from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

from calm_coder.store.defs import Composition, helper_name
from calm_coder.store.derive import reachable
from calm_coder.store.normalize import RECEIVERS, first_param, has_decorator
from calm_coder.store.store import Store

if TYPE_CHECKING:
    from calm_coder.task import Task

STUB_EXC = "class CalmStubHit(BaseException):\n    pass\n"   # BaseException: `except Exception` can't swallow it


def _bare_helper_calls_to_class(fn: ast.AST, helper_names: set[str], class_name: str) -> None:
    """Bare `_h_x(...)` (module-level helper in the emission) -> `Cls._h_x(...)`."""
    for node in ast.walk(fn):
        for fld, value in ast.iter_fields(node):
            if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load) and value.id in helper_names:
                setattr(node, fld, ast.Attribute(value=ast.Name(class_name, ast.Load()), attr=value.id, ctx=ast.Load()))
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, ast.Name) and isinstance(v.ctx, ast.Load) and v.id in helper_names:
                        value[i] = ast.Attribute(value=ast.Name(class_name, ast.Load()), attr=v.id, ctx=ast.Load())


def _method(src: str, *, name: str, static: bool, helper_names: set[str], class_name: str) -> str:
    fn = ast.parse(src).body[0]
    fn.name = name
    _bare_helper_calls_to_class(fn, helper_names, class_name)
    if static and first_param(fn) not in RECEIVERS and not has_decorator(fn, "staticmethod") \
            and not has_decorator(fn, "classmethod"):
        fn.decorator_list = [ast.Name("staticmethod", ast.Load())] + fn.decorator_list
    return textwrap.indent(ast.unparse(ast.fix_missing_locations(fn)), "    ")


def materialize(task: "Task", store: Store, comp: Composition, *, stub_unbound: bool = True) -> str:
    bound = comp.binding
    hashes, _ = reachable(store, comp)
    helpers = sorted((store.get(h) for h in hashes if store.get(h) and store.get(h).kind == "helper"),
                     key=lambda d: d.hash)
    hnames = {helper_name(d.hash) for d in helpers}

    parts = [STUB_EXC, *task.import_statement, ""]
    header = [f"class {task.class_name}{task.bases_src}:"]
    if task.class_docstring is not None:
        header.append(textwrap.indent(ast.unparse(ast.Expr(ast.Constant(task.class_docstring))), "    "))
    if task.init_src:
        header.append(textwrap.indent(task.init_src, "    "))
    body = ["\n".join(header)]

    for slot in task.slots:
        d = store.get(bound[slot.id]) if slot.id in bound else None
        if d is not None:
            body.append(_method(d.canonical_src, name=slot.id, static=slot.is_static or d.is_static,
                                helper_names=hnames, class_name=task.class_name))
        elif stub_unbound:
            deco = "    @staticmethod\n" if slot.is_static else ""
            body.append(f"{deco}    {slot.signature}\n        raise CalmStubHit({slot.id!r})")
    for d in helpers:
        body.append(_method(d.canonical_src, name=helper_name(d.hash), static=d.is_static,
                            helper_names=hnames, class_name=task.class_name))
    parts.append("\n\n".join(body))
    return "\n".join(parts) + "\n"
