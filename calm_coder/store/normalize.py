"""Canonical form of an emitted function (spec §3.1).

Canonicalization is what makes identity = body: docstrings, comments, formatting and
local variable names are erased; helper names are replaced by their content hashes.

Deviation from §3.1 (logged in §9): a *fill's* parameters are not alpha-renamed. Materialization
is done from the canonical form, and tests call methods with keyword arguments, so a slot's
parameter names are part of the interface. Helper parameters are renamed (drop_name=True):
helpers are private, and a keyword call into a renamed helper fails its tests, which is a fact.
"""
from __future__ import annotations

import ast
import copy
from typing import Iterable

RECEIVERS = ("self", "cls")


def first_param(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    ps = fn.args.posonlyargs + fn.args.args
    return ps[0].arg if ps else None


def has_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    return any(isinstance(d, ast.Name) and d.id == name for d in fn.decorator_list)


def is_static_fn(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return has_decorator(fn, "staticmethod") or first_param(fn) not in RECEIVERS


def _param_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arguments):
            for a in node.posonlyargs + node.args + node.kwonlyargs:
                out.add(a.arg)
            if node.vararg:
                out.add(node.vararg.arg)
            if node.kwarg:
                out.add(node.kwarg.arg)
    return out


def _top_params(fn) -> set[str]:
    """Parameters of fn itself, minus a self/cls receiver."""
    a = fn.args
    ps = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    ps += [x.arg for x in (a.vararg, a.kwarg) if x]
    return set(ps) - set(RECEIVERS)


def _bound_locals(fn: ast.AST, rename_params: bool) -> set[str]:
    """Names bound in the function that are safe to alpha-rename."""
    bound: set[str] = set()
    declared: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            declared |= set(node.names)
    top = _top_params(fn) if rename_params else set()
    nested = _param_names(ast.Module(body=fn.body, type_ignores=[]))
    return ((bound | top) - declared - nested - set(RECEIVERS)) if rename_params else \
        (bound - declared - _param_names(fn) - set(RECEIVERS))


class _Order(ast.NodeVisitor):
    """Records bound names in order of first appearance (field order, name-invariant)."""

    def __init__(self, bound: set[str]):
        self.bound = bound
        self.order: list[str] = []
        self._seen: set[str] = set()

    def _see(self, name: str | None) -> None:
        if name in self.bound and name not in self._seen:
            self._seen.add(name)
            self.order.append(name)

    def visit_Name(self, node: ast.Name) -> None:
        self._see(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self._see(node.arg)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._see(node.name)
        self.generic_visit(node)

    def _visit_named(self, node) -> None:
        self._see(node.name)
        self.generic_visit(node)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _visit_named


class _Rename(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str], top: ast.AST):
        self.m = mapping
        self.top = top

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self.m.get(node.id, node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self.m.get(node.arg, node.arg)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            node.name = self.m.get(node.name, node.name)
        return self.generic_visit(node)

    def _named(self, node):
        if node is not self.top:
            node.name = self.m.get(node.name, node.name)
        return self.generic_visit(node)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _named


def alpha_rename(fn: ast.AST, *, rename_params: bool = False) -> None:
    """In place: rename bound locals (and, for helpers, parameters) to v0, v1, ... avoiding free names."""
    bound = _bound_locals(fn, rename_params)
    if not bound:
        return
    free = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} - bound
    free |= _param_names(fn) - bound
    order = _Order(bound)
    order.visit(fn)
    mapping: dict[str, str] = {}
    i = 0
    for name in order.order:
        while f"v{i}" in free:
            i += 1
        mapping[name] = f"v{i}"
        i += 1
    _Rename(mapping, fn).visit(fn)


def referenced_names(fn: ast.AST, names: Iterable[str], receivers: Iterable[str]) -> set[str]:
    """Members of `names` referenced as `<receiver>.n` or as a bare Load `n`."""
    names = set(names)
    recv = set(receivers)
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in recv:
            if node.attr in names:
                out.add(node.attr)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
            out.add(node.id)
    return out


def rewrite_refs(fn: ast.AST, rename: dict[str, str], receivers: Iterable[str]) -> None:
    """In place: `<receiver>.n` -> `<receiver>.rename[n]`, bare Load `n` -> `rename[n]`."""
    recv = set(receivers)
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in recv:
            if node.attr in rename:
                node.attr = rename[node.attr]
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in rename:
            node.id = rename[node.id]


def canonicalize(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    rename_helper: dict[str, str],
    drop_name: bool,
    class_name: str | None = None,
    alpha: bool = True,
) -> str:
    fn = copy.deepcopy(fn)
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    fn.body = body or [ast.Pass()]
    fn.decorator_list = [d for d in fn.decorator_list
                         if not (isinstance(d, ast.Name) and d.id == "staticmethod")]
    if alpha:
        alpha_rename(fn, rename_params=drop_name)
    receivers = RECEIVERS + ((class_name,) if class_name else ())
    rewrite_refs(fn, rename_helper, receivers)
    if drop_name:
        fn.name = "_h"
    return ast.unparse(fn)
