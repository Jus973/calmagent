"""Where a slot's completion can be cut off (the decode-side latency lever).

A slot request asks for one method and, optionally, the private helpers it calls. Everything a model
emits after that — a closing fence, an explanation, another declared method it was told not to write —
is decoded at the same tokens/second as the part we keep and then thrown away by `extract`. Tokens are
the wall clock on a local server, so the cut is a latency win and nothing else changes: the fill that
lands in the store is the one the full completion would have produced.

`truncation_point` is a pure function of the text so far. It returns an index only when the text
already contains a complete method for the slot *and* something after it shows the method ended;
otherwise None, meaning keep decoding.
"""
from __future__ import annotations

import ast
import builtins
import re
from typing import Iterable, Mapping

from calm_coder.serve.extract import ExtractError, extract_functions
from calm_coder.store.normalize import RECEIVERS

_DEF = re.compile(r"^(?P<indent>[ \t]*)(?:async[ \t]+)?def[ \t]+(?P<name>\w+)[ \t]*\(")
_DECORATOR = re.compile(r"^[ \t]*@")
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for line in text.splitlines(keepends=True):
        out.append((pos, line.rstrip("\n").rstrip("\r")))
        pos += len(line)
    return out


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _bound(fn: ast.FunctionDef) -> set[str]:
    """Names the function itself binds: parameters, assignment targets, imports, nested defs."""
    out = {a.arg for n in ast.walk(fn) if isinstance(n, ast.arguments)
           for a in n.posonlyargs + n.args + n.kwonlyargs + ([n.vararg] if n.vararg else [])
           + ([n.kwarg] if n.kwarg else [])}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.alias):
            out.add((node.asname or node.name).split(".")[0])
    return out


def _referenced(fn: ast.FunctionDef, recv: set[str], emitted: Mapping[str, ast.FunctionDef]) -> set[str]:
    """What `fn` calls that a class member could satisfy: `<receiver>.n(...)` and bare `n(...)`.

    Those are the two call forms `emission_to_defs` resolves against emitted helpers. A bare call to
    something the function binds itself, or to a builtin, is nobody's helper and is ignored.
    """
    bound = _bound(fn) | set(dir(builtins))
    out: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in recv:
            out.add(f.attr)
        elif isinstance(f, ast.Name) and (f.id in emitted or f.id not in bound):
            out.add(f.id)
    return out


def _pending_helpers(fill: ast.FunctionDef, emitted: Mapping[str, ast.FunctionDef],
                     slots: Iterable[str], class_name: str) -> bool:
    """True if anything reachable from the fill calls a method that is neither a declared slot nor emitted.

    The walk follows emitted helpers, so a helper needed only by another helper still holds the cut,
    and a helper called bare or through the class name counts the same as `self._h()`.
    """
    recv = set(RECEIVERS) | ({class_name} if class_name else set())
    declared, seen, queue = set(slots), set(), [fill]
    while queue:
        for name in _referenced(queue.pop(), recv, emitted):
            if name in emitted:
                if name not in seen:
                    seen.add(name)
                    queue.append(emitted[name])
            elif name not in declared and not (name.startswith("__") and name.endswith("__")):
                return True
    return False


def truncation_point(text: str, slot: str, slots: Iterable[str] = (), class_name: str = "") -> int | None:
    """Index at which `text` may be cut without losing the fill for `slot`, or None to keep decoding.

    The method is over once a non-blank line is no more indented than its `def`. That line is only a
    stop if it does not start another definition the fill might need: a decorator, or a `def` at the
    same indentation for anything other than a declared slot, means helpers are still arriving, so
    decoding continues through them. A fill whose helper graph is incomplete is never cut either.
    """
    declared = set(slots)
    if _THINK_OPEN in text and _THINK_CLOSE not in text:
        return None
    lines = _lines_with_offsets(text)
    start = next((i for i, (_, ln) in enumerate(lines)
                  if (m := _DEF.match(ln)) and m.group("name") == slot), None)
    if start is None:
        return None
    body_indent = _indent(lines[start][1])
    i = start + 1
    while i < len(lines):
        pos, ln = lines[i]
        if not ln.strip() or _indent(ln) > body_indent:
            i += 1
            continue
        if _DECORATOR.match(ln) and _indent(ln) == body_indent:   # decorates whatever comes next
            i += 1
            continue
        m = _DEF.match(ln)
        if m and _indent(ln) == body_indent and m.group("name") not in declared:
            i += 1                                                # a helper the fill may call
            continue
        stop = _cut(text, pos, slot, declared, class_name)
        if stop is not None:
            return stop
        i += 1          # a boundary we can't take yet: a helper is still missing, keep looking
    return None


def _cut(text: str, pos: int, slot: str, slots: set[str], class_name: str) -> int | None:
    """Confirm the prefix really extracts into the slot's fill before calling it a stop."""
    fns = extract_functions(text[:pos], class_name=class_name)
    if isinstance(fns, ExtractError):
        return None
    fill = next((f for f in fns if f.name == slot), None)
    if fill is None:
        return None
    emitted = {f.name: f for f in fns if f.name != slot and f.name not in slots}
    return None if _pending_helpers(fill, emitted, slots, class_name) else pos
