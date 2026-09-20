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
import re
from typing import Iterable

from calm_coder.serve.extract import ExtractError, extract_functions

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


def _pending_helpers(fn: ast.FunctionDef, defined: Iterable[str], slots: Iterable[str]) -> bool:
    """True if the method calls a private helper on self/cls that has not been emitted yet."""
    have, declared = set(defined), set(slots)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        recv, name = node.func.value, node.func.attr
        if isinstance(recv, ast.Name) and recv.id in ("self", "cls") \
                and name.startswith("_") and name not in have and name not in declared:
            return True
    return False


def truncation_point(text: str, slot: str, slots: Iterable[str] = ()) -> int | None:
    """Index at which `text` may be cut without losing the fill for `slot`, or None to keep decoding.

    The method is over once a non-blank line is no more indented than its `def`. That line is only a
    stop if it does not start another definition the fill might need: a decorator or a `def _helper`
    at the same indentation means helpers are still arriving, so decoding continues through them. A
    method that calls a helper it has not emitted yet is never cut either.
    """
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
        if m and _indent(ln) == body_indent and m.group("name").startswith("_"):
            i += 1                                                # a helper the fill may call
            continue
        return _cut(text, pos, slot, slots)
    return None


def _cut(text: str, pos: int, slot: str, slots: Iterable[str]) -> int | None:
    """Confirm the prefix really extracts into the slot's fill before calling it a stop."""
    fns = extract_functions(text[:pos], class_name="")
    if isinstance(fns, ExtractError):
        return None
    fill = next((f for f in fns if f.name == slot), None)
    if fill is None or _pending_helpers(fill, {f.name for f in fns}, slots):
        return None
    return pos
