"""Model text -> function ASTs (§3.8), and whole-class extraction for the baselines."""
from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calm_coder.store.normalize import RECEIVERS, first_param, has_decorator

if TYPE_CHECKING:
    from calm_coder.task import Task

_FENCE = re.compile(r"```([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class ExtractError:
    reason: str


def _candidates(text: str) -> list[str]:
    text = _THINK.sub("", text)
    blocks = _FENCE.findall(text)
    py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3")]
    anyb = [b for _, b in blocks]
    out = []
    if py:
        out += ["\n\n".join(py), *py]
    if anyb:
        out += ["\n\n".join(anyb), *anyb]
    out.append(text)
    seen, uniq = set(), []
    for c in out:
        if c.strip() and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _parse(src: str) -> ast.Module | None:
    for attempt in (lambda: ast.parse(textwrap.dedent(src)),
                    lambda: ast.parse("class _W:\n" + textwrap.indent(textwrap.dedent(src), "    "))):
        try:
            return attempt()
        except (SyntaxError, ValueError):
            continue
    return None


def _functions(mod: ast.Module, class_name: str) -> list[ast.FunctionDef]:
    fns = [n for n in mod.body if isinstance(n, FUNC)]
    classes = [n for n in mod.body if isinstance(n, ast.ClassDef)]
    target = next((c for c in classes if c.name == class_name), None) or \
        next((c for c in classes if c.name == "_W"), None) or (classes[0] if classes else None)
    if target is not None:
        fns += [n for n in target.body if isinstance(n, FUNC)]
    return fns


def extract_functions(text: str, *, class_name: str) -> list[ast.FunctionDef] | ExtractError:
    for src in _candidates(text):
        mod = _parse(src)
        if mod is None:
            continue
        fns = _functions(mod, class_name)
        if fns:
            return fns
    return ExtractError("no parseable function in output")


# ---------------------------------------------------------------- whole class (baselines A/C)

def add_static_statement(cdef: ast.ClassDef) -> None:
    """Mirror the official pipeline: a method whose first param isn't self/cls gets @staticmethod."""
    for n in cdef.body:
        if isinstance(n, FUNC) and first_param(n) not in RECEIVERS and not has_decorator(n, "staticmethod") \
                and not has_decorator(n, "classmethod"):
            n.decorator_list = [ast.Name("staticmethod", ast.Load())] + n.decorator_list


def extract_class(text: str, task: "Task") -> str | ExtractError:
    """-> module source: task imports + model's module-level code, with add_static_statement applied."""
    for src in _candidates(text):
        mod = _parse(src)
        if mod is None:
            continue
        cdef = next((n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == task.class_name), None)
        if cdef is None:
            # Bare methods: splice them into the skeleton class (keeps its constructor and docstring).
            fns = _functions(mod, task.class_name)
            if not fns:
                continue
            skel = ast.parse(task.skeleton)
            cdef = next(n for n in skel.body if isinstance(n, ast.ClassDef))
            by_name = {f.name: f for f in fns}
            cdef.body = [by_name.pop(n.name, n) if isinstance(n, FUNC) else n for n in cdef.body] + list(by_name.values())
            mod = ast.Module(body=[cdef], type_ignores=[])
        add_static_statement(cdef)
        return "\n".join(task.import_statement) + "\n\n" + ast.unparse(ast.fix_missing_locations(mod)) + "\n"
    return ExtractError(f"no class {task.class_name} in output")
