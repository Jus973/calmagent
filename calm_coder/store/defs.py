"""Data model (§2) and emission -> definitions (§3.2)."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, FrozenSet

from calm_coder.store.normalize import (
    RECEIVERS, canonicalize, is_static_fn, referenced_names,
)

if TYPE_CHECKING:
    from calm_coder.task import Task


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def helper_name(h: str) -> str:
    return f"_h_{h[:12]}"


@dataclass(frozen=True)
class Slot:
    id: str
    signature: str
    docstring: str
    is_static: bool
    order: int


@dataclass(frozen=True, eq=False)
class Definition:
    """Identity is the hash. raw_src and meta are provenance of the first emission seen;
    nothing derived reads them (materialization uses canonical_src)."""
    hash: str
    kind: Literal["fill", "helper"]
    slot: str | None
    name: str
    canonical_src: str
    raw_src: str
    slot_refs: FrozenSet[str]
    helper_refs: FrozenSet[str]
    unresolved_refs: FrozenSet[str]
    is_static: bool
    meta: Mapping[str, object] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Definition) and other.hash == self.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    def to_json(self) -> dict:
        return {
            "hash": self.hash, "kind": self.kind, "slot": self.slot, "name": self.name,
            "canonical_src": self.canonical_src, "raw_src": self.raw_src,
            "slot_refs": sorted(self.slot_refs), "helper_refs": sorted(self.helper_refs),
            "unresolved_refs": sorted(self.unresolved_refs), "is_static": self.is_static,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, d: dict) -> "Definition":
        return cls(
            hash=d["hash"], kind=d["kind"], slot=d["slot"], name=d["name"],
            canonical_src=d["canonical_src"], raw_src=d["raw_src"],
            slot_refs=frozenset(d["slot_refs"]), helper_refs=frozenset(d["helper_refs"]),
            unresolved_refs=frozenset(d["unresolved_refs"]), is_static=d["is_static"],
            meta=d.get("meta", {}),
        )


@dataclass(frozen=True, eq=False)
class Composition:
    id: str
    bindings: tuple[tuple[str, str], ...]   # sorted (slot_id, fill_hash); unbound slots are stubs

    @classmethod
    def make(cls, bindings: Mapping[str, str]) -> "Composition":
        pairs = tuple(sorted(bindings.items()))
        return cls(id=sha256("comp|" + "|".join(f"{s}={h}" for s, h in pairs)), bindings=pairs)

    @property
    def binding(self) -> dict[str, str]:
        return dict(self.bindings)

    def with_(self, slot_id: str, fill_hash: str) -> "Composition":
        return Composition.make({**self.binding, slot_id: fill_hash})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Composition) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class Outcome:
    """Identity is (test_id, comp_id, result). A rerun with the same result is a duplicate;
    a flaky rerun with a different result is a second fact (verified uses ∃ pass)."""
    test_id: str
    comp_id: str
    result: Literal["pass", "fail", "error", "timeout", "inconclusive"]
    detail: str = field(default="", compare=False)
    wall_ms: int = field(default=0, compare=False)
    bindings: tuple[tuple[str, str], ...] = field(default=(), compare=False)

    def to_json(self) -> dict:
        return {"test_id": self.test_id, "comp_id": self.comp_id, "result": self.result,
                "detail": self.detail, "wall_ms": self.wall_ms,
                "bindings": [list(p) for p in self.bindings]}

    @classmethod
    def from_json(cls, d: dict) -> "Outcome":
        return cls(test_id=d["test_id"], comp_id=d["comp_id"], result=d["result"],
                   detail=d.get("detail", ""), wall_ms=d.get("wall_ms", 0),
                   bindings=tuple(tuple(p) for p in d.get("bindings", ())))


# ---------------------------------------------------------------- emission -> definitions

def _sccs(nodes: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan. Returns SCCs callees-first (reverse topological order of the call graph)."""
    # Local scratch state only; written without list/set removal calls so the I1 grep stays literal.
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: dict[str, bool] = {}
    out: list[list[str]] = []

    def visit(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on[v] = True
        for w in sorted(edges[v]):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif on.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            cut = len(stack) - stack[::-1].index(v) - 1
            comp = stack[cut:]
            stack[cut:] = []
            for w in comp:
                on[w] = False
            out.append(sorted(comp))

    for v in nodes:
        if v not in index:
            visit(v)
    return out


def _unresolved(fn: ast.AST, known: set[str], class_name: str) -> set[str]:
    recv = set(RECEIVERS) | {class_name}
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in recv:
            n = node.func.attr
            if n not in known and not (n.startswith("__") and n.endswith("__")):
                out.add(n)
    return out


def emission_to_defs(task: "Task", slot: Slot, emitted_fns: list[ast.FunctionDef],
                     meta: Mapping[str, object], *, log: list[dict] | None = None,
                     alpha: bool = True) -> list[Definition]:
    log = log if log is not None else []
    fill: ast.FunctionDef | None = None
    helpers: dict[str, ast.FunctionDef] = {}
    for fn in emitted_fns:
        if fn.name == slot.id:
            if fill is None:
                fill = fn
            else:
                log.append({"event": "duplicate_fill", "slot": slot.id})
        elif fn.name in task.slot_ids or fn.name == "__init__":
            log.append({"event": "stray_slot_def", "slot": slot.id, "name": fn.name})
        elif fn.name in helpers:
            log.append({"event": "duplicate_helper", "slot": slot.id, "name": fn.name})
        else:
            helpers[fn.name] = fn
    if fill is None:
        log.append({"event": "no_fill", "slot": slot.id})
        return []

    receivers = RECEIVERS + (task.class_name,)
    hnames = set(helpers)
    edges = {h: referenced_names(fn, hnames, receivers) for h, fn in helpers.items()}

    rename: dict[str, str] = {}       # helper name -> _h_<hash12>
    full: dict[str, str] = {}         # helper name -> full hash
    canon: dict[str, str] = {}
    for scc in _sccs(sorted(helpers), edges):
        cyclic = len(scc) > 1 or scc[0] in edges[scc[0]]
        if not cyclic:
            h = scc[0]
            canon[h] = canonicalize(helpers[h], rename_helper=rename, drop_name=True,
                                    class_name=task.class_name, alpha=alpha)
            full[h] = sha256("helper|" + canon[h])
            rename = {**rename, h: helper_name(full[h])}
            continue
        log.append({"event": "cyclic_helpers", "slot": slot.id, "names": scc})
        placeholder = {**rename, **{m: "_h_rec" for m in scc}}
        pre = {m: canonicalize(helpers[m], rename_helper=placeholder, drop_name=True,
                               class_name=task.class_name, alpha=alpha) for m in scc}
        ranked = sorted(scc, key=lambda m: (pre[m], m))
        scc_hash = sha256("scc|" + "\n".join(sorted(pre.values())))
        for rank, m in enumerate(ranked):
            full[m] = sha256(f"helper-scc|{scc_hash}|{rank}")
        rename = {**rename, **{m: helper_name(full[m]) for m in scc}}
        for m in scc:
            canon[m] = canonicalize(helpers[m], rename_helper=rename, drop_name=True,
                                    class_name=task.class_name, alpha=alpha)

    slot_ids = task.slot_ids - {"__init__"}
    known = set(slot_ids) | hnames | set(task.fields)
    defs: list[Definition] = []
    for h, fn in helpers.items():
        defs.append(Definition(
            hash=full[h], kind="helper", slot=None, name=h, canonical_src=canon[h],
            raw_src=ast.unparse(fn),
            slot_refs=frozenset(referenced_names(fn, slot_ids, receivers)),
            helper_refs=frozenset(full[x] for x in edges[h]),
            unresolved_refs=frozenset(_unresolved(fn, known, task.class_name)),
            is_static=is_static_fn(fn), meta=dict(meta),
        ))
    fill_canon = canonicalize(fill, rename_helper=rename, drop_name=False,
                              class_name=task.class_name, alpha=alpha)
    defs.append(Definition(
        hash=sha256("fill|" + slot.id + "|" + fill_canon), kind="fill", slot=slot.id,
        name=slot.id, canonical_src=fill_canon, raw_src=ast.unparse(fill),
        slot_refs=frozenset(referenced_names(fill, slot_ids, receivers)),
        helper_refs=frozenset(full[x] for x in referenced_names(fill, hnames, receivers)),
        unresolved_refs=frozenset(_unresolved(fill, known, task.class_name)),
        is_static=slot.is_static or is_static_fn(fill), meta=dict(meta),
    ))
    return defs
