"""Monotone, pure derivations over the store (I3). ∃ / ∧ / closure only; no argmax, no latest."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, FrozenSet

from calm_coder.store.defs import Composition, Definition
from calm_coder.store.store import Store

if TYPE_CHECKING:
    from calm_coder.task import Task


def reachable(store: Store, comp: Composition) -> tuple[set[str], bool]:
    """Hashes reachable from comp's bound fills through helper_refs, and whether all were present."""
    seen: set[str] = set()
    frontier = [h for _, h in comp.bindings]
    ok = True
    while frontier:
        h = frontier[-1]
        frontier = frontier[:-1]
        if h in seen:
            continue
        seen.add(h)
        d = store.get(h)
        if d is None:
            ok = False
            continue
        frontier = frontier + sorted(d.helper_refs)
    return seen, ok


def complete(store: Store, task: "Task", comp: Composition) -> bool:
    """Every slot referenced (transitively) is bound, every helper hash is present, and no reached
    def has unresolved refs. For a fixed comp, only False -> True as the store grows."""
    bound = comp.binding
    hashes, present = reachable(store, comp)
    if not present:
        return False
    for h in hashes:
        d = store.get(h)
        if d.unresolved_refs or not d.slot_refs <= bound.keys():
            return False
    return all(store.get(h).slot == s for s, h in bound.items())


def slot_evidence(store: Store, task: "Task", d: Definition) -> dict[str, int]:
    """Scheduler input only: results of d.slot's own test class on compositions binding d."""
    t = task.slot_tests.get(d.slot)
    c = Counter({"pass": 0, "fail": 0, "error": 0, "timeout": 0, "inconclusive": 0})
    for o in store.outcomes():
        if o.test_id == t and (d.slot, d.hash) in o.bindings:
            c[o.result] += 1
    return dict(c)


def verified(store: Store, task: "Task") -> FrozenSet[str]:
    """comp_ids with an Outcome(t, comp, 'pass') for every t in task.test_classes."""
    passed: dict[str, set[str]] = {}
    for o in store.outcomes():
        if o.result == "pass":
            passed.setdefault(o.comp_id, set()).add(o.test_id)
    need = set(task.test_classes)
    return frozenset(c for c, ts in passed.items() if need <= ts)


def done(store: Store, task: "Task") -> bool:
    return len(verified(store, task)) > 0
