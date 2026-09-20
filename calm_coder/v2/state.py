"""Policy reads over the store: the current best class and the dead slots to repair.

These are *scheduling* reads. `best_class` is an argmax and `dead_slots` is a negation, so neither
may live in `store/derive.py`: I3 keeps derivations monotone. What these functions decide is *what
to generate next*, never which facts are derivable. Everything they read is immutable, so a repair
round is reproducible from the store contents at the round boundary (see ARCHITECTURE.md).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from calm_coder.runner import tests as rt
from calm_coder.store.defs import Composition
from calm_coder.store.store import Store
from calm_coder.v2.feedback import Failure, parse_failure

if TYPE_CHECKING:
    from calm_coder.task import Task

PASS = "pass"


def comp_results(store: Store) -> dict[str, dict[str, tuple[str, str]]]:
    """comp_id -> test class -> (result, detail). A pass anywhere wins (outcomes are never retracted)."""
    out: dict[str, dict[str, tuple[str, str]]] = {}
    for o in sorted(store.outcomes(), key=lambda o: (o.comp_id, o.test_id, o.result)):
        cur = out.setdefault(o.comp_id, {})
        if cur.get(o.test_id, ("", ""))[0] != PASS:
            cur[o.test_id] = (o.result, o.detail)
    return out


def comp_bindings(store: Store) -> dict[str, dict[str, str]]:
    return {o.comp_id: dict(o.bindings) for o in sorted(store.outcomes(), key=lambda o: o.comp_id) if o.bindings}


def best_class(store: Store, task: "Task") -> Composition | None:
    """The fully bound composition with the most passing test classes.

    Ties: fewest failing tests, then lowest comp id. Deterministic given the store contents.
    """
    results, binds = comp_results(store), comp_bindings(store)
    slot_ids = {s.id for s in task.slots}
    best: tuple[tuple[int, int, str], Composition] | None = None
    for comp_id, res in sorted(results.items()):
        b = binds.get(comp_id)
        if not b or set(b) != slot_ids:
            continue
        passed = sum(1 for t in task.test_classes if res.get(t, ("", ""))[0] == PASS)
        failed = sum(1 for t in task.test_classes if res.get(t, ("", ""))[0] not in (PASS, ""))
        key = (-passed, failed, comp_id)
        if best is None or key < best[0]:
            best = (key, Composition.make(b))
    return best[1] if best is not None else None


def passing_candidates(store: Store, task: "Task", slot: str) -> frozenset[str]:
    """Candidates for `slot` that passed the slot's own test class in some evaluated context."""
    t = task.slot_tests.get(slot)
    if t is None:
        return frozenset()
    return frozenset(h for o in store.outcomes() if o.test_id == t and o.result == PASS
                     for s, h in o.bindings if s == slot)


def failing_tests(store: Store, task: "Task", comp: Composition) -> dict[str, tuple[str, str]]:
    res = comp_results(store).get(comp.id, {})
    return {t: res[t] for t in task.test_classes if t in res and res[t][0] != PASS}


def blamed_slots(task: "Task", test_class: str, detail: str) -> set[str]:
    return rt.test_slot_deps(task, test_class) | rt.slots_in_traceback(task, detail)


def dead_slots(store: Store, task: "Task", comp: Composition | None = None) -> tuple[str, ...]:
    """Slots with no candidate that passes their own test class, plus — for slots that have no own
    test class — the slots blamed for the current best class's failures."""
    comp = comp if comp is not None else best_class(store, task)
    blamed: set[str] = set()
    if comp is not None:
        for t, (_, detail) in failing_tests(store, task, comp).items():
            blamed |= blamed_slots(task, t, detail)
    out = []
    for slot in task.slots:
        if task.slot_tests.get(slot.id) is not None:
            if not passing_candidates(store, task, slot.id):
                out.append(slot.id)
        elif slot.id in blamed:
            out.append(slot.id)
    if not out and comp is not None and failing_tests(store, task, comp):
        # Nothing is attributable: the class fails but no slot owns the failure. Targeting every
        # slot is a scheduling guess, so `unattributed_slots` names it as one rather than letting
        # it be read as "this task has N dead slots".
        out = sorted(blamed) or [s.id for s in task.slots]
    return tuple(sorted(out))


def unattributed(store: Store, task: "Task", comp: Composition | None = None) -> bool:
    """True when the dead slots are the fallback guess, not slots a failure actually blames."""
    comp = comp if comp is not None else best_class(store, task)
    if comp is None or not failing_tests(store, task, comp):
        return False
    owned = any(task.slot_tests.get(s.id) is not None and not passing_candidates(store, task, s.id)
                for s in task.slots)
    blamed = set()
    for t, (_, detail) in failing_tests(store, task, comp).items():
        blamed |= blamed_slots(task, t, detail)
    return not owned and not blamed


def slot_failures(store: Store, task: "Task", slot: str, comp: Composition | None) -> list[Failure]:
    """Failures to show a repair prompt for `slot`: its own test class first, then class-level
    failures that blame it."""
    if comp is None:
        return []
    own = task.slot_tests.get(slot)
    out: list[Failure] = []
    for t, (_, detail) in sorted(failing_tests(store, task, comp).items(), key=lambda kv: (kv[0] != own, kv[0])):
        if t != own and slot not in blamed_slots(task, t, detail):
            continue
        f = parse_failure(t, detail, task.test_src)
        if f is not None:
            out.append(f)
    return out


def class_source(store: Store, task: "Task", comp: Composition | None) -> str:
    """Source of the current best class, for the repair prompt. Empty when nothing has been tested."""
    if comp is None:
        return ""
    from calm_coder.store.materialize import materialize
    return materialize(task, store, comp)


def slot_candidates(store: Store, task: "Task", slots: Sequence[str] | None = None) -> dict[str, list[str]]:
    ids = [s.id for s in task.slots] if slots is None else list(slots)
    return {s: sorted(d.hash for d in store.defs_for_slot(s)) for s in ids}


def dead_slot_report(store: Store, task: "Task", comp: Composition | None = None) -> dict[str, Mapping]:
    """Per dead slot: how many distinct candidates exist and what their failures look like."""
    comp = comp if comp is not None else best_class(store, task)
    guess = unattributed(store, task, comp)
    out = {}
    for slot in dead_slots(store, task, comp):
        fs = slot_failures(store, task, slot, comp)
        out[slot] = {
            "candidates": len(store.defs_for_slot(slot)),
            "exc_types": sorted({f.exc_type for f in fs}),
            "interface_error": any(f.interface_error for f in fs),
            "has_own_test": task.slot_tests.get(slot) is not None,
            "unattributed": guess,
        }
    return out
