"""Latency mechanics: concurrent phase 2, stub-test dedup, and cross-run outcome reuse.

Every assertion here is about *when* facts appear or how often a subprocess runs. The facts themselves
(verified compositions, recorded outcomes) must be the same as the sequential path.
"""
import asyncio

import pytest

import calm_coder.agents.scheduler as sched_mod
from calm_coder.agents.scheduler import Scheduler
from calm_coder.runner.cache import OutcomeCache
from calm_coder.store.defs import Composition, emission_to_defs
from calm_coder.store.derive import verified
from calm_coder.store.store import Store
from calm_coder.task import Task
from conftest import TOY_SKELETON, TOY_TESTS, fn

GOOD = {
    "put": "def put(self, key, value):\n    self.items[key.strip().lower()] = value",
    "get": "def get(self, key):\n    return self.items[key.strip().lower()]",
    "norm": "def norm(key):\n    return key.strip().lower()",
}
BAD = {
    "put": "def put(self, key, value):\n    self.items[key] = value",
    "get": "def get(self, key):\n    return self.items[key]",
    "norm": "def norm(key):\n    return key",
}


def _add(task, store: Store, slot: str, src: str) -> str:
    ds = emission_to_defs(task, task.slot(slot), [fn(src)], {})
    for d in ds:
        store.add_def(d)
    return next(d.hash for d in ds if d.kind == "fill")


def _good_store(toy) -> tuple[Store, Composition]:
    s = Store()
    return s, Composition.make({slot: _add(toy, s, slot, src) for slot, src in GOOD.items()})


def _spy(monkeypatch) -> list[list[str]]:
    """Record the test classes handed to the sandbox, so reuse is visible as an absent subprocess."""
    calls: list[list[str]] = []
    real = sched_mod.run_tests_async

    async def spy(src, test_src, classes, **kw):
        calls.append(list(classes))
        return await real(src, test_src, classes, **kw)

    monkeypatch.setattr(sched_mod, "run_tests_async", spy)
    return calls


def test_concurrent_stub_tests_of_one_fill_run_once(toy, monkeypatch):
    calls = _spy(monkeypatch)
    store, comp = _good_store(toy)
    h = comp.binding["put"]
    sched = Scheduler(toy, store)

    async def main():
        return await asyncio.gather(*(sched.stub_test(h) for _ in range(5)))

    assert asyncio.run(main()) == ["pass"] * 5
    assert calls == [["ToyTestPut"]]


def test_class_key_covers_reachable_fills_and_the_test_module(toy):
    store = Store()
    put = _add(toy, store, "put", GOOD["put"])
    good_get, bad_get = _add(toy, store, "get", GOOD["get"]), _add(toy, store, "get", BAD["get"])
    sched = Scheduler(toy, store)
    k = lambda t, b: sched.class_key(t, Composition.make(b))
    # ToyTestPut cannot reach `get`, so swapping or unbinding it is the same reachable content...
    assert k("ToyTestPut", {"put": put, "get": good_get}) == k("ToyTestPut", {"put": put, "get": bad_get})
    assert k("ToyTestPut", {"put": put}) == k("ToyTestPut", {"put": put, "get": good_get})
    # ...while ToyTestGet does reach it, so both the fill and whether it is stubbed matter.
    assert k("ToyTestGet", {"put": put, "get": good_get}) != k("ToyTestGet", {"put": put, "get": bad_get})
    assert k("ToyTestGet", {"put": put}) != k("ToyTestGet", {"put": put, "get": good_get})
    other = Task.from_skeleton(task_id="toy2", skeleton=TOY_SKELETON, test_src=TOY_TESTS + "\n# changed\n",
                               slot_tests=dict(toy.slot_tests))
    assert Scheduler(other, store).class_key("ToyTestPut", Composition.make({"put": put})) \
        != k("ToyTestPut", {"put": put})


def test_run_full_reuses_class_results_across_compositions(toy, monkeypatch):
    calls = _spy(monkeypatch)
    store, comp = _good_store(toy)
    other_get = "def get(self, key):\n    k = key.strip().lower()\n    return self.items[k]"
    alt = comp.with_("get", _add(toy, store, "get", other_get))
    sched = Scheduler(toy, store, reuse_class_outcomes=True)
    asyncio.run(sched.run_full(comp))
    res, _, cached = asyncio.run(sched.run_full(alt))
    assert res == {t: "pass" for t in toy.test_classes}
    assert calls[-1] == ["ToyTestGet"]        # only the class that can reach the changed slot re-ran
    assert not cached
    assert verified(store, toy) == {comp.id, alt.id}


def test_cache_reuse_across_runs_records_outcomes_and_marks_them(toy, monkeypatch, tmp_path):
    calls = _spy(monkeypatch)
    cache = tmp_path / "outcomes.jsonl"
    first, comp = _good_store(toy)
    asyncio.run(Scheduler(toy, first, cache=OutcomeCache(cache)).run_full(comp))
    assert calls and set(calls[0]) == set(toy.test_classes)

    second = Store()
    for slot, src in GOOD.items():
        _add(toy, second, slot, src)
    calls.clear()
    res, _, cached = asyncio.run(Scheduler(toy, second, cache=OutcomeCache(cache)).run_full(comp))
    assert calls == [] and cached                      # nothing executed on the second run
    assert res == {t: "pass" for t in toy.test_classes}
    assert verified(second, toy) == {comp.id}
    assert all(o.detail.startswith("reused:") for o in second.outcomes())


def test_phase1_stub_outcome_is_reused_in_phase2(toy, monkeypatch):
    calls = _spy(monkeypatch)
    store, comp = _good_store(toy)
    sched = Scheduler(toy, store, reuse_class_outcomes=True)
    asyncio.run(sched.stub_test(comp.binding["norm"]))       # `norm` calls nothing, so the stub context
    calls.clear()                                            # is the same reachable content as the full one
    asyncio.run(sched.run_full(comp))
    assert "ToyTestNorm" not in [t for c in calls for t in c]


def test_fail_fast_stops_a_composition_at_the_first_failing_class(toy, monkeypatch):
    calls = _spy(monkeypatch)
    store = Store()
    comp = Composition.make({s: _add(toy, store, s, GOOD[s] if s != "get" else BAD["get"]) for s in GOOD})
    sched = Scheduler(toy, store, fail_fast=True)
    res, _, _ = asyncio.run(sched.run_full(comp))
    assert all(len(c) == 1 for c in calls)                   # a class at a time, so one failure can end it
    assert len(calls) <= len(toy.test_classes)
    assert res["ToyTestGet"] != "pass"                       # the failure that ended it is still a fact
    assert {o.test_id for o in store.outcomes()} == set(res)
    assert comp.id not in verified(store, toy)


@pytest.mark.parametrize("width", [1, 4])
def test_parallel_search_derives_the_same_facts(toy, width):
    store = Store()
    cands, arrival = {}, {}
    for slot in GOOD:
        bad, good = _add(toy, store, slot, BAD[slot]), _add(toy, store, slot, GOOD[slot])
        cands[slot] = [bad, good]
        arrival[bad], arrival[good] = 0, 1             # the broken fill is tried first
    sched = Scheduler(toy, store, width=width)
    res = asyncio.run(sched.search(cands, {}, arrival))
    assert res.verified_comp in verified(store, toy)
    assert res.test_wall_ms >= 0
    good_only = Composition.make({s: cands[s][1] for s in cands})
    assert res.verified_comp == good_only.id           # the only composition that can verify
