import ast

import pytest

from calm_coder.bench.classeval import reference_emissions, task_from_row
from calm_coder.runner.sandbox import run_tests
from calm_coder.store.defs import Composition, emission_to_defs
from calm_coder.store.derive import complete
from calm_coder.store.materialize import materialize
from calm_coder.store.store import Store
from conftest import fn


def _reference_store(row):
    task = task_from_row(row)
    s = Store()
    bind = {}
    for slot_id, fns in reference_emissions(row, task).items():
        for d in emission_to_defs(task, task.slot(slot_id), fns, {"ref": True}):
            s.add_def(d)
            if d.kind == "fill":
                bind[slot_id] = d.hash
    return task, s, Composition.make(bind)


@pytest.mark.parametrize("i", range(5))
def test_reference_materializes_and_passes(rows, i):
    task, s, comp = _reference_store(rows[i])
    assert complete(s, task, comp)
    src = materialize(task, s, comp)
    res = run_tests(src, task.test_src, list(task.test_classes))
    assert {k: v["result"] for k, v in res.items()} == {t: "pass" for t in task.test_classes}, res


def test_same_helper_name_different_bodies_no_clash(toy):
    s = Store()
    put = emission_to_defs(toy, toy.slot("put"), [
        fn("def put(self, key, value):\n    self.items[self._norm(key)] = value"),
        fn("def _norm(self, k):\n    return k.strip().lower()")], {})
    get = emission_to_defs(toy, toy.slot("get"), [
        fn("def get(self, key):\n    return self.items[self._norm(key)]"),
        fn("def _norm(k):\n    return k.lower().strip()")], {})
    norm = emission_to_defs(toy, toy.slot("norm"), [fn("def norm(key):\n    return key.strip().lower()")], {})
    for d in put + get + norm:
        s.add_def(d)
    f = lambda ds: next(d for d in ds if d.kind == "fill").hash
    comp = Composition.make({"put": f(put), "get": f(get), "norm": f(norm)})
    src = materialize(toy, s, comp)
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Toy")
    names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
    assert len(names) == len(set(names)) and sum(n.startswith("_h_") for n in names) == 2
    res = run_tests(src, toy.test_src, list(toy.test_classes))
    assert all(v["result"] == "pass" for v in res.values()), res


def test_bare_module_level_helper_call(toy):
    s = Store()
    put = emission_to_defs(toy, toy.slot("put"), [
        fn("def put(self, key, value):\n    self.items[clean(key)] = value"),
        fn("def clean(k):\n    return k.strip().lower()")], {})
    for d in put:
        s.add_def(d)
    comp = Composition.make({"put": next(d for d in put if d.kind == "fill").hash})
    res = run_tests(materialize(toy, s, comp), toy.test_src, ["ToyTestPut", "ToyTestNorm"])
    assert res["ToyTestPut"]["result"] == "pass"
    assert res["ToyTestNorm"]["result"] == "inconclusive"


def test_unbound_slot_stub_parses(classeval_task0):
    src = materialize(classeval_task0, Store(), Composition.make({}))
    ast.parse(src)
    assert src.count("raise CalmStubHit") == len(classeval_task0.slots)
