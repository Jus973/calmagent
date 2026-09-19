from calm_coder.store.defs import Composition, emission_to_defs
from conftest import fn


def fills(defs):
    return [d for d in defs if d.kind == "fill"]


def helpers(defs):
    return [d for d in defs if d.kind == "helper"]


def test_same_body_same_hash_across_agents(toy):
    s = toy.slot("get")
    a = emission_to_defs(toy, s, [fn("def get(self, key):\n    val = self.items.get(key)\n    return val")], {"agent": 1})
    b = emission_to_defs(toy, s, [fn("def get(self, key):\n    # x\n    out = self.items.get(key)\n    return out")], {"agent": 2})
    assert fills(a)[0].hash == fills(b)[0].hash


def test_helper_name_is_metadata(toy):
    s = toy.slot("put")
    e1 = [fn("def put(self, key, value):\n    self.items[self._norm(key)] = value"),
          fn("def _norm(self, k):\n    return k.strip().lower()")]
    e2 = [fn("def put(self, key, value):\n    self.items[self._clean(key)] = value"),
          fn("def _clean(self, s):\n    return s.strip().lower()")]
    d1, d2 = emission_to_defs(toy, s, e1, {}), emission_to_defs(toy, s, e2, {})
    assert helpers(d1)[0].hash == helpers(d2)[0].hash
    assert fills(d1)[0].hash == fills(d2)[0].hash
    assert fills(d1)[0].helper_refs == {helpers(d1)[0].hash}


def test_different_helper_body_different_fill_hash(toy):
    s = toy.slot("put")
    e1 = [fn("def put(self, key, value):\n    self.items[self._norm(key)] = value"),
          fn("def _norm(self, k):\n    return k.strip().lower()")]
    e2 = [fn("def put(self, key, value):\n    self.items[self._norm(key)] = value"),
          fn("def _norm(self, k):\n    return k.lower()")]
    assert fills(emission_to_defs(toy, s, e1, {}))[0].hash != fills(emission_to_defs(toy, s, e2, {}))[0].hash


def test_refs_classified(toy):
    log = []
    defs = emission_to_defs(toy, toy.slot("get"), [
        fn("def get(self, key):\n    return self.items.get(self.norm(key)) or self.missing(key)"),
        fn("def put(self, key, value):\n    pass"),
    ], {}, log=log)
    (f,) = defs
    assert f.slot_refs == {"norm"} and f.unresolved_refs == {"missing"} and f.helper_refs == frozenset()
    assert log == [{"event": "stray_slot_def", "slot": "get", "name": "put"}]


def test_recursive_and_mutually_recursive_helpers(toy):
    s = toy.slot("get")
    e1 = [fn("def get(self, key):\n    return self._a(key)"),
          fn("def _a(self, n):\n    return 0 if not n else self._b(n[1:])"),
          fn("def _b(self, n):\n    return self._a(n)")]
    e2 = [fn("def get(self, key):\n    return self._x(key)"),
          fn("def _x(self, n):\n    return 0 if not n else self._y(n[1:])"),
          fn("def _y(self, n):\n    return self._x(n)")]
    log = []
    d1, d2 = emission_to_defs(toy, s, e1, {}, log=log), emission_to_defs(toy, s, e2, {})
    assert {d.hash for d in d1} == {d.hash for d in d2}
    assert len({d.hash for d in helpers(d1)}) == 2
    assert any(e["event"] == "cyclic_helpers" for e in log)


def test_no_fill_yields_nothing(toy):
    assert emission_to_defs(toy, toy.slot("get"), [fn("def _h(self):\n    pass")], {}) == []


def test_composition_id_order_independent():
    assert Composition.make({"a": "1", "b": "2"}).id == Composition.make({"b": "2", "a": "1"}).id
