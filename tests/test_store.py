import random
import re
import subprocess
from pathlib import Path

from calm_coder.store.defs import Composition, Outcome, emission_to_defs
from calm_coder.store.derive import complete, done, verified
from calm_coder.store.store import Store
from conftest import fn

STORE_DIR = Path(__file__).parents[1] / "calm_coder" / "store"


def test_I1_no_mutating_calls_in_store():
    pat = re.compile(r"\b(del|remove|pop|discard|clear|update)\(")
    hits = [f"{p.name}:{i}: {ln.strip()}" for p in STORE_DIR.glob("*.py")
            for i, ln in enumerate(p.read_text().splitlines(), 1) if pat.search(ln)]
    assert hits == []
    out = subprocess.run(["grep", "-rnE", r"\b(del|remove|pop|discard|clear|update)\(", str(STORE_DIR)],
                         capture_output=True, text=True)
    assert out.stdout == ""


def _toy_events(toy):
    s = Store()
    put_a = emission_to_defs(toy, toy.slot("put"), [
        fn("def put(self, key, value):\n    self.items[self._n(key)] = value"),
        fn("def _n(self, k):\n    return k.strip().lower()")], {"agent": 0})
    put_b = emission_to_defs(toy, toy.slot("put"), [fn("def put(self, key, value):\n    self.items[key] = value")], {})
    get_a = emission_to_defs(toy, toy.slot("get"), [fn("def get(self, key):\n    return self.items[self.norm(key)]")], {})
    norm_a = emission_to_defs(toy, toy.slot("norm"), [fn("def norm(key):\n    return key.strip().lower()")], {})
    for d in put_a + put_b + get_a + norm_a + put_a:
        s.add_def(d)
    fill = lambda defs: next(d for d in defs if d.kind == "fill").hash
    good = Composition.make({"put": fill(put_a), "get": fill(get_a), "norm": fill(norm_a)})
    bad = good.with_("put", fill(put_b))
    for comp, res in ((good, "pass"), (bad, "fail")):
        for t in toy.test_classes:
            r = "pass" if (res == "pass" or t != "ToyTestPut") else "fail"
            s.add_outcome(Outcome(t, comp.id, r, bindings=comp.bindings))
    s.add_outcome(Outcome("ToyTestPut", bad.id, "pass", bindings=bad.bindings))  # flaky rerun: second fact
    return s, [good, bad]


def test_add_idempotent(toy):
    s = Store()
    (d,) = emission_to_defs(toy, toy.slot("get"), [fn("def get(self, key):\n    return 1")], {})
    assert s.add_def(d) is True and s.add_def(d) is False
    assert len(s.defs()) == 1 and len(s.event_log()) == 1
    o = Outcome("T", "c", "pass", wall_ms=3)
    assert s.add_outcome(o) and not s.add_outcome(Outcome("T", "c", "pass", wall_ms=9))
    assert len(s.event_log()) == 2


def _facts(s, toy, comps):
    return (frozenset(d.hash for d in s.defs()), s.outcomes(), verified(s, toy), done(s, toy),
            {c.id: complete(s, toy, c) for c in comps})


def test_confluence_shuffled_replay(toy):
    s, comps = _toy_events(toy)
    base = _facts(s, toy, comps)
    assert base[3] is True and base[4] == {c.id: True for c in comps}
    rng = random.Random(0)
    for _ in range(20):
        log = s.event_log()
        rng.shuffle(log)
        assert _facts(Store.from_events(log), toy, comps) == base


def test_verified_monotone_under_append(toy):
    s, comps = _toy_events(toy)
    rng = random.Random(1)
    for _ in range(20):
        log = s.event_log()
        rng.shuffle(log)
        r = Store()
        prev_v, prev_c = frozenset(), {c.id: False for c in comps}
        for e in log:
            r = Store.from_events(r.event_log() + [e])
            v = verified(r, toy)
            cc = {c.id: complete(r, toy, c) for c in comps}
            assert prev_v <= v
            assert all(cc[k] or not prev_c[k] for k in cc)
            prev_v, prev_c = v, cc


def test_complete_requires_referenced_slots_and_helpers(toy):
    s, (good, _) = _toy_events(toy)
    partial = Composition.make({k: v for k, v in good.bindings if k != "norm"})  # get references norm
    assert not complete(s, toy, partial)
    assert not complete(Store(), toy, good)
