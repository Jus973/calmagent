"""v2 acceptance: prefix alignment, leak guard, no-loss decomposition, replay determinism."""
import asyncio
import json
import random
import textwrap
from dataclasses import replace

import httpx
import pytest

from calm_coder.serve.client import Client
from calm_coder.serve.extract import extract_class
from calm_coder.serve.prompts import (
    shared_prefix, slot_messages, slot_repair_messages, whole_class_messages,
    whole_class_repair_messages,
)
from calm_coder.store.defs import Composition
from calm_coder.store.derive import verified
from calm_coder.store.materialize import materialize
from calm_coder.store.store import Store
from calm_coder.v2 import state
from calm_coder.v2.decompose import ingest_class_sample
from calm_coder.v2.feedback import LEAK_WINDOW, REDACTED, Failure, parse_failure, render
from calm_coder.v2.harness import run_v2, run_wcr

GOOD = {
    "put": "    def put(self, key, value):\n        self.items[self.norm(key)] = value\n",
    "get": "    def get(self, key):\n        return self.items[self.norm(key)]\n",
    "norm": "    def norm(key):\n        return key.strip().lower()\n",
}
BAD_GET = "    def get(self, key):\n        return self.items[key]\n"


def toy_class(get_src: str) -> str:
    return ("import re\n\nclass Toy:\n    \"\"\"A toy.\"\"\"\n\n"
            "    def __init__(self):\n        self.items = {}\n\n"
            + GOOD["put"] + "\n" + get_src + "\n" + GOOD["norm"])


def fake_server(calls: list[dict], *, repair_fixes: bool = True):
    """Whole class -> a class whose `get` ignores normalization; repair -> the correct method."""

    def h(req):
        body = json.loads(req.content)
        user = body["messages"][-1]["content"]
        calls.append(body)
        if user.rstrip().endswith("full class."):
            src = toy_class(GOOD["get"] if repair_fixes and "current implementation" in user else BAD_GET)
            text = f"```python\n{src}\n```"
        elif "Rewrite only" in user:
            slot = user.rsplit("Rewrite only `", 1)[1].split("`", 1)[0]
            src = GOOD[slot] if repair_fixes else BAD_GET
            text = f"```python\n{textwrap.dedent(src)}\n```"
        elif "Implement `" not in user:
            text = "ok"                                   # warm-up request
        else:
            slot = user.rsplit("Implement `", 1)[1].split("`", 1)[0]
            text = f"```python\n{textwrap.dedent(GOOD[slot] if slot != 'get' else BAD_GET)}\n```"
        n = body.get("n", 1)
        return httpx.Response(200, json={
            "choices": [{"index": i, "message": {"content": text}, "finish_reason": "stop"} for i in range(n)],
            "usage": {"prompt_tokens": 50, "completion_tokens": 8 * n,
                      "prompt_tokens_details": {"cached_tokens": 40}}})
    return h


@pytest.fixture
def client(request):
    calls: list[dict] = []
    c = Client("http://fake/v1", "fake", no_n=False, transport=httpx.MockTransport(fake_server(calls)))
    c.calls = calls
    return c


# ------------------------------------------------------------------ prefix alignment
def test_every_prompt_starts_with_the_shared_prefix(toy):
    prefix = shared_prefix(toy)
    users = [slot_messages(toy, "get")[-1]["content"],
             whole_class_messages(toy)[-1]["content"],
             slot_repair_messages(toy, "get", "class Toy: pass", "tests failed")[-1]["content"],
             whole_class_repair_messages(toy, "class Toy: pass", "tests failed")[-1]["content"]]
    for u in users:
        assert u.startswith(prefix), u[:120]
    # and within a repair round only the trailing block differs
    a = slot_repair_messages(toy, "get", "class Toy: pass", "f")[-1]["content"]
    b = slot_repair_messages(toy, "put", "class Toy: pass", "f")[-1]["content"]
    common = len(prefix)
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    assert common > len(prefix) + 20


# ------------------------------------------------------------------ leak guard
def _longest_common_substring(a: str, b: str) -> int:
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


@pytest.fixture
def leaky_traceback(toy):
    line = next(ln for ln in toy.test_src.splitlines() if "assertEqual" in ln).strip()
    return line, ('Traceback (most recent call last):\n'
                  '  File "/tmp/t.py", line 7, in test_get\n'
                  f'    {line}\n'
                  f"KeyError: 'a' while running {line}\n")


@pytest.fixture
def failures(toy, leaky_traceback):
    return [parse_failure("ToyTestGet", leaky_traceback[1], toy.test_src)]


def test_f0_and_f1_never_carry_test_source(toy, failures):
    for level in ("F0", "F1"):
        out = render(failures, level, slot_id="get")
        assert _longest_common_substring(out, toy.test_src) <= LEAK_WINDOW, (level, out)
    assert "KeyError" not in render(failures, "F0")
    assert "KeyError" in render(failures, "F1")


def test_f2_adds_only_the_raising_line(toy, failures, leaky_traceback):
    line, _ = leaky_traceback
    f1, f2 = render(failures, "F1", slot_id="get"), render(failures, "F2", slot_id="get")
    extra = [ln.strip() for ln in f2.splitlines() if ln not in f1.splitlines()]
    assert extra == [f"raised at: {line}"]
    assert f2.count(line) == 1


@pytest.mark.parametrize("expected", [
    "NAME    | AGE\n--------+----\nalice   | 30\nbob     | 41\n",      # real newlines -> \n in the repr
    {"zulu": 1, "alpha": "beta", "gamma": "delta", "omega": "final-value"},
])
def test_the_leak_guard_survives_repr_reformatting(expected):
    """An expected value written across several source lines is one escaped line in the message."""
    src = ("class T(unittest.TestCase):\n    def test_x(self):\n"
           f"        self.assertEqual(out, {expected!r})\n")
    f = parse_failure("ToyTestGet", f"AssertionError: 'wrong' != {expected!r}\n", src)
    out = render([f], "F1", slot_id="get")
    assert _longest_common_substring(out, src) <= LEAK_WINDOW, out
    assert REDACTED in out


def test_f1_truncates_to_three_failures_of_300_chars(toy):
    fs = [Failure(f"T{i}", "ValueError", "x" * 1000, "y") for i in range(10)]
    out = render(fs, "F1")
    assert out.count("ValueError") == 3 and all(len(ln) < 400 for ln in out.splitlines())


# ------------------------------------------------------------------ no-loss
def test_whole_class_sample_survives_decomposition(toy):
    store = Store()
    ing = ingest_class_sample(toy, store, f"```python\n{toy_class(GOOD['get'])}\n```", {"seed": 1})
    assert ing.complete and not ing.error
    comp = Composition.make(ing.bindings)
    src = materialize(toy, store, comp)
    ns: dict = {}
    exec(compile(src, "<m>", "exec"), ns)
    t = ns["Toy"]()
    t.put(" A ", 1)
    assert t.get("a") == 1 and ns["Toy"].norm(" Ab ") == "ab"


SAMPLE_WITH_PRELUDE = '''
import string

PAD = "-"

def _clean(key):
    return key.strip(string.whitespace + PAD)

class Toy:
    """A toy."""

    def __init__(self):
        self.items = {}

    def put(self, key, value):
        self.items[self.norm(key)] = value

    def get(self, key):
        return self.items[self.norm(key)]

    def norm(key):
        return _clean(key).lower()
'''


def _run_class(src: str):
    ns: dict = {}
    exec(compile(src, "<m>", "exec"), ns)
    t = ns["Toy"]()
    t.put(" -A- ", 1)
    return t.get("a"), ns["Toy"].norm(" Ab ")


def test_a_samples_own_composition_behaves_like_the_sample(toy):
    """No-loss is a claim about the program, so the sample's imports and module code come along."""
    store = Store()
    text = f"```python\n{SAMPLE_WITH_PRELUDE}\n```"
    ing = ingest_class_sample(toy, store, text, {"seed": 1})
    assert ing.complete and not ing.error and not ing.dropped
    src = materialize(toy, store, Composition.make(ing.bindings))
    assert _run_class(src) == _run_class(extract_class(text, toy)) == (1, "ab")


def test_context_a_composition_cannot_carry_is_recorded(toy):
    """The skeleton owns the constructor, so a sample that rewrites it is not its own composition."""
    src = SAMPLE_WITH_PRELUDE.replace("self.items = {}", "self.items = {}\n        self.n = 0")
    ing = ingest_class_sample(toy, Store(), f"```python\n{src}\n```", {"seed": 1})
    assert ing.dropped == ["__init__"]


def test_v2_run_keeps_every_sample_composition(toy, client):
    res = asyncio.run(run_v2(client, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=1, warm=False))
    assert res.row["seed_comps"] == 2
    assert all(res.store.get(h) is not None for h in state.slot_candidates(res.store, toy)["get"])


# ------------------------------------------------------------------ arms
def test_v2_repairs_the_dead_slot(toy, client):
    res = asyncio.run(run_v2(client, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=3, warm=True))
    row = res.row
    assert row["rounds"][0]["dead_after"] == ["get"]            # whole-class samples all miss `get`
    assert row["solved"] and row["verified_comp"]
    assert any(r["producer"] == "repair" and r["targets"] == ["get"] for r in row["rounds"])
    assert row["dead_slots"] == () or list(row["dead_slots"]) == []
    assert row["decode_tokens"] > 0 and not row["over_budget"]
    assert verified(res.store, toy)


def test_v2_respects_the_budget(toy, client):
    res = asyncio.run(run_v2(client, toy, budget_tokens=1, k=2, repair_n=2, rounds=3, warm=False))
    assert [r["producer"] for r in res.row["rounds"]] == ["whole_class"]   # no repair round is funded
    assert res.row["decode_tokens"] == 0 and not res.row["over_budget"]    # nor the first round


def test_the_budget_caps_a_round_before_it_is_sent(toy, client):
    """max_tokens is the only lever that stops a request, so it carries the remaining budget."""
    res = asyncio.run(run_v2(client, toy, budget_tokens=200, k=2, repair_n=2, rounds=3, warm=False))
    assert [c["max_tokens"] for c in client.calls][0] == 100               # 200 // 2 samples
    assert res.row["decode_tokens"] <= 200 and not res.row["over_budget"]


def test_wcr_baseline_gets_the_same_feedback(toy, client):
    res = asyncio.run(run_wcr(client, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=2, warm=False))
    assert res.row["solved"]                                   # its repair sample fixes the class
    assert [r["producer"] for r in res.row["rounds"]] == ["whole_class", "whole_class_repair"]
    repair_prompts = [c for c in client.calls if "current implementation" in c["messages"][-1]["content"]]
    assert repair_prompts and "ToyTestGet" in repair_prompts[0]["messages"][-1]["content"]


# ------------------------------------------------------------------ determinism
def test_replay_of_a_v2_run_is_order_independent(toy, client):
    res = asyncio.run(run_v2(client, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=3, warm=False))
    events = res.store.event_log()
    base = Store.from_events(events)
    rng = random.Random(0)
    for _ in range(20):
        rng.shuffle(events)
        s = Store.from_events(events)
        assert verified(s, toy) == verified(base, toy)
        assert s.defs() == base.defs() and s.outcomes() == base.outcomes()
        assert state.dead_slots(s, toy) == state.dead_slots(base, toy)
        assert state.best_class(s, toy) == state.best_class(base, toy)


def test_the_run_seed_changes_what_is_sampled(toy):
    """Two seeds of an arm must be two samples of it, not the same run logged twice."""
    def seeds(run_seed: int):
        c = Client("http://fake/v1", "fake", no_n=False, transport=httpx.MockTransport(fake_server([])))
        r = asyncio.run(run_v2(c, toy, budget_tokens=10_000, seed=run_seed, k=2, repair_n=2,
                               rounds=3, warm=False))
        return [e["sample_seed"] for e in r.emissions]
    assert seeds(0) and not set(seeds(0)) & set(seeds(1))


def test_the_warmup_prompt_is_a_prefix_of_the_requests_it_warms(toy, client):
    """A served prompt includes its system message, so a warm-up without one warms nothing."""
    asyncio.run(run_v2(client, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=3, warm=True))
    def ser(c):
        return "\n".join(f"{m['role']}:{m['content']}" for m in c["messages"])
    warms = [c for c in client.calls if c["max_tokens"] == 1]
    assert len(warms) >= 2
    for w in warms:
        after = client.calls[client.calls.index(w) + 1]
        assert ser(after).startswith(ser(w)), ser(w)[:200]


def test_repair_rounds_use_the_repair_client(toy):
    """A hosted fixer is a second producer into the same store, not a judge and not the headline."""
    gen_calls, fix_calls = [], []
    gen = Client("http://gen/v1", "qwen", no_n=False,
                 transport=httpx.MockTransport(fake_server(gen_calls)))
    fixer = Client("https://api.x.ai/v1", "grok-build-0.1", no_n=False,
                   transport=httpx.MockTransport(fake_server(fix_calls)))
    res = asyncio.run(run_v2(gen, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=3,
                             warm=False, repair_client=fixer))
    assert res.row["solved"]
    assert res.row["gen_model"] == "qwen" and res.row["repair_model"] == "grok-build-0.1"
    assert all("Rewrite only" not in c["messages"][-1]["content"] for c in gen_calls)
    assert any("Rewrite only" in c["messages"][-1]["content"] for c in fix_calls)
    assert any(e.get("model") == "grok-build-0.1" for e in res.emissions)
    assert any(e.get("model") == "qwen" for e in res.emissions)


def test_a_repair_round_asks_for_as_many_samples_as_the_baseline(toy, client):
    """`repair_n` is a round total in both arms, so the targeted round is split across dead slots."""
    v2 = asyncio.run(run_v2(client, toy, budget_tokens=10_000, k=2, repair_n=4, rounds=1, warm=False))
    rnd = next(r for r in v2.row["rounds"] if r["producer"] == "repair")
    assert rnd["samples_per_target"] * len(rnd["targets"]) == 4


def test_two_v2_runs_with_the_same_seeds_agree(toy):
    def once():
        c = Client("http://fake/v1", "fake", no_n=False, transport=httpx.MockTransport(fake_server([])))
        r = asyncio.run(run_v2(c, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=3, warm=False))
        return r.row["verified_comp"], sorted(d.hash for d in r.store.defs())
    assert once() == once()


# ------------------------------------------------------------------ policy reads
def _outcome(comp: Composition, test_id: str, result: str, detail: str = ""):
    from calm_coder.store.defs import Outcome
    return Outcome(test_id=test_id, comp_id=comp.id, result=result, detail=detail,
                   bindings=tuple(sorted(comp.binding.items())))


def test_best_class_breaks_ties_by_comp_id(toy):
    """A tie has to resolve the same way in every replica, so it resolves on content."""
    store = Store()
    comps = [Composition.make({"put": "a" * 64, "get": g, "norm": "c" * 64})
             for g in ("b" * 64, "d" * 64)]
    for c in comps:
        store.add_outcome(_outcome(c, "ToyTestPut", "pass"))
        store.add_outcome(_outcome(c, "ToyTestGet", "fail", "boom"))
    assert state.best_class(store, toy).id == min(c.id for c in comps)


def test_add_def_keeps_the_first_body_for_a_hash(toy):
    """add_def is an idempotent set insert: a second write of the same hash changes nothing."""
    from calm_coder.store.defs import Definition
    d = Definition(hash="h" * 64, kind="fill", slot="get", name="get", canonical_src="def get(self): pass",
                   raw_src="def get(self): pass", slot_refs=frozenset(), helper_refs=frozenset(),
                   unresolved_refs=frozenset(), is_static=False)
    store = Store()
    assert store.add_def(d)
    assert not store.add_def(replace(d, raw_src="def get(self): return 1", canonical_src="x"))
    assert [x.canonical_src for x in store.defs()] == ["def get(self): pass"]
    assert len(store.event_log()) == 1


def test_a_class_failure_that_blames_no_slot_is_reported_as_a_guess():
    """Repair has to aim somewhere, but "every slot is dead" is not a measurement."""
    from calm_coder.task import Task
    from tests.conftest import TOY_SKELETON
    src = ("import unittest\n\nclass ToyTestAll(unittest.TestCase):\n"
           "    def test_all(self):\n        self.assertTrue(Toy())\n")
    toy = Task.from_skeleton(task_id="toy", skeleton=TOY_SKELETON, test_src=src, slot_tests={})
    store = Store()
    comp = Composition.make({s.id: f"{i}" * 64 for i, s in enumerate(toy.slots)})
    store.add_outcome(_outcome(comp, toy.test_classes[0], "fail", "nothing here names a method"))
    assert state.unattributed(store, toy, comp)
    assert all(v["unattributed"] for v in state.dead_slot_report(store, toy, comp).values())
