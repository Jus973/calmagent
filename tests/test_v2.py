"""v2 acceptance: prefix alignment, leak guard, no-loss decomposition, replay determinism."""
import asyncio
import json
import random
import textwrap

import httpx
import pytest

from calm_coder.serve.client import Client
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
from calm_coder.v2.feedback import Failure, parse_failure, render
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
        assert _longest_common_substring(out, toy.test_src) < 30, (level, out)
    assert "KeyError" not in render(failures, "F0")
    assert "KeyError" in render(failures, "F1")


def test_f2_adds_only_the_raising_line(toy, failures, leaky_traceback):
    line, _ = leaky_traceback
    f1, f2 = render(failures, "F1", slot_id="get"), render(failures, "F2", slot_id="get")
    extra = [ln.strip() for ln in f2.splitlines() if ln not in f1.splitlines()]
    assert extra == [f"raised at: {line}"]
    assert f2.count(line) == 1


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


def test_two_v2_runs_with_the_same_seeds_agree(toy):
    def once():
        c = Client("http://fake/v1", "fake", no_n=False, transport=httpx.MockTransport(fake_server([])))
        r = asyncio.run(run_v2(c, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=3, warm=False))
        return r.row["verified_comp"], sorted(d.hash for d in r.store.defs())
    assert once() == once()
