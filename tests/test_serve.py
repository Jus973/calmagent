import asyncio
import json
import os

import httpx
import pytest

from calm_coder.serve.client import Client, Fleet
from calm_coder.serve.prompts import repair_messages, slot_messages, whole_class_messages


def _handler(calls, usage=True, fail_first=0):
    state = {"n": 0}

    def h(req: httpx.Request):
        state["n"] += 1
        if state["n"] <= fail_first:
            return httpx.Response(503)
        body = json.loads(req.content)
        calls.append(body)
        n = body.get("n", 1)
        data = {"choices": [{"index": i, "message": {"content": f"x{i}" * (i + 1)}, "finish_reason": "stop"}
                            for i in range(n)]}
        if usage:
            data["usage"] = {"prompt_tokens": 100, "completion_tokens": 30 * n,
                             "prompt_tokens_details": {"cached_tokens": 64}}
        return httpx.Response(200, json=data)
    return h


def run(coro):
    return asyncio.run(coro)


def test_server_side_n():
    calls = []
    c = Client("http://x/v1", "m", no_n=False, transport=httpx.MockTransport(_handler(calls)))
    out = run(c.sample([{"role": "user", "content": "hi"}], n=4, seed=7))
    assert len(calls) == 1 and calls[0]["n"] == 4 and calls[0]["seed"] == 7
    assert len(out) == 4 and sum(s.completion_tokens for s in out) == 120
    assert [s.prompt_tokens for s in out] == [100, 0, 0, 0] and out[0].cached_tokens == 64


def test_no_n_fans_out_with_distinct_seeds():
    calls = []
    c = Client("http://x/v1", "m", no_n=True, transport=httpx.MockTransport(_handler(calls)))
    out = run(c.sample([{"role": "user", "content": "hi"}], n=4, seed=10))
    assert sorted(b["seed"] for b in calls) == [10, 11, 12, 13] and all(b["n"] == 1 for b in calls)
    assert [s.seed for s in out] == [10, 11, 12, 13] and all(s.prompt_tokens == 100 for s in out)


def test_missing_usage_is_estimated_cached_is_none():
    c = Client("http://x/v1", "m", no_n=True, transport=httpx.MockTransport(_handler([], usage=False)))
    (s,) = run(c.sample([{"role": "user", "content": "hi"}], n=1))
    assert s.tokens_estimated and s.cached_tokens is None


def test_retries_transient_errors(monkeypatch):
    async def fast(_):
        return None
    monkeypatch.setattr("calm_coder.serve.client.asyncio.sleep", fast)
    c = Client("http://x/v1", "m", transport=httpx.MockTransport(_handler([], fail_first=2)))
    assert len(run(c.sample([{"role": "user", "content": "hi"}]))) == 1


def test_slot_prompts_share_prefix(classeval_task0):
    t = classeval_task0
    a, b = (slot_messages(t, s.id) for s in t.slots[:2])
    assert a[0] == b[0]
    ua, ub = a[1]["content"], b[1]["content"]
    assert ua.rsplit("\n", 1)[0] == ub.rsplit("\n", 1)[0]
    assert ua.endswith(f"Implement `{t.slots[0].id}` now.")


def test_other_prompts_build(classeval_task0):
    assert "Complete the class" in whole_class_messages(classeval_task0)[1]["content"]
    m = repair_messages(classeval_task0, "filter", "\n".join(str(i) for i in range(100)))
    assert "\n59\n" not in m[1]["content"] and "\n60\n" in m[1]["content"]


@pytest.mark.skipif(not os.environ.get("CALM_SMOKE"), reason="set CALM_SMOKE=1 to hit the live endpoint")
def test_smoke_live_endpoint():
    async def go():
        async with Client() as c:
            return await c.sample([{"role": "user", "content": "Say hi."}], n=4, max_tokens=16)
    out = run(go())
    assert len(out) == 4 and all(s.text for s in out)


def test_fleet_splits_samples_across_models_with_distinct_seeds():
    calls = []
    f = Fleet([Client("http://a/v1", "m1", no_n=False, transport=httpx.MockTransport(_handler(calls))),
               Client("http://b/v1", "m2", no_n=False, transport=httpx.MockTransport(_handler(calls))),
               Client("http://c/v1", "m3", no_n=False, transport=httpx.MockTransport(_handler(calls)))])
    out = run(f.sample([{"role": "user", "content": "hi"}], n=8, seed=100))
    # 8 split three ways, remainder to the earlier members, and no two members share a seed range.
    assert f.shares(8) == [3, 3, 2]
    assert sorted((b["n"], b["seed"]) for b in calls) == [(2, 106), (3, 100), (3, 103)]
    assert len(out) == 8
    assert sorted({s.model for s in out}) == ["m1", "m2", "m3"]


def test_fleet_share_is_a_function_of_n_alone():
    f = Fleet([Client("http://a/v1", "m1"), Client("http://b/v1", "m2")])
    assert [f.shares(i) for i in range(5)] == [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]
    assert f.model == "m1+m2" and [c.model for c in f.members] == ["m1", "m2"]


def test_fleet_from_spec_falls_back_to_env_base_url(monkeypatch):
    monkeypatch.setenv("CALM_BASE_URL", "http://env/v1")
    f = Fleet.from_spec("m1@http://a/v1, m2")
    assert [(c.model, c.base_url) for c in f.members] == [("m1", "http://a/v1"), ("m2", "http://env/v1")]


def test_single_client_is_a_one_member_fleet():
    c = Client("http://a/v1", "m1")
    assert c.members == [c]
