"""Streamed decoding, the early cut, and the latency benchmark's fake server.

The claim under test is that cutting a completion short costs nothing: the definition that lands in the
store is the one the full completion would have produced, and only the discarded tail differs.
"""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from calm_coder.agents.fill import generate_fills
from calm_coder.bench.fake_model import FakeModel, FakeModelConfig
from calm_coder.bench.latency import MODES, run_mode
from calm_coder.bench.latency_samples import SAMPLES, TAILS
from calm_coder.serve.client import Client
from calm_coder.serve.stream import truncation_point
from calm_coder.store.store import Store
from calm_coder.task import task_from_files

FAST = FakeModelConfig(ttft_ms=0.0, tok_s=100_000.0)
SLOTS = ("set", "get", "delete", "keys_with_prefix")

METHOD = "def get(self, key):\n    return self.data[key]\n"


def cut(text: str, slot: str = "get", class_name: str = "KVStore") -> str | None:
    i = truncation_point(text, slot, SLOTS, class_name)
    return None if i is None else text[:i]


# ------------------------------------------------------------------ truncation_point

def test_no_cut_while_the_method_may_still_grow():
    assert cut("```python\n") is None
    assert cut("```python\n" + METHOD) is None          # the next line could be more body
    assert cut("```python\ndef get(self, key)") is None


def test_cut_at_the_fence_prose_or_another_declared_method():
    for tail in ("```", "That returns the value.", "def set(self, key, value):"):
        kept = cut("```python\n" + METHOD + tail)
        assert kept == "```python\n" + METHOD, tail


def test_cut_keeps_the_helpers_the_method_calls():
    text = ("```python\ndef get(self, key):\n    return self.data[self._norm(key)]\n"
            "\ndef _norm(self, key):\n    return key.strip().lower()\n```")
    assert cut(text) == text[:text.rindex("```")]
    partial = text[:text.index("\ndef _norm")] + "\n"   # helper not emitted yet: keep decoding
    assert cut(partial) is None


def test_cut_inside_a_class_body_and_after_reasoning():
    indented = "class KVStore:\n    " + METHOD.replace("\n    ", "\n        ") + "    def set(self):\n"
    assert cut(indented) is not None
    assert cut("<think>the key must be normalized") is None
    assert cut("<think>ok</think>\n" + METHOD + "```") is not None


def test_no_cut_until_the_whole_helper_graph_is_emitted():
    """A helper the fill reaches only through another helper holds the cut just like a direct call."""
    head = ("```python\ndef get(self, key):\n    return self.data[self._norm(key)]\n"
            "\ndef _norm(self, key):\n    return self._strip(key).lower()\n")
    stray = "\ndef set(self, key, value):\n    self.data[key] = value\n"
    tail = "\ndef _strip(self, key):\n    return key.strip()\n"
    assert cut(head + "```") is None                      # _strip is still missing
    assert cut(head + stray) is None                      # ... even past another declared method
    assert cut(head + stray + tail + "```") == head + stray + tail


def test_helper_calls_count_bare_and_through_the_class_name():
    for call, helper in (("KVStore._norm(key)", "def _norm(key):"), ("_norm(key)", "def _norm(key):")):
        head = f"```python\ndef get(self, key):\n    return self.data[{call}]\n"
        assert cut(head + "```") is None
        whole = head + f"\n{helper}\n    return key.strip()\n"
        assert cut(whole + "```") == whole


def test_a_helper_without_a_leading_underscore_still_holds_the_cut():
    """`emission_to_defs` calls every emitted non-slot function a helper, underscore or not."""
    head = "```python\ndef get(self, key):\n    return self.data[self.norm(key)]\n"
    assert cut(head + "```") is None
    whole = head + "\ndef norm(self, key):\n    return key.strip()\n"
    assert cut(whole + "```") == whole


def test_fields_and_declared_slots_do_not_hold_the_cut():
    text = "```python\ndef get(self, key):\n    return self.data.get(self.set(key))\n"
    assert cut(text + "```") == text


def test_no_cut_for_a_different_slot():
    assert cut("```python\n" + METHOD + "```", slot="set") is None


# ------------------------------------------------------------------ streamed client

async def _one(stop: bool):
    model = FakeModel({"get": [METHOD + TAILS["long"]]}, cfg=FAST)
    async with Client(base_url="http://fake.invalid/v1", model="fake", stream=True,
                      transport=model.transport) as client:
        rule = (lambda t: truncation_point(t, "get", SLOTS)) if stop else None
        (s,) = await client.sample([{"role": "user", "content": "Implement `get` now."}],
                                   n=1, seed=0, stop_when=rule)
    return s, model.stats


def test_streaming_measures_ttft_and_returns_the_whole_completion():
    s, stats = asyncio.run(_one(stop=False))
    assert s.ttft_ms is not None and not s.stopped_early
    assert stats.decoded_tokens == stats.offered_tokens
    assert s.text.endswith("doctests.")


def test_the_cut_ends_the_request_so_the_tail_is_never_decoded():
    s, stats = asyncio.run(_one(stop=True))
    assert s.stopped_early and s.finish_reason == "stopped_early"
    assert s.text.rstrip().endswith("self.data[key]")
    assert 0 < stats.decoded_tokens < stats.offered_tokens


def test_a_server_that_ignores_stream_is_still_read():
    """Not every OpenAI-compatible server streams. The sample must survive that, minus the TTFT."""
    def h(req):
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": METHOD}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 9}})

    async def go():
        async with Client(base_url="http://fake.invalid/v1", model="fake", stream=True,
                          transport=httpx.MockTransport(h)) as client:
            return await client.sample([{"role": "user", "content": "Implement `get` now."}], n=1, seed=0,
                                       stop_when=lambda t: truncation_point(t, "get", SLOTS))

    (s,) = asyncio.run(go())
    assert s.text == METHOD and s.completion_tokens == 9
    assert s.ttft_ms is None and not s.stopped_early


def _stream_body(text: str) -> bytes:
    frames = [f'data: {{"choices":[{{"index":0,"delta":{{"content":{json.dumps(c)}}}}}]}}'
              for c in text.splitlines(keepends=True)]
    return ("\n\n".join(frames) + "\n\ndata: [DONE]\n\n").encode()


def _retrying_client(fail):
    """A client whose first request fails the way `fail(request)` says and whose second streams METHOD."""
    calls = []

    def h(req):
        calls.append(req)
        if len(calls) == 1:
            return fail(req)
        return httpx.Response(200, content=_stream_body(METHOD + TAILS["long"]),
                              headers={"content-type": "text/event-stream"})

    async def go():
        async with Client(base_url="http://fake.invalid/v1", model="fake", stream=True,
                          transport=httpx.MockTransport(h)) as client:
            return await client.sample([{"role": "user", "content": "Implement `get` now."}], n=1, seed=0,
                                       stop_when=lambda t: truncation_point(t, "get", SLOTS, "KVStore"))
    return go, calls


def _raise_connect(req):
    raise httpx.ConnectError("server is restarting", request=req)


@pytest.mark.parametrize("fail", [lambda req: httpx.Response(503, text="overloaded"), _raise_connect],
                         ids=["retryable-status", "transport-error"])
def test_a_transient_streaming_failure_is_retried_not_fatal(fail):
    """The latency-first path always streams, so a flaky server must not end the solve."""
    go, calls = _retrying_client(fail)
    (s,) = asyncio.run(go())
    assert len(calls) == 2
    assert s.stopped_early and s.text.rstrip().endswith("self.data[key]")
    assert s.text.count("def get") == 1          # nothing survives from the attempt that failed


def test_a_streaming_failure_that_never_clears_raises():
    def h(req):
        return httpx.Response(503, text="overloaded")

    async def go():
        async with Client(base_url="http://fake.invalid/v1", model="fake", stream=True,
                          transport=httpx.MockTransport(h)) as client:
            return await client.sample([{"role": "user", "content": "hi"}], n=1, seed=0)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(go())


def test_a_cut_sample_estimates_tokens_from_its_text():
    """An early-stopped stream has no usage frame; SSE chunk counts are not tokens."""
    s, _ = asyncio.run(_one(stop=True))
    assert s.tokens_estimated and s.completion_tokens == len(s.text) // 4


def test_unstreamed_requests_have_no_ttft():
    model = FakeModel({"get": [METHOD]}, cfg=FAST)

    async def go():
        async with Client(base_url="http://fake.invalid/v1", model="fake",
                          transport=model.transport) as client:
            return await client.sample([{"role": "user", "content": "Implement `get` now."}], n=1, seed=0)

    (s,) = asyncio.run(go())
    assert s.ttft_ms is None and not s.stopped_early and s.completion_tokens > 0


# ------------------------------------------------------------------ same facts, fewer tokens

@pytest.fixture(scope="module")
def demo_task():
    import calm_coder.demo as demo
    d = Path(demo.__file__).parent
    return task_from_files(d / "task.py", d / "test_task.py")


def _fills(store) -> dict[str, set[str]]:
    return {s: {d.hash for d in store.defs_for_slot(s)} for s in SLOTS}


def test_cutting_a_completion_does_not_change_the_definitions_it_produces(demo_task):
    """Same fixtures, same seeds, with and without the cut: identical fill hashes in the store."""
    def run(stop_at_fill: bool):
        model = FakeModel(SAMPLES, tail=TAILS["long"], cfg=FAST)
        store = Store()

        async def go():
            async with Client(base_url="http://fake.invalid/v1", model="fake",
                              transport=model.transport) as client:
                await generate_fills(client, demo_task, store, n=4, seed=0, stop_at_fill=stop_at_fill)
        asyncio.run(go())
        return _fills(store), model.stats

    whole, whole_stats = run(False)
    cut_, cut_stats = run(True)
    assert cut_ == whole
    assert cut_stats.decoded_tokens < whole_stats.decoded_tokens


# ------------------------------------------------------------------ the benchmark itself

def test_benchmark_modes_verify_and_the_levers_do_what_they_claim(demo_task):
    runs = {m: asyncio.run(run_mode(demo_task, MODES[m], n=4, seed=0, cfg=FAST, tail=TAILS["long"]))
            for m in ("sequential", "adaptive", "stop-at-fill")}
    assert all(r.verified for r in runs.values())
    assert runs["adaptive"].requests <= runs["sequential"].requests    # how many depends on the timing
    assert runs["stop-at-fill"].decoded_tokens < runs["stop-at-fill"].offered_tokens
    assert runs["sequential"].decoded_tokens == runs["sequential"].offered_tokens
