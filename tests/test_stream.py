"""Streamed decoding, the early cut, and the latency benchmark's fake server.

The claim under test is that cutting a completion short costs nothing: the definition that lands in the
store is the one the full completion would have produced, and only the discarded tail differs.
"""
import asyncio
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


def cut(text: str, slot: str = "get") -> str | None:
    i = truncation_point(text, slot, SLOTS)
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
