"""Targeted repair vs whole-file rewrite, the agent loop this repo actually improves.

An existing class has working methods and one bug. `repair` keeps the working fills and only
samples the dead slot. `rewrite` regenerates the whole class from the same feedback — the usual
agent move, which can regress a passing method while fixing the reported one.

The fixture is the KVStore demo with a `get` that forgets to normalize. The rewrite model "fixes"
`get` and drops normalization from `set`. No live model: the comparison is the policy.

ClassEval evidence is the 7B v2-vs-WCR pilot in results/pilot.md, cited at the bottom.

python -m calm_coder.bench.repair_bench [--out results/repair.md]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from calm_coder.runner.sandbox import run_tests
from calm_coder.serve.client import Client
from calm_coder.task import task_from_implementation
from calm_coder.v2.harness import run_from_seed

HERE = Path(__file__).resolve().parents[1]
DEMO = HERE / "demo"
BROKEN = DEMO / "broken.py"
TESTS = DEMO / "test_task.py"

FIXED_GET = (
    "def get(self, key, default=None):\n"
    "    return self.data.get(key.strip().lower(), default)\n"
)

# Typical rewrite: the reported bug is fixed, a passing method is not.
REWRITE_CLASS = '''
class KVStore:
    """A tiny in-memory key-value store."""
    def __init__(self):
        self.data = {}
    def set(self, key, value):
        self.data[key] = value
    def get(self, key, default=None):
        return self.data.get(key.strip().lower(), default)
    def delete(self, key):
        k = key.strip().lower()
        if k in self.data:
            del self.data[k]
            return True
        return False
    def keys_with_prefix(self, prefix):
        p = prefix.strip().lower()
        return sorted(k for k in self.data if k.startswith(p))
'''


def fake_agent(calls: list[dict]):
    """Repair returns a correct `get`. Rewrite returns a class that regresses `set`."""

    def h(req: httpx.Request):
        body = json.loads(req.content)
        calls.append(body)
        user = body["messages"][-1]["content"]
        if body.get("max_tokens") == 1:
            text = "ok"
        elif "Rewrite only `" in user:
            text = f"```python\n{FIXED_GET}\n```"
        else:
            text = f"```python\n{REWRITE_CLASS}\n```"
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 40,
                      "prompt_tokens_details": {"cached_tokens": 50}}})
    return h


def _client(calls: list[dict]) -> Client:
    return Client("http://fake/v1", "fake", no_n=True, transport=httpx.MockTransport(fake_agent(calls)))


async def _run(mode: str) -> dict:
    calls: list[dict] = []
    task, seed_src = task_from_implementation(BROKEN, TESTS)
    res = await run_from_seed(_client(calls), task, seed_src, mode=mode, budget_tokens=2000,
                              repair_n=2, rounds=2, warm=False, seed=0)
    row = dict(res.row)
    row["n_model_calls"] = len(calls)
    row["n_slots"] = len(task.slots)
    return row


def to_md(repair: dict, rewrite: dict) -> str:
    def line(name, r):
        kept = r.get("seed_slots_kept") or []
        return (f"| {name} | {int(bool(r['solved']))} | {r['decode_tokens']} | {r['n_model_calls']} "
                f"| {len(kept)}/{r['n_slots']} | {', '.join(kept) or '-'} |")
    return "\n".join([
        "# Targeted repair vs whole-file rewrite",
        "",
        "An existing KVStore has working `set` / `delete` / `keys_with_prefix` and a `get` that",
        "forgets to normalize keys. Two policies see the same failing tests:",
        "",
        "- **repair** — keep passing fills, sample only dead slots, recombine.",
        "- **rewrite** — regenerate the whole class (the usual agent loop). The model “fixes”",
        "  `get` and drops normalization from `set`.",
        "",
        "The store never retracts a passing method. The rewrite arm does not recombine with the",
        "seed, so the regression stands. Same fake model, same budget, same feedback.",
        "",
        "| policy | solved | decode tokens | model calls | seed methods kept | kept |",
        "| --- | --- | --- | --- | --- | --- |",
        line("repair", repair),
        line("rewrite", rewrite),
        "",
        "## What this is evidence for",
        "",
        "Not “CALM beats whole-class sampling on ClassEval.” That bet (H2) failed. This is the",
        "narrower claim that *did* show up: when a class is already mostly right, spending new",
        "tokens only on dead slots does not throw away working methods, and whole-file rewrite can.",
        "",
        "## ClassEval (7B pilot, not this fixture)",
        "",
        "On 10 ClassEval tasks at equal decode budget, v2 (targeted repair + recombination) solved",
        "8/10 and whole-class repair solved 6/10. Both arms solved the same 6 by sampling; the gap",
        "was repair closing 9 of 11 dead slots while WCR closed none (`results/pilot.md`).",
        "Underpowered (McNemar p = 0.50). Directionally the same policy as this fixture.",
        "",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/repair.md")
    a = ap.parse_args()
    repair, rewrite = asyncio.run(_run("repair")), asyncio.run(_run("rewrite"))
    md = to_md(repair, rewrite)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md)
    (Path(a.out).with_suffix(".json")).write_text(json.dumps({"repair": repair, "rewrite": rewrite},
                                                             indent=1, default=str))
    print(md)
    task, src = task_from_implementation(BROKEN, TESTS)
    failing = run_tests(src, task.test_src, list(task.test_classes))
    assert any(v["result"] != "pass" for v in failing.values()), "fixture is not broken"


if __name__ == "__main__":
    main()
