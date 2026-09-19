import ast
import asyncio
import json
import random
import re

import httpx
import pytest

from calm_coder.bench.classeval import task_from_row
from calm_coder.bench.experiment import RunDir, run_a_greedy, run_c, run_calm
from calm_coder.serve.client import Client
from calm_coder.store.derive import done, verified
from calm_coder.store.store import Store


def fake_server(row, wrong_every=3):
    """Slot prompts -> reference method (every `wrong_every`-th seed: a broken one). Class -> reference."""
    sol = ast.parse(row["solution_code"])
    cls = next(n for n in sol.body if isinstance(n, ast.ClassDef))
    methods = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}

    def h(req):
        body = json.loads(req.content)
        user = body["messages"][-1]["content"]
        m = re.search(r"Implement `(\w+)` now\.$", user)
        if m:
            name = m.group(1)
            src = methods[name]
            if body.get("seed", 0) % wrong_every == 1:
                src = re.sub(r"(def \w+\([^)]*\)[^:]*:)", r"\1\n    raise ValueError('wrong')", src, count=1)
            text = f"```python\n{src}\n```"
        else:
            text = f"```python\n{row['solution_code']}\n```"
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": text}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 50, "completion_tokens": len(text) // 4}})
    return h


@pytest.fixture
def client_for(rows):
    def make(i, **kw):
        return Client("http://fake/v1", "fake", no_n=True, transport=httpx.MockTransport(fake_server(rows[i], **kw)))
    return make


def test_calm_arm_solves_and_logs_are_confluent(rows, client_for, tmp_path):
    task = task_from_row(rows[0])
    rd = RunDir(tmp_path)
    out = asyncio.run(run_calm(client_for(0), task, 0, [1, 2, 4], rd, max_comps=16))
    by_n = {r["N"]: r for r in out}
    assert [r["N"] for r in out] == [1, 2, 4]
    assert by_n[4]["solved"] and by_n[2]["solved"]
    assert by_n[4]["distinct_alpha"] < by_n[4]["emitted"]          # reference fill repeats -> dedup
    assert by_n[4]["completion_tokens"] > by_n[1]["completion_tokens"]
    events = [json.loads(l) for l in (tmp_path / "events" / f"{task.task_id}__calm__s0.jsonl").read_text().splitlines()]
    base = Store.from_events(events)
    assert done(base, task)
    rng = random.Random(0)
    for _ in range(20):
        rng.shuffle(events)
        s = Store.from_events(events)
        assert verified(s, task) == verified(base, task) and s.outcomes() == base.outcomes()
    assert len(rd.rows("emissions.jsonl")) == 4 * len(task.slots)


def test_c_and_greedy_arms(rows, client_for, tmp_path):
    task = task_from_row(rows[1])
    rd = RunDir(tmp_path)
    c = asyncio.run(run_c(client_for(1), task, 0, [1, 2], rd))
    assert [r["solved"] for r in c] == [True, True] and c[1]["n_pass"] == 2
    a = asyncio.run(run_a_greedy(client_for(1), task, 0, rd))
    assert a[0]["solved"]
