"""The demo page is fed by exported run logs, so the exporter is what has to be right."""
import asyncio
import json
import random

from calm_coder.demo.export_web import build, store_hash
from calm_coder.store.store import Store
from calm_coder.v2.harness import run_v2
from tests.test_v2 import client, toy_class  # noqa: F401  (client fixture)


def _run_dir(tmp_path, toy, client):
    res = asyncio.run(run_v2(client, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=3, warm=False))
    run = tmp_path / "20260101T000000Z_demo"
    (run / "events").mkdir(parents=True)
    with open(run / "events" / "toy__v2__s0.jsonl", "w") as f:
        for ev in res.store.event_log():
            f.write(json.dumps(ev) + "\n")
    with open(run / "emissions.jsonl", "w") as f:
        for e in res.emissions:
            f.write(json.dumps({**e, "seed": 0, "round": e["seed"]}) + "\n")
    row = dict(res.row)
    row["completion_tokens"] = row["decode_tokens"]
    (run / "results.jsonl").write_text(json.dumps(row) + "\n")
    return run, res


def test_export_describes_the_run_the_page_replays(tmp_path, toy, client):
    run, res = _run_dir(tmp_path, toy, client)
    d = build(run, "toy", "v2", 0)
    assert d["slots"] and set(d["slots"]) <= {s.id for s in toy.slots}
    assert d["left"]["samples"] and d["left"]["decode_tokens"] > 0       # best-of-N pane
    kinds = {e["kind"] for e in d["right"]["events"]}
    assert kinds == {"def", "outcome"}
    assert any(e["kind"] == "def" and "repair" in e["producer"] for e in d["right"]["events"])
    assert d["right"]["solved"] and d["right"]["decode_tokens"] > 0


def test_the_store_hash_the_shuffle_button_shows_is_order_independent(tmp_path, toy, client):
    run, res = _run_dir(tmp_path, toy, client)
    events = res.store.event_log()
    base = store_hash(Store.from_events(events))
    rng = random.Random(0)
    for _ in range(10):
        rng.shuffle(events)
        assert store_hash(Store.from_events(events)) == base
    assert build(run, "toy", "v2", 0)["store_hash"] == base
