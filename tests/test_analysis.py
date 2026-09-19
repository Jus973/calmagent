import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from calm_coder.serve.client import Client
from calm_coder.store.store import Store
from calm_coder.v2.decompose import ingest_class_sample
from calm_coder.v2.harness import run_v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import dead_slots, final_report, pool_existing  # noqa: E402

from test_v2 import BAD_GET, GOOD, fake_server, toy_class  # noqa: E402


def row(task, arm, seed, solved, **kw):
    return {"task_id": task, "arm": arm, "seed": seed, "N": 8, "solved": solved,
            "completion_tokens": 1000, "prompt_tokens": 500, "requests": 8, "ttfv_ms": 1000, **kw}


@pytest.fixture
def run_dir(tmp_path):
    rows = []
    for i in range(10):
        t = f"ClassEval_{i}"
        rows.append(row(t, "c", 0, i < 4))
        rows.append(row(t, "v2", 0, i < 7, over_budget=False, feedback="F1", seed_comp_solved=i < 4,
                        dead_slots=[] if i < 7 else ["b"],
                        rounds=[{"round": 0, "producer": "whole_class", "dead_after": [] if i < 4 else ["b"]},
                                {"round": 1, "producer": "repair", "targets": ["b"], "dead_after": []}]))
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_final_report_pairs_arms_and_writes_tables(run_dir, tmp_path):
    m = final_report.compute(final_report.load([run_dir]), ["c"])
    assert m["solve"]["v2"]["rate"] == 0.7 and m["solve"]["c"]["rate"] == 0.4
    p = m["paired"]["v2-c"]
    assert p["arm_only"] == 3 and p["base_only"] == 0 and p["diff"] == pytest.approx(0.3)
    assert p["mcnemar_p"] < 0.3 and p["bootstrap_ci"][0] >= 0
    assert m["v2"]["v2"]["repair_rounds_mean"] == 1.0
    assert m["v2"]["v2"]["solved_by_seed_samples"] == 4
    md = final_report.to_md(m)
    assert "| v2 | 8 | 10 | 1 | 0.70 |" in md and "McNemar" in md
    assert m["solve"]["v2"]["ci_kind"] == "wilson"        # one seed: a task is a Bernoulli trial


def test_a_seed_averaged_task_is_not_reported_as_a_coin_flip(tmp_path):
    """With two seeds a task can be half-solved, which Wilson has no way to read; the interval
    over tasks does, and the tasks the two arms disagree about per seed stay out of McNemar."""
    rows = []
    for i in range(6):
        t = f"ClassEval_{i}"
        rows += [row(t, "c", 0, i < 2), row(t, "c", 1, i < 2),
                 row(t, "v2", 0, i < 4), row(t, "v2", 1, i < 3)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = final_report.compute(final_report.load([tmp_path]), ["c"])
    s = m["solve"]["v2"]
    assert s["rate"] == pytest.approx(3.5 / 6) and s["ci_kind"] == "bootstrap"
    assert 0.0 <= s["ci"][0] <= s["rate"] <= s["ci"][1] <= 1.0
    p = m["paired"]["v2-c"]
    assert p["n_tasks"] == 6 and p["mcnemar_tasks"] == 5 and p["ambiguous_tasks"] == 1
    assert p["arm_only"] == 1 and p["base_only"] == 0
    assert p["diff"] == pytest.approx(3.5 / 6 - 2 / 6)


def test_an_arm_that_ran_fewer_tasks_says_so(tmp_path):
    """Two arms with two denominators is the one way a report can lie without a wrong number."""
    rows = [row(f"ClassEval_{i}", "c", 0, i < 2) for i in range(6)]
    rows += [row(f"ClassEval_{i}", "v2", 0, True) for i in range(3)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "excluded.jsonl").write_text(json.dumps({"task_id": "ClassEval_9", "why": "imports"}) + "\n")
    m = final_report.compute(final_report.load([tmp_path]), ["c"])
    m["excluded"] = final_report.load_excluded([tmp_path])
    assert m["solve"]["v2"]["n"] == 3 and m["solve"]["c"]["n"] == 6
    assert m["solve"]["v2"]["missing_tasks"] == ["ClassEval_3", "ClassEval_4", "ClassEval_5"]
    assert m["paired"]["v2-c"]["n_tasks"] == 3          # only what both arms ran is compared
    md = final_report.to_md(m)
    assert "1 tasks excluded" in md and "ClassEval_3, ClassEval_4, ClassEval_5" in md


def test_pooling_reports_one_denominator_per_configuration(tmp_path):
    """Rows from N=4 and N=8 answer different questions and may not share a table."""
    out = tmp_path / "analysis"
    out.mkdir()
    rs = [{"task_id": "t1", "seed": 0, "N": 4, "c_solved": False, "calm_solved": False,
           "pooled_solved": False, "slots": {}},
          {"task_id": "t1", "seed": 0, "N": 8, "c_solved": False, "calm_solved": False,
           "pooled_solved": True, "slots": {}}]
    (out / "pool_existing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    pool_existing.write_md(out)
    md = (out / "pool_existing.md").read_text()
    assert "## seed 0, N 4" in md and "## seed 0, N 8" in md
    assert md.count("| pooled store, same search | 0 | 1 |") == 1
    assert md.count("| pooled store, same search | 1 | 1 |") == 1


# ------------------------------------------------- Phase 0 recon on a store, no model needed
def test_pooling_solves_what_neither_producer_solves_alone(toy):
    """The recon's headline column: the per-method fills have no working `get`, the whole-class
    sample has one but no `put`, and the pooled store recombines them with no new tokens."""
    store = Store()
    ingest_class_sample(toy, store, toy_class(BAD_GET), {"from": "calm", "sample_idx": 0})
    calm_events = [e for e in store.event_log() if e["kind"] == "def"]
    sample = {"sample_idx": 0, "text": toy_class(GOOD["get"]).replace(GOOD["put"], "")}
    r = asyncio.run(pool_existing.pool_task(toy, calm_events, [sample], cap=50, par=2, max_comps=32))
    assert r["slots"]["get"]["calm"] == 1 and r["slots"]["get"]["pooled"] == 2
    assert r["seed_comps"] == 0                      # the sample on its own is missing `put`
    assert r["pooled_solved"]


def test_dead_slots_names_the_slot_no_candidate_passes(toy):
    calls: list[dict] = []
    client = Client("http://fake/v1", "fake", no_n=False,
                    transport=httpx.MockTransport(fake_server(calls, repair_fixes=False)))
    res = asyncio.run(run_v2(client, toy, budget_tokens=10_000, k=2, repair_n=2, rounds=2, warm=False))
    assert not res.row["solved"]
    info = dead_slots.analyze(toy, res.store.event_log())
    assert info["dead_slots"] == ["get"] and info["report"]["get"]["candidates"] >= 1
    assert info["best_class"] is not None and info["interface_errors"] + info["value_errors"] > 0
