"""Targeted repair keeps passing methods; whole-file rewrite can regress them."""
import asyncio

import httpx

import calm_coder.cli as cli
from calm_coder.bench.repair_bench import BROKEN, FIXED_GET, TESTS, fake_agent, to_md
from calm_coder.runner.sandbox import run_tests
from calm_coder.serve.client import Client
from calm_coder.task import skeletonize, task_from_implementation
from calm_coder.v2.harness import run_from_seed
import httpx


def _client(calls=None):
    calls = [] if calls is None else calls
    return Client("http://fake/v1", "fake", no_n=True, transport=httpx.MockTransport(fake_agent(calls)))


def test_skeletonize_drops_bodies_keeps_docs_and_init():
    src = BROKEN.read_text()
    skel = skeletonize(src)
    assert "self.data.get(key, default)" not in skel
    assert "whitespace is stripped" in skel
    assert "self.data = {}" in skel
    task, _ = task_from_implementation(BROKEN, TESTS)
    assert [s.id for s in task.slots] == ["set", "get", "delete", "keys_with_prefix"]
    assert "pass" in task.skeleton


def test_broken_fixture_fails_get_and_passes_the_other_slot_tests():
    task, src = task_from_implementation(BROKEN, TESTS)
    res = run_tests(src, task.test_src, list(task.test_classes))
    assert res["KVStoreTestGet"]["result"] != "pass"
    assert res["KVStoreTestSet"]["result"] == "pass"
    assert res["KVStoreTestDelete"]["result"] == "pass"
    assert res["KVStoreTestKeysWithPrefix"]["result"] == "pass"


def test_repair_keeps_working_methods_and_solves():
    task, src = task_from_implementation(BROKEN, TESTS)
    res = asyncio.run(run_from_seed(_client(), task, src, mode="repair", budget_tokens=2000,
                                    repair_n=2, rounds=2, warm=False))
    assert res.row["solved"] and not res.row["solved_by_seed"]
    kept = res.row["seed_slots_kept"]
    assert "get" not in kept
    assert set(kept) >= {"set", "delete", "keys_with_prefix"}
    assert any(r["producer"] == "repair" and "get" in r.get("targets", []) for r in res.row["rounds"])


def test_rewrite_regresses_set_and_does_not_recombine():
    task, src = task_from_implementation(BROKEN, TESTS)
    res = asyncio.run(run_from_seed(_client(), task, src, mode="rewrite", budget_tokens=2000,
                                    repair_n=2, rounds=2, warm=False))
    assert not res.row["solved"]
    assert res.row["seed_slots_kept"] == []
    assert any(r["producer"] == "whole_class_repair" for r in res.row["rounds"])


def test_cli_repair_writes_a_class_that_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "Client", lambda: _client())
    out = tmp_path / "fixed.py"
    rc = cli.main(["repair", str(BROKEN), "--tests", str(TESTS), "--N", "2",
                   "--budget-tokens", "2000", "--out", str(out)])
    assert rc == 0
    task, _ = task_from_implementation(BROKEN, TESTS)
    res = run_tests(out.read_text(), task.test_src, list(task.test_classes))
    assert all(v["result"] == "pass" for v in res.values()), res
    assert "strip().lower()" in out.read_text()


def test_repair_report_names_the_kept_methods():
    repair = {"solved": True, "decode_tokens": 40, "n_model_calls": 1, "n_slots": 4,
              "seed_slots_kept": ["delete", "keys_with_prefix", "set"]}
    rewrite = {"solved": False, "decode_tokens": 80, "n_model_calls": 2, "n_slots": 4,
               "seed_slots_kept": []}
    md = to_md(repair, rewrite)
    assert "| repair | 1 | 40 |" in md
    assert "set" in md and "rewrite" in md.lower()
    assert FIXED_GET
