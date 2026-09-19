"""Experiment driver (§3.14). Append-only JSONL under runs/<UTC>_<name>/. Resumable.

python -m calm_coder.bench.experiment --arms calm,c,a_greedy --N 8 --seeds 0 --subset data/subset.json
python -m calm_coder.bench.experiment --resume runs/<dir>          # skip (task, arm, seed) already done
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from calm_coder.agents.fill import Emission, generate_fills
from calm_coder.agents.scheduler import Scheduler
from calm_coder.bench.baselines import whole_class_samples
from calm_coder.bench.classeval import SUBSET, load_subset
from calm_coder.runner import tests as rt
from calm_coder.serve.client import Client
from calm_coder.serve.prompts import SAMPLING
from calm_coder.store.derive import reachable
from calm_coder.store.defs import Composition
from calm_coder.store.store import Store

console = Console(stderr=True)
RESULTS = ("pass", "fail", "error", "timeout", "inconclusive")


class RunDir:
    def __init__(self, path: Path):
        self.path = path
        (path / "events").mkdir(parents=True, exist_ok=True)

    def append(self, name: str, row: dict) -> None:
        with open(self.path / name, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def rows(self, name: str) -> list[dict]:
        p = self.path / name
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    def done(self) -> set[tuple[str, str, int]]:
        return {(r["task_id"], r["arm"], r["seed"]) for r in self.rows("results.jsonl")}


# ---------------------------------------------------------------- CALM arm

def _budget_view(emissions: list[Emission], n: int):
    used = [e for e in emissions if e.sample_idx < n]
    cands: dict[str, list[str]] = defaultdict(list)
    fan_in: Counter = Counter()
    arrival: dict[str, int] = {}
    for e in sorted(used, key=lambda e: e.t_done_ms):
        if e.fill_hash:
            if e.fill_hash not in arrival:
                cands[e.slot].append(e.fill_hash)
                arrival[e.fill_hash] = e.t_done_ms
            fan_in[e.fill_hash] += 1
    return used, dict(cands), fan_in, arrival


async def run_calm(client: Client, task, seed: int, Ns: list[int], rd: RunDir, *, max_comps: int,
                   arm: str = "calm", on_event=None) -> list[dict]:
    store = Store()
    t_start = time.monotonic()
    emissions = await generate_fills(client, task, store, n=max(Ns), seed=seed, arm=arm, t_start=t_start,
                                     on_event=on_event)
    for e in emissions:
        rd.append("emissions.jsonl", e.to_json())
    sched = Scheduler(task, store, max_comps=max_comps, on_event=on_event)
    rows = []
    phase1_ms = 0
    stub: dict[str, str] = {}
    for n in sorted(Ns):
        used, cands, fan_in, arrival = _budget_view(emissions, n)
        new = [h for hs in cands.values() for h in hs if h not in stub]
        t0 = time.monotonic()
        stub = {**stub, **await sched.phase1(new)}
        phase1_ms += int((time.monotonic() - t0) * 1000)
        res = await sched.search(cands, fan_in, arrival)
        gen_ms = max((e.t_done_ms for e in used), default=0)
        per_slot = {}
        for s in task.slots:
            es = [e for e in used if e.slot == s.id]
            c = Counter(stub[e.fill_hash] if e.fill_hash else "extract_error" for e in es)
            distinct = {e.fill_hash for e in es if e.fill_hash}
            per_slot[s.id] = {
                "emitted": len(es), "distinct_alpha": len(distinct),
                "distinct_exact": len({e.fill_hash_exact for e in es if e.fill_hash_exact}),
                "stub": {k: c.get(k, 0) for k in (*RESULTS, "extract_error")},
                "fills_passing_slot_test_in_some_comp": sum(
                    1 for e in es if e.fill_hash and any(
                        tr.bindings.get(s.id) == e.fill_hash and tr.results.get(task.slot_tests.get(s.id)) == "pass"
                        for tr in res.trace)),
            }
        heldout = next((tr for tr in res.trace if tr.slot_level_pass), None)
        all_defs = {h for e in used for h in e.def_hashes}
        reach = reachable(store, Composition.make(dict(next(tr.bindings for tr in res.trace
                                                             if tr.comp_id == res.verified_comp).items())))[0] \
            if res.verified_comp else set()
        row = {
            "task_id": task.task_id, "arm": arm, "seed": seed, "N": n,
            "solved": res.verified_comp is not None, "verified_comp": res.verified_comp,
            "unsolvable_reason": res.unsolvable_reason,
            "ttfv_ms": gen_ms + phase1_ms + res.test_ms if res.verified_comp else None,
            "gen_ms": gen_ms, "phase1_ms": phase1_ms, "test_ms": res.test_ms,
            "comps_tested": len(res.trace),
            "completion_tokens": sum(e.completion_tokens for e in used),
            "prompt_tokens": sum(e.prompt_tokens for e in used),
            "cached_tokens": sum(e.cached_tokens for e in used) if all(e.cached_tokens is not None for e in used) else None,
            "tokens_estimated": any(e.tokens_estimated for e in used),
            "requests": len(used),
            "emitted": sum(1 for e in used if e.fill_hash), "extract_errors": sum(1 for e in used if not e.fill_hash),
            "distinct_alpha": sum(p["distinct_alpha"] for p in per_slot.values()),
            "distinct_exact": sum(p["distinct_exact"] for p in per_slot.values()),
            "per_slot": per_slot,
            "all_slots_have_stub_pass": all(p["stub"]["pass"] > 0 for p in per_slot.values()),
            "comps_all_stub_pass": sum(1 for tr in res.trace if tr.all_stub_pass),
            "comps_all_stub_pass_failed": sum(1 for tr in res.trace if tr.all_stub_pass
                                              and not all(r == "pass" for r in tr.results.values())),
            "heldout_found": heldout is not None,
            "heldout_class_level_pass": heldout.class_level_pass if heldout else None,
            "has_class_level_tests": bool(rt.class_level_tests(task)),
            "defs_total": len(all_defs), "defs_reachable_verified": len(reach & all_defs),
        }
        rd.append("traces.jsonl", {"task_id": task.task_id, "arm": arm, "seed": seed, "N": n,
                                   "trace": [asdict(t) for t in res.trace]})
        rows.append(row)
    with open(rd.path / "events" / f"{task.task_id}__{arm}__s{seed}.jsonl", "w") as f:
        for ev in store.event_log():
            f.write(json.dumps(ev) + "\n")
    return rows


# ---------------------------------------------------------------- C / A arms

async def run_c(client: Client, task, seed: int, Ns: list[int], rd: RunDir) -> list[dict]:
    samples = await whole_class_samples(client, task, n=max(Ns), seed=seed)
    for s in samples:
        rd.append("class_samples.jsonl", s.to_json())
    rows = []
    for n in sorted(Ns):
        used = [s for s in samples if s.sample_idx < n]
        passing = [s for s in used if s.passed]
        rows.append({
            "task_id": task.task_id, "arm": "c", "seed": seed, "N": n, "solved": bool(passing),
            "ttfv_ms": min((s.t_done_ms + s.test_ms for s in passing), default=None),
            "n_pass": len(passing), "extract_errors": sum(1 for s in used if s.extract_error),
            "completion_tokens": sum(s.completion_tokens for s in used),
            "prompt_tokens": sum(s.prompt_tokens for s in used),
            "cached_tokens": sum(s.cached_tokens for s in used) if all(s.cached_tokens is not None for s in used) else None,
            "tokens_estimated": any(s.tokens_estimated for s in used), "requests": len(used),
            "sample_tokens": [s.completion_tokens for s in sorted(used, key=lambda s: s.sample_idx)],
            "sample_passed": [s.passed for s in sorted(used, key=lambda s: s.sample_idx)],
        })
    return rows


async def run_a_greedy(client: Client, task, seed: int, rd: RunDir) -> list[dict]:
    (s,) = await whole_class_samples(client, task, n=1, seed=seed, arm="a_greedy", temperature=0.0, top_p=None)
    rd.append("class_samples.jsonl", s.to_json())
    return [{"task_id": task.task_id, "arm": "a_greedy", "seed": seed, "N": 1, "solved": s.passed,
             "ttfv_ms": s.t_done_ms + s.test_ms if s.passed else None,
             "completion_tokens": s.completion_tokens, "prompt_tokens": s.prompt_tokens,
             "extract_errors": int(bool(s.extract_error)), "requests": 1}]


# ---------------------------------------------------------------- driver

async def main_async(a) -> Path:
    if a.resume:
        rd = RunDir(Path(a.resume))
        cfg = json.loads((rd.path / "config.json").read_text())
        # config.json is never rewritten; widening the task limit is an appended, logged override
        for o in rd.rows("config_overrides.jsonl"):
            cfg = {**cfg, **o["set"]}
        if a.limit and a.limit != cfg.get("limit"):
            o = {"at": datetime.now(timezone.utc).isoformat(), "set": {"limit": a.limit}}
            rd.append("config_overrides.jsonl", o)
            cfg = {**cfg, **o["set"]}
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rd = RunDir(Path(a.out) / f"{ts}_{a.name}")
        cfg = {"arms": a.arms.split(","), "Ns": sorted({int(x) for x in a.Ns.split(",")} | {a.N}),
               "seeds": [int(x) for x in a.seeds.split(",")], "subset": str(a.subset), "limit": a.limit,
               "tasks": a.tasks.split(",") if a.tasks else None, "max_comps": a.max_comps,
               "sampling": SAMPLING, "per_class_timeout_s": 5, "wall_timeout_s": 20,
               "model": os.environ.get("CALM_MODEL"), "base_url": os.environ.get("CALM_BASE_URL"),
               "no_n": os.environ.get("CALM_NO_N"),
               "server_env": {k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")},
               "started": datetime.now(timezone.utc).isoformat()}
        (rd.path / "config.json").write_text(json.dumps(cfg, indent=1))
    tasks = load_subset(Path(cfg["subset"]))
    if cfg.get("tasks"):
        tasks = [(r, t) for r, t in tasks if t.task_id in cfg["tasks"]]
    if cfg.get("limit"):
        tasks = tasks[: cfg["limit"]]
    done = rd.done()
    async with Client() as client:
        cfg_client = client.config()
        if not (rd.path / "client.json").exists():
            (rd.path / "client.json").write_text(json.dumps({**cfg_client, "metrics_start": await client.metrics_snapshot()}))
        jobs = [(seed, i, task) for seed in cfg["seeds"] for i, (_, task) in enumerate(tasks)]
        queue: asyncio.Queue = asyncio.Queue()
        for j in jobs:
            queue.put_nowait(j)

        async def worker() -> None:
            while not queue.empty():
                seed, i, task = queue.get_nowait()
                for arm in cfg["arms"]:
                    if (task.task_id, arm, seed) in done:
                        continue
                    t0 = time.monotonic()
                    try:
                        if arm == "calm":
                            rows = await run_calm(client, task, seed, cfg["Ns"], rd, max_comps=cfg["max_comps"])
                        elif arm == "c":
                            rows = await run_c(client, task, seed, cfg["Ns"], rd)
                        elif arm == "a_greedy":
                            rows = await run_a_greedy(client, task, seed, rd)
                        else:
                            raise ValueError(f"unknown arm {arm}")
                    except Exception as e:  # a task that doesn't fit is excluded and logged, never special-cased
                        rd.append("excluded.jsonl", {"task_id": task.task_id, "arm": arm, "seed": seed,
                                                     "reason": f"{type(e).__name__}: {e}"})
                        console.print(f"[red]{task.task_id} {arm} s{seed}: {e!r}[/red]")
                        continue
                    for r in rows:
                        rd.append("results.jsonl", r)
                    top = rows[-1]
                    console.print(f"[{i + 1}/{len(tasks)}] {task.task_id} {arm} s{seed} N={top['N']} "
                                  f"solved={top['solved']} tok={top['completion_tokens']} "
                                  f"{time.monotonic() - t0:.0f}s")

        await asyncio.gather(*(worker() for _ in range(max(1, a.workers))))
        (rd.path / "client_end.json").write_text(json.dumps({"metrics_end": await client.metrics_snapshot()}))
    return rd.path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="calm,c,a_greedy")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--Ns", default="1,2,4,8")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--subset", default=str(SUBSET))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tasks")
    ap.add_argument("--max-comps", type=int, default=64)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default="main")
    ap.add_argument("--resume")
    ap.add_argument("--workers", type=int, default=1, help="tasks in flight at once (one process, one writer)")
    a = ap.parse_args()
    print(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
