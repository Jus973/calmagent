"""Zero-token recon: what does pooling already-sampled candidates buy? (Phase 0 of the v2 plan.)

For every task of a finished run, build one store out of

  * the per-method fills the CALM arm sampled (from its event log), and
  * the methods of every whole-class sample arm C produced (decomposed on the way in),

then run the normal search. The three columns that matter are: solved by C alone, solved by CALM
alone, solved by the pool. The gap is the headroom a producer-agnostic store gives us before a
single new token is spent.

python -m analysis.pool_existing runs/<dir> [--seed 0] [--N 8] [--cap 2000] [--par 6]
Writes <dir>/analysis/pool_existing.{jsonl,md}; the run's own logs are never touched.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from calm_coder.agents.scheduler import Scheduler
from calm_coder.bench.classeval import load_rows, task_from_row
from calm_coder.bench.posthoc import _ranked_cands, exhaustive
from calm_coder.jsonl import append_jsonl, read_jsonl, write_jsonl
from calm_coder.store.derive import verified
from calm_coder.store.store import Store
from calm_coder.v2.decompose import ingest_class_sample


def _within_n(events: list[dict], n: int) -> list[dict]:
    """Definitions from the first `n` samples per slot.

    One run logs one event stream at its largest N, so asking for a smaller N has to mean the
    pool that run *would* have had at N, not the whole pool relabelled.
    """
    out = []
    for e in events:
        if e["kind"] != "def":
            continue
        idx = (e["data"].get("meta") or {}).get("sample_idx")
        if idx is None or idx < n:
            out.append(e)
    return out


async def pool_task(task, calm_events: list[dict], class_samples: list[dict], *, cap: int, par: int,
                    max_comps: int, n: int | None = None) -> dict:
    store = Store.from_events(_within_n(calm_events, n) if n is not None
                              else [e for e in calm_events if e["kind"] == "def"])
    from_calm = {s.id: len(store.defs_for_slot(s.id)) for s in task.slots}
    seed_comps = []
    for s in sorted(class_samples, key=lambda s: s["sample_idx"]):
        ing = ingest_class_sample(task, store, s["text"], {"from": "c_sample", "sample_idx": s["sample_idx"]})
        if ing.complete and not ing.error:
            seed_comps.append(ing.bindings)
    cands = {s.id: sorted(d.hash for d in store.defs_for_slot(s.id)) for s in task.slots}
    out = {"slots": {s.id: {"calm": from_calm[s.id], "pooled": len(cands[s.id])} for s in task.slots},
           "seed_comps": len(seed_comps), "pooled_solved": False, "pooled_comps": 0,
           "exhaustive": None, "events": store.event_log()}
    if any(not v for v in cands.values()):
        out["unsolvable_reason"] = "no_fill_for_some_slot"
        return out
    sched = Scheduler(task, store, max_comps=max_comps)
    await sched.phase1([h for hs in cands.values() for h in hs])
    res = await sched.search(cands, {}, {})
    out["pooled_solved"] = res.verified_comp is not None
    out["pooled_comps"] = len(res.trace)
    if not out["pooled_solved"]:
        stub_pass = {h for h, r in sched._stub.items() if r == "pass"}
        out["exhaustive"] = await exhaustive(task, store, _ranked_cands(task, store, stub_pass), cap, par)
        out["pooled_solved_exhaustive"] = bool(verified(store, task))
    out["events"] = store.event_log()
    return out


async def main_async(d: Path, seed: int, N: int, cap: int, par: int, max_comps: int) -> None:
    out_dir = d / "analysis"
    (out_dir / "events").mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r.get("seed"), r.get("N")) for r in read_jsonl(out_dir / "pool_existing.jsonl")}
    rows = {r["task_id"]: r for r in load_rows()}
    results = read_jsonl(d / "results.jsonl")
    pick = lambda arm: {r["task_id"]: r for r in results                                # noqa: E731
                        if r["arm"] == arm and r.get("N") == N and r["seed"] == seed}
    calm, c = pick("calm"), pick("c")
    if not calm and not c:
        raise SystemExit(f"no c/calm results in {d} at N={N} seed={seed}; "
                         f"present: {sorted({(r['arm'], r.get('N'), r['seed']) for r in results})}")
    uniq: dict[tuple, dict] = {}
    for s in read_jsonl(d / "class_samples.jsonl"):
        if s["arm"] == "c" and s["seed"] == seed and s["sample_idx"] < N:
            uniq[(s["task_id"], s["arm"], s["seed"], s["sample_idx"])] = s   # a resumed run re-logs
    samples: dict[str, list[dict]] = {}
    for s in uniq.values():
        samples.setdefault(s["task_id"], []).append(s)
    for tid in sorted(set(calm) | set(c)):
        if (tid, seed, N) in done or tid not in rows:
            continue
        task = task_from_row(rows[tid])
        r = await pool_task(task, read_jsonl(d / "events" / f"{tid}__calm__s{seed}.jsonl"),
                            samples.get(tid, []), cap=cap, par=par, max_comps=max_comps, n=N)
        write_jsonl(out_dir / "events" / f"{tid}__pool__s{seed}_N{N}.jsonl", r.pop("events", []))
        row = {"task_id": tid, "seed": seed, "N": N, "calm_solved": calm.get(tid, {}).get("solved"),
               "c_solved": c.get(tid, {}).get("solved"), **r}
        append_jsonl(out_dir / "pool_existing.jsonl", row)
        print(row["task_id"], "c=", row["c_solved"], "calm=", row["calm_solved"], "pooled=", row["pooled_solved"])
    write_md(out_dir)


def write_md(out_dir: Path) -> None:
    rows = read_jsonl(out_dir / "pool_existing.jsonl")
    if not rows:
        return
    lines = ["# Pooling existing samples (zero new tokens)", ""]
    configs: dict[tuple, list[dict]] = {}
    for r in rows:                                 # one denominator per (seed, N), never mixed
        configs.setdefault((r.get("seed"), r.get("N")), []).append(r)
    for (seed, N), rs in sorted(configs.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        n = len(rs)
        got = lambda k: sum(1 for r in rs if r.get(k))            # noqa: E731
        new = [r["task_id"] for r in rs if r["pooled_solved"] and not (r["c_solved"] or r["calm_solved"])]
        lines += [f"## seed {seed}, N {N}", "",
                  f"| arm | solved | of {n} |", "| --- | --- | --- |",
                  f"| C | {got('c_solved')} | {n} |",
                  f"| CALM | {got('calm_solved')} | {n} |",
                  f"| pooled store, same search | {got('pooled_solved')} | {n} |",
                  f"| pooled store, exhaustive | {got('pooled_solved') + got('pooled_solved_exhaustive')} | {n} |",
                  "", f"Tasks only the pool solves: {', '.join(new) or 'none'}", "",
                  "| task | C | CALM | pooled | candidates/slot (CALM -> pooled) |",
                  "| --- | --- | --- | --- | --- |"]
        for r in rs:
            cand = ", ".join(f"{s}:{v['calm']}->{v['pooled']}" for s, v in r["slots"].items())
            lines.append(f"| {r['task_id']} | {r['c_solved']} | {r['calm_solved']} | "
                         f"{r['pooled_solved']} | {cand} |")
        lines.append("")
    (out_dir / "pool_existing.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--max-comps", type=int, default=64)
    a = ap.parse_args()
    asyncio.run(main_async(Path(a.run_dir), a.seed, a.N, a.cap, a.par, a.max_comps))


if __name__ == "__main__":
    main()
