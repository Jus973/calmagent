"""Turn a recorded run into the JSON the demo page replays.

The demo is replay-first on purpose: the page never talks to a model, so it cannot fail on stage,
and every number it shows came out of a real run's append-only logs.

python -m calm_coder.demo.export_web runs/<dir> --task ClassEval_21 --arm v2 [--seed 0]
  -> calm_coder/demo/web/logs/<task>__<arm>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from calm_coder.store.store import Store
from calm_coder.v2 import state

WEB = Path(__file__).parent / "web"


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def store_hash(store: Store) -> str:
    """The claim the shuffle button demonstrates: the store's contents, not its arrival order."""
    h = hashlib.sha256()
    for d in sorted(d.hash for d in store.defs()):
        h.update(b"d" + d.encode())
    for o in sorted(f"{o.test_id}|{o.comp_id}|{o.result}" for o in store.outcomes()):
        h.update(b"o" + o.encode())
    return h.hexdigest()[:16]


def _task(task_id: str):
    """The slot list and its test classes. None when the dataset isn't downloaded on this box —
    the page then shows candidates without per-slot lights rather than refusing to export."""
    try:
        from calm_coder.bench.classeval import SUBSET, load_rows, task_from_row
        if task_id not in json.loads(SUBSET.read_text())["task_ids"]:
            return None
        return task_from_row({r["task_id"]: r for r in load_rows()}[task_id])
    except Exception:
        return None


def build(run: Path, task_id: str, arm: str, seed: int) -> dict:
    task = _task(task_id)
    events = _rows(run / "events" / f"{task_id}__{arm}__s{seed}.jsonl")
    store = Store.from_events(events)
    results = _rows(run / "results.jsonl")
    row = next((r for r in results if r["task_id"] == task_id and r["arm"] == arm and r["seed"] == seed), {})
    emissions = [e for e in _rows(run / "emissions.jsonl")
                 if e["task_id"] == task_id and e["arm"] == arm and e["seed"] == seed]

    # left pane: whole-class best-of-N, from arm C when the run has it, else this arm's round-0 samples
    samples = [s for s in _rows(run / "class_samples.jsonl") if s["task_id"] == task_id and s["seed"] == seed]
    if samples:
        left = {"label": "best-of-N (whole class)",
                "samples": [{"idx": s["sample_idx"], "tokens": s["completion_tokens"], "passed": s["passed"]}
                            for s in sorted(samples, key=lambda s: s["sample_idx"])]}
        c_row = next((r for r in results if r["task_id"] == task_id and r["arm"] == "c" and r["seed"] == seed), {})
        left["decode_tokens"] = c_row.get("completion_tokens", sum(s["completion_tokens"] for s in samples))
        left["prompt_tokens"] = c_row.get("prompt_tokens")
        left["solved"] = bool(c_row.get("solved"))
    else:
        r0 = [e for e in emissions if e["slot"] == "__class__" and not e.get("round")]
        left = {"label": "whole-class samples (this run)", "solved": bool(row.get("seed_comp_solved")),
                "samples": [{"idx": e["sample_idx"], "tokens": e["completion_tokens"], "passed": None} for e in r0],
                "decode_tokens": sum(e["completion_tokens"] for e in r0), "prompt_tokens": None}

    binds = state.comp_bindings(store)
    tokens = {}
    for e in emissions:
        for h in e.get("def_hashes") or ([e["fill_hash"]] if e.get("fill_hash") else []):
            tokens[h] = e["completion_tokens"]
    ev: list[dict] = []
    for i, e in enumerate(events):
        d = e["data"]
        if e["kind"] == "def" and d.get("slot"):
            ev.append({"kind": "def", "i": i, "slot": d["slot"], "hash": d["hash"][:12],
                       "producer": (d.get("meta") or {}).get("producer", "sample"),
                       "round": (d.get("meta") or {}).get("round", 0), "tokens": tokens.get(d["hash"])})
        elif e["kind"] == "outcome":
            ev.append({"kind": "outcome", "i": i, "test": d["test_id"], "comp": d["comp_id"][:12],
                       "result": d["result"],
                       "bindings": {k: v[:12] for k, v in binds.get(d["comp_id"], {}).items()}})
    return {
        "task": task_id, "arm": arm, "seed": seed, "run": str(run),
        "slots": [s.id for s in task.slots] if task else sorted({d.slot for d in store.defs() if d.slot}),
        "slot_tests": dict(task.slot_tests) if task else {},
        "left": left,
        "right": {"label": f"CALM Coder ({arm})", "events": ev,
                  "decode_tokens": row.get("decode_tokens") or row.get("completion_tokens"),
                  "prompt_tokens": row.get("prompt_tokens"), "cached_tokens": row.get("cached_tokens"),
                  "requests": row.get("requests"), "solved": bool(row.get("solved")),
                  "rounds": row.get("rounds") or [], "dead_slots": row.get("dead_slots") or [],
                  "budget_tokens": row.get("budget_tokens"), "feedback": row.get("feedback")},
        "store_hash": store_hash(store),
        "n_defs": len(store.defs()), "n_outcomes": len(store.outcomes()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", default="v2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = build(Path(a.run_dir), a.task, a.arm, a.seed)
    out = WEB / "logs" / f"{a.task}__{a.arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    index = sorted(p.name for p in out.parent.glob("*.json") if p.name != "index.json")
    (out.parent / "index.json").write_text(json.dumps(index, indent=1))
    print(out, f"{data['n_defs']} defs, {data['n_outcomes']} outcomes, store {data['store_hash']}")


if __name__ == "__main__":
    main()
