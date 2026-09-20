"""Final report over one or more run directories (Phase 4 of the v2 plan).

Arm-agnostic on purpose: it reads results.jsonl, groups by arm (and N, for the arms that sweep a
budget), and produces

  * solve rate per arm, tasks as the unit of inference: a Wilson interval where a task's outcome
    is binary (one seed, or every seed agreeing) and a bootstrap interval over tasks otherwise,
    because a seed-averaged task is not a Bernoulli trial and Wilson would read it as one;
  * every pairwise comparison against the configured baselines: a paired bootstrap difference
    with 10,000 resamples over the tasks both arms ran, and exact McNemar over the tasks where
    both arms are unanimous across seeds (the rest are reported as ambiguous, never dropped
    silently);
  * per-arm coverage: how many tasks the arm ran, which of the union it is missing, and the
    tasks the run excluded;
  * a systems table (decode tokens, prompt tokens, cached prompt tokens, requests, wall time);
  * for the v2 arms: repair rounds used, dead slots before and after, and what the budget bought.

python -m analysis.final_report runs/<dir> [runs/<dir2> ...] [--baseline c --baseline calm]
Writes results/final.{md,json} and a copy under <first run dir>/analysis/ (run logs stay untouched).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calm_coder.bench.metrics import dollars_for, mcnemar_exact, paired_bootstrap, price_for_model, wilson
from calm_coder.jsonl import read_jsonl

V2_ARMS = ("v2", "v2_pm", "v2_f0", "v2_f2", "wcr")


def load_excluded(dirs: list[Path]) -> list[dict]:
    """Tasks a run refused to attempt. They are not failures and they are not silence either."""
    return [r for d in dirs for r in read_jsonl(d / "excluded.jsonl")]


def load_emissions(dirs: list[Path]) -> list[dict]:
    return [r for d in dirs for r in read_jsonl(d / "emissions.jsonl")]


def tokens_by_model(emissions: list[dict]) -> dict:
    """What-if dollars from logged emissions. Local models price at $0; Grok uses list prices."""
    agg: dict[str, dict] = defaultdict(lambda: {
        "prompt": 0, "cached": 0, "completion": 0, "requests": 0, "emissions": 0})
    for e in emissions:
        m = e.get("model") or "unknown"
        a = agg[m]
        a["prompt"] += e.get("prompt_tokens") or 0
        a["cached"] += e.get("cached_tokens") or 0
        a["completion"] += e.get("completion_tokens") or 0
        a["emissions"] += 1
        if e.get("prompt_tokens"):
            a["requests"] += 1
    out = {}
    for m, a in sorted(agg.items()):
        label, prices = price_for_model(m)
        out[m] = {**a, "price_table": label,
                  "what_if_usd": dollars_for(a["prompt"], a["cached"], a["completion"], prices)}
    return out


def load(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        if not (d / "results.jsonl").exists():
            raise SystemExit(f"no results.jsonl in {d}")
        rows += [{**r, "run": str(d)} for r in read_jsonl(d / "results.jsonl")]
    # a resumed run may re-log a (task, arm, seed, N); the last copy wins
    return list({(r["task_id"], r["arm"], r["seed"], r.get("N")): r for r in rows}.values())


def _mean_ci(v: np.ndarray, reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over tasks, for the case where a task's value is a seed mean."""
    if not len(v):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def top_n(rows: list[dict], arm: str) -> int | None:
    ns = {r.get("N") for r in rows if r["arm"] == arm and r.get("N") is not None}
    return max(ns) if ns else None


def solved_by(rows: list[dict]) -> dict[str, dict[tuple[str, int], bool]]:
    """arm -> (task, seed) -> solved, at each arm's largest N."""
    out: dict[str, dict[tuple[str, int], bool]] = defaultdict(dict)
    for arm in sorted({r["arm"] for r in rows}):
        n = top_n(rows, arm)
        for r in rows:
            if r["arm"] == arm and (n is None or r.get("N") == n):
                out[arm][(r["task_id"], r["seed"])] = bool(r["solved"])
    return out


def compute(rows: list[dict], baselines: list[str]) -> dict:
    by_arm = solved_by(rows)
    arms = sorted(by_arm)
    tasks = sorted({t for m in by_arm.values() for t, _ in m})
    seeds = sorted({s for m in by_arm.values() for _, s in m})
    out: dict = {"arms": arms, "n_tasks": len(tasks), "seeds": seeds,
                 "N_per_arm": {a: top_n(rows, a) for a in arms}, "solve": {}, "paired": {}, "systems": {},
                 "v2": {}}

    def seed_values(arm: str, t: str) -> list[bool]:
        return [by_arm[arm][(t, s)] for s in seeds if (t, s) in by_arm[arm]]

    def per_task(arm: str, over: list[str] | None = None) -> np.ndarray:
        """Mean over seeds per task: tasks, not samples, are the unit of inference."""
        vals = [float(np.mean(xs)) for t in (over or tasks) if (xs := seed_values(arm, t))]
        return np.array(vals)

    for arm in arms:
        v = per_task(arm)
        ran = [t for t in tasks if seed_values(arm, t)]
        binary = bool(len(v)) and all(x in (0.0, 1.0) for x in v)
        if binary:
            _, lo, hi = wilson(v.sum(), len(v))
            ci = "wilson"
        else:
            lo, hi = _mean_ci(v)
            ci = "bootstrap"
        out["solve"][arm] = {"rate": float(v.mean()) if len(v) else None, "ci": [lo, hi],
                             "ci_kind": ci, "k": float(v.sum()), "n": int(len(v)),
                             "seeds_per_task": sorted({len(seed_values(arm, t)) for t in ran}),
                             "missing_tasks": [t for t in tasks if t not in ran]}

    for arm in arms:
        for base in baselines:
            if base == arm or base not in by_arm:
                continue
            ct = sorted({t for t, _ in set(by_arm[arm]) & set(by_arm[base])})
            a_v, b_v = per_task(arm, ct), per_task(base, ct)
            diff, blo, bhi = paired_bootstrap(a_v, b_v)
            # McNemar is a test on discordant *pairs*, so a task only enters it when both arms
            # gave the same answer on every seed; anything else is counted and reported instead.
            firm = [t for t in ct if len(set(seed_values(arm, t))) == 1
                    and len(set(seed_values(base, t))) == 1]
            b = sum(1 for t in firm if seed_values(arm, t)[0] and not seed_values(base, t)[0])
            c = sum(1 for t in firm if seed_values(base, t)[0] and not seed_values(arm, t)[0])
            out["paired"][f"{arm}-{base}"] = {
                "unit": "task", "n_tasks": len(ct), "mcnemar_tasks": len(firm),
                "ambiguous_tasks": len(ct) - len(firm), "arm_only": b, "base_only": c,
                "mcnemar_p": mcnemar_exact(b, c), "diff": diff, "bootstrap_ci": [blo, bhi]}

    for arm in arms:
        n = top_n(rows, arm)
        rs = [r for r in rows if r["arm"] == arm and (n is None or r.get("N") == n)]
        num = lambda k: [r[k] for r in rs if isinstance(r.get(k), (int, float))]          # noqa: E731
        ttfv = [r["ttfv_ms"] for r in rs if r.get("ttfv_ms")]
        out["systems"][arm] = {
            "tasks": len(rs),
            "decode_tokens_mean": float(np.mean(num("completion_tokens") or [0])),
            "prompt_tokens_mean": float(np.mean(num("prompt_tokens") or [0])),
            "cached_tokens_mean": float(np.mean(num("cached_tokens"))) if num("cached_tokens") else None,
            "requests_mean": float(np.mean(num("requests") or [0])),
            "ttfv_ms_median": float(np.median(ttfv)) if ttfv else None,
            "over_budget": sum(1 for r in rs if r.get("over_budget")),
        }

    for arm in [a for a in arms if a in V2_ARMS]:
        rs = [r for r in rows if r["arm"] == arm]
        rounds = [len([x for x in r.get("rounds", []) if x["producer"] != "whole_class"]) for r in rs]
        dead0 = [len(r["rounds"][0].get("dead_after") or []) for r in rs if r.get("rounds")]
        out["v2"][arm] = {
            "repair_rounds_mean": float(np.mean(rounds)) if rounds else 0.0,
            "solved_by_seed_samples": sum(1 for r in rs if r.get("seed_comp_solved")),
            "solved_total": sum(1 for r in rs if r["solved"]),
            "dead_slots_after_sampling_mean": float(np.mean(dead0)) if dead0 else None,
            "dead_slots_at_end_mean": float(np.mean([len(r.get("dead_slots") or []) for r in rs])) if rs else None,
            "feedback": sorted({r.get("feedback") for r in rs if r.get("feedback")}),
        }
    return out


def to_md(m: dict) -> str:
    L = ["# CALM Coder v2 — final report", "",
         f"{m['n_tasks']} tasks in the union, seeds {m['seeds']}, "
         f"{len(m.get('excluded', []))} tasks excluded by the run.", "",
         "## Solve rate (tasks are the unit of inference)", "",
         "| arm | N | tasks | seeds/task | solve rate | 95% CI | interval | missing tasks |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, s in sorted(m["solve"].items(), key=lambda kv: -(kv[1]["rate"] or 0)):
        lo, hi = s["ci"]
        miss = ", ".join(s["missing_tasks"]) or "-"
        L.append(f"| {arm} | {m['N_per_arm'].get(arm) or '-'} | {s['n']} | "
                 f"{','.join(str(x) for x in s['seeds_per_task']) or '-'} | {s['rate']:.2f} | "
                 f"[{lo:.2f}, {hi:.2f}] | {s['ci_kind']} | {miss} |")
    L += ["", "## Paired comparisons (unit: task)", "",
          "| comparison | tasks | diff | bootstrap 95% CI | McNemar tasks | arm only | baseline only "
          "| McNemar p | ambiguous |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, p in sorted(m["paired"].items()):
        lo, hi = p["bootstrap_ci"]
        L.append(f"| {k} | {p['n_tasks']} | {p['diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                 f"{p['mcnemar_tasks']} | {p['arm_only']} | {p['base_only']} | "
                 f"{p['mcnemar_p']:.3f} | {p['ambiguous_tasks']} |")
    L += ["", "## Systems", "",
          "| arm | tasks | decode tok | prompt tok | cached prompt tok | requests | median TTFV ms | over budget |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, s in sorted(m["systems"].items()):
        cached = f"{s['cached_tokens_mean']:.0f}" if s["cached_tokens_mean"] is not None else "-"
        ttfv = f"{s['ttfv_ms_median']:.0f}" if s["ttfv_ms_median"] is not None else "-"
        L.append(f"| {arm} | {s['tasks']} | {s['decode_tokens_mean']:.0f} | {s['prompt_tokens_mean']:.0f} | "
                 f"{cached} | {s['requests_mean']:.1f} | {ttfv} | {s['over_budget']} |")
    if m["v2"]:
        L += ["", "## Repair", "",
              "| arm | feedback | repair rounds | solved by samples alone | solved | dead slots after sampling | "
              "dead slots at end |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for arm, s in sorted(m["v2"].items()):
            d0 = f"{s['dead_slots_after_sampling_mean']:.2f}" if s["dead_slots_after_sampling_mean"] is not None else "-"
            d1 = f"{s['dead_slots_at_end_mean']:.2f}" if s["dead_slots_at_end_mean"] is not None else "-"
            L.append(f"| {arm} | {','.join(s['feedback']) or '-'} | {s['repair_rounds_mean']:.2f} | "
                     f"{s['solved_by_seed_samples']} | {s['solved_total']} | {d0} | {d1} |")
    by_model = m.get("tokens_by_model") or {}
    if by_model:
        L += ["", "## What-if cost by model", "",
              "Labeled prices, not a measurement. Local Qwen/Llama are $0. Grok rows use xAI list "
              "prices (input / cached input / output per 1M). Cached tokens shrink the uncached "
              "input line; they are never estimated.", "",
              "| model | emissions | prompt | cached | completion | price table | what-if $ |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for model, s in by_model.items():
            usd = s["what_if_usd"]["total"]
            L.append(f"| {model} | {s['emissions']} | {s['prompt']} | {s['cached']} | "
                     f"{s['completion']} | {s['price_table']} | {usd:.4f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--baseline", action="append", default=None, help="repeatable; default: c and calm")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    dirs = [Path(d) for d in a.run_dirs]
    rows = load(dirs)
    m = compute(rows, a.baseline or ["c", "calm"])
    m["runs"] = [str(d) for d in dirs]
    m["excluded"] = load_excluded(dirs)
    m["tokens_by_model"] = tokens_by_model(load_emissions(dirs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final.json").write_text(json.dumps(m, indent=1))
    (out / "final.md").write_text(to_md(m))
    copy = dirs[0] / "analysis"          # a run directory's own logs are never written to
    copy.mkdir(exist_ok=True)
    (copy / "final.md").write_text(to_md(m))
    print(to_md(m))


if __name__ == "__main__":
    main()
