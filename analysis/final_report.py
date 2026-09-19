"""Final report over one or more run directories (Phase 4 of the v2 plan).

Arm-agnostic on purpose: it reads results.jsonl, groups by arm (and N, for the arms that sweep a
budget), and produces

  * solve rate per arm with a Wilson interval, tasks as the unit of inference;
  * every pairwise comparison against the configured baselines: paired McNemar (exact) and a
    paired bootstrap difference with 10,000 resamples;
  * a systems table (decode tokens, prompt tokens, cached prompt tokens, requests, wall time);
  * for the v2 arms: repair rounds used, dead slots before and after, and what the budget bought.

python -m analysis.final_report runs/<dir> [runs/<dir2> ...] [--baseline c --baseline calm]
Writes results/final.{md,json} (and a copy inside the first run dir).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calm_coder.bench.metrics import mcnemar_exact, paired_bootstrap, wilson

V2_ARMS = ("v2", "v2_pm", "v2_f0", "v2_f2", "wcr")


def load(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        p = d / "results.jsonl"
        if not p.exists():
            raise SystemExit(f"no results.jsonl in {d}")
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["run"] = str(d)
                rows.append(r)
    # a resumed run may re-log a (task, arm, seed, N); the last copy wins
    return list({(r["task_id"], r["arm"], r["seed"], r.get("N")): r for r in rows}.values())


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

    def per_task(arm: str) -> np.ndarray:
        """Mean over seeds per task: tasks, not samples, are the unit of inference."""
        vals = []
        for t in tasks:
            xs = [by_arm[arm][(t, s)] for s in seeds if (t, s) in by_arm[arm]]
            if xs:
                vals.append(float(np.mean(xs)))
        return np.array(vals)

    for arm in arms:
        v = per_task(arm)
        p, lo, hi = wilson(v.sum(), len(v))
        out["solve"][arm] = {"rate": float(v.mean()) if len(v) else None, "wilson": [lo, hi],
                             "k": float(v.sum()), "n": int(len(v))}

    for arm in arms:
        for base in baselines:
            if base == arm or base not in by_arm:
                continue
            common = sorted(set(by_arm[arm]) & set(by_arm[base]))
            b = sum(1 for k in common if by_arm[arm][k] and not by_arm[base][k])
            c = sum(1 for k in common if by_arm[base][k] and not by_arm[arm][k])
            ct = sorted({t for t, _ in common})
            a_v = np.array([np.mean([by_arm[arm][(t, s)] for s in seeds if (t, s) in by_arm[arm]]) for t in ct])
            b_v = np.array([np.mean([by_arm[base][(t, s)] for s in seeds if (t, s) in by_arm[base]]) for t in ct])
            diff, blo, bhi = paired_bootstrap(a_v, b_v)
            out["paired"][f"{arm}-{base}"] = {"n_pairs": len(common), "arm_only": b, "base_only": c,
                                              "mcnemar_p": mcnemar_exact(b, c),
                                              "diff": diff, "bootstrap_ci": [blo, bhi]}

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
         f"{m['n_tasks']} tasks, seeds {m['seeds']}.", "",
         "## Solve rate (tasks are the unit of inference, Wilson 95%)", "",
         "| arm | N | solve rate | 95% CI |", "| --- | --- | --- | --- |"]
    for arm, s in sorted(m["solve"].items(), key=lambda kv: -(kv[1]["rate"] or 0)):
        lo, hi = s["wilson"]
        L.append(f"| {arm} | {m['N_per_arm'].get(arm) or '-'} | {s['rate']:.2f} | [{lo:.2f}, {hi:.2f}] |")
    L += ["", "## Paired comparisons", "",
          "| comparison | pairs | arm only | baseline only | McNemar p | diff | bootstrap 95% CI |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, p in sorted(m["paired"].items()):
        lo, hi = p["bootstrap_ci"]
        L.append(f"| {k} | {p['n_pairs']} | {p['arm_only']} | {p['base_only']} | {p['mcnemar_p']:.3f} | "
                 f"{p['diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
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
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final.json").write_text(json.dumps(m, indent=1))
    (out / "final.md").write_text(to_md(m))
    (dirs[0] / "final.md").write_text(to_md(m))
    print(to_md(m))


if __name__ == "__main__":
    main()
