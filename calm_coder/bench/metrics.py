"""Metrics over a run directory (§3.15). Pure functions of the append-only logs; no new runs.

python -m calm_coder.bench.metrics runs/<dir>    # writes metrics.json + summary.md into the run dir
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from calm_coder.jsonl import read_jsonl

# What-if prices per 1M tokens (input, cached input, output). Labeled as hypothetical, never a measurement.
WHAT_IF_PRICES = {"small-hosted-model (what-if)": {"input": 0.15, "cached_input": 0.075, "output": 0.60}}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on discordant pairs b (A only), c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, reps: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = a - b
    if len(d) == 0:
        return (float("nan"),) * 3
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _pct(xs, q):
    return float(np.percentile(xs, q)) if xs else None


class Run:
    def __init__(self, d: Path):
        self.d = d
        self.config = json.loads((d / "config.json").read_text())
        self.results = read_jsonl(d / "results.jsonl")
        # An arm interrupted mid-task is redone on resume; keep the last copy of each re-logged sample.
        self.emissions = list({(e["task_id"], e["arm"], e["seed"], e["slot"], e["sample_idx"]): e
                               for e in read_jsonl(d / "emissions.jsonl")}.values())
        self.class_samples = list({(e["task_id"], e["arm"], e["seed"], e["sample_idx"]): e
                                   for e in read_jsonl(d / "class_samples.jsonl")}.values())
        self.excluded = read_jsonl(d / "excluded.jsonl")
        self.by = {(r["task_id"], r["arm"], r["seed"], r["N"]): r for r in self.results}
        self.Ns = sorted({r["N"] for r in self.results if r["arm"] == "calm"})
        self.seeds = sorted({r["seed"] for r in self.results})
        # Only (task, seed) pairs where every compared arm finished count (paired analysis).
        arms = set(self.config["arms"])
        have = defaultdict(set)
        for r in self.results:
            have[(r["task_id"], r["seed"])].add(r["arm"])
        self.pairs = sorted(k for k, v in have.items() if {"calm", "c"} <= v and ({"a_greedy"} & arms <= v))
        self.tasks = sorted({t for t, _ in self.pairs}, key=lambda t: int(t.split("_")[1]))

    def row(self, task, arm, seed, N):
        return self.by.get((task, arm, seed, N))

    # ---- C@tokens: the first M C samples that fit within CALM's completion tokens at the same N
    def c_at_tokens(self, task, seed, N) -> tuple[bool, int, int]:
        calm = self.row(task, "calm", seed, N)
        c = self.row(task, "c", seed, max(self.Ns))
        spent, m, solved = 0, 0, False
        for tok, ok in zip(c["sample_tokens"], c["sample_passed"]):
            if spent + tok > calm["completion_tokens"]:
                break
            spent += tok
            m += 1
            solved = solved or ok
        return solved, m, spent

    def per_task(self, fn) -> np.ndarray:
        """Mean over seeds per task (tasks are the unit of inference, §6.5)."""
        return np.array([np.mean([fn(t, s) for s in self.seeds if (t, s) in set(self.pairs)]) for t in self.tasks])


def compute(d: Path) -> dict:
    run = Run(d)
    Ns, Nmax = run.Ns, max(run.Ns)
    n_tasks = len(run.tasks)
    out: dict = {"run": str(d), "config": run.config, "n_tasks": n_tasks, "seeds": run.seeds,
                 "excluded": run.excluded}

    # 1. solve rates
    series = {
        "calm": lambda N: (lambda t, s: float(run.row(t, "calm", s, N)["solved"])),
        "c@N": lambda N: (lambda t, s: float(run.row(t, "c", s, N)["solved"])),
        "c@tokens": lambda N: (lambda t, s: float(run.c_at_tokens(t, s, N)[0])),
    }
    solve: dict = defaultdict(dict)
    for name, f in series.items():
        for N in Ns:
            v = run.per_task(f(N))
            solve[name][N] = {"rate": float(v.mean()) if len(v) else None,
                              "wilson": wilson(v.sum(), len(v))[1:], "k": float(v.sum()), "n": len(v)}
    a_pass1 = run.per_task(lambda t, s: run.row(t, "c", s, Nmax)["n_pass"] / Nmax)
    solve["a_pass1"] = {1: {"rate": float(a_pass1.mean()) if len(a_pass1) else None,
                            "wilson": wilson(a_pass1.sum(), len(a_pass1))[1:], "n": len(a_pass1)}}
    if "a_greedy" in run.config["arms"]:
        g = run.per_task(lambda t, s: float(run.row(t, "a_greedy", s, 1)["solved"]))
        solve["a_greedy"] = {1: {"rate": float(g.mean()) if len(g) else None,
                                 "wilson": wilson(g.sum(), len(g))[1:], "n": len(g)}}
    out["solve"] = solve

    paired = {}
    for N in Ns:
        a = run.per_task(series["calm"](N))
        for other in ("c@N", "c@tokens"):
            b = run.per_task(series[other](N))
            paired[f"calm-{other}@{N}"] = paired_bootstrap(a, b)
    out["paired_bootstrap"] = paired
    # McNemar on (task, seed) pairs: CALM@Nmax vs C@tokens
    b = c = 0
    for t, s in run.pairs:
        x = run.row(t, "calm", s, Nmax)["solved"]
        y = run.c_at_tokens(t, s, Nmax)[0]
        b += x and not y
        c += y and not x
    out["mcnemar_calm_vs_ctokens"] = {"calm_only": b, "ctokens_only": c, "p": mcnemar_exact(b, c)}

    # 2. TTFV at Nmax
    tt = {"calm": [], "c": []}
    ratios = []
    for t, s in run.pairs:
        x, y = run.row(t, "calm", s, Nmax)["ttfv_ms"], run.row(t, "c", s, Nmax)["ttfv_ms"]
        if x is not None:
            tt["calm"].append(x)
        if y is not None:
            tt["c"].append(y)
        if x is not None and y is not None:
            ratios.append(x / y)
    out["ttfv"] = {k: {"median_ms": _pct(v, 50), "p90_ms": _pct(v, 90), "n": len(v), "all_ms": v} for k, v in tt.items()}
    out["ttfv"]["ratio_calm_over_c"] = {"median": _pct(ratios, 50), "q25": _pct(ratios, 25), "q75": _pct(ratios, 75),
                                        "n_cosolved": len(ratios)}

    # 3. independence prediction vs observed
    def p_s(ps: dict) -> float:
        st = ps["stub"]
        den = st["pass"] + st["fail"] + st["error"] + st["timeout"] + st["extract_error"]
        if den > 0:
            return st["pass"] / den
        return ps["fills_passing_slot_test_in_some_comp"] / max(ps["emitted"], 1)
    indep = {}
    for N in Ns:
        pred = run.per_task(lambda t, s: float(np.prod([1 - (1 - p_s(ps)) ** N for ps in
                                                        run.row(t, "calm", s, Nmax)["per_slot"].values()])))
        indep[N] = {"predicted": float(pred.mean()) if len(pred) else None, "observed": solve["calm"][N]["rate"]}
    out["independence"] = indep
    out["stub_inconclusive_share"] = float(np.mean([
        ps["stub"]["inconclusive"] / max(ps["emitted"], 1)
        for t, s in run.pairs for ps in run.row(t, "calm", s, Nmax)["per_slot"].values()])) if run.pairs else None

    # 4. interface mismatch + falsifier F1
    tot = sum(run.row(t, "calm", s, Nmax)["comps_all_stub_pass"] for t, s in run.pairs)
    bad = sum(run.row(t, "calm", s, Nmax)["comps_all_stub_pass_failed"] for t, s in run.pairs)
    elig = [(t, s) for t, s in run.pairs if run.row(t, "calm", s, Nmax)["all_slots_have_stub_pass"]]
    sysm = [(t, s) for t, s in elig if not run.row(t, "calm", s, Nmax)["solved"]]
    out["mismatch"] = {"comps_all_stub_pass": tot, "failed": bad, "rate": bad / tot if tot else None,
                       "tasks_all_slots_stub_pass": len(elig), "systematic": len(sysm),
                       "systematic_rate_of_all_tasks": len(sysm) / len(run.pairs) if run.pairs else None,
                       "systematic_tasks": sorted({t for t, _ in sysm})}

    # 5. held-out: select on method-level tests, score on class-level tests
    ho = [run.row(t, "calm", s, Nmax) for t, s in run.pairs]
    ho = [r for r in ho if r["has_class_level_tests"] and r["heldout_found"]]
    out["heldout"] = {"tasks_with_class_level_tests": sum(run.row(t, "calm", s, Nmax)["has_class_level_tests"]
                                                          for t, s in run.pairs),
                      "selected": len(ho), "class_level_pass": sum(bool(r["heldout_class_level_pass"]) for r in ho),
                      "rate": (sum(bool(r["heldout_class_level_pass"]) for r in ho) / len(ho)) if ho else None}

    # 6. dedup + fan-in
    dd = {}
    for N in Ns:
        rs = [run.row(t, "calm", s, N) for t, s in run.pairs]
        em = sum(r["emitted"] for r in rs)
        dd[N] = {"emitted": em, "alpha": (em - sum(r["distinct_alpha"] for r in rs)) / em if em else None,
                 "exact": (em - sum(r["distinct_exact"] for r in rs)) / em if em else None}
    out["dedup"] = dd
    fan = Counter(e["fill_hash"] for e in run.emissions if e["fill_hash"] and e["arm"] == "calm")
    out["fan_in_hist"] = dict(sorted(Counter(fan.values()).items()))

    # 7. unreachable fraction + comps tested before first verified
    sol = [run.row(t, "calm", s, Nmax) for t, s in run.pairs if run.row(t, "calm", s, Nmax)["solved"]]
    out["unreachable_fraction"] = float(np.mean([1 - r["defs_reachable_verified"] / r["defs_total"] for r in sol])) if sol else None
    out["comps_before_verified"] = sorted(r["comps_tested"] for r in sol)

    # 8. confluence
    cf = d / "confluence.json"
    out["confluence"] = json.loads(cf.read_text()) if cf.exists() else None

    # 9. cost
    cost = {}
    for arm in ("calm", "c"):
        for N in Ns:
            rs = [run.row(t, arm, s, N) for t, s in run.pairs]
            solved = sum(r["solved"] for r in rs)
            pt, ct = sum(r["prompt_tokens"] for r in rs), sum(r["completion_tokens"] for r in rs)
            cost[f"{arm}@{N}"] = {"prompt": pt, "completion": ct, "per_task": (pt + ct) / len(rs) if rs else None,
                                  "per_solved": (pt + ct) / solved if solved else None, "solved": solved,
                                  "cached": (sum(r["cached_tokens"] for r in rs)
                                             if rs and all(r.get("cached_tokens") is not None for r in rs) else None)}

    def first_solve_spend(t, s, arm):
        """Doubling schedule: stop at the smallest N that solves (stop-when-verified); unsolved pay Nmax."""
        for N in Ns:
            r = run.row(t, arm, s, N)
            if r["solved"]:
                return r["prompt_tokens"] + r["completion_tokens"], True
        r = run.row(t, arm, s, Nmax)
        return r["prompt_tokens"] + r["completion_tokens"], False
    ratios, per = [], {"calm": [], "c": []}
    for t, s in run.pairs:
        (x, xs), (y, ys) = first_solve_spend(t, s, "calm"), first_solve_spend(t, s, "c")
        per["calm"].append((x, xs))
        per["c"].append((y, ys))
        ratios.append(x / y)
    cost["doubling"] = {
        "median_ratio_calm_over_c": _pct(ratios, 50),
        "calm_tokens_per_solved": (sum(x for x, _ in per["calm"]) / max(1, sum(s for _, s in per["calm"]))),
        "c_tokens_per_solved": (sum(x for x, _ in per["c"]) / max(1, sum(s for _, s in per["c"]))),
    }
    cost["what_if_prices"] = WHAT_IF_PRICES
    out["cost"] = cost

    out["hypotheses"] = hypotheses(out, Nmax)
    return out


def hypotheses(m: dict, Nmax: int) -> list[dict]:
    s = m["solve"]
    calm, a = s["calm"][Nmax]["rate"], s["a_pass1"][1]["rate"]
    ct = s["c@tokens"][Nmax]["rate"]
    mc = m["mcnemar_calm_vs_ctokens"]
    r = m["ttfv"]["ratio_calm_over_c"]["median"]
    dd = m["dedup"][Nmax]["alpha"]
    cf = m["confluence"]
    cd = m["cost"]["doubling"]
    f1 = m["mismatch"]["systematic_rate_of_all_tasks"]

    def v(ok):
        return "n/a" if ok is None else ("HOLDS" if ok else "FAILS")
    rows = [
        ("H1", f"CALM@{Nmax} ≥ 1.5× holistic pass@1", f"{calm:.2f} vs {a:.2f}" if a is not None else "-",
         v(None if not a else calm >= 1.5 * a)),
        ("H2", f"CALM@{Nmax} > C@tokens, McNemar p<0.1", f"{calm:.2f} vs {ct:.2f}, p={mc['p']:.3f}",
         v(calm > ct and mc["p"] < 0.1)),
        ("H2b", "… by ≥ 1.2× relative", f"{calm:.2f} vs {ct:.2f}", v(None if not ct else calm >= 1.2 * ct)),
        ("H3", "median TTFV ratio CALM/C ≤ 0.5 (co-solved)", f"{r:.2f} (n={m['ttfv']['ratio_calm_over_c']['n_cosolved']})" if r else "no co-solved",
         v(None if r is None else r <= 0.5)),
        ("H4", f"≥10% of fills collapse at N={Nmax} (alpha)", f"{dd:.1%}" if dd is not None else "-", v(None if dd is None else dd >= 0.10)),
        ("H5", "20 shuffled replays, 0 diffs on every log", f"{cf['pass']}/{cf['files']} files" if cf else "not run",
         v(None if not cf else cf["fail"] == 0)),
        ("H6", "doubling-schedule token ratio CALM/C < 1 and fewer tokens per solved",
         f"ratio {cd['median_ratio_calm_over_c']:.2f}; {cd['calm_tokens_per_solved']:.0f} vs {cd['c_tokens_per_solved']:.0f} tok/solved",
         v(cd["median_ratio_calm_over_c"] < 1 and cd["calm_tokens_per_solved"] < cd["c_tokens_per_solved"])),
        ("F1", "falsifier: systematic mismatch > 30% of tasks", f"{f1:.1%}" if f1 is not None else "-",
         "TRIGGERED" if f1 is not None and f1 > 0.30 else "not triggered"),
    ]
    return [{"id": i, "claim": c, "observed": o, "verdict": x} for i, c, o, x in rows]


def summary_md(m: dict) -> str:
    Nmax = max(int(k) for k in m["solve"]["calm"])
    cfg = m["config"]
    L = [f"# Results — {Path(m['run']).name}", "",
         f"Model `{cfg.get('model')}`, T={cfg['sampling']['temperature']}, {m['n_tasks']} ClassEval tasks, "
         f"seeds {m['seeds']}. Verifier = hidden oracle tests for every arm (coverage, §6.1).", "",
         "| Arm | " + " | ".join(f"N={n}" for n in sorted(m["solve"]["calm"])) + " |",
         "|---|" + "---|" * len(m["solve"]["calm"])]
    for arm in ("calm", "c@N", "c@tokens"):
        cells = []
        for n in sorted(m["solve"][arm]):
            x = m["solve"][arm][n]
            cells.append(f"{x['rate']:.2f} [{x['wilson'][0]:.2f}, {x['wilson'][1]:.2f}]")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", f"Holistic pass@1 (unbiased, from C samples): {m['solve']['a_pass1'][1]['rate']:.2f}"
          + (f"; greedy: {m['solve']['a_greedy'][1]['rate']:.2f}" if "a_greedy" in m["solve"] else ""), "",
          "## Pre-registered hypotheses", "", "| ID | Claim | Observed | Verdict |", "|---|---|---|---|"]
    L += [f"| {h['id']} | {h['claim']} | {h['observed']} | {h['verdict']} |" for h in m["hypotheses"]]
    mm = m["mismatch"]
    L += ["", "## Coupling", "",
          f"- Independence-predicted vs observed CALM solve rate: " +
          ", ".join(f"N={n}: {v['predicted']:.2f} vs {v['observed']:.2f}" for n, v in sorted(m["independence"].items())),
          f"- Share of stub-context slot tests that were inconclusive: {m['stub_inconclusive_share']:.0%}",
          f"- Interface mismatch: {mm['failed']}/{mm['comps_all_stub_pass']} compositions whose every fill passed "
          f"its own slot test still failed the module.",
          f"- Systematic mismatch (F1): {mm['systematic']}/{m['n_tasks']} tasks had a stub-passing fill for every "
          f"slot but no verified composition: {mm['systematic_tasks']}.",
          f"- Held-out: {m['heldout']['class_level_pass']}/{m['heldout']['selected']} compositions selected on "
          f"method-level tests also passed class-level tests ({m['heldout']['tasks_with_class_level_tests']} tasks have class-level tests).",
          "", "## Dedup and cost", "",
          f"- Fills collapsed at N={Nmax}: {m['dedup'][Nmax]['alpha']:.1%} alpha-normalized, "
          f"{m['dedup'][Nmax]['exact']:.1%} exact-unparse. Dedup saves test executions, not tokens.",
          f"- Doubling schedule, median tokens-to-first-solve CALM/C: {m['cost']['doubling']['median_ratio_calm_over_c']:.2f}.",
          "- CALM sends the skeleton once per slot request; C once per class sample. Prompt tokens are counted as billed "
          "per request (Ollama has no server-side n and reports no cached tokens).", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    d = Path(ap.parse_args().run_dir)
    m = compute(d)
    (d / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    (d / "summary.md").write_text(summary_md(m))
    print(summary_md(m))


if __name__ == "__main__":
    main()
