"""Turn two A/B run directories into the table the overnight contract asks for.

    python -m bench_agent.quick_report runs/<ts>_ab_off runs/<ts>_ab_on

Written as the fallback for `analysis/proxy_report.py` (CF's). If that script lands, prefer it;
this one exists so the A/B is never blocked on a document.

Every figure here is read out of `results.jsonl` and `trace.jsonl`. Two are derived rather than
measured, and are labelled as such wherever they are printed:

* `prefill_s` -- upstream wall minus `completion_tokens x decode_ms_per_token`. Ollama does not
  separate the two on the OpenAI-compatible endpoint, and mini-swe-agent does not stream, so there
  is no first-token boundary to read. The decode constant is measured, not guessed
  (`bench_agent/probe_cache.py`), but it is a constant, so treat `prefill_s` as a decomposition of
  a measured total rather than as a second measurement.
* `no_cache_s` -- what the same prompts would have cost with no prefix cache at all, at the
  machine's measured cold rate. It is the denominator for "how much of the available saving is the
  cache actually capturing", not a claim about any run that happened.

The solve-rate difference is reported with a Wilson interval on each arm and an exact McNemar
p-value on the paired tasks, because with 10 tasks nothing else is honest: a one-task difference
is well inside the noise, and the report says so rather than leaving the reader to assume.

**Total wall clock is not evidence about the lever, and this report says so where it prints it.**
At temperature 0 the agent is deterministic given its prompt, so the first message the lever
rewrites forks the trajectory; from there the arms are different agents taking different numbers of
turns. A run ending at 127 s instead of the 480 s cap usually means the agent gave up early, not
that tokens got cheaper. The per-request block below normalises that out, and
`bench_agent/replay_bench.py` measures the lever with the agent removed entirely.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from itertools import combinations

COLD_MS_PER_PROMPT_TOKEN = 5.96
DECODE_MS_PER_TOKEN = 50.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs (b on-only, c off-only)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_arm(d: pathlib.Path) -> dict:
    results = {}
    rp = d / "results.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                results[r["task_id"]] = r
    trace = []
    tp = d / "trace.jsonl"
    if tp.exists():
        for line in tp.read_text().splitlines():
            if line.strip():
                trace.append(json.loads(line))
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    return {"dir": d, "results": results, "trace": trace, "config": cfg}


def trace_totals(trace: list[dict]) -> dict:
    pt = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in trace)
    ct = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in trace)
    wall = sum((r.get("timing_ms") or {}).get("done") or 0 for r in trace)
    dedup_replaced = sum((r.get("dedup") or {}).get("replaced") or 0 for r in trace)
    dedup_saved = sum((r.get("dedup") or {}).get("bytes_saved") or 0 for r in trace)
    decode_ms = ct * DECODE_MS_PER_TOKEN
    return {
        "requests": len(trace), "prompt_tokens": pt, "completion_tokens": ct,
        "upstream_wall_s": wall / 1000,
        "decode_s": decode_ms / 1000,
        "prefill_s": max(wall - decode_ms, 0) / 1000,
        "no_cache_s": pt * COLD_MS_PER_PROMPT_TOKEN / 1000,
        "dedup_replaced": dedup_replaced, "dedup_bytes_saved": dedup_saved,
    }


def main() -> int:
    dirs = [pathlib.Path(a) for a in sys.argv[1:]]
    if len(dirs) < 1:
        print(__doc__)
        return 2
    arms = {}
    for d in dirs:
        name = d.name.rsplit("_", 1)[-1]
        arms[name] = load_arm(d)

    print("# A/B report\n")
    for name, a in arms.items():
        cfg = a["config"]
        print(f"- **{name}** `{a['dir'].name}` agent={cfg.get('agent')} model={cfg.get('model')} "
              f"proxy={cfg.get('proxy')} flags={cfg.get('proxy_flags')} "
              f"commit={str(cfg.get('git_commit'))[:8]} ollama={cfg.get('ollama_version')}")
    print()

    # ---- per task ----
    all_tasks = sorted({t for a in arms.values() for t in a["results"]})
    print("## Per task\n")
    head = f"| task | " + " | ".join(f"{n}: solved / wall s / reason" for n in arms) + " |"
    print(head)
    print("|" + "---|" * (len(arms) + 1))
    for t in all_tasks:
        cells = []
        for a in arms.values():
            r = a["results"].get(t)
            cells.append("—" if not r else
                         f"{'yes' if r['solved'] else 'no'} / {r['wall_s']:.0f} / {r['agent_reason']}")
        print(f"| {t} | " + " | ".join(cells) + " |")
    print()

    # ---- pooled ----
    print("## Pooled\n")
    print("| metric | " + " | ".join(arms) + " |")
    print("|" + "---|" * (len(arms) + 1))
    tot = {n: trace_totals(a["trace"]) for n, a in arms.items()}
    solved = {n: sum(1 for r in a["results"].values() if r["solved"]) for n, a in arms.items()}
    ntask = {n: len(a["results"]) for n, a in arms.items()}

    def row(label: str, fmt):
        print(f"| {label} | " + " | ".join(fmt(n) for n in arms) + " |")

    row("tasks run", lambda n: str(ntask[n]))
    row("solved", lambda n: f"{solved[n]}/{ntask[n]}")
    row("solve rate (Wilson 95%)", lambda n: (
        f"{solved[n]/max(ntask[n],1):.0%} "
        f"[{wilson(solved[n], ntask[n])[0]:.0%}, {wilson(solved[n], ntask[n])[1]:.0%}]"))
    row("agent wall, total s", lambda n: f"{sum(r['wall_s'] for r in arms[n]['results'].values()):.0f}")
    row("requests", lambda n: str(tot[n]["requests"]))
    row("prompt tokens", lambda n: f"{tot[n]['prompt_tokens']:,}")
    row("completion tokens", lambda n: f"{tot[n]['completion_tokens']:,}")
    row("upstream wall s", lambda n: f"{tot[n]['upstream_wall_s']:.0f}")
    row("· decode s (derived)", lambda n: f"{tot[n]['decode_s']:.0f}")
    row("· prefill s (derived)", lambda n: f"{tot[n]['prefill_s']:.0f}")
    row("prefill with no cache s (derived)", lambda n: f"{tot[n]['no_cache_s']:.0f}")
    row("cache capture of available saving", lambda n: (
        f"{100*(tot[n]['no_cache_s']-tot[n]['prefill_s'])/max(tot[n]['no_cache_s'],1e-9):.1f}%"))
    row("dedup: messages replaced", lambda n: str(tot[n]["dedup_replaced"]))
    row("dedup: bytes removed from prompts", lambda n: f"{tot[n]['dedup_bytes_saved']:,}")
    print()
    print("> Total wall clock and total token counts above are **confounded**: the arms take "
          "different numbers of turns because the lever forks the agent's trajectory. Compare the "
          "per-request block below, and see `bench_agent/replay_bench.py` for the lever measured "
          "without an agent.\n")

    print("## Per request (normalised for how many turns the agent took)\n")
    print("| metric | " + " | ".join(arms) + " |")
    print("|" + "---|" * (len(arms) + 1))

    def per_req(n: str, key: str) -> float:
        r = tot[n]["requests"]
        return tot[n][key] / r if r else 0.0

    row("requests per task", lambda n: f"{tot[n]['requests']/max(ntask[n],1):.1f}")
    row("prompt tokens per request", lambda n: f"{per_req(n, 'prompt_tokens'):,.0f}")
    row("completion tokens per request", lambda n: f"{per_req(n, 'completion_tokens'):,.0f}")
    row("upstream ms per request", lambda n: f"{1000*per_req(n, 'upstream_wall_s'):,.0f}")
    row("upstream ms per completion token", lambda n: (
        f"{1000*tot[n]['upstream_wall_s']/max(tot[n]['completion_tokens'],1):,.0f}"))
    print()

    # ---- paired ----
    if len(arms) == 2:
        (na, a), (nb, b) = list(arms.items())
        paired = [t for t in all_tasks if t in a["results"] and t in b["results"]]
        bb = sum(1 for t in paired if b["results"][t]["solved"] and not a["results"][t]["solved"])
        cc = sum(1 for t in paired if a["results"][t]["solved"] and not b["results"][t]["solved"])
        p = mcnemar_exact(bb, cc)
        print("## Paired\n")
        print(f"- {len(paired)} paired tasks; discordant: {nb}-only {bb}, {na}-only {cc}")
        print(f"- exact McNemar p = {p:.3f}")
        verdict = ("solve rate is unchanged within noise" if p > 0.1 else
                   f"solve rate differs (p={p:.3f})")
        print(f"- **{verdict}.** With {len(paired)} tasks this test can only detect a large "
              f"difference; a one- or two-task gap is not evidence either way.")
        if cc > bb:
            print(f"- **`{nb}` lost {cc - bb} net solve(s) against `{na}`.** Per the contract that "
                  f"is a cut, not something to explain away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
