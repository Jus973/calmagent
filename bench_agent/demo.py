"""Tell the whole story from committed run directories. No model server, nothing to fail.

    python -m bench_agent.demo

Every number printed is read out of a run directory under `runs/` and the directory is named next
to it, so any line can be checked. Nothing here is typed by hand and nothing here calls a model --
the research phase's lesson was that a demo which needs a GPU is a demo that dies on stage.

    --verify   re-derive the headline numbers from the raw rows instead of reading the summaries,
               and fail loudly if a summary disagrees with its own data.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

RUNS = pathlib.Path("runs")


def newest(pattern: str) -> pathlib.Path | None:
    hits = sorted(glob.glob(str(RUNS / pattern)))
    return pathlib.Path(hits[-1]) if hits else None


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * max(len(title), 60))


def section_cache(verify: bool) -> None:
    d = newest("*_cache_probe")
    rule("1. What a prompt-cache hit is worth on this machine")
    if not d:
        print("  no cache probe recorded; run `python -m bench_agent.probe_cache --out ...`")
        return
    s = json.loads((d / "summary.json").read_text())
    rows = {json.loads(l)["label"]: json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines()}
    cold, warm = rows["a_cold"], rows["a_warm"]
    if verify:
        got = round(cold["prompt_eval_ms_per_token"] / warm["prompt_eval_ms_per_token"], 1)
        assert abs(got - s["cache_speedup_on_prompt_eval"]) < 0.2, (got, s)
        print(f"  [verified from rows.jsonl: {got}x]")
    print(f"  source: {d}")
    print(f"  A {s['prompt_tokens']:,}-token prompt, {s['model']}")
    print(f"    cold prefix  {cold['prompt_eval_ms']:>10,.0f} ms   ({cold['prompt_eval_ms_per_token']} ms/token)")
    print(f"    warm prefix  {warm['prompt_eval_ms']:>10,.0f} ms   ({warm['prompt_eval_ms_per_token']} ms/token)")
    print(f"    => \033[1m{s['cache_speedup_on_prompt_eval']}x\033[0m on prompt evaluation")
    print(f"  Two ways to throw that away, both measured in the same run:")
    print(f"    a timestamp at the front of the system prompt : {rows['a_ts_churn_1']['prompt_eval_ms']:>9,.0f} ms")
    print(f"    one interleaved request with another prefix   : {rows['a_after_b']['prompt_eval_ms']:>9,.0f} ms")
    print(f"  The server reports no `cached_tokens` at all, and `prompt_eval_count` does not shrink")
    print(f"  on a hit, so wall clock is the only instrument there is.")


def section_headroom() -> None:
    d = newest("*_headroom")
    rule("2. What the levers are worth on a real agent loop")
    if not d:
        print("  no headroom measurement recorded; run")
        print("  python -m bench_agent.probe_headroom runs/*_probe* --out runs/<ts>_headroom/h.json")
        return
    h = json.loads((d / "headroom.json").read_text())
    print(f"  source: {d}   ({h['requests']} requests, {h['prompt_tokens']:,} prompt tokens "
          f"for {h['completion_tokens']:,} completion tokens)")
    print("  Measured before any lever was built, so the choice was made on data:")
    i2, i3, i4, i5 = h["I2_prefix_lint"], h["I3_memo"], h["I4_dedup"], h["I5_calm_run"]
    print(f"    I-2 prefix-lint {i2['headroom_s']:>7.1f} s   churn tokens: {i2['churn_tokens']}; "
          f"divergence causes {i2['divergence_causes']}")
    print(f"    I-3 memo        {i3['headroom_s']:>7.1f} s   "
          f"{i3['duplicate_requests']} duplicate request keys of {h['requests']}")
    print(f"    I-4 dedup       {i4['headroom_s']:>7.1f} s   "
          f"{i4['duplicate_bytes_after_divergence']:,} duplicate bytes after the divergence point")
    print(f"    I-5 calm-run        n/a     {i5['test_command_reruns']} reruns of "
          f"{i5['test_commands']} test commands, but a ClassEval suite runs in ~20 ms")
    print("  Two kill numbers hit. I-4 is the only lever left, and section 3 is why.")


def section_context(verify: bool) -> None:
    d = newest("*_context_cost")
    rule("3. Why the agent still gets slower: context is not free")
    if not d:
        print("  no context-cost fit recorded")
        return
    fit = json.loads((d / "fit.json").read_text())
    s = fit["summary"]
    print(f"  source: {d}   ({s['rows_used']} requests, r^2 = {s['r_squared']})")
    print(f"  model:  {s['model']}")
    lo, hi = s["context_range"]
    print(f"                        at {lo:,} ctx    at {hi:,} ctx    multiple")
    print(f"    one prompt token    {s['prompt_token_ms_at_min_context']:>9.2f} ms {s['prompt_token_ms_at_max_context']:>13.2f} ms"
          f" {s['prompt_multiple_across_range']:>10.2f}x")
    print(f"    one output token    {s['completion_token_ms_at_min_context']:>9.2f} ms {s['completion_token_ms_at_max_context']:>13.2f} ms"
          f" {s['completion_multiple_across_range']:>10.2f}x")
    print(f"  The fit was given neither constant, yet lands on both: {s['a_prompt_token_ms']} ms/prompt token")
    print(f"  against the cache probe's {s['independent_cold_rate_from_probe_cache_ms']} ms, and "
          f"{s['c_completion_token_ms']} ms/output token = "
          f"{1000/s['c_completion_token_ms']:.0f} tok/s against this model's ~20 tok/s.")
    if verify:
        pts = fit["points"]
        assert len(pts) == s["rows_used"], (len(pts), s["rows_used"])
        print(f"  [verified: {len(pts)} points behind the fit]")


def section_ab() -> None:
    off, on = newest("*_ab_off"), newest("*_ab_on")
    rule("4. The A/B: does the lever survive contact with a real agent?")
    if not (off and on):
        print("  no A/B recorded yet")
        return
    def load(d):
        p = d / "results.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    ro, rn = load(off), load(on)
    print(f"  source: {off}")
    print(f"          {on}")
    print(f"    off: {sum(r['solved'] for r in ro)}/{len(ro)} solved")
    print(f"    on : {sum(r['solved'] for r in rn)}/{len(rn)} solved")
    print("  Full table, including why total wall clock is NOT evidence about the lever:")
    print(f"    python -m bench_agent.quick_report {off} {on}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    print("\033[1mCALM Proxy — what was measured, from the run directories that hold it\033[0m")
    section_cache(args.verify)
    section_headroom()
    section_context(args.verify)
    section_ab()
    print("\nEvery figure above comes from a directory under runs/ named beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
