"""Measure what a prompt-prefix cache hit is worth on this machine's model server.

Ollama 0.21.2 does not report `usage.prompt_tokens_details.cached_tokens` on /v1, and
/api/chat reports `prompt_eval_count` as the FULL prompt whether or not the prefix was
cached. So token accounting cannot see the cache at all here. `prompt_eval_duration` can:
a cached prefix is not re-evaluated, so the duration collapses.

This writes a run directory with one JSON row per request. Every number in the docs comes
from here; nothing is typed by hand.

    python -m bench_agent.probe_cache --out runs/<ts>_cache_probe/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:11434"


def chat(model: str, messages: list[dict], num_predict: int = 5) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        BASE + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    wall_ms = (time.time() - t0) * 1000
    return {
        "wall_ms": round(wall_ms, 1),
        "prompt_tokens": d["prompt_eval_count"],
        "prompt_eval_ms": round(d["prompt_eval_duration"] / 1e6, 1),
        "completion_tokens": d.get("eval_count"),
        "eval_ms": round(d.get("eval_duration", 0) / 1e6, 1),
    }


def system_message(tag: str, lines: int) -> str:
    filler = "".join(f"# filler line {i} about python semantics\n" for i in range(lines))
    return f"You are a helpful coding assistant. Session {tag}\n{filler}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--lines", type=int, default=400, help="filler lines in the system message")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="cold prefix then warm prefix only: the demo moment, ~40 s instead of ~3 min")
    ap.add_argument("--warmup", action="store_true",
                    help="load the model first so the demo's first timing is prefill, not a 7 GB read")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # A unique tag per invocation, so "cold" is genuinely cold every time this is run. Without it
    # the second run of the day measures the FIRST run's cache and reports cold == warm, which on
    # a demo looks exactly like the lever not working.
    run_tag = uuid.uuid4().hex[:8]
    a = system_message(f"AAA-{run_tag}", args.lines)
    b = system_message(f"BBB-{run_tag}", args.lines)

    if args.warmup or args.quick:
        # Load the weights on a throwaway prompt. Without this the first timing includes a ~7 GB
        # read from disk, which is not the thing being measured and makes the cold number a lie.
        print("warming up the model (not measured)...", flush=True)
        chat(args.model, [{"role": "user", "content": "hi"}], num_predict=1)

    # (label, system message, what the label means)
    plan_quick = [
        ("a_cold", a, "first sight of prefix A"),
        ("a_warm", a, "same prefix A, different last user message"),
    ]
    plan = [
        ("a_cold", a, "first sight of prefix A"),
        ("a_warm", a, "same prefix A, different last user message"),
        ("b_cold", b, "a different prefix arrives on the same server slot"),
        ("a_after_b", a, "back to A: was A's prefix evicted by B?"),
        ("a_again", a, "A immediately after A"),
        ("a_ts_churn_1", f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + a,
         "a timestamp at the FRONT of the system message"),
        ("a_ts_churn_2", f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +1] " + a, "same, one tick later"),
    ]

    rows = []
    for i, (label, sysmsg, note) in enumerate(plan_quick if args.quick else plan):
        r = chat(args.model, [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": f"Say OK {i}"},
        ])
        r.update(label=label, note=note, seq=i, model=args.model)
        r["prompt_eval_ms_per_token"] = round(r["prompt_eval_ms"] / max(r["prompt_tokens"], 1), 3)
        rows.append(r)
        print(f"{label:<14} wall={r['wall_ms']:>9.1f}ms  p_tok={r['prompt_tokens']:>6}  "
              f"p_eval={r['prompt_eval_ms']:>9.1f}ms  {r['prompt_eval_ms_per_token']:>6.3f} ms/tok")

    by = {r["label"]: r for r in rows}
    cold = by["a_cold"]["prompt_eval_ms_per_token"]
    warm = by["a_warm"]["prompt_eval_ms_per_token"]
    if args.quick:
        c, w = by["a_cold"], by["a_warm"]
        print(f"\n  same {c['prompt_tokens']:,}-token prompt, one cold and one warm:")
        print(f"    cold  {c['prompt_eval_ms']:>9,.0f} ms")
        print(f"    warm  {w['prompt_eval_ms']:>9,.0f} ms")
        print(f"    => {cold / warm:.0f}x on prompt evaluation\n")
    summary = {
        "model": args.model,
        "run_tag": run_tag,
        "prompt_tokens": by["a_cold"]["prompt_tokens"],
        "cold_ms_per_prompt_token": cold,
        "warm_ms_per_prompt_token": warm,
        "cache_speedup_on_prompt_eval": round(cold / warm, 1) if warm else None,
        "evicted_by_one_interleaved_request":
            by["a_after_b"]["prompt_eval_ms"] > 0.5 * by["a_cold"]["prompt_eval_ms"]
            if "a_after_b" in by else None,
        "front_churn_costs_full_reprefill":
            by["a_ts_churn_1"]["prompt_eval_ms"] > 0.5 * by["a_cold"]["prompt_eval_ms"]
            if "a_ts_churn_1" in by else None,
        "quick_mode": args.quick,
        "cached_tokens_reported_by_server": False,
        "note": "cached_tokens is absent from Ollama /v1 usage; prompt_eval_count does not "
                "shrink on a cache hit. prompt_eval_duration is the only cache instrument here.",
    }
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out}/rows.jsonl and {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
