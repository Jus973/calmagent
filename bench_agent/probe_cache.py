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
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a = system_message("AAA", args.lines)
    b = system_message("BBB", args.lines)

    # (label, system message, what the label means)
    plan = [
        ("a_cold", a, "first sight of prefix A"),
        ("a_warm", a, "same prefix A, different last user message"),
        ("b_cold", b, "a different prefix arrives on the same server slot"),
        ("a_after_b", a, "back to A: was A's prefix evicted by B?"),
        ("a_again", a, "A immediately after A"),
        ("a_ts_churn_1", "[2026-09-20 02:38:11] " + a, "a timestamp at the FRONT of the system message"),
        ("a_ts_churn_2", "[2026-09-20 02:38:47] " + a, "same, one tick later"),
    ]

    rows = []
    for i, (label, sysmsg, note) in enumerate(plan):
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
    summary = {
        "model": args.model,
        "prompt_tokens": by["a_cold"]["prompt_tokens"],
        "cold_ms_per_prompt_token": cold,
        "warm_ms_per_prompt_token": warm,
        "cache_speedup_on_prompt_eval": round(cold / warm, 1) if warm else None,
        "evicted_by_one_interleaved_request": by["a_after_b"]["prompt_eval_ms"] > 0.5 * by["a_cold"]["prompt_eval_ms"],
        "front_churn_costs_full_reprefill": by["a_ts_churn_1"]["prompt_eval_ms"] > 0.5 * by["a_cold"]["prompt_eval_ms"],
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
