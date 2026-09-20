"""Run the replay bench over every recorded conversation, not just the convenient one.

    python -m bench_agent.replay_sweep runs/<trace dir> --out runs/<ts>_replay_sweep/

`replay_bench.py` answers "what did the lever do to this conversation". Quoting it on one
conversation invites the obvious objection, and it is a fair one: the longest conversation is
where dedup has the most to remove, so picking it picks the answer. This runs all of them and
reports the distribution, including the ones where the lever does nothing.

One order per conversation, not two. Order-independence was established separately on two
conversations (both agreed to within 4%: `runs/*_replay_prefill/`, `runs/*_replay_full/`), and
spending the second order again here would halve the number of conversations instead, which is the
axis that actually matters now. Arm order alternates between conversations so that whatever drift
remains does not accumulate on one arm.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

from bench_agent.replay_bench import load_conversation, replay, split_conversations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--mode", default="full", choices=["prefill", "full"])
    ap.add_argument("--max-conversations", type=int, default=0)
    ap.add_argument("--min-requests", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_scratch"
    scratch.mkdir(exist_ok=True)

    d = pathlib.Path(args.trace_dir)
    rows = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines() if l.strip()]
    n_convs = len(split_conversations(rows))
    if args.max_conversations:
        n_convs = min(n_convs, args.max_conversations)

    results = []
    t0 = time.time()
    for i in range(n_convs):
        conv = load_conversation(d, i)
        if len(conv) < args.min_requests:
            print(f"conversation {i}: {len(conv)} requests, skipping", flush=True)
            continue
        order = ("off", "on") if i % 2 == 0 else ("on", "off")
        arms = {}
        for arm in order:
            arms[arm] = replay(conv, dedup=(arm == "on"), mode=args.mode, tmp=scratch)
        off, on = arms["off"], arms["on"]
        r = {
            "conversation": i, "requests": len(conv), "arm_order": list(order),
            "off_wall_s": off["total_wall_s"], "on_wall_s": on["total_wall_s"],
            "off_prompt_tokens": off["prompt_tokens"], "on_prompt_tokens": on["prompt_tokens"],
            "messages_replaced": on["messages_replaced"],
            "speedup": round(off["total_wall_s"] / max(on["total_wall_s"], 1e-9), 3),
            "prompt_token_reduction": round(
                1 - on["prompt_tokens"] / max(off["prompt_tokens"], 1), 4),
            "deepest_prompt_tokens": max((x["prompt_tokens"] or 0) for x in off["rows"]),
        }
        results.append(r)
        print(f"conv {i:>2} ({len(conv):>2} reqs, order {'/'.join(order)}): "
              f"off {off['total_wall_s']:>7.1f}s  on {on['total_wall_s']:>7.1f}s  "
              f"speedup {r['speedup']:>5.2f}x  prompt -{r['prompt_token_reduction']:.1%}  "
              f"[{int(time.time()-t0)}s elapsed]", flush=True)

    if not results:
        print("no conversations long enough to replay")
        return 1

    speedups = [r["speedup"] for r in results]
    tot_off = sum(r["off_wall_s"] for r in results)
    tot_on = sum(r["on_wall_s"] for r in results)
    summary = {
        "trace_dir": args.trace_dir, "mode": args.mode,
        "conversations": len(results),
        "total_requests": sum(r["requests"] for r in results),
        "pooled_off_wall_s": round(tot_off, 1),
        "pooled_on_wall_s": round(tot_on, 1),
        "pooled_speedup": round(tot_off / max(tot_on, 1e-9), 3),
        "median_speedup": round(statistics.median(speedups), 3),
        "min_speedup": min(speedups), "max_speedup": max(speedups),
        "conversations_with_no_effect": sum(1 for s in speedups if s < 1.05),
        "pooled_prompt_token_reduction": round(
            1 - sum(r["on_prompt_tokens"] for r in results)
            / max(sum(r["off_prompt_tokens"] for r in results), 1), 4),
        "note": "One arm order per conversation, alternating. Order-independence was checked "
                "separately on two conversations and held to within 4%.",
    }
    (out / "summary.json").write_text(json.dumps(
        {"summary": summary, "conversations": results}, indent=2) + "\n")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
