"""Measure what the dedup lever costs and saves, with the agent taken out of the loop.

    python -m bench_agent.replay_bench runs/<a trace dir> --task ClassEval_10 \
        --out runs/<ts>_replay/

**Why this exists.** In the agent A/B the two arms stop being comparable after the first deduped
message: at temperature 0 the agent is deterministic given its prompt, so any change to the prompt
forks the trajectory, and from there the arms are running different agents that take different
numbers of turns and give up at different points. The wall-clock difference that falls out of that
is mostly "did it happen to quit early", which the lever perturbs but does not control.

Here there is no agent. A conversation recorded in a trace is replayed request by request, once
with dedup off and once with it on, and the only difference between the two runs is the lever. The
prompts are the ones the agent really sent, so this is not a synthetic benchmark; it is the same
traffic with one variable changed.

**Two things this is careful about.**

*Cache coherence.* Each arm's requests are replayed contiguously, in order, because that is what
the server's single KV slot sees in real use. Interleaving the arms per request would evict both
prefixes on every switch and make both arms look uniformly cold — the cache probe measures that
effect at 27.3 s on a 4.3k prompt — which would hide exactly what is being measured.

*Drift.* Replaying arm A then arm B gives B whatever the machine's state has drifted to. So the
whole pair is replayed twice, in both orders, and both are reported. If the two orders disagree
the run is telling you the machine moved, not that the lever did something.

`--mode prefill` caps generation at one token, so what is measured is prompt processing alone.
`--mode full` asks for as many tokens as the recorded response had, so the decode-side attention
term is in the measurement too. They answer different questions and both are reported.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import urllib.request

from bench_agent.trace_proxy import Tracer

UPSTREAM = "http://127.0.0.1:11434/v1/chat/completions"


def load_conversation(trace_dir: pathlib.Path, task: str | None) -> list[dict]:
    """Rebuild the real request bodies from a trace directory."""
    bodies = {p.stem: p.read_text(errors="replace") for p in (trace_dir / "bodies").glob("*.txt")}
    rows = [json.loads(l) for l in (trace_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    # One trace directory holds every task the arm ran; a session id identifies a conversation.
    by_session: dict[str, list[dict]] = {}
    for r in rows:
        by_session.setdefault(r["session"], []).append(r)
    sessions = sorted(by_session.values(), key=len, reverse=True)
    if task:
        for s in sessions:
            first = bodies.get(s[0]["messages"][-1]["sha"], "") if s[0]["messages"] else ""
            if task in first:
                sessions = [s]
                break
    chosen = sessions[0]
    out = []
    for r in sorted(chosen, key=lambda x: x["seq"]):
        msgs = [{"role": m["role"], "content": bodies.get(m["sha"], "")} for m in r["messages"]]
        if any(not m["content"] for m in msgs):
            continue  # a body we do not have; skip rather than send a truncated prompt
        out.append({"model": r["model"], "messages": msgs,
                    "completion_tokens": (r.get("usage") or {}).get("completion_tokens") or 32})
    return out


def send(model: str, messages: list[dict], max_tokens: int) -> tuple[float, dict]:
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(UPSTREAM, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return (time.time() - t0) * 1000, (d.get("usage") or {})


def replay(conv: list[dict], dedup: bool, mode: str, tmp: pathlib.Path) -> dict:
    tracer = Tracer(tmp, cold_ms=5.96, dedup=dedup, dedup_min_bytes=200)
    rows = []
    for i, req in enumerate(conv):
        msgs, replaced, saved = tracer.apply_dedup("replay", req["messages"])
        max_tokens = 1 if mode == "prefill" else max(req["completion_tokens"], 1)
        wall, usage = send(req["model"], msgs, max_tokens)
        rows.append({"i": i, "wall_ms": round(wall, 1),
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "dedup_replaced": replaced, "dedup_bytes_saved": saved})
    return {
        "dedup": dedup, "mode": mode, "requests": len(rows),
        "total_wall_s": round(sum(r["wall_ms"] for r in rows) / 1000, 2),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
        "median_wall_ms": round(statistics.median(r["wall_ms"] for r in rows), 1),
        "messages_replaced": sum(r["dedup_replaced"] for r in rows),
        "bytes_removed": sum(r["dedup_bytes_saved"] for r in rows),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--task", default="")
    ap.add_argument("--mode", default="prefill", choices=["prefill", "full"])
    ap.add_argument("--max-requests", type=int, default=0, help="0 = the whole conversation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conv = load_conversation(pathlib.Path(args.trace_dir), args.task or None)
    if args.max_requests:
        conv = conv[: args.max_requests]
    if not conv:
        print("no replayable requests found")
        return 1
    print(f"replaying {len(conv)} requests, mode={args.mode}", flush=True)

    tmp = out / "_scratch"
    tmp.mkdir(exist_ok=True)
    results = {}
    # Both orders, so a machine that drifts says so instead of being mistaken for the lever.
    for order in (("off", "on"), ("on", "off")):
        for arm in order:
            key = f"{'_'.join(order)}::{arm}"
            r = replay(conv, dedup=(arm == "on"), mode=args.mode, tmp=tmp)
            results[key] = r
            print(f"  [{'/'.join(order)}] {arm:>3}: {r['total_wall_s']:>7.2f} s  "
                  f"prompt={r['prompt_tokens']:>8,}  replaced={r['messages_replaced']}", flush=True)

    summary = {"trace_dir": args.trace_dir, "task": args.task, "mode": args.mode,
               "requests": len(conv), "arms": results}
    for order in ("off_on", "on_off"):
        off, on = results[f"{order}::off"], results[f"{order}::on"]
        summary[f"speedup_{order}"] = round(off["total_wall_s"] / max(on["total_wall_s"], 1e-9), 3)
        summary[f"prompt_token_reduction_{order}"] = round(
            1 - on["prompt_tokens"] / max(off["prompt_tokens"], 1), 4)
    a, b = summary["speedup_off_on"], summary["speedup_on_off"]
    summary["orders_agree"] = abs(a - b) < 0.15 * max(a, b)
    summary["verdict"] = (
        f"dedup speedup {min(a,b):.2f}-{max(a,b):.2f}x on {args.mode}"
        if summary["orders_agree"] else
        f"ORDERS DISAGREE ({a:.2f} vs {b:.2f}): the machine drifted during the replay; "
        f"do not quote a speedup from this run")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "arms"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
