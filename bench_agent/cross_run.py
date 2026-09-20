"""How much of a run's prompt could have been served from the *previous* run's cache.

    python -m bench_agent.cross_run runs/<dir_a> runs/<dir_b> [--out ...]

`probe_headroom.py` asks whether an agent breaks its own prefix *within* a session, and for
mini-swe-agent the answer is a clean no: it only ever appends. That is the whole within-session
story and it is why the lint had no headroom there.

This asks the other question. A developer runs their agent on the same repo again an hour later.
The system prompt is identical, the task is identical, the first command is identical -- so a KV
cache that survived should serve most of the second run's early turns for free. Does it?

For mini-swe-agent on this bench: no. Every conversation diverges at message 3, and the median
prompt is only 13.7% reusable from the run before.

**Read the cause carefully, because this bench is partly responsible for it.** The agent's opening
move is `ls -la`, and exactly one line of that output differs between runs: the `..` entry's mtime.
That is the *parent temp directory*, which `run_agent.py` creates fresh for every task so runs
cannot contaminate each other. So the trigger here is the harness, not the agent and not a real
repository, where `..` would sit still.

What the number is therefore evidence *for*: the mechanism, exactly and quantitatively. One line of
volatile text, forty tokens in, makes 86% of every later turn uncacheable, because a prefix cache
is a prefix cache -- nothing after the first differing byte survives. And because the model is
deterministic given its prompt, it also makes the two runs stop being the same experiment, which
is how this was found at all.

What it is *not*: an estimate of what a real user loses. A real repo would not churn `..`. It would
churn something else -- an agent that edits files changes their mtimes, `git status` prints
branch state, a test runner prints durations -- and the wall would arrive later and cost less.
Claiming this 396 s for a real workload would be dishonest; the right claim is that the proxy finds
the line, and that finding it is worth something because of how much sits behind it.

The output is the number of prompt tokens that were *available* to reuse and were not, priced at
the machine's measured cold rate, plus exactly where and why the prefix broke.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from bench_agent.replay_bench import split_conversations

COLD_MS_PER_PROMPT_TOKEN = 5.96


def conversations(d: pathlib.Path) -> tuple[list[list[dict]], dict[str, str]]:
    bodies = {p.stem: p.read_text(errors="replace") for p in (d / "bodies").glob("*.txt")}
    rows = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines() if l.strip()]
    return sorted(split_conversations(rows), key=lambda c: c[0]["ts"]), bodies


def first_divergence(a: list[dict], b: list[dict], ba: dict, bb: dict) -> tuple[int, int, str, str]:
    """Index of the first differing message, the byte offset inside it, and both sides' text."""
    for i, (x, y) in enumerate(zip(a, b)):
        tx, ty = ba.get(x["sha"], ""), bb.get(y["sha"], "")
        if x["role"] != y["role"] or tx != ty:
            n = 0
            for n in range(min(len(tx), len(ty))):
                if tx[n] != ty[n]:
                    break
            else:
                n = min(len(tx), len(ty))
            return i, n, tx, ty
    return min(len(a), len(b)), 0, "", ""


def classify(role: str, a: str, b: str, at: int) -> str:
    window_a, window_b = a[max(0, at - 40):at + 40], b[max(0, at - 40):at + 40]
    import re
    time_re = re.compile(r"\d{1,2}:\d{2}|\b\d{4}-\d{2}-\d{2}\b|\b[A-Z][a-z]{2} +\d{1,2}\b")
    if time_re.search(window_a) and time_re.search(window_b):
        return "timestamp in tool output"
    if "/tmp" in window_a or "/var/folders" in window_a or "/private" in window_a:
        return "temporary path in tool output"
    if role == "system":
        return "system prompt differs"
    return f"{role} content differs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if len(args.dirs) < 2:
        print("need at least two run directories to compare across runs")
        return 2

    loaded = [(pathlib.Path(d), *conversations(pathlib.Path(d))) for d in args.dirs]
    base_dir, base_convs, base_bodies = loaded[0]

    findings = []
    for other_dir, other_convs, other_bodies in loaded[1:]:
        for i, (a, b) in enumerate(zip(base_convs, other_convs)):
            # compare the deepest request of each run: that is the whole transcript
            ra, rb = a[-1], b[-1]
            idx, byte, ta, tb = first_divergence(ra["messages"], rb["messages"],
                                                 base_bodies, other_bodies)
            shared_bytes = sum(m["bytes"] for m in ra["messages"][:idx]) + byte
            total_bytes = sum(m["bytes"] for m in ra["messages"])
            pt = (ra.get("usage") or {}).get("prompt_tokens") or 0
            share = shared_bytes / max(total_bytes, 1)
            reusable = pt * share
            wasted = pt - reusable
            role = ra["messages"][idx]["role"] if idx < len(ra["messages"]) else "?"
            findings.append({
                "conversation": i,
                "runs": [base_dir.name, other_dir.name],
                "messages_in_deepest_prompt": len(ra["messages"]),
                "first_divergent_message": idx,
                "first_divergent_byte": byte,
                "cause": classify(role, ta, tb, byte) if idx < len(ra["messages"]) else "identical",
                "reusable_prefix_share": round(share, 4),
                "prompt_tokens": pt,
                "tokens_reusable_across_runs": round(reusable),
                "tokens_wasted_across_runs": round(wasted),
                "cost_of_waste_s": round(wasted * COLD_MS_PER_PROMPT_TOKEN / 1000, 1),
                "sample_a": ta[max(0, byte - 60):byte + 60].replace("\n", " | "),
                "sample_b": tb[max(0, byte - 60):byte + 60].replace("\n", " | "),
            })

    print(f"{'conv':>4} {'msgs':>5} {'diverge@':>9} {'reusable':>9} {'wasted tok':>11} {'cost':>8}  cause")
    for f in findings:
        print(f"{f['conversation']:>4} {f['messages_in_deepest_prompt']:>5} "
              f"{f['first_divergent_message']:>9} {f['reusable_prefix_share']:>8.1%} "
              f"{f['tokens_wasted_across_runs']:>11,} {f['cost_of_waste_s']:>7.1f}s  {f['cause']}")
    if findings:
        print("\nfirst divergence, as the two runs wrote it:")
        f = findings[0]
        print(f"  run A: ...{f['sample_a']}...")
        print(f"  run B: ...{f['sample_b']}...")

    total_wasted = sum(f["tokens_wasted_across_runs"] for f in findings)
    summary = {
        "comparisons": len(findings),
        "median_reusable_prefix_share": round(
            sorted(f["reusable_prefix_share"] for f in findings)[len(findings) // 2], 4)
        if findings else None,
        "total_tokens_wasted_across_runs": total_wasted,
        "total_cost_s": round(total_wasted * COLD_MS_PER_PROMPT_TOKEN / 1000, 1),
        "causes": {c: sum(1 for f in findings if f["cause"] == c)
                   for c in {f["cause"] for f in findings}},
        "cold_ms_per_prompt_token": COLD_MS_PER_PROMPT_TOKEN,
        "note": "Upper bound twice over, and neither bound is this bench's real-world cost. It "
                "assumes the server still holds the earlier run's prefix, which with one KV slot "
                "it usually will not; and on this bench the diverging line is the parent temp "
                "directory's mtime, which run_agent.py creates fresh per task, so the harness is "
                "the trigger rather than the agent or a real repo. What the figure does show "
                "exactly is the mechanism: one volatile line 40 tokens in costs everything "
                "behind it.",
        "caveat": "trigger is harness-created; see module docstring",
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(
            {"summary": summary, "findings": findings}, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
