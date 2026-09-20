"""Score every candidate lever's headroom on a recorded probe trace, before any of them is built.

    python -m bench_agent.probe_headroom runs/<ts>_probe10_off [more dirs...]

The overnight contract's decision rule is "the headline lever is the one with the largest measured
headroom on the probe trace". This computes each lever's headroom from `trace.jsonl` + `bodies/`
and nothing else, so the decision is made on this agent's real behaviour rather than on a guess
about it.

Headroom definitions, each deliberately an *upper bound* on what the lever could win:

I-2 prefix-lint   tokens the agent threw away by breaking its own prefix -- NOT simply the tokens
                  that are new this turn. An agent that only appends has nothing to lint however
                  much it appends, so `churn_tokens` counts only requests whose divergence has a
                  cause other than `appended_only`, and charges everything from the divergence
                  point to the end of the prompt at the cold rate. The separate `new_tokens` line
                  is content that never existed before and that no cache could have held.
I-3 memo          requests whose (model, messages, params) key is byte-identical to an earlier one.
                  Saves the whole request: prefill and decode.
I-4 dedup         bytes in repeated messages that sit AFTER the divergence point, which are the only
                  ones a rewrite could shorten without invalidating cached prefix behind it. The
                  repeats inside the shared prefix are counted separately and explicitly NOT
                  charged: they are already cached, and rewriting one would cost a full re-prefill.
I-5 calm-run      shell commands the agent re-issued verbatim. Counted from the assistant turns in
                  the prompt bodies, split into test commands and everything else, because only the
                  test commands have a cached-execution story behind them (`results/cache.md`).
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
import sys

COLD_MS_PER_PROMPT_TOKEN = 5.96   # bench_agent/probe_cache.py on this machine
DECODE_MS_PER_TOKEN = 50.0        # ~20 tok/s single stream, qwen2.5-coder:7b on the M4
BASH_RE = re.compile(r"```(?:mswea_bash_command|bash)\n(.*?)```", re.S)
TEST_RE = re.compile(r"\b(pytest|unittest|python -m pytest|nosetests)\b")


def load(dirs: list[pathlib.Path]) -> tuple[list[dict], dict[str, str]]:
    rows, bodies = [], {}
    for d in dirs:
        t = d / "trace.jsonl"
        if not t.exists():
            continue
        for line in t.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_dir"] = d.name
                rows.append(r)
        for b in (d / "bodies").glob("*.txt"):
            bodies[b.stem] = b.read_text(errors="replace")
    return rows, bodies


def main() -> int:
    dirs = [pathlib.Path(a) for a in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return 2
    rows, bodies = load(dirs)
    if not rows:
        print("no trace rows found")
        return 1

    n = len(rows)
    prompt_tokens = sum((r["usage"] or {}).get("prompt_tokens") or 0 for r in rows)
    completion_tokens = sum((r["usage"] or {}).get("completion_tokens") or 0 for r in rows)
    wall_ms = sum((r["timing_ms"] or {}).get("done") or 0 for r in rows)
    sessions = {r["session"] for r in rows}

    # ---- I-2 -------------------------------------------------------------------------------
    achievable = sum(r["prefix"]["shared_tokens"] for r in rows)
    # char/4 units; rescale to the server's own token count so the two are comparable
    approx_total = sum(r["prefix"]["total_tokens"] for r in rows) or 1
    scale = prompt_tokens / approx_total
    achievable_tok = achievable * scale
    new_tok = prompt_tokens - achievable_tok
    # Churn is the part the lint could actually recover: prompt behind a divergence that the agent
    # caused itself. Appending is not churn, no matter how much of it there is.
    churn_tok = sum(((r["usage"] or {}).get("prompt_tokens") or 0)
                    - r["prefix"]["shared_tokens"] * scale
                    for r in rows
                    if r["prefix"]["cause"] not in ("appended_only", "first_request_in_session"))
    i2_ms = max(churn_tok, 0) * COLD_MS_PER_PROMPT_TOKEN
    causes = collections.Counter(r["prefix"]["cause"] for r in rows)

    # ---- I-3 -------------------------------------------------------------------------------
    seen: set[str] = set()
    memo_hits = [r for r in rows if (r["prompt_sha"] in seen) or seen.add(r["prompt_sha"])]
    memo_hits = [r for r in memo_hits if r is not None]
    counts = collections.Counter(r["prompt_sha"] for r in rows)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    i3_ms = sum(((r["usage"] or {}).get("prompt_tokens") or 0) * COLD_MS_PER_PROMPT_TOKEN
                + ((r["usage"] or {}).get("completion_tokens") or 0) * DECODE_MS_PER_TOKEN
                for r in rows if counts[r["prompt_sha"]] > 1) * (repeats / max(sum(
                    c for c in counts.values() if c > 1), 1))

    # ---- I-4 -------------------------------------------------------------------------------
    dup_after_divergence = dup_inside_prefix = dup_tool_bytes = 0
    for sess in sessions:
        first: set[str] = set()
        for r in [x for x in rows if x["session"] == sess]:
            div = r["prefix"]["first_divergent_message"]
            for i, m in enumerate(r["messages"]):
                key = m["sha"]
                if key in first:
                    if i >= div:
                        dup_after_divergence += m["bytes"]
                        if m["role"] == "tool":
                            dup_tool_bytes += m["bytes"]
                    else:
                        dup_inside_prefix += m["bytes"]
                else:
                    first.add(key)
    i4_ms = (dup_after_divergence / 4) * COLD_MS_PER_PROMPT_TOKEN

    # ---- I-5 -------------------------------------------------------------------------------
    cmds: list[str] = []
    for text in bodies.values():
        for m in BASH_RE.finditer(text):
            cmds.append(m.group(1).strip())
    cmd_counts = collections.Counter(cmds)
    repeated_cmds = {c: k for c, k in cmd_counts.items() if k > 1}
    test_cmds = {c: k for c, k in cmd_counts.items() if TEST_RE.search(c)}
    repeated_test = {c: k for c, k in test_cmds.items() if k > 1}
    reruns = sum(k - 1 for k in repeated_test.values())

    out = {
        "trace_dirs": [d.name for d in dirs],
        "requests": n, "sessions": len(sessions),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "upstream_wall_s": round(wall_ms / 1000, 1),
        "constants": {"cold_ms_per_prompt_token": COLD_MS_PER_PROMPT_TOKEN,
                      "decode_ms_per_token": DECODE_MS_PER_TOKEN},
        "I2_prefix_lint": {
            "achievable_prefix_share": round(achievable_tok / max(prompt_tokens, 1), 4),
            "new_tokens_no_cache_could_hold": round(new_tok),
            "churn_tokens": round(max(churn_tok, 0)),
            "headroom_s": round(i2_ms / 1000, 1),
            "divergence_causes": dict(causes),
        },
        "I3_memo": {
            "duplicate_requests": repeats,
            "duplicate_share": round(repeats / max(n, 1), 4),
            "headroom_s": round(i3_ms / 1000, 1),
        },
        "I4_dedup": {
            "duplicate_bytes_after_divergence": dup_after_divergence,
            "duplicate_bytes_inside_cached_prefix_not_charged": dup_inside_prefix,
            "duplicate_tool_result_bytes": dup_tool_bytes,
            "headroom_s": round(i4_ms / 1000, 1),
        },
        "I5_calm_run": {
            "shell_commands_seen": len(cmds),
            "distinct": len(cmd_counts),
            "repeated_commands": sum(k - 1 for k in repeated_cmds.values()),
            "test_commands": sum(test_cmds.values()),
            "test_command_reruns": reruns,
            "top_repeated_test_commands": sorted(repeated_test.items(), key=lambda kv: -kv[1])[:5],
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
