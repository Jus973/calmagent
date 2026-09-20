"""`calm-proxy stats <trace-dir>` — the pooled table, in seconds.

`lint` answers "which session is losing its prefix and why". `stats` answers
"what did the proxy do, and what did it save", pooled over the whole trace, so
the demo can print it live. Wall clock is the unit wherever it is available:
LC's 02:41 measurement says a lost prefix is 5.96 ms per prompt token on this
class of machine, and nobody feels a token.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Iterable, Sequence

from .lint import load_records
from .middleware.prefix import APPEND_ONLY, NO_PREVIOUS

DEFAULT_COLD_MS_PER_TOKEN = 5.96


def _percentile(values: Sequence[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))
    return ordered[index]


def _human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def collect(records: Iterable[dict[str, Any]], cold_ms_per_token: float) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "requests": 0,
        "sessions": set(),
        "errors": 0,
        "streamed": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "cached_reported": False,
        "completion_tokens": 0,
        "achievable_tokens": 0,
        "memo_eligible": 0,
        "memo_hits": 0,
        "memo_tokens_replayed": 0,
        "dedup_replaced": 0,
        "dedup_bytes_saved": 0,
        "divergences": Counter(),
        "ttft_ms": [],
        "done_ms": [],
        "prompt_eval_ms": 0.0,
        "eval_ms": 0.0,
        "native_requests": 0,
    }
    for record in records:
        stats["requests"] += 1
        stats["sessions"].add(record.get("session"))
        if record.get("error"):
            stats["errors"] += 1
        if record.get("stream"):
            stats["streamed"] += 1

        usage = record.get("usage") or {}
        prompt = usage.get("prompt_tokens") or 0
        stats["prompt_tokens"] += prompt
        cached = usage.get("cached_tokens")
        if cached is not None:
            stats["cached_reported"] = True
            stats["cached_tokens"] += cached
        stats["completion_tokens"] += usage.get("completion_tokens") or 0

        prefix = record.get("prefix") or {}
        estimate = prefix.get("achievable_cached_tokens_est") or 0
        stats["achievable_tokens"] += min(estimate, prompt) if prompt else estimate
        divergence = prefix.get("divergence")
        if divergence and divergence not in {NO_PREVIOUS, APPEND_ONLY}:
            stats["divergences"][divergence] += 1

        memo = record.get("memo") or {}
        if memo.get("key"):
            stats["memo_eligible"] += 1
        if memo.get("hit"):
            stats["memo_hits"] += 1
            stats["memo_tokens_replayed"] += usage.get("completion_tokens") or 0

        dedup = record.get("dedup") or {}
        stats["dedup_replaced"] += dedup.get("replaced") or 0
        stats["dedup_bytes_saved"] += dedup.get("bytes_saved") or 0

        timing = record.get("timing_ms") or {}
        if timing.get("first_token") is not None:
            stats["ttft_ms"].append(float(timing["first_token"]))
        if timing.get("done") is not None:
            stats["done_ms"].append(float(timing["done"]))

        upstream = record.get("upstream") or {}
        if upstream.get("prompt_eval_ms") is not None:
            stats["native_requests"] += 1
            stats["prompt_eval_ms"] += float(upstream["prompt_eval_ms"])
            stats["eval_ms"] += float(upstream.get("eval_ms") or 0.0)

    gap = max(0, stats["achievable_tokens"] - stats["cached_tokens"])
    stats["prefix_gap_tokens"] = gap
    stats["prefix_gap_s"] = gap * cold_ms_per_token / 1000.0
    return stats


def report(records: list[dict[str, Any]], cold_ms_per_token: float = DEFAULT_COLD_MS_PER_TOKEN) -> str:
    s = collect(records, cold_ms_per_token)
    if not s["requests"]:
        return "no records in this trace"

    lines = [
        f"requests {s['requests']}   sessions {len(s['sessions'])}   "
        f"streamed {s['streamed']}   errors {s['errors']}",
    ]

    p50 = _percentile(s["ttft_ms"], 0.5)
    p95 = _percentile(s["ttft_ms"], 0.95)
    total_s = sum(s["done_ms"]) / 1000.0
    lines.append(
        f"wall clock       total {total_s:.1f} s   "
        f"ttft p50 {p50:.0f} ms   p95 {p95:.0f} ms" if p50 is not None
        else f"wall clock       total {total_s:.1f} s"
    )

    cached = f"{s['cached_tokens']:,}" if s["cached_reported"] else "n/a"
    share = (
        f" ({100 * s['cached_tokens'] / s['prompt_tokens']:.1f}%)"
        if s["cached_reported"] and s["prompt_tokens"]
        else ""
    )
    lines.append(
        f"prompt tokens    {s['prompt_tokens']:,} sent   cached {cached}{share}   "
        f"completion {s['completion_tokens']:,}"
    )

    if s["divergences"]:
        reasons = "  ".join(f"{name} {count}" for name, count in s["divergences"].most_common())
        lines.append(f"prefix lost to   {reasons}")
    lines.append(
        f"prefix gap       {s['prefix_gap_tokens']:,} tok "
        f"\u2248 {s['prefix_gap_s']:.1f} s of cold prefill "
        f"at {cold_ms_per_token} ms/tok (lower bound, not a measured saving)"
    )

    hit_rate = 100 * s["memo_hits"] / s["memo_eligible"] if s["memo_eligible"] else 0.0
    lines.append(
        f"memo             {s['memo_hits']}/{s['memo_eligible']} eligible requests replayed "
        f"({hit_rate:.1f}%)   {s['memo_tokens_replayed']:,} completion tok not generated"
    )
    lines.append(
        f"dedup            {s['dedup_replaced']} tool results referenced   "
        f"{_human_bytes(s['dedup_bytes_saved'])} not re-sent"
    )
    if s["native_requests"]:
        lines.append(
            f"upstream timing  prompt_eval {s['prompt_eval_ms'] / 1000:.1f} s   "
            f"eval {s['eval_ms'] / 1000:.1f} s   over {s['native_requests']} native requests"
        )
    else:
        lines.append(
            "upstream timing  n/a on the /v1 surface; point the agent at /api/chat "
            "for prompt_eval_ms"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calm-proxy stats")
    parser.add_argument("trace_dir", help="run directory, or a trace.jsonl path")
    parser.add_argument(
        "--cold-ms-per-token",
        type=float,
        default=DEFAULT_COLD_MS_PER_TOKEN,
        help="cold prefill cost, calibrated per machine (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    print(report(load_records(args.trace_dir), args.cold_ms_per_token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
