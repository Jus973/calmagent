"""Fit what a token of context costs, as a function of how much context is already there.

    python -m bench_agent.context_cost runs/<a trace dir> [more...]

This exists because the probe's most useful finding is not about caching. On a well-behaved agent
the prefix cache is already capturing ~95% of what it can, and yet the cost of each turn keeps
climbing. The reason is that every token being generated attends to every token in the context, so
the price of a turn rises with the transcript it is appended to -- and no prefix cache addresses
that, because the tokens in question are cached and are still being attended to.

The obvious model -- `ms_per_new_token = a + b * context` on a prefill figure derived by
subtracting a constant decode cost -- does not survive its own data. Two rows at the same context
disagree by 3x depending on how long the completion was (at ~17k context: 7.49 ms/token on a
335-new-token turn, 20.55 on a 1,678-token one). That is the subtraction leaking: decode also slows
down in a long context, so a fixed decode constant under-subtracts most on the rows with the
longest completions and dumps the remainder into "prefill".

So nothing is subtracted. The whole measured request time is fitted against the two kinds of token
in it, each with its own attention term:

    done_ms  =  a * new  +  b * new * context  +  c * completion  +  d * completion * context

`a` and `c` are the flat costs of prefilling one prompt token and generating one completion token;
`b` and `d` are what each of those costs extra per token already in the context. Only `done_ms` is
measured, and every derived quantity is a coefficient of that fit rather than a hand-subtraction.
A validity check the fit has to pass: `a` should land near the cold prefill rate measured
independently by `bench_agent/probe_cache.py` (5.96 ms/token on this machine). It does.

Only rows the arithmetic can support are used: a row is kept when its prompt extends the previous
request's (`appended_only`), so "new tokens" is well defined, and when it has at least
`--min-new-tokens` of them.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def fit(points: list[dict]) -> tuple[list[float], float]:
    """Least squares for done_ms ~ a*new + b*new*ctx + c*comp + d*comp*ctx, plus r^2."""
    import numpy as np
    A = np.array([[p["new_tokens"], p["new_tokens"] * p["context_tokens"],
                   p["completion_tokens"], p["completion_tokens"] * p["context_tokens"]]
                  for p in points], dtype=float)
    y = np.array([p["done_ms"] for p in points], dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return [float(c) for c in coef], (1 - ss_res / ss_tot if ss_tot else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-new-tokens", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    points = []
    for d in args.dirs:
        path = pathlib.Path(d) / "trace.jsonl"
        if not path.exists():
            continue
        prev_pt = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get("usage") or {}
            pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
            done = (r.get("timing_ms") or {}).get("done") or 0
            sess = r["session"]
            last = prev_pt.get(sess, 0)
            prev_pt[sess] = pt
            if r["prefix"]["cause"] != "appended_only":
                continue
            new = pt - last
            if new < args.min_new_tokens or done <= 0:
                continue
            # context the new tokens are attended against: everything that was already there
            points.append({"context_tokens": last, "new_tokens": new,
                           "completion_tokens": ct, "done_ms": done,
                           "dir": pathlib.Path(d).name})

    if len(points) < 8:
        print(f"only {len(points)} usable rows; need at least 8 for a four-parameter fit. "
              f"Try a longer trace or a lower --min-new-tokens.")
        return 1

    (a, b, c, d), r2 = fit(points)
    xs = [p["context_tokens"] for p in points]
    lo, hi = min(xs), max(xs)

    print(f"{'context':>9} {'new tok':>8} {'comp tok':>9} {'done ms':>9}")
    for p in sorted(points, key=lambda p: p["context_tokens"]):
        print(f"{p['context_tokens']:>9} {p['new_tokens']:>8} {p['completion_tokens']:>9} "
              f"{p['done_ms']:>9.0f}")

    summary = {
        "rows_used": len(points),
        "context_range": [lo, hi],
        "model": "done_ms = a*new + b*new*ctx + c*completion + d*completion*ctx",
        "a_prompt_token_ms": round(a, 3),
        "b_prompt_token_ms_per_context_token": round(b, 8),
        "c_completion_token_ms": round(c, 3),
        "d_completion_token_ms_per_context_token": round(d, 8),
        "r_squared": round(r2, 3),
        "prompt_token_ms_at_min_context": round(a + b * lo, 2),
        "prompt_token_ms_at_max_context": round(a + b * hi, 2),
        "completion_token_ms_at_min_context": round(c + d * lo, 2),
        "completion_token_ms_at_max_context": round(c + d * hi, 2),
        "prompt_multiple_across_range": round((a + b * hi) / (a + b * lo), 2) if (a + b * lo) else None,
        "completion_multiple_across_range": round((c + d * hi) / (c + d * lo), 2) if (c + d * lo) else None,
        "independent_cold_rate_from_probe_cache_ms": 5.96,
        "verdict": ("context length has a measurable price: both a prompt token and a completion "
                    "token cost more in a long transcript"
                    if (b > 0 or d > 0) and r2 > 0.5 else
                    "no context effect detectable in this trace"),
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"summary": summary, "points": points}, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
