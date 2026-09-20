"""How much of a real completion is decoded after the harness already has what it asked for.

`bench/latency.py` measures the stop-at-fill lever against canned completions, so its speedup is a
function of the tail those completions happen to carry. This measures the tail itself, on the
completions a real model actually produced: for every recorded per-slot emission, where
`truncation_point` would have cut it, and what fraction of the characters came after that.

Characters, not tokens: a request is cut mid-stream, so no usage frame reports the split. The
character fraction is the honest proxy and is named as one. Only per-slot emissions are counted —
a whole-class sample is asked for the whole class, so none of it is tail.

A completion that stops exactly at the end of its method has no cut point, because nothing after it
proves the method ended, so it is excluded. That drops the tightest completions from the
denominator, which can only push the measured tail up: the number here is an upper bound.

python -m analysis.tail_waste runs/<dir> [runs/<dir2> ...] [--out results/tail.md]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from calm_coder.bench.classeval import load_rows, task_from_row
from calm_coder.jsonl import read_jsonl
from calm_coder.serve.stream import truncation_point

SPINE = "__class__"          # a whole-class emission: asked for everything, so nothing is tail


def measure(run_dirs: list[Path]) -> dict:
    rows = {r["task_id"]: r for r in load_rows()}
    tasks: dict[str, object] = {}
    per: list[dict] = []
    reasons: Counter = Counter()
    for d in run_dirs:
        for e in read_jsonl(d / "emissions.jsonl"):
            slot, text = e.get("slot"), e.get("text") or ""
            if slot == SPINE or not text or e.get("task_id") not in rows:
                reasons["whole_class_or_empty"] += slot == SPINE or not text
                continue
            task = tasks.setdefault(e["task_id"], task_from_row(rows[e["task_id"]]))
            if slot not in {s.id for s in task.slots}:
                continue
            cut = truncation_point(text, slot, {s.id for s in task.slots}, task.class_name)
            if cut is None:
                reasons["no_cut_point"] += 1
                continue
            per.append({"task_id": e["task_id"], "arm": e.get("arm"), "slot": slot,
                        "chars": len(text), "kept": cut, "tail": len(text) - cut,
                        "tokens": e.get("completion_tokens") or 0})
    if not per:
        return {"emissions": 0, "skipped": dict(reasons)}
    fracs = [p["tail"] / p["chars"] for p in per]
    chars, tail = sum(p["chars"] for p in per), sum(p["tail"] for p in per)
    toks = sum(p["tokens"] for p in per)
    return {
        "emissions": len(per), "skipped": dict(reasons),
        "chars": chars, "tail_chars": tail, "tail_frac": tail / chars,
        "median_tail_frac": statistics.median(fracs),
        "p90_tail_frac": sorted(fracs)[int(0.9 * (len(fracs) - 1))],
        "cut_at_all_frac": sum(f > 0.01 for f in fracs) / len(fracs),
        "decode_tokens": toks, "projected_tokens_saved": int(toks * tail / chars),
    }


def render(m: dict, runs: list[str]) -> str:
    if not m["emissions"]:
        return "# Tail waste\n\nNo per-slot emissions with a cut point in those runs.\n"
    return "\n".join([
        "# How much of a real completion is tail", "",
        f"{m['emissions']} per-slot emissions from {', '.join(runs)}. A completion's *tail* is "
        "everything after the point the harness could have cut it and still stored the same fill "
        "(`serve/stream.truncation_point`). Measured in characters, since a cut request never "
        "reports a usage split; the token column applies the character rate and is a projection.",
        "",
        f"These are upper bounds: {m['skipped'].get('no_cut_point', 0)} completions stopped at the "
        "end of their method with nothing after to prove it, so they have no cut point and are not "
        "in the denominator — the tightest completions are the ones excluded.", "",
        "| | value |",
        "| --- | --- |",
        f"| completions with a tail worth cutting (>1%) | {m['cut_at_all_frac']:.1%} |",
        f"| tail share of all characters decoded | {m['tail_frac']:.1%} |",
        f"| median tail share of one completion | {m['median_tail_frac']:.1%} |",
        f"| p90 tail share of one completion | {m['p90_tail_frac']:.1%} |",
        f"| decode tokens in these emissions | {m['decode_tokens']} |",
        f"| tokens the cut would not have decoded (projected) | {m['projected_tokens_saved']} |",
        "",
        "On a local server tokens are the wall clock, so this is the ceiling on what stop-at-fill "
        "can buy for per-slot requests on this model — the controlled speedups in `results/latency.md` "
        "are what the scheduler does with it.", "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="results/tail.md")
    a = ap.parse_args()
    dirs = [Path(d) for d in a.run_dirs]
    m = measure(dirs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(m, a.run_dirs))
    out.with_suffix(".json").write_text(json.dumps(m, indent=1))
    print(out.read_text())


if __name__ == "__main__":
    main()
