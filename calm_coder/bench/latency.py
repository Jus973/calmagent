"""Time to first verified composition, one lever at a time (§latency).

The experiment (`bench/experiment.py`) measures solve rate at a token budget; this measures the other
thing the harness claims: that a monotone store lets generation, testing and search overlap, so a
verified class arrives sooner than the phase-by-phase path allows. Four modes, each adding one lever:

  sequential     generate every sample, then stub-test everything, then search one composition at a time
  pipelined      fills are stub-tested as they land, the search tests `--width` at once and fails fast
  adaptive       a slot stops sampling once one of its fills passes its own test in stub context
  stop-at-fill   each request ends when its method is complete; the tail is never decoded

Every mode runs against the same canned completions (`latency_samples.py`) served by `FakeModel`: one
queue, one aggregate throughput, so a token costs the same server time in every mode and a difference
in wall clock is scheduling, not sampling. The tests are real subprocesses. What the fake fixes is the
model's text and its rate (`--ttft-ms`, `--tok-s`), both printed with the results.

python -m calm_coder.bench.latency [--N 4] [--repeat 3] [--tail short|long|none] [--json out.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

from calm_coder.agents.fill import generate_fills
from calm_coder.agents.scheduler import Scheduler
from calm_coder.bench.experiment import _budget_view
from calm_coder.bench.fake_model import FakeModel, FakeModelConfig
from calm_coder.bench.latency_samples import SAMPLES, TAILS
from calm_coder.cli import _pipelined
from calm_coder.serve.client import Client
from calm_coder.store.store import Store
from calm_coder.task import task_from_files

DEMO = Path(__file__).resolve().parents[1] / "demo"


@dataclass(frozen=True)
class Mode:
    name: str
    sequential: bool = False
    width: int = 4
    fail_fast: bool = True
    reuse: bool = True
    adaptive: bool = True
    stop_at_fill: bool = True


MODES: dict[str, Mode] = {
    "sequential": Mode("sequential", sequential=True, width=1, fail_fast=False, reuse=False,
                       adaptive=False, stop_at_fill=False),
    "pipelined": Mode("pipelined", adaptive=False, stop_at_fill=False),
    "adaptive": Mode("adaptive", stop_at_fill=False),
    "stop-at-fill": Mode("stop-at-fill"),
}


@dataclass
class Run:
    mode: str
    verified: bool
    ttfv_ms: int
    requests: int
    decoded_tokens: int
    offered_tokens: int
    executions: int
    comps_tested: int

    def to_json(self) -> dict:
        return dict(self.__dict__)


async def run_mode(task, mode: Mode, *, n: int, seed: int, cfg: FakeModelConfig, tail: str) -> Run:
    model = FakeModel(SAMPLES, tail=tail, cfg=cfg)
    store = Store()
    sched = Scheduler(task, store, max_comps=64, width=mode.width, fail_fast=mode.fail_fast,
                      reuse_class_outcomes=mode.reuse)
    t0 = time.monotonic()
    async with Client(base_url="http://fake.invalid/v1", model="fake", max_inflight=64,
                      transport=model.transport) as client:
        if mode.sequential:
            emissions = await generate_fills(client, task, store, n=n, seed=seed)
            _, cands, fan_in, arrival = _budget_view(emissions, n)
            await sched.phase1([h for hs in cands.values() for h in hs])
            res = await sched.search(cands, fan_in, arrival)
        else:
            res = await _pipelined(client, task, store, sched, n=n, seed=seed, cb=None,
                                   stop_at_fill=mode.stop_at_fill, adaptive=mode.adaptive)
    return Run(mode=mode.name, verified=res.verified_comp is not None,
               ttfv_ms=int((time.monotonic() - t0) * 1000), requests=model.stats.requests,
               decoded_tokens=model.stats.decoded_tokens, offered_tokens=model.stats.offered_tokens,
               executions=sched.executions, comps_tested=len(res.trace))


def median_run(runs: list[Run]) -> Run:
    """Median of each measure. Wall clock is the noisy one; the counts are identical across repeats."""
    med = lambda f: int(statistics.median([getattr(r, f) for r in runs]))  # noqa: E731
    return replace(runs[0], ttfv_ms=med("ttfv_ms"), executions=med("executions"),
                   comps_tested=med("comps_tested"), verified=all(r.verified for r in runs))


def table(rows: list[Run]) -> str:
    base = next((r.ttfv_ms for r in rows if r.mode == "sequential"), None)
    out = ["| mode | verified | TTFV (ms) | speedup | requests | tokens decoded | of offered | test procs |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sp = f"{base / r.ttfv_ms:.2f}x" if base and r.ttfv_ms else "—"
        out.append(f"| {r.mode} | {'yes' if r.verified else 'no'} | {r.ttfv_ms} | {sp} | "
                   f"{r.requests} | {r.decoded_tokens} | {r.offered_tokens} | {r.executions} |")
    return "\n".join(out)


async def main_async(a) -> int:
    task = task_from_files(DEMO / "task.py", DEMO / "test_task.py")
    cfg = FakeModelConfig(ttft_ms=a.ttft_ms, tok_s=a.tok_s)
    tail = TAILS[a.tail]
    rows, raw = [], []
    for name in a.modes:
        runs = [await run_mode(task, MODES[name], n=a.N, seed=a.seed, cfg=cfg, tail=tail)
                for _ in range(a.repeat)]
        raw += runs
        rows.append(median_run(runs))
    header = (f"KVStore demo task · N={a.N} · seed={a.seed} · median of {a.repeat} · fake server "
              f"(prefill {a.ttft_ms:.0f} ms, {a.tok_s:.0f} tok/s shared, {a.tail} tail)")
    print(header)
    print(table(rows))
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"header": header, "config": vars(a), "median": [r.to_json() for r in rows],
             "runs": [r.to_json() for r in raw]}, indent=2, default=str) + "\n")
    return 0 if all(r.verified for r in rows) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calm-latency", description=__doc__.splitlines()[0])
    ap.add_argument("--N", type=int, default=4, help="samples per method")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--ttft-ms", type=float, default=400.0, dest="ttft_ms")
    ap.add_argument("--tok-s", type=float, default=24.0, dest="tok_s")
    ap.add_argument("--tail", default="short", choices=list(TAILS),
                    help="what the model appends after the method it was asked for")
    ap.add_argument("--json", help="write every run here")
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
