"""Run every packaged task under each proxy arm, interleaved, one at a time.

    python -m bench_agent.ab --arms off,on --agent mini-swe --out-root runs/ --label ab

Why interleaved (`off,on` per task rather than all `off` then all `on`): this machine's throughput
drifts — thermal, background work, another model resident — and a drift that lands on one arm
would read as a result. Alternating per task spreads it across both.

**What `off` means here.** Not "no proxy": both arms go through the same proxy process, `off` with
tracing only and `on` with the shipped levers enabled. The alternative — `off` straight to Ollama —
would leave the control arm with no trace at all, and Ollama reports no usage on a streamed
request and no cache figures on any request, so there would be nothing to compare against. The
`trace` middleware does not touch the request or the response: it relays bytes and writes a line.
The one exception is documented and deliberate: on a streamed request the proxy sets
`stream_options.include_usage` so the server returns a token count. That is additive, identical in
both arms, and cannot reach the prompt.

Concurrency is 1. The laptop has one GPU and one KV slot; two tasks at once would have them evict
each other's prefix and both arms would measure the contention instead of the lever.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from bench_agent.run_agent import run_one

ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
PORT = 8999


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def ollama_version() -> str:
    try:
        return json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/version",
                                                 timeout=10).read())["version"]
    except Exception:  # noqa: BLE001
        return "unknown"


def start_proxy(trace_dir: pathlib.Path, flags: list[str], port: int,
                impl: str = "bench") -> subprocess.Popen:
    """Start the proxy the A/B will run through.

    This used to prefer `calm_proxy` whenever that package existed. It landed implementing
    `trace` only -- `--enable dedup` is accepted and does nothing -- and its entry point is
    `python -m calm_proxy`, not `calm_proxy.server`, so the automatic preference both failed to
    start and would have measured a no-op lever if it had. Every result in `runs/` was produced by
    `bench_agent.trace_proxy`, so that is the default; `--proxy calm_proxy` selects the other one
    explicitly, for tracing.
    """
    if impl == "calm_proxy":
        cmd = [sys.executable, "-m", "calm_proxy", "--port", str(port),
               "--trace-dir", str(trace_dir), *flags]
    else:
        cmd = [sys.executable, "-m", "bench_agent.trace_proxy", "--port", str(port),
               "--trace-dir", str(trace_dir), *flags]
    log = (trace_dir / "proxy.log").open("w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
            return p
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
    raise SystemExit(f"proxy on {port} never came up; see {trace_dir}/proxy.log")


def stop_proxy(p: subprocess.Popen) -> None:
    p.send_signal(signal.SIGINT)
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="off,on")
    ap.add_argument("--agent", default="mini-swe")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--tasks", default=str(TASKS / "TASKS.json"))
    ap.add_argument("--only", default="", help="comma-separated task ids, default all")
    ap.add_argument("--cap-s", type=int, default=480)
    ap.add_argument("--proxy-flags", default="", help="flags for the `on` arm, from DECISIONS.md")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--label", default="ab")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--proxy", default="bench", choices=["bench", "calm_proxy"],
                    help="which proxy implementation to run through (default: bench, the one "
                         "that implements the dedup lever)")
    args = ap.parse_args()

    manifest = json.loads(pathlib.Path(args.tasks).read_text())
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    if args.only:
        wanted = [t.strip() for t in args.only.split(",")]
        task_ids = [t for t in task_ids if t in wanted]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    on_flags = args.proxy_flags.split() if args.proxy_flags else []

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dirs = {}
    for arm in arms:
        d = pathlib.Path(args.out_root) / f"{ts}_{args.label}_{arm}"
        d.mkdir(parents=True, exist_ok=True)
        dirs[arm] = d
        (d / "config.json").write_text(json.dumps({
            "arm": arm, "agent": args.agent, "model": args.model, "cap_s": args.cap_s,
            "proxy_flags": on_flags if arm == "on" else [],
            "git_commit": git_commit(), "ollama_version": ollama_version(),
            "tasks": task_ids, "interleaved": True, "concurrency": 1,
            "proxy": "calm_proxy" if args.proxy == "calm_proxy" else "bench_agent.trace_proxy",
            "started_utc": ts,
        }, indent=2) + "\n")

    t_start = time.time()
    for i, task_id in enumerate(task_ids):
        for arm in arms:
            d = dirs[arm]
            done = {json.loads(l)["task_id"] for l in (d / "results.jsonl").read_text().splitlines()
                    if l.strip()} if (d / "results.jsonl").exists() else set()
            if task_id in done:
                print(f"[{i+1}/{len(task_ids)}] {task_id} {arm}: already done, skipping", flush=True)
                continue
            proxy = start_proxy(d, on_flags if arm == "on" else [], args.port, args.proxy)
            try:
                r = run_one(args.agent, task_id, f"http://127.0.0.1:{args.port}/v1",
                            args.model, d, args.cap_s)
            finally:
                stop_proxy(proxy)
            print(f"[{i+1}/{len(task_ids)}] {task_id} {arm}: solved={r['solved']} "
                  f"wall={r['wall_s']}s reason={r['agent_reason']} ({r['tail']}) "
                  f"[elapsed {int(time.time()-t_start)}s]", flush=True)

    for arm, d in dirs.items():
        rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()] \
            if (d / "results.jsonl").exists() else []
        solved = sum(1 for r in rows if r["solved"])
        print(f"{arm}: {solved}/{len(rows)} solved -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
