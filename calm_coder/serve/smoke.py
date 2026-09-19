"""Hours 2–4 acceptance: n=4 slot fills for one ClassEval task, extracted into definitions.

python -m calm_coder.serve.smoke [--task 0] [--slot filter] [--n 4]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rich import print

from calm_coder.bench.classeval import task_from_row
from calm_coder.serve.client import Client
from calm_coder.serve.extract import ExtractError, extract_functions
from calm_coder.serve.prompts import SAMPLING, slot_messages
from calm_coder.store.defs import emission_to_defs
from calm_coder.store.store import Store

FIX = Path(__file__).parents[2] / "tests" / "fixtures" / "classeval_head5.json"


async def main(task_idx: int, slot_id: str | None, n: int) -> None:
    task = task_from_row(json.loads(FIX.read_text())[task_idx])
    slot = task.slot(slot_id) if slot_id else task.slots[0]
    store, log = Store(), []
    async with Client() as c:
        print(c.config())
        samples = await c.sample(slot_messages(task, slot.id), n=n, temperature=SAMPLING["temperature"],
                                 top_p=SAMPLING["top_p"], max_tokens=SAMPLING["max_tokens_slot"], seed=0)
    for i, s in enumerate(samples):
        fns = extract_functions(s.text, class_name=task.class_name)
        if isinstance(fns, ExtractError):
            print(f"[red]sample {i}: {fns.reason}[/red]\n{s.text[:400]}")
            continue
        defs = emission_to_defs(task, slot, fns, {"sample_idx": i, "seed": s.seed}, log=log)
        new = [store.add_def(d) for d in defs]
        print(f"sample {i}: {s.completion_tokens} tok, {s.latency_ms} ms, defs={[d.kind[0] + ':' + d.hash[:8] for d in defs]} new={new}")
    fills = store.defs_for_slot(slot.id)
    print(f"[bold]{task.task_id}.{slot.id}: {len(samples)} samples -> {len(fills)} distinct fills, "
          f"{len(store.helpers())} helpers[/bold]; log={log}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--slot")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    asyncio.run(main(a.task, a.slot, a.n))
