"""v2 arms.

V2   k whole-class samples -> store -> recombination search -> targeted repair of dead slots,
     round after round, until the task verifies or the decode budget is gone.
WCR  the fair baseline: same k samples, same feedback level, same budget, but each repair round
     regenerates the whole class and the verdict comes from the samples themselves — no
     recombination, so the only difference between the arms is where new tokens are spent.

Both arms test every sample's own composition before anything else: that is the no-loss property.
A pooled run can therefore never do worse than the samples it pooled.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from calm_coder.agents.scheduler import Scheduler
from calm_coder.store.defs import Composition
from calm_coder.store.derive import verified
from calm_coder.store.store import Store
from calm_coder.v2 import state
from calm_coder.v2.feedback import Level
from calm_coder.v2.producers import (
    Budget, ProduceResult, RepairProducer, PerMethodProducer, WholeClassProducer,
    WholeClassRepairProducer, warmup,
)

if TYPE_CHECKING:
    from calm_coder.serve.client import Client
    from calm_coder.task import Task

DEFAULT_K = 4
DEFAULT_ROUNDS = 3
DEFAULT_REPAIR_N = 8


@dataclass
class RunResult:
    row: dict
    store: Store
    emissions: list[dict] = field(default_factory=list)


def _acc(row: dict, r: ProduceResult) -> None:
    row["decode_tokens"] += r.decode_tokens
    row["prompt_tokens"] += r.prompt_tokens
    row["requests"] += r.requests
    if r.cached_tokens is not None:
        row["cached_tokens"] = (row["cached_tokens"] or 0) + r.cached_tokens


def _row(task: "Task", arm: str, seed: int, budget: Budget, level: Level, k: int) -> dict:
    return {"task_id": task.task_id, "arm": arm, "seed": seed, "N": k, "feedback": level,
            "budget_tokens": budget.decode_tokens, "solved": False, "verified_comp": None,
            "decode_tokens": 0, "prompt_tokens": 0, "cached_tokens": None, "requests": 0,
            "rounds": [], "ttfv_ms": None, "test_ms": 0, "gen_ms": 0}


async def _test_seed_comps(sched: Scheduler, comps: list[dict]) -> None:
    """The no-loss step: every whole-class sample is evaluated as its own composition."""
    for bindings in comps:
        await sched.run_full(Composition.make(bindings))


async def run_v2(client: "Client", task: "Task", *, budget_tokens: int, seed: int = 0,
                 k: int = DEFAULT_K, repair_n: int = DEFAULT_REPAIR_N, rounds: int = DEFAULT_ROUNDS,
                 level: Level = "F1", max_comps: int = 64, per_method: bool = False,
                 warm: bool = True, arm: str = "v2", store: Store | None = None,
                 on_event: Callable[[dict], None] | None = None) -> RunResult:
    store = store if store is not None else Store()
    budget = Budget(budget_tokens)
    sched = Scheduler(task, store, max_comps=max_comps, on_event=on_event)
    row = _row(task, arm, seed, budget, level, k)
    emissions: list[dict] = []
    t0 = time.monotonic()

    if warm:
        _acc(row, await warmup(client, task, budget))
    wc = await WholeClassProducer(client, k=k, arm=arm).produce(task, store, budget)
    _acc(row, wc)
    emissions += wc.emissions
    if per_method:
        pm = await PerMethodProducer(client, n=k, arm=arm).produce(task, store, budget)
        _acc(row, pm)
        emissions += pm.emissions
    row["gen_ms"] = int((time.monotonic() - t0) * 1000)

    await _test_seed_comps(sched, wc.seed_comps)
    row["seed_comps"] = len(wc.seed_comps)
    row["seed_comp_solved"] = bool(verified(store, task))

    async def search_round() -> None:
        if verified(store, task):
            row["solved"] = True
            row["verified_comp"] = next(iter(sorted(verified(store, task))))
            return
        cands = state.slot_candidates(store, task)
        if any(not v for v in cands.values()):
            return
        await sched.phase1([h for hs in cands.values() for h in hs])
        res = await sched.search(cands, {}, {})
        row["test_ms"] += res.test_ms
        if res.verified_comp:
            row["solved"], row["verified_comp"] = True, res.verified_comp

    await search_round()
    row["rounds"].append({"round": 0, "producer": "whole_class", "dead_before": None,
                          "dead_after": list(state.dead_slots(store, task)),
                          "decode_tokens": row["decode_tokens"], "solved": row["solved"]})

    for r in range(1, rounds + 1):
        if row["solved"] or budget.exhausted:
            break
        rp = RepairProducer(client, n=repair_n, arm=arm, rnd=r, level=level)
        dead_before = list(state.dead_slots(store, task))
        if not dead_before:
            break
        if warm:
            _acc(row, await warmup(client, task, budget))
        pr = await rp.produce(task, store, budget)
        if not pr.emissions:
            break
        _acc(row, pr)
        emissions += pr.emissions
        await search_round()
        row["rounds"].append({"round": r, "producer": "repair", "targets": list(rp.targets),
                              "dead_before": dead_before, "dead_after": list(state.dead_slots(store, task)),
                              "decode_tokens": row["decode_tokens"], "solved": row["solved"]})

    row["dead_slots"] = list(state.dead_slots(store, task))
    row["dead_slot_report"] = dict(state.dead_slot_report(store, task))
    row["over_budget"] = budget.used > budget.decode_tokens
    if row["solved"]:
        row["ttfv_ms"] = int((time.monotonic() - t0) * 1000)
    return RunResult(row=row, store=store, emissions=emissions)


async def run_wcr(client: "Client", task: "Task", *, budget_tokens: int, seed: int = 0,
                  k: int = DEFAULT_K, repair_n: int = DEFAULT_K, rounds: int = DEFAULT_ROUNDS,
                  level: Level = "F1", max_comps: int = 64, warm: bool = True,
                  arm: str = "wcr", store: Store | None = None,
                  on_event: Callable[[dict], None] | None = None) -> RunResult:
    store = store if store is not None else Store()
    budget = Budget(budget_tokens)
    sched = Scheduler(task, store, max_comps=max_comps, on_event=on_event)
    row = _row(task, arm, seed, budget, level, k)
    emissions: list[dict] = []
    t0 = time.monotonic()

    if warm:
        _acc(row, await warmup(client, task, budget))
    wc = await WholeClassProducer(client, k=k, arm=arm).produce(task, store, budget)
    _acc(row, wc)
    emissions += wc.emissions
    row["gen_ms"] = int((time.monotonic() - t0) * 1000)
    await _test_seed_comps(sched, wc.seed_comps)
    row["seed_comps"] = len(wc.seed_comps)
    row["solved"] = bool(verified(store, task))
    row["verified_comp"] = next(iter(sorted(verified(store, task))), None)
    row["rounds"].append({"round": 0, "producer": "whole_class", "dead_after": list(state.dead_slots(store, task)),
                          "decode_tokens": row["decode_tokens"], "solved": row["solved"]})

    for r in range(1, rounds + 1):
        if row["solved"] or budget.exhausted:
            break
        if warm:
            _acc(row, await warmup(client, task, budget))
        pr = await WholeClassRepairProducer(client, n=repair_n, arm=arm, rnd=r, level=level) \
            .produce(task, store, budget)
        if not pr.emissions:
            break
        _acc(row, pr)
        emissions += pr.emissions
        await _test_seed_comps(sched, pr.seed_comps)     # selection by tests, never recombined
        row["solved"] = bool(verified(store, task))
        row["verified_comp"] = next(iter(sorted(verified(store, task))), None)
        row["rounds"].append({"round": r, "producer": "whole_class_repair",
                              "dead_after": list(state.dead_slots(store, task)),
                              "decode_tokens": row["decode_tokens"], "solved": row["solved"]})

    row["dead_slots"] = list(state.dead_slots(store, task))
    row["over_budget"] = budget.used > budget.decode_tokens
    if row["solved"]:
        row["ttfv_ms"] = int((time.monotonic() - t0) * 1000)
    return RunResult(row=row, store=store, emissions=emissions)
