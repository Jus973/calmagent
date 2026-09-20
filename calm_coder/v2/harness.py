"""v2 arms.

V2   k whole-class samples -> store -> recombination search -> targeted repair of dead slots,
     round after round, until the task verifies or the decode budget is gone.
WCR  the fair baseline: same k samples, same feedback level, same budget, but each repair round
     regenerates the whole class and the verdict comes from the samples themselves — no
     recombination, so the only difference between the arms is where new tokens are spent.

Both arms test every sample's own composition before anything else: that is the no-loss property.
A pooled run can therefore never do worse than the samples it pooled, except for samples whose
context a composition cannot carry (`ClassIngest.dropped`, §3.5); those are counted, not assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from calm_coder.agents.scheduler import Scheduler
from calm_coder.serve.prompts import CLASS_SYSTEM, SLOT_SYSTEM
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


@dataclass
class _Run:
    """What both arms keep while they run: the control loops differ, the bookkeeping does not."""

    task: "Task"
    store: Store
    budget: Budget
    sched: Scheduler
    row: dict
    emissions: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.monotonic)

    @classmethod
    def start(cls, task: "Task", arm: str, seed: int, budget_tokens: int, level: Level, k: int,
              max_comps: int, store: Store | None,
              on_event: Callable[[dict], None] | None) -> "_Run":
        store = store if store is not None else Store()
        budget = Budget(budget_tokens)
        row = {"task_id": task.task_id, "arm": arm, "seed": seed, "N": k, "feedback": level,
               "budget_tokens": budget.decode_tokens, "solved": False, "verified_comp": None,
               "decode_tokens": 0, "prompt_tokens": 0, "cached_tokens": None, "requests": 0,
               "rounds": [], "ttfv_ms": None, "test_ms": 0, "gen_ms": 0}
        return cls(task=task, store=store, budget=budget, row=row,
                   sched=Scheduler(task, store, max_comps=max_comps, on_event=on_event))

    def add(self, r: ProduceResult) -> ProduceResult:
        """A producer's tokens, requests and emissions, into the run's row."""
        self.row["decode_tokens"] += r.decode_tokens
        self.row["prompt_tokens"] += r.prompt_tokens
        self.row["requests"] += r.requests
        if r.cached_tokens is not None:
            self.row["cached_tokens"] = (self.row["cached_tokens"] or 0) + r.cached_tokens
        self.emissions += r.emissions
        return r

    async def warm(self, client: "Client", system: str) -> None:
        self.add(await warmup(client, self.task, self.budget, system))

    async def test_seed_comps(self, comps: list[dict]) -> None:
        """The no-loss step: every whole-class sample is evaluated as its own composition."""
        for bindings in comps:
            await self.sched.run_full(Composition.make(bindings))

    def verified_comp(self) -> str | None:
        return next(iter(sorted(verified(self.store, self.task))), None)

    def log_round(self, rnd: int, producer: str, **extra) -> None:
        self.row["rounds"].append({"round": rnd, "producer": producer, **extra,
                                   "dead_after": list(state.dead_slots(self.store, self.task)),
                                   "decode_tokens": self.row["decode_tokens"],
                                   "solved": self.row["solved"]})

    def finish(self) -> RunResult:
        self.row["dead_slots"] = list(state.dead_slots(self.store, self.task))
        self.row["dead_slot_report"] = dict(state.dead_slot_report(self.store, self.task))
        self.row["over_budget"] = self.budget.used > self.budget.decode_tokens
        if self.row["solved"]:
            self.row["ttfv_ms"] = int((time.monotonic() - self.t0) * 1000)
        return RunResult(row=self.row, store=self.store, emissions=self.emissions)


async def run_v2(client: "Client", task: "Task", *, budget_tokens: int, seed: int = 0,
                 k: int = DEFAULT_K, repair_n: int = DEFAULT_REPAIR_N, rounds: int = DEFAULT_ROUNDS,
                 level: Level = "F1", max_comps: int = 64, per_method: bool = False,
                 warm: bool = True, arm: str = "v2", store: Store | None = None,
                 on_event: Callable[[dict], None] | None = None) -> RunResult:
    run = _Run.start(task, arm, seed, budget_tokens, level, k, max_comps, store, on_event)
    store, budget, row = run.store, run.budget, run.row

    if warm:
        await run.warm(client, CLASS_SYSTEM)
    wc = run.add(await WholeClassProducer(client, n=k, arm=arm, run_seed=seed)
                 .produce(task, store, budget))
    if per_method:
        run.add(await PerMethodProducer(client, n=k, arm=arm, run_seed=seed)
                .produce(task, store, budget))
    row["gen_ms"] = int((time.monotonic() - run.t0) * 1000)

    await run.test_seed_comps(wc.seed_comps)
    row["seed_comps"] = len(wc.seed_comps)
    row["seed_comp_solved"] = bool(verified(store, task))

    async def search_round() -> None:
        if verified(store, task):
            row["solved"] = True
            row["verified_comp"] = run.verified_comp()
            return
        cands = state.slot_candidates(store, task)
        if any(not v for v in cands.values()):
            return
        await run.sched.phase1([h for hs in cands.values() for h in hs])
        res = await run.sched.search(cands, {}, {})
        row["test_ms"] += res.test_ms
        if res.verified_comp:
            row["solved"], row["verified_comp"] = True, res.verified_comp

    await search_round()
    run.log_round(0, "whole_class", dead_before=None)

    for r in range(1, rounds + 1):
        if row["solved"] or budget.exhausted:
            break
        rp = RepairProducer(client, n=repair_n, arm=arm, rnd=r, run_seed=seed, level=level)
        dead_before = list(state.dead_slots(store, task))
        if not dead_before:
            break
        if warm:
            await run.warm(client, SLOT_SYSTEM)
        pr = await rp.produce(task, store, budget)
        if not pr.emissions:
            break
        run.add(pr)
        await search_round()
        run.log_round(r, "repair", targets=list(rp.targets), samples_per_target=rp.per_slot,
                      dead_before=dead_before)

    return run.finish()


async def run_wcr(client: "Client", task: "Task", *, budget_tokens: int, seed: int = 0,
                  k: int = DEFAULT_K, repair_n: int = DEFAULT_REPAIR_N, rounds: int = DEFAULT_ROUNDS,
                  level: Level = "F1", max_comps: int = 64, warm: bool = True,
                  arm: str = "wcr", store: Store | None = None,
                  on_event: Callable[[dict], None] | None = None) -> RunResult:
    run = _Run.start(task, arm, seed, budget_tokens, level, k, max_comps, store, on_event)
    store, budget, row = run.store, run.budget, run.row

    if warm:
        await run.warm(client, CLASS_SYSTEM)
    wc = run.add(await WholeClassProducer(client, n=k, arm=arm, run_seed=seed)
                 .produce(task, store, budget))
    row["gen_ms"] = int((time.monotonic() - run.t0) * 1000)
    await run.test_seed_comps(wc.seed_comps)
    row["seed_comps"] = len(wc.seed_comps)
    row["solved"] = bool(verified(store, task))
    row["seed_comp_solved"] = row["solved"]          # the same field the v2 arm reports (§6.3)
    row["verified_comp"] = run.verified_comp()
    run.log_round(0, "whole_class")

    for r in range(1, rounds + 1):
        if row["solved"] or budget.exhausted:
            break
        if warm:
            await run.warm(client, CLASS_SYSTEM)
        pr = await WholeClassRepairProducer(client, n=repair_n, arm=arm, rnd=r, run_seed=seed,
                                            level=level).produce(task, store, budget)
        if not pr.emissions:
            break
        run.add(pr)
        await run.test_seed_comps(pr.seed_comps)         # selection by tests, never recombined
        row["solved"] = bool(verified(store, task))
        row["verified_comp"] = run.verified_comp()
        run.log_round(r, "whole_class_repair")

    return run.finish()
