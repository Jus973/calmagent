"""Producers: anything that writes candidates into the store.

    produce(task, store, budget) -> ProduceResult

Whole-class samples, per-method samples and targeted repairs are the same kind of event once they
are in the store; only the prompt differs. Every producer spends from the task's decode budget and
stops when it is gone, so arms are comparable at equal tokens.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar, Sequence

from calm_coder.agents.fill import Emission, ingest
from calm_coder.serve.prompts import (
    CLASS_SYSTEM, SAMPLING, slot_messages, slot_repair_messages, warmup_messages,
    whole_class_messages, whole_class_repair_messages,
)
from calm_coder.store.store import Store
from calm_coder.v2 import state
from calm_coder.v2.decompose import ingest_class_sample
from calm_coder.v2.feedback import Level, render

if TYPE_CHECKING:
    from calm_coder.serve.client import Client
    from calm_coder.store.defs import Composition
    from calm_coder.task import Task

SPINE = "__class__"          # slot id used for a whole-class emission's own composition in logs


def seed_for(run_seed: int, task_id: str, arm: str, slot: str, rnd: int, idx: int) -> int:
    """Sampling seed per request = hash(run_seed, task_id, arm, slot, round, sample_idx).

    The run seed is in the key because two runs of an arm that differ only in their seed must be
    two samples of the arm and not the same run twice — otherwise repeated seeds are
    pseudo-replication, and the arm is not comparable with a baseline that does vary with it.
    """
    key = f"{run_seed}|{task_id}|{arm}|{slot}|{rnd}|{idx}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


@dataclass
class Budget:
    """Decode tokens a task may spend. Warm-up and repair decode tokens count."""
    decode_tokens: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.decode_tokens - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.decode_tokens

    def spend(self, n: int) -> None:
        self.used += n


# A request can only be stopped at max_tokens, so the budget is enforced before it is sent: the
# round's per-sample cap is the remaining budget split over the samples it is about to ask for.
# Below MIN_TOKENS a sample cannot produce a usable method, so the round is skipped instead.
MIN_TOKENS = 32


def cap_tokens(budget: Budget, samples: int, max_tokens: int) -> int:
    """Per-sample max_tokens for a round of `samples` samples, or 0 if the budget cannot fund it."""
    per = budget.remaining // max(1, samples)
    return min(max_tokens, per) if per >= MIN_TOKENS else 0


@dataclass
class ProduceResult:
    producer: str
    emissions: list[dict] = field(default_factory=list)
    decode_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int | None = None
    requests: int = 0
    seed_comps: list[dict] = field(default_factory=list)   # whole-class samples' own bindings


async def warmup(client: "Client", task: "Task", budget: Budget,
                 system: str = CLASS_SYSTEM) -> ProduceResult:
    """One request with the shared prefix and max_tokens=1, before a task's fan-out.

    Requests that arrive in the same scheduling step are not guaranteed to share a prefix that is
    still being computed; this removes the question. `system` is the system message of the round
    about to be sent, so what is warmed is a real prefix of it. Its tokens are counted like any
    other.
    """
    if budget.exhausted:
        return ProduceResult(producer="warmup")
    (s,) = await client.sample(warmup_messages(task, system), n=1,
                               temperature=0.0, max_tokens=1, seed=0)
    budget.spend(s.completion_tokens)
    return ProduceResult(producer="warmup", decode_tokens=s.completion_tokens,
                         prompt_tokens=s.prompt_tokens, cached_tokens=s.cached_tokens, requests=1)


def _emission(task: "Task", arm: str, slot: str, rnd: int, idx: int, s, t_start: float,
              seed: int) -> Emission:
    return Emission(task_id=task.task_id, arm=arm, slot=slot, seed=rnd, sample_idx=idx,
                    sample_seed=seed, text=s.text, completion_tokens=s.completion_tokens,
                    prompt_tokens=s.prompt_tokens, cached_tokens=s.cached_tokens,
                    tokens_estimated=s.tokens_estimated, latency_ms=s.latency_ms,
                    t_done_ms=int((time.monotonic() - t_start) * 1000), finish_reason=s.finish_reason)


def _totals(res: ProduceResult, emissions: Sequence[Emission], requests: int) -> ProduceResult:
    """`requests` counts prompts the producer issued, not samples: a server that honours `n`
    computes the shared prefix once per prompt, which is the number the prefix claim is about."""
    res.decode_tokens += sum(e.completion_tokens for e in emissions)
    res.prompt_tokens += sum(e.prompt_tokens for e in emissions)
    res.requests += requests
    cached = [e.cached_tokens for e in emissions if e.cached_tokens is not None]
    if cached:
        res.cached_tokens = (res.cached_tokens or 0) + sum(cached)
    res.emissions += [e.to_json() for e in emissions]
    return res


@dataclass
class _Producer:
    """Sampling parameters and the two round shapes every producer is built out of: `n` samples of
    the whole class, or `per_slot` samples of each of several slots. Subclasses only choose the
    prompts and how many samples the round asks for."""

    name: ClassVar[str] = ""

    client: "Client"
    n: int
    arm: str
    rnd: int = 0
    run_seed: int = 0
    temperature: float = SAMPLING["temperature"]
    top_p: float | None = SAMPLING["top_p"]
    max_tokens: int = SAMPLING["max_tokens_slot"]

    async def _group(self, task: "Task", slot: str, n: int, messages: list[dict], t_start: float,
                     max_tokens: int) -> list[Emission]:
        """One request per (slot, round) with n choices: the server computes the shared prefix once.

        Clients whose server ignores `n` fan the request out themselves with seed+i, so the sample
        identities are the same either way.
        """
        seed = seed_for(self.run_seed, task.task_id, self.arm, slot, self.rnd, 0)
        samples = await self.client.sample(messages, n=n, temperature=self.temperature,
                                           top_p=self.top_p, max_tokens=max_tokens, seed=seed)
        return [_emission(task, self.arm, slot, self.rnd, i, s, t_start, seed + i)
                for i, s in enumerate(samples)]

    async def _class_round(self, task: "Task", store: Store, budget: Budget,
                           messages: Callable[[], list[dict]]) -> ProduceResult:
        """n whole-class samples, each decomposed into slot candidates and kept as its own
        composition (`seed_comps`), which is what makes pooling lossless."""
        t_start = time.monotonic()
        res = ProduceResult(producer=self.name)
        max_tokens = cap_tokens(budget, self.n, self.max_tokens)
        if not max_tokens:
            return res
        emissions = await self._group(task, SPINE, self.n, messages(), t_start, max_tokens)
        for e in emissions:
            ing = ingest_class_sample(task, store, e.text,
                                      {"producer": self.name, "arm": self.arm, "round": self.rnd,
                                       "sample_idx": e.sample_idx, "seed": e.sample_seed})
            e.extract_error = ing.error
            e.def_hashes = ing.def_hashes
            if ing.complete and not ing.error:
                res.seed_comps.append(dict(ing.bindings))
        budget.spend(sum(e.completion_tokens for e in emissions))
        return _totals(res, emissions, 1)

    async def _slot_round(self, task: "Task", store: Store, budget: Budget,
                          prompts: dict[str, list[dict]], per_slot: int, max_tokens: int,
                          t_start: float, key) -> ProduceResult:
        """per_slot samples of every slot in `prompts`, one request per slot."""
        res = ProduceResult(producer=self.name)
        groups = await asyncio.gather(*(
            self._group(task, slot, per_slot, msgs, t_start, max_tokens)
            for slot, msgs in prompts.items()))
        emissions = sorted((e for g in groups for e in g), key=key)
        for e in emissions:
            ingest(task, store, e, extra={"producer": self.name, "round": self.rnd})
        budget.spend(sum(e.completion_tokens for e in emissions))
        return _totals(res, emissions, len(prompts))


@dataclass
class WholeClassProducer(_Producer):
    """n whole-class samples, decomposed into slot candidates on insert."""

    name = "whole_class"
    max_tokens: int = SAMPLING["max_tokens_class"]

    async def produce(self, task: "Task", store: Store, budget: Budget) -> ProduceResult:
        return await self._class_round(task, store, budget, lambda: whole_class_messages(task))


@dataclass
class PerMethodProducer(_Producer):
    """The v1 arm's producer: n independent samples per slot, straight from the skeleton."""

    name = "per_method"

    async def produce(self, task: "Task", store: Store, budget: Budget) -> ProduceResult:
        t_start = time.monotonic()
        max_tokens = cap_tokens(budget, self.n * len(task.slots), self.max_tokens)
        if not max_tokens:
            return ProduceResult(producer=self.name)
        prompts = {s.id: slot_messages(task, s.id) for s in task.slots}
        return await self._slot_round(task, store, budget, prompts, self.n, max_tokens, t_start,
                                      key=lambda e: (task.slot(e.slot).order, e.sample_idx))


@dataclass
class RepairProducer(_Producer):
    """Targeted repair: for each dead slot, regenerate only that method, conditioned on the
    skeleton, the current best class, and that slot's failing test output at the feedback level."""

    name = "repair"
    level: Level = "F1"
    targets: tuple[str, ...] = ()
    per_slot: int = 0

    def plan(self, store: Store, task: "Task") -> tuple["Composition | None", tuple[str, ...]]:
        comp = state.best_class(store, task)
        return comp, state.dead_slots(store, task, comp)

    async def produce(self, task: "Task", store: Store, budget: Budget) -> ProduceResult:
        t_start = time.monotonic()
        comp, targets = self.plan(store, task)
        self.targets = targets
        if not targets:
            return ProduceResult(producer=self.name)
        # `n` is the round's sample count, not a per-slot one: a round of targeted repair asks for
        # as many samples as a whole-class repair round does, so the two arms spend alike.
        self.per_slot = per_slot = max(1, self.n // len(targets))
        max_tokens = cap_tokens(budget, per_slot * len(targets), self.max_tokens)
        if not max_tokens:
            return ProduceResult(producer=self.name)
        current = state.class_source(store, task, comp)
        # Every request of this round shares everything up to the end of the current-class block.
        prompts = {slot: slot_repair_messages(
            task, slot, current,
            render(state.slot_failures(store, task, slot, comp), self.level, slot_id=slot))
            for slot in targets}
        return await self._slot_round(task, store, budget, prompts, per_slot, max_tokens, t_start,
                                      key=lambda e: (e.slot, e.sample_idx))


@dataclass
class WholeClassRepairProducer(_Producer):
    """The fair baseline for repair: same feedback, same information, whole class regenerated.
    Candidates are still decomposed into the store so the arms log identically, but the WCR arm's
    verdict is taken from the samples themselves (no recombination)."""

    name = "whole_class_repair"
    level: Level = "F1"
    max_tokens: int = SAMPLING["max_tokens_class"]

    def _messages(self, task: "Task", store: Store) -> list[dict]:
        comp = state.best_class(store, task)
        current = state.class_source(store, task, comp)
        seen: set[str] = set()
        uniq = []
        for slot in state.dead_slots(store, task, comp):
            for f in state.slot_failures(store, task, slot, comp):
                if f.test_class not in seen:
                    seen.add(f.test_class)
                    uniq.append(f)
        return whole_class_repair_messages(task, current, render(uniq, self.level))

    async def produce(self, task: "Task", store: Store, budget: Budget) -> ProduceResult:
        return await self._class_round(task, store, budget, lambda: self._messages(task, store))
