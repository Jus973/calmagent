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
from typing import TYPE_CHECKING, Sequence

from calm_coder.agents.fill import Emission, ingest
from calm_coder.serve.prompts import (
    SAMPLING, shared_prefix, slot_messages, slot_repair_messages, whole_class_messages,
    whole_class_repair_messages,
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


def seed_for(task_id: str, arm: str, slot: str, rnd: int, idx: int) -> int:
    """Sampling seed per request = hash(task_id, arm, slot, round, sample_idx)."""
    key = f"{task_id}|{arm}|{slot}|{rnd}|{idx}"
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
    new_fills: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)


async def warmup(client: "Client", task: "Task", budget: Budget) -> ProduceResult:
    """One request with the shared prefix and max_tokens=1, before a task's fan-out.

    Requests that arrive in the same scheduling step are not guaranteed to share a prefix that is
    still being computed; this removes the question. Its tokens are counted like any other.
    """
    if budget.exhausted:
        return ProduceResult(producer="warmup")
    (s,) = await client.sample([{"role": "user", "content": shared_prefix(task)}], n=1,
                               temperature=0.0, max_tokens=1, seed=0)
    budget.spend(s.completion_tokens)
    return ProduceResult(producer="warmup", decode_tokens=s.completion_tokens,
                         prompt_tokens=s.prompt_tokens, cached_tokens=s.cached_tokens, requests=1)


async def _sample_group(client: "Client", task: "Task", arm: str, slot: str, rnd: int, n: int,
                        messages: list[dict], t_start: float, *, temperature: float,
                        top_p: float | None, max_tokens: int) -> list[Emission]:
    """One request per (slot, round) with n choices: the server computes the shared prefix once.

    Clients whose server ignores `n` fan the request out themselves with seed+i, so the sample
    identities are the same either way.
    """
    seed = seed_for(task.task_id, arm, slot, rnd, 0)
    samples = await client.sample(messages, n=n, temperature=temperature, top_p=top_p,
                                  max_tokens=max_tokens, seed=seed)
    return [_emission(task, arm, slot, rnd, i, s, t_start, seed + i) for i, s in enumerate(samples)]


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


class WholeClassProducer:
    """k whole-class samples, decomposed into slot candidates on insert."""

    name = "whole_class"

    def __init__(self, client: "Client", *, k: int, arm: str, rnd: int = 0,
                 temperature: float = SAMPLING["temperature"], top_p: float | None = SAMPLING["top_p"],
                 max_tokens: int = SAMPLING["max_tokens_class"]):
        self.client, self.k, self.arm, self.rnd = client, k, arm, rnd
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens

    async def produce(self, task: "Task", store: Store, budget: Budget) -> ProduceResult:
        t_start = time.monotonic()
        res = ProduceResult(producer=self.name)
        max_tokens = cap_tokens(budget, self.k, self.max_tokens)
        if not max_tokens:
            return res

        emissions = await _sample_group(self.client, task, self.arm, SPINE, self.rnd, self.k,
                                        whole_class_messages(task), t_start,
                                        temperature=self.temperature, top_p=self.top_p,
                                        max_tokens=max_tokens)
        for e in emissions:
            ing = ingest_class_sample(task, store, e.text,
                                      {"producer": self.name, "arm": self.arm, "round": self.rnd,
                                       "sample_idx": e.sample_idx, "seed": e.sample_seed})
            e.extract_error = ing.error
            e.def_hashes = ing.def_hashes
            res.new_fills += [h for h in ing.bindings.values()]
            res.texts.append(e.text)
            if ing.complete and not ing.error:
                res.seed_comps.append(dict(ing.bindings))
        budget.spend(sum(e.completion_tokens for e in emissions))
        return _totals(res, emissions, 1)


class PerMethodProducer:
    """The v1 arm's producer: n independent samples per slot, straight from the skeleton."""

    name = "per_method"

    def __init__(self, client: "Client", *, n: int, arm: str, rnd: int = 0,
                 temperature: float = SAMPLING["temperature"], top_p: float | None = SAMPLING["top_p"],
                 max_tokens: int = SAMPLING["max_tokens_slot"]):
        self.client, self.n, self.arm, self.rnd = client, n, arm, rnd
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens

    async def produce(self, task: "Task", store: Store, budget: Budget) -> ProduceResult:
        t_start = time.monotonic()
        res = ProduceResult(producer=self.name)
        max_tokens = cap_tokens(budget, self.n * len(task.slots), self.max_tokens)
        if not max_tokens:
            return res

        groups = await asyncio.gather(*(
            _sample_group(self.client, task, self.arm, s.id, self.rnd, self.n,
                          slot_messages(task, s.id), t_start, temperature=self.temperature,
                          top_p=self.top_p, max_tokens=max_tokens)
            for s in task.slots))
        emissions = sorted((e for g in groups for e in g),
                           key=lambda e: (task.slot(e.slot).order, e.sample_idx))
        for e in emissions:
            ingest(task, store, e, extra={"producer": self.name, "round": self.rnd})
            if e.fill_hash:
                res.new_fills.append(e.fill_hash)
        budget.spend(sum(e.completion_tokens for e in emissions))
        return _totals(res, emissions, len(task.slots))


class RepairProducer:
    """Targeted repair: for each dead slot, regenerate only that method, conditioned on the
    skeleton, the current best class, and that slot's failing test output at the feedback level."""

    name = "repair"

    def __init__(self, client: "Client", *, n: int, arm: str, rnd: int, level: Level = "F1",
                 temperature: float = SAMPLING["temperature"], top_p: float | None = SAMPLING["top_p"],
                 max_tokens: int = SAMPLING["max_tokens_slot"]):
        self.client, self.n, self.arm, self.rnd, self.level = client, n, arm, rnd, level
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.targets: tuple[str, ...] = ()

    def plan(self, store: Store, task: "Task") -> tuple["Composition | None", tuple[str, ...]]:
        comp = state.best_class(store, task)
        return comp, state.dead_slots(store, task, comp)

    async def produce(self, task: "Task", store: Store, budget: Budget) -> ProduceResult:
        t_start = time.monotonic()
        res = ProduceResult(producer=self.name)
        comp, targets = self.plan(store, task)
        self.targets = targets
        if not targets:
            return res
        max_tokens = cap_tokens(budget, self.n * len(targets), self.max_tokens)
        if not max_tokens:
            return res
        current = state.class_source(store, task, comp)
        # Every request of this round shares everything up to the end of the current-class block.
        prompts = {slot: slot_repair_messages(
            task, slot, current,
            render(state.slot_failures(store, task, slot, comp), self.level, slot_id=slot))
            for slot in targets}

        groups = await asyncio.gather(*(
            _sample_group(self.client, task, self.arm, slot, self.rnd, self.n, prompts[slot], t_start,
                          temperature=self.temperature, top_p=self.top_p, max_tokens=max_tokens)
            for slot in targets))
        emissions = sorted((e for g in groups for e in g), key=lambda e: (e.slot, e.sample_idx))
        for e in emissions:
            ingest(task, store, e, extra={"producer": self.name, "round": self.rnd})
            if e.fill_hash:
                res.new_fills.append(e.fill_hash)
        budget.spend(sum(e.completion_tokens for e in emissions))
        return _totals(res, emissions, len(targets))


class WholeClassRepairProducer:
    """The fair baseline for repair: same feedback, same information, whole class regenerated.
    Candidates are still decomposed into the store so the arms log identically, but the WCR arm's
    verdict is taken from the samples themselves (no recombination)."""

    name = "whole_class_repair"

    def __init__(self, client: "Client", *, n: int, arm: str, rnd: int, level: Level = "F1",
                 temperature: float = SAMPLING["temperature"], top_p: float | None = SAMPLING["top_p"],
                 max_tokens: int = SAMPLING["max_tokens_class"]):
        self.client, self.n, self.arm, self.rnd, self.level = client, n, arm, rnd, level
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens

    async def produce(self, task: "Task", store: Store, budget: Budget) -> ProduceResult:
        t_start = time.monotonic()
        res = ProduceResult(producer=self.name)
        max_tokens = cap_tokens(budget, self.n, self.max_tokens)
        if not max_tokens:
            return res
        comp = state.best_class(store, task)
        current = state.class_source(store, task, comp)
        failures = [f for slot in state.dead_slots(store, task, comp)
                    for f in state.slot_failures(store, task, slot, comp)]
        seen, uniq = set(), []
        for f in failures:
            if f.test_class not in seen:
                seen.add(f.test_class)
                uniq.append(f)
        messages = whole_class_repair_messages(task, current, render(uniq, self.level))

        emissions = await _sample_group(self.client, task, self.arm, SPINE, self.rnd, self.n, messages,
                                        t_start, temperature=self.temperature, top_p=self.top_p,
                                        max_tokens=max_tokens)
        for e in emissions:
            ing = ingest_class_sample(task, store, e.text,
                                      {"producer": self.name, "arm": self.arm, "round": self.rnd,
                                       "sample_idx": e.sample_idx, "seed": e.sample_seed})
            e.extract_error = ing.error
            e.def_hashes = ing.def_hashes
            res.texts.append(e.text)
            if ing.complete and not ing.error:
                res.seed_comps.append(dict(ing.bindings))
        budget.spend(sum(e.completion_tokens for e in emissions))
        return _totals(res, emissions, 1)
