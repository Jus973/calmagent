"""Phase 0 (§3.12): N concurrent fills per slot, extracted into the grow-only store."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from calm_coder.serve.extract import ExtractError, extract_functions
from calm_coder.serve.prompts import SAMPLING, slot_messages
from calm_coder.store.defs import Outcome, emission_to_defs, sha256
from calm_coder.store.store import Store

if TYPE_CHECKING:
    from calm_coder.serve.client import Client
    from calm_coder.task import Task


@dataclass
class Emission:
    task_id: str
    arm: str
    slot: str
    seed: int
    sample_idx: int
    sample_seed: int | None
    text: str
    completion_tokens: int
    prompt_tokens: int
    cached_tokens: int | None
    tokens_estimated: bool
    latency_ms: int
    t_done_ms: int                     # since the task's t_start
    finish_reason: str | None
    extract_error: str | None = None
    fill_hash: str | None = None       # alpha-normalized identity
    fill_hash_exact: str | None = None  # exact-unparse identity (no alpha-renaming)
    def_hashes: list[str] = field(default_factory=list)
    helper_hashes_exact: list[str] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return dict(self.__dict__)


def sample_seed(seed: int, slot_order: int, idx: int) -> int:
    """Seeds are a pure function of (run seed, slot, sample index)."""
    return seed * 10_000 + slot_order * 100 + idx


def ingest(task: "Task", store: Store, e: Emission, on_event: Callable[[dict], None] | None = None) -> None:
    """Extract -> definitions -> store. Extraction failure is recorded as a positive fact."""
    slot = task.slot(e.slot)
    fns = extract_functions(e.text, class_name=task.class_name)
    if isinstance(fns, ExtractError):
        e.extract_error = fns.reason
        cid = sha256(f"extract|{task.task_id}|{e.arm}|{e.slot}|{e.seed}|{e.sample_idx}")
        store.add_outcome(Outcome("__extract__", cid, "error", detail=fns.reason))
        return
    meta = {"agent_id": f"{e.slot}#{e.sample_idx}", "seed": e.sample_seed, "sample_idx": e.sample_idx,
            "arm": e.arm, "ts": time.time(), "completion_tokens": e.completion_tokens}
    defs = emission_to_defs(task, slot, fns, meta, log=e.log)
    exact = emission_to_defs(task, slot, fns, meta, alpha=False)
    if not defs:
        e.extract_error = "no_fill"
        cid = sha256(f"extract|{task.task_id}|{e.arm}|{e.slot}|{e.seed}|{e.sample_idx}")
        store.add_outcome(Outcome("__extract__", cid, "error", detail="no_fill"))
        return
    for d in defs:
        new = store.add_def(d)
        if on_event:
            on_event({"kind": "def", "new": new, "def": d})
    e.def_hashes = [d.hash for d in defs]
    e.fill_hash = next(d.hash for d in defs if d.kind == "fill")
    e.fill_hash_exact = next(d.hash for d in exact if d.kind == "fill")
    e.helper_hashes_exact = [d.hash for d in exact if d.kind == "helper"]


async def generate_fills(client: "Client", task: "Task", store: Store, *, n: int, seed: int,
                         arm: str = "calm", t_start: float | None = None,
                         temperature: float = SAMPLING["temperature"], top_p: float = SAMPLING["top_p"],
                         max_tokens: int = SAMPLING["max_tokens_slot"],
                         on_event: Callable[[dict], None] | None = None) -> list[Emission]:
    """Every slot concurrently; each sample is ingested the moment it returns."""
    t_start = t_start or time.monotonic()
    out: list[Emission] = []

    async def one(slot, idx: int) -> None:
        s_seed = sample_seed(seed, slot.order, idx)
        (s,) = await client.sample(slot_messages(task, slot.id), n=1, temperature=temperature,
                                   top_p=top_p, max_tokens=max_tokens, seed=s_seed)
        e = Emission(task_id=task.task_id, arm=arm, slot=slot.id, seed=seed, sample_idx=idx,
                     sample_seed=s_seed, text=s.text, completion_tokens=s.completion_tokens,
                     prompt_tokens=s.prompt_tokens, cached_tokens=s.cached_tokens,
                     tokens_estimated=s.tokens_estimated, latency_ms=s.latency_ms,
                     t_done_ms=int((time.monotonic() - t_start) * 1000), finish_reason=s.finish_reason)
        ingest(task, store, e, on_event)
        out.append(e)

    # Issue sample 0 of every slot first, then sample 1, ...: nested prefixes (N=1,2,4) finish early.
    await asyncio.gather(*(one(slot, i) for i in range(n) for slot in task.slots))
    return sorted(out, key=lambda e: (task.slot(e.slot).order, e.sample_idx))
