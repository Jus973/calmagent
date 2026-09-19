"""Baselines (§3.13). C = N whole-class samples, oracle-verified. A = pass@1 estimated from C's
samples (unbiased c/n) plus one greedy sample reported as A_greedy. B (incremental) is cut (§8)."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from calm_coder.runner.sandbox import run_tests_async
from calm_coder.serve.extract import ExtractError, extract_class
from calm_coder.serve.prompts import SAMPLING, whole_class_messages

if TYPE_CHECKING:
    from calm_coder.serve.client import Client
    from calm_coder.task import Task


@dataclass
class ClassSample:
    task_id: str
    arm: str
    seed: int
    sample_idx: int
    sample_seed: int | None
    text: str
    completion_tokens: int
    prompt_tokens: int
    cached_tokens: int | None
    tokens_estimated: bool
    latency_ms: int
    t_done_ms: int
    finish_reason: str | None
    extract_error: str | None = None
    results: dict[str, str] = field(default_factory=dict)
    passed: bool = False
    test_ms: int = 0

    def to_json(self) -> dict:
        return dict(self.__dict__)


def class_seed(seed: int, idx: int) -> int:
    return seed * 10_000 + 9_000 + idx


async def whole_class_samples(client: "Client", task: "Task", *, n: int, seed: int, arm: str = "c",
                              temperature: float = SAMPLING["temperature"], top_p: float | None = SAMPLING["top_p"],
                              max_tokens: int = SAMPLING["max_tokens_class"], per_class_timeout_s: float = 5,
                              wall_timeout_s: float = 20) -> list[ClassSample]:
    """Each sample is tested the moment it arrives (so TTFV is time-to-first-passing-sample)."""
    t_start = time.monotonic()

    async def one(idx: int) -> ClassSample:
        s_seed = class_seed(seed, idx)
        (s,) = await client.sample(whole_class_messages(task), n=1, temperature=temperature, top_p=top_p,
                                   max_tokens=max_tokens, seed=s_seed)
        cs = ClassSample(task_id=task.task_id, arm=arm, seed=seed, sample_idx=idx, sample_seed=s_seed, text=s.text,
                         completion_tokens=s.completion_tokens, prompt_tokens=s.prompt_tokens,
                         cached_tokens=s.cached_tokens, tokens_estimated=s.tokens_estimated,
                         latency_ms=s.latency_ms, t_done_ms=int((time.monotonic() - t_start) * 1000),
                         finish_reason=s.finish_reason)
        src = extract_class(s.text, task)
        if isinstance(src, ExtractError):
            cs.extract_error = src.reason
            return cs
        t0 = time.monotonic()
        res = await run_tests_async(src, task.test_src, list(task.test_classes),
                                    per_class_timeout_s=per_class_timeout_s, wall_timeout_s=wall_timeout_s)
        cs.test_ms = int((time.monotonic() - t0) * 1000)
        cs.results = {t: r["result"] for t, r in res.items()}
        cs.passed = all(cs.results.get(t) == "pass" for t in task.test_classes)
        return cs

    return list(await asyncio.gather(*(one(i) for i in range(n))))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1 - k / i
    return 1 - p
