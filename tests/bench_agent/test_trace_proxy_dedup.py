"""The dedup lever's correctness property is *stability*, so that is what these test.

A rewrite that moved between turns would invalidate the shared prefix behind it. At the measured
~6 ms per prompt token, re-prefilling a 17k-token transcript costs about 100 s -- more than dedup
saves across a whole task. So "the same message is always rewritten to the same bytes" is not a
nicety, it is the difference between a lever and a regression.
"""
from __future__ import annotations

import pathlib

import pytest

from bench_agent.trace_proxy import Tracer


@pytest.fixture()
def tracer(tmp_path: pathlib.Path) -> Tracer:
    return Tracer(tmp_path, cold_ms=5.96, dedup=True, dedup_min_bytes=50)


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


LONG = "the same pytest output, repeated verbatim by an agent that re-ran the suite. " * 4
OTHER = "a different command's output entirely, also long enough to be worth replacing. " * 4


def test_first_occurrence_is_untouched(tracer: Tracer) -> None:
    out, replaced, saved = tracer.apply_dedup("s", [msg("system", "sys"), msg("user", LONG)])
    assert replaced == 0 and saved == 0
    assert out[1]["content"] == LONG


def test_repeat_is_replaced_by_a_reference(tracer: Tracer) -> None:
    conv = [msg("system", "sys"), msg("user", LONG)]
    tracer.apply_dedup("s", conv)
    out, replaced, saved = tracer.apply_dedup("s", conv + [msg("assistant", "ok"), msg("user", LONG)])
    assert replaced == 1
    assert saved > 0
    assert out[1]["content"] == LONG, "the first copy must survive"
    assert out[3]["content"].startswith("[identical to the content of message #1")


def test_rewrite_is_stable_across_turns(tracer: Tracer) -> None:
    """The same message must come out byte-identical every turn, or the prefix breaks."""
    conv = [msg("system", "sys"), msg("user", LONG)]
    tracer.apply_dedup("s", conv)
    conv = conv + [msg("assistant", "a"), msg("user", LONG)]
    first_pass, _, _ = tracer.apply_dedup("s", conv)
    rendered = [m["content"] for m in first_pass]
    for turn in range(5):
        conv = conv + [msg("assistant", f"step {turn}"), msg("user", OTHER + str(turn))]
        out, _, _ = tracer.apply_dedup("s", conv)
        assert [m["content"] for m in out][: len(rendered)] == rendered, (
            f"turn {turn} rewrote the shared prefix differently; the cache would be invalidated"
        )


def test_prefix_only_ever_grows(tracer: Tracer) -> None:
    """Across a whole session, each request's output must extend the previous one's."""
    conv = [msg("system", "sys")]
    previous: list[str] = []
    for turn in range(8):
        conv = conv + [msg("assistant", f"cmd {turn}"),
                       msg("user", LONG if turn % 2 else OTHER + str(turn))]
        out, _, _ = tracer.apply_dedup("s", conv)
        rendered = [m["content"] for m in out]
        assert rendered[: len(previous)] == previous, f"turn {turn} broke the prefix"
        previous = rendered


def test_short_messages_are_left_alone(tracer: Tracer) -> None:
    conv = [msg("system", "sys"), msg("user", "hi"), msg("assistant", "a"), msg("user", "hi")]
    out, replaced, _ = tracer.apply_dedup("s", conv)
    assert replaced == 0
    assert out == conv


def test_sessions_do_not_share_a_dedup_memory(tracer: Tracer) -> None:
    tracer.apply_dedup("s1", [msg("system", "sys"), msg("user", LONG)])
    out, replaced, _ = tracer.apply_dedup("s2", [msg("system", "sys"), msg("user", LONG)])
    assert replaced == 0
    assert out[1]["content"] == LONG


def test_dedup_off_is_a_no_op(tmp_path: pathlib.Path) -> None:
    t = Tracer(tmp_path, cold_ms=5.96, dedup=False)
    conv = [msg("system", "sys"), msg("user", LONG), msg("assistant", "a"), msg("user", LONG)]
    out, replaced, saved = t.apply_dedup("s", conv)
    assert (out, replaced, saved) == (conv, 0, 0)


# --- conversation boundaries ---------------------------------------------------------------
#
# A session id is the hash of the first system message, and an agent framework sends the same
# system message for every task it runs. With a long-lived proxy that puts ten tasks under one id,
# and the dedup memory is keyed by it -- so without a boundary, task 2's first sight of a string
# gets rewritten into a reference to a message number that only existed in task 1.

SHARED_SYSTEM = "You are a helpful assistant that can interact with a computer. " * 8


def test_a_shorter_message_list_starts_a_new_conversation(tracer: Tracer) -> None:
    long_conv = [msg("system", SHARED_SYSTEM)] + [msg("user", f"turn {i}" * 40) for i in range(6)]
    k1 = tracer.conversation_key("sess", long_conv)
    tracer.prev["sess"] = long_conv
    k2 = tracer.conversation_key("sess", [msg("system", SHARED_SYSTEM), msg("user", "fresh task")])
    assert k1 != k2


def test_a_new_task_does_not_inherit_the_previous_task_s_dedup_memory(tracer: Tracer) -> None:
    """The bug this guards: content first seen in task 2 must not be replaced by a reference."""
    task1 = [msg("system", SHARED_SYSTEM), msg("user", "a"), msg("assistant", "b"),
             msg("user", LONG)]
    k1 = tracer.conversation_key("sess", task1)
    tracer.apply_dedup(k1, task1)
    tracer.prev["sess"] = task1

    task2 = [msg("system", SHARED_SYSTEM), msg("user", LONG)]
    k2 = tracer.conversation_key("sess", task2)
    out, replaced, _ = tracer.apply_dedup(k2, task2)
    assert replaced == 0, "task 2's first sight of this content was deleted"
    assert out[1]["content"] == LONG


def test_an_extending_request_stays_in_the_same_conversation(tracer: Tracer) -> None:
    conv = [msg("system", SHARED_SYSTEM), msg("user", "a")]
    k1 = tracer.conversation_key("sess", conv)
    tracer.prev["sess"] = conv
    conv = conv + [msg("assistant", "b"), msg("user", "c")]
    assert tracer.conversation_key("sess", conv) == k1


def test_a_changed_first_message_starts_a_new_conversation(tracer: Tracer) -> None:
    conv = [msg("system", SHARED_SYSTEM), msg("user", "a"), msg("assistant", "b")]
    k1 = tracer.conversation_key("sess", conv)
    tracer.prev["sess"] = conv
    k2 = tracer.conversation_key("sess", [msg("system", "a different system prompt"),
                                          msg("user", "a"), msg("assistant", "b")])
    assert k1 != k2
