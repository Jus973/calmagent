"""Replay recorded agent request bodies through the whole proxy.

The fixtures under ``fixtures/probe_bodies/`` are the shapes LC's harness
sends. Point ``CALM_PROBE_BODIES`` at a directory of real recorded bodies
(one JSON request per file) to replay those instead — same assertions, no
model required, so a recording from a real run is a regression test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiohttp
import pytest

from .harness import post, run, sse_chunks, stable, stack, trace_records

FIXTURES = Path(__file__).parent / "fixtures" / "probe_bodies"
PROBE_DIR = Path(os.environ.get("CALM_PROBE_BODIES") or FIXTURES)
BODIES = sorted(PROBE_DIR.glob("*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_for(body: dict) -> str:
    return "/v1/completions" if "prompt" in body else "/v1/chat/completions"


@pytest.mark.skipif(not BODIES, reason=f"no recorded bodies in {PROBE_DIR}")
@pytest.mark.parametrize("probe", BODIES, ids=lambda p: p.stem)
def test_recorded_body_replays_identically(probe: Path, tmp_path: Path):
    body = _load(probe)
    path = _path_for(body)

    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix,dedup,memo") as st:
            if body.get("stream"):
                async with aiohttp.ClientSession() as http:
                    async with http.post(
                        st.upstream_url + path, json=body, headers={"X-Calm-Session": "d"}
                    ) as direct:
                        plain = sse_chunks(await direct.read())
                    async with http.post(
                        st.proxy_url + path, json=body, headers={"X-Calm-Session": "p"}
                    ) as via:
                        assert via.status == 200
                        proxied = sse_chunks(await via.read())
                return plain, proxied
            _, plain, _ = await post(
                st.upstream_url, body, path=path, headers={"X-Calm-Session": "d"}
            )
            status, proxied, _ = await post(
                st.proxy_url, body, path=path, headers={"X-Calm-Session": "p"}
            )
            assert status == 200
            return plain, proxied

    plain, proxied = run(go())
    if isinstance(plain, list):  # streamed
        assert [c for c in plain if c != "[DONE]"] != [], "the probe must produce a stream"
        assert len(plain) == len(proxied)
        assert plain[-1] == proxied[-1] == "[DONE]"
    else:
        assert stable(plain)["choices"] == stable(proxied)["choices"]

    record = trace_records(tmp_path)[0]
    assert record["upstream_status"] == 200
    assert record["error"] is None
    assert record["usage"]["prompt_tokens"]


@pytest.mark.skipif(not BODIES, reason=f"no recorded bodies in {PROBE_DIR}")
def test_a_whole_recorded_session_replays_in_order(tmp_path: Path):
    """All probes in one session: the levers have to survive each other."""

    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix,dedup,memo") as st:
            async with aiohttp.ClientSession() as http:
                for probe in BODIES:
                    body = _load(probe)
                    async with http.post(
                        st.proxy_url + _path_for(body),
                        json=body,
                        headers={"X-Calm-Session": "replay"},
                    ) as resp:
                        assert resp.status == 200, probe.name
                        await resp.read()

    run(go())
    records = trace_records(tmp_path)
    assert [r["seq"] for r in records] == list(range(len(BODIES)))
    assert all(r["error"] is None for r in records)
    assert any(r["dedup"]["replaced"] for r in records), "repeated tool results must dedup"
