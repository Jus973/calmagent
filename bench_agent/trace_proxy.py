"""A passthrough proxy that records what an agent actually sends a local model server.

Fallback per the overnight contract: this exists so the bench is not blocked on `calm_proxy`.
It writes the contract's trace record byte-compatibly, so `analysis/proxy_report.py` reads a
trace from either proxy, and it is deleted the moment DV's lands.

    python -m bench_agent.trace_proxy --port 8999 --trace-dir runs/<ts>_probe/

Design notes that are not obvious:

* **Passthrough, not translation.** The request body is relayed to `/v1/chat/completions`
  unchanged, including `stream: true`, and the response bytes are relayed back as they arrive.
  The agent cannot tell the proxy is there. Nothing is buffered, so time-to-first-token survives.

* **TTFT is the cache instrument.** Ollama reports no `cached_tokens`, and `prompt_eval_count`
  does not shrink on a cache hit (`runs/*_cache_probe/`). What does move is the clock: prefill on
  this machine costs ~6 ms per prompt token cold and ~0.06 ms warm, so on a streaming request the
  time to the first token *is* the prefill measurement. `computed_prompt_tokens` below is that
  reading divided by the calibrated cold cost — clearly an estimate, labelled as one, with the
  calibration constant recorded next to it.

* **`--dedup` is the one lever this fallback implements** (I-4), because the probe measured it as
  the only candidate whose kill number was not hit. When a message's content has already appeared
  earlier in the same session, its repeat is replaced by a short reference to the first copy. The
  replacement is keyed by content and is therefore *stable*: the same message is rewritten to the
  same bytes in every later request, so the shared prefix stays aligned. That stability is the
  whole lever -- a rewrite that moved would invalidate the prefix behind it, and at ~6 ms per
  prompt token a single broken 17k-token prefix costs about 100 s, which is more than the lever
  saves over a whole task.

* **The proxy sees both requests, so it knows the achievable prefix.** `prefix.shared_tokens` is
  how much of this request's prompt is a prefix of the previous request in the same session — what
  a perfectly-behaved cache *could* have reused. The gap between that and the reading above is the
  lint's headroom, and it is arithmetic, not a guess.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import pathlib
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request

UPSTREAM = "http://127.0.0.1:11434"
# ms of prefill per prompt token with a cold cache, from bench_agent/probe_cache.py on this
# machine. Overridable; recorded in every trace row so a reader can redo the arithmetic.
COLD_MS_PER_PROMPT_TOKEN = 5.96


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def approx_tokens(text: str) -> int:
    """Characters/4. Only used for the *relative* prefix share, where the bias cancels."""
    return max(1, len(text) // 4)


class Tracer:
    """Owns the trace file, the body store and the per-session previous-request memory."""

    def __init__(self, trace_dir: pathlib.Path, cold_ms: float, dedup: bool = False,
                 dedup_min_bytes: int = 200):
        self.dir = trace_dir
        self.bodies = trace_dir / "bodies"
        self.bodies.mkdir(parents=True, exist_ok=True)
        self.trace = (trace_dir / "trace.jsonl").open("a", buffering=1)
        self.cold_ms = cold_ms
        self.lock = threading.Lock()
        self.seq = 0
        self.prev: dict[str, list[dict]] = {}   # session -> previous request's messages
        self.dedup = dedup
        self.dedup_min_bytes = dedup_min_bytes
        # session -> content sha -> the label the first copy was given. Grow-only and keyed by
        # content, which is what makes every rewrite reproducible across turns.
        self.seen_content: dict[str, dict[str, int]] = {}

    def apply_dedup(self, session: str, messages: list[dict]) -> tuple[list[dict], int, int]:
        """Replace repeated message content with a stable reference to its first occurrence."""
        if not self.dedup:
            return messages, 0, 0
        first = self.seen_content.setdefault(session, {})
        out, replaced, saved = [], 0, 0
        for i, m in enumerate(messages):
            content = m.get("content")
            if i == 0 or not isinstance(content, str) or len(content) < self.dedup_min_bytes:
                out.append(m)
                continue
            h = sha(content)
            # `first` is keyed by content and remembers WHERE the first copy sits. The index test
            # matters: every turn re-sends the whole transcript, so without it the original copy
            # would be rewritten into a reference to itself the moment it was seen a second time,
            # and the prefix would move under the cache on every single turn.
            if h not in first:
                first[h] = i
            if first[h] != i:
                ref = (f"[identical to the content of message #{first[h]} earlier in this "
                       f"conversation, sha256:{h[:16]}]")
                replaced += 1
                saved += len(content) - len(ref)
                out.append({**m, "content": ref})
            else:
                out.append(m)
        return out, replaced, saved

    def store_body(self, text: str) -> str:
        h = sha(text)
        p = self.bodies / f"{h}.txt"
        if not p.exists():
            p.write_text(text, encoding="utf-8", errors="replace")
        return h

    def prefix_against_previous(self, session: str, messages: list[dict]) -> dict:
        """How much of this prompt a perfect cache could have reused from the last request."""
        prev = self.prev.get(session)
        self.prev[session] = messages
        total = sum(approx_tokens(str(m.get("content", ""))) for m in messages)
        if prev is None:
            return {"shared_tokens": 0, "total_tokens": total, "first_divergent_message": 0,
                    "first_divergent_byte": 0, "cause": "first_request_in_session"}
        shared = 0
        idx = 0
        for i, (a, b) in enumerate(zip(prev, messages)):
            ca, cb = str(a.get("content", "")), str(b.get("content", ""))
            if a.get("role") == b.get("role") and ca == cb:
                shared += approx_tokens(ca)
                idx = i + 1
                continue
            # partial overlap inside the first message that differs
            n = 0
            for n in range(min(len(ca), len(cb))):
                if ca[n] != cb[n]:
                    break
            else:
                n = min(len(ca), len(cb))
            shared += n // 4
            return {"shared_tokens": shared, "total_tokens": total, "first_divergent_message": i,
                    "first_divergent_byte": n, "cause": classify(prev, messages, i, n)}
        return {"shared_tokens": shared, "total_tokens": total, "first_divergent_message": idx,
                "first_divergent_byte": 0, "cause": "appended_only"}

    def write(self, row: dict) -> None:
        with self.lock:
            self.seq += 1
            row["seq"] = self.seq
            self.trace.write(json.dumps(row) + "\n")


def classify(prev: list[dict], cur: list[dict], i: int, byte: int) -> str:
    """Why the prefix broke. Coarse on purpose; the lint refines it."""
    role = cur[i].get("role") if i < len(cur) else "?"
    if i == 0:
        return "system_prompt_churn"
    if len(cur) < len(prev):
        return "compaction"
    if role == "tool":
        return "tool_result_change"
    if byte < 200:
        return f"{role}_message_head_change"
    return f"{role}_message_change"


def make_handler(tracer: Tracer, upstream: str):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _relay(self, path: str, body: bytes, headers: dict):
            req = urllib.request.Request(upstream + path, data=body, headers=headers,
                                         method="POST" if body is not None else "GET")
            return urllib.request.urlopen(req, timeout=1800)

        def do_GET(self):
            try:
                resp = self._relay(self.path, None, {})
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:  # noqa: BLE001
                self.send_error(502, str(exc))

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if not self.path.startswith("/v1/chat/completions"):
                try:
                    resp = self._relay(self.path, raw, {"Content-Type": "application/json"})
                    data = resp.read()
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as exc:  # noqa: BLE001
                    self.send_error(502, str(exc))
                return

            t_submit = time.time()
            try:
                payload = json.loads(raw)
            except Exception:  # noqa: BLE001
                self.send_error(400, "bad json")
                return

            messages = payload.get("messages") or []
            stream = bool(payload.get("stream"))
            messages, dedup_replaced, dedup_saved = tracer.apply_dedup(
                sha(str(messages[0].get("content", "")) if messages else "empty")[:16], messages)
            if dedup_replaced:
                payload["messages"] = messages
                raw = json.dumps(payload).encode()
            # Ollama omits `usage` from a streamed response unless asked. Asking is additive and
            # touches nothing in the prompt, so it cannot move the cache or the model's output --
            # it just means a streaming agent is accounted for as well as a blocking one.
            injected_usage = False
            if stream and not payload.get("stream_options", {}).get("include_usage"):
                payload.setdefault("stream_options", {})["include_usage"] = True
                raw = json.dumps(payload).encode()
                injected_usage = True
            session = sha(str(messages[0].get("content", "")) if messages else "empty")[:16]
            msg_rows = []
            for m in messages:
                content = str(m.get("content", ""))
                h = tracer.store_body(content)
                row = {"role": m.get("role"), "sha": h, "bytes": len(content)}
                if m.get("tool_call_id"):
                    row["tool_call_id"] = m["tool_call_id"]
                msg_rows.append(row)
            prompt_sha = sha(json.dumps(msg_rows, sort_keys=True))
            prefix = tracer.prefix_against_previous(session, messages)

            record = {
                "ts": t_submit, "session": session, "model": payload.get("model"),
                "stream": stream,
                "params": {"temperature": payload.get("temperature"),
                           "seed": payload.get("seed"),
                           "max_tokens": payload.get("max_tokens")},
                "messages": msg_rows, "prompt_sha": prompt_sha,
                "prefix": prefix,
                "memo": {"hit": False, "key": prompt_sha},
                "dedup": {"replaced": dedup_replaced, "bytes_saved": dedup_saved},
                "cold_ms_per_prompt_token": tracer.cold_ms,
                "proxy": {"injected_stream_usage": injected_usage},
            }

            try:
                resp = self._relay(self.path, raw, {"Content-Type": "application/json"})
            except urllib.error.HTTPError as exc:
                record.update(upstream_status=exc.code, error=exc.read()[:500].decode("utf-8", "replace"),
                              usage=None, timing_ms={"submit": 0, "first_token": None, "done": 0})
                tracer.write(record)
                self.send_error(exc.code, "upstream error")
                return
            except Exception as exc:  # noqa: BLE001
                record.update(upstream_status=None, error=str(exc)[:500], usage=None,
                              timing_ms={"submit": 0, "first_token": None, "done": 0})
                tracer.write(record)
                self.send_error(502, str(exc))
                return

            self.send_response(resp.status)
            ctype = resp.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            first_token_ms = None
            usage = None
            completion_chars = 0
            collected = bytearray()
            try:
                while True:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                    if not chunk:
                        break
                    if first_token_ms is None:
                        first_token_ms = (time.time() - t_submit) * 1000
                    collected += chunk
                    self.wfile.write(f"{len(chunk):X}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"relay interrupted: {exc}"

            done_ms = (time.time() - t_submit) * 1000
            text = collected.decode("utf-8", "replace")
            if stream:
                for line in text.splitlines():
                    if line.startswith("data: ") and '"usage"' in line:
                        try:
                            d = json.loads(line[6:])
                            if d.get("usage"):
                                usage = d["usage"]
                        except Exception:  # noqa: BLE001
                            pass
                completion_chars = text.count('"content":')
            else:
                try:
                    d = json.loads(text)
                    usage = d.get("usage")
                except Exception:  # noqa: BLE001
                    pass

            pt = (usage or {}).get("prompt_tokens")
            # On a streamed request the first token arrives when prefill ends, so TTFT is the
            # prefill reading. On a non-streamed one there is no such boundary and we say so.
            if stream and first_token_ms is not None:
                ceiling = pt if pt is not None else prefix["total_tokens"]
                computed = min(ceiling, round(first_token_ms / tracer.cold_ms))
                basis = "ttft" if pt is not None else "ttft_prompt_tokens_estimated"
            else:
                computed, basis = None, "unavailable_non_streaming"
            record.update(
                usage={"prompt_tokens": pt,
                       "cached_tokens": (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens")
                       if isinstance((usage or {}).get("prompt_tokens_details"), dict) else None,
                       "completion_tokens": (usage or {}).get("completion_tokens")},
                timing_ms={"submit": 0,
                           "first_token": round(first_token_ms, 1) if first_token_ms else None,
                           "done": round(done_ms, 1)},
                estimate={"computed_prompt_tokens": computed, "basis": basis,
                          "achievable_shared_tokens": prefix["shared_tokens"]},
                upstream_status=resp.status,
                error=record.get("error"),
            )
            tracer.write(record)

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8999)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--cold-ms-per-prompt-token", type=float, default=COLD_MS_PER_PROMPT_TOKEN)
    ap.add_argument("--dedup", action="store_true",
                    help="I-4: replace a repeated message with a stable reference to its first copy")
    ap.add_argument("--dedup-min-bytes", type=int, default=200)
    args = ap.parse_args()

    tracer = Tracer(pathlib.Path(args.trace_dir), args.cold_ms_per_prompt_token,
                    dedup=args.dedup, dedup_min_bytes=args.dedup_min_bytes)
    srv = Server(("127.0.0.1", args.port), make_handler(tracer, args.upstream))
    print(f"trace proxy on http://127.0.0.1:{args.port} -> {args.upstream}, "
          f"trace -> {args.trace_dir}/trace.jsonl, dedup={args.dedup}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
