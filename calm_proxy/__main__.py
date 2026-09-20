from __future__ import annotations

import argparse
from pathlib import Path

from .config import KNOWN_MIDDLEWARES, Config, parse_enabled
from .server import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calm_proxy",
        description=(
            "CALM Proxy: an OpenAI-compatible, content-addressed proxy for local "
            "model servers. Point an agent at it instead of the model server."
        ),
    )
    parser.add_argument(
        "--upstream",
        default="http://127.0.0.1:11434",
        help="model server base URL (default: %(default)s)",
    )
    parser.add_argument("--port", type=int, default=8787, help="listen port (default: %(default)s)")
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: %(default)s)")
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="directory for trace.jsonl, bodies/ and memo/ (no tracing if omitted)",
    )
    parser.add_argument(
        "--enable",
        default="trace",
        help=(
            "comma-separated middlewares to enable, from "
            f"{','.join(KNOWN_MIDDLEWARES)} (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dedup-min-bytes",
        type=int,
        default=200,
        help="minimum message size the dedup lever will replace (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="upstream socket read timeout in seconds (default: %(default)s)",
    )
    return parser


def config_from_args(argv: list[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    enabled = parse_enabled(args.enable)
    if args.trace_dir is None and ("trace" in enabled or "memo" in enabled):
        raise SystemExit("--trace-dir is required when trace or memo is enabled")
    return Config(
        upstream=args.upstream,
        port=args.port,
        host=args.host,
        trace_dir=Path(args.trace_dir) if args.trace_dir else None,
        enabled=enabled,
        dedup_min_bytes=args.dedup_min_bytes,
        timeout_s=args.timeout_s,
    )


def main(argv: list[str] | None = None) -> int:
    run(config_from_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
