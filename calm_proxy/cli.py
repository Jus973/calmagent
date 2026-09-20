from __future__ import annotations

import sys

from . import lint as lint_cmd
from .__main__ import build_parser, config_from_args
from .server import run as run_server

USAGE = """calm-proxy <command> [options]

commands:
  serve   run the proxy (same flags as `python -m calm_proxy --help`)
  lint    per-session prompt-cache report for a trace directory
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        build_parser().print_help()
        return 0
    command, rest = argv[0], argv[1:]
    if command == "serve":
        run_server(config_from_args(rest))
        return 0
    if command == "lint":
        return lint_cmd.main(rest)
    print(f"calm-proxy: unknown command {command!r}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
