# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cli implementation."""

import argparse
import logging
from pathlib import Path

import tactics2d

LOGGER = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tactics2D Command Line Tool")
    parser.add_argument("--version", action="version", version=f"tactics2d {tactics2d.__version__}")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start the browser frontend.")
    start_parser.add_argument("--host", default="127.0.0.1")
    start_parser.add_argument("--port", type=int, default=8765)
    start_parser.add_argument("--max-fps", type=int, default=30)
    start_parser.add_argument("--demo", action="store_true")
    start_parser.add_argument("--open", action="store_true", dest="open_browser")
    start_parser.add_argument("--background", action="store_true")
    start_parser.add_argument("--pid-file", type=Path, default=None)

    stop_parser = subparsers.add_parser("stop", help="Stop a background frontend.")
    stop_parser.add_argument("--pid-file", type=Path, default=None)

    return parser.parse_args(argv)


def _start(args):
    from tactics2d.frontend import FrontendRenderer, run_server
    from tactics2d.frontend.renderer import start_server_process

    if args.background:
        process = start_server_process(
            args.host, args.port, args.demo, args.max_fps, args.open_browser, args.pid_file
        )
        renderer = FrontendRenderer(args.host, args.port)
        if renderer.wait_until_ready(timeout=5.0):
            LOGGER.info("Tactics2D frontend started at http://%s:%s", args.host, args.port)
        else:
            LOGGER.warning("Frontend process %s started, but health check timed out.", process.pid)
        return

    run_server(args.host, args.port, args.demo, args.max_fps, args.open_browser)


def _stop(args):
    from tactics2d.frontend.renderer import stop_server_process

    pid = stop_server_process(args.pid_file)
    LOGGER.info("Stopped Tactics2D frontend process %s.", pid)


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    if args.command == "start":
        _start(args)
    elif args.command == "stop":
        _stop(args)
    else:
        parse_args(["--help"])


if __name__ == "__main__":
    main()
