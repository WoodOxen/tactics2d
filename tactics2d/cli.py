# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cli implementation."""

import argparse
import json
import logging
from pathlib import Path

import tactics2d
from tactics2d.display.renderers.web.preview import LEVELX_DATASETS

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
    start_parser.add_argument("--open", action="store_true", dest="open_browser", default=True)
    start_parser.add_argument("--no-open", action="store_false", dest="open_browser")
    start_parser.add_argument("--background", action="store_true")
    start_parser.add_argument("--pid-file", type=Path, default=None)

    stop_parser = subparsers.add_parser("stop", help="Stop a background frontend.")
    stop_parser.add_argument("--pid-file", type=Path, default=None)

    status_parser = subparsers.add_parser("status", help="Show frontend server status.")
    status_parser.add_argument("--host", default="127.0.0.1")
    status_parser.add_argument("--port", type=int, default=8765)

    preview_parser = subparsers.add_parser("preview", help="Preview frontend scenes.")
    preview_subparsers = preview_parser.add_subparsers(dest="preview_command")
    demo_parser = preview_subparsers.add_parser("demo", help="Preview the built-in demo scene.")
    demo_parser.add_argument("--host", default="127.0.0.1")
    demo_parser.add_argument("--port", type=int, default=8765)
    demo_parser.add_argument("--max-fps", type=int, default=30)
    demo_parser.add_argument("--open", action="store_true", dest="open_browser", default=True)
    demo_parser.add_argument("--no-open", action="store_false", dest="open_browser")

    map_parser = preview_subparsers.add_parser("map", help="Preview an OSM map file.")
    map_parser.add_argument("osm", type=Path)
    map_parser.add_argument("--host", default="127.0.0.1")
    map_parser.add_argument("--port", type=int, default=8765)
    map_parser.add_argument("--max-fps", type=int, default=30)
    map_parser.add_argument("--open", action="store_true", dest="open_browser", default=True)
    map_parser.add_argument("--no-open", action="store_false", dest="open_browser")
    map_parser.add_argument("--lanelet2", action="store_true", default=True)
    map_parser.add_argument("--plain-osm", action="store_false", dest="lanelet2")
    map_parser.add_argument("--map-config", default=None)

    dataset_parser = preview_subparsers.add_parser("dataset", help="Preview a dataset scene.")
    dataset_parser.add_argument("--dataset", choices=LEVELX_DATASETS, required=True)
    dataset_parser.add_argument("--folder", type=Path, required=True)
    dataset_parser.add_argument("--file", required=True)
    dataset_parser.add_argument("--osm", type=Path, default=None)
    dataset_parser.add_argument("--map-config", default=None)
    dataset_parser.add_argument("--host", default="127.0.0.1")
    dataset_parser.add_argument("--port", type=int, default=8765)
    dataset_parser.add_argument("--max-fps", type=int, default=30)
    dataset_parser.add_argument("--open", action="store_true", dest="open_browser", default=True)
    dataset_parser.add_argument("--no-open", action="store_false", dest="open_browser")
    dataset_parser.add_argument("--lanelet2", action="store_true", default=True)
    dataset_parser.add_argument("--plain-osm", action="store_false", dest="lanelet2")
    dataset_parser.add_argument("--frames", type=int, default=300)
    dataset_parser.add_argument("--start-time-ms", type=int, default=None)
    dataset_parser.add_argument("--ids", type=int, nargs="+", default=None)
    dataset_parser.add_argument("--follow-id", type=int, default=None)
    dataset_parser.add_argument("--perception-range", type=float, default=80.0)
    dataset_parser.add_argument("--loop", action="store_true")

    return parser.parse_args(argv)


def _start(args):
    from tactics2d.display.renderers.web import FrontendRenderer, run_server
    from tactics2d.display.renderers.web.renderer import start_server_process

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
    from tactics2d.display.renderers.web.renderer import stop_server_process

    pid = stop_server_process(args.pid_file)
    LOGGER.info("Stopped Tactics2D frontend process %s.", pid)


def _status(args):
    from tactics2d.display.renderers.web import FrontendRenderer

    renderer = FrontendRenderer(args.host, args.port)
    print(json.dumps(renderer.health(), indent=2, sort_keys=True))


def _preview(args):
    from tactics2d.display.renderers.web.preview import ensure_frontend_server

    if args.preview_command == "demo":
        renderer = ensure_frontend_server(
            args.host, args.port, args.max_fps, open_browser=args.open_browser
        )
        renderer.preview_demo(max_fps=args.max_fps)
        LOGGER.info("Previewing demo scene at %s.", renderer.base_url)
    elif args.preview_command == "map":
        renderer = ensure_frontend_server(
            args.host, args.port, args.max_fps, open_browser=args.open_browser
        )

        renderer.preview_map(args.osm, lanelet2=args.lanelet2, map_config=args.map_config)
        LOGGER.info("Previewing %s at %s", args.osm, renderer.base_url)
    elif args.preview_command == "dataset":
        renderer = ensure_frontend_server(
            args.host, args.port, args.max_fps, open_browser=args.open_browser
        )
        renderer.preview_dataset(
            dataset=args.dataset,
            folder=args.folder,
            file=args.file,
            osm_path=args.osm,
            map_config=args.map_config,
            max_fps=args.max_fps,
            lanelet2=args.lanelet2,
            frames=args.frames,
            start_time_ms=args.start_time_ms,
            ids=args.ids,
            follow_id=args.follow_id,
            perception_range=args.perception_range,
            loop=args.loop,
        )
        LOGGER.info("Previewing %s recording %s at %s.", args.dataset, args.file, renderer.base_url)
    else:
        raise SystemExit("Please choose a preview target, for example `tactics2d preview demo`.")


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    if args.command == "start":
        _start(args)
    elif args.command == "stop":
        _stop(args)
    elif args.command == "status":
        _status(args)
    elif args.command == "preview":
        _preview(args)
    else:
        parse_args(["--help"])


if __name__ == "__main__":
    main()
