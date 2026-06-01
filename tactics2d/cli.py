# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cli implementation."""

import argparse
import json
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


def _status(args):
    from tactics2d.frontend import FrontendRenderer

    renderer = FrontendRenderer(args.host, args.port)
    print(json.dumps(renderer.health(), indent=2, sort_keys=True))


def _preview(args):
    from tactics2d.frontend import FrontendRenderer, run_server
    from tactics2d.frontend.renderer import start_server_process

    if args.preview_command == "demo":
        run_server(
            args.host, args.port, demo=True, max_fps=args.max_fps, open_browser=args.open_browser
        )
    elif args.preview_command == "map":
        renderer = FrontendRenderer(args.host, args.port)
        if not renderer.wait_until_ready(timeout=0.5):
            start_server_process(
                args.host,
                args.port,
                demo=False,
                max_fps=args.max_fps,
                open_browser=args.open_browser,
            )
            if not renderer.wait_until_ready(timeout=5.0):
                raise RuntimeError(f"Tactics2D frontend did not start on {renderer.base_url}.")

        renderer.send_frame(
            [_build_map_preview_sensor(args.osm, args.lanelet2)],
            frame=0,
            layout="grid",
            wait_ack=False,
            drop_if_busy=False,
        )
        LOGGER.info("Previewing %s at %s", args.osm, renderer.base_url)
    else:
        raise SystemExit("Please choose a preview target, for example `tactics2d preview demo`.")


def _build_map_preview_sensor(osm_path: Path, lanelet2: bool) -> dict:
    from shapely.geometry import Point

    from tactics2d.map.parser import OSMParser
    from tactics2d.sensor import BEVCamera

    map_ = OSMParser(lanelet2=lanelet2).parse(str(osm_path))
    camera = BEVCamera(id_=0, map_=map_)
    x_center = 0.5 * (map_.boundary[0] + map_.boundary[1])
    y_center = 0.5 * (map_.boundary[2] + map_.boundary[3])
    geometry_data, _, _ = camera.update(0, {}, [], set(), set(), Point(x_center, y_center), 0)

    coords = [
        point
        for element in geometry_data["map_data"]["road_elements"]
        for point in element["geometry"]
    ]
    if coords:
        x_values = [point[0] for point in coords]
        y_values = [point[1] for point in coords]
        x_center = 0.5 * (min(x_values) + max(x_values))
        y_center = 0.5 * (min(y_values) + max(y_values))
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)
    else:
        x_span = map_.boundary[1] - map_.boundary[0]
        y_span = map_.boundary[3] - map_.boundary[2]

    preview_range = max(20.0, min(max(x_span, y_span) / 2, max(y_span * 3, x_span / 8)))

    return {
        "id": f"map-{osm_path.stem}",
        "perception_range": float(preview_range),
        "position": [x_center, y_center],
        "yaw": 0,
        "frame": 0,
        "map_data": geometry_data["map_data"],
        "participant_data": geometry_data["participant_data"],
    }


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
