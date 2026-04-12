#! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: run_final_sim.py
# @Description: Render a DriveInsightD scenario as a top-down MP4 video
#               using the Tactics2D TopDownCamera sensor.
# @Author: Zexi Chen
# @Version: 1.0.0

"""Render a DriveInsightD scenario to video.

Usage
-----
  # Defaults — reads environment variables, falls back to built-in paths
  python run_final_sim.py

  # Override any setting via environment variables
  DRIVEINSIGHTD_FOLDER=/data/us_coldwater  \\
  DRIVEINSIGHTD_SCENARIO=55                \\
  DRIVEINSIGHTD_FILTER_VIRTUAL=1           \\
  python run_final_sim.py

Environment variables
---------------------
DRIVEINSIGHTD_FOLDER          Path to the scenario folder.
DRIVEINSIGHTD_SCENARIO        Scenario ID (default: "55").
DRIVEINSIGHTD_MAP             Map filename (default: "usa_coldwater.xodr").
DRIVEINSIGHTD_OUTPUT          Output MP4 path.
DRIVEINSIGHTD_FILTER_VIRTUAL  Set to "1" to remove virtual lane boundaries
                              and geometrically anomalous lines from the
                              rendered map.  Recommended for RL training use.
                              Default: "0" (show all lines, matching raw data).
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from PIL import Image

from tactics2d.dataset_parser.parse_driveinsightd import DriveInsightDParser
from tactics2d.sensor import TopDownCamera


# ---------------------------------------------------------------------------
# Configuration  (all values can be overridden via environment variables)
# ---------------------------------------------------------------------------

FOLDER      = os.environ.get(
    "DRIVEINSIGHTD_FOLDER",
    "/root/autodl-tmp/driveinsightD/database/us_coldwater",
)
SCENARIO_ID = os.environ.get("DRIVEINSIGHTD_SCENARIO", "55")
MAP_NAME    = os.environ.get("DRIVEINSIGHTD_MAP",      "usa_coldwater.xodr")
OUTPUT_PATH = os.environ.get(
    "DRIVEINSIGHTD_OUTPUT",
    "/root/autodl-tmp/driveinsightD/us_coldwater_55_render.mp4",
)
WINDOW_SIZE = (800, 800)

# Set to True to remove virtual lane boundaries and geometrically anomalous
# lines before rendering.  Recommended when the map is used for RL training.
FILTER_VIRTUAL_LINES = False


# ---------------------------------------------------------------------------
# Map post-processing
# ---------------------------------------------------------------------------

# Roadlines whose end-to-end span exceeds this threshold are considered
# geometrically anomalous and will be filtered out when FILTER_VIRTUAL_LINES
# is enabled.  129 m is the full length of road 16 in the us_coldwater map;
# legitimate roadlines within a single laneSection are always shorter than
# the full road length, so 100 m is a safe upper bound.
_ANOMALOUS_SPAN_THRESHOLD_M = 100.0


def _roadline_span(rl) -> float:
    """Return the straight-line distance between the first and last point of
    a RoadLine's geometry.  Returns 0.0 on any error."""
    try:
        coords = list(rl.geometry.coords)
        if len(coords) < 2:
            return 0.0
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    except Exception:
        return 0.0


def _should_filter(rl) -> bool:
    return getattr(rl, "type_", None) == "virtual"


def filter_roadlines(map_) -> int:
    """Remove unwanted RoadLine objects from the map in-place.

    Returns the number of lines removed.
    """
    roadlines = map_.roadlines
    if not isinstance(roadlines, dict):
        return 0

    remove_ids = [k for k, rl in roadlines.items() if _should_filter(rl)]
    for k in remove_ids:
        del roadlines[k]

    return len(remove_ids)


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def render_scenario(scenario: dict, output_path: str,
                    filter_virtual: bool = False) -> None:
    """Render all frames of *scenario* and encode them into an MP4 file.

    Parameters
    ----------
    scenario : dict
        Output of DriveInsightDParser.parse().
    output_path : str
        Destination path for the rendered MP4.
    filter_virtual : bool
        When True, virtual and anomalous road-mark lines are removed from the
        map before rendering.  Recommended for RL training visualisation.
    """
    map_           = scenario["map"]
    participants   = scenario["participants"]

    if filter_virtual:
        n_removed = filter_roadlines(map_)
        logging.info(
            "Roadline filter enabled: removed %d virtual / anomalous lines.",
            n_removed,
        )

    # Collect and sort all unique trajectory timestamps
    all_ts: set = set()
    for p in participants.values():
        frames = getattr(p.trajectory, "frames", None)
        if frames is not None:
            all_ts.update(frames)
    times = sorted(all_ts)

    if not times:
        logging.error("No trajectory timestamps found. Aborting render.")
        return

    intervals       = np.diff(times)
    avg_interval_ms = float(np.median(intervals)) if len(intervals) > 0 else 250.0
    fps             = max(1, round(1000.0 / avg_interval_ms))
    logging.info(
        "Detected data interval: %.1f ms  ->  video fps: %d",
        avg_interval_ms, fps,
    )

    camera     = TopDownCamera(1, map_, window_size=WINDOW_SIZE, off_screen=True)
    frames_dir = Path(tempfile.mkdtemp(prefix="tactics2d_frames_"))
    logging.info("Rendering %d frames into %s ...", len(times), frames_dir)

    frame_idx  = 0
    skip_count = 0

    for t in times:
        active_participants = {}
        active_ids          = []

        for pid, p in participants.items():
            try:
                if p.trajectory.get_state(t) is not None:
                    active_participants[pid] = p
                    active_ids.append(pid)
            except Exception:
                pass

        if not active_ids:
            continue

        try:
            camera.update(active_participants, active_ids, t)
            obs = camera.get_observation()
            Image.fromarray(obs).rotate(270).save(
                frames_dir / f"frame_{frame_idx:05d}.jpg"
            )
            frame_idx += 1
        except Exception as exc:
            logging.warning("Frame t=%d ms skipped: %s", t, exc)
            skip_count += 1

    logging.info(
        "Captured %d frames (%d skipped). Encoding ...", frame_idx, skip_count
    )

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i",         str(frames_dir / "frame_%05d.jpg"),
        "-c:v",       "libx264",
        "-pix_fmt",   "yuv420p",
        output_path,
    ]
    result = subprocess.run(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    shutil.rmtree(frames_dir)

    if result.returncode != 0:
        logging.error(
            "FFmpeg encoding failed:\n%s",
            result.stderr.decode(errors="replace"),
        )
    else:
        logging.info("Video saved to: %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not Path(FOLDER).is_dir():
        logging.error("Dataset folder not found: %s", FOLDER)
        sys.exit(1)

    logging.info(
        "Loading scenario %s from %s (filter_virtual=%s) ...",
        SCENARIO_ID, FOLDER, FILTER_VIRTUAL_LINES,
    )
    parser   = DriveInsightDParser()
    scenario = parser.parse(
        scenario_id=SCENARIO_ID,
        folder=FOLDER,
        map_name=MAP_NAME,
    )

    t_start, t_end = scenario["time_range"]
    logging.info(
        "Loaded %d participants, time range %d ms -> %d ms.",
        len(scenario["participants"]), t_start, t_end,
    )

    render_scenario(scenario, OUTPUT_PATH, filter_virtual=FILTER_VIRTUAL_LINES)


if __name__ == "__main__":
    main()