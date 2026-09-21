# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared helpers for the behavior-model tutorial notebooks."""

import inspect
import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from shapely.geometry import Point

from tactics2d.display.renderers import MatplotlibRenderer
from tactics2d.display.sensor import BEVCamera
from tactics2d.map.element import Map
from tactics2d.map.parser import OSMParser
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

# BEVCamera perception range, in meters.
PERCEPTION_RANGE = 200
# Local view window centred on the ego, in meters.
VIEW_WIDTH = 120
VIEW_HEIGHT = 80
# Colour the renderer gives the ego (takeover) vehicle.
EGO_COLOR = "light-pink"
# Warm-up window used by select_ego, in milliseconds. Mirrors the InterSim
# planning warm-up (planning_warmup_steps * step_ms = 11 * 100).
EGO_WARMUP_MS = 1100
# Fraction of the scenario span a candidate ego has to survive into.
EGO_COVERAGE = 0.75

# Take-over shared by every behavior demo; the horizon is the shortest any
# model supports.
COMPARISON_SPLIT = "validation_interactive"
COMPARISON_FILE = "validation_interactive.tfrecord-00000-of-00150"
COMPARISON_SCENARIO = 2
COMPARISON_EGO = 8
COMPARISON_FRAME_MS = 1100
COMPARISON_HORIZON_STEPS = 20

# Second shared WOMD scene, in the same shard as the first: the ego sits on the
# yielding side here, so the obligation turns into actual braking.
SECOND_SCENARIO = 9
SECOND_EGO = 2478
SECOND_FRAME_MS = COMPARISON_FRAME_MS
SECOND_HORIZON_STEPS = COMPARISON_HORIZON_STEPS

# The nuPlan log the demos replay; the window is recomputed at run time, since the
# parser stamps frames relative to ``datetime(2021, 1, 1)`` in local time.
NUPLAN_SCENARIO_FOLDER = "train_boston"
NUPLAN_SCENARIO_FILE = "2021.08.26.18.24.36_veh-28_00578_00663.db"
NUPLAN_SCENARIO_MAP = "us-ma-boston/9.12.1817"
NUPLAN_SCENARIO_EGO = 26
NUPLAN_SCENARIO_TAG = "traversing_intersection"
# Recording is 20 Hz; the models are laid out on a 100 ms lattice.
NUPLAN_NATIVE_STEP_MS = 50
NUPLAN_CONTEXT_MS = 5000
NUPLAN_FUTURE_MS = 8000


def apply_notebook_style():
    """Apply the figure style shared by the behavior tutorial notebooks."""

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "font.family": "DejaVu Sans Mono",
            "font.size": 8,
            "animation.html": "html5",
            "animation.embed_limit": 100 * 1024 * 1024,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def select_ego(participants: Dict[object, object], warmup_ms=EGO_WARMUP_MS, coverage=EGO_COVERAGE):
    """Select a persistent early vehicle as the ego (the takeover target).

    Prefers a vehicle already present within ``warmup_ms`` of the scenario's
    earliest frame that survives into at least ``coverage`` of the scenario
    span, falling back to every vehicle when none qualifies. Ties break on the
    earliest start, then the longest track, then ``str(agent_id)``.

    The thresholds are in milliseconds, so the same defaults serve datasets
    parsed at any frame rate.

    Args:
        participants (Dict): All participants in the scenario.
        warmup_ms (int, optional): Warm-up window measured from the earliest
            candidate start, in milliseconds. Defaults to 1100.
        coverage (float, optional): Fraction of the scenario span a candidate
            has to survive into. Defaults to 0.75.

    Returns:
        The selected agent id.

    Raises:
        ValueError: If the scenario contains no vehicle with a trajectory.
    """

    candidates = []
    for agent_id, participant in participants.items():
        if not isinstance(participant, Vehicle):
            continue
        first_frame = participant.trajectory.first_frame
        last_frame = participant.trajectory.last_frame
        if first_frame is None or last_frame is None:
            continue
        candidates.append((agent_id, first_frame, last_frame, len(participant.trajectory.frames)))
    if not candidates:
        raise ValueError("The scenario contains no vehicle with a trajectory.")

    first_frame = min(item[1] for item in candidates)
    last_frame = max(item[2] for item in candidates)
    warmup = first_frame + warmup_ms
    late = first_frame + coverage * (last_frame - first_frame)
    persistent = [item for item in candidates if item[1] <= warmup and item[2] >= late]
    pool = persistent or candidates
    return min(pool, key=lambda item: (item[1], -item[3], str(item[0])))[0]


def render_replay_animation(
    participants,
    map_,
    playback_frames,
    ego_id,
    plans: Optional[Dict[int, Sequence[Tuple[int, float, float]]]] = None,
    resolution=(1200, 800),
    fps=10,
    perception_range=PERCEPTION_RANGE,
    view_width=VIEW_WIDTH,
    view_height=VIEW_HEIGHT,
    title_prefix=None,
    driven_colormap="BuPu",
    plan_colormap="GnBu",
):
    """Animate a replayed ego trajectory on a map with two gradient traces.

    The camera follows the ego in a fixed-size window, the path the ego has
    already driven is drawn with ``driven_colormap`` and its current plan with
    ``plan_colormap``, clipped to the part the ego has not driven yet. Every
    trace is cleared and redrawn on each frame.

    Args:
        participants (Dict): The participants to render, keyed by agent id.
        map_ (Map): The map drawn behind them.
        playback_frames (List[int]): Frame timestamps to animate, in milliseconds.
        ego_id (object): The agent the camera follows.
        plans (Dict, optional): Plans keyed by the frame that issued them, in
            milliseconds. Each value is a sequence of ``(frame_ms, x, y)``
            waypoints in world coordinates. Defaults to None.
        resolution (Tuple[float, float], optional): Output size in pixels.
            Defaults to (1200, 800).
        fps (float, optional): Playback rate of ``playback_frames``. Match it to
            the spacing of those frames, not to the parsed dataset's rate.
            Defaults to 10.
        perception_range (float, optional): BEVCamera perception range, in meters.
            Defaults to 200.
        view_width (float, optional): Window width centred on the ego, in meters.
            Defaults to 120.
        view_height (float, optional): Window height centred on the ego, in
            meters. Defaults to 80.
        title_prefix (str, optional): Prefix of the per-frame title. Defaults to
            None, which reads "Closed loop".
        driven_colormap (str, optional): Colormap of the driven path. Defaults to
            "BuPu".
        plan_colormap (str, optional): Colormap of the current plan. Defaults to
            "GnBu".

    Returns:
        The ``FuncAnimation`` driving ``playback_frames``.
    """

    for roadline in map_.roadlines.values():
        if roadline.type_ is None:
            roadline.type_ = "roadline"
    camera = BEVCamera(id_=0, map_=map_, perception_range=perception_range)
    prev_road, prev_part = set(), set()
    renderer = MatplotlibRenderer(
        xlim=(-view_width / 2, view_width / 2),
        ylim=(-view_height / 2, view_height / 2),
        resolution=resolution,
        auto_scale=False,
    )
    renderer.enable_trajectory_gradient()
    ego_trajectory = participants[ego_id].trajectory
    plan_frames = sorted(plans) if plans else []

    def update(frame):
        nonlocal prev_road, prev_part
        renderer._remove_trajectory_lines()
        pids = [pid for pid, p in participants.items() if frame in p.trajectory.history_states]
        if ego_trajectory.has_state(frame):
            ego_pose = ego_trajectory.get_state(frame)
            cam_pos = Point(ego_pose.x, ego_pose.y)
            renderer.ax.set_xlim(ego_pose.x - view_width / 2, ego_pose.x + view_width / 2)
            renderer.ax.set_ylim(ego_pose.y - view_height / 2, ego_pose.y + view_height / 2)
        else:
            ego_pose = None
            cam_pos = Point(0.0, 0.0)
        geometry_data, prev_road, prev_part = camera.update(
            frame, participants, pids, prev_road, prev_part, cam_pos
        )
        renderer.update(geometry_data)
        if ego_pose is not None:
            driven = [f for f in playback_frames if f <= frame and ego_trajectory.has_state(f)]
            if len(driven) >= 2:
                points = [
                    (ego_trajectory.get_state(f).x, ego_trajectory.get_state(f).y) for f in driven
                ]
                renderer.draw_gradient_trace(list(reversed(points)), driven_colormap)
            issued = [f for f in plan_frames if f <= frame]
            if issued:
                # Only the part of the plan that has not been driven yet, so the
                # trace always reads forward from the ego.
                ahead = [(x, y) for f, x, y in plans[issued[-1]] if f > frame]
                if len(ahead) >= 2:
                    renderer.draw_gradient_trace([(ego_pose.x, ego_pose.y)] + ahead, plan_colormap)
        renderer.ax.set_title(
            f"{title_prefix or 'Closed loop'}: {ego_id}  |  frame {frame}  |  active: {len(pids)}",
            fontsize=7,
        )

    interval_ms = max(10, int(1000 / fps))
    return FuncAnimation(
        renderer.fig, update, frames=playback_frames, interval=interval_ms, repeat=True
    )


def nuplan_intersection_window(
    db_path,
    tag: str = NUPLAN_SCENARIO_TAG,
    context_ms: int = NUPLAN_CONTEXT_MS,
    future_ms: int = NUPLAN_FUTURE_MS,
):
    """Return the millisecond window around a log's longest intersection pass.

    The window is the longest unbroken run of *tag* frames, padded by
    *context_ms* ahead and *future_ms* behind.

    Args:
        db_path (str): Path to the nuPlan ``.db`` log.
        tag (str, optional): Scenario type to look for. Defaults to
            ``NUPLAN_SCENARIO_TAG``.
        context_ms (int, optional): Milliseconds of run-up to keep. Defaults to
            ``NUPLAN_CONTEXT_MS``.
        future_ms (int, optional): Milliseconds of run-out to keep. Defaults to
            ``NUPLAN_FUTURE_MS``.

    Returns:
        A ``(start_ms, end_ms)`` tuple in the parser's own frame convention.

    Raises:
        ValueError: If the log carries no frame tagged *tag*.
    """

    import sqlite3

    from tactics2d.dataset_parser.parse_nuplan import NuPlanParser

    with sqlite3.connect(str(db_path)) as connection:
        stamps = dict(connection.execute("SELECT token, timestamp FROM lidar_pc"))
        tagged = sorted(
            stamps[token]
            for (token,) in connection.execute(
                "SELECT lidar_pc_token FROM scenario_tag WHERE type = ?", (tag,)
            )
        )
    if not tagged:
        raise ValueError(f"{db_path} has no frame tagged {tag!r}.")

    runs, current = [], [tagged[0]]
    for previous, moment in zip(tagged, tagged[1:]):
        # Tags are stamped per lidar sweep; a wider gap means another intersection.
        if moment - previous <= 4 * NUPLAN_NATIVE_STEP_MS * 1000:
            current.append(moment)
        else:
            runs.append(current)
            current = [moment]
    runs.append(current)
    longest = max(runs, key=len)

    # ``_DATETIME`` is the parser's own epoch; frames it hands out are relative to it.
    epoch_ms = NuPlanParser._DATETIME
    return (
        int(longest[0] / 1000 - epoch_ms) - context_ms,
        int(longest[-1] / 1000 - epoch_ms) + future_ms,
    )


def recorded_future(trajectory: Trajectory, frame_ms: int) -> Trajectory:
    """Take a copy of the part of a trajectory recorded after ``frame_ms``.

    Args:
        trajectory (Trajectory): The vehicle's recorded trajectory.
        frame_ms (int): The take-over frame. The unit is millisecond (ms).

    Returns:
        A stand-alone ``Trajectory`` holding every state after ``frame_ms``.
    """

    future = Trajectory(id_=trajectory.id_, fps=trajectory.fps, stable_freq=trajectory.stable_freq)
    for frame in sorted(trajectory.frames):
        if frame > frame_ms:
            future.add_state(trajectory.get_state(frame))
    return future


def displacement_errors(
    predicted: Trajectory,
    recorded: Trajectory,
    horizon_steps: Optional[int] = None,
    tolerance_ms: int = 50,
):
    """Score a predicted future against the recorded one.

    Args:
        predicted (Trajectory): The predicted future.
        recorded (Trajectory): The recorded future, from ``recorded_future``.
        horizon_steps (int, optional): Number of matched steps to score, counted
            from the take-over. Defaults to None, i.e. every matched step.
        tolerance_ms (int, optional): Largest frame gap a pair may have, in
            milliseconds. Defaults to 50.

    Returns:
        A tuple of:
            - ade (float): Mean displacement over the scored steps, in metres.
            - fde (float): Displacement at the last scored step, in metres.
            - matched (int): How many steps were scored.

        Both errors are ``inf`` when no pair survives the tolerance.
    """

    predicted_frames = sorted(predicted.frames)
    errors = []
    for frame in sorted(recorded.frames):
        if not predicted_frames:
            break
        nearest = min(predicted_frames, key=lambda other: abs(other - frame))
        if abs(nearest - frame) > tolerance_ms:
            continue
        state = predicted.get_state(nearest)
        truth = recorded.get_state(frame)
        errors.append(math.hypot(state.x - truth.x, state.y - truth.y))
    if horizon_steps is not None:
        errors = errors[:horizon_steps]
    if not errors:
        return float("inf"), float("inf"), 0
    return sum(errors) / len(errors), errors[-1], len(errors)


def load_map(
    parser,
    file_name=None,
    folder=None,
    map_file=None,
    map_folder=None,
    map_path=None,
    map_config=None,
    **parse_kwargs,
) -> Map:
    """Load the map of a scenario, with the fallbacks every demo needs.

    The scenario parser is asked first. Not every dataset ships a map next to its
    trajectories, so a stand-alone map file is loaded through ``OSMParser`` when
    that fails, and an empty map is returned when there is neither - the models
    accept one, and the demos then differ only in how much context they draw.

    Args:
        parser: The dataset parser the scenario came from.
        file_name (str, optional): Scenario file the trajectories were parsed
            from, used as the map's when the dataset keeps them together.
            Defaults to None.
        folder (str, optional): Folder of the scenario file. Defaults to None.
        map_file (str, optional): Map file, when the dataset keeps the map and
            the scenario in separate files. Defaults to None, i.e. ``file_name``.
        map_folder (str, optional): Folder of the map file. Defaults to None,
            i.e. ``folder``.
        map_path (str, optional): Path of a stand-alone OpenStreetMap file.
            Defaults to None.
        map_config (dict, optional): Configuration passed to the OSM parser.
            Defaults to None.
        **parse_kwargs: Scenario arguments, forwarded to the parser for the keys
            its map interface takes; the scenario's own window is the
            trajectory's business.

    Returns:
        The parsed map, or an empty one.
    """

    map_ = None
    if hasattr(parser, "parse_map"):
        takes = inspect.signature(parser.parse_map).parameters
        map_ = parser.parse_map(
            **{name: value for name, value in parse_kwargs.items() if name in takes},
            file=map_file if map_file is not None else file_name,
            folder=map_folder if map_folder is not None else folder,
        )
    if map_ is None and map_path is not None:
        print(f"  loading map from {map_path}")
        map_ = OSMParser(lanelet2=True).parse(file_path=map_path, configs=map_config)
    return map_ if map_ is not None else Map("empty_map", scenario_type="demo")


def rebuild_render_participants(poses, participants, step_ms, frame_ms0=0) -> Dict[object, object]:
    """Turn the pose arrays of a runner into participants a camera can render.

    The arrays carry positions and headings only, so the speed is recovered from
    the last pose seen, over the frames it took to reach it; the models read that
    speed as the agent's initial one, and a state without velocity would plan
    every agent as if it were standing still. Rows whose first column is ``-1.0``
    hold no pose: they are filled with the last one seen, so a trajectory covers
    the span the runner actually simulated.

    Args:
        poses (Dict): Pose arrays keyed by agent id, one ``(x, y, _, heading)``
            row per frame.
        participants (Dict): The parsed participants, used for the vehicle class,
            its dimensions and the recorded goal of each agent.
        step_ms (int): Length of one frame, in milliseconds.
        frame_ms0 (int, optional): Timestamp of the first row. Defaults to 0.

    Returns:
        The rebuilt participants, keyed by agent id.
    """

    dt = step_ms / 1000.0
    rebuilt = {}
    for agent_id, array in poses.items():
        valid = [index for index, row in enumerate(array) if row[0] != -1.0]
        if not valid:
            continue
        source = participants.get(agent_id)
        trajectory = Trajectory(id_=agent_id, fps=round(1000.0 / step_ms, 3), stable_freq=True)
        seen = None
        for index in range(valid[0], valid[-1] + 1):
            row = array[index]
            if row[0] != -1.0:
                seen = (float(row[0]), float(row[1]), float(row[3]))
            speed = 0.0
            for back in (1, 2, 3, 4, 5):
                past = index - back
                if past >= 0 and array[past][0] != -1.0:
                    speed = math.hypot(seen[0] - array[past][0], seen[1] - array[past][1]) / (
                        back * dt
                    )
                    break
            trajectory.add_state(
                State(
                    frame=int(frame_ms0 + index * step_ms),
                    x=seen[0],
                    y=seen[1],
                    heading=seen[2],
                    vx=speed * math.cos(seen[2]),
                    vy=speed * math.sin(seen[2]),
                    speed=speed,
                )
            )
        if source is None:
            rebuilt[agent_id] = Vehicle(agent_id, "vehicle", trajectory=trajectory)
            continue
        rebuilt[agent_id] = type(source)(
            agent_id,
            source.type_,
            trajectory=trajectory,
            length=float(getattr(source, "length", 0.0) or 4.8),
            width=float(getattr(source, "width", 0.0) or 1.9),
        )
        # The closed loop hands every agent its last recorded position as a
        # routing goal; mirror it so a re-derived plan follows the same lanes.
        if source.trajectory.last_state is not None:
            rebuilt[agent_id].goal_xy = (
                source.trajectory.last_state.x,
                source.trajectory.last_state.y,
            )
    return rebuilt


def collect_ego_plans(
    plan_at: Callable[[int], Optional[Trajectory]],
    warmup_steps: int,
    steps: int,
    interval_steps: int,
    step_ms: int,
    frame_ms0: int = 0,
) -> Dict[int, Sequence[Tuple[int, float, float]]]:
    """Re-derive the ego's plan at every planning frame of a closed loop.

    The runners commit their plans into the pose arrays but do not return them,
    so the plan trace is recovered by asking the same model, at the same frames,
    for the ego's plan on the simulated state. Over the window in which a plan is
    authoritative (up to the next replan) this reproduces what the runner
    committed.

    Args:
        plan_at (Callable): Returns the trajectory the model plans for the ego at
            a frame, or None when it has none.
        warmup_steps (int): Frames of warm-up the model needs before planning.
        steps (int): Number of simulated frames, counted from the first one and
            exclusive as in ``range``.
        interval_steps (int): Frames between two plans (the replan cadence).
        step_ms (int): Length of one frame, in milliseconds.
        frame_ms0 (int, optional): Timestamp of the first frame. Defaults to 0.

    Returns:
        Plans keyed by the frame that issued them, each a sequence of
        ``(frame_ms, x, y)`` waypoints.
    """

    plans = {}
    for index in range(warmup_steps, steps, interval_steps):
        frame = frame_ms0 + index * step_ms
        trajectory = plan_at(frame)
        if trajectory is None:
            continue
        plans[frame] = [
            (frame_, trajectory.get_state(frame_).x, trajectory.get_state(frame_).y)
            for frame_ in trajectory.frames
        ]
    return plans
