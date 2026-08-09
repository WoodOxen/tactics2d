#!/usr/bin/env python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Benchmark LimSim RoI multi-vehicle takeover across datasets.

Usage::

    python limsim_takeover_single.py --datasets highD inD rounD WOMD --scenarios 100

Output: ``./runtime/limsim/<dataset>/``  (CSV + GIFs).
"""

import argparse
import csv
import gc
import hashlib
import os
import random
import statistics
import time
from functools import partial
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from shapely.geometry import Point
from tqdm import tqdm

matplotlib.use("Agg")
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from tactics2d.behavior import LimSimBehaviorModel, LimSimConfig
from tactics2d.behavior.limsim import (
    InteractiveReplayController,
    apply_rollout_states,
    restore_recorded_snapshots,
    snapshot_vehicle_trajectories,
)
from tactics2d.behavior.limsim.lane_follower import route_lanes_from_agent
from tactics2d.behavior.limsim.schema import AgentDecisionState
from tactics2d.dataset_parser import (
    LevelXParser,
    WOMDParser,
    extract_lane_sequence,
    infer_lane_topology,
)
from tactics2d.dataset_parser.womd_proto import scenario_pb
from tactics2d.display.renderers import MatplotlibRenderer
from tactics2d.display.sensor import BEVCamera
from tactics2d.geometry import spatial
from tactics2d.map.map_config import HIGHD_MAP_CONFIG, IND_MAP_CONFIG, ROUND_MAP_CONFIG
from tactics2d.map.parser import OSMParser
from tactics2d.participant.element import Vehicle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIGS = {
    "fast_preview": LimSimConfig(
        mcts_iterations=50,
        terminal_depth=3,
        horizon_steps=30,
        max_group_size=2,
        interaction_distance=20.0,
        max_routes_per_agent=10,
        use_frenet_refinement=True,
    ),
    "balanced_demo": LimSimConfig(
        mcts_iterations=200,
        terminal_depth=4,
        horizon_steps=50,
        max_group_size=3,
        interaction_distance=30.0,
        max_routes_per_agent=10,
        use_frenet_refinement=True,
    ),
    "high_fidelity": LimSimConfig(
        mcts_iterations=400,
        terminal_depth=4,
        horizon_steps=80,
        max_group_size=3,
        interaction_distance=50.0,
        max_routes_per_agent=10,
        use_frenet_refinement=True,
    ),
}

DATA_ROOT = Path(__file__).resolve().parent / "data"
OUT_ROOT = Path(__file__).resolve().parent / "runtime" / "limsim"
HISTORY_SECONDS = 1
VIEW_WIDTH, VIEW_HEIGHT = 120, 80

DATASET_FPS = {"WOMD": 10, "highD": 25, "inD": 25, "rounD": 25}
DATASET_SIMULATION_SECONDS = {"WOMD": 5, "highD": 10, "inD": 10, "rounD": 10}
DEFAULT_ROI_RADIUS_M = 50.0
DEFAULT_ROI_OUTER_RADIUS_M = 100.0
DEFAULT_RANDOM_SEED = 42


def _simulation_ms(ds_name):
    return int(DATASET_SIMULATION_SECONDS.get(ds_name, 10) * 1000)


def _dataset_sampling_seed(base_seed, dataset):
    """Derive the historical per-dataset sampling seed from a base seed."""

    return base_seed + sum((index + 1) * ord(char) for index, char in enumerate(dataset))


def _planner_seed(base_seed, dataset, info, takeover_frame, config_name):
    """Return a stable 32-bit planner seed for one configuration rollout."""

    payload = "\n".join(
        (
            "limsim-planner-v1",
            str(base_seed),
            dataset,
            scenario_label(info),
            str(takeover_frame),
            config_name,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big")


def _reset_planner_random_state(seed):
    """Reset all random generators used by the LimSim planning path."""

    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def check_planned_collisions(trajectories, frame, participants, horizon_ms=5000):
    """Count conflicting pairs among all RoI trajectories planned by LimSim."""

    collision_pairs = 0
    pairs = list(combinations(trajectories.items(), 2))
    for (first_id, first), (second_id, second) in pairs:
        common_frames = sorted(set(first.frames) & set(second.frames))
        for future_frame in common_frames:
            if not frame < future_frame <= frame + horizon_ms:
                continue
            first_state = first.get_state(future_frame)
            second_state = second.get_state(future_frame)
            first_vehicle = participants.get(first_id)
            second_vehicle = participants.get(second_id)
            first_shape = spatial.oriented_box(
                first_state.x,
                first_state.y,
                first_state.heading,
                getattr(first_vehicle, "length", None) or 4.8,
                getattr(first_vehicle, "width", None) or 1.9,
            )
            second_shape = spatial.oriented_box(
                second_state.x,
                second_state.y,
                second_state.heading,
                getattr(second_vehicle, "length", None) or 4.8,
                getattr(second_vehicle, "width", None) or 1.9,
            )
            if first_shape.intersects(second_shape):
                collision_pairs += 1
                break
    return collision_pairs, len(pairs)


def scenario_label(info):
    if "scenario_id" in info:
        return info["scenario_id"]
    if "file_id" in info:
        return f"file_{info['file_id']}"
    return info.get("file_name", "unknown")


def interpolate_controlled_rollout_for_render(
    participants,
    recorded_snapshots,
    rollout_states,
    controlled_ids_by_frame,
    ego_id,
    takeover_frame,
    target_frames,
):
    """Interpolate controlled trajectories onto the dataset frame grid."""

    controlled_ids = (
        set().union(*controlled_ids_by_frame.values()) if controlled_ids_by_frame else set()
    )
    control_start_by_id = {}
    for frame, participant_ids in controlled_ids_by_frame.items():
        for participant_id in participant_ids:
            control_start_by_id[participant_id] = min(
                frame, control_start_by_id.get(participant_id, frame)
            )
    control_start_by_id[ego_id] = takeover_frame

    for participant_id in controlled_ids:
        participant = participants.get(participant_id)
        if not isinstance(participant, Vehicle):
            continue

        control_start = control_start_by_id.get(participant_id, takeover_frame)
        trajectory = participant.trajectory
        recorded_states = recorded_snapshots.get(participant_id, {})
        retained = {
            frame: state for frame, state in recorded_states.items() if frame <= control_start
        }
        trajectory._history_states = retained
        trajectory._frames = sorted(retained)
        trajectory._current_state = retained[trajectory._frames[-1]] if trajectory._frames else None

        anchors = {
            frame: state
            for frame, state in rollout_states.get(participant_id, {}).items()
            if frame >= control_start
        }
        if control_start == takeover_frame and control_start in recorded_states:
            anchors[control_start] = recorded_states[control_start]
        elif control_start in rollout_states.get(participant_id, {}):
            anchors[control_start] = rollout_states[participant_id][control_start]
        anchor_frames = sorted(anchors)
        if not anchor_frames:
            continue

        render_frames = [
            frame for frame in target_frames if anchor_frames[0] <= frame <= anchor_frames[-1]
        ]
        if not render_frames:
            continue

        anchor_x = np.asarray([anchors[frame].x for frame in anchor_frames], dtype=float)
        anchor_y = np.asarray([anchors[frame].y for frame in anchor_frames], dtype=float)
        anchor_heading = np.unwrap(
            np.asarray([anchors[frame].heading for frame in anchor_frames], dtype=float)
        )
        interpolated_x = np.interp(render_frames, anchor_frames, anchor_x)
        interpolated_y = np.interp(render_frames, anchor_frames, anchor_y)
        interpolated_heading = np.interp(render_frames, anchor_frames, anchor_heading)
        state_cls = type(anchors[anchor_frames[0]])
        for frame, x, y, heading in zip(
            render_frames,
            interpolated_x,
            interpolated_y,
            interpolated_heading,
        ):
            state = state_cls(
                frame=frame,
                x=float(x),
                y=float(y),
                heading=float(np.arctan2(np.sin(heading), np.cos(heading))),
            )
            if trajectory.has_state(frame):
                trajectory._history_states[frame] = state
            else:
                trajectory.add_state(state)


def expand_controlled_ids_for_render(
    controlled_ids_by_frame,
    simulation_frames,
    playback_frames,
    takeover_frame,
):
    """Map planner-step ownership onto intermediate dataset frames."""

    if not simulation_frames:
        return {}
    expanded = {}
    simulation_frames = sorted(simulation_frames)
    for frame in playback_frames:
        if frame <= takeover_frame:
            continue
        right_index = int(np.searchsorted(simulation_frames, frame, side="left"))
        if right_index < len(simulation_frames):
            expanded[frame] = set(
                controlled_ids_by_frame.get(simulation_frames[right_index], set())
            )
    return expanded


# ---------------------------------------------------------------------------
# Parse one file
# ---------------------------------------------------------------------------
def parse_scenario(info):
    ds = info["dataset"]
    if ds in ("highD", "inD", "rounD"):
        parser = LevelXParser(ds)
        participants, _ = parser.parse_trajectory(file=info["file_id"], folder=info["folder"])
        loc = parser.get_location(info["file_id"], info["folder"])
        loc_key = f"{ds}_{loc}" if ds != "inD" else f"inD_{loc}"
        map_path = os.path.join(info["map_dir"], f"{loc_key}.osm")
        map_configs = {"highD": HIGHD_MAP_CONFIG, "inD": IND_MAP_CONFIG, "rounD": ROUND_MAP_CONFIG}
        map_ = OSMParser(lanelet2=True).parse(file_path=map_path, configs=map_configs[ds][loc_key])
        infer_lane_topology(map_)
        return participants, map_
    else:
        parser = WOMDParser()
        participants, _ = parser.parse_trajectory(
            scenario_id=info["scenario_id"], file=info["file_name"], folder=info["folder"]
        )
        map_ = parser.parse_map(
            scenario_id=info["scenario_id"], file=info["file_name"], folder=info["folder"]
        )
        infer_lane_topology(map_)
        return participants, map_


# ---------------------------------------------------------------------------
# Clip discovery — each long recording yields multiple windows
# ---------------------------------------------------------------------------
def _check_ego_motion(
    participants, ego_id, tf, window_ms=1000, min_displacement=1.0, min_speed=0.5
):
    """Return True if the ego vehicle moves enough in the window after tf."""
    traj = participants[ego_id].trajectory
    s0 = traj.get_state(tf)
    future = sorted(f for f in traj.history_states if tf < f <= tf + window_ms)
    if not future:
        return False
    s1 = traj.get_state(future[-1])
    displacement = ((s1.x - s0.x) ** 2 + (s1.y - s0.y) ** 2) ** 0.5
    if displacement < min_displacement:
        return False
    dt_s = (future[-1] - tf) / 1000.0
    if dt_s > 0 and displacement / dt_s < min_speed:
        return False
    return True


def discover_clips(info):
    """Yield (info, takeover_frame) for every viable window.

    Only yields clips where the longest-trajectory vehicle (future ego) is
    active at the takeover frame, has enough remaining track, and is moving.
    """
    if info["dataset"] == "WOMD":
        yield from discover_clips_womd(info)
        return

    try:
        participants, _ = parse_scenario(info)
    except Exception:
        return

    veh_ids = [
        pid
        for pid, p in participants.items()
        if isinstance(p, Vehicle) and len(p.trajectory.history_states) >= 10
    ]
    if len(veh_ids) < 2:
        return

    veh_sorted = sorted(
        veh_ids, key=lambda pid: len(participants[pid].trajectory.history_states), reverse=True
    )

    simulation_ms = _simulation_ms(info["dataset"])
    headway = int(0.5 * 1000)
    yielded = set()  # avoid yielding duplicate takeover_frame values

    # LevelX: 按轨迹长度降序尝试候选 ego，能找到运动的就 yield
    for candidate in veh_sorted:
        ego_frames = sorted(participants[candidate].trajectory.history_states.keys())
        ego_first = ego_frames[0]
        ego_last = ego_frames[-1]
        tf = ego_first + 1000
        while tf + simulation_ms <= ego_last:
            if tf in yielded:
                tf += headway
                continue
            active = [
                pid
                for pid in veh_ids
                if pid != candidate and participants[pid].trajectory.has_state(tf)
            ]
            if len(active) >= 1 and _check_ego_motion(participants, candidate, tf):
                clip_info = dict(info)
                clip_info["ego_id"] = candidate
                yielded.add(tf)
                yield (clip_info, tf)
            tf += headway


def discover_clips_womd(info):
    """Iterate all scenarios in a WOMD file once (cached), yield viable clips.

    Uses cached protobuf data directly instead of reopening the file per scenario.
    """
    try:
        parser = WOMDParser()
        dataset = parser._get_dataset(file=info["file_name"], folder=info["folder"])
        scenario_ids, cached_data = parser.get_scenario_ids(dataset, cache_data=True)
    except Exception:
        return

    simulation_ms = _simulation_ms(info["dataset"])

    for i, sid in enumerate(scenario_ids):
        scenario = scenario_pb.Scenario()
        scenario.ParseFromString(cached_data[i])
        timestamps_ms = [int(round(t * 1000)) for t in scenario.timestamps_seconds]

        # Quick vehicle info from raw protobuf (avoid Participant/Trajectory overhead)
        track_info = []  # (track_id, frames_list, state_indices, states)
        for track in scenario.tracks:
            if track.object_type != 1:  # not a vehicle
                continue
            frames = []
            valid_idx = []
            for j, s in enumerate(track.states):
                if s.valid:
                    frames.append(timestamps_ms[j])
                    valid_idx.append(j)
            if len(frames) >= 10:
                track_info.append((track.id, frames, valid_idx, track.states))

        if len(track_info) < 2:
            continue

        # Sort by trajectory length descending. Try each as ego until one passes.
        track_info.sort(key=lambda x: len(x[1]), reverse=True)
        yielded_ego_ids = set()

        for c_ego_id, c_frames, c_idx, c_states in track_info:
            if c_ego_id in yielded_ego_ids:
                continue
            yielded_ego_ids.add(c_ego_id)

            c_first = c_frames[0]
            c_last = c_frames[-1]
            # Find the frame closest to 1s after first appearance
            target_tf = c_first + 1000
            tf = min(c_frames, key=lambda f: abs(f - target_tf))
            if abs(tf - target_tf) > 500:  # too far from 1s mark
                continue
            if not (tf + simulation_ms <= c_last):
                continue

            # At least one other vehicle active at takeover frame
            other_active = any(tid != c_ego_id and tf in fr for tid, fr, _, _ in track_info)
            if not other_active:
                continue

            # Motion check: displacement in 1s after tf
            try:
                pos0 = c_states[c_idx[c_frames.index(tf)]]
            except (ValueError, IndexError):
                continue

            end_candidate = [f for f in c_frames if tf < f <= tf + 1000]
            if not end_candidate:
                continue
            end_pos = c_frames.index(end_candidate[-1])
            pos1 = c_states[c_idx[end_pos]]
            dx = pos1.center_x - pos0.center_x
            dy = pos1.center_y - pos0.center_y
            displacement = (dx * dx + dy * dy) ** 0.5
            dt_s = (c_frames[end_pos] - tf) / 1000.0
            if displacement < 1.0 or (dt_s > 0 and displacement / dt_s < 0.5):
                continue

            clip_info = dict(info)
            clip_info["scenario_id"] = sid
            clip_info["ego_id"] = c_ego_id
            yield (clip_info, tf)
            break  # one clip per scenario


def sample_clips(file_infos, num_clips=100, seed=DEFAULT_RANDOM_SEED, takeover_frame=None):
    """Per-file balanced clip sampling."""
    rng = random.Random(seed)
    shuffled_infos = list(file_infos)
    rng.shuffle(shuffled_infos)
    target_files = min(num_clips, len(shuffled_infos))
    pools = {}
    for info in tqdm(shuffled_infos, desc="discover", unit="file"):
        clips = list(discover_clips(info))
        if takeover_frame is not None:
            clips = [clip for clip in clips if clip[1] == takeover_frame]
        if clips:
            pools[scenario_label(info)] = clips
            if len(pools) >= target_files:
                break

    if not pools:
        return []

    per_file = (num_clips + len(pools) - 1) // len(pools)
    sampled = []
    for key, clips in pools.items():
        sampled.extend(clips if len(clips) <= per_file else rng.sample(clips, per_file))

    if len(sampled) > num_clips:
        sampled = rng.sample(sampled, num_clips)
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Dataset route adapter
# ---------------------------------------------------------------------------
def update_local_route_map(
    participants,
    map_,
    frame,
    ego_id,
    roi_outer_radius,
    route_map,
    *,
    end_frame=None,
    failed_ids=None,
    max_routes=10,
):
    """Cache route-compatible lane sequences for vehicles inside the outer RoI."""

    ego = participants.get(ego_id)
    if ego is None or not ego.trajectory.has_state(frame):
        return
    ego_state = ego.trajectory.get_state(frame)
    for participant_id, participant in participants.items():
        if (
            participant_id in route_map
            or (failed_ids is not None and participant_id in failed_ids)
            or not isinstance(participant, Vehicle)
            or not participant.trajectory.has_state(frame)
        ):
            continue
        state = participant.trajectory.get_state(frame)
        if np.hypot(state.x - ego_state.x, state.y - ego_state.y) > roi_outer_radius:
            continue
        try:
            route = extract_lane_sequence(
                participant,
                map_,
                start_frame=frame,
                end_frame=end_frame,
            )
        except Exception:
            if failed_ids is not None:
                failed_ids.add(participant_id)
            continue
        if route:
            route_state = AgentDecisionState(
                agent_id=participant_id,
                x=state.x,
                y=state.y,
                heading=state.heading,
                speed=max(state.speed or 0.0, 0.0),
                lane_id=route[0],
                route_lane_ids=tuple(route),
            )
            route_map[participant_id] = tuple(
                route_lanes_from_agent(
                    route_state,
                    map_,
                    max(max_routes, len(route)),
                )
            )
        elif failed_ids is not None:
            failed_ids.add(participant_id)


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------
def render_gif(
    participants,
    map_,
    playback_frames,
    ego_id,
    ego_plans,
    out_path,
    fps,
    show_history=True,
    controlled_ids_by_frame=None,
):
    for rl in map_.roadlines.values():
        if rl.type_ is None:
            rl.type_ = "roadline"

    camera = BEVCamera(id_=0, map_=map_, perception_range=200)
    r = MatplotlibRenderer(
        xlim=(-VIEW_WIDTH / 2, VIEW_WIDTH / 2),
        ylim=(-VIEW_HEIGHT / 2, VIEW_HEIGHT / 2),
        resolution=(1200, 800),
        auto_scale=False,
    )
    r.enable_trajectory_gradient()
    fig = r.fig
    canvas = fig.canvas

    frames_pil = []
    original_colors = {
        participant_id: participant.color
        for participant_id, participant in participants.items()
        if isinstance(participant, Vehicle)
    }
    controlled_ids_by_frame = controlled_ids_by_frame or {}
    previous_controlled_ids = set()
    previous_road_ids = set()
    previous_participant_ids = set()
    try:
        for frame in tqdm(playback_frames, desc="frame", leave=False):
            controlled_ids = set(controlled_ids_by_frame.get(frame, set()))
            for participant_id, color in original_colors.items():
                participants[participant_id].color = (
                    "orange"
                    if participant_id != ego_id and participant_id in controlled_ids
                    else color
                )

            p = participants[ego_id]
            if p.trajectory.has_state(frame):
                ego_pose = p.trajectory.get_state(frame)
                cx, cy = ego_pose.x, ego_pose.y
            else:
                cx, cy = 0.0, 0.0
                ego_pose = None
            r.ax.set_xlim(cx - VIEW_WIDTH / 2, cx + VIEW_WIDTH / 2)
            r.ax.set_ylim(cy - VIEW_HEIGHT / 2, cy + VIEW_HEIGHT / 2)

            pids = [
                pid for pid, p2 in participants.items() if frame in p2.trajectory.history_states
            ]
            camera._position = None
            try:
                gd, previous_road_ids, previous_participant_ids = camera.update(
                    frame,
                    participants,
                    pids,
                    previous_road_ids,
                    previous_participant_ids,
                    Point(cx, cy),
                )
            except Exception:
                continue  # skip frames with invalid geometry
            recolor_ids = (controlled_ids ^ previous_controlled_ids) & set(pids)
            if recolor_ids:
                participant_data = gd["participant_data"]
                participant_data["participant_id_to_remove"] = list(
                    set(participant_data["participant_id_to_remove"]) | recolor_ids
                )
                participant_data["participant_id_to_create"] = list(
                    set(participant_data["participant_id_to_create"]) | recolor_ids
                )
            previous_controlled_ids = controlled_ids
            r.update(gd)

            # remove old traces before drawing new ones
            r._remove_trajectory_lines()

            if ego_pose is not None:
                if show_history:
                    past = [f for f in playback_frames if f <= frame and p.trajectory.has_state(f)]
                    if len(past) >= 2:
                        pts = [
                            (p.trajectory.get_state(f).x, p.trajectory.get_state(f).y) for f in past
                        ]
                        r.draw_gradient_trace(
                            list(reversed(pts)), cm.BuPu, linewidth=0.8, alpha=0.55
                        )
                if ego_plans:
                    cand = [f for f in sorted(ego_plans) if f <= frame]
                    if cand:
                        planned_points = ego_plans[cand[-1]]
                        fpts = [(ego_pose.x, ego_pose.y)]
                        fpts.extend(planned_points)
                        if len(fpts) >= 2:
                            r.draw_gradient_trace(fpts, cm.GnBu, linewidth=0.8, alpha=0.55)

            visible = sum(1 for pid in pids if isinstance(participants.get(pid), Vehicle))
            controlled_visible = len(set(pids) & set(controlled_ids))
            r.ax.set_title(
                f"frame {frame}  |  vehicles: {visible}  |  controlled: {controlled_visible}",
                fontsize=6,
            )

            canvas.draw()
            buf = canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype="uint8").reshape(canvas.get_width_height()[::-1] + (3,))
            frames_pil.append(Image.fromarray(img))
    finally:
        for participant_id, color in original_colors.items():
            participants[participant_id].color = color

    r.destroy()
    if not frames_pil:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_pil[0].save(
        out_path, save_all=True, append_images=frames_pil[1:], duration=int(1000 / fps), loop=0
    )


# ---------------------------------------------------------------------------
# Process one clip
# ---------------------------------------------------------------------------
def process_clip(
    info,
    takeover_frame,
    ds_name,
    gif_dir,
    show_history,
    roi_radius,
    roi_outer_radius,
    configs,
    gif_config_name,
    base_seed=DEFAULT_RANDOM_SEED,
    sampling_seed=None,
):
    """Parse one clip, run RoI multi-vehicle planning, and render a GIF."""
    results = []
    try:
        participants, map_ = parse_scenario(info)
    except Exception:
        return results

    # Preserve the ego selected during clip discovery.  Silently replacing it
    # would make a supposedly fixed debug clip a different experiment.
    ego_id = info.get("ego_id")
    if ego_id is not None:
        if ego_id not in participants or not participants[ego_id].trajectory.has_state(
            takeover_frame
        ):
            return results
    else:
        # Fall back to the longest active vehicle trajectory.
        veh = [(pid, p) for pid, p in participants.items() if isinstance(p, Vehicle)]
        if veh:
            active = [(pid, p) for pid, p in veh if p.trajectory.has_state(takeover_frame)]
            if active:
                ego_id = max(active, key=lambda x: len(x[1].trajectory.history_states))[0]
            else:
                ego_id = max(veh, key=lambda x: len(x[1].trajectory.history_states))[0]
    if ego_id is None or not participants[ego_id].trajectory.has_state(takeover_frame):
        return results
    participants[ego_id].color = "light-pink"

    veh_ids = [
        pid
        for pid, participant in participants.items()
        if isinstance(participant, Vehicle) and len(participant.trajectory.history_states) >= 10
    ]
    s_label = f"{scenario_label(info)}_t{takeover_frame}"
    route_map = {}
    route_failed_ids = set()
    max_planning_ms = max(config.horizon_steps * config.step_ms for config in configs.values())
    max_routes = max(config.max_routes_per_agent for config in configs.values())
    route_end_frame = takeover_frame + _simulation_ms(ds_name) + max_planning_ms
    planner_seeds = {
        cfg_name: _planner_seed(base_seed, ds_name, info, takeover_frame, cfg_name)
        for cfg_name in configs
    }
    update_local_route_map(
        participants,
        map_,
        takeover_frame,
        ego_id,
        roi_outer_radius,
        route_map,
        end_frame=route_end_frame,
        failed_ids=route_failed_ids,
        max_routes=max_routes,
    )

    # ---- per-frame average active vehicles in a fixed 5 s window ----
    window_end = takeover_frame + 5000
    fps = DATASET_FPS.get(ds_name, 10)
    step = max(1, int(1000 / fps))
    active_sum = 0
    active_count = 0
    for f in range(takeover_frame, window_end, step):
        cnt = sum(1 for pid in veh_ids if participants[pid].trajectory.has_state(f))
        active_sum += cnt
        active_count += 1
    avg_active_vehicles = active_sum / active_count if active_count > 0 else float(len(veh_ids))

    # ---- Initial multi-vehicle plan and timing for three configs ----
    pids_all = [
        pid for pid, p in participants.items() if takeover_frame in p.trajectory.history_states
    ]
    cam_pos = Point(0.0, 0.0)
    if ego_id in participants and participants[ego_id].trajectory.has_state(takeover_frame):
        s = participants[ego_id].trajectory.get_state(takeover_frame)
        cam_pos = Point(s.x, s.y)

    timing_cam = BEVCamera(id_=0, map_=map_, perception_range=200)
    timing_renderer = MatplotlibRenderer(
        xlim=(-60, 60), ylim=(-40, 40), resolution=(1200, 800), auto_scale=False
    )

    for cfg_name, cfg in configs.items():
        planner_seed = planner_seeds[cfg_name]
        _reset_planner_random_state(planner_seed)
        model = LimSimBehaviorModel(cfg)
        t0 = time.perf_counter()
        try:
            initial_plan = model.plan(
                participants=participants,
                map_=map_,
                frame=takeover_frame,
                route_map=route_map,
                ego_id=ego_id,
                roi_radius=roi_radius,
                roi_outer_radius=roi_outer_radius,
            )
        except Exception:
            initial_plan = None
        plan_time = time.perf_counter() - t0
        if initial_plan is None:
            collision_pairs, total_pairs = 0, 0
            initial_roi_vehicles = 0
            initial_background_vehicles = 0
            initial_groups = 0
            initial_joint_groups = 0
        else:
            collision_pairs, total_pairs = check_planned_collisions(
                initial_plan.trajectories, takeover_frame, participants
            )
            initial_roi_vehicles = len(initial_plan.roi_agent_ids)
            initial_background_vehicles = len(initial_plan.background_agent_ids)
            initial_groups = len(initial_plan.groups)
            initial_joint_groups = sum(len(group) > 1 for group in initial_plan.groups)

        t0 = time.perf_counter()
        try:
            timing_cam._position = None
            gd, _, _ = timing_cam.update(
                takeover_frame, participants, pids_all, set(), set(), cam_pos
            )
        except Exception:
            gd = {
                "frame": takeover_frame,
                "map_data": {"road_id_to_remove": [], "road_elements": []},
                "participant_data": {
                    "participant_id_to_create": [],
                    "participant_id_to_remove": [],
                    "participants": [],
                },
                "metadata": {
                    "perception_range": timing_cam._perception_range,
                    "sensor_position": cam_pos,
                    "sensor_yaw": 0.0,
                },
            }
        timing_renderer.update(gd)
        render_time = time.perf_counter() - t0

        results.append(
            {
                "dataset": ds_name,
                "scenario": s_label,
                "ego_id": ego_id,
                "config": cfg_name,
                "takeover_frame": takeover_frame,
                "base_seed": base_seed,
                "sampling_seed": sampling_seed if sampling_seed is not None else "",
                "planner_seed": planner_seed,
                "roi_radius_m": roi_radius,
                "roi_outer_radius_m": roi_outer_radius,
                "avg_active_vehicles": round(avg_active_vehicles, 2),
                "mcts_iterations": cfg.mcts_iterations,
                "horizon_steps": cfg.horizon_steps,
                "max_group_size": cfg.max_group_size,
                "interaction_distance": cfg.interaction_distance,
                "plan_time_s": round(plan_time, 6),
                "render_time_s": round(render_time, 6),
                "initial_roi_vehicles": initial_roi_vehicles,
                "initial_background_vehicles": initial_background_vehicles,
                "initial_interaction_groups": initial_groups,
                "initial_joint_groups": initial_joint_groups,
                "planned_collision_pairs": collision_pairs,
                "planned_total_pairs": total_pairs,
                "closed_loop_cycles": "",
                "closed_loop_completion": "",
                "closed_loop_avg_plan_ms": "",
                "closed_loop_avg_roi_vehicles": "",
                "closed_loop_avg_background_vehicles": "",
                "closed_loop_avg_planned_vehicles": "",
                "closed_loop_avg_controlled_vehicles": "",
                "closed_loop_avg_new_takeovers": "",
                "closed_loop_avg_replayed_candidates": "",
                "closed_loop_avg_interaction_groups": "",
                "closed_loop_multi_agent_cycles": "",
                "closed_loop_owned_vehicles": "",
                "closed_loop_retired_vehicles": "",
                "closed_loop_collision_events": "",
                "closed_loop_collision_steps": "",
                "closed_loop_status": "",
                "closed_loop_error": "",
                "gif_frames": "",
            }
        )
        del model
    timing_renderer.destroy()
    del timing_renderer, timing_cam

    # ---- RoI multi-vehicle closed loop for each config ----
    recorded_snapshots = snapshot_vehicle_trajectories(participants)
    gif_rollout = None
    for idx, (cfg_name, cfg) in enumerate(configs.items()):
        restore_recorded_snapshots(participants, recorded_snapshots)
        config_route_map = dict(route_map)
        config_failed_route_ids = set(route_failed_ids)
        _reset_planner_random_state(planner_seeds[cfg_name])
        model = LimSimBehaviorModel(cfg)
        controller = InteractiveReplayController(
            model=model,
            participants=participants,
            map_=map_,
            ego_id=ego_id,
            route_map=config_route_map,
            recorded_snapshots=recorded_snapshots,
            roi_radius=roi_radius,
            roi_outer_radius=roi_outer_radius,
            route_updater=partial(
                update_local_route_map,
                end_frame=route_end_frame,
                failed_ids=config_failed_route_ids,
                max_routes=max_routes,
            ),
        )
        closed_loop = controller.run(
            start_frame=takeover_frame,
            steps=int(_simulation_ms(ds_name) / cfg.step_ms),
        )
        if cfg_name == gif_config_name:
            gif_rollout = closed_loop
        expected = closed_loop.expected_cycles

        results[idx]["closed_loop_cycles"] = closed_loop.num_cycles
        results[idx]["closed_loop_completion"] = round(closed_loop.num_cycles / max(expected, 1), 3)
        results[idx]["closed_loop_avg_plan_ms"] = (
            round(statistics.mean(closed_loop.plan_times) * 1000, 1)
            if closed_loop.plan_times
            else 0
        )
        results[idx]["closed_loop_avg_roi_vehicles"] = round(closed_loop.avg_roi_vehicles, 2)
        results[idx]["closed_loop_avg_background_vehicles"] = round(
            closed_loop.avg_background_vehicles, 2
        )
        results[idx]["closed_loop_avg_planned_vehicles"] = round(
            closed_loop.avg_planned_vehicles, 2
        )
        results[idx]["closed_loop_avg_controlled_vehicles"] = round(
            closed_loop.avg_controlled_vehicles, 2
        )
        results[idx]["closed_loop_avg_new_takeovers"] = round(closed_loop.avg_new_takeovers, 2)
        results[idx]["closed_loop_avg_replayed_candidates"] = round(
            closed_loop.avg_replayed_candidates, 2
        )
        results[idx]["closed_loop_avg_interaction_groups"] = round(
            closed_loop.avg_interaction_groups, 2
        )
        results[idx]["closed_loop_multi_agent_cycles"] = closed_loop.multi_agent_cycles
        results[idx]["closed_loop_owned_vehicles"] = closed_loop.owned_vehicle_count
        results[idx]["closed_loop_retired_vehicles"] = closed_loop.retired_vehicle_count
        results[idx]["closed_loop_collision_events"] = closed_loop.collision_events
        results[idx]["closed_loop_collision_steps"] = closed_loop.collision_steps
        results[idx]["closed_loop_status"] = closed_loop.status
        results[idx]["closed_loop_error"] = closed_loop.error_message
        restore_recorded_snapshots(participants, recorded_snapshots)

    # ---- GIF (one selected configuration) ----
    gif_path = gif_dir / (f"{ds_name}_{s_label}_{gif_config_name}_seed{base_seed}.gif")
    if gif_rollout is not None:
        restore_recorded_snapshots(participants, recorded_snapshots)
        apply_rollout_states(participants, gif_rollout.rollout_states)

        full_ego_frames = sorted(recorded_snapshots[ego_id])
        window_end = takeover_frame + _simulation_ms(ds_name)
        render_end = (
            min(window_end, gif_rollout.simulation_frames[-1])
            if gif_rollout.simulation_frames
            else takeover_frame
        )
        his_cut = takeover_frame - int(HISTORY_SECONDS * 1000)
        history_frames = [frame for frame in full_ego_frames if his_cut <= frame <= takeover_frame]
        if takeover_frame not in history_frames:
            history_frames.append(takeover_frame)
        future_frames = [frame for frame in full_ego_frames if takeover_frame < frame <= render_end]
        interpolate_controlled_rollout_for_render(
            participants,
            recorded_snapshots,
            gif_rollout.rollout_states,
            gif_rollout.controlled_ids_by_frame,
            ego_id,
            takeover_frame,
            future_frames,
        )
        playback = sorted(set(history_frames + future_frames))
        controlled_ids_by_frame = expand_controlled_ids_for_render(
            gif_rollout.controlled_ids_by_frame,
            gif_rollout.simulation_frames,
            playback,
            takeover_frame,
        )

        if len(playback) >= 2:
            render_gif(
                participants,
                map_,
                playback,
                ego_id,
                gif_rollout.ego_plans,
                gif_path,
                fps=DATASET_FPS.get(ds_name, 10),
                show_history=show_history,
                controlled_ids_by_frame=controlled_ids_by_frame,
            )

        for row in results:
            row["gif_frames"] = len(playback)
        restore_recorded_snapshots(participants, recorded_snapshots)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def cmd_benchmark(
    datasets=None,
    num_scenarios=100,
    show_history=True,
    roi_radius=DEFAULT_ROI_RADIUS_M,
    roi_outer_radius=DEFAULT_ROI_OUTER_RADIUS_M,
    config_names=None,
    seed=DEFAULT_RANDOM_SEED,
    file_id=None,
    takeover_frame=None,
):
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if file_id is not None and file_id < 0:
        raise ValueError("file_id must be non-negative.")
    if takeover_frame is not None and takeover_frame < 0:
        raise ValueError("takeover_frame must be non-negative.")

    if config_names is None:
        configs = CONFIGS
    else:
        configs = {name: CONFIGS[name] for name in config_names}
    if not configs:
        raise ValueError("At least one LimSim configuration is required.")
    gif_config_name = "balanced_demo" if "balanced_demo" in configs else next(iter(configs))

    all_scenarios = {
        "highD": [
            {
                "dataset": "highD",
                "file_id": i,
                "folder": str(DATA_ROOT / "highD" / "data"),
                "map_config": HIGHD_MAP_CONFIG,
                "map_dir": str(DATA_ROOT / "highD_map"),
            }
            for i in range(1, 61)
        ],
        "inD": [
            {
                "dataset": "inD",
                "file_id": i,
                "folder": str(DATA_ROOT / "inD" / "data"),
                "map_config": IND_MAP_CONFIG,
                "map_dir": str(DATA_ROOT / "inD_map"),
            }
            for i in range(33)
        ],
        "rounD": [
            {
                "dataset": "rounD",
                "file_id": i,
                "folder": str(DATA_ROOT / "rounD" / "data"),
                "map_config": ROUND_MAP_CONFIG,
                "map_dir": str(DATA_ROOT / "rounD_map"),
            }
            for i in range(24)
        ],
    }

    womd_dir = DATA_ROOT / "WOMD"
    if womd_dir.exists():
        import glob

        womd_list = []
        for tf_path in sorted(glob.glob(str(womd_dir / "*.tfrecord*"))):
            fname = os.path.basename(tf_path)
            womd_list.append(
                {
                    "dataset": "WOMD",
                    "file_name": fname,
                    "folder": str(womd_dir),
                }
            )
        all_scenarios["WOMD"] = womd_list

    if datasets is None:
        datasets = list(all_scenarios.keys())
    if file_id is not None or takeover_frame is not None:
        selected_datasets = list(dict.fromkeys(datasets))
        if len(selected_datasets) != 1:
            raise ValueError("--file-id/--takeover-frame require exactly one dataset.")
        if selected_datasets[0] == "WOMD":
            raise ValueError(
                "--file-id/--takeover-frame select LevelX clips only; WOMD uses scenario IDs."
            )
        if takeover_frame is not None and file_id is None:
            raise ValueError("--takeover-frame requires --file-id.")
    all_scenarios = {k: all_scenarios[k] for k in datasets if k in all_scenarios}

    if file_id is not None:
        ds_name = next(iter(all_scenarios), None)
        if ds_name is None:
            raise ValueError("The selected dataset is not available.")
        matching_infos = [info for info in all_scenarios[ds_name] if info.get("file_id") == file_id]
        if not matching_infos:
            raise ValueError(f"{ds_name} file_id {file_id} is outside the available range.")
        all_scenarios[ds_name] = matching_infos

    all_rows = []
    for ds_name, file_infos in all_scenarios.items():
        if not file_infos:
            print(f"[{ds_name}] — no files, skip")
            continue

        ds_dir = OUT_ROOT / ds_name
        gif_dir = ds_dir / "gifs"
        ds_dir.mkdir(parents=True, exist_ok=True)
        gif_dir.mkdir(parents=True, exist_ok=True)

        dataset_seed = _dataset_sampling_seed(seed, ds_name)
        clips = sample_clips(
            file_infos,
            num_scenarios,
            seed=dataset_seed,
            takeover_frame=takeover_frame,
        )
        if not clips:
            if takeover_frame is not None:
                raise ValueError(
                    f"{ds_name} file_id {file_id} has no viable clip at "
                    f"takeover_frame {takeover_frame}."
                )
            print(f"[{ds_name}] — no clips, skip")
            continue
        print(f"[{ds_name}] — {len(clips)} clips")

        csv_files = {}
        csv_writers = {}
        for cfg_name in configs:
            path = ds_dir / f"{cfg_name}.csv"
            f = open(path, "w", newline="")
            csv_files[cfg_name] = f
            csv_writers[cfg_name] = None

        results = []
        for info, tf in tqdm(clips, desc=ds_name, unit="clip"):
            try:
                rows = process_clip(
                    info,
                    tf,
                    ds_name,
                    gif_dir,
                    show_history,
                    roi_radius,
                    roi_outer_radius,
                    configs,
                    gif_config_name,
                    base_seed=seed,
                    sampling_seed=dataset_seed,
                )
            except Exception as error:
                tqdm.write(
                    f"[{ds_name}] {scenario_label(info)} t={tf} failed: "
                    f"{type(error).__name__}: {error}"
                )
                rows = []
            results.extend(rows)
            for row in rows:
                cfg = row["config"]
                if csv_writers[cfg] is None:
                    csv_writers[cfg] = csv.DictWriter(csv_files[cfg], fieldnames=list(row.keys()))
                    csv_writers[cfg].writeheader()
                csv_writers[cfg].writerow(row)
                csv_files[cfg].flush()
            gc.collect()

        for f in csv_files.values():
            f.close()
        all_rows.extend(results)

    # summary
    if all_rows:
        path = OUT_ROOT / "summary.csv"
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nSummary: {len(all_rows)} rows → {path}")

        header = f"{'ds':>8s}  {'config':16s}  {'plan_ms':>7s}  {'rend_ms':>7s}  {'roi':>5s}  {'ctrl':>5s}  {'joint%':>7s}  {'plan_coll%':>10s}  {'loop_coll':>9s}  {'loop_ms':>7s}  {'loop%':>6s}  {'n':>4s}"
        print(header)
        print("-" * len(header))
        for ds in sorted(set(r["dataset"] for r in all_rows)):
            for cfg in configs:
                rows = [r for r in all_rows if r["dataset"] == ds and r["config"] == cfg]
                if rows:
                    plan_collision_rates = [
                        int(row["planned_collision_pairs"])
                        / max(int(row["planned_total_pairs"]), 1)
                        for row in rows
                    ]
                    loop_collisions = sum(
                        int(row.get("closed_loop_collision_events", 0) or 0) for row in rows
                    )
                    loop_completion = statistics.mean(
                        float(row["closed_loop_completion"])
                        for row in rows
                        if row["closed_loop_completion"] != ""
                    )
                    loop_times = [
                        float(row["closed_loop_avg_plan_ms"])
                        for row in rows
                        if row.get("closed_loop_avg_plan_ms") != ""
                    ]
                    joint_rates = [
                        int(row["closed_loop_multi_agent_cycles"])
                        / max(int(row["closed_loop_cycles"]), 1)
                        for row in rows
                    ]
                    print(
                        f"{ds:>8s}  {cfg:16s}  "
                        f"{statistics.mean(float(row['plan_time_s']) * 1000 for row in rows):7.1f}  "
                        f"{statistics.mean(float(row['render_time_s']) * 1000 for row in rows):7.1f}  "
                        f"{statistics.mean(float(row['closed_loop_avg_roi_vehicles']) for row in rows):5.1f}  "
                        f"{statistics.mean(float(row['closed_loop_avg_controlled_vehicles']) for row in rows):5.1f}  "
                        f"{statistics.mean(joint_rates):7.1%}  "
                        f"{statistics.mean(plan_collision_rates):10.1%}  "
                        f"{loop_collisions:9d}  "
                        f"{statistics.mean(loop_times) if loop_times else 0.0:7.1f}  "
                        f"{loop_completion:6.1%}  {len(rows):4d}"
                    )


def main():
    parser = argparse.ArgumentParser(description="LimSim RoI multi-vehicle takeover benchmark")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["highD", "inD", "rounD", "WOMD"],
        help="Datasets to benchmark",
    )
    parser.add_argument("--scenarios", type=int, default=100, help="Scenarios per dataset")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Base seed for clip sampling and deterministic per-config planner seeds",
    )
    parser.add_argument(
        "--file-id",
        type=int,
        default=None,
        help="LevelX recording ID; requires selecting exactly one non-WOMD dataset",
    )
    parser.add_argument(
        "--takeover-frame",
        type=int,
        default=None,
        help="Exact LevelX takeover frame in milliseconds; requires --file-id",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIGS),
        default=None,
        help="Planner configurations to run (default: all)",
    )
    parser.add_argument("--no-history", action="store_true", help="Skip past traces in GIF")
    parser.add_argument(
        "--roi-radius",
        type=float,
        default=DEFAULT_ROI_RADIUS_M,
        help="Inner RoI radius in meters; vehicles inside are PDP candidates",
    )
    parser.add_argument(
        "--roi-outer-radius",
        type=float,
        default=DEFAULT_ROI_OUTER_RADIUS_M,
        help="Outer RoI radius in meters; vehicles in the outer ring are obstacles",
    )
    args = parser.parse_args()
    if args.roi_radius <= 0:
        parser.error("--roi-radius must be positive")
    if args.roi_outer_radius < args.roi_radius:
        parser.error("--roi-outer-radius must be greater than or equal to --roi-radius")
    try:
        cmd_benchmark(
            datasets=args.datasets,
            num_scenarios=args.scenarios,
            show_history=not args.no_history,
            roi_radius=args.roi_radius,
            roi_outer_radius=args.roi_outer_radius,
            config_names=args.configs,
            seed=args.seed,
            file_id=args.file_id,
            takeover_frame=args.takeover_frame,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
