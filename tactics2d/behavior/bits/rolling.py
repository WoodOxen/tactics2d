# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Receding-horizon closed-loop replay for the BITS behavior model."""

from typing import Dict, Iterable, Optional

import numpy as np

from tactics2d.map.element import Map
from tactics2d.participant.trajectory import Trajectory

from ..rolling_utils import to_lattice
from .config import BitsConfig
from .schema import BitsRollingResult

# Default history kept before the take-over frame, in milliseconds.
DEFAULT_WARMUP_MS = 1000

# Steps of one plan a cycle commits before re-planning.
DEFAULT_REPLAN_INTERVAL = 20

# Smallest step, in metres, that still counts as a heading sample.
_MIN_HEADING_STEP = 1e-6


def _write_states(trajectory: Trajectory, states: Dict[int, object]) -> None:
    """Write a snapshot of states back into a trajectory."""

    trajectory._history_states = dict(states)
    trajectory._frames = sorted(states)
    trajectory._current_state = states[trajectory._frames[-1]] if trajectory._frames else None


def _resample(ego_id, recorded_states, stable_freq, fps, history_frames, future_frames, samples):
    """Lay the replayed future back onto the recorded frame grid."""

    replayed = Trajectory(id_=ego_id, fps=fps, stable_freq=stable_freq)
    for frame in history_frames:
        replayed.add_state(recorded_states[frame])
    if not future_frames or not samples:
        return replayed

    sample_frames = [frame for frame, _, _ in samples]
    xs = np.interp(future_frames, sample_frames, [x for _, x, _ in samples])
    ys = np.interp(future_frames, sample_frames, [y for _, _, y in samples])

    state_cls = type(recorded_states[history_frames[-1]])
    heading = recorded_states[history_frames[-1]].heading
    for position, frame in enumerate(future_frames):
        if position < len(future_frames) - 1:
            dx = xs[position + 1] - xs[position]
            dy = ys[position + 1] - ys[position]
            if abs(dx) > _MIN_HEADING_STEP or abs(dy) > _MIN_HEADING_STEP:
                heading = np.arctan2(dy, dx)
        replayed.add_state(
            state_cls(
                frame=frame, x=float(xs[position]), y=float(ys[position]), heading=float(heading)
            )
        )
    return replayed


class BitsRollingRunner:
    """Replay one vehicle's future in a receding-horizon loop.

    Attributes:
        model (BitsBehaviorModel): The model being replayed.
        config (BitsConfig): The model's configuration.
        horizon_ms (int): How far ahead the replay runs, in milliseconds.
        replan_interval (int): States committed per cycle.
    """

    def __init__(
        self,
        model,
        config: Optional[BitsConfig] = None,
        horizon_ms: Optional[int] = None,
        replan_interval: int = DEFAULT_REPLAN_INTERVAL,
    ):
        """Initialize the runner.

        Args:
            model (BitsBehaviorModel): The model to replay, exposing ``predict``.
            config (Optional[BitsConfig], optional): Configuration to replay with.
                Defaults to None, which uses the model's own.
            horizon_ms (int, optional): How far ahead to replay, in milliseconds.
                Defaults to None, which uses the model's planning horizon.
            replan_interval (int, optional): States committed per cycle. Defaults
                to 20, i.e. one re-plan every 2 s at 10 Hz.

        Raises:
            ValueError: If *replan_interval* is below 1.
        """

        self.model = model
        self.config = config if config is not None else model.config
        self.horizon_ms = (
            self.config.planning_steps * self.config.step_ms
            if horizon_ms is None
            else int(horizon_ms)
        )
        if replan_interval < 1:
            raise ValueError("replan_interval must be at least 1.")
        self.replan_interval = int(replan_interval)

    def run(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        frame_ms: Optional[int] = None,
        controlled_ids: Optional[Iterable[object]] = None,
    ) -> BitsRollingResult:
        """Replay the vehicles and return the re-simulated track.

        Args:
            participants (Dict[object, object]): All participants in the scenario.
            map_ (Optional[Map]): The map to predict on.
            ego_id (object): The vehicle to replay.
            frame_ms (int, optional): The take-over frame, in milliseconds.
                Defaults to None, i.e. ``DEFAULT_WARMUP_MS`` after the vehicle's
                first frame.
            controlled_ids (Optional[Iterable], optional): Every vehicle to
                re-simulate, the ego included. Defaults to None, i.e. the ego
                alone. Each is truncated at the take-over frame and replanned
                from there.

        Returns:
            The replay, resampled onto the recorded frame grid. The committed
            future of every controlled vehicle is written back into
            *participants*.

        Raises:
            ValueError: If the vehicle has no recorded trajectory, if *frame_ms*
                falls past the end of it, or if *controlled_ids* omits *ego_id*.
        """

        if controlled_ids is not None:
            controlled_ids = list(controlled_ids)
            if ego_id not in controlled_ids:
                raise ValueError(
                    "controlled_ids must contain ego_id {!r}; got {!r}".format(
                        ego_id, controlled_ids
                    )
                )
        controlled = [ego_id] if controlled_ids is None else controlled_ids

        step_ms = self.config.step_ms
        participants = to_lattice(participants, step_ms)
        ego = participants[ego_id]
        recorded_frames = sorted(ego.trajectory.frames)
        if not recorded_frames:
            raise ValueError(f"participant {ego_id!r} has no recorded trajectory to replay.")

        if frame_ms is None:
            frame_ms = recorded_frames[0] + DEFAULT_WARMUP_MS
        take_over = next((frame for frame in recorded_frames if frame >= frame_ms), None)
        if take_over is None:
            raise ValueError(
                f"participant {ego_id!r} has no frame at or after {frame_ms} ms; its track "
                f"spans {recorded_frames[0]}-{recorded_frames[-1]} ms."
            )
        index = recorded_frames.index(take_over)
        history_frames = recorded_frames[: index + 1]
        future_frames = [f for f in recorded_frames if take_over < f <= take_over + self.horizon_ms]

        # Truncate every controlled vehicle at the take-over frame.
        saved_states = {
            agent_id: dict(participants[agent_id].trajectory.history_states)
            for agent_id in controlled
        }
        stable_freq = ego.trajectory.stable_freq
        fps = ego.trajectory.fps
        for agent_id in controlled:
            _write_states(
                participants[agent_id].trajectory,
                {f: s for f, s in saved_states[agent_id].items() if f <= take_over},
            )
        recorded_states = saved_states[ego_id]

        total_future_frames = self.horizon_ms // step_ms
        samples = []
        plans = {}
        current = take_over
        steps_done = 0
        while steps_done < total_future_frames:
            # One call for the whole set, so every controlled vehicle is planned
            # against the same snapshot rather than in sequence.
            predicted = self.model.predict(participants, map_, current, agent_ids=controlled)
            if ego_id not in predicted:
                break
            plan = predicted[ego_id]
            ahead = sorted(frame for frame in plan.frames if frame > current)
            if not ahead:
                break

            # Absolute world coordinates, keyed by the frame the plan was issued at.
            plans[current] = [
                (frame, plan.get_state(frame).x, plan.get_state(frame).y) for frame in ahead
            ]

            step_count = min(self.replan_interval, len(ahead), total_future_frames - steps_done)
            window_end = ahead[step_count - 1]
            for agent_id in controlled:
                own = predicted.get(agent_id)
                if own is None:
                    continue
                for frame in sorted(f for f in own.frames if current < f <= window_end):
                    state = own.get_state(frame)
                    participants[agent_id].trajectory.add_state(state)
                    if agent_id == ego_id:
                        samples.append((frame, state.x, state.y))

            steps_done += step_count
            current = window_end

        for agent_id in controlled:
            _write_states(participants[agent_id].trajectory, saved_states[agent_id])
            participants[agent_id].trajectory.stable_freq = stable_freq
        return BitsRollingResult(
            ego_id=ego_id,
            frames=history_frames + future_frames,
            trajectory=_resample(
                ego_id, recorded_states, stable_freq, fps, history_frames, future_frames, samples
            ),
            plans=plans,
            cycles=len(plans),
            controlled_ids=list(controlled),
        )
