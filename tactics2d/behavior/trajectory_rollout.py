# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared receding-horizon replay for trajectory-producing behavior models."""

from typing import Callable, Dict, Iterable, Optional

import numpy as np

from tactics2d.participant.trajectory import Trajectory

from .results import TrajectoryRolloutResult
from .trajectory_processing import resample_participants

DEFAULT_WARMUP_MS = 1000
_MIN_HEADING_STEP = 1e-6


def resample_replayed_trajectory(
    ego_id, recorded_states, stable_freq, fps, history_frames, future_frames, samples
) -> Trajectory:
    """Lay a replayed future back onto the recording's frame grid."""

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


class TrajectoryReplayRunner:
    """Run a shared closed-loop schedule around a model-specific predictor."""

    def __init__(
        self,
        step_ms: int,
        horizon_ms: int,
        commit_steps: int = 1,
        *,
        suppress_prediction_errors: bool = False,
    ):
        if commit_steps < 1:
            raise ValueError("commit_steps must be at least 1.")
        self.step_ms = int(step_ms)
        self.horizon_ms = int(horizon_ms)
        self.commit_steps = int(commit_steps)
        self.suppress_prediction_errors = suppress_prediction_errors

    def run(
        self,
        participants: Dict[object, object],
        ego_id: object,
        predictor: Callable,
        frame_ms: Optional[int] = None,
        controlled_ids: Optional[Iterable[object]] = None,
        prepare: Optional[Callable] = None,
    ) -> TrajectoryRolloutResult:
        """Replay participants using ``predictor(participants, frame, ids)``."""

        controlled = [ego_id] if controlled_ids is None else list(controlled_ids)
        if ego_id not in controlled:
            raise ValueError(f"controlled_ids must contain ego_id {ego_id!r}; got {controlled!r}")
        participants = resample_participants(participants, self.step_ms)
        ego = participants[ego_id]
        recorded_frames = sorted(ego.trajectory.frames)
        if not recorded_frames:
            raise ValueError(f"participant {ego_id!r} has no recorded trajectory to replay.")
        requested = recorded_frames[0] + DEFAULT_WARMUP_MS if frame_ms is None else frame_ms
        take_over = next((frame for frame in recorded_frames if frame >= requested), None)
        if take_over is None:
            raise ValueError(
                f"participant {ego_id!r} has no frame at or after {requested} ms; its track "
                f"spans {recorded_frames[0]}-{recorded_frames[-1]} ms."
            )
        split = recorded_frames.index(take_over)
        history_frames = recorded_frames[: split + 1]
        future_frames = [
            frame for frame in recorded_frames if take_over < frame <= take_over + self.horizon_ms
        ]
        saved_states = {
            agent_id: dict(participants[agent_id].trajectory.history_states)
            for agent_id in controlled
        }
        saved_stable_freq = {
            agent_id: participants[agent_id].trajectory.stable_freq for agent_id in controlled
        }
        stable_freq = ego.trajectory.stable_freq
        fps = ego.trajectory.fps
        for agent_id in controlled:
            participants[agent_id].trajectory.replace_states(
                {
                    frame: state
                    for frame, state in saved_states[agent_id].items()
                    if frame <= take_over
                }
            )
        context = prepare(participants, take_over, controlled) if prepare is not None else None

        total_steps = self.horizon_ms // self.step_ms
        completed = 0
        current = take_over
        samples = []
        plans = {}
        try:
            while completed < total_steps:
                try:
                    predicted = predictor(participants, current, controlled, context)
                except Exception:
                    if self.suppress_prediction_errors:
                        break
                    raise
                if ego_id not in predicted:
                    break
                plan = predicted[ego_id]
                ahead = sorted(frame for frame in plan.frames if frame > current)
                if not ahead:
                    break
                plans[current] = [
                    (frame, plan.get_state(frame).x, plan.get_state(frame).y) for frame in ahead
                ]
                count = min(self.commit_steps, len(ahead), total_steps - completed)
                window_end = ahead[count - 1]
                for agent_id in controlled:
                    own = predicted.get(agent_id)
                    if own is None:
                        continue
                    for frame in sorted(
                        frame for frame in own.frames if current < frame <= window_end
                    ):
                        state = own.get_state(frame)
                        participants[agent_id].trajectory.add_state(state)
                        if agent_id == ego_id:
                            samples.append((frame, state.x, state.y))
                completed += count
                current = window_end
        finally:
            for agent_id in controlled:
                participants[agent_id].trajectory.replace_states(saved_states[agent_id])
                participants[agent_id].trajectory.stable_freq = saved_stable_freq[agent_id]

        return TrajectoryRolloutResult(
            ego_id=ego_id,
            frames=history_frames + future_frames,
            trajectory=resample_replayed_trajectory(
                ego_id,
                saved_states[ego_id],
                stable_freq,
                fps,
                history_frames,
                future_frames,
                samples,
            ),
            plans=plans,
            cycles=len(plans),
            controlled_ids=controlled,
        )
