# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Token batch assembly from native participants and maps."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tactics2d.behavior.limsim.roi import RoISelector
from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle

from .config import SmartConfig
from .map_tokenizer import MapTokenizer
from .motion_tokenizer import MotionTokenizer
from .schema import AgentType, SmartTokenBatch

# Participant classes the motion codebook covers, with its own type index.
_PARTICIPANT_AGENT_TYPES = (
    (Vehicle, AgentType.VEHICLE),
    (Cyclist, AgentType.CYCLIST),
    (Pedestrian, AgentType.PEDESTRIAN),
)


def participant_agent_type(participant) -> Optional[int]:
    """Return the motion codebook type index of a participant.

    Args:
        participant: A tactics2d participant.

    Returns:
        The codebook index, or None if the participant has no category.
    """

    for participant_class, index in _PARTICIPANT_AGENT_TYPES:
        if isinstance(participant, participant_class):
            return index
    return None


def _observed_frames(participants: Dict[object, object]) -> List[int]:
    """Return every frame any participant is observed at, ascending.

    Args:
        participants (Dict[object, object]): All participants, keyed by id.

    Returns:
        The frame numbers in ascending order. The unit is millisecond (ms).
    """

    frames = set()
    for participant in participants.values():
        frames.update(int(frame) for frame in participant.trajectory.frames)
    return sorted(frames)


def _dimensions(participant) -> Tuple[float, float, float]:
    """Return a participant's extent as length, width and height.

    Args:
        participant: A tactics2d participant.

    Returns:
        The length, width and height in metres, zero for a missing dimension.
    """

    extent = []
    for name in ("length", "width", "height"):
        value = getattr(participant, name, None)
        extent.append(0.0 if value is None else float(value))
    return (extent[0], extent[1], extent[2])


class SmartBatchBuilder:
    """Turn tactics2d participants and a native map into a SMART token batch.

    Histories are tokenized against the motion codebook and the map against the
    map codebook; history gaps are zero-filled, never interpolated.

    Attributes:
        config (SmartConfig): The builder configuration.
        motion_tokenizer (MotionTokenizer): Quantizer for agent histories.
        map_tokenizer (MapTokenizer): Quantizer for the map.
    """

    def __init__(
        self,
        config: Optional[SmartConfig] = None,
        motion_tokenizer: Optional[MotionTokenizer] = None,
        map_tokenizer: Optional[MapTokenizer] = None,
    ):
        """Initialize the builder and its tokenizers.

        Args:
            config (Optional[SmartConfig], optional): Builder configuration.
                Defaults to None.
            motion_tokenizer (Optional[MotionTokenizer], optional): Tokenizer
                to reuse. Defaults to None.
            map_tokenizer (Optional[MapTokenizer], optional): Tokenizer to
                reuse. Defaults to None.
        """

        self.config = config if config is not None else SmartConfig()
        self.motion_tokenizer = (
            motion_tokenizer if motion_tokenizer is not None else MotionTokenizer(self.config)
        )
        self.map_tokenizer = (
            map_tokenizer if map_tokenizer is not None else MapTokenizer(self.config)
        )

    def history_frames(
        self, frame: int, observed_frames: Optional[Sequence[int]] = None
    ) -> List[int]:
        """Return the frames that feed one tokenized history.

        Args:
            frame (int): Newest observed frame. The unit is millisecond (ms).
            observed_frames (Optional[Sequence[int]], optional): The scenario's
                frame grid. Defaults to None, which lays the window on the
                ``config.step_ms`` lattice.

        Returns:
            The frame numbers in ascending order, ending at the newest frame at
            or before *frame*.
        """

        step = self.config.step_ms
        newest = int(frame)
        count = self.config.history_steps
        if observed_frames is None:
            return [newest - step * (count - 1 - index) for index in range(count)]
        observed = sorted({int(frame_) for frame_ in observed_frames if int(frame_) <= newest})
        if len(observed) < count:
            return [
                newest - step * (count - len(observed) - 1 - index)
                for index in range(count - len(observed))
            ] + observed
        return observed[-count:]

    def select_agent_ids(
        self, participants: Dict[object, object], frame: int, center_id: Optional[object] = None
    ) -> List[object]:
        """Choose the agents to model, deterministically.

        Args:
            participants (Dict[object, object]): All participants, keyed by id.
            frame (int): Newest observed frame. The unit is millisecond (ms).
            center_id (Optional[object], optional): Participant the selection
                radius is measured from. Defaults to None.

        Returns:
            The modelled participant ids, nearest to the centre first.

        Raises:
            ValueError: If no modelled participant is active at *frame*, or if
                *center_id* is not one of them.
        """

        candidates = [
            agent_id
            for agent_id, participant in participants.items()
            if participant_agent_type(participant) is not None
            and participant.trajectory.has_state(frame)
        ]
        if self.config.vehicle_only:
            candidates = [
                agent_id for agent_id in candidates if isinstance(participants[agent_id], Vehicle)
            ]
        if not candidates:
            raise ValueError(f"no modelled participant is active at frame {frame}.")
        if center_id is None:
            center_id = candidates[0]
        elif center_id not in candidates:
            raise ValueError(
                f"center_id {center_id!r} is not a modelled participant active at frame {frame}."
            )

        center = participants[center_id].trajectory.get_state(frame).location
        selection = RoISelector.select_by_radius(
            participants=participants,
            frame=frame,
            center=center,
            radius=self.config.predicted_radius,
            candidate_ids=candidates,
        )
        # Distance ties fall back to participant order, so the cap is
        # reproducible and ids need no ordering of their own.
        order = {agent_id: index for index, agent_id in enumerate(participants)}
        ranked = sorted(
            selection.agent_ids,
            key=lambda agent_id: (
                spatial.euclidean_distance(
                    participants[agent_id].trajectory.get_state(frame).location, center
                ),
                order[agent_id],
            ),
        )
        return ranked[: self.config.max_predicted_agents]

    def build(
        self,
        participants: Dict[object, object],
        map_: Map,
        frame: int,
        center_id: Optional[object] = None,
    ) -> SmartTokenBatch:
        """Assemble a token batch for one scenario at one frame.

        Args:
            participants (Dict[object, object]): All participants, keyed by id.
            map_ (Map): The native map to tokenize.
            frame (int): Newest observed frame. The unit is millisecond (ms).
            center_id (Optional[object], optional): Participant the selection
                radius is measured from. Defaults to None.

        Returns:
            The tokenized agent history and map, stamped with *frame*.

        Raises:
            ValueError: If no modelled participant is active at *frame*, or if
                *center_id* is not one of them.
        """

        # Anchor on the newest frame the scenario is actually observed at, not
        # on the requested timestamp.
        grid = _observed_frames(participants)
        observed = [frame_ for frame_ in grid if frame_ <= int(frame)]
        anchor = observed[-1] if observed else int(frame)
        # Read the map one frame on, so a light that changes as the rollout
        # begins is the state the model sees.
        later = [frame_ for frame_ in grid if frame_ > anchor]
        map_frame = later[0] if later else anchor + self.config.step_ms

        agent_ids = self.select_agent_ids(participants, anchor, center_id)
        frames = self.history_frames(anchor, grid)

        num_agent = len(agent_ids)
        history = self.config.history_steps
        positions = np.zeros((num_agent, history, 2), dtype=np.float64)
        headings = np.zeros((num_agent, history), dtype=np.float64)
        valid = np.zeros((num_agent, history), dtype=bool)
        velocity = np.zeros((num_agent, history, 2), dtype=np.float64)
        shape = np.zeros((num_agent, 3), dtype=np.float64)
        agent_type = np.zeros((num_agent,), dtype=np.int64)

        for row, agent_id in enumerate(agent_ids):
            participant = participants[agent_id]
            trajectory = participant.trajectory
            agent_type[row] = participant_agent_type(participant)
            shape[row] = _dimensions(participant)
            for column, history_frame in enumerate(frames):
                if not trajectory.has_state(history_frame):
                    continue
                state = trajectory.get_state(history_frame)
                positions[row, column] = state.location
                headings[row, column] = state.heading
                valid[row, column] = True
                observed_velocity = state.velocity
                if observed_velocity is not None:
                    velocity[row, column] = observed_velocity

        agents = self.motion_tokenizer.tokenize(
            positions, headings, valid, velocity, agent_type, agent_ids=list(agent_ids), shape=shape
        )
        map_tokens = self.map_tokenizer.build(map_, map_frame)
        return SmartTokenBatch(agents=agents, map_tokens=map_tokens, frame_ms=int(anchor))
