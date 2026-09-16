# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Agent motion tokenizer for the SMART network."""

from typing import Dict, List, Optional

import numpy as np
import torch

from .config import SmartConfig, load_codebook
from .schema import AgentType, SmartAgentTokens

# Codebook key per agent type index; any other type stays at token 0.
_CATEGORY_BY_TYPE = {
    AgentType.VEHICLE: "veh",
    AgentType.PEDESTRIAN: "ped",
    AgentType.CYCLIST: "cyc",
}

# (length, width) in metres of the footprint matched against the codebook.
_FOOTPRINT = {"veh": (4.8, 2.0), "cyc": (2.0, 1.0), "ped": (1.0, 1.0)}


def _oriented_contour(x, y, heading, length, width):
    """Build per-agent oriented footprints in the world frame.

    Corner order is ``[left-front, right-front, right-back, left-back]``; 0 minus
    3 is the heading axis read back from a token.

    Args:
        x (torch.Tensor): Centre x-coordinates of shape ``(A,)``.
        y (torch.Tensor): Centre y-coordinates of shape ``(A,)``.
        heading (torch.Tensor): Headings in radians of shape ``(A,)``.
        length (float): Footprint length along the heading in metres.
        width (float): Footprint width across the heading in metres.

    Returns:
        A ``(A, 4, 2)`` float32 tensor of corners.
    """

    cos, sin = heading.cos(), heading.sin()
    half_length, half_width = 0.5 * length, 0.5 * width
    left_front = torch.stack(
        [x + half_length * cos - half_width * sin, y + half_length * sin + half_width * cos], dim=-1
    )
    right_front = torch.stack(
        [x + half_length * cos + half_width * sin, y + half_length * sin - half_width * cos], dim=-1
    )
    right_back = torch.stack(
        [x - half_length * cos + half_width * sin, y - half_length * sin - half_width * cos], dim=-1
    )
    left_back = torch.stack(
        [x - half_length * cos - half_width * sin, y - half_length * sin + half_width * cos], dim=-1
    )
    return torch.stack([left_front, right_front, right_back, left_back], dim=1)


def _clean_heading(heading, valid):
    """Pull heading across wrap-around jumps onto the previous frame.

    Where a valid pair of consecutive frames differs by more than one radian,
    the later frame copies the earlier one's heading.

    Args:
        heading (torch.Tensor): Headings of shape ``(A, F)``, modified in place.
        valid (torch.Tensor): Validity mask of shape ``(A, F)``.
    """

    pi = float(torch.pi)
    n_frames = heading.shape[1]

    diff_raw = heading[:, :-1] - heading[:, 1:]
    diff = torch.remainder(diff_raw + pi, 2 * pi) - pi
    diff[diff > pi] -= 2 * pi
    diff[diff < -pi] += 2 * pi

    valid_pairs = valid[:, :-1] & valid[:, 1:]

    for i in range(n_frames - 1):
        change_needed = (diff[:, i : i + 1].abs() > 1.0) & valid_pairs[:, i : i + 1]
        selector = change_needed.squeeze(1)
        heading[:, i + 1][selector] = heading[:, i][selector]
        if i < n_frames - 2:
            diff_raw = heading[:, i + 1] - heading[:, i + 2]
            diff[:, i + 1] = torch.remainder(diff_raw + pi, 2 * pi) - pi
            diff[diff[:, i + 1] > pi] -= 2 * pi
            diff[diff[:, i + 1] < -pi] += 2 * pi


class MotionTokenizer:
    """Quantize agent histories into the motion codebook.

    Slot ``j`` spans frames ``[j * shift, j * shift + shift]`` and is available
    only when both its first and last frame are.

    Attributes:
        config (SmartConfig): The tokenizer configuration.
        dt (float): Frame interval in seconds.
        shift (int): Frames covered by one motion token.
        current_step (int): Index of the newest observed frame.
    """

    def __init__(self, config: Optional[SmartConfig] = None):
        """Initialize the tokenizer and load the motion codebook.

        Args:
            config (Optional[SmartConfig], optional): Tokenizer configuration.
                Defaults to ``SmartConfig()``.

        Raises:
            FileNotFoundError: If the configured codebook does not exist.
        """

        self.config = config if config is not None else SmartConfig()
        self.dt = self.config.dt
        self.shift = self.config.shift
        self.current_step = self.config.history_steps - 1
        self._load_codebook()

    def _load_codebook(self) -> None:
        """Load the motion codebook and precompute the tail token set."""

        codebook = load_codebook(self.config.motion_codebook, "motion codebook")

        for key, contour in codebook["token"].items():
            if len(contour) != self.config.token_size:
                raise ValueError(
                    f"motion codebook {self.config.motion_codebook!r} carries "
                    f"{len(contour)} {key} tokens but config.token_size is "
                    f"{self.config.token_size}; point motion_codebook at the matching file."
                )

        self._token: Dict[str, torch.Tensor] = {}
        self._token_last: Dict[str, torch.Tensor] = {}
        for key, contour in codebook["token"].items():
            contour = torch.as_tensor(np.asarray(contour), dtype=torch.float32)
            self._token[key] = contour
            self._token_last[key] = self._build_tail(codebook["token_all"][key])

    def _build_tail(self, token_all) -> torch.Tensor:
        """Build the previous-frame token set used for late-appearing agents.

        Args:
            token_all: The ``(T, 6, 4, 2)`` contour sequence of one category.

        Returns:
            A ``(T, 4, 2)`` float32 tensor.
        """

        token_all = torch.as_tensor(np.asarray(token_all), dtype=torch.float32)
        last_two = token_all[:, -2:]
        axis = last_two[:, 0, 0] - last_two[:, 0, 3]
        theta = torch.arctan2(axis[:, 1], axis[:, 0])
        cos, sin = theta.cos(), theta.sin()
        rot = theta.new_zeros(theta.shape[0], 2, 2)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = -sin
        rot[:, 1, 0] = sin
        rot[:, 1, 1] = cos
        tail = torch.bmm(last_two[:, 1], rot)
        return tail - last_two[:, 0].mean(dim=1)[:, None, :]

    def _world_tokens(
        self, src: torch.Tensor, theta: torch.Tensor, origin: torch.Tensor
    ) -> torch.Tensor:
        """Rotate a canonical token set into the world frame about a pose.

        Args:
            src (torch.Tensor): Canonical contours of shape ``(T, 4, 2)``.
            theta (torch.Tensor): Anchor headings of shape ``(N,)``.
            origin (torch.Tensor): Anchor positions of shape ``(N, 2)``.

        Returns:
            A ``(N, T, 4, 2)`` tensor of world-frame contours.
        """

        num_agent = theta.shape[0]
        token_num = src.shape[0]
        cos, sin = theta.cos(), theta.sin()
        rot = theta.new_zeros(num_agent, 2, 2)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = sin
        rot[:, 1, 0] = -sin
        rot[:, 1, 1] = cos
        flat = src.reshape(1, token_num * 4, 2).expand(num_agent, token_num * 4, 2)
        world = torch.bmm(flat, rot).reshape(num_agent, token_num, 4, 2)
        return world + origin[:, None, None, :]

    @staticmethod
    def _nearest_token(contour: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
        """Return the codebook index closest to each agent's footprint.

        Score is the mean Euclidean distance over the four corners, ties to the
        lower index; the reduce is done in numpy to match upstream's last bits.

        Args:
            contour (torch.Tensor): Target footprints of shape ``(N, 4, 2)``.
            world (torch.Tensor): Candidate tokens of shape ``(N, T, 4, 2)``.

        Returns:
            An ``(N,)`` int64 tensor of codebook indices.
        """

        target = contour.detach().cpu().numpy()[:, None, ...]
        candidates = world.detach().cpu().numpy()
        score = np.mean(np.sqrt(np.sum((target - candidates) ** 2, axis=-1)), axis=2)
        return torch.from_numpy(np.argmin(score, axis=-1))

    def _match_token(
        self,
        pos: torch.Tensor,
        valid: torch.Tensor,
        heading: torch.Tensor,
        key: str,
        extra_mask: torch.Tensor,
    ) -> tuple:
        """Match one agent-type group against the codebook over every slot.

        Args:
            pos (torch.Tensor): Positions of shape ``(N, F, 2)``.
            valid (torch.Tensor): Validity of shape ``(N, F)``.
            heading (torch.Tensor): Headings of shape ``(N, F)``.
            key (str): Codebook category, one of ``veh``/``ped``/``cyc``.
            extra_mask (torch.Tensor): Agents needing the tail token set, ``(N,)``.

        Returns:
            A tuple of:
                - token_index (torch.Tensor): ``(N, S)`` int64 codebook indices.
                - token_contour (torch.Tensor): ``(N, S, 4, 2)`` world contours.
        """

        length, width = _FOOTPRINT[key]
        src = self._token[key]
        num_agent = pos.shape[0]

        prev_heading = heading[:, 0]
        prev_pos = pos[:, 0]
        index_list: List[torch.Tensor] = []
        contour_list: List[torch.Tensor] = []

        for i in range(self.shift, pos.shape[1], self.shift):
            world = self._world_tokens(src, prev_heading, prev_pos)
            contour = _oriented_contour(pos[:, i, 0], pos[:, i, 1], heading[:, i], length, width)
            index = self._nearest_token(contour, world)
            selected = world[torch.arange(num_agent), index]

            axis = selected[:, 0] - selected[:, 3]
            # Masked after ``arctan2``, as upstream: a compacted input rounds differently.
            anchor_heading = torch.arctan2(axis[:, 1], axis[:, 0])
            anchor_valid = valid[:, i - self.shift]
            prev_heading = heading[:, i].clone()
            prev_heading[anchor_valid] = anchor_heading[anchor_valid]
            prev_pos = pos[:, i].clone()
            prev_pos[anchor_valid] = selected.mean(dim=1)[anchor_valid]

            index_list.append(index[:, None])
            contour_list.append(selected[:, None, ...])

        token_index = torch.cat(index_list, dim=1)
        token_contour = torch.cat(contour_list, dim=1)

        num_extra = int(extra_mask.sum())
        if num_extra:
            # Inference-only: second slot from the tail set, anchored one frame back.
            world = self._world_tokens(
                self._token_last[key],
                heading[extra_mask, self.current_step - 1],
                pos[extra_mask, self.current_step - 1],
            )
            contour = _oriented_contour(
                pos[extra_mask, self.current_step, 0],
                pos[extra_mask, self.current_step, 1],
                heading[extra_mask, self.current_step],
                length,
                width,
            )
            index = self._nearest_token(contour, world)
            token_index[extra_mask, 1] = index
            token_contour[extra_mask, 1] = world[torch.arange(num_extra), index]

        return token_index, token_contour

    def tokenize(
        self,
        positions,
        headings,
        valid_mask,
        velocity,
        agent_type,
        agent_ids: Optional[List[object]] = None,
        shape=None,
    ) -> SmartAgentTokens:
        """Quantize agent histories into motion tokens.

        Args:
            positions: Positions of shape ``(A, F, 2)`` in metres, world frame.
            headings: Headings of shape ``(A, F)`` in radians.
            valid_mask: Per-frame presence of shape ``(A, F)``; usable frames are decided here.
            velocity: Velocities of shape ``(A, F, 2)``, used only to impute one missing pose.
            agent_type: Codebook type index per agent, shape ``(A,)``.
            agent_ids (Optional[List[object]], optional): Id per agent row. Defaults to the index.
            shape (optional): Agent extent as ``(A, 3)`` length, width, height. Defaults to zeros.

        Returns:
            The tokenized history.

        Raises:
            ValueError: If an input shape is inconsistent or fewer than ``shift + 1`` frames.
        """

        pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32).clone()
        head = torch.as_tensor(np.asarray(headings), dtype=torch.float32).clone()
        valid = torch.as_tensor(np.asarray(valid_mask), dtype=torch.bool).clone()
        vel = torch.as_tensor(np.asarray(velocity), dtype=torch.float32)
        types = torch.as_tensor(np.asarray(agent_type), dtype=torch.int64)

        if pos.dim() != 3 or pos.shape[-1] != 2:
            raise ValueError(f"positions must have shape (A, F, 2), got {tuple(pos.shape)}")
        if head.shape != pos.shape[:2] or valid.shape != pos.shape[:2]:
            raise ValueError("headings and valid_mask must have shape (A, F)")
        if vel.shape != pos.shape:
            raise ValueError("velocity must have shape (A, F, 2)")
        if types.shape != (pos.shape[0],):
            raise ValueError("agent_type must have shape (A,)")
        if pos.shape[1] < self.shift + 1:
            raise ValueError(
                f"need at least shift + 1 = {self.shift + 1} frames, got {pos.shape[1]}"
            )

        step = self.current_step
        num_agent = pos.shape[0]

        # A history frame counts only with its predecessor; frame 0 never counts
        # alone. Must run before the imputation patches below.
        history = step + 1
        valid[:, 1:history] = valid[:, : history - 1] & valid[:, 1:history]
        valid[:, 0] = False

        # Impute backwards from velocity: invalid at the newest frame but positioned there.
        interplote = (valid[:, step] == False) & (pos[:, step, 0] != 0)  # noqa: E712
        pos[interplote, step - 1, :] = pos[interplote, step, :] - vel[interplote, step, :] * self.dt
        valid[interplote, step - 1 : step + 1] = True
        head[interplote, step - 1] = head[interplote, step]
        vel[interplote, step - 1] = vel[interplote, step]

        _clean_heading(head, valid)

        matching_extra = valid[:, step] & (~valid[:, step - self.shift])

        # First frame unobserved but positioned: only the validity flag is missing.
        interplote_first = (valid[:, 0] == False) & (pos[:, 0, 0] != 0)  # noqa: E712
        valid[interplote_first, 0] = True

        slot_valid = valid.unfold(1, self.shift + 1, self.shift)
        token_valid = slot_valid[:, :, 0] & slot_valid[:, :, -1]
        num_slot = token_valid.shape[1]

        token_index = torch.zeros((num_agent, num_slot), dtype=torch.int64)
        token_contour = torch.zeros((num_agent, num_slot, 4, 2), dtype=torch.float32)
        for type_index, key in _CATEGORY_BY_TYPE.items():
            mask = types == type_index
            if not bool(mask.any()):
                continue
            matched_index, matched_contour = self._match_token(
                pos[mask], valid[mask], head[mask], key, matching_extra[mask]
            )
            token_index[mask] = matched_index
            token_contour[mask] = matched_contour

        # Inference-only: the tail branch above filled their second slot.
        if matching_extra.any():
            token_valid[matching_extra, 1] = True

        token_pos = token_contour.mean(dim=2)
        axis = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]
        token_heading = torch.arctan2(axis[:, :, 1], axis[:, :, 0])

        token_velocity = torch.cat(
            [
                token_pos.new_zeros((num_agent, 1, 2)),
                (token_pos[:, 1:] - token_pos[:, :-1]) / (self.dt * self.shift),
            ],
            dim=1,
        )
        velocity_valid = torch.cat(
            [
                torch.zeros((num_agent, 1), dtype=torch.bool),
                (token_valid & token_valid.roll(1, dims=1))[:, 1:],
            ],
            dim=1,
        )
        token_velocity[~velocity_valid] = 0
        observed = valid[:, step]
        token_velocity[observed, 1] = vel[observed, step, :]

        if shape is None:
            agent_shape = torch.zeros((num_agent, 3), dtype=torch.float32)
        else:
            agent_shape = torch.as_tensor(np.asarray(shape), dtype=torch.float32)
            if agent_shape.shape != (num_agent, 3):
                raise ValueError(
                    "shape must have shape (A, 3), got {}".format(tuple(agent_shape.shape))
                )

        return SmartAgentTokens(
            agent_ids=list(agent_ids) if agent_ids is not None else list(range(num_agent)),
            token_idx=token_index,
            token_pos=token_pos,
            token_heading=token_heading,
            token_velocity=token_velocity,
            agent_valid_mask=token_valid,
            agent_type=np.asarray(agent_type),
            agent_shape=agent_shape,
            # Read from the patched local ``valid``, not the raw argument.
            eval_mask=observed.clone(),
        )
