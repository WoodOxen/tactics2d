# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Configuration for SMART-style joint behavior generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_BUNDLED_TOKENS = Path(__file__).resolve().parent / "tokens"


@dataclass(frozen=True)
class SmartConfig:
    """Parameters for the SMART joint motion generation port.

    Decoder dimensions must not be changed when loading a released checkpoint,
    because each one determines a parameter shape.

    Attributes:
        history_steps (int): Number of observed frames. Defaults to 11.
        future_steps (int): Number of predicted frames. Defaults to 80.
        dt (float): Sampling interval in seconds. Defaults to 0.1.
        shift (int): Frames covered by one motion token. Defaults to 5.
        motion_codebook (Optional[str]): Motion codebook path. Defaults to None (bundled, by
            ``token_size``).
        map_codebook (Optional[str]): Map codebook path. Defaults to None (the bundled copy).
        token_size (int): Number of motion tokens in the codebook.
        map_token_size (int): Number of map tokens in the codebook.
        hidden_dim (int): Transformer hidden width.
        num_heads (int): Number of attention heads.
        head_dim (int): Per-head attention width.
        dropout (float): Dropout probability.
        num_freq_bands (int): Fourier feature bands.
        num_map_layers (int): Map decoder layers.
        num_agent_layers (int): Agent decoder layers.
        pl2pl_radius (float): Point-to-point map attention radius in meters.
        pl2a_radius (float): Map-to-agent cross attention radius in meters.
        a2a_radius (float): Agent-to-agent attention radius in meters.
        time_span (int): Temporal attention window in frames.
        max_pt2pt_neighbors (int): Neighbour cap for map map attention.
        max_pl2a_neighbors (int): Neighbour cap for map agent attention.
        max_a2a_neighbors (int): Neighbour cap for agent agent attention.
        beam_size (int): Number of sampled token candidates.
        seed (Optional[int]): Sampling seed. Defaults to None (unseeded).
        device (Optional[str]): Torch device string. None selects automatically.
        predicted_radius (float): Agent-selection radius around the centre agent, in meters.
        max_predicted_agents (int): Upper bound on modelled agents.
        valid_radius (float): Distance beyond which agents are masked out.
        vehicle_only (bool): Whether to model vehicles only.
    """

    # token time base
    history_steps: int = 11
    future_steps: int = 80
    dt: float = 0.1
    shift: int = 5
    # codebooks
    motion_codebook: Optional[str] = None
    map_codebook: Optional[str] = None
    token_size: int = 2048
    map_token_size: int = 1024
    # decoder
    hidden_dim: int = 128
    num_heads: int = 8
    head_dim: int = 16
    dropout: float = 0.1
    num_freq_bands: int = 64
    num_map_layers: int = 3
    num_agent_layers: int = 6
    pl2pl_radius: float = 10.0
    pl2a_radius: float = 30.0
    a2a_radius: float = 60.0
    time_span: int = 30
    max_pt2pt_neighbors: int = 100
    max_pl2a_neighbors: int = 300
    max_a2a_neighbors: int = 300
    # sampling and device
    beam_size: int = 5
    seed: Optional[int] = None
    device: Optional[str] = None
    # agent selection
    predicted_radius: float = 100.0
    max_predicted_agents: int = 32
    valid_radius: float = 150.0
    vehicle_only: bool = True

    def __post_init__(self) -> None:
        if self.shift <= 0:
            raise ValueError("shift must be positive.")
        if self.future_steps % self.shift != 0:
            raise ValueError(
                "future_steps must be divisible by shift so that every "
                "autoregressive step covers a whole number of tokens."
            )
        if self.history_token_slots < 2:
            raise ValueError(
                "history_steps must cover at least two motion tokens, because "
                "the autoregressive loop reads the last two history slots."
            )
        if self.motion_codebook is None:
            object.__setattr__(
                self,
                "motion_codebook",
                str(_BUNDLED_TOKENS / f"cluster_frame_5_{self.token_size}.pkl"),
            )
        if self.map_codebook is None:
            object.__setattr__(self, "map_codebook", str(_BUNDLED_TOKENS / "map_traj_token5.pkl"))

    @property
    def step_ms(self) -> int:
        """Return the configured sampling interval in milliseconds."""

        return int(round(self.dt * 1000))

    @property
    def planning_steps(self) -> int:
        """Return the number of future planning states."""

        return self.future_steps

    @property
    def history_token_slots(self) -> int:
        """Return how many motion tokens the observed history covers."""

        return (self.history_steps - 1) // self.shift

    @property
    def future_token_slots(self) -> int:
        """Return how many motion tokens the autoregressive loop predicts."""

        return self.future_steps // self.shift
