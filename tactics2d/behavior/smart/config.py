# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Configuration for SMART-style joint behavior generation."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# SMART's codebooks are not redistributed with the package: download them
# alongside the checkpoint and point ``asset_root`` (or the two explicit paths)
# at that directory.
MOTION_CODEBOOK_NAME = "motion_codebook.pkl"
MAP_CODEBOOK_NAME = "map_codebook.pkl"


def load_codebook(path: Optional[str], name: str) -> Dict[str, object]:
    """Load one of SMART's downloadable codebooks.

    Args:
        path (Optional[str]): Configured codebook path.
        name (str): Human-readable codebook name used in the message.

    Returns:
        The unpickled codebook.

    Raises:
        ValueError: If no path was configured.
    """

    if path is None:
        raise ValueError(
            f"The SMART {name} is not redistributed with the package. Download it "
            f"together with the checkpoint and set SmartConfig(asset_root=...) or pass "
            f"the path directly."
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


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
        asset_root (Optional[str]): Directory holding the downloaded SMART codebooks.
            Defaults to None, which requires the two paths below to be set explicitly.
        motion_codebook (Optional[str]): Motion codebook path. Defaults to None
            (``asset_root``/``motion_codebook.pkl``).
        map_codebook (Optional[str]): Map codebook path. Defaults to None
            (``asset_root``/``map_codebook.pkl``).
        token_size (int): Number of motion tokens in the codebook. Validated against the
            loaded codebook.
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
    asset_root: Optional[str] = None
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
        if self.asset_root is not None:
            root = Path(self.asset_root)
            if self.motion_codebook is None:
                object.__setattr__(self, "motion_codebook", str(root / MOTION_CODEBOOK_NAME))
            if self.map_codebook is None:
                object.__setattr__(self, "map_codebook", str(root / MAP_CODEBOOK_NAME))

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
