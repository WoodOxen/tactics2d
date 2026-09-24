# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""SMART rollout policy implementation."""

from typing import Optional

import numpy as np
import torch

from .prediction import SmartPrediction
from .tokens import SmartTokenBatch


class TorchSmartPolicy:
    """Wrap the ported SMART network behind the standard policy interface.

    The model only needs the duck-typed ``predict_batch`` method below.

    Attributes:
        model (SmartTorchModel): The network being driven.
        device (Optional[torch.device]): Device the batch is moved to. Defaults
            to None, which leaves it where the builder put it.
        seed (Optional[int]): Seed applied before every rollout. Defaults to
            None, which draws from the ambient generator.
    """

    def __init__(self, model, device=None, dtype=None, seed: Optional[int] = None):
        """Initialize the policy.

        Args:
            model (SmartTorchModel): The network to drive.
            device (Optional[str], optional): Device name. Defaults to None.
            dtype (Optional[torch.dtype], optional): Parameter dtype. Defaults to None.
            seed (Optional[int], optional): Rollout seed. Defaults to None.
        """

        self.model = model
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.seed = seed

    def predict_batch(self, batch: SmartTokenBatch) -> SmartPrediction:
        """Run the autoregressive rollout for one batch.

        Args:
            batch (SmartTokenBatch): Tokenized scene.

        Returns:
            The joint rollout, in the world frame.
        """

        if self.device is not None:
            self.model.to(self.device)
        if self.dtype is not None:
            self.model.to(dtype=self.dtype)
        # Re-applied per call, so every step of a closed loop draws identically.
        if self.seed is not None:
            torch.manual_seed(self.seed)
        with torch.no_grad():
            output = self.model.inference(batch)

        agents = batch.agents
        positions = output["pred_traj"].detach().cpu().numpy()
        headings = output["pred_head"].detach().cpu().numpy()
        return SmartPrediction(
            agent_ids=list(agents.agent_ids),
            positions=positions,
            headings=headings,
            availabilities=np.ones(positions.shape[:2], dtype=bool),
            frame_ms0=int(batch.frame_ms),
            step_ms=int(self.model.config.step_ms),
        )
