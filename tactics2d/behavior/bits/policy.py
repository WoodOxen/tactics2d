# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""BITS policy interface and PyTorch policy wrapper."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .adapters import (
    agent_prediction_from_tensor,
    batch_to_tensor,
    collate_batches,
    plan_from_tensor,
    prediction_from_module_output,
    squeeze_batch_prediction,
)
from .batch import BitsBatch
from .prediction import BitsPlan, BitsPrediction
from .scorer import BitsPlanScorer


class BitsPolicy(ABC):
    """Base interface for BITS-compatible policies."""

    @abstractmethod
    def predict_batch(self, batch: BitsBatch) -> BitsPrediction:
        """Predict ego-frame future trajectories for one BITS batch."""

    def predict_batches(self, batches: Sequence[BitsBatch]) -> List[BitsPrediction]:
        """Predict several prepared batches, falling back to serial inference."""

        return [self.predict_batch(batch) for batch in batches]


class TorchBitsPolicy(BitsPolicy):
    """Wrap a PyTorch module behind the standard BITS policy interface."""

    def __init__(
        self,
        module,
        device=None,
        dtype=None,
        include_optional: bool = True,
        module_input: str = "tensors",
        plan_scorer: Optional[BitsPlanScorer] = None,
        select_best_plan: bool = True,
        module_kwargs: Optional[Dict[str, object]] = None,
    ):
        if module_input not in {"tensors", "batch"}:
            raise ValueError("module_input must be either 'tensors' or 'batch'.")
        self.module = module
        self.device = device
        self.dtype = dtype
        self.include_optional = include_optional
        self.module_input = module_input
        self.plan_scorer = plan_scorer
        self.select_best_plan = select_best_plan
        self.module_kwargs = dict(module_kwargs or {})
        self.last_plan: Optional[BitsPlan] = None
        self.last_plan_scores = None
        self.last_selected_plan: Optional[BitsPlan] = None

    def predict_batch(self, batch: BitsBatch) -> BitsPrediction:
        """Run a torch module and return the normal numpy BITS prediction."""

        torch_batch = batch_to_tensor(
            batch, device=self.device, dtype=self.dtype, include_optional=self.include_optional
        )
        module_arg = torch_batch if self.module_input == "batch" else torch_batch.tensors

        with torch.inference_mode():
            output = self.module(module_arg, **self.module_kwargs)

        if isinstance(output, dict) and "plan" in output and "predictions" in output:
            return self._prediction_from_bilevel_output(batch, output)
        return prediction_from_module_output(output)

    def predict_batches(self, batches: Sequence[BitsBatch]) -> List[BitsPrediction]:
        """Run same-shaped agent-centric samples in one model forward pass."""

        batches = list(batches)
        if not batches:
            return []
        torch_batch = collate_batches(
            batches, device=self.device, dtype=self.dtype, include_optional=self.include_optional
        )
        module_arg = torch_batch if self.module_input == "batch" else torch_batch.tensors
        with torch.inference_mode():
            output = self.module(module_arg, **self.module_kwargs)

        predictions = []
        for index, batch in enumerate(batches):
            sample_output = _slice_batch_output(output, index, len(batches))
            if (
                isinstance(sample_output, dict)
                and "plan" in sample_output
                and "predictions" in sample_output
            ):
                prediction = self._prediction_from_bilevel_output(batch, sample_output)
            else:
                prediction = squeeze_batch_prediction(prediction_from_module_output(sample_output))
            predictions.append(prediction)
        return predictions

    def _prediction_from_bilevel_output(
        self, batch: BitsBatch, output: Dict[str, object]
    ) -> BitsPrediction:
        prediction = squeeze_batch_prediction(prediction_from_module_output(output["predictions"]))
        plan = plan_from_tensor(output["plan"], prediction)
        agent_prediction = agent_prediction_from_tensor(output["predictions"])
        scorer = self.plan_scorer or BitsPlanScorer()
        self.last_plan = plan
        self.last_plan_scores = scorer.score_batch(batch, plan, agent_prediction=agent_prediction)
        if self.select_best_plan:
            best_index = int(np.argmax(self.last_plan_scores.total))
            self.last_selected_plan = scorer.select_plan(plan, self.last_plan_scores)
            return BitsPrediction(
                positions=prediction.positions[[best_index]].copy(),
                yaws=prediction.yaws[[best_index]].copy(),
                availabilities=prediction.availabilities[[best_index]].copy(),
                scores=self.last_plan_scores.total[[best_index]].copy(),
            )

        self.last_selected_plan = BitsPlan(
            positions=plan.positions,
            yaws=plan.yaws,
            availabilities=plan.availabilities,
            scores=self.last_plan_scores.total,
        )
        return BitsPrediction(
            positions=prediction.positions,
            yaws=prediction.yaws,
            availabilities=prediction.availabilities,
            scores=self.last_plan_scores.total,
        )


def _slice_batch_output(value, index: int, batch_size: int):
    """Keep one leading batch row in a nested model output."""

    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value
    if isinstance(value, dict):
        return {key: _slice_batch_output(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_slice_batch_output(item, index, batch_size) for item in value)
    if isinstance(value, list):
        return [_slice_batch_output(item, index, batch_size) for item in value]
    return value
