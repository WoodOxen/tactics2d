# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch-to-tensor conversion and tensor-to-prediction adapters for BITS."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from .predictor import BitsAgentPrediction, BitsPlan, BitsPrediction
from .schema import BitsBatch


@dataclass(frozen=True)
class TensorBatch:
    """A torch-backed view of one or more BITS samples."""

    tensors: Dict[str, object]
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"tensors": self.tensors, "metadata": self.metadata}

    def to(self, device) -> "TensorBatch":
        """Move all tensor values to another torch device."""

        return TensorBatch(
            tensors={
                name: value.to(device) if hasattr(value, "to") else value
                for name, value in self.tensors.items()
            },
            metadata=self.metadata,
        )


_FLOAT_BATCH_KEYS = (
    "history_positions",
    "history_yaws",
    "target_positions",
    "target_yaws",
    "curr_speed",
    "centroid",
    "yaw",
    "extent",
    "agent_from_world",
    "world_from_agent",
    "all_other_agents_history_positions",
    "all_other_agents_history_yaws",
    "all_other_agents_future_positions",
    "all_other_agents_future_yaws",
    "all_other_agents_curr_speed",
    "all_other_agents_extents",
    "all_other_agents_history_extents",
)

_BOOL_BATCH_KEYS = (
    "history_availabilities",
    "target_availabilities",
    "all_other_agents_history_availability",
    "all_other_agents_future_availability",
)

_INT_BATCH_KEYS = ("type", "all_other_agents_types")

_OPTIONAL_FLOAT_BATCH_KEYS = (
    "image",
    "raster_from_agent",
    "agent_from_raster",
    "static_image",
    "dynamic_image",
)

_OPTIONAL_BOOL_BATCH_KEYS = ("drivable_map",)


def batch_to_tensor(
    batch: BitsBatch, device=None, dtype=None, include_optional: bool = True
) -> TensorBatch:
    """Convert one BITS batch to tensors for BITS PyTorch models.

    This marks the boundary between Tactics2D scene data (``BitsBatch``) and
    BITS torch modules: batches describe scenarios, while modules operate on
    tensors.

    Args:
        batch: A single BITS sample.
        device: Target torch device.
        dtype: Target torch dtype. Defaults to ``torch.float32``.
        include_optional: Whether to include optional raster fields.

    Returns:
        A ``TensorBatch`` where all fields are ``torch.Tensor``.
    """

    resolved_dtype = dtype or torch.float32
    tensors = {}

    for key in _FLOAT_BATCH_KEYS:
        tensors[key] = torch.as_tensor(getattr(batch, key), dtype=resolved_dtype, device=device)
    for key in _BOOL_BATCH_KEYS:
        tensors[key] = torch.as_tensor(getattr(batch, key), dtype=torch.bool, device=device)
    for key in _INT_BATCH_KEYS:
        tensors[key] = torch.as_tensor(getattr(batch, key), dtype=torch.long, device=device)

    if include_optional:
        for key in _OPTIONAL_FLOAT_BATCH_KEYS:
            value = getattr(batch, key)
            if value is not None:
                tensors[key] = torch.as_tensor(value, dtype=resolved_dtype, device=device)
        for key in _OPTIONAL_BOOL_BATCH_KEYS:
            value = getattr(batch, key)
            if value is not None:
                tensors[key] = torch.as_tensor(value, dtype=torch.bool, device=device)

    return TensorBatch(
        tensors=tensors,
        metadata={
            "ego_id": batch.ego_id,
            "frame": batch.frame,
            "agent_ids": list(batch.agent_ids),
            "lane_id": batch.lane_id,
        },
    )


def collate_batches(
    batches: Iterable[BitsBatch], device=None, dtype=None, include_optional: bool = True
) -> TensorBatch:
    """Stack same-shaped BITS batches into a torch mini-batch.

    Args:
        batches: An iterable of BITS samples.
        device, dtype, include_optional: Forwarded to :func:`batch_to_tensor`.

    Returns:
        A ``TensorBatch`` with an added leading batch dimension.

    Raises:
        ValueError: If samples have different tensor field keys.
    """

    samples = [
        batch_to_tensor(batch, device=device, dtype=dtype, include_optional=include_optional)
        for batch in batches
    ]
    if not samples:
        return TensorBatch(
            tensors={}, metadata={"ego_id": [], "frame": [], "agent_ids": [], "lane_id": []}
        )

    keys = set(samples[0].tensors.keys())
    for sample in samples[1:]:
        if set(sample.tensors.keys()) != keys:
            raise ValueError("Cannot collate BITS samples with different tensor fields.")

    return TensorBatch(
        tensors={
            key: torch.stack([sample.tensors[key] for sample in samples], dim=0)
            for key in sorted(keys)
        },
        metadata={
            "ego_id": [sample.metadata["ego_id"] for sample in samples],
            "frame": [sample.metadata["frame"] for sample in samples],
            "agent_ids": [sample.metadata["agent_ids"] for sample in samples],
            "lane_id": [sample.metadata["lane_id"] for sample in samples],
        },
    )


# ---------------------------------------------------------------------------
# Output adapters: torch predictions → numpy data classes
# ---------------------------------------------------------------------------


def prediction_from_tensor(positions, yaws, availabilities=None, scores=None) -> BitsPrediction:
    """Convert torch model outputs back to the numpy prediction schema.

    Args:
        positions: Tensor or array with shape ``(*, T, 2)``.
        yaws: Tensor or array with shape ``(*, T, 1)``.
        availabilities: Optional bool tensor/array. Defaults to all-True.
        scores: Optional float tensor/array. Defaults to all-1.0.

    Returns:
        A ``BitsPrediction`` with detached numpy arrays.
    """

    positions_np = (
        positions.detach().cpu().numpy() if hasattr(positions, "detach") else np.asarray(positions)
    ).astype(float, copy=False)
    yaws_np = (yaws.detach().cpu().numpy() if hasattr(yaws, "detach") else np.asarray(yaws)).astype(
        float, copy=False
    )
    if availabilities is None:
        availabilities_np = np.ones(positions_np.shape[:2], dtype=bool)
    else:
        availabilities_np = (
            availabilities.detach().cpu().numpy()
            if hasattr(availabilities, "detach")
            else np.asarray(availabilities)
        ).astype(bool, copy=False)
    if scores is None:
        scores_np = np.ones(positions_np.shape[0], dtype=float)
    else:
        scores_np = (
            scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        ).astype(float, copy=False)

    return BitsPrediction(
        positions=positions_np, yaws=yaws_np, availabilities=availabilities_np, scores=scores_np
    )


def prediction_from_module_output(output) -> BitsPrediction:
    """Normalise a BITS torch module output to a ``BitsPrediction``.

    Accepts a ``BitsPrediction`` directly, a dict with keys ``positions``,
    ``yaws``, (optional) ``availabilities``, (optional) ``scores``, or a
    tuple/list of 2--4 elements ``(positions, yaws, *availabilities, *scores)``.

    Returns:
        A ``BitsPrediction`` in the numpy schema.
    """

    if isinstance(output, BitsPrediction):
        return output

    if isinstance(output, dict):
        positions = output["positions"]
        yaws = output["yaws"]
        availabilities = output.get("availabilities")
        scores = output.get("scores")
    elif isinstance(output, (tuple, list)):
        if len(output) < 2 or len(output) > 4:
            raise ValueError("Torch BITS module output must contain 2 to 4 values.")
        positions = output[0]
        yaws = output[1]
        availabilities = output[2] if len(output) >= 3 else None
        scores = output[3] if len(output) >= 4 else None
    else:
        raise TypeError("Torch BITS module output must be a dict, tuple, or BitsPrediction.")

    return prediction_from_tensor(positions, yaws, availabilities, scores)


def squeeze_batch_prediction(prediction: BitsPrediction) -> BitsPrediction:
    """Remove the leading batch dimension when batch size is 1.

    Args:
        prediction: A ``BitsPrediction`` with optional batch dim.

    Returns:
        Squeezed prediction.

    Raises:
        ValueError: If batch size > 1.
    """

    if prediction.positions.ndim != 4:
        return prediction
    if prediction.positions.shape[0] != 1:
        raise ValueError("TorchBitsPolicy only supports one BITS batch at a time.")
    return BitsPrediction(
        positions=prediction.positions[0],
        yaws=prediction.yaws[0],
        availabilities=prediction.availabilities[0],
        scores=prediction.scores.reshape(-1),
    )


def plan_from_tensor(plan_output, fallback_prediction: BitsPrediction) -> BitsPlan:
    """Convert a torch plan output to a ``BitsPlan``.

    Args:
        plan_output: A ``BitsPlan`` or a dict with ``log_likelihood`` or
            ``scores`` key.
        fallback_prediction: Fallback when ``plan_output`` has no score field.

    Returns:
        A ``BitsPlan`` with numpy arrays.
    """

    if isinstance(plan_output, BitsPlan):
        return plan_output
    if not isinstance(plan_output, dict):
        raise TypeError("Torch BITS plan output must be a dict or BitsPlan.")

    scores = plan_output.get("log_likelihood", plan_output.get("scores"))

    if scores is None:
        plan_scores = np.asarray(fallback_prediction.scores, dtype=float).reshape(-1)
    else:
        scores_np = (
            scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        ).astype(float, copy=False)
        plan_scores = scores_np.reshape(-1)

    return BitsPlan(
        positions=fallback_prediction.positions,
        yaws=fallback_prediction.yaws,
        availabilities=fallback_prediction.availabilities,
        scores=plan_scores,
    )


def agent_prediction_from_tensor(prediction_output) -> Optional[BitsAgentPrediction]:
    """Extract neighbour-agent predictions from a prediction dict.

    Args:
        prediction_output: A dict optionally containing ``agent_positions``,
            ``agent_yaws``, ``agent_availabilities``.

    Returns:
        A ``BitsAgentPrediction`` or ``None`` if the dict has no agent fields.
    """

    if not isinstance(prediction_output, dict):
        return None
    if "agent_positions" in prediction_output and "agent_yaws" in prediction_output:
        positions = (
            prediction_output["agent_positions"].detach().cpu().numpy()
            if hasattr(prediction_output["agent_positions"], "detach")
            else np.asarray(prediction_output["agent_positions"])
        )
        yaws = (
            prediction_output["agent_yaws"].detach().cpu().numpy()
            if hasattr(prediction_output["agent_yaws"], "detach")
            else np.asarray(prediction_output["agent_yaws"])
        )
        availabilities = prediction_output.get("agent_availabilities")
        if availabilities is None:
            availabilities = prediction_output.get("scene_availabilities")
            if availabilities is not None:
                availabilities = (
                    availabilities.detach().cpu().numpy()
                    if hasattr(availabilities, "detach")
                    else np.asarray(availabilities)
                )[..., 1:, :]
        if availabilities is None:
            availabilities = np.ones(positions.shape[:-1], dtype=bool)
        else:
            availabilities = (
                availabilities.detach().cpu().numpy()
                if hasattr(availabilities, "detach")
                else np.asarray(availabilities)
            ).astype(bool, copy=False)
        if positions.ndim == 5 and positions.shape[0] == 1:
            positions = positions[0]
        if yaws.ndim == 5 and yaws.shape[0] == 1:
            yaws = yaws[0]
        if availabilities.ndim == 4 and availabilities.shape[0] == 1:
            availabilities = availabilities[0]
        return BitsAgentPrediction(
            positions=np.asarray(positions, dtype=float),
            yaws=np.asarray(yaws, dtype=float),
            availabilities=availabilities,
        )
    return None


# ---------------------------------------------------------------------------
# Goal tensor helpers
# ---------------------------------------------------------------------------


def collate_goals(goals, device=None, dtype=None):
    """Stack spatial planner supervision for a torch mini-batch."""

    goal_tensors = [goal_to_tensor(g, device, dtype) for g in goals]
    if not goal_tensors:
        return {}
    keys = set(goal_tensors[0].keys())
    return {k: torch.stack([g[k] for g in goal_tensors], dim=0) for k in keys}


def goal_to_tensor(goal, device=None, dtype=None):
    """Convert BITS spatial planner supervision to torch tensors."""

    resolved = dtype or torch.float32
    tensors = {
        "goal_position": torch.as_tensor(goal.goal_position, dtype=resolved, device=device),
        "goal_yaw": torch.as_tensor(goal.goal_yaw, dtype=resolved, device=device),
        "goal_index": torch.as_tensor(goal.goal_index, dtype=torch.long, device=device),
    }
    if goal.goal_position_pixel is not None:
        tensors["goal_position_pixel"] = torch.as_tensor(
            goal.goal_position_pixel, dtype=torch.long, device=device
        )
    if goal.goal_position_pixel_flat is not None:
        tensors["goal_position_pixel_flat"] = torch.as_tensor(
            goal.goal_position_pixel_flat, dtype=torch.long, device=device
        )
    if goal.goal_position_residual is not None:
        tensors["goal_position_residual"] = torch.as_tensor(
            goal.goal_position_residual, dtype=resolved, device=device
        )
    if goal.goal_spatial_map is not None:
        tensors["goal_spatial_map"] = torch.as_tensor(
            goal.goal_spatial_map, dtype=resolved, device=device
        )
    return tensors
