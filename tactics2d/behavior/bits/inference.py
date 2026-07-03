# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Inference model loading and checkpoint compatibility for BITS."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch

from .config import BitsConfig
from .torch_model import BitsBiLevelTorchModel


@dataclass(frozen=True)
class BitsCheckpointMetadata:
    """Metadata saved next to BITS model weights."""

    epoch: int
    image_channels: int
    future_steps: int
    hidden_dim: int
    model_arch: str
    context_size: int
    roi_feature_size: int
    roi_layer_key: str
    config: Dict[str, object]
    history_conditioning: bool = False
    use_transformer: bool = False
    schedule: Dict[str, object] = field(default_factory=dict)
    split: Dict[str, object] = field(default_factory=dict)
    official_checkpoint_note: Optional[str] = None


@dataclass(frozen=True)
class BitsCheckpointShapeMismatch:
    """One tensor whose mapped checkpoint shape does not match the target model."""

    key: str
    expected_shape: Tuple[int, ...]
    found_shape: Tuple[int, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "expected_shape": self.expected_shape,
            "found_shape": self.found_shape,
        }


@dataclass(frozen=True)
class BitsCheckpointCompatibilityReport:
    """Key and shape compatibility report for a mapped BITS checkpoint."""

    matched_keys: Tuple[str, ...]
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    shape_mismatches: Tuple[BitsCheckpointShapeMismatch, ...] = ()

    @property
    def is_compatible(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys and not self.shape_mismatches

    def as_dict(self) -> Dict[str, object]:
        return {
            "is_compatible": self.is_compatible,
            "matched_keys": self.matched_keys,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "shape_mismatches": [mismatch.as_dict() for mismatch in self.shape_mismatches],
        }


@dataclass(frozen=True)
class BitsInferenceLoadResult:
    """Loaded BITS inference model and optional checkpoint compatibility report."""

    model: BitsBiLevelTorchModel
    source: str
    metadata: Dict[str, object] = field(default_factory=dict)
    compatibility: Optional[BitsCheckpointCompatibilityReport] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "metadata": self.metadata,
            "compatibility": None if self.compatibility is None else self.compatibility.as_dict(),
        }


def load_bits_inference_model(
    checkpoint_path=None,
    tactics2d_planner_checkpoint=None,
    planner_checkpoint=None,
    predictor_checkpoint=None,
    image_channels: Optional[int] = None,
    future_steps: Optional[int] = None,
    hidden_dim: int = 128,
    model_arch: str = "resnet18",
    context_size: int = 30,
    roi_feature_size: int = 7,
    roi_layer_key: str = "layer2",
    history_conditioning: bool = False,
    use_transformer: bool = False,
    config: Optional[BitsConfig] = None,
    map_location=None,
    strict: bool = True,
) -> BitsInferenceLoadResult:
    """Load a BITS inference model from either local or official checkpoints."""

    has_local_checkpoint = checkpoint_path is not None
    has_weight_parts = (
        tactics2d_planner_checkpoint is not None
        or planner_checkpoint is not None
        or predictor_checkpoint is not None
    )
    if has_local_checkpoint == has_weight_parts:
        raise ValueError(
            "Provide either checkpoint_path or planner/predictor checkpoint parts, not both."
        )
    if tactics2d_planner_checkpoint is not None and planner_checkpoint is not None:
        raise ValueError(
            "Provide only one planner source: tactics2d_planner_checkpoint or planner_checkpoint."
        )

    if has_local_checkpoint:
        model, metadata, _payload = load_bits_checkpoint(checkpoint_path, map_location=map_location)
        return BitsInferenceLoadResult(
            model=model, source="tactics2d", metadata=dict(metadata), compatibility=None
        )

    if image_channels is None or future_steps is None:
        raise ValueError(
            "image_channels and future_steps are required when loading checkpoint parts."
        )
    resolved_config = config or BitsConfig(future_steps=int(future_steps))
    model = BitsBiLevelTorchModel(
        image_channels=int(image_channels),
        future_steps=int(future_steps),
        hidden_dim=hidden_dim,
        model_arch=model_arch,
        context_size=context_size,
        roi_feature_size=roi_feature_size,
        roi_layer_key=roi_layer_key,
        history_conditioning=history_conditioning,
        use_transformer=use_transformer,
        config=resolved_config,
    )
    mapped_state_dict = {}
    if tactics2d_planner_checkpoint is not None:
        mapped_state_dict.update(
            _load_tactics2d_planner_state_dict(
                tactics2d_planner_checkpoint, map_location=map_location
            )
        )
    if planner_checkpoint is not None or predictor_checkpoint is not None:
        planner_state_dict = (
            None
            if planner_checkpoint is None
            else _load_checkpoint_state_dict(planner_checkpoint, map_location=map_location)
        )
        predictor_state_dict = (
            None
            if predictor_checkpoint is None
            else _load_checkpoint_state_dict(predictor_checkpoint, map_location=map_location)
        )
        official_state_dict = merge_tbsim_bits_state_dicts(
            planner_state_dict=planner_state_dict, predictor_state_dict=predictor_state_dict
        )
        overlap = set(mapped_state_dict).intersection(official_state_dict)
        non_encoder_overlap = [
            key for key in sorted(overlap) if not key.startswith("shared_encoder.encoder_heads.")
        ]
        if non_encoder_overlap:
            raise ValueError(f"Overlapping BITS checkpoint keys: {non_encoder_overlap[:3]}")
        mapped_state_dict.update(official_state_dict)
    compatibility = _load_mapped_bits_inference_weights(model, mapped_state_dict, strict=strict)
    source = "mixed" if tactics2d_planner_checkpoint is not None else "tbsim"
    return BitsInferenceLoadResult(
        model=model,
        source=source,
        metadata={
            "image_channels": int(image_channels),
            "future_steps": int(future_steps),
            "hidden_dim": int(hidden_dim),
            "model_arch": model_arch,
            "context_size": int(context_size),
            "roi_feature_size": int(roi_feature_size),
            "roi_layer_key": roi_layer_key,
            "history_conditioning": bool(history_conditioning),
            "use_transformer": bool(use_transformer),
            "uses_tactics2d_planner_checkpoint": tactics2d_planner_checkpoint is not None,
            "uses_tbsim_planner_checkpoint": planner_checkpoint is not None,
            "uses_tbsim_predictor_checkpoint": predictor_checkpoint is not None,
        },
        compatibility=compatibility,
    )


def load_bits_checkpoint(path, map_location=None):
    """Load a BITS checkpoint and reconstruct the compact torch model."""

    payload = torch.load(path, map_location=map_location)
    metadata = payload["metadata"]
    config = BitsConfig(**metadata["config"])
    model = BitsBiLevelTorchModel(
        image_channels=int(metadata["image_channels"]),
        future_steps=int(metadata["future_steps"]),
        hidden_dim=int(
            metadata.get("hidden_dim", metadata.get("schedule", {}).get("hidden_dim", 128))
        ),
        model_arch=str(
            metadata.get("model_arch", metadata.get("schedule", {}).get("model_arch", "resnet18"))
        ),
        context_size=int(
            metadata.get("context_size", metadata.get("schedule", {}).get("context_size", 30))
        ),
        roi_feature_size=int(
            metadata.get(
                "roi_feature_size", metadata.get("schedule", {}).get("roi_feature_size", 7)
            )
        ),
        roi_layer_key=str(
            metadata.get(
                "roi_layer_key", metadata.get("schedule", {}).get("roi_layer_key", "layer2")
            )
        ),
        history_conditioning=bool(
            metadata.get(
                "history_conditioning",
                metadata.get("schedule", {}).get("history_conditioning", False),
            )
        ),
        use_transformer=bool(
            metadata.get(
                "use_transformer", metadata.get("schedule", {}).get("use_transformer", False)
            )
        ),
        config=config,
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, metadata, payload


def load_tbsim_bits_inference_weights(
    model: BitsBiLevelTorchModel,
    planner_checkpoint=None,
    predictor_checkpoint=None,
    map_location=None,
    strict: bool = True,
) -> BitsCheckpointCompatibilityReport:
    """Load official planner/predictor checkpoints into an inference BITS model."""

    planner_state_dict = (
        None
        if planner_checkpoint is None
        else _load_checkpoint_state_dict(planner_checkpoint, map_location=map_location)
    )
    predictor_state_dict = (
        None
        if predictor_checkpoint is None
        else _load_checkpoint_state_dict(predictor_checkpoint, map_location=map_location)
    )
    mapped_state_dict = merge_tbsim_bits_state_dicts(
        planner_state_dict=planner_state_dict, predictor_state_dict=predictor_state_dict
    )
    return _load_mapped_bits_inference_weights(model, mapped_state_dict, strict=strict)


def map_tbsim_bits_planner_state_dict(state_dict: Mapping[str, object]) -> Dict[str, object]:
    """Map official TBSIM SpatialPlanner keys onto the Tactics2D planner module."""

    mapped = _map_tbsim_state_dict_prefixes(
        state_dict,
        prefix_map={
            "nets.policy.decoder.": "planner.spatial_goal_decoder.decoder.",
            "policy.decoder.": "planner.spatial_goal_decoder.decoder.",
            "nets.policy.encoder_heads.": "shared_encoder.encoder_heads.",
            "policy.encoder_heads.": "shared_encoder.encoder_heads.",
            "nets.policy.": "planner.spatial_goal_decoder.",
            "policy.": "planner.spatial_goal_decoder.",
        },
    )
    return _normalize_bits_state_dict_keys(mapped)


def map_tbsim_bits_predictor_state_dict(state_dict: Mapping[str, object]) -> Dict[str, object]:
    """Map official TBSIM MATrafficModel keys onto the Tactics2D predictor module."""

    mapped = _map_tbsim_state_dict_prefixes(state_dict, prefix_map={"model.": "predictor."})
    return _normalize_bits_state_dict_keys(mapped)


def merge_tbsim_bits_state_dicts(
    planner_state_dict: Optional[Mapping[str, object]] = None,
    predictor_state_dict: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Merge official planner/predictor state dicts into one BITS model state dict."""

    merged: Dict[str, object] = {}
    if planner_state_dict is not None:
        merged.update(map_tbsim_bits_planner_state_dict(planner_state_dict))
    if predictor_state_dict is not None:
        mapped_predictor = map_tbsim_bits_predictor_state_dict(predictor_state_dict)
        overlap = set(merged).intersection(mapped_predictor)
        non_encoder_overlap = [
            key for key in sorted(overlap) if not key.startswith("shared_encoder.encoder_heads.")
        ]
        if non_encoder_overlap:
            raise ValueError(f"Overlapping BITS checkpoint keys: {non_encoder_overlap[:3]}")
        merged.update(mapped_predictor)
    return merged


def _load_mapped_bits_inference_weights(
    model: BitsBiLevelTorchModel, mapped_state_dict: Mapping[str, object], strict: bool = True
) -> BitsCheckpointCompatibilityReport:
    """Load already-mapped BITS weights into ``model`` and report compatibility."""

    normalized_state_dict = _normalize_shared_encoder_state_dict(mapped_state_dict)
    report = _build_bits_checkpoint_compatibility_report(model, normalized_state_dict)
    if strict and not report.is_compatible:
        raise ValueError(_format_checkpoint_compatibility_error(report))

    model_state = model.state_dict()
    loadable_state = {
        key: normalized_state_dict[key] for key in report.matched_keys if key in model_state
    }
    model_state.update(loadable_state)
    model.load_state_dict(model_state)
    return report


def _load_tactics2d_planner_state_dict(checkpoint, map_location=None) -> Dict[str, object]:
    """Extract only ``planner.*`` tensors from a Tactics2D BITS checkpoint."""

    if isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        payload = torch.load(checkpoint, map_location=map_location)
    state_dict = payload.get("model_state_dict", payload)
    normalized_state_dict = _normalize_bits_state_dict_keys(state_dict)
    planner_state = {
        key: value
        for key, value in normalized_state_dict.items()
        if str(key).startswith("planner.") or str(key).startswith("shared_encoder.")
    }
    if not planner_state:
        raise ValueError(
            "Tactics2D planner checkpoint does not contain planner.* or shared_encoder.* weights."
        )
    return planner_state


def _normalize_bits_state_dict_keys(state_dict: Mapping[str, object]) -> Dict[str, object]:
    """Translate legacy/official BITS keys onto the shared-encoder module tree."""

    normalized: Dict[str, object] = {}
    replacements = (
        ("predictor.map_encoder.encoder_heads.", "shared_encoder.encoder_heads."),
        ("predictor.map_encoder.roi_head.", "predictor.roi_head."),
        ("predictor.map_encoder.agent_net.", "predictor.roi_head.agent_net."),
        ("predictor.goal_encoder.", "predictor.policy_head.goal_encoder."),
        ("predictor.ego_decoder.", "predictor.policy_head.ego_decoder."),
        ("predictor.agents_decoder.", "predictor.future_state_head.agents_decoder."),
        ("planner.raster_unet.decoder.", "planner.spatial_goal_decoder.decoder."),
        ("planner.raster_unet.encoder_heads.", "shared_encoder.encoder_heads."),
        ("planner.raster_unet.", "planner.spatial_goal_decoder."),
    )
    for key, value in state_dict.items():
        target_key = str(key)
        for old_prefix, new_prefix in replacements:
            if target_key.startswith(old_prefix):
                target_key = new_prefix + target_key[len(old_prefix) :]
                break
        normalized[target_key] = value
    return normalized


def _normalize_shared_encoder_state_dict(state_dict: Mapping[str, object]) -> Dict[str, object]:
    """Drop duplicate planner encoder tensors when predictor encoder is the shared source."""

    normalized = _normalize_bits_state_dict_keys(state_dict)
    return normalized


def _build_bits_checkpoint_compatibility_report(
    model: BitsBiLevelTorchModel, state_dict: Mapping[str, object]
) -> BitsCheckpointCompatibilityReport:
    """Compare mapped checkpoint weights against a BITS model without mutating weights."""

    model_state = model.state_dict()
    state_dict = _normalize_bits_state_dict_keys(state_dict)
    model_keys = set(model_state)
    checkpoint_keys = set(state_dict)
    common_keys = sorted(model_keys.intersection(checkpoint_keys))
    missing_keys = tuple(sorted(model_keys - checkpoint_keys))
    unexpected_keys = tuple(sorted(checkpoint_keys - model_keys))
    matched_keys = []
    shape_mismatches = []
    for key in common_keys:
        expected_shape = _tensor_shape_tuple(model_state[key])
        found_shape = _tensor_shape_tuple(state_dict[key])
        if expected_shape == found_shape:
            matched_keys.append(key)
        else:
            shape_mismatches.append(
                BitsCheckpointShapeMismatch(
                    key=key, expected_shape=expected_shape, found_shape=found_shape
                )
            )
    return BitsCheckpointCompatibilityReport(
        matched_keys=tuple(matched_keys),
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=tuple(shape_mismatches),
    )


def _format_checkpoint_compatibility_error(report: BitsCheckpointCompatibilityReport) -> str:
    return (
        "Official BITS checkpoint is not compatible with the current model: "
        f"{len(report.missing_keys)} missing, "
        f"{len(report.unexpected_keys)} unexpected, "
        f"{len(report.shape_mismatches)} shape mismatches."
    )


def _map_tbsim_state_dict_prefixes(
    state_dict: Mapping[str, object], prefix_map: Mapping[str, str]
) -> Dict[str, object]:
    mapped: Dict[str, object] = {}
    for key, value in state_dict.items():
        normalized_key = _strip_lightning_state_prefix(str(key))
        target_key = normalized_key
        for source_prefix, target_prefix in prefix_map.items():
            if normalized_key.startswith(source_prefix):
                target_key = target_prefix + normalized_key[len(source_prefix) :]
                break
        mapped[target_key] = value
    return mapped


def _strip_lightning_state_prefix(key: str) -> str:
    return key[len("state_dict.") :] if key.startswith("state_dict.") else key


def _load_checkpoint_state_dict(checkpoint, map_location=None) -> Mapping[str, object]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location=map_location)
    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], Mapping):
            return checkpoint["model_state_dict"]
        return checkpoint
    raise TypeError("checkpoint must be a path or a mapping containing tensor weights.")


def _tensor_shape_tuple(value) -> Tuple[int, ...]:
    return tuple(value.shape) if hasattr(value, "shape") else ()


__all__ = [
    "BitsCheckpointMetadata",
    "BitsCheckpointShapeMismatch",
    "BitsCheckpointCompatibilityReport",
    "BitsInferenceLoadResult",
    "load_bits_inference_model",
    "load_bits_checkpoint",
    "load_tbsim_bits_inference_weights",
    "map_tbsim_bits_planner_state_dict",
    "map_tbsim_bits_predictor_state_dict",
    "merge_tbsim_bits_state_dicts",
]
