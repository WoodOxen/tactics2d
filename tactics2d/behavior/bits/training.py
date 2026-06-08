# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Training and checkpoint helpers for the BITS torch reproduction."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .config import BitsConfig
from .dataset import NuPlanBitsDataset
from .evaluation import evaluate_bits_rolling_result
from .model import BitsBehaviorModel, BitsPlanScorer
from .rolling import BitsRollingRunner
from .torch_model import (
    BitsBiLevelTorchModel,
    BitsTorchEpochResult,
    TorchBitsPolicy,
    run_bits_planner_torch_epoch,
    run_bits_torch_epoch,
)


@dataclass(frozen=True)
class NuPlanLogSpec:
    """One NuPlan log/map pair used by the BITS reproduction pipeline."""

    data_file: str
    map_file: str
    data_folder: str = ""
    map_folder: Optional[str] = None
    time_range: Optional[Tuple[int, int]] = None
    frame_range: Optional[Tuple[int, int]] = None
    ego_ids: Optional[Tuple[object, ...]] = None


@dataclass(frozen=True)
class NuPlanBitsSplit:
    """Train/validation/test split for NuPlan-backed BITS runs."""

    train: Tuple[NuPlanLogSpec, ...] = ()
    val: Tuple[NuPlanLogSpec, ...] = ()
    test: Tuple[NuPlanLogSpec, ...] = ()

    def logs(self, split: str) -> Tuple[NuPlanLogSpec, ...]:
        if split == "train":
            return self.train
        if split in {"val", "valid", "validation"}:
            return self.val
        if split == "test":
            return self.test
        raise ValueError("split must be one of 'train', 'val', or 'test'.")


@dataclass(frozen=True)
class BitsTrainingSchedule:
    """Small schedule object for reproducible BITS training runs."""

    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 128
    model_arch: str = "resnet18"
    context_size: int = 30
    roi_feature_size: int = 7
    roi_layer_key: str = "layer2"
    history_conditioning: bool = False
    use_transformer: bool = False
    use_ground_truth_goal: bool = True
    num_samples: Optional[int] = None
    mask_drivable: bool = False
    max_train_samples_per_log: Optional[int] = None
    max_val_samples_per_log: Optional[int] = None
    checkpoint_every_epochs: int = 1
    seed: Optional[int] = None


@dataclass(frozen=True)
class BitsRunConfig:
    """Serializable NuPlan BITS run configuration."""

    config: BitsConfig
    split: NuPlanBitsSplit
    schedule: BitsTrainingSchedule = field(default_factory=BitsTrainingSchedule)


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
            "compatibility": None
            if self.compatibility is None
            else self.compatibility.as_dict(),
        }


@dataclass(frozen=True)
class BitsTrainingHistory:
    """Epoch-by-epoch training/validation losses."""

    train: Tuple[BitsTorchEpochResult, ...]
    val: Tuple[BitsTorchEpochResult, ...] = ()
    checkpoints: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BitsProtocolResult:
    """Serializable summary from a small reproducible BITS experiment."""

    protocol: str
    run_config: BitsRunConfig
    output_dir: str
    config_path: str
    result_path: Optional[str]
    train: Tuple[Dict[str, object], ...]
    val: Tuple[Dict[str, object], ...] = ()
    test: Optional[Dict[str, object]] = None
    checkpoints: Tuple[str, ...] = ()
    inference: Optional[Dict[str, object]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "protocol": self.protocol,
            "run_config": bits_run_config_to_dict(self.run_config),
            "output_dir": self.output_dir,
            "config_path": self.config_path,
            "result_path": self.result_path,
            "train": list(self.train),
            "val": list(self.val),
            "test": self.test,
            "checkpoints": list(self.checkpoints),
            "inference": self.inference,
        }


OFFICIAL_TBSIM_BITS_CHECKPOINTS = {
    "source": "NVlabs/traffic-behavior-simulation README",
    "download_url": "https://drive.google.com/drive/folders/1y3_HO1c721pFrFOYeGGjORV58g6zNEds?usp=drive_link",
    "example_yaml": "evaluation/BITS_example.yaml",
    "note": (
        "These checkpoints target the original TBSIM architecture. The Tactics2D "
        "implementation reproduces the core BITS model/data flow directly, but it does "
        "not claim binary checkpoint compatibility with the external training framework."
    ),
}


def load_bits_run_config(path) -> BitsRunConfig:
    """Load a JSON NuPlan BITS run config into typed dataclasses."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise ValueError("BITS run config must be a JSON object.")
    return bits_run_config_from_dict(payload)


def save_bits_run_config(path, run_config: BitsRunConfig) -> None:
    """Save a typed BITS run config as JSON for repeatable training/evaluation."""

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(bits_run_config_to_dict(run_config), file, indent=2, sort_keys=True)
        file.write("\n")


def bits_run_config_from_dict(payload: Mapping[str, object]) -> BitsRunConfig:
    """Build a typed BITS run config from a plain dictionary."""

    bits_config_payload = payload.get("config", {})
    split_payload = payload.get("split")
    schedule_payload = payload.get("schedule", {})
    if not isinstance(bits_config_payload, Mapping):
        raise ValueError("config must be an object.")
    if not isinstance(split_payload, Mapping):
        raise ValueError("split must be an object with train/val/test lists.")
    if not isinstance(schedule_payload, Mapping):
        raise ValueError("schedule must be an object.")
    return BitsRunConfig(
        config=BitsConfig(**dict(bits_config_payload)),
        split=_split_from_dict(split_payload),
        schedule=BitsTrainingSchedule(**dict(schedule_payload)),
    )


def bits_run_config_to_dict(run_config: BitsRunConfig) -> Dict[str, object]:
    """Convert a typed BITS run config into a JSON-serializable dictionary."""

    return {
        "config": asdict(run_config.config),
        "split": _split_to_dict(run_config.split),
        "schedule": asdict(run_config.schedule),
    }


def iter_nuplan_bits_batches(
    logs: Sequence[NuPlanLogSpec],
    config: Optional[BitsConfig] = None,
    include_raster: bool = True,
    parser=None,
    require_full_history: bool = True,
    require_full_future: bool = True,
    max_samples_per_log: Optional[int] = None,
    map_cache: Optional[dict] = None,
) -> Iterable:
    """Yield BITS batches from a sequence of NuPlan log specs."""

    for spec in logs:
        dataset = build_nuplan_bits_dataset(
            spec,
            config=config,
            include_raster=include_raster,
            parser=parser,
            require_full_history=require_full_history,
            require_full_future=require_full_future,
            map_cache=map_cache,
        )
        limit = len(dataset) if max_samples_per_log is None else min(max_samples_per_log, len(dataset))
        for index in range(limit):
            yield dataset[index]


def build_nuplan_bits_dataset(
    spec: NuPlanLogSpec,
    config: Optional[BitsConfig] = None,
    include_raster: bool = True,
    parser=None,
    require_full_history: bool = True,
    require_full_future: bool = True,
    map_cache: Optional[dict] = None,
) -> NuPlanBitsDataset:
    """Build one NuPlan-backed BITS dataset from a log spec."""

    return NuPlanBitsDataset(
        data_file=spec.data_file,
        data_folder=spec.data_folder,
        map_file=spec.map_file,
        map_folder=spec.map_folder,
        time_range=spec.time_range,
        parser=parser,
        config=config,
        include_raster=include_raster,
        ego_ids=spec.ego_ids,
        frame_range=spec.frame_range,
        require_full_history=require_full_history,
        require_full_future=require_full_future,
        map_cache=map_cache,
    )


def infer_image_channels(
    logs: Sequence[NuPlanLogSpec],
    config: Optional[BitsConfig] = None,
    parser=None,
) -> int:
    """Read one sample to infer raster channel count."""

    for spec in logs:
        dataset = build_nuplan_bits_dataset(spec, config=config, include_raster=True, parser=parser)
        if len(dataset) > 0:
            return int(dataset[0].image.shape[0])
    raise ValueError("Cannot infer image channels from an empty NuPlan split.")


def train_nuplan_bits_model(
    split: NuPlanBitsSplit,
    output_dir,
    config: Optional[BitsConfig] = None,
    schedule: Optional[BitsTrainingSchedule] = None,
    parser=None,
    device=None,
    dtype=None,
) -> Tuple[BitsBiLevelTorchModel, BitsTrainingHistory]:
    """Train the compact BITS reproduction on a NuPlan split."""

    resolved_config = config or BitsConfig()
    resolved_schedule = schedule or BitsTrainingSchedule()
    if resolved_schedule.epochs < 0:
        raise ValueError("epochs must be non-negative.")
    if resolved_schedule.seed is not None:
        torch.manual_seed(int(resolved_schedule.seed))

    train_logs = split.logs("train")
    if not train_logs:
        raise ValueError("NuPlanBitsSplit.train must contain at least one log.")
    image_channels = infer_image_channels(train_logs, config=resolved_config, parser=parser)
    model = BitsBiLevelTorchModel(
        image_channels=image_channels,
        future_steps=resolved_config.future_steps,
        hidden_dim=resolved_schedule.hidden_dim,
        model_arch=resolved_schedule.model_arch,
        context_size=resolved_schedule.context_size,
        roi_feature_size=resolved_schedule.roi_feature_size,
        roi_layer_key=resolved_schedule.roi_layer_key,
        history_conditioning=resolved_schedule.history_conditioning,
        use_transformer=resolved_schedule.use_transformer,
        config=resolved_config,
    )
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_schedule.learning_rate,
        weight_decay=resolved_schedule.weight_decay,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_history: List[BitsTorchEpochResult] = []
    val_history: List[BitsTorchEpochResult] = []
    checkpoint_paths: List[str] = []
    for epoch in range(1, resolved_schedule.epochs + 1):
        train_result = run_bits_torch_epoch(
            model=model,
            batches=iter_nuplan_bits_batches(
                train_logs,
                config=resolved_config,
                include_raster=True,
                parser=parser,
                max_samples_per_log=resolved_schedule.max_train_samples_per_log,
            ),
            optimizer=optimizer,
            batch_size=resolved_schedule.batch_size,
            device=device,
            dtype=dtype,
            use_ground_truth_goal=resolved_schedule.use_ground_truth_goal,
            num_samples=resolved_schedule.num_samples,
            mask_drivable=resolved_schedule.mask_drivable,
            config=resolved_config,
        )
        train_history.append(train_result)

        if split.val:
            val_result = evaluate_nuplan_bits_split(
                model=model,
                logs=split.val,
                config=resolved_config,
                schedule=resolved_schedule,
                parser=parser,
                device=device,
                dtype=dtype,
            )
            val_history.append(val_result)

        if _should_save_checkpoint(epoch, resolved_schedule):
            checkpoint_path = output_path / f"bits_epoch_{epoch:04d}.pt"
            save_bits_checkpoint(
                checkpoint_path,
                model=model,
                epoch=epoch,
                image_channels=image_channels,
                config=resolved_config,
                schedule=resolved_schedule,
                split=split,
                optimizer=optimizer,
            )
            checkpoint_paths.append(str(checkpoint_path))

    return model, BitsTrainingHistory(
        train=tuple(train_history),
        val=tuple(val_history),
        checkpoints=tuple(checkpoint_paths),
    )


def train_nuplan_bits_planner(
    split: NuPlanBitsSplit,
    output_dir,
    config: Optional[BitsConfig] = None,
    schedule: Optional[BitsTrainingSchedule] = None,
    predictor_checkpoint=None,
    parser=None,
    device=None,
    dtype=None,
    map_location=None,
    freeze_shared_encoder: bool = False,
) -> Tuple[BitsBiLevelTorchModel, BitsTrainingHistory]:
    """Train only the high-level BITS spatial planner on a NuPlan split."""

    resolved_config = config or BitsConfig()
    resolved_schedule = schedule or BitsTrainingSchedule()
    if resolved_schedule.epochs < 0:
        raise ValueError("epochs must be non-negative.")
    if resolved_schedule.seed is not None:
        torch.manual_seed(int(resolved_schedule.seed))

    train_logs = split.logs("train")
    if not train_logs:
        raise ValueError("NuPlanBitsSplit.train must contain at least one log.")
    image_channels = infer_image_channels(train_logs, config=resolved_config, parser=parser)
    model = BitsBiLevelTorchModel(
        image_channels=image_channels,
        future_steps=resolved_config.future_steps,
        hidden_dim=resolved_schedule.hidden_dim,
        model_arch=resolved_schedule.model_arch,
        context_size=resolved_schedule.context_size,
        roi_feature_size=resolved_schedule.roi_feature_size,
        roi_layer_key=resolved_schedule.roi_layer_key,
        history_conditioning=resolved_schedule.history_conditioning,
        use_transformer=resolved_schedule.use_transformer,
        config=resolved_config,
    )
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    if predictor_checkpoint is not None:
        # Initialize the shared encoder from official predictor weights; planner
        # training only learns the spatial goal decoder.
        load_tbsim_bits_inference_weights(
            model,
            predictor_checkpoint=predictor_checkpoint,
            map_location=map_location,
            strict=False,
        )

    optimizer = torch.optim.Adam(
        _spatial_goal_trainable_parameters(
            model,
            freeze_shared_encoder=freeze_shared_encoder,
        ),
        lr=resolved_schedule.learning_rate,
        weight_decay=resolved_schedule.weight_decay,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_history: List[BitsTorchEpochResult] = []
    val_history: List[BitsTorchEpochResult] = []
    checkpoint_paths: List[str] = []
    for epoch in range(1, resolved_schedule.epochs + 1):
        train_result = run_bits_planner_torch_epoch(
            model=model,
            batches=iter_nuplan_bits_batches(
                train_logs,
                config=resolved_config,
                include_raster=True,
                parser=parser,
                max_samples_per_log=resolved_schedule.max_train_samples_per_log,
            ),
            optimizer=optimizer,
            batch_size=resolved_schedule.batch_size,
            device=device,
            dtype=dtype,
            config=resolved_config,
            freeze_shared_encoder=freeze_shared_encoder,
        )
        train_history.append(train_result)

        if split.val:
            val_result = evaluate_nuplan_bits_planner_split(
                model=model,
                logs=split.val,
                config=resolved_config,
                schedule=resolved_schedule,
                parser=parser,
                device=device,
                dtype=dtype,
            )
            val_history.append(val_result)

        if _should_save_checkpoint(epoch, resolved_schedule):
            checkpoint_path = output_path / f"bits_planner_epoch_{epoch:04d}.pt"
            save_bits_checkpoint(
                checkpoint_path,
                model=model,
                epoch=epoch,
                image_channels=image_channels,
                config=resolved_config,
                schedule=resolved_schedule,
                split=split,
                optimizer=optimizer,
                official_checkpoint_note=(
                    "Tactics2D-trained spatial planner decoder with frozen shared encoder."
                    if freeze_shared_encoder
                    else "Tactics2D-trained spatial planner only."
                ),
            )
            checkpoint_paths.append(str(checkpoint_path))

    return model, BitsTrainingHistory(
        train=tuple(train_history),
        val=tuple(val_history),
        checkpoints=tuple(checkpoint_paths),
    )


def run_nuplan_bits_planner_protocol(
    run_config: BitsRunConfig,
    output_dir,
    predictor_checkpoint=None,
    parser=None,
    device=None,
    dtype=None,
    map_location=None,
    freeze_shared_encoder: bool = False,
    save_result: bool = True,
) -> Tuple[BitsBiLevelTorchModel, BitsProtocolResult]:
    """Run a small reproducible NuPlan BITS planner protocol.

    The protocol intentionally stays planner-only: it trains the high-level
    spatial planner, evaluates optional validation/test splits with the same
    planner loss, and writes config/results JSON files for repeatability.
    """

    if not isinstance(run_config, BitsRunConfig):
        raise TypeError("run_config must be a BitsRunConfig.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "bits_run_config.json"
    save_bits_run_config(config_path, run_config)

    model, history = train_nuplan_bits_planner(
        run_config.split,
        output_path,
        config=run_config.config,
        schedule=run_config.schedule,
        predictor_checkpoint=predictor_checkpoint,
        parser=parser,
        device=device,
        dtype=dtype,
        map_location=map_location,
        freeze_shared_encoder=freeze_shared_encoder,
    )
    test_result = None
    if run_config.split.test:
        test_result = evaluate_nuplan_bits_planner_split(
            model,
            run_config.split.test,
            config=run_config.config,
            schedule=run_config.schedule,
            parser=parser,
            device=device,
            dtype=dtype,
        )

    result_path = output_path / "bits_planner_protocol_result.json" if save_result else None
    result = BitsProtocolResult(
        protocol="nuplan_bits_planner_v0",
        run_config=run_config,
        output_dir=str(output_path),
        config_path=str(config_path),
        result_path=None if result_path is None else str(result_path),
        train=tuple(_epoch_result_to_dict(item) for item in history.train),
        val=tuple(_epoch_result_to_dict(item) for item in history.val),
        test=None if test_result is None else _epoch_result_to_dict(test_result),
        checkpoints=history.checkpoints,
        inference={
            "source": "tbsim_predictor_initialized_planner"
            if predictor_checkpoint is not None
            else "tactics2d_planner",
            "uses_tbsim_predictor_checkpoint": predictor_checkpoint is not None,
            "freeze_shared_encoder": bool(freeze_shared_encoder),
        },
    )
    if result_path is not None:
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result.as_dict(), file, indent=2, sort_keys=True)
            file.write("\n")
    return model, result


def run_nuplan_bits_open_loop_protocol(
    run_config: BitsRunConfig,
    output_dir,
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
    parser=None,
    device=None,
    dtype=None,
    map_location=None,
    strict: bool = True,
    save_result: bool = True,
) -> Tuple[BitsBiLevelTorchModel, BitsProtocolResult]:
    """Evaluate a loaded BITS model on NuPlan open-loop splits."""

    if not isinstance(run_config, BitsRunConfig):
        raise TypeError("run_config must be a BitsRunConfig.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "bits_open_loop_run_config.json"
    save_bits_run_config(config_path, run_config)

    resolved_future_steps = future_steps
    if resolved_future_steps is None:
        resolved_future_steps = run_config.config.future_steps
    inference = load_bits_inference_model(
        checkpoint_path=checkpoint_path,
        tactics2d_planner_checkpoint=tactics2d_planner_checkpoint,
        planner_checkpoint=planner_checkpoint,
        predictor_checkpoint=predictor_checkpoint,
        image_channels=image_channels,
        future_steps=resolved_future_steps,
        hidden_dim=hidden_dim,
        model_arch=model_arch,
        context_size=context_size,
        roi_feature_size=roi_feature_size,
        roi_layer_key=roi_layer_key,
        history_conditioning=history_conditioning,
        use_transformer=use_transformer,
        config=run_config.config,
        map_location=map_location,
        strict=strict,
    )
    model = inference.model
    train_result = None
    val_result = None
    test_result = None
    if run_config.split.train:
        train_result = evaluate_nuplan_bits_split(
            model,
            run_config.split.train,
            config=run_config.config,
            schedule=run_config.schedule,
            parser=parser,
            device=device,
            dtype=dtype,
        )
    if run_config.split.val:
        val_result = evaluate_nuplan_bits_split(
            model,
            run_config.split.val,
            config=run_config.config,
            schedule=run_config.schedule,
            parser=parser,
            device=device,
            dtype=dtype,
        )
    if run_config.split.test:
        test_result = evaluate_nuplan_bits_split(
            model,
            run_config.split.test,
            config=run_config.config,
            schedule=run_config.schedule,
            parser=parser,
            device=device,
            dtype=dtype,
        )

    result_path = output_path / "bits_open_loop_protocol_result.json" if save_result else None
    result = BitsProtocolResult(
        protocol="nuplan_bits_open_loop_v0",
        run_config=run_config,
        output_dir=str(output_path),
        config_path=str(config_path),
        result_path=None if result_path is None else str(result_path),
        train=() if train_result is None else (_epoch_result_to_dict(train_result),),
        val=() if val_result is None else (_epoch_result_to_dict(val_result),),
        test=None if test_result is None else _epoch_result_to_dict(test_result),
        checkpoints=tuple(
            str(path)
            for path in (
                checkpoint_path,
                tactics2d_planner_checkpoint,
                planner_checkpoint,
                predictor_checkpoint,
            )
            if path is not None and not isinstance(path, Mapping)
        ),
        inference=inference.as_dict(),
    )
    if result_path is not None:
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result.as_dict(), file, indent=2, sort_keys=True)
            file.write("\n")
    return model, result


def run_nuplan_bits_rolling_protocol(
    run_config: BitsRunConfig,
    output_dir,
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
    split_name: str = "test",
    simulation_steps: int = 1,
    parser=None,
    device=None,
    dtype=None,
    map_location=None,
    strict: bool = True,
    num_samples: Optional[int] = None,
    mask_drivable: bool = True,
    save_result: bool = True,
) -> Tuple[BitsBiLevelTorchModel, BitsProtocolResult]:
    """Run short-horizon BITS closed-loop rollouts on a NuPlan split."""

    if not isinstance(run_config, BitsRunConfig):
        raise TypeError("run_config must be a BitsRunConfig.")
    if simulation_steps < 0:
        raise ValueError("simulation_steps must be non-negative.")
    logs = run_config.split.logs(split_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "bits_rolling_run_config.json"
    save_bits_run_config(config_path, run_config)

    resolved_future_steps = future_steps
    if resolved_future_steps is None:
        resolved_future_steps = run_config.config.future_steps
    inference = load_bits_inference_model(
        checkpoint_path=checkpoint_path,
        tactics2d_planner_checkpoint=tactics2d_planner_checkpoint,
        planner_checkpoint=planner_checkpoint,
        predictor_checkpoint=predictor_checkpoint,
        image_channels=image_channels,
        future_steps=resolved_future_steps,
        hidden_dim=hidden_dim,
        model_arch=model_arch,
        context_size=context_size,
        roi_feature_size=roi_feature_size,
        roi_layer_key=roi_layer_key,
        history_conditioning=history_conditioning,
        use_transformer=use_transformer,
        config=run_config.config,
        map_location=map_location,
        strict=strict,
    )
    model = inference.model
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    scorer = BitsPlanScorer(run_config.config)
    policy = TorchBitsPolicy(
        model,
        device=device,
        dtype=dtype,
        plan_scorer=scorer,
        module_kwargs={
            "use_ground_truth_goal": False,
            "num_samples": num_samples,
            "mask_drivable": mask_drivable,
        },
    )
    behavior_model = BitsBehaviorModel(
        run_config.config,
        policy=policy,
        include_raster=True,
    )
    runner = BitsRollingRunner(run_config.config, behavior_model=behavior_model)

    log_results = []
    for spec in logs:
        dataset = build_nuplan_bits_dataset(
            spec,
            config=run_config.config,
            include_raster=True,
            parser=parser,
        )
        if len(dataset) == 0:
            log_results.append(_rolling_log_result_to_dict(spec, None, skipped_reason="empty_dataset"))
            continue
        start_index = dataset.indices[0]
        rolling_result = runner.run(
            dataset.participants,
            dataset.map,
            start_frame=start_index.frame,
            simulation_steps=simulation_steps,
            agent_ids=[start_index.ego_id],
        )
        evaluation = evaluate_bits_rolling_result(
            rolling_result,
            reference_participants=dataset.participants,
            map_=dataset.map,
        )
        log_results.append(
            _rolling_log_result_to_dict(
                spec,
                evaluation,
                start_frame=start_index.frame,
                ego_id=start_index.ego_id,
            )
        )

    result_path = output_path / "bits_rolling_protocol_result.json" if save_result else None
    rolling_summary = _summarize_rolling_logs(log_results)
    result = BitsProtocolResult(
        protocol="nuplan_bits_rolling_v0",
        run_config=run_config,
        output_dir=str(output_path),
        config_path=str(config_path),
        result_path=None if result_path is None else str(result_path),
        train=(),
        val=(),
        test={
            "split": split_name,
            "simulation_steps": simulation_steps,
            "log_results": log_results,
            "summary": rolling_summary,
        },
        checkpoints=tuple(
            str(path)
            for path in (
                checkpoint_path,
                tactics2d_planner_checkpoint,
                planner_checkpoint,
                predictor_checkpoint,
            )
            if path is not None and not isinstance(path, Mapping)
        ),
        inference=inference.as_dict(),
    )
    if result_path is not None:
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result.as_dict(), file, indent=2, sort_keys=True)
            file.write("\n")
    return model, result


def evaluate_nuplan_bits_split(
    model: BitsBiLevelTorchModel,
    logs: Sequence[NuPlanLogSpec],
    config: Optional[BitsConfig] = None,
    schedule: Optional[BitsTrainingSchedule] = None,
    parser=None,
    device=None,
    dtype=None,
) -> BitsTorchEpochResult:
    """Evaluate a BITS model on a NuPlan split without updating weights."""

    resolved_config = config or BitsConfig()
    resolved_schedule = schedule or BitsTrainingSchedule()
    return run_bits_torch_epoch(
        model=model,
        batches=iter_nuplan_bits_batches(
            logs,
            config=resolved_config,
            include_raster=True,
            parser=parser,
            max_samples_per_log=resolved_schedule.max_val_samples_per_log,
        ),
        optimizer=None,
        batch_size=resolved_schedule.batch_size,
        device=device,
        dtype=dtype,
        use_ground_truth_goal=resolved_schedule.use_ground_truth_goal,
        num_samples=resolved_schedule.num_samples,
        mask_drivable=resolved_schedule.mask_drivable,
        config=resolved_config,
    )


def evaluate_nuplan_bits_planner_split(
    model: BitsBiLevelTorchModel,
    logs: Sequence[NuPlanLogSpec],
    config: Optional[BitsConfig] = None,
    schedule: Optional[BitsTrainingSchedule] = None,
    parser=None,
    device=None,
    dtype=None,
) -> BitsTorchEpochResult:
    """Evaluate only the high-level BITS spatial planner on a NuPlan split."""

    resolved_config = config or BitsConfig()
    resolved_schedule = schedule or BitsTrainingSchedule()
    return run_bits_planner_torch_epoch(
        model=model,
        batches=iter_nuplan_bits_batches(
            logs,
            config=resolved_config,
            include_raster=True,
            parser=parser,
            max_samples_per_log=resolved_schedule.max_val_samples_per_log,
        ),
        optimizer=None,
        batch_size=resolved_schedule.batch_size,
        device=device,
        dtype=dtype,
        config=resolved_config,
    )


def run_nuplan_bits_torch_validation(
    data_file: str,
    data_folder: str,
    map_file: str,
    map_folder: Optional[str] = None,
    time_range=None,
    frame_range=None,
    config: Optional[BitsConfig] = None,
    parser=None,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    hidden_dim: int = 128,
    model_arch: str = "resnet18",
    context_size: int = 30,
    roi_feature_size: int = 7,
    roi_layer_key: str = "layer2",
    history_conditioning: bool = False,
    use_transformer: bool = False,
    use_ground_truth_goal: bool = True,
    num_samples: Optional[int] = None,
    mask_drivable: bool = False,
    device=None,
    dtype=None,
    loss_weights: Optional[Dict[str, float]] = None,
) -> BitsTorchEpochResult:
    """Build one NuPlan-backed dataset and run a compact BITS validation pass."""

    resolved_config = config or BitsConfig()
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative.")
    dataset = NuPlanBitsDataset(
        data_file=data_file,
        data_folder=data_folder,
        map_file=map_file,
        map_folder=map_folder,
        time_range=time_range,
        frame_range=frame_range,
        parser=parser,
        config=resolved_config,
        include_raster=True,
    )
    if max_samples is not None:
        batches = [dataset[index] for index in range(min(max_samples, len(dataset)))]
    else:
        batches = dataset

    image_channels = dataset[0].image.shape[0] if len(dataset) > 0 else 4
    model = BitsBiLevelTorchModel(
        image_channels=image_channels,
        future_steps=resolved_config.future_steps,
        hidden_dim=hidden_dim,
        model_arch=model_arch,
        context_size=context_size,
        roi_feature_size=roi_feature_size,
        roi_layer_key=roi_layer_key,
        history_conditioning=history_conditioning,
        use_transformer=use_transformer,
        config=resolved_config,
    )
    return run_bits_torch_epoch(
        model=model,
        batches=batches,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        use_ground_truth_goal=use_ground_truth_goal,
        num_samples=num_samples,
        mask_drivable=mask_drivable,
        config=resolved_config,
        loss_weights=loss_weights,
    )


def save_bits_checkpoint(
    path,
    model: BitsBiLevelTorchModel,
    epoch: int,
    image_channels: int,
    config: BitsConfig,
    schedule: Optional[BitsTrainingSchedule] = None,
    split: Optional[NuPlanBitsSplit] = None,
    optimizer=None,
    official_checkpoint_note: Optional[str] = None,
) -> None:
    """Save BITS model weights and reproduction metadata."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = BitsCheckpointMetadata(
        epoch=int(epoch),
        image_channels=int(image_channels),
        future_steps=int(config.future_steps),
        hidden_dim=int(getattr(model, "hidden_dim", 128)),
        model_arch=str(getattr(model, "model_arch", "resnet18")),
        context_size=int(getattr(model, "context_size", 30)),
        roi_feature_size=int(getattr(model, "roi_feature_size", 7)),
        roi_layer_key=str(getattr(model, "roi_layer_key", "layer2")),
        history_conditioning=bool(getattr(model, "history_conditioning", False)),
        use_transformer=bool(getattr(model, "use_transformer", False)),
        config=asdict(config),
        schedule={} if schedule is None else asdict(schedule),
        split={} if split is None else _split_to_dict(split),
        official_checkpoint_note=official_checkpoint_note,
    )
    payload = {
        "metadata": asdict(metadata),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)


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

    mapped = _map_tbsim_state_dict_prefixes(
        state_dict,
        prefix_map={
            "model.": "predictor.",
        },
    )
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


def load_tbsim_bits_inference_weights(
    model: BitsBiLevelTorchModel,
    planner_checkpoint=None,
    predictor_checkpoint=None,
    map_location=None,
    strict: bool = True,
) -> BitsCheckpointCompatibilityReport:
    """Load official planner/predictor checkpoints into an inference BITS model."""

    planner_state_dict = None if planner_checkpoint is None else _load_checkpoint_state_dict(
        planner_checkpoint,
        map_location=map_location,
    )
    predictor_state_dict = None if predictor_checkpoint is None else _load_checkpoint_state_dict(
        predictor_checkpoint,
        map_location=map_location,
    )
    mapped_state_dict = merge_tbsim_bits_state_dicts(
        planner_state_dict=planner_state_dict,
        predictor_state_dict=predictor_state_dict,
    )
    return _load_mapped_bits_inference_weights(
        model,
        mapped_state_dict,
        strict=strict,
    )


def _load_mapped_bits_inference_weights(
    model: BitsBiLevelTorchModel,
    mapped_state_dict: Mapping[str, object],
    strict: bool = True,
) -> BitsCheckpointCompatibilityReport:
    """Load already-mapped BITS weights into ``model`` and report compatibility."""

    normalized_state_dict = _normalize_shared_encoder_state_dict(mapped_state_dict)
    report = _build_bits_checkpoint_compatibility_report(model, normalized_state_dict)
    if strict and not report.is_compatible:
        raise ValueError(_format_checkpoint_compatibility_error(report))

    model_state = model.state_dict()
    loadable_state = {
        key: normalized_state_dict[key]
        for key in report.matched_keys
        if key in model_state
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
        (
            "predictor.map_encoder.encoder_heads.",
            "shared_encoder.encoder_heads.",
        ),
        (
            "predictor.map_encoder.roi_head.",
            "predictor.roi_head.",
        ),
        (
            "predictor.map_encoder.agent_net.",
            "predictor.roi_head.agent_net.",
        ),
        (
            "predictor.goal_encoder.",
            "predictor.policy_head.goal_encoder.",
        ),
        (
            "predictor.ego_decoder.",
            "predictor.policy_head.ego_decoder.",
        ),
        (
            "predictor.agents_decoder.",
            "predictor.future_state_head.agents_decoder.",
        ),
        (
            "planner.raster_unet.decoder.",
            "planner.spatial_goal_decoder.decoder.",
        ),
        (
            "planner.raster_unet.encoder_heads.",
            "shared_encoder.encoder_heads.",
        ),
        (
            "planner.raster_unet.",
            "planner.spatial_goal_decoder.",
        ),
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
            model=model,
            source="tactics2d",
            metadata=dict(metadata),
            compatibility=None,
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
                tactics2d_planner_checkpoint,
                map_location=map_location,
            )
        )
    if planner_checkpoint is not None or predictor_checkpoint is not None:
        planner_state_dict = None if planner_checkpoint is None else _load_checkpoint_state_dict(
            planner_checkpoint,
            map_location=map_location,
        )
        predictor_state_dict = None if predictor_checkpoint is None else _load_checkpoint_state_dict(
            predictor_checkpoint,
            map_location=map_location,
        )
        official_state_dict = merge_tbsim_bits_state_dicts(
            planner_state_dict=planner_state_dict,
            predictor_state_dict=predictor_state_dict,
        )
        overlap = set(mapped_state_dict).intersection(official_state_dict)
        non_encoder_overlap = [
            key for key in sorted(overlap) if not key.startswith("shared_encoder.encoder_heads.")
        ]
        if non_encoder_overlap:
            raise ValueError(f"Overlapping BITS checkpoint keys: {non_encoder_overlap[:3]}")
        mapped_state_dict.update(official_state_dict)
    compatibility = _load_mapped_bits_inference_weights(
        model,
        mapped_state_dict,
        strict=strict,
    )
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


def _build_bits_checkpoint_compatibility_report(
    model: BitsBiLevelTorchModel,
    state_dict: Mapping[str, object],
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
                    key=key,
                    expected_shape=expected_shape,
                    found_shape=found_shape,
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


def load_bits_checkpoint(path, map_location=None):
    """Load a BITS checkpoint and reconstruct the compact torch model."""

    payload = torch.load(path, map_location=map_location)
    metadata = payload["metadata"]
    config = BitsConfig(**metadata["config"])
    model = BitsBiLevelTorchModel(
        image_channels=int(metadata["image_channels"]),
        future_steps=int(metadata["future_steps"]),
        hidden_dim=int(metadata.get("hidden_dim", metadata.get("schedule", {}).get("hidden_dim", 128))),
        model_arch=str(metadata.get("model_arch", metadata.get("schedule", {}).get("model_arch", "resnet18"))),
        context_size=int(metadata.get("context_size", metadata.get("schedule", {}).get("context_size", 30))),
        roi_feature_size=int(metadata.get("roi_feature_size", metadata.get("schedule", {}).get("roi_feature_size", 7))),
        roi_layer_key=str(metadata.get("roi_layer_key", metadata.get("schedule", {}).get("roi_layer_key", "layer2"))),
        history_conditioning=bool(
            metadata.get("history_conditioning", metadata.get("schedule", {}).get("history_conditioning", False))
        ),
        use_transformer=bool(
            metadata.get("use_transformer", metadata.get("schedule", {}).get("use_transformer", False))
        ),
        config=config,
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, metadata, payload


def _map_tbsim_state_dict_prefixes(
    state_dict: Mapping[str, object],
    prefix_map: Mapping[str, str],
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


def _epoch_result_to_dict(result: BitsTorchEpochResult) -> Dict[str, object]:
    return {
        "sample_count": result.sample_count,
        "step_count": result.step_count,
        "mean_total_loss": result.mean_total_loss,
        "mean_losses": dict(result.mean_losses),
    }


def _spatial_goal_trainable_parameters(
    model: BitsBiLevelTorchModel,
    freeze_shared_encoder: bool = False,
):
    """Return spatial goal parameters for planner-only training."""

    params = []
    if not freeze_shared_encoder:
        params.extend(model.shared_encoder.parameters())
    params.extend(model.planner.spatial_goal_decoder.parameters())
    return list(params)


def _rolling_log_result_to_dict(
    spec: NuPlanLogSpec,
    evaluation,
    start_frame: Optional[int] = None,
    ego_id=None,
    skipped_reason: Optional[str] = None,
) -> Dict[str, object]:
    payload = {
        "log": asdict(spec),
        "start_frame": start_frame,
        "ego_id": ego_id,
        "skipped_reason": skipped_reason,
    }
    if evaluation is None:
        payload["metrics"] = None
        return payload

    trajectory_errors = {
        str(agent_id): {
            "ade": error.ade,
            "fde": error.fde,
            "samples": error.samples,
        }
        for agent_id, error in evaluation.trajectory_errors.items()
    }
    controlled_error = evaluation.trajectory_errors.get(ego_id)
    payload["metrics"] = {
        "frame_count": evaluation.frame_count,
        "prediction_round_count": evaluation.prediction_round_count,
        "min_distance": evaluation.min_distance,
        "collision_count": evaluation.collision_count,
        "first_collision": evaluation.first_collision,
        "off_drivable_count": evaluation.off_drivable_count,
        "off_drivable_rate": evaluation.off_drivable_rate,
        "first_off_drivable": evaluation.first_off_drivable,
        "mean_ade": evaluation.mean_ade,
        "mean_fde": evaluation.mean_fde,
        "controlled_ade": None if controlled_error is None else controlled_error.ade,
        "controlled_fde": None if controlled_error is None else controlled_error.fde,
        "controlled_samples": None if controlled_error is None else controlled_error.samples,
        "trajectory_errors": trajectory_errors,
    }
    return payload


def _summarize_rolling_logs(log_results: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    metrics = [item.get("metrics") for item in log_results if item.get("metrics") is not None]
    if not metrics:
        return {
            "evaluated_log_count": 0,
            "skipped_log_count": len(log_results),
        }
    mean_ade_values = [item["mean_ade"] for item in metrics if item.get("mean_ade") is not None]
    mean_fde_values = [item["mean_fde"] for item in metrics if item.get("mean_fde") is not None]
    controlled_ade_values = [
        item["controlled_ade"] for item in metrics if item.get("controlled_ade") is not None
    ]
    controlled_fde_values = [
        item["controlled_fde"] for item in metrics if item.get("controlled_fde") is not None
    ]
    return {
        "evaluated_log_count": len(metrics),
        "skipped_log_count": len(log_results) - len(metrics),
        "collision_count": int(sum(item["collision_count"] for item in metrics)),
        "off_drivable_count": int(sum(item["off_drivable_count"] for item in metrics)),
        "mean_off_drivable_rate": float(
            sum(item["off_drivable_rate"] for item in metrics) / len(metrics)
        ),
        "mean_ade": None
        if not mean_ade_values
        else float(sum(mean_ade_values) / len(mean_ade_values)),
        "mean_fde": None
        if not mean_fde_values
        else float(sum(mean_fde_values) / len(mean_fde_values)),
        "mean_controlled_ade": None
        if not controlled_ade_values
        else float(sum(controlled_ade_values) / len(controlled_ade_values)),
        "mean_controlled_fde": None
        if not controlled_fde_values
        else float(sum(controlled_fde_values) / len(controlled_fde_values)),
    }


def _should_save_checkpoint(epoch: int, schedule: BitsTrainingSchedule) -> bool:
    every = int(schedule.checkpoint_every_epochs)
    return every > 0 and epoch % every == 0


def _split_to_dict(split: NuPlanBitsSplit) -> Dict[str, object]:
    return {
        "train": [asdict(spec) for spec in split.train],
        "val": [asdict(spec) for spec in split.val],
        "test": [asdict(spec) for spec in split.test],
    }


def _split_from_dict(payload: Mapping[str, object]) -> NuPlanBitsSplit:
    return NuPlanBitsSplit(
        train=tuple(_log_spec_from_dict(item) for item in payload.get("train", ())),
        val=tuple(_log_spec_from_dict(item) for item in payload.get("val", ())),
        test=tuple(_log_spec_from_dict(item) for item in payload.get("test", ())),
    )


def _log_spec_from_dict(payload: object) -> NuPlanLogSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("Each NuPlan log spec must be an object.")
    values = dict(payload)
    if "data_file" not in values or "map_file" not in values:
        raise ValueError("Each NuPlan log spec requires data_file and map_file.")
    for key in ("time_range", "frame_range", "ego_ids"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return NuPlanLogSpec(**values)
