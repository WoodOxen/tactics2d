# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Default BITS policy artifacts used by the public behavior model."""

import os
from pathlib import Path
from typing import Optional


DEFAULT_BITS_RUN_NAME = "nuplan_mini_all_unet_v1"
DEFAULT_BITS_TRAIN_RUN = "train_valid57_4sample_clean_decoder_e120"
DEFAULT_BITS_PLANNER_SUBDIR = "planner_frozen_predictor"
DEFAULT_BITS_PREDICTOR_RUN = "nusc_dynUnicycle_gl0_yrl0_tfTrue_4130553"
DEFAULT_BITS_PREDICTOR_CHECKPOINT = "iter94000_ep6_valLoss0.06.ckpt"


def default_bits_checkpoint_root() -> Path:
    """Return the root folder that contains default BITS artifacts."""

    return Path(
        os.environ.get("TACTICS2D_BITS_CHECKPOINT_ROOT", "D:/tactics/checkpoints/bits")
    ).expanduser()


def default_bits_artifact_paths(checkpoint_root: Optional[Path] = None) -> dict:
    """Return the default run config, planner, and predictor artifact paths."""

    root = default_bits_checkpoint_root() if checkpoint_root is None else Path(checkpoint_root)
    run_root = root / DEFAULT_BITS_RUN_NAME
    train_run_dir = run_root / DEFAULT_BITS_TRAIN_RUN
    planner_dir = train_run_dir / DEFAULT_BITS_PLANNER_SUBDIR
    predictor_checkpoint = (
        root
        / DEFAULT_BITS_PREDICTOR_RUN
        / "run0"
        / "checkpoints"
        / DEFAULT_BITS_PREDICTOR_CHECKPOINT
    )
    return {
        "checkpoint_root": root,
        "run_config": planner_dir / "bits_run_config.json",
        "planner_checkpoint": planner_dir / "bits_planner_best_val.pt",
        "predictor_checkpoint": predictor_checkpoint,
    }


def load_default_bits_policy(
    *,
    device=None,
    dtype=None,
    map_location=None,
    num_samples: Optional[int] = 8,
    mask_drivable: bool = True,
) -> tuple:
    """Load the default weighted BITS policy and its matching config.

    Returns:
        ``(config, policy)`` where ``policy`` implements ``predict_batch``.
    """

    paths = default_bits_artifact_paths()
    missing = {
        name: path
        for name, path in paths.items()
        if name != "checkpoint_root" and not path.exists()
    }
    if missing:
        raise FileNotFoundError(
            "Missing default BITS artifacts. Set TACTICS2D_BITS_CHECKPOINT_ROOT "
            "to the checkpoint root or install the default BITS artifacts: "
            + str(missing)
        )

    try:
        from .model import BitsPlanScorer
        from .torch_model import TorchBitsPolicy
        from .training import load_bits_inference_model, load_bits_run_config
    except ImportError as exc:
        raise ImportError(
            "The default BITS policy requires the optional BITS dependencies. "
            "Install them with tactics2d[bits]."
        ) from exc

    run_config = load_bits_run_config(paths["run_config"])
    resolved_map_location = map_location if map_location is not None else device
    inference = load_bits_inference_model(
        tactics2d_planner_checkpoint=paths["planner_checkpoint"],
        predictor_checkpoint=paths["predictor_checkpoint"],
        image_channels=int(run_config.config.history_steps + 4),
        future_steps=run_config.config.future_steps,
        hidden_dim=run_config.schedule.hidden_dim,
        model_arch=run_config.schedule.model_arch,
        context_size=run_config.schedule.context_size,
        roi_feature_size=run_config.schedule.roi_feature_size,
        roi_layer_key=run_config.schedule.roi_layer_key,
        history_conditioning=run_config.schedule.history_conditioning,
        use_transformer=run_config.schedule.use_transformer,
        config=run_config.config,
        map_location=resolved_map_location,
        strict=True,
    )
    module = inference.model
    if device is not None:
        module.to(device)
    if dtype is not None:
        module.to(dtype=dtype)
    module.eval()
    policy = TorchBitsPolicy(
        module,
        device=device,
        dtype=dtype,
        plan_scorer=BitsPlanScorer(run_config.config),
        module_kwargs={
            "use_ground_truth_goal": False,
            "num_samples": num_samples,
            "mask_drivable": mask_drivable,
        },
    )
    return run_config.config, policy


__all__ = [
    "DEFAULT_BITS_RUN_NAME",
    "DEFAULT_BITS_TRAIN_RUN",
    "default_bits_artifact_paths",
    "default_bits_checkpoint_root",
    "load_default_bits_policy",
]
