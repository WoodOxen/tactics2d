from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from tactics2d.behavior.bits.cache import (
    build_bits_batch_cache,
    build_bits_batch_cache_parallel,
    load_bits_batch_cache,
    rebuild_bits_batch_cache_manifest,
)
from tactics2d.behavior.bits.training import (
    bits_run_config_to_dict,
    load_bits_checkpoint,
    load_bits_inference_model,
    load_bits_run_config,
    save_bits_checkpoint,
    save_bits_run_config,
)
from tactics2d.behavior.bits.torch_model import run_bits_planner_torch_epoch


def _epoch_result_to_dict(result):
    return {
        "sample_count": result.sample_count,
        "step_count": result.step_count,
        "mean_total_loss": result.mean_total_loss,
        "mean_losses": dict(result.mean_losses),
    }


def _print_cache_progress(split_name, count, elapsed):
    print(f"[cache] {split_name}: {count} samples in {elapsed / 60:.1f} min", flush=True)


def _ensure_disk_cache(run_config, cache_dir, splits, overwrite=False, max_seconds=None, workers=1):
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "manifest.json"
    if manifest_path.exists() and not overwrite:
        print(f"[cache] inspect manifest={manifest_path}", flush=True)

    print(f"[cache] build cache at {cache_path}", flush=True)
    builder = build_bits_batch_cache_parallel if workers and workers > 1 else build_bits_batch_cache
    kwargs = {
        "splits": splits,
        "overwrite": overwrite,
        "max_seconds": max_seconds,
        "progress_callback": _print_cache_progress,
    }
    if workers and workers > 1:
        kwargs["max_workers"] = workers
    else:
        kwargs["progress_interval"] = 5
    manifest = builder(run_config, cache_path, **kwargs)
    return manifest


def _load_cached_split(cache_dir, split_name):
    started = time.perf_counter()
    batches = list(load_bits_batch_cache(cache_dir, split=split_name))
    elapsed = time.perf_counter() - started
    print(f"[cache] loaded {split_name}: {len(batches)} samples in {elapsed:.2f}s", flush=True)
    return batches, elapsed


def _save_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--predictor-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples-per-log", type=int, default=4)
    parser.add_argument("--max-val-samples-per-log", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--eval-every-epochs", type=int, default=0)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--rebuild-cache-manifest-only", action="store_true")
    parser.add_argument("--max-cache-seconds", type=float, default=None)
    parser.add_argument("--cache-workers", type=int, default=1)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(int(args.seed))

    run_config = load_bits_run_config(args.config)
    schedule = replace(
        run_config.schedule,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_samples_per_log=args.max_train_samples_per_log,
        max_val_samples_per_log=args.max_val_samples_per_log,
        seed=args.seed,
    )
    run_config = replace(run_config, schedule=schedule)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_bits_run_config(output_dir / "bits_run_config.json", run_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] warm_start={args.warm_start}", flush=True)
    print(f"[setup] predictor_checkpoint={args.predictor_checkpoint}", flush=True)
    print(
        "[setup] target="
        f"{len(run_config.split.train)} logs * "
        f"{schedule.max_train_samples_per_log} samples/log * "
        f"{schedule.epochs} epochs",
        flush=True,
    )

    cache_splits = ["train"]
    if args.final_eval:
        cache_splits.extend(["val", "test"])
    if args.rebuild_cache_manifest_only:
        manifest = rebuild_bits_batch_cache_manifest(run_config, args.cache_dir, splits=cache_splits)
        train_count = manifest.get("splits", {}).get("train", {}).get("sample_count", 0)
        print(f"[cache] rebuilt manifest train sample_count={train_count}", flush=True)
        return

    cache_started = time.perf_counter()
    manifest = _ensure_disk_cache(
        run_config,
        args.cache_dir,
        cache_splits,
        overwrite=args.overwrite_cache and not args.reuse_cache,
        max_seconds=args.max_cache_seconds,
        workers=args.cache_workers,
    )
    cache_seconds = time.perf_counter() - cache_started
    train_count = manifest.get("splits", {}).get("train", {}).get("sample_count", 0)
    print(f"[cache] train sample_count={train_count}", flush=True)
    if args.build_cache_only:
        print(f"[done] cache_only total_time={cache_seconds / 60:.1f} min", flush=True)
        return
    if train_count <= 0:
        raise RuntimeError("No train batches were cached.")

    train_batches, load_seconds = _load_cached_split(args.cache_dir, "train")
    image_channels = int(train_batches[0].image.shape[0])
    val_batches = []
    val_cache_seconds = 0.0
    if args.eval_every_epochs and args.eval_every_epochs > 0:
        split_manifest = manifest.get("splits", {}).get("val")
        if split_manifest and split_manifest.get("sample_count", 0) > 0:
            val_batches, val_cache_seconds = _load_cached_split(args.cache_dir, "val")

    if args.predictor_checkpoint:
        inference = load_bits_inference_model(
            tactics2d_planner_checkpoint=args.warm_start,
            predictor_checkpoint=args.predictor_checkpoint,
            image_channels=image_channels,
            future_steps=run_config.config.future_steps,
            hidden_dim=schedule.hidden_dim,
            model_arch=schedule.model_arch,
            context_size=schedule.context_size,
            roi_feature_size=schedule.roi_feature_size,
            roi_layer_key=schedule.roi_layer_key,
            history_conditioning=schedule.history_conditioning,
            use_transformer=schedule.use_transformer,
            config=run_config.config,
            map_location=device,
            strict=args.warm_start is not None,
        )
        model = inference.model
        metadata = dict(inference.metadata)
        metadata["epoch"] = None
    elif args.warm_start:
        model, metadata, _ = load_bits_checkpoint(args.warm_start, map_location=device)
    else:
        raise ValueError("Provide either --warm-start, --predictor-checkpoint, or both.")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.planner.spatial_goal_decoder.parameters(),
        lr=schedule.learning_rate,
        weight_decay=schedule.weight_decay,
    )

    history = []
    checkpoints = []
    best_val_loss = None
    best_checkpoint = None
    train_started = time.perf_counter()

    def save_epoch_checkpoint(epoch, name=None):
        checkpoint_name = name or f"bits_planner_epoch_{epoch:04d}.pt"
        checkpoint_path = output_dir / checkpoint_name
        save_bits_checkpoint(
            checkpoint_path,
            model=model,
            epoch=epoch,
            image_channels=image_channels,
            config=run_config.config,
            schedule=schedule,
            split=run_config.split,
            optimizer=optimizer,
            official_checkpoint_note=(
                "Tactics2D spatial planner decoder warm-started from the previous "
                "all-mini planner checkpoint; official predictor/shared encoder kept frozen."
                if args.predictor_checkpoint and args.warm_start
                else "Tactics2D spatial planner decoder trained from random initialization; "
                "official predictor/shared encoder kept frozen."
                if args.predictor_checkpoint
                else "Tactics2D spatial planner decoder warm-started from the previous "
                "all-mini planner checkpoint; local shared encoder kept frozen."
            ),
        )
        return checkpoint_path

    for epoch in range(1, schedule.epochs + 1):
        epoch_started = time.perf_counter()
        result = run_bits_planner_torch_epoch(
            model=model,
            batches=train_batches,
            optimizer=optimizer,
            batch_size=schedule.batch_size,
            device=device,
            dtype=torch.float32,
            config=run_config.config,
            freeze_shared_encoder=True,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        result_payload = _epoch_result_to_dict(result)
        result_payload["seconds"] = epoch_seconds
        result_payload["minutes"] = epoch_seconds / 60

        val_payload = None
        if val_batches and epoch % int(args.eval_every_epochs) == 0:
            eval_started = time.perf_counter()
            val_result = run_bits_planner_torch_epoch(
                model=model,
                batches=val_batches,
                optimizer=None,
                batch_size=schedule.batch_size,
                device=device,
                dtype=torch.float32,
                config=run_config.config,
                freeze_shared_encoder=True,
            )
            val_seconds = time.perf_counter() - eval_started
            val_payload = {
                **_epoch_result_to_dict(val_result),
                "cache_seconds": val_cache_seconds,
                "seconds": val_seconds,
                "minutes": val_seconds / 60,
            }
            result_payload["val"] = val_payload
            if best_val_loss is None or val_result.mean_total_loss < best_val_loss:
                best_val_loss = val_result.mean_total_loss
                best_checkpoint = save_epoch_checkpoint(epoch, name="bits_planner_best_val.pt")

        should_save_interval = (
            args.checkpoint_every_epochs
            and args.checkpoint_every_epochs > 0
            and epoch % int(args.checkpoint_every_epochs) == 0
        )
        should_save_final = epoch == schedule.epochs
        checkpoint_path = None
        if should_save_interval or should_save_final:
            checkpoint_path = save_epoch_checkpoint(epoch)
            checkpoints.append(str(checkpoint_path))

        history.append(result_payload)

        print(
            "[epoch] "
            f"{epoch}/{schedule.epochs} "
            f"samples={result.sample_count} steps={result.step_count} "
            f"loss={result.mean_total_loss:.4f} "
            + (
                f"val_loss={val_payload['mean_total_loss']:.4f} "
                if val_payload is not None
                else ""
            )
            + f"time={epoch_seconds / 60:.1f} min "
            + (f"checkpoint={checkpoint_path}" if checkpoint_path is not None else ""),
            flush=True,
        )

        _save_result(
            output_dir / "bits_planner_cached_train_result.json",
            {
                "protocol": "nuplan_bits_planner_cached_warm_start_v0",
                "output_dir": str(output_dir),
                "warm_start": None if args.warm_start is None else str(args.warm_start),
                "predictor_checkpoint": None
                if args.predictor_checkpoint is None
                else str(args.predictor_checkpoint),
                "warm_start_epoch": metadata.get("epoch"),
                "run_config": bits_run_config_to_dict(run_config),
                "cache_seconds": cache_seconds,
                "cache_load_seconds": load_seconds,
                "cache_dir": str(args.cache_dir),
                "train_seconds": time.perf_counter() - train_started,
                "train": history,
                "best_val_loss": best_val_loss,
                "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
                "checkpoints": checkpoints,
            },
        )

    eval_payload = {}
    if args.final_eval:
        for split_name in ("val", "test"):
            split_manifest = manifest.get("splits", {}).get(split_name)
            if not split_manifest or split_manifest.get("sample_count", 0) <= 0:
                continue
            eval_batches, eval_cache_seconds = _load_cached_split(args.cache_dir, split_name)
            eval_started = time.perf_counter()
            eval_result = run_bits_planner_torch_epoch(
                model=model,
                batches=eval_batches,
                optimizer=None,
                batch_size=schedule.batch_size,
                device=device,
                dtype=torch.float32,
                config=run_config.config,
                freeze_shared_encoder=True,
            )
            eval_seconds = time.perf_counter() - eval_started
            eval_payload[split_name] = {
                **_epoch_result_to_dict(eval_result),
                "cache_seconds": eval_cache_seconds,
                "seconds": eval_seconds,
                "minutes": eval_seconds / 60,
            }
            print(
                f"[eval] {split_name} samples={eval_result.sample_count} "
                f"loss={eval_result.mean_total_loss:.4f} time={eval_seconds / 60:.1f} min",
                flush=True,
            )

    total_seconds = cache_seconds + (time.perf_counter() - train_started)
    _save_result(
        output_dir / "bits_planner_cached_train_result.json",
        {
            "protocol": "nuplan_bits_planner_cached_warm_start_v0",
            "output_dir": str(output_dir),
            "warm_start": None if args.warm_start is None else str(args.warm_start),
            "predictor_checkpoint": None
            if args.predictor_checkpoint is None
            else str(args.predictor_checkpoint),
            "warm_start_epoch": metadata.get("epoch"),
            "run_config": bits_run_config_to_dict(run_config),
            "cache_seconds": cache_seconds,
            "cache_load_seconds": load_seconds,
            "cache_dir": str(args.cache_dir),
            "train_seconds": time.perf_counter() - train_started,
            "total_seconds": total_seconds,
            "train": history,
            "eval": eval_payload,
            "best_val_loss": best_val_loss,
            "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
            "checkpoints": checkpoints,
        },
    )
    print(f"[done] total_time={total_seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
