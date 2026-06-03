# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Persistent sample cache helpers for NuPlan-backed BITS experiments."""

from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from .schema import BitsBatch
from .training import (
    BitsRunConfig,
    bits_run_config_from_dict,
    bits_run_config_to_dict,
    iter_nuplan_bits_batches,
)


BITS_BATCH_CACHE_VERSION = 1


def build_bits_batch_cache(
    run_config: BitsRunConfig,
    cache_dir,
    splits: Sequence[str] = ("train",),
    overwrite: bool = False,
    drop_visual_rasters: bool = True,
    parser=None,
    progress_interval: int = 25,
    max_seconds: Optional[float] = None,
    reuse_maps: bool = True,
    progress_callback: Optional[Callable[[str, int, float], None]] = None,
) -> Dict[str, object]:
    """Build a persistent cache of BITS batches from a NuPlan run config.

    The cache stores one pickled :class:`BitsBatch` per sample plus a JSON
    manifest. It is intentionally simple so repeated planner experiments can
    skip NuPlan parsing and rasterization.
    """

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    manifest = _load_existing_manifest(manifest_path, run_config, overwrite=overwrite)
    started_all = time.perf_counter()
    map_cache = {} if reuse_maps else None

    for split_name in splits:
        _validate_split_name(split_name)
        split_dir = cache_root / split_name
        if split_dir.exists() and overwrite:
            _clear_cache_split(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        existing = manifest.get("splits", {}).get(split_name)
        logs = run_config.split.logs(split_name)
        if existing and not overwrite and _split_cache_complete(cache_root, existing, len(logs)):
            continue

        max_samples = _max_samples_for_split(run_config, split_name)
        split_started = time.perf_counter()
        files = _collect_existing_split_files(split_dir, split_name) if not overwrite else []
        sample_offset = len(files)

        for local_index, batch in enumerate(
            iter_nuplan_bits_batches(
                logs,
                config=run_config.config,
                include_raster=True,
                parser=parser,
                max_samples_per_log=max_samples,
                map_cache=map_cache,
            )
        ):
            sample_index = sample_offset + local_index
            if max_seconds is not None and time.perf_counter() - started_all > max_seconds:
                break

            cached_batch = _prepare_batch_for_cache(batch, drop_visual_rasters=drop_visual_rasters)
            relative_path = Path(split_name) / f"sample_{sample_index:06d}.pkl"
            save_bits_batch(cache_root / relative_path, cached_batch)
            files.append(
                {
                    "path": relative_path.as_posix(),
                    "frame": int(cached_batch.frame),
                    "ego_id": str(cached_batch.ego_id),
                    "agent_count": len(cached_batch.agent_ids),
                }
            )

            if progress_callback is not None and progress_interval > 0:
                if (sample_index + 1) % progress_interval == 0:
                    progress_callback(split_name, sample_index + 1, time.perf_counter() - split_started)
                    _update_split_manifest(
                        manifest,
                        split_name,
                        files,
                        split_started,
                        max_samples,
                        drop_visual_rasters,
                        reuse_maps,
                    )
                    _save_manifest(manifest_path, manifest)

        _update_split_manifest(
            manifest,
            split_name,
            files,
            split_started,
            max_samples,
            drop_visual_rasters,
            reuse_maps,
        )
        _save_manifest(manifest_path, manifest)

    manifest["total_seconds"] = time.perf_counter() - started_all
    _save_manifest(manifest_path, manifest)
    return manifest


def build_bits_batch_cache_parallel(
    run_config: BitsRunConfig,
    cache_dir,
    splits: Sequence[str] = ("train",),
    overwrite: bool = False,
    drop_visual_rasters: bool = True,
    progress_callback: Optional[Callable[[str, int, float], None]] = None,
    max_seconds: Optional[float] = None,
    max_workers: int = 2,
) -> Dict[str, object]:
    """Build a BITS batch cache by processing NuPlan logs in parallel."""

    if max_workers <= 1:
        return build_bits_batch_cache(
            run_config,
            cache_dir,
            splits=splits,
            overwrite=overwrite,
            drop_visual_rasters=drop_visual_rasters,
            progress_interval=5,
            max_seconds=max_seconds,
            progress_callback=progress_callback,
        )

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    manifest = _load_existing_manifest(manifest_path, run_config, overwrite=overwrite)
    started_all = time.perf_counter()
    run_config_payload = bits_run_config_to_dict(run_config)

    for split_name in splits:
        _validate_split_name(split_name)
        split_dir = cache_root / split_name
        if split_dir.exists() and overwrite:
            _clear_cache_split(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        existing = manifest.get("splits", {}).get(split_name)
        logs = run_config.split.logs(split_name)
        if existing and not overwrite and _split_cache_complete(cache_root, existing, len(logs)):
            continue

        files = _collect_existing_split_files(split_dir, split_name) if not overwrite else []
        completed_log_indices = {
            int(item["log_index"])
            for item in files
            if item.get("log_index") is not None and item.get("sample_index") == 0
        }
        max_samples = _max_samples_for_split(run_config, split_name)
        split_started = time.perf_counter()
        next_log_index = 0
        futures = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            while next_log_index < len(logs) or futures:
                if max_seconds is not None and time.perf_counter() - started_all > max_seconds:
                    for future in futures:
                        future.cancel()
                    break

                while next_log_index < len(logs) and len(futures) < max_workers:
                    if next_log_index in completed_log_indices:
                        next_log_index += 1
                        continue
                    future = executor.submit(
                        _build_log_cache_worker,
                        run_config_payload,
                        split_name,
                        next_log_index,
                        str(cache_root),
                        max_samples,
                        drop_visual_rasters,
                    )
                    futures[future] = next_log_index
                    next_log_index += 1

                if not futures:
                    continue

                done, _pending = wait(tuple(futures.keys()), timeout=2.0, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    log_index = futures.pop(future)
                    result = future.result()
                    files.extend(result["files"])
                    completed_log_indices.add(log_index)
                    if progress_callback is not None:
                        progress_callback(split_name, len(files), time.perf_counter() - split_started)
                    _update_split_manifest(
                        manifest,
                        split_name,
                        _sort_cache_files(files),
                        split_started,
                        max_samples,
                        drop_visual_rasters,
                        reuse_maps=False,
                    )
                    manifest["splits"][split_name]["parallel_workers"] = max_workers
                    _save_manifest(manifest_path, manifest)

        files = _sort_cache_files(files)
        _update_split_manifest(
            manifest,
            split_name,
            files,
            split_started,
            max_samples,
            drop_visual_rasters,
            reuse_maps=False,
        )
        manifest["splits"][split_name]["parallel_workers"] = max_workers
        _save_manifest(manifest_path, manifest)

    manifest["total_seconds"] = time.perf_counter() - started_all
    _save_manifest(manifest_path, manifest)
    return manifest


def iter_bits_batch_cache(cache_dir, split: str = "train") -> Iterator[BitsBatch]:
    """Yield cached BITS batches for a split in manifest order."""

    cache_root = Path(cache_dir)
    manifest = load_bits_batch_cache_manifest(cache_root)
    split_manifest = manifest.get("splits", {}).get(split)
    if not isinstance(split_manifest, Mapping):
        raise ValueError(f"Cache split {split!r} is not available.")
    for item in split_manifest.get("files", ()):
        yield load_bits_batch(cache_root / item["path"])


def load_bits_batch_cache(cache_dir, split: str = "train") -> Tuple[BitsBatch, ...]:
    """Load a cached split into memory."""

    return tuple(iter_bits_batch_cache(cache_dir, split=split))


def load_bits_batch_cache_manifest(cache_dir) -> Dict[str, object]:
    """Load a BITS batch cache manifest."""

    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"BITS batch cache manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("version") != BITS_BATCH_CACHE_VERSION:
        raise ValueError("Unsupported BITS batch cache version.")
    return manifest


def rebuild_bits_batch_cache_manifest(
    run_config: BitsRunConfig,
    cache_dir,
    splits: Sequence[str] = ("train",),
) -> Dict[str, object]:
    """Recreate a manifest from already-written cache sample files."""

    cache_root = Path(cache_dir)
    manifest = {
        "version": BITS_BATCH_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_config": bits_run_config_to_dict(run_config),
        "splits": {},
    }
    for split_name in splits:
        _validate_split_name(split_name)
        split_dir = cache_root / split_name
        files = _collect_existing_split_files(split_dir, split_name)
        manifest["splits"][split_name] = {
            "sample_count": len(files),
            "seconds": None,
            "files": files,
            "max_samples_per_log": _max_samples_for_split(run_config, split_name),
            "drop_visual_rasters": None,
            "reuse_maps": None,
            "rebuilt_from_files": True,
        }
    _save_manifest(cache_root / "manifest.json", manifest)
    return manifest


def save_bits_batch(path, batch: BitsBatch) -> None:
    """Save one BITS batch with pickle."""

    batch_path = Path(path)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with batch_path.open("wb") as file:
        pickle.dump(batch, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_bits_batch(path) -> BitsBatch:
    """Load one pickled BITS batch."""

    with Path(path).open("rb") as file:
        batch = pickle.load(file)
    if not isinstance(batch, BitsBatch):
        raise TypeError(f"Cached object is not a BitsBatch: {path}")
    return batch


def _build_log_cache_worker(
    run_config_payload,
    split_name: str,
    log_index: int,
    cache_dir: str,
    max_samples_per_log,
    drop_visual_rasters: bool,
) -> Dict[str, object]:
    run_config = bits_run_config_from_dict(run_config_payload)
    spec = run_config.split.logs(split_name)[log_index]
    cache_root = Path(cache_dir)
    files = []
    started = time.perf_counter()

    for sample_index, batch in enumerate(
        iter_nuplan_bits_batches(
            (spec,),
            config=run_config.config,
            include_raster=True,
            max_samples_per_log=max_samples_per_log,
        )
    ):
        cached_batch = _prepare_batch_for_cache(batch, drop_visual_rasters=drop_visual_rasters)
        relative_path = Path(split_name) / f"log_{log_index:04d}_sample_{sample_index:04d}.pkl"
        save_bits_batch(cache_root / relative_path, cached_batch)
        files.append(
            {
                "path": relative_path.as_posix(),
                "frame": int(cached_batch.frame),
                "ego_id": str(cached_batch.ego_id),
                "agent_count": len(cached_batch.agent_ids),
                "log_index": log_index,
                "sample_index": sample_index,
            }
        )

    return {
        "split": split_name,
        "log_index": log_index,
        "sample_count": len(files),
        "seconds": time.perf_counter() - started,
        "files": files,
    }


def _load_existing_manifest(path: Path, run_config: BitsRunConfig, overwrite: bool) -> Dict[str, object]:
    run_config_payload = bits_run_config_to_dict(run_config)
    if path.exists() and not overwrite:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not _cache_run_config_matches(manifest.get("run_config"), run_config_payload):
            raise ValueError(
                "Existing BITS batch cache was built for a different run_config; "
                "choose another cache directory or rebuild with overwrite=True."
            )
        return manifest
    return {
        "version": BITS_BATCH_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config_payload,
        "splits": {},
    }


def _cache_run_config_matches(existing, current) -> bool:
    if not isinstance(existing, Mapping) or not isinstance(current, Mapping):
        return False
    if _json_normalize(existing.get("config")) != _json_normalize(current.get("config")):
        return False
    if _json_normalize(existing.get("split")) != _json_normalize(current.get("split")):
        return False
    existing_schedule = existing.get("schedule", {})
    current_schedule = current.get("schedule", {})
    for key in ("max_train_samples_per_log", "max_val_samples_per_log"):
        if existing_schedule.get(key) != current_schedule.get(key):
            return False
    return True


def _json_normalize(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _save_manifest(path: Path, manifest: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


def _clear_cache_split(path: Path) -> None:
    if not path.exists():
        return
    for child in path.glob("*.pkl"):
        child.unlink()


def _split_files_exist(cache_root: Path, split_manifest: Mapping[str, object]) -> bool:
    files = split_manifest.get("files", ())
    return bool(files) and all((cache_root / item["path"]).exists() for item in files)


def _split_cache_complete(
    cache_root: Path,
    split_manifest: Mapping[str, object],
    expected_logs: int,
) -> bool:
    if not _split_files_exist(cache_root, split_manifest):
        return False
    if split_manifest.get("parallel_workers"):
        completed = {
            int(item["log_index"])
            for item in split_manifest.get("files", ())
            if item.get("log_index") is not None
        }
        return len(completed) >= expected_logs
    return True


def _collect_existing_split_files(split_dir: Path, split_name: str) -> list:
    files = []
    for path in sorted(split_dir.glob("*.pkl")):
        log_index = None
        sample_index = None
        parts = path.stem.split("_")
        if len(parts) == 4 and parts[0] == "log" and parts[2] == "sample":
            log_index = int(parts[1])
            sample_index = int(parts[3])
        files.append(
            {
                "path": (Path(split_name) / path.name).as_posix(),
                "frame": None,
                "ego_id": None,
                "agent_count": None,
                "log_index": log_index,
                "sample_index": sample_index,
            }
        )
    return files


def _sort_cache_files(files: Iterable[Mapping[str, object]]) -> list:
    return sorted(
        files,
        key=lambda item: (
            item.get("log_index") if item.get("log_index") is not None else 10**9,
            item.get("sample_index") if item.get("sample_index") is not None else 10**9,
            item.get("path", ""),
        ),
    )


def _update_split_manifest(
    manifest: Dict[str, object],
    split_name: str,
    files: list,
    split_started: float,
    max_samples,
    drop_visual_rasters: bool,
    reuse_maps: bool,
) -> None:
    split_seconds = time.perf_counter() - split_started
    manifest.setdefault("splits", {})[split_name] = {
        "sample_count": len(files),
        "seconds": split_seconds,
        "files": files,
        "max_samples_per_log": max_samples,
        "drop_visual_rasters": bool(drop_visual_rasters),
        "reuse_maps": bool(reuse_maps),
    }


def _prepare_batch_for_cache(batch: BitsBatch, drop_visual_rasters: bool) -> BitsBatch:
    if not drop_visual_rasters:
        return batch
    return replace(batch, static_image=None, dynamic_image=None)


def _max_samples_for_split(run_config: BitsRunConfig, split: str) -> Optional[int]:
    if split == "train":
        return run_config.schedule.max_train_samples_per_log
    return run_config.schedule.max_val_samples_per_log


def _validate_split_name(split: str) -> None:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of 'train', 'val', or 'test'.")
