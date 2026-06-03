# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model-facing interfaces for BITS-style trajectory prediction."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from tactics2d.geometry import normalize_angle, oriented_box
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import State, Trajectory

from .config import BitsConfig
from .dataset import BitsBatchBuilder
from .schema import BitsBatch


@dataclass(frozen=True)
class BitsPrediction:
    """BITS prediction in the current ego frame."""

    positions: np.ndarray
    yaws: np.ndarray
    availabilities: np.ndarray
    scores: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "positions": self.positions,
            "yaws": self.yaws,
            "availabilities": self.availabilities,
            "scores": self.scores,
        }


@dataclass(frozen=True)
class BitsPlan:
    """Spatial plan candidates in the current ego frame."""

    positions: np.ndarray
    yaws: np.ndarray
    availabilities: np.ndarray
    scores: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "positions": self.positions,
            "yaws": self.yaws,
            "availabilities": self.availabilities,
            "scores": self.scores,
        }


@dataclass(frozen=True)
class BitsAgentPrediction:
    """Neighboring-agent predictions aligned with BITS plan candidates."""

    positions: np.ndarray
    yaws: np.ndarray
    availabilities: np.ndarray


@dataclass(frozen=True)
class BitsPlanScoreBreakdown:
    """Rule-based score terms for spatial plan candidates."""

    total: np.ndarray
    likelihood: np.ndarray
    progress: np.ndarray
    lane: np.ndarray
    collision: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "total": self.total,
            "likelihood": self.likelihood,
            "progress": self.progress,
            "lane": self.lane,
            "collision": self.collision,
        }


class BitsPolicy:
    """Base interface for BITS-compatible policies."""

    def predict_batch(self, batch: BitsBatch) -> BitsPrediction:
        raise NotImplementedError


class BitsPlanScorer:
    """Score BITS candidate trajectories with planner likelihood and closed-loop costs."""

    def __init__(self, config: Optional[BitsConfig] = None):
        self.config = config or BitsConfig()

    def score_batch(
        self,
        batch: BitsBatch,
        plan: BitsPlan,
        agent_prediction: Optional[BitsAgentPrediction] = None,
    ) -> BitsPlanScoreBreakdown:
        likelihood = self.config.likelihood_weight * np.asarray(plan.scores, dtype=float)
        progress = self.config.progress_weight * self._progress_scores(plan)
        lane = -self.config.lane_weight * self._drivable_violations(batch, plan)
        collision = -self.config.collision_weight * self._collision_violations(
            batch,
            plan,
            agent_prediction,
        )
        total = likelihood + progress + lane + collision
        return BitsPlanScoreBreakdown(
            total=total,
            likelihood=likelihood,
            progress=progress,
            lane=lane,
            collision=collision,
        )

    @staticmethod
    def select_plan(plan: BitsPlan, scores: BitsPlanScoreBreakdown) -> BitsPlan:
        index = int(np.argmax(scores.total))
        return BitsPlan(
            positions=plan.positions[[index]].copy(),
            yaws=plan.yaws[[index]].copy(),
            availabilities=plan.availabilities[[index]].copy(),
            scores=scores.total[[index]].copy(),
        )

    @staticmethod
    def _progress_scores(plan: BitsPlan) -> np.ndarray:
        scores = np.zeros(plan.positions.shape[0], dtype=float)
        for index in range(plan.positions.shape[0]):
            available = np.flatnonzero(plan.availabilities[index])
            if available.size < 2:
                continue
            points = plan.positions[index, available]
            distances = np.linalg.norm(points[1:] - points[:-1], axis=-1)
            scores[index] = float(np.sum(distances))
        return scores

    def _drivable_violations(self, batch: BitsBatch, plan: BitsPlan) -> np.ndarray:
        violations = np.zeros(plan.positions.shape[0], dtype=float)
        if batch.drivable_map is None or batch.raster_from_agent is None:
            return violations

        distance_map = self._drivable_distance_map(batch.drivable_map)
        height, width = distance_map.shape
        for index in range(plan.positions.shape[0]):
            available = np.flatnonzero(plan.availabilities[index])
            if available.size == 0:
                continue
            distances = []
            for step in available:
                raster_point = self._transform_point(
                    plan.positions[index, step], batch.raster_from_agent
                )
                col = int(np.rint(raster_point[0]))
                row = int(np.rint(raster_point[1]))
                if row < 0 or row >= height or col < 0 or col >= width:
                    distances.append(self.config.drivable_distance_clip)
                else:
                    distances.append(float(distance_map[row, col]))
            violations[index] = float(np.mean(distances))
        return violations

    def _drivable_distance_map(self, drivable_map: np.ndarray) -> np.ndarray:
        drivable = np.asarray(drivable_map, dtype=bool)
        if drivable.size == 0:
            return np.zeros_like(drivable, dtype=float)
        if not np.any(drivable):
            return np.full(
                drivable.shape,
                float(self.config.drivable_distance_clip),
                dtype=float,
            )

        distance_pixels = distance_transform_edt(~drivable)
        distance_meters = distance_pixels * float(self.config.pixel_size)
        return np.minimum(distance_meters, float(self.config.drivable_distance_clip))

    def _collision_violations(
        self,
        batch: BitsBatch,
        plan: BitsPlan,
        agent_prediction: Optional[BitsAgentPrediction] = None,
    ) -> np.ndarray:
        violations = np.zeros(plan.positions.shape[0], dtype=float)
        agent_positions, agent_yaws, agent_availability = self._agent_prediction_arrays(
            batch,
            plan,
            agent_prediction,
        )
        if agent_availability.size == 0 or not np.any(agent_availability):
            return violations

        for index in range(plan.positions.shape[0]):
            available = np.flatnonzero(plan.availabilities[index])
            if available.size == 0:
                continue
            collision_steps = 0
            mode_index = index if agent_positions.shape[0] > 1 else 0
            for step in available:
                if step >= agent_availability.shape[2]:
                    continue
                # Score each candidate in the ego frame, using predicted neighbors
                # for closed-loop collision filtering when available.
                ego_box = oriented_box(
                    plan.positions[index, step, 0],
                    plan.positions[index, step, 1],
                    float(plan.yaws[index, step, 0]),
                    batch.extent[0],
                    batch.extent[1],
                )
                if self._has_collision_at_step(
                    ego_box,
                    agent_positions[mode_index],
                    agent_yaws[mode_index],
                    agent_availability[mode_index],
                    batch.all_other_agents_extents,
                    int(step),
                ):
                    collision_steps += 1
            violations[index] = collision_steps / float(available.size)
        return violations

    @staticmethod
    def _agent_prediction_arrays(
        batch: BitsBatch,
        plan: BitsPlan,
        agent_prediction: Optional[BitsAgentPrediction],
    ) -> tuple:
        if agent_prediction is None:
            positions = np.asarray(batch.all_other_agents_future_positions, dtype=float)[None]
            yaws = np.asarray(batch.all_other_agents_future_yaws, dtype=float)[None]
            availabilities = np.asarray(batch.all_other_agents_future_availability, dtype=bool)[None]
        else:
            positions = np.asarray(agent_prediction.positions, dtype=float)
            yaws = np.asarray(agent_prediction.yaws, dtype=float)
            availabilities = np.asarray(agent_prediction.availabilities, dtype=bool)
            if positions.ndim == 3:
                positions = positions[None]
            if yaws.ndim == 3:
                yaws = yaws[None]
            if availabilities.ndim == 2:
                availabilities = availabilities[None]

        mode_count = plan.positions.shape[0]
        if positions.shape[0] not in (1, mode_count):
            raise ValueError("agent_prediction mode count must be 1 or match plan candidates.")
        return positions, yaws, availabilities

    @staticmethod
    def _has_collision_at_step(
        ego_box,
        agent_positions: np.ndarray,
        agent_yaws: np.ndarray,
        agent_availability: np.ndarray,
        agent_extents: np.ndarray,
        step: int,
    ) -> bool:
        for agent_index in range(agent_positions.shape[0]):
            if not bool(agent_availability[agent_index, step]):
                continue
            extent = agent_extents[agent_index]
            if not np.any(extent):
                continue
            other_box = oriented_box(
                agent_positions[agent_index, step, 0],
                agent_positions[agent_index, step, 1],
                float(agent_yaws[agent_index, step, 0]),
                extent[0],
                extent[1],
            )
            if ego_box.intersects(other_box):
                return True
        return False

    @staticmethod
    def _transform_point(point, transform: np.ndarray) -> np.ndarray:
        transformed = transform @ np.asarray([point[0], point[1], 1.0], dtype=float)
        return transformed[:2]


class BitsBehaviorModel:
    """Public BITS-style behavior model entry point."""

    def __init__(
        self,
        config: Optional[BitsConfig] = None,
        policy: Optional[BitsPolicy] = None,
        builder: Optional[BitsBatchBuilder] = None,
        include_raster: bool = True,
    ):
        self.config = config or BitsConfig()
        if policy is None:
            raise ValueError(
                "policy is required; use TorchBitsPolicy with BitsBiLevelTorchModel "
                "for the BITS reproduction path."
            )
        self.policy = policy
        self.builder = builder or BitsBatchBuilder(self.config)
        self.include_raster = include_raster

    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> Dict[object, Trajectory]:
        """Predict future trajectories for selected agents."""

        selected_ids = list(participants.keys()) if agent_ids is None else list(agent_ids)
        trajectories = {}
        for ego_id in selected_ids:
            if ego_id not in participants or not participants[ego_id].trajectory.has_state(frame):
                continue

            # Prediction has three stages: build a BITS batch, run the bi-level
            # policy in ego frame, then convert results back to world trajectories.
            batch = self.builder.build(
                participants=participants,
                frame=frame,
                ego_id=ego_id,
                map_=map_,
                include_raster=self.include_raster,
            )
            prediction = self.policy.predict_batch(batch)
            trajectories[ego_id] = self._prediction_to_trajectory(ego_id, frame, batch, prediction)
        return trajectories

    def predict_batch(self, batch: BitsBatch) -> BitsPrediction:
        """Expose the policy-level batch API for training/evaluation code."""

        return self.policy.predict_batch(batch)

    def _prediction_to_trajectory(
        self,
        ego_id: object,
        frame: int,
        batch: BitsBatch,
        prediction: BitsPrediction,
    ) -> Trajectory:
        best_index = int(np.argmax(prediction.scores))
        trajectory = Trajectory(
            id_=ego_id,
            fps=round(1.0 / self.config.dt, 3),
            stable_freq=True,
        )
        for step in range(self.config.future_steps):
            if not bool(prediction.availabilities[best_index, step]):
                continue
            local_position = prediction.positions[best_index, step]
            local_yaw = float(prediction.yaws[best_index, step, 0])
            world_position = self._transform_point(local_position, batch.world_from_agent)
            world_yaw = normalize_angle(batch.yaw + local_yaw)
            state_frame = frame + self.config.step_ms * (step + 1)
            trajectory.add_state(
                State(
                    frame=state_frame,
                    x=float(world_position[0]),
                    y=float(world_position[1]),
                    heading=world_yaw,
                    vx=0.0,
                    vy=0.0,
                )
            )
        return trajectory

    @staticmethod
    def _transform_point(point, transform: np.ndarray) -> np.ndarray:
        transformed = transform @ np.asarray([point[0], point[1], 1.0], dtype=float)
        return transformed[:2]
