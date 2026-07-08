# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared behavior model interfaces."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional

from tactics2d.map.element import Map
from tactics2d.participant.trajectory import Trajectory


class BehaviorModelBase(ABC):
    """Base interface for Tactics2D behavior models.

    Attributes:
        parallel_workers (int): Default thread count for :meth:`predict_batch`.
            0 disables parallelism. Subclasses may override this in ``__init__``.
    """

    parallel_workers: int = 0

    @abstractmethod
    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> Dict[object, Trajectory]:
        """Plan future trajectories for the specified traffic participants."""

    def predict_batch(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frames: List[int],
        agent_ids: Optional[Iterable[object]] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[int, Dict[object, Trajectory]]:
        """Predict trajectories for multiple frames, optionally in parallel.

        When *max_workers* > 1, frames are dispatched across a
        ``ThreadPoolExecutor``.  Subclasses whose :meth:`predict` mutates
        shared instance state should override this method or ensure
        thread-safety.

        Args:
            participants: Traffic participants keyed by agent id.
            map_: Semantic map, or ``None``.
            frames: Frame timestamps to predict for, in any order.
            agent_ids: Optional explicit ids to control. Forwarded to
                :meth:`predict`.
            max_workers: Maximum thread count. Defaults to
                :attr:`parallel_workers`.  0 or 1 runs sequentially.

        Returns:
            ``{frame: {agent_id: Trajectory}}`` — the same structure the
            notebook ``compute_predictions`` helper used to produce.
        """
        workers = max_workers if max_workers is not None else self.parallel_workers
        if workers <= 1:
            result = {}
            for f in frames:
                try:
                    result[f] = self.predict(participants, map_, f, agent_ids)
                except Exception:
                    result[f] = {}
            return result

        result = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_frame = {
                executor.submit(self.predict, participants, map_, f, agent_ids): f for f in frames
            }
            for future in as_completed(future_to_frame):
                f = future_to_frame[future]
                try:
                    result[f] = future.result()
                except Exception:
                    result[f] = {}
        return result
