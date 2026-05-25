# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""RoadSegmentGenerator implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .road_elements.fork import fork as _fork
from .road_elements.intersection import intersection as _intersection
from .road_elements.lane_adapter import lane_adapter as _lane_adapter
from .road_elements.merge import merge as _merge
from .road_elements.one_way import one_way as _one_way
from .road_elements.ramp import entrance_ramp as _entrance_ramp
from .road_elements.ramp import exit_ramp as _exit_ramp
from .road_elements.ramp import freeway_entrance_ramp as _freeway_entrance_ramp
from .road_elements.ramp import freeway_exit_ramp as _freeway_exit_ramp
from .road_elements.ramp import urban_entrance_ramp as _urban_entrance_ramp
from .road_elements.ramp import urban_exit_ramp as _urban_exit_ramp
from .road_elements.roundabout import roundabout as _roundabout
from .road_elements.two_way import two_way as _two_way
from .rules.lane_marking_rules import get_standard, set_standard
from .rules.module_types import RoadModuleResult, RoadPort


class RoadSegmentGenerator:
    """Procedural road-segment factory with auto-chained element IDs.

    Wraps all road-element generators into a single stateful interface.
    After each call the internal id counter is updated to the result's
    ``id_counter``, so consecutive calls automatically produce non-colliding
    element ids without manual chaining.

    Example::

        gen = RoadSegmentGenerator(standard="MUTCD")
        road = gen.two_way(start_port, end_port, forward_lane_num=2)
        adapter = gen.lane_adapter(road.ports["forward_out"], next_port)
        junction = gen.intersection(center, arms)

    Attributes:
        id_counter: Next element id that will be assigned to the following
            generator call (read-only; updated automatically after each call).
        standard: Active lane-marking standard (``"MUTCD"`` or ``"GB"``).
            Changing this property calls :func:`set_standard` globally.
    """

    def __init__(self, standard: str = "MUTCD", id_offset: int = 0) -> None:
        """Initialise the generator.

        Args:
            standard: Initial lane-marking standard. Must be ``"MUTCD"`` or
                ``"GB"``.
            id_offset: Starting value for the element id counter.
        """
        set_standard(standard)
        self._id_counter: int = id_offset

    @property
    def id_counter(self) -> int:
        """Next available element id."""
        return self._id_counter

    @property
    def standard(self) -> str:
        """Active lane-marking standard."""
        return get_standard()

    @standard.setter
    def standard(self, value: str) -> None:
        set_standard(value)

    def reset(self, id_offset: int = 0) -> None:
        """Reset the id counter.

        Args:
            id_offset: New starting value for the counter. Defaults to 0.
        """
        self._id_counter = id_offset

    def _run(self, fn, *args, **kwargs) -> RoadModuleResult:
        """Call a generator function with auto-chained id_offset.

        Args:
            fn: Underlying generator function to call.
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``. If ``id_offset``
                is not present it defaults to the current :attr:`id_counter`.

        Returns:
            The :class:`RoadModuleResult` returned by ``fn``.
        """
        kwargs.setdefault("id_offset", self._id_counter)
        result = fn(*args, **kwargs)
        self._id_counter = result.id_counter
        return result

    def one_way(
        self,
        start_port: RoadPort,
        end_port: RoadPort,
        *,
        lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        step_size: float = 0.1,
    ) -> RoadModuleResult:
        """Generate a one-way road segment between two ports.

        Args:
            start_port: Upstream socket defining entry geometry and lane metadata.
            end_port: Downstream socket defining exit geometry.
            lane_num: Lane count. Defaults to ``start_port.lane_num``.
            lane_width: Lane width in metres. Defaults to
                ``start_port.lane_width``.
            speed_limit: Speed limit in km/h. Defaults to
                ``start_port.speed_limit``.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"forward_in"`` and
            ``"forward_out"``.
        """
        return self._run(
            _one_way,
            start_port,
            end_port,
            lane_num=lane_num,
            lane_width=lane_width,
            speed_limit=speed_limit,
            step_size=step_size,
        )

    def two_way(
        self,
        start_port: RoadPort,
        end_port: RoadPort,
        *,
        forward_lane_num: int | None = None,
        backward_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        step_size: float = 0.1,
    ) -> RoadModuleResult:
        """Generate a two-way road segment between two ports.

        Args:
            start_port: Forward-direction start socket.
            end_port: Forward-direction end socket.
            forward_lane_num: Forward lane count. Defaults to
                ``start_port.lane_num``.
            backward_lane_num: Backward lane count. Defaults to
                ``start_port.lane_num``.
            lane_width: Lane width in metres. Defaults to
                ``start_port.lane_width``.
            speed_limit: Speed limit in km/h. Defaults to
                ``start_port.speed_limit``.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with directional road ports.
        """
        return self._run(
            _two_way,
            start_port,
            end_port,
            forward_lane_num=forward_lane_num,
            backward_lane_num=backward_lane_num,
            lane_width=lane_width,
            speed_limit=speed_limit,
            step_size=step_size,
        )

    def lane_adapter(
        self,
        start_port: RoadPort,
        end_port: RoadPort,
        *,
        start_lane_num: int | None = None,
        end_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        change_side: str = "right",
        step_size: float = 0.1,
    ) -> RoadModuleResult:
        """Generate a lane-count adapter between two ports (±1 lane).

        Args:
            start_port: Upstream socket.
            end_port: Downstream socket.
            start_lane_num: Entry lane count. Defaults to
                ``start_port.lane_num``.
            end_lane_num: Exit lane count. Defaults to ``end_port.lane_num``.
            lane_width: Lane width in metres. Defaults to
                ``start_port.lane_width``.
            speed_limit: Speed limit in km/h. Defaults to
                ``start_port.speed_limit``.
            change_side: Side where the lane is added or dropped.
                ``"right"`` or ``"left"``.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"entry"`` and ``"exit"``.

        Raises:
            ValueError: If lane counts differ by more than 1 or any parameter
                is invalid.
        """
        return self._run(
            _lane_adapter,
            start_port,
            end_port,
            start_lane_num=start_lane_num,
            end_lane_num=end_lane_num,
            lane_width=lane_width,
            speed_limit=speed_limit,
            change_side=change_side,
            step_size=step_size,
        )

    def fork(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        branch_out: RoadPort,
        *,
        fork_side: str = "right",
        main_lane_num: int | None = None,
        branch_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        taper_length: float = 65.0,
        branch_length: float = 85.0,
        diverge_s_ratio: float | None = None,
        step_size: float = 0.1,
    ) -> RoadModuleResult:
        """Generate a lane-level fork where traffic diverges from the main road.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            branch_out: Downstream branch socket.
            fork_side: Side where the branch leaves (``"left"`` or
                ``"right"``).
            main_lane_num: Main lane count. Defaults to ``main_in.lane_num``.
            branch_lane_num: Branch lane count. Defaults to
                ``branch_out.lane_num``.
            lane_width: Lane width in metres. Defaults to
                ``main_in.lane_width``.
            speed_limit: Main-road speed limit in km/h. Defaults to
                ``main_in.speed_limit``.
            taper_length: Minimum distance reserved for the fork opening.
            branch_length: Nominal branch length for diverge point inference.
            diverge_s_ratio: Optional normalised diverge position override.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``,
            ``"main_out"``, and ``"branch_out"``.

        Raises:
            ValueError: If ``fork_side`` is invalid or ``branch_lane_num``
                exceeds ``main_lane_num``.
        """
        return self._run(
            _fork,
            main_in,
            main_out,
            branch_out,
            fork_side=fork_side,
            main_lane_num=main_lane_num,
            branch_lane_num=branch_lane_num,
            lane_width=lane_width,
            speed_limit=speed_limit,
            taper_length=taper_length,
            branch_length=branch_length,
            diverge_s_ratio=diverge_s_ratio,
            step_size=step_size,
        )

    def merge(
        self,
        main_in: RoadPort,
        branch_in: RoadPort,
        main_out: RoadPort,
        *,
        merge_side: str = "right",
        main_lane_num: int | None = None,
        branch_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        taper_length: float = 65.0,
        branch_length: float = 85.0,
        merge_s_ratio: float | None = None,
        step_size: float = 0.1,
    ) -> RoadModuleResult:
        """Generate a lane-level merge where traffic converges onto the main road.

        Args:
            main_in: Upstream main-road socket.
            branch_in: Upstream branch socket.
            main_out: Downstream main-road socket.
            merge_side: Side where the branch joins (``"left"`` or
                ``"right"``).
            main_lane_num: Main lane count. Defaults to ``main_in.lane_num``.
            branch_lane_num: Branch lane count. Defaults to
                ``branch_in.lane_num``.
            lane_width: Lane width in metres. Defaults to
                ``main_in.lane_width``.
            speed_limit: Main-road speed limit in km/h. Defaults to
                ``main_in.speed_limit``.
            taper_length: Minimum distance reserved for the merge opening.
            branch_length: Nominal branch length for merge point inference.
            merge_s_ratio: Optional normalised merge position override.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``,
            ``"branch_in"``, and ``"main_out"``.

        Raises:
            ValueError: If ``merge_side`` is invalid or ``branch_lane_num``
                exceeds ``main_lane_num``.
        """
        return self._run(
            _merge,
            main_in,
            branch_in,
            main_out,
            merge_side=merge_side,
            main_lane_num=main_lane_num,
            branch_lane_num=branch_lane_num,
            lane_width=lane_width,
            speed_limit=speed_limit,
            taper_length=taper_length,
            branch_length=branch_length,
            merge_s_ratio=merge_s_ratio,
            step_size=step_size,
        )

    def exit_ramp(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        ramp_out: RoadPort,
        *,
        main_road_type: str = "freeway",
        ramp_side: str = "right",
        backward_lane_num: int = 1,
        lane_width: float | None = None,
        main_speed_limit: float | None = None,
        ramp_speed_limit: float | None = None,
        taper_length: float = 50.0,
        parallel_length: float = 60.0,
        step_size: float = 0.5,
    ) -> RoadModuleResult:
        """Generate an exit ramp where traffic peels off the main road.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            ramp_out: Downstream ramp socket where exiting traffic departs.
            main_road_type: ``"freeway"`` for a one-way main road; ``"urban"``
                for a two-way road.
            ramp_side: Side of the main road where the ramp exits.
                Only ``"right"`` is supported for ``"urban"`` type.
            backward_lane_num: Backward lane count; ignored for ``"freeway"``.
            lane_width: Lane width in metres. Defaults to
                ``main_in.lane_width``.
            main_speed_limit: Main-road speed limit in km/h.
            ramp_speed_limit: Ramp speed limit in km/h.
            taper_length: Longitudinal length of the auxiliary lane taper.
            parallel_length: Longitudinal length of the full-width auxiliary
                lane running parallel to the main road.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``,
            ``"main_out"``, and ``"ramp"``.

        Raises:
            ValueError: If ``ramp_side="left"`` with ``main_road_type="urban"``.
        """
        return self._run(
            _exit_ramp,
            main_in,
            main_out,
            ramp_out,
            main_road_type=main_road_type,
            ramp_side=ramp_side,
            backward_lane_num=backward_lane_num,
            lane_width=lane_width,
            main_speed_limit=main_speed_limit,
            ramp_speed_limit=ramp_speed_limit,
            taper_length=taper_length,
            parallel_length=parallel_length,
            step_size=step_size,
        )

    def entrance_ramp(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        ramp_in: RoadPort,
        *,
        main_road_type: str = "freeway",
        ramp_side: str = "right",
        backward_lane_num: int = 1,
        lane_width: float | None = None,
        main_speed_limit: float | None = None,
        ramp_speed_limit: float | None = None,
        taper_length: float = 50.0,
        parallel_length: float = 60.0,
        step_size: float = 0.5,
    ) -> RoadModuleResult:
        """Generate an entrance ramp where traffic merges onto the main road.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            ramp_in: Upstream ramp socket where entering traffic arrives.
            main_road_type: ``"freeway"`` or ``"urban"``.
            ramp_side: Side of the main road where the ramp joins.
            backward_lane_num: Backward lane count; ignored for ``"freeway"``.
            lane_width: Lane width in metres. Defaults to
                ``main_in.lane_width``.
            main_speed_limit: Main-road speed limit in km/h.
            ramp_speed_limit: Ramp speed limit in km/h.
            taper_length: Longitudinal length of the auxiliary lane taper.
            parallel_length: Longitudinal length of the full-width auxiliary
                lane running parallel to the main road.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``,
            ``"main_out"``, and ``"ramp"``.

        Raises:
            ValueError: If ``ramp_side="left"`` with ``main_road_type="urban"``.
        """
        return self._run(
            _entrance_ramp,
            main_in,
            main_out,
            ramp_in,
            main_road_type=main_road_type,
            ramp_side=ramp_side,
            backward_lane_num=backward_lane_num,
            lane_width=lane_width,
            main_speed_limit=main_speed_limit,
            ramp_speed_limit=ramp_speed_limit,
            taper_length=taper_length,
            parallel_length=parallel_length,
            step_size=step_size,
        )

    def freeway_exit_ramp(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        ramp_out: RoadPort,
        *,
        ramp_side: str = "right",
        lane_width: float | None = None,
        main_speed_limit: float | None = None,
        ramp_speed_limit: float | None = None,
        taper_length: float = 50.0,
        parallel_length: float = 60.0,
        step_size: float = 0.5,
    ) -> RoadModuleResult:
        """Generate a freeway exit ramp (one-way main road).

        Convenience alias for :meth:`exit_ramp` with
        ``main_road_type="freeway"``.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            ramp_out: Downstream ramp socket.
            ramp_side: ``"right"`` or ``"left"``.
            lane_width: Lane width in metres.
            main_speed_limit: Main-road speed limit in km/h.
            ramp_speed_limit: Ramp speed limit in km/h.
            taper_length: Auxiliary lane taper length in metres.
            parallel_length: Auxiliary lane parallel length in metres.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``,
            ``"main_out"``, and ``"ramp"``.
        """
        return self._run(
            _freeway_exit_ramp,
            main_in,
            main_out,
            ramp_out,
            ramp_side=ramp_side,
            lane_width=lane_width,
            main_speed_limit=main_speed_limit,
            ramp_speed_limit=ramp_speed_limit,
            taper_length=taper_length,
            parallel_length=parallel_length,
            step_size=step_size,
        )

    def freeway_entrance_ramp(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        ramp_in: RoadPort,
        *,
        ramp_side: str = "right",
        lane_width: float | None = None,
        main_speed_limit: float | None = None,
        ramp_speed_limit: float | None = None,
        taper_length: float = 50.0,
        parallel_length: float = 60.0,
        step_size: float = 0.5,
    ) -> RoadModuleResult:
        """Generate a freeway entrance ramp (one-way main road).

        Convenience alias for :meth:`entrance_ramp` with
        ``main_road_type="freeway"``.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            ramp_in: Upstream ramp socket.
            ramp_side: ``"right"`` or ``"left"``.
            lane_width: Lane width in metres.
            main_speed_limit: Main-road speed limit in km/h.
            ramp_speed_limit: Ramp speed limit in km/h.
            taper_length: Auxiliary lane taper length in metres.
            parallel_length: Auxiliary lane parallel length in metres.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``,
            ``"main_out"``, and ``"ramp"``.
        """
        return self._run(
            _freeway_entrance_ramp,
            main_in,
            main_out,
            ramp_in,
            ramp_side=ramp_side,
            lane_width=lane_width,
            main_speed_limit=main_speed_limit,
            ramp_speed_limit=ramp_speed_limit,
            taper_length=taper_length,
            parallel_length=parallel_length,
            step_size=step_size,
        )

    def urban_exit_ramp(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        ramp_out: RoadPort,
        *,
        backward_lane_num: int = 1,
        lane_width: float | None = None,
        main_speed_limit: float | None = None,
        ramp_speed_limit: float | None = None,
        taper_length: float = 50.0,
        parallel_length: float = 60.0,
        step_size: float = 0.5,
    ) -> RoadModuleResult:
        """Generate an urban exit ramp (two-way main road, right side only).

        Convenience alias for :meth:`exit_ramp` with
        ``main_road_type="urban"`` and ``ramp_side="right"``.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            ramp_out: Downstream ramp socket.
            backward_lane_num: Number of backward lanes on the main road.
            lane_width: Lane width in metres.
            main_speed_limit: Main-road speed limit in km/h.
            ramp_speed_limit: Ramp speed limit in km/h.
            taper_length: Auxiliary lane taper length in metres.
            parallel_length: Auxiliary lane parallel length in metres.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``,
            ``"main_out"``, ``"ramp"``, ``"backward_in"``, and
            ``"backward_out"``.
        """
        return self._run(
            _urban_exit_ramp,
            main_in,
            main_out,
            ramp_out,
            backward_lane_num=backward_lane_num,
            lane_width=lane_width,
            main_speed_limit=main_speed_limit,
            ramp_speed_limit=ramp_speed_limit,
            taper_length=taper_length,
            parallel_length=parallel_length,
            step_size=step_size,
        )

    def urban_entrance_ramp(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        ramp_in: RoadPort,
        *,
        backward_lane_num: int = 1,
        lane_width: float | None = None,
        main_speed_limit: float | None = None,
        ramp_speed_limit: float | None = None,
        taper_length: float = 50.0,
        parallel_length: float = 60.0,
        step_size: float = 0.5,
    ) -> RoadModuleResult:
        """Generate an urban entrance ramp (two-way main road, right side only).

        Convenience alias for :meth:`entrance_ramp` with
        ``main_road_type="urban"`` and ``ramp_side="right"``.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            ramp_in: Upstream ramp socket.
            backward_lane_num: Number of backward lanes on the main road.
            lane_width: Lane width in metres.
            main_speed_limit: Main-road speed limit in km/h.
            ramp_speed_limit: Ramp speed limit in km/h.
            taper_length: Auxiliary lane taper length in metres.
            parallel_length: Auxiliary lane parallel length in metres.
            step_size: Reference-line sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``,
            ``"main_out"``, ``"ramp"``, ``"backward_in"``, and
            ``"backward_out"``.
        """
        return self._run(
            _urban_entrance_ramp,
            main_in,
            main_out,
            ramp_in,
            backward_lane_num=backward_lane_num,
            lane_width=lane_width,
            main_speed_limit=main_speed_limit,
            ramp_speed_limit=ramp_speed_limit,
            taper_length=taper_length,
            parallel_length=parallel_length,
            step_size=step_size,
        )

    def intersection(
        self,
        center: np.ndarray,
        arms: list[dict[str, Any] | RoadPort],
        *,
        lane_width: float = 3.5,
        radius: float = 8.0,
        speed_limit: float = 30.0,
        step_size: float = 0.1,
    ) -> RoadModuleResult:
        """Generate a T-junction or cross-junction.

        Args:
            center: 2-D world coordinate of the junction centre.
            arms: List of 3 or 4 arm descriptors. Each element is either a
                :class:`RoadPort` or a plain ``dict`` with at least a
                ``"heading"`` key and optionally ``"lane_num"``,
                ``"lane_width"``, ``"speed_limit"``, ``"curvature"``,
                ``"radius"``.
            lane_width: Default lane width in metres.
            radius: Default arm boundary radius for dict-style arms.
            speed_limit: Default speed limit in km/h.
            step_size: Arc sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with one named port per arm and a
            single :class:`~tactics2d.map.element.Junction`.

        Raises:
            ValueError: If the number of arms is not 3 or 4.
        """
        return self._run(
            _intersection,
            center,
            arms,
            lane_width=lane_width,
            radius=radius,
            speed_limit=speed_limit,
            step_size=step_size,
        )

    def roundabout(
        self,
        center: np.ndarray,
        arms: list[dict[str, Any] | RoadPort],
        *,
        ring_radius: float = 15.0,
        ring_lane_num: int = 1,
        lane_width: float = 3.5,
        speed_limit: float = 20.0,
        step_size: float = 0.1,
    ) -> RoadModuleResult:
        """Generate a roundabout with a circulating ring road.

        Args:
            center: 2-D world coordinate of the roundabout centre.
            arms: List of arm descriptors. Each element is a :class:`RoadPort`
                or a ``dict`` with ``"heading"`` and optionally ``"lane_num"``,
                ``"lane_width"``, ``"speed_limit"``.
            ring_radius: Radius of the inner edge of the ring road in metres.
            ring_lane_num: Number of circulating lanes in the ring.
            lane_width: Lane width in metres (ring and arm stubs).
            speed_limit: Speed limit in km/h (ring and arm stubs).
            step_size: Arc sampling interval in metres.

        Returns:
            :class:`RoadModuleResult` with one named port per arm and a
            single :class:`~tactics2d.map.element.Junction`.
        """
        return self._run(
            _roundabout,
            center,
            arms,
            ring_radius=ring_radius,
            ring_lane_num=ring_lane_num,
            lane_width=lane_width,
            speed_limit=speed_limit,
            step_size=step_size,
        )
