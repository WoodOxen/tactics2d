# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Junction implementation."""

import logging
from typing import Optional


class Junction:
    """This class implements a junction.

    The junction absorbs the connection semantics previously held by a separate
    Connection class.  OpenDRIVE-specific fields (``incoming_road``,
    ``connecting_road``, ``contact_point``, ``lane_links``) are available
    directly on every Junction instance and default to ``None`` / empty so
    that junctions without explicit connection data (e.g. SUMO junctions that
    only carry geometric shape information) can still be constructed without
    supplying those fields.

    Multiple connections belonging to the same physical junction are stored in
    the ``connections`` dictionary, where each value is also a ``Junction``
    instance carrying the per-connection attributes.

    !!! quote "Reference"
        [OpenDRIVE's description of a junction](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/12_junctions/12_01_introduction.html)
        [SUMO Road Networks](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html)

    Attributes:
        id_ (str): The unique identifier of the junction.
        incoming_road (str, optional): The id of the incoming road. Used in
            OpenDRIVE connection semantics. Defaults to None.
        connecting_road (str, optional): The id of the connecting road. Used in
            OpenDRIVE connection semantics. Defaults to None.
        contact_point (str): The contact point of the connection. Defaults to
            ``"start"``.
        lane_links (list): The lane links of the connection. Each element is a
            tuple of (from_lane_id, to_lane_id). Defaults to [].
        connections (dict): Child connections of the junction, keyed by their
            ``id_``. Defaults to {}.
        custom_tags (dict): Format-specific metadata. For SUMO junctions this
            stores keys ``sumo_id``, ``x``, ``y``, ``type``, and ``shape``.
            Defaults to {}.
    """

    __slots__ = (
        "id_",
        "incoming_road",
        "connecting_road",
        "contact_point",
        "lane_links",
        "connections",
        "custom_tags",
    )

    def __init__(
        self,
        id_: str,
        incoming_road: Optional[str] = None,
        connecting_road: Optional[str] = None,
        contact_point: str = "start",
        lane_links: Optional[list] = None,
        connections: Optional[dict] = None,
        custom_tags: Optional[dict] = None,
    ):
        """Initialize the junction.

        Args:
            id_ (str): The unique identifier of the junction.
            incoming_road (str, optional): The id of the incoming road.
                Defaults to None.
            connecting_road (str, optional): The id of the connecting road.
                Defaults to None.
            contact_point (str, optional): The contact point of the connection.
                Defaults to ``"start"``.
            lane_links (list, optional): The lane links of the connection.
                Defaults to an empty list.
            connections (dict, optional): Child connections of the junction.
                Defaults to an empty dict.
            custom_tags (dict, optional): Format-specific metadata.
                Defaults to an empty dict.
        """
        self.id_ = id_
        self.incoming_road = incoming_road
        self.connecting_road = connecting_road
        self.contact_point = contact_point
        self.lane_links = lane_links if lane_links is not None else []
        self.connections = connections if connections is not None else {}
        self.custom_tags = custom_tags if custom_tags is not None else {}

    def add_lane_link(self, lane_link: tuple):
        """Add a lane link to the junction.

        Args:
            lane_link (tuple): The lane link to be added. Shape is (2,).
                The first element is the id of the lane in the incoming road
                and the second element is the id of the lane in the connecting
                road.
        """
        self.lane_links.append(lane_link)

    def add_connection(self, connection: "Junction"):
        """Add a child connection to the junction.

        Args:
            connection (Junction): The connection to be added. Each connection
                is itself a Junction instance carrying the per-connection
                attributes (``incoming_road``, ``connecting_road``, etc.).
        """
        if connection.id_ in self.connections:
            logging.warning(
                "Connection %s already exists in junction %s. "
                "Overwriting the existing connection.",
                connection.id_, self.id_,
            )
        self.connections[connection.id_] = connection