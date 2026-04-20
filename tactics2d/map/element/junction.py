# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Junction implementation."""

import logging
from typing import Optional


class Junction:
    """This class implements a junction.

    !!! quote "Reference"
        [OpenDRIVE's description of a junction](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/12_junctions/12_01_introduction.html)
        [SUMO Road Networks](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html)

    Attributes:
        id_ (str): The unique identifier of the junction.
        connections (dict): The connections of the junction. Defaults to {}.
        custom_tags (dict): Format-specific metadata. For SUMO junctions, stores
            keys: sumo_id, x, y, type, shape. Defaults to {}.
    """

    class Connection:
        """This class implements a connection between roads at a junction.

        Supports both OpenDRIVE and SUMO connection semantics. OpenDRIVE-specific
        fields (incoming_road, connecting_road, contact_point, lane_links) are used
        when parsing .xodr files. SUMO-specific fields and any other format-specific
        data are stored in custom_tags.

        !!! quote "Reference"
            [OpenDRIVE's description of a connection](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/12_junctions/12_01_introduction.html)
            [SUMO Road Networks](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html)

        Attributes:
            id_ (str): The unique identifier of the connection.
            incoming_road (str, optional): The id of the incoming road. Used in OpenDRIVE.
            connecting_road (str, optional): The id of the connecting road. Used in OpenDRIVE.
            contact_point (str): The contact point of the connection. Defaults to "start".
            lane_links (list): The lane links of the connection. Each element is a
                tuple of (from_lane_id, to_lane_id). Defaults to [].
            custom_tags (dict): Format-specific metadata. Defaults to {}.
        """

        __slots__ = (
            "id_",
            "incoming_road",
            "connecting_road",
            "contact_point",
            "lane_links",
            "custom_tags",
        )

        def __init__(
            self,
            id_: str,
            incoming_road: Optional[str] = None,
            connecting_road: Optional[str] = None,
            contact_point: str = "start",
            lane_links: Optional[list] = None,
            custom_tags: Optional[dict] = None,
        ):
            self.id_ = id_
            self.incoming_road = incoming_road
            self.connecting_road = connecting_road
            self.contact_point = contact_point
            self.lane_links = lane_links if lane_links is not None else []
            self.custom_tags = custom_tags if custom_tags is not None else {}

        def add_lane_link(self, lane_link: tuple):
            """Add a lane link to the connection.

            Args:
                lane_link (tuple): The lane link to be added. Shape is (2,).
                    The first element is the id of the source lane and the second
                    element is the id of the destination lane.
            """
            self.lane_links.append(lane_link)

    __slots__ = ("id_", "connections", "custom_tags")

    def __init__(
        self,
        id_: str,
        connections: Optional[dict] = None,
        custom_tags: Optional[dict] = None,
    ):
        """Initialize the junction.

        Args:
            id_ (str): The unique identifier of the junction.
            connections (dict, optional): The connections of the junction.
                Defaults to an empty dict.
            custom_tags (dict, optional): Format-specific metadata.
                Defaults to an empty dict.
        """
        self.id_ = id_
        self.connections = connections if connections is not None else {}
        self.custom_tags = custom_tags if custom_tags is not None else {}

    def add_connection(self, connection: "Junction.Connection"):
        """Add a connection to the junction.

        Args:
            connection (Junction.Connection): The connection to be added.
        """
        if connection.id_ in self.connections:
            logging.warning(
                f"Connection {connection.id_} already exists in junction "
                f"{self.id_}. Overwriting the existing connection."
            )
        self.connections[connection.id_] = connection