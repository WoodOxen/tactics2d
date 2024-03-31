##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: junction.py
# @Description: This file defines a class for a junction.
# @Author: Yueyuan Li
# @Version: 1.0.0


import logging


class Connection:
    """This class implements a connection.

    !!! quote "Reference"
        [OpenDRIVE's description of a connection](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/12_junctions/12_01_introduction.html)

    Attributes:
        id_ (str): The unique identifier of the connect. The id should be unique within the junction.
        incoming_road (str): The id of the incoming road.
        connecting_road (str): The id of the connecting road.
        contact_point (str): The contact point of the junction. Defaults to "start".
        lane_links (list): The lane links of the junction. Defaults to [].
    """

    def __init__(
        self,
        id_: str,
        incoming_road: str,
        connecting_road: str,
        contact_point: str = "start",
        lane_links: list = [],
    ):
        """Initialize the connection.

        Args:
            id_ (str): The unique identifier of the connect. The id should be unique within the junction.
            incoming_road (str): The id of the incoming road.
            connecting_road (str): The id of the connecting road.
            contact_point (str, optional): The contact point of the junction.
            lane_links (list, optional): The lane links of the junction.
        """
        self.id_ = id_
        self.incoming_road = incoming_road
        self.connecting_road = connecting_road
        self.contact_point = contact_point
        self.lane_links = lane_links

    def add_lane_link(self, lane_link: tuple):
        """Add a lane link to the junction.

        Args:
            lane_link (tuple): The lane link to be added. The shape is (2,). The first element is the id of the lane in the incoming road, and the second element is the id of the lane in the connecting road.
        """
        self.lane_links.append(lane_link)


class Junction:
    """This class implements a junction.

    !!! quote "Reference"
        [OpenDRIVE's description of a junction](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/12_junctions/12_01_introduction.html)

    Attributes:
        id_ (str): The unique identifier of the junction.
        connections (dict): The connections of the junction. Defaults to an empty dictionary.
    """

    def __init__(self, id_: str, connections: dict = {}):
        """Initialize the junction.

        Args:
            id_ (str): The unique identifier of the junction.
            connections (dict, optional): The connections of the junction. Defaults to an empty dictionary.
        """
        self.id_ = id_
        self.connections = connections

    def add_connection(self, connection: Connection):
        """Add a connection to the junction.

        Args:
            connection (Connection): The connection to be added.
        """
        if connection.id_ in self.connections:
            logging.warning(
                f"Connection {connection.id_} already exists in junction {self.id_}. Overwrite the existing connection."
            )

        self.connections[connection.id_] = connection
