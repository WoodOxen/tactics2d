##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: regulatory.py
# @Description: This file defines a class for a regulatory element.
# @Author: Yueyuan Li
# @Version: 1.0.0


class RegulatoryMember:
    """This class implements the subelement of the Regulatory class."""

    def __init__(self, ref: str, type_: str, role: str):
        self.ref = ref
        self.type_ = type_
        self.role = role


class Regulatory:
    """This class implements the [lanelet2-style map element *RegulatoryElement*](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/RegulatoryElementTagging.md).

    !!! note
        This class is still under development. It is supposed to support the detection of traffic events in the future.

    Attributes:
        id_ (str): The id of the regulatory element.
        relations (dict): A dictionary of the relations that the regulatory element belongs to. The key is the id of the relation, and the value is the role of the relation. Defaults to an empty dictionary.
        ways (dict): A dictionary of the ways that the regulatory element belongs to. The key is the id of the way, and the value is the role of the way. Defaults to an empty dictionary.
        nodes (dict): A dictionary of the nodes that the regulatory element belongs to. The key is the id of the node, and the value is the role of the node. Defaults to an empty dictionary.
        type_ (str): The type of the regulatory element. Defaults to "regulatory_element".
        subtype (str): The subtype of the regulatory element.
        position (str): The position of the regulatory element. Defaults to None.
        location (str): The location of the regulatory element. Defaults to None.
        dynamic (bool): Whether the regulatory element is dynamic. Defaults to False.
        fallback (bool): Whether the regulatory element is a fallback. Defaults to False.
        custom_tags (dict): The custom tags of the regulatory element. Defaults to None.
    """

    def __init__(
        self,
        id_: str,
        relations: dict = dict(),
        ways: dict = dict(),
        nodes: dict = dict(),
        type_: str = "regulatory_element",
        subtype: str = None,
        position: str = None,
        location: str = None,
        dynamic: bool = False,
        fallback: bool = False,
        custom_tags: dict = None,
    ):
        """Initialize the attributes in the class.

        Args:
            id_ (str): The id of the regulatory element.
            relations (dict, optional): A dictionary of the relations that the regulatory element belongs to. The key is the id of the relation, and the value is the role of the relation.
            ways (dict, optional): A dictionary of the ways that the regulatory element belongs to. The key is the id of the way, and the value is the role of the way.
            nodes (dict, optional): A dictionary of the nodes that the regulatory element belongs to. The key is the id of the node, and the value is the role of the node.
            type_ (str, optional): The type of the regulatory element.
            subtype (str, optional): The subtype of the regulatory element.
            position (str, optional): The position of the regulatory element.
            location (str, optional): The location of the regulatory element.
            dynamic (bool, optional): Whether the regulatory element is dynamic.
            fallback (bool, optional): Whether the regulatory element is a fallback.
            custom_tags (dict, optional): The custom tags of the regulatory element.
        """

        if subtype is None:
            raise ValueError("The subtype of Regulatory %s is not defined!" % id_)

        self.id_ = id_
        self.relations = relations
        self.ways = ways
        self.nodes = nodes
        self.type_ = type_
        self.subtype = subtype
        self.position = position
        self.location = location
        self.dynamic = dynamic
        self.fallback = fallback
        self.custom_tags = custom_tags
