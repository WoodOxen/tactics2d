from shapely.geometry import Point, LineString


class RegulatoryMember:
    def __init__(self, ref: str, type_: str, role: str):
        self.ref = ref
        self.type_ = type_
        self.role = role


class Regulatory:
    """This class implements the lenelet2-style map element *RegulatoryElement*.

    This class is still under development. It is supposed to support the detection of traffic
        events in the future.

    Detailed definition of lanelet2-style lane:
        [LaneletPrimitives](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md)

    Attributes:
        id_ (str): The unique identifier of the regulatory element.
        type_ (str): The type of the regulatory element. The default value is
            "regulatory_element".
        subtype (str, optional): By default it is one of [traffic_sign, traffic_light, speed_limit,
            right_of_way, all_way_stop]
        dynamic (bool, optional): Whether the regulatory element will be changed by time or not.
        fallback (bool, optional): Whether the regulatory element has a lower priority than other
            RegulatoryElements.
        custom_tags (dict, optional): The custom tags of the regulatory element. Defaults to None.
    """

    def __init__(
        self,
        id_: str,
        relation_ids: set = None,
        way_ids: set = None,
        type_: str = "regulatory_element",
        subtype: str = None,
        position: str = None,
        location: str = None,
        dynamic: bool = False,
        fallback: bool = False,
        custom_tags: dict = None,
    ):
        """Initialize the attributes in the class."""
        if subtype is None:
            raise ValueError("The subtype of Regulatory %s is not defined!" % id_)

        self.id_ = id_
        self.relation_ids = relation_ids
        self.way_ids = way_ids
        self.type_ = type_
        self.subtype = subtype
        self.position = position
        self.location = location
        self.dynamic = dynamic
        self.fallback = fallback
        self.custom_tags = custom_tags
