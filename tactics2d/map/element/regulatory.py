class RegulatoryElement(object):
    """Implementation of the lanelet2-style regulatory element.

    Detailed definition of lanelet2-style regulatory element:
        <https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/RegulatoryElementTagging.md>

    Attributes:
        id_ (str): The unique identifier of the regulatory element.
        relation_ids ():
        way_ids ():
        type_ (str): The type of the regulatory element. The default value is 
            "regulatory_element".
        subtype (): By default it is one of [traffic_sign, traffic_light, speed_limit,
            right_of_way, all_way_stop]
        dynamic (bool): Whether the RegulatoryElement will be changed by time or not.
        fallback (bool): Whether the Regulatory Element has a lower priority than other 
            RegulatoryElements. 
        custom_tags (dict): 
    """

    def __init__(
        self, id_: str, relation_ids: set = None, way_ids: set = None,
        type_: str = "regulatory_element", subtype: str = None,
        location: str = None, dynamic: bool = False, fallback: bool = False,
        custom_tags: dict = None,
    ):
        if subtype is None:
            raise ValueError(
                "The subtype of RegulatoryElement %s is not defined!" % id_
            )

        self.id_ = id_
        self.relation_ids = relation_ids
        self.way_ids = way_ids
        self.type_ = type_
        self.subtype = subtype
        self.location = location
        self.dynamic = dynamic
        self.fallback = fallback
        self.custom_tags = custom_tags
