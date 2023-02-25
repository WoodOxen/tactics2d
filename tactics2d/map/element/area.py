import warnings

from shapely.geometry import Polygon

from .defaults import LEGAL_SPEED_UNIT


class Area(object):
    """## tactics2d.map.element.Area

    An implementation of the lanelet2-style area.

    The detailed definition of the lanelet2-style area can be found here:
    https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md

    Attributes:
        id_ (str): _description_.
        polygon (Polygon): _description_.
        subtype (str, optional): _description_. Defaults to None.
        location (str, optional): _description_. Defaults to None.
        inferred_participants (list, optional): _description_. Defaults to None.
        speed_limit (float, optional): _description_. Defaults to None.
        speed_limit_unit (str, optional): _description_. Defaults to "km/h".
        speed_limit_mandatory (bool, optional): _description_. Defaults to True.
    """

    def __init__(
        self,
        id_: str,
        polygon: Polygon,
        line_ids: dict,
        type_: str = "multipolygon",
        subtype: str = None,
        color: tuple = None,
        location: str = None,
        inferred_participants: list = None,
        speed_limit: float = None,
        speed_limit_unit: str = "km/h",
        speed_limit_mandatory: bool = True,
        custom_tags: dict = None,
    ):
        self.id_ = id_
        self.polygon = polygon
        self.line_ids = line_ids
        self.type_ = type_
        self.subtype = subtype
        self.color = color
        self.location = location
        self.inferred_participants = inferred_participants
        self.speed_limit = speed_limit
        self.speed_limit_unit = speed_limit_unit
        self.speed_limit_mandatory = speed_limit_mandatory
        self.custom_tags = custom_tags

    @property
    def shape(self, outer_only: bool = False):
        """Get shape of the area"""
        outer_shape = list(self.polygon.exterior.coords)
        if outer_only:
            return outer_shape
        inners_shape = list(self.polygon.interiors.coords)
        return outer_shape, inners_shape

    def is_valid(self) -> bool:
        """ """
        if self.speed_limit_unit not in LEGAL_SPEED_UNIT:
            warnings.warn(
                "Invalid speed limit unit %s. The legal units types are %s"
                % (self.speed_limit_unit, ", ".join(LEGAL_SPEED_UNIT))
            )
