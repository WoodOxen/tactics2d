import warnings

from shapely.geometry import Polygon

from .defaults import LEGAL_SPEED_UNIT


class Area:
    """This class implements the lenelet2-style map element *area*.

    Detailed definition of lanelet2-style area: 
        [LaneletPrimitives.md](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md)

    Attributes:
        id_ (str): The unique identifier of the area.
        geometry (Polygon): The shape of the area expressed in geometry format.
        line_ids (dict, optional): The ids of the lines that circle this area. Defaults to None.
        type_ (str): The type of the area. The default value is "multipolygon".
        subtype (str, optional): The subtype of the area. Defaults to None.
        color (tuple, optional): The color of the area. Defaults to None.
        location (str, optional): The location of the area (urban, nonurban, etc.). Defaults to
            None.
        inferred_participants (list, optional): The allowing type of traffic participants that
            can pass the area. Defaults to None.
        speed_limit (float, optional): The speed limit in this area. Defaults to None.
        speed_limit_unit (str, optional): The unit of speed limit in this area. The valid units
            are `"km/h"`, `"mi/h"`, and `"m/s"`. Defaults to `"km/h"`.
        speed_limit_mandatory (bool, optional): Whether the speed limit is mandatory or
            not. Defaults to True.
        custom_tags (dict, optional): The custom tags of the area. Defaults to None.
    """

    def __init__(
        self,
        id_: str,
        geometry: Polygon,
        line_ids: dict = None,
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
        """Initialize the attributes in the class."""
        self.id_ = id_
        self.geometry = geometry
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

        if self.speed_limit_unit not in LEGAL_SPEED_UNIT:
            warnings.warn(
                "Invalid speed limit unit %s. The legal units types are %s"
                % (self.speed_limit_unit, ", ".join(LEGAL_SPEED_UNIT))
            )

    def shape(self, outer_only: bool = False):
        outer_shape = list(self.polygon.exterior.coords)
        if outer_only:
            return outer_shape
        inners_shape = list(self.polygon.interiors.coords)
        return outer_shape, inners_shape
