from enum import Enum


class TrafficEvent(Enum):
    """All the possible traffic events of an agent during driving.

    - 1 (Normal): The agent behaves normally.
    - 2 (Collision_static): The agent collides into a static object, such as a building.
    - 3 (Collision_vehicle): The agent collides with an npc vehicle.
    - 4 (Collision_pedestrian): The agent collides with an npc pedestrian.
    - 5 (Route_deviation): The agent is deviated from the arbitrary route.
    - 6 (Route_completed): The agent manages to finish the arbitrary route.
    - 7 (Violation_retrograde): The agent violates the traffic rule because it is on the reversed direction of the lane.
    - 8 (Violation_non_drivable): The agent violates the traffic rule because it is in a non-drivable region, such as the bicycle lane, side walk, and traffic island.
    - 9 (Violation_outside_lane):
    - 10 (Out_lane): The agent has driven out of the lane.
    - 11 (Out_Map): The agent has driven out of the map.
    - 12 (Out-time): The agent fails to complete the arbitrary route within the time limit.
    """

    NORMAL = 1
    COLLISION_STATIC = 2
    COLLISION_VEHICLE = 3
    COLLISION_PEDESTRIAN = 4
    ROUTE_DEVIATION = 5
    ROUTE_COMPLETED = 6
    VIOLATION_RETROGRADE = 7
    VIOLATION_NON_DRIVABLE = 8
    VIOLATION_TRAFFIC_LIGHT = 9
    OUTSIDE_LANE = 10
    OUTSIDE_MAP = 11
    TIME_EXCEED = 12