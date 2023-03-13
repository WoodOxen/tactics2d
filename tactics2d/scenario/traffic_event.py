from enum import Enum


class TrafficEvent(Enum):
    """All the possible traffic status in the scenario.

    - 1 (Normal): The agent behaves normally.
    - 2 (Completed): The agent completed the goal in the scenario.
    - 3 (Collision_static): The agent collides into a static object, such as a building.
    - 4 (Collision_vehicle): The agent collides with an npc vehicle.
    - 5 (Collision_pedestrian): The agent collides with an npc pedestrian.
    - 6 (Route_deviation): The agent is deviated from the arbitrary route.
    - 7 (Route_completed): The agent manages to finish the arbitrary route.
    - 8 (Violation_retrograde): The agent violates the traffic rule because it is on the
        reversed direction of the lane.
    - 9 (Violation_non_drivable): The agent violates the traffic rule because it is in a
        non-drivable region, such as the bicycle lane, side walk, and traffic island.
    - 10 (Violation_outside_lane):
    - 11 (Out_lane): The agent has driven out of the lane.
    - 12 (Out_Map): The agent has driven out of the map.
    - 13 (Out-time): The agent fails to complete the arbitrary route within the time limit.
    """

    NORMAL = 1
    COMPLETED = 2
    COLLISION_STATIC = 3
    COLLISION_VEHICLE = 4
    COLLISION_PEDESTRIAN = 5
    ROUTE_DEVIATION = 6
    ROUTE_COMPLETED = 7
    VIOLATION_RETROGRADE = 8
    VIOLATION_NON_DRIVABLE = 9
    VIOLATION_TRAFFIC_LIGHT = 10
    OUTSIDE_LANE = 11
    OUTSIDE_MAP = 12
    TIME_EXCEED = 13
