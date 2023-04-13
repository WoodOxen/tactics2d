from enum import Enum


class TrafficEvent(Enum):
    """All the possible traffic status in the scenario.

    - 1 (Normal): The agent behaves normally.
    - 2 (Completed): The agent completed the goal in the scenario.
    - 3 (Collision_static): The agent collides into a static object, such as a building.
    - 4 (Collision_vehicle): The agent collides with an npc vehicle.
    - 5 (Collision_pedestrian): The agent collides with an npc pedestrian.
    - 6 (Route_deviation): The agent is deviated from the arbitrary route.
    - 7 (Violation_retrograde): The agent violates the traffic rule because it is on the
        reversed direction of the lane.
    - 8 (Violation_non_drivable): The agent violates the traffic rule because it is in a
        non-drivable region, such as the bicycle lane, side walk, and traffic island.
    - 9 (Violation_outside_lane):
    - 10 (Outside lane): The agent has driven out of the lane.
    - 11 (Outside Map): The agent has driven out of the map.
    - 12 (Time exceeded): The agent fails to complete the arbitrary route within the time limit.
    - 13 (No action): The agent does not take any action for a while.
    """

    NORMAL = 1
    COMPLETED = 2
    COLLISION_STATIC = 3
    COLLISION_VEHICLE = 4
    COLLISION_PEDESTRIAN = 5
    ROUTE_DEVIATION = 6
    VIOLATION_RETROGRADE = 7
    VIOLATION_NON_DRIVABLE = 8
    VIOLATION_TRAFFIC_LIGHT = 9
    OUTSIDE_LANE = 10
    OUTSIDE_MAP = 11
    TIME_EXCEEDED = 12
    NO_ACTION = 13
