##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: status.py
# @Description: This script defines the status by enum class for a traffic scenario.
# @Author: Yueyuan Li
# @Version: 1.0.0

from enum import Enum


class ScenarioStatus(Enum):
    """This class defines the high-level status of a traffic scenario.

    The possible status are as follows:

    1. `NORMAL`: The scenario is running normally.
    2. `COMPLETED`: The scenario is completed.
    3. `TIME_EXCEEDED`: The scenario is not completed within the time limit.
    4. `FAILED`: The scenario is failed. The reasons for the failure are various and should be specified by [TrafficStatus](#tactics2d.traffic.TrafficStatus) or user defined status.
    """

    NORMAL = 1
    COMPLETED = 2
    TIME_EXCEEDED = 3
    FAILED = 4


class TrafficStatus(Enum):
    """This class provides a sample for describing the low-level status of a traffic scenario. The user can custom their own status class.

    The preliminaries of the status description are as follows:
        - This class is used by **the ego vehicle** controlled by an driving policy-maker in the scenario.
        - All the other agents are either controlled by humans or other policy-makers.

    The possible status are as follows:

    1. `NORMAL`: The ego vehicle behaves normally.
    2. `UNKNOWN`: The traffic status of the ego vehicle is unchecked and unknown for some reasons.
    3. `COLLISION_STATIC`: The ego vehicle collides into a static object, such as a building.
    4. `COLLISION_DYNAMIC`: The ego vehicle collides with a movable traffic participant, such as a vehicle or pedestrian.
    5. `OFF_ROUTE`: The ego vehicle is deviated from the arbitrary route.
    6. `OFF_LANE`: The ego vehicle is out of the lane.
    7. `VIOLATION_RETROGRADE`: The ego vehicle violates the traffic rule because it is on the reversed direction of the lane.
    8. `VIOLATION_NON_DRIVABLE`: The ego vehicle violates the traffic rule because it is in a non-drivable region, such as the bicycle lane, side walk, and traffic island.
    9. `VIOLATION_TRAFFIC_LIGHT`: The ego vehicle violates the traffic rule because it does not obey the traffic light.
    10. `VIOLATION_TRAFFIC_SIGN`: The ego vehicle violates the traffic rule because it does not obey the traffic sign.
    11. `NO_ACTION`: The ego vehicle does not take any action for a given time period.
    """

    NORMAL = 1
    UNKNOWN = 2
    COLLISION_STATIC = 3
    COLLISION_DYNAMIC = 4
    OFF_ROUTE = 5
    OFF_LANE = 6
    VIOLATION_RETROGRADE = 7
    VIOLATION_NON_DRIVABLE = 8
    VIOLATION_TRAFFIC_LIGHT = 9
    VIOLATION_TRAFFIC_SIGN = 10
    NO_ACTION = 11
