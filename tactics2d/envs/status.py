from enum import Enum


class Status(Enum):
    """All the possible statuses of an agent in the training environment.

    - 1 (Normal): The agent behaves normally. It has no sign of breaking any rule.
    - 2 (Arrived): The agent has arrived at the destination.
    - 3 (Retrograde): The agent is moving in the reversed direction of the lane.
    - 4 (Collided): The agent collided with another object.
    - 5 (Non-drivable): The agent drives into a non-drivable region of the map.
    - 6 (Outbound): The agent has driven out of the map's boundary.
    - 7 (Out-time): The agent fails to arrive at the destination within the time limit.
    """
    NORMAL = 1
    ARRIVED = 2
    RETROGRADE = 3
    COLLIDED = 4
    NON_DRIVABLE = 5
    OUTBOUND = 6
    OUT_TIME = 7

    def __str__(self) -> str:
        return self.name