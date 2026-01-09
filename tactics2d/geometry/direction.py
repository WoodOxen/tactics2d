# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Direction implementation."""


from enum import Enum


class RelativeDirection(Enum):
    """Enum representing relative directions from a reference point.

    Attributes:
        LEFT: Represents the left direction ("L").
        RIGHT: Represents the right direction ("R").
        FRONT: Represents the forward direction ("F").
        BACK: Represents the backward direction ("B").
    """

    LEFT = "L"
    RIGHT = "R"
    FRONT = "F"
    BACK = "B"

    @classmethod
    def from_string(cls, direction: str) -> "RelativeDirection":
        """Convert a string to a RelativeDirection enum member.

        Args:
            direction (str): The string representation of the direction.

        Returns:
            RelativeDirection: The corresponding enum member.

        Raises:
            ValueError: If the input string does not match any enum member.
        """
        for member in cls:
            if member.value == direction.upper():
                return member

        raise ValueError(
            f"Invalid direction: {direction}. Must be one of {list(cls.__members__.values())}."
        )


class CardinalDirection(Enum):
    """Enum representing cardinal (global) compass directions.

    Attributes:
        NORTH: Represents north direction ("N").
        SOUTH: Represents south direction ("S").
        EAST: Represents east direction ("E").
        WEST: Represents west direction ("W").
    """

    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"

    @classmethod
    def from_string(cls, direction: str) -> "CardinalDirection":
        """Convert a string to a CardinalDirection enum member.

        Args:
            direction (str): The string representation of the direction.

        Returns:
            CardinalDirection: The corresponding enum member.

        Raises:
            ValueError: If the input string does not match any enum member.
        """
        for member in cls:
            if member.value == direction.upper():
                return member

        raise ValueError(
            f"Invalid direction: {direction}. Must be one of {list(cls.__members__.values())}."
        )
