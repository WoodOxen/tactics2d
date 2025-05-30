##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: render_template.py
# @Description: This file contains the default color palette and zorder for rendering different classes of map elements.
# @Author: Yueyuan Li
# @Version: 1.0.0

COLOR_PALETTE = {
    "red": "#fc5c65",
    "light-red": "#fc5c65",
    "orange": "#fa8231",
    "light-orange": "#fd9644",
    "yellow": "#f7b731",
    "light-yellow": "#fed330",
    "green": "#20bf6b",
    "light-green": "#26de81",
    "turquoise": "#0fb9b1",
    "light-turquoise": "#2bcbba",
    "blue": "#2d98da",
    "light-blue": "#45aaf2",
    "royal-blue": "#3867d6",
    "light-royal-blue": "#4b7bec",
    "purple": "#8854d0",
    "light-purple": "#a55eea",
    "white": "#f1f2f6",
    "light-gray": "#a5b1c2",
    "gray": "#b2bec3",
    "dark-gray": "#4b6584",
    "black": "#2f3542",
}

DEFAULT_COLOR = {
    # default color for lane class subtypes
    "lane": COLOR_PALETTE["black"],
    "slipLane": COLOR_PALETTE["black"],
    "road": COLOR_PALETTE["black"],
    "driving": COLOR_PALETTE["black"],
    "bidirectional": COLOR_PALETTE["black"],
    "highway": COLOR_PALETTE["black"],
    "play_street": COLOR_PALETTE["gray"],
    "emergency_lane": COLOR_PALETTE["red"],
    "bus_lane": COLOR_PALETTE["dark-gray"],
    "bus": COLOR_PALETTE["dark-gray"],
    "bicycle_lane": COLOR_PALETTE["dark-gray"],
    "biking": COLOR_PALETTE["dark-gray"],
    "offRamp": COLOR_PALETTE["dark-gray"],
    "onRamp": COLOR_PALETTE["dark-gray"],
    "entry": COLOR_PALETTE["green"],
    "exit": COLOR_PALETTE["green"],
    "sidewalk": COLOR_PALETTE["gray"],
    "walking": COLOR_PALETTE["gray"],
    "walkway": COLOR_PALETTE["gray"],
    "shared_walkway": "gray",
    "crosswalk": COLOR_PALETTE["dark-gray"],
    "stairs": COLOR_PALETTE["gray"],
    # default color for area class subtypes
    "area": COLOR_PALETTE["black"],
    "hole": COLOR_PALETTE["white"],
    "parking": COLOR_PALETTE["black"],
    "parkingSpace": COLOR_PALETTE["black"],
    "rail": COLOR_PALETTE["dark-gray"],
    "freespace": COLOR_PALETTE["black"],
    "vegetation": COLOR_PALETTE["green"],
    "keepout": COLOR_PALETTE["red"],
    "stop": COLOR_PALETTE["red"],
    "building": COLOR_PALETTE["gray"],
    "traffic_island": COLOR_PALETTE["dark-gray"],
    "obstacle": COLOR_PALETTE["gray"],
    # default color for roadline class subtypes
    "roadline": COLOR_PALETTE["white"],
    "curbstone": COLOR_PALETTE["light-gray"],
    "road_border": COLOR_PALETTE["light-gray"],
    # default color for vehicle class subtypes
    "vehicle": COLOR_PALETTE["light-turquoise"],
    "car": COLOR_PALETTE["light-turquoise"],
    "truck": COLOR_PALETTE["light-turquoise"],
    "bus": COLOR_PALETTE["light-turquoise"],
    # default color for cyclist class subtypes
    "motorcycle": COLOR_PALETTE["light-orange"],
    "cyclist": COLOR_PALETTE["light-orange"],
    "bicycle": COLOR_PALETTE["light-orange"],
    # default color for pedestrian class subtypes
    "pedestrian": COLOR_PALETTE["light-blue"],
}

DEFAULT_ORDER = {
    # default zorder for lane class subtypes
    "lane": 3,
    "slipLane": 3,
    "road": 3,
    "driving": 3,
    "bidirectional": 3,
    "highway": 3,
    "play_street": 3,
    "emergency_lane": 3,
    "bus_lane": 2,
    "bus": 2,
    "bicycle_lane": 2,
    "biking": 2,
    "exit": 3,
    "entry": 3,
    "offRamp": 3,
    "onRamp": 3,
    "walkway": 3,
    "shared_walkway": 3,
    "crosswalk": 2,
    "stairs": 3,
    # default zorder for area class subtypes
    "area": 2,
    "hole": 3,
    "parking": 3,
    "parkingSpace": 3,
    "freespace": 3,
    "vegetation": 3,
    "keepout": 3,
    "building": 5,
    "traffic_island": 3,
    "obstacle": 5,
    # default zorder for roadline class subtypes
    "roadline": 4,
    "road_border": 4,
    "curbstone": 5,
    "vehicle": 6,
    "car": 6,
    "truck": 6,
    "bus": 6,
    # default zorder for cyclist class subtypes
    "motorcycle": 6,
    "cyclist": 6,
    "bicycle": 6,
    # default zorder for pedestrian class subtypes
    "pedestrian": 6,
}
