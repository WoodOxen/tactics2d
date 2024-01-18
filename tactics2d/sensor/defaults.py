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
    "gray": "#778ca3",
    "dark-gray": "#4b6584",
    "black": "#2f3542",
}

DEFAULT_COLOR = {
    # default color for lane class subtypes
    "lane": COLOR_PALETTE["black"],
    "road": COLOR_PALETTE["black"],
    "highway": COLOR_PALETTE["black"],
    "play_street": COLOR_PALETTE["gray"],
    "emergency_lane": COLOR_PALETTE["red"],
    "bus_lane": COLOR_PALETTE["dark-gray"],
    "bicycle_lane": COLOR_PALETTE["dark-gray"],
    "exit": COLOR_PALETTE["green"],
    "walkway": COLOR_PALETTE["gray"],
    "shared_walkway": "gray",
    "crosswalk": COLOR_PALETTE["dark-gray"],
    "stairs": COLOR_PALETTE["gray"],
    # default color for area class subtypes
    "area": COLOR_PALETTE["black"],
    "hole": COLOR_PALETTE["white"],
    "parking": COLOR_PALETTE["dark-gray"],
    "freespace": COLOR_PALETTE["black"],
    "vegetation": COLOR_PALETTE["green"],
    "keepout": COLOR_PALETTE["red"],
    "building": COLOR_PALETTE["gray"],
    "traffic_island": COLOR_PALETTE["gray"],
    "obstacle": COLOR_PALETTE["gray"],
    # default color for roadline class subtypes
    "roadline": COLOR_PALETTE["white"],
    # default color for vehicle class subtypes
    "vehicle": COLOR_PALETTE["light-turquoise"],
    # default color for cyclist class subtypes
    "cyclist": COLOR_PALETTE["light-orange"],
    # default color for pedestrian class subtypes
    "pedestrian": COLOR_PALETTE["light-blue"],
}

DEFAULT_ORDERS = {
    "lane": 1,
    
}