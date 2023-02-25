from pygame.colordict import THECOLORS


LANE_COLOR = {
    "default": THECOLORS["darkgray"],
    "road": THECOLORS["darkgray"],
    "highway": THECOLORS["dimgray"],
    "play_street": THECOLORS["slategray"],
    "emergency_lane": THECOLORS["red4"],
    "bus_lane": THECOLORS["dodgerblue4"],
    "bicycle_lane": THECOLORS["darkgreen"],
    "exit": THECOLORS["palegreen4"],
    "walkway": THECOLORS["azure3"],
    "shared_walkway": THECOLORS["darkgray"],
    "crosswalk": THECOLORS["silver"],
    "stairs": THECOLORS["lightslategray"],
}


AREA_COLOR = {
    "default": THECOLORS["slategray"],
    "hole": THECOLORS["white"],
    "parking": THECOLORS["darkgray"],
    "freespace": THECOLORS["slategray"],
    "vegetation": THECOLORS["forestgreen"],
    "keepout": THECOLORS["red2"],
    "building": THECOLORS["steelblue1"],
    "traffic_island": THECOLORS["silver"],
}


ROADLINE_COLOR = {
    "default": THECOLORS["white"],
}


VEHICLE_COLOR = {
    "default": THECOLORS["turquoise1"],
}


CYCLIST_COLOR = {
    "default": THECOLORS["cyan1"],
}


PEDESTRIAN_COLOR = {
    "default": THECOLORS["lightpink1"],
}
