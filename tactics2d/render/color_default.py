from tactics2d.render.color_dict import COLORS


LANE_COLOR = {
    "road": COLORS["darkgray"],
    "highway": COLORS["dimgray"],
    "play_street": COLORS["slategray"],
    "emergency_lane": COLORS["red4"],
    "bus_lane": COLORS["dodgerblue4"],
    "bicycle_lane": COLORS["darkgreen"],
    "exit": COLORS["palegreen4"],
    "walkway": COLORS["azure3"],
    "shared_walkway": COLORS["darkgray"],
    "crosswalk": COLORS["silver"],
    "stairs": COLORS["lightslategray"]
}


AREA_COLOR = {
    "parking": COLORS["darkgray"],
    "freespace": COLORS["slategray"],
    "vegetation": COLORS["springgreen2"],
    "keepout": COLORS["red2"],
    "building": COLORS["steelblue1"],
    "traffic_island": COLORS["silver"],
}
