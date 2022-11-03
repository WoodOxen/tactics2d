ALL_VEHICLE = {"car", "bus", "truck", "taxi"}
MOTOR = {"motorcycle", "motorbike"}
CYCLIST = {"cyclist"}
PEDESTRIAN = {"pedestrian", "walker"}
ALL_PARTICIPANT = ALL_VEHICLE.union(MOTOR, CYCLIST, PEDESTRIAN)

DEFAULT_LANE = {
    "road": {
        "drivable": True, 
        "participants": ALL_VEHICLE,
        "color": (190, 190, 190, 255),
        "speed_limit": 20,
    }, 
    "highway": {
        "drivable": True, 
        "participants": ALL_VEHICLE,
        "color": (190, 190, 190, 255),
        "speed_limit": 35,
    }, 
    "play_street": {
        "drivable": True, 
        "participants": ALL_PARTICIPANT,
        "color": (190, 190, 190, 255),
        "speed_limit": 15,
    }, 
    "emergency_lane": {
        "drivable": True, 
        "participants": [],
        "color": (190, 190, 190, 255),
        "speed_limit": 35,
    }, 
    "bus_lane": {
        "drivable": True, 
        "participants": [],
        "color": (190, 190, 190, 255),
        "speed_limit": 15,
    }, 
    "bicycle_lane": {
        "drivable": False, 
        "participants": CYCLIST.union(MOTOR),
        "color": (190, 190, 190, 255),
        "speed_limit": 10,
    }, 
    "exit": {
        "drivable": False, 
        "participants": PEDESTRIAN.union(CYCLIST, MOTOR),
        "color": (190, 190, 190, 255),
        "speed_limit": 5
    }, 
    "walkway": {
        "drivable": False, 
        "participants": PEDESTRIAN,
        "color": (190, 190, 190, 255),
        "speed_limit": 5
    }, 
    "shared_way": {
        "drivable": True, 
        "participants": PEDESTRIAN.union(CYCLIST),
        "color": (190, 190, 190, 255),
        "speed_limit": 10,
    }, 
    "crosswalk": {
        "drivable": False, 
        "participants": PEDESTRIAN, 
        "color": (190, 190, 190, 255),
        "speed_limit": 5,
    }, 
    "stairs": {
        "drivable": False, 
        "participants": PEDESTRIAN, 
        "color": (190, 190, 190, 255),
        "speed_limit": 5,
    }, 
}


DEFAULT_AREA = {
    "parking": {"color": (190, 190, 190, 255)},
    "freespace": {"color": (190, 190, 190, 255)},
    "vegetation": {"color": (190, 190, 190, 255)},
    "building": {"color": (190, 190, 190, 255)},
    "traffic_island": {"color": (190, 190, 190, 255)},
}