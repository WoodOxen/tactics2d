VEHICLE_TYPE = {"car", "van", "truck", "bus", "trailer"}
MOTORCYCLE_TYPE = {"motorcycle"}
BICYCLE_TYPE = {"cycle", "bicycle"}
ALL_VEHICLE_TYPE = VEHICLE_TYPE | MOTORCYCLE_TYPE | BICYCLE_TYPE
PEDESTRIAN_TYPE = {"pedestrian", "walker"}

NAME_MAPPING = {
    "car": "car",
    "Car": "car",
    "van": "van",
    "truck": "truck",
    "Truck": "truck",
    "truck_bus": "bus",
    "bus": "bus",
    "trailer": "trailer",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "cycle": "cycle",
    "pedestrian": "pedestrian"
}

# Refer to https://sumo.dlr.de/docs/Vehicle_Type_Parameter_OBJECTs.html
OBJECT_SIZE = {
    "bicycle": [1.6, 0.65, 1.7],
    "motorcycle": [2.2, 0.9, 1.5],
    "car": [5, 1.8, 1.5], # passenger
    "van": [4.7, 1.9, 1.73], # passenger/van
    "truck": [7.1, 2.4, 2.4],
    "trailer": [18.75, 2.55, 4], # truck/trailer
    "bus": [12, 2.5, 3.4]
}

# unit: km/h
OBJECT_SPEED_LIMIT= {
    "pedestrian": 10,
    "bicycle": 20,
    "motorcycle": 200,
    "car": 200,
    "van": 200,
    "truck": 130,
    "trailer": 130,
    "bus": 85
}

# unit: m/s^2
OBJECT_MAX_ACCEL = {
    "pedestrian": 1.5,
    "bicycle": 1.2,
    "motorcycle": 6,
    "car": 2.6,
    "van": 2.6,
    "truck": 1.3,
    "trailer": 1,
    "bus": 1.2
}

# The max deceleration considering the passengers' comfort
OBJECT_MAX_DECEL = {
    "pedestrian": -2,
    "bicycle": -3,
    "motorcycle": -10,
    "car": -4.5,
    "van": -4.5,
    "truck": -4,
    "trailer": -4,
    "bus": -4
}

OBJECT_EMERGENCY_DECEL = {
    "pedestrian": -5,
    "bicycle": -7,
    "motorcycle": -10,
    "car": -9,
    "van": -9,
    "truck": -7,
    "trailer": -7,
    "bus": -7
}