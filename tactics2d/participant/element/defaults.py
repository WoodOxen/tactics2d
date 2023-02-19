"""
This script provides parameters for building models of traffic participants. 

The definition of vehicle types is based on the European Emissions Standards (EEC, https://en.wikipedia.org/wiki/Vehicle_size_class) due to its clarity and simplicity. To obtain the parameters, we chose one specific vehicle from each type of vehicle based on typical (highest selling) vehicles and available data found online. These choices were made to ensure the data used is as representative and accurate as possible.

The parameters of the pedestrians are referred to the data provided by ChatGPT.
"""


CAR_MODEL = {
    "mini_car": {
        # Prototype: Volkswagen Up! (https://en.wikipedia.org/wiki/Volkswagen_Up)
        "length": 3.540,
        "width": 1.641,
        "height": 1.478,
        "wheel_base": 2.420,
        "front_hang": 0,
        "rear_hang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "small_car": {
        # Prototype: Volkswagen Polo (https://en.wikipedia.org/wiki/Volkswagen_Polo)
        "length": 4.053,
        "width": 1.751,
        "height": 1.438,
        "wheel_base": 2.548,
        "front_hang": 0,
        "rear_hang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "medium_car": {
        # Prototype: Ford Focus (fourth_generation) (wagon) (https://en.wikipedia.org/wiki/Ford_Focus_(fourth_generation))
        "length": 4.668,
        "width": 1.825,
        "height": 1.481,
        "wheel_base": 2.700,
        "front_hang": 0,
        "rear_hang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "large_car": {
        # Prototype: Volkswagen Passat (B8) (Magotan) (https://en.wikipedia.org/wiki/Volkswagen_Passat_(B8))
        "length": 4.866,
        "width": 1.832,
        "height": 1.464,
        "wheel_base": 2.871,
        "front_hang": 0,
        "rear_hang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "large_car": {
        # Prototype: Volkswagen Passat (B8) (Magotan) (https://en.wikipedia.org/wiki/Volkswagen_Passat_(B8))
        "length": 4.866,
        "width": 1.832,
        "height": 1.464,
        "wheel_base": 2.871,
        "front_hang": 0,
        "rear_hang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "executive_car": {
        # Prototype: Audi A6 (https://en.wikipedia.org/wiki/Audi_A6#C8_(Typ_4K,_2018%E2%80%93present))
        "length": 4.939,
        "width": 1.886,
        "height": 1.457,
        "wheel_base": 2.924,
        "front_hang": 0,
        "rear_hang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "luxury_car": {
        # Prototype: Audi A8 (Magotan) (https://en.wikipedia.org/wiki/Volkswagen_Passat_(B8))
        "length": 5.172,
        "width": 1.945,
        "height": 1.485,
        "wheel_base": 2.998,
        "front_hang": 0,
        "rear_hang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "SUV": {
        # Prototype: Jeep Wagoneer (WS) (https://en.wikipedia.org/wiki/Jeep_Wagoneer_(WS))
        "length": 5.453,
        "width": 2.123,
        "height": 1.920,
        "wheel_base": 3.124,
        "front_hang": 0,
        "rear_hang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    }
}

TRUCK_MODEL = {

}
    
BUS_MODEL = {

}


PEDESTRIAN_MODEL = {
    "pedestrian/male": {
        "length": 0.24,
        "width": 0.41,
        "height": 1.75,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "pedestrian/female": {
        "length": 0.22,
        "width": 0.37,
        "height": 1.63,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "pedestrian/six_years_old_children": {
        "length": 0.18,
        "width": 0.25,
        "height": 1.16,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "pedestrian/tem_years_old_children": {
        "length": 0.22,
        "width": 0.34,
        "height": 1.42,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "cyclist": {
        "length": 1.80,
        "width": 0.65,
        "height": 1.70,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "moped": {
        "length": 2.00,
        "width": 0.70,
        "height": 1.70,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "motorcycle": {
        "length": 2.40,
        "width": 0.80,
        "height": 1.70,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    }
}