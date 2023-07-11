"""
This script provides parameters for building models of traffic participants.

The definition of vehicle types is based on the European Emissions Standards (EEC, https://en.wikipedia.org/wiki/Vehicle_size_class) due to its clarity and simplicity. To obtain the parameters, we chose one specific vehicle from each type of vehicle based on typical (highest selling) vehicles and available data found online. These choices were made to ensure the data used is as representative and accurate as possible.

The parameters of the pedestrians are referred to the data provided by ChatGPT.
"""


VEHICLE_MODEL = {
    "micro_car": {
        # Prototype: Smart Fortwo (W453) (https://en.wikipedia.org/wiki/Smart_Fortwo)
        "length": 2.695,
        "width": 1.663,
        "height": 1.555,
        "wheel_base": 1.873,
        "front_overhang": 0.424,
        "rear_overhang": 0.398,
        "max_speed": 41.9,
        "max_accel": 1.941,
        "max_decel": -7.5,
        "0_100_km/h": 15.1,
    },
    "mini_car": {
        # Prototype: Volkswagen Up! (https://en.wikipedia.org/wiki/Volkswagen_Up)
        "length": 3.540,
        "width": 1.641,
        "height": 1.478,
        "wheel_base": 2.420,
        "front_overhang": 0.585,
        "rear_overhang": 0.535,
        "max_speed": 47.5,
        "max_accel": 2.5,
        "max_decel": -7.5,
        "0_100_km/h": 13.9,
    },
    "small_car": {
        # Prototype: Volkswagen Polo(AW/BZ) (https://en.wikipedia.org/wiki/Volkswagen_Polo)
        "length": 4.053,
        "width": 1.751,
        "height": 1.461,
        "wheel_base": 2.548,
        "front_overhang": 0.824,
        "rear_overhang": 0.681,
        "max_speed": 52.8,
        "max_accel": 3.517,
        "max_decel": -9.8,
        "0_100_km/h": 12,
    },
    "sedan": {
        "length": 4.668,
        "width": 1.825,
        "height": 1.481,
        "wheel_base": 2.700,
        "front_overhang": 0,
        "rear_overhang": 0,
        "max_speed": 0,
        "max_accel": 0,
        "max_decel": 0,
    },
    "medium_car": {
        # Prototype: Volkswagen Golf (https://car.autohome.com.cn/config/series/871-10281.html#pvareaid=3454437)
        "length": 4.259,
        "width": 1.799,
        "height": 1.452,
        "wheel_base": 2.637,
        "front_overhang": 0.841,
        "rear_overhang": 0.781,
        "max_speed": 52.8,
        "max_accel": 5.586,
        "max_decel": -10.7,
        "0_100_km/h": 10.9,
    },
    # "medium_car": {
    #     # Prototype: Volkswagen Golf (https://car.autohome.com.cn/config/series/871-10281.html#pvareaid=3454437)
    #     "length": 2.8+0.96+0.929,
    #     "width": 1.942,
    #     "height": 1.452,
    #     "wheel_base": 2.8,
    #     "front_overhang": 0.96,
    #     "rear_overhang": 0.929,
    #     "max_speed": 52.8,
    #     "max_accel": 5.586,
    #     "max_decel": -10.7,
    #     "0_100_km/h": 10.9,
    # },
    "large_car": {
        # Prototype: Volkswagen Passat (B8) (Magotan) (https://en.wikipedia.org/wiki/Volkswagen_Passat_(B8))
        "length": 4.866,
        "width": 1.832,
        "height": 1.464,
        "wheel_base": 2.871,
        "front_overhang": 0.955,
        "rear_overhang": 1.040,
        "max_speed": 58.3,
        "max_accel": 5.66,
        "max_decel": -11.27,
        "0_100_km/h": 9.1,
    },
    "executive_car": {
        # Prototype: Audi A6L (https://price.pcauto.com.cn/sg4313/config.html)
        "length": 5.050,
        "width": 1.886,
        "height": 1.475,
        "wheel_base": 3.024,
        "front_overhang": 0.921,
        "rear_overhang": 1.105,
        "max_speed": 63.9,
        "max_accel": 5.88,
        "max_decel": -14.7,
        "0_100_km/h": 8.3,
    },
    "luxury_car": {
        # Prototype: Audi A8L(https://car.autohome.com.cn/config/series/146.html)
        "length": 5.320,
        "width": 1.945,
        "height": 1.488,
        "wheel_base": 2.998,
        "front_overhang": 0.989,
        "rear_overhang": 1.185,
        "max_speed": 69.4,
        "max_accel": 8.52,
        "max_decel": -13.23,
        "0_100_km/h": 6.7,
    },
    "sports_coupes": {
        # Prototype: Chevrolet Camaro(https://en.wikipedia.org/wiki/Chevrolet_Camaro)
        "length": 4.786,
        "width": 1.897,
        "height": 1.356,
        "wheel_base": 2.811,
        "front_overhang": 0.829,
        "rear_overhang": 1.146,
        "max_speed": 66.7,
        "max_accel": 9.65,
        "max_decel": -14.02,
        "0_100_km/h": 5.9,
    },
    "mini_MPV": {
        # Prototype: Opel Meriva(https://en.wikipedia.org/wiki/Opel_Meriva)
        "length": 4.288,
        "width": 1.812,
        "height": 1.615,
        "wheel_base": 2.644,
        "front_overhang": 0.931,
        "rear_overhang": 0.713,
        "max_speed": 53.4,
        "max_accel": 3.8,
        "max_decel": -8,
        "0_100_km/h": 9.9,
    },
    "large_MPV": {
        # Prototype: Volkswagen Sharan(https://en.wikipedia.org/wiki/Volkswagen_Sharan)
        "length": 4.850,
        "width": 1.924,
        "height": 1.720,
        "wheel_base": 2.920,
        "front_overhang": 0.960,
        "rear_overhang": 0.970,
        "max_speed": 62.8,
        "max_accel": 3.8,
        "max_decel": -8,
        "0_100_km/h": 9.9,
    },
    "full_size_cargo_van": {
        # Prototype: Mercedes-Benz Sprinter(https://en.wikipedia.org/wiki/Mercedes-Benz_Sprinter#Third_generation_(2019%E2%80%93present,_VS30))
        "length": 5.910,
        "width": 1.993,
        "height": 2.955,
        "wheel_base": 3.665,
        "front_overhang": 1.004,
        "rear_overhang": 1.240,
        "max_speed": 35,
        "max_accel": 3.8,
        "max_decel": -7.5,
    },
    "SUV": {
        # Prototype: Kia Mohave (https://en.wikipedia.org/wiki/Kia_Mohave)
        "length": 4.930,
        "width": 1.920,
        "height": 1.790,
        "wheel_base": 2.895,
        "front_overhang": 0.874,
        "rear_overhang": 1.161,
        "max_speed": 53.8,
        "max_accel": 4.27,
        "max_decel": -8,
        "0_100_km/h": 8.5,
    },
}

TRUCK_MODEL = {
    "truck": {
        "length": 8.7,
        "width": 2.4,
        "height": 2.7,
        "max_speed": 33.3,
        "max_accel": 2,
        "max_decel": -7.5,
    }
}

BUS_MODEL = {
    "bus": {
        "length": 11.530,
        "width": 2.5,
        "height": 3.04,
        "max_speed": 16.7,
        "max_accel": 2,
        "max_decel": -7.5,
    }
}


PEDESTRIAN_MODEL = {
    "pedestrian/male": {
        "length": 0.24,
        "width": 0.41,
        "height": 1.75,
        "max_speed": 8,
        "max_accel": 1.5,
        "max_decel": -4,
    },
    "pedestrian/female": {
        "length": 0.22,
        "width": 0.37,
        "height": 1.63,
        "max_speed": 7,
        "max_accel": 1.7,
        "max_decel": -4,
    },
    "pedestrian/six_years_old_children": {
        "length": 0.18,
        "width": 0.25,
        "height": 1.16,
        "max_speed": 3.5,
        "max_accel": 1,
        "max_decel": -2,
    },
    "pedestrian/ten_years_old_children": {
        "length": 0.22,
        "width": 0.34,
        "height": 1.42,
        "max_speed": 4.5,
        "max_accel": 1.5,
        "max_decel": -3,
    },
    "cyclist": {
        "length": 1.80,
        "width": 0.65,
        "height": 1.70,
        "max_speed": 7,
        "max_accel": 2.9,
        "max_decel": -4,
    },
    "moped": {
        "length": 2.00,
        "width": 0.70,
        "height": 1.70,
        "max_speed": 13.4,
        "max_accel": 3.5,
        "max_decel": -7,
    },
    "motorcycle": {
        "length": 2.40,
        "width": 0.80,
        "height": 1.70,
        "max_speed": 16.7,
        "max_accel": 4,
        "max_decel": -7.5,
    },
}
