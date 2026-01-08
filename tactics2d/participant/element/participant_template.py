##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: templates.py
# @Description: The parameter templates for different types of vehicles and participants.
# @Author: Yueyuan Li
# @Version: 0.1.8rc1

from tabulate import tabulate

EURO_SEGMENT_MAPPING = {
    "A": "mini_car",
    "B": "small_car",
    "C": "medium_car",
    "D": "large_car",
    "E": "executive_car",
    "F": "luxury_car",
    "S": "sports_coupe",
    "M": "multi_purpose_car",
    "J": "sports_utility_car",
}

NCAP_MAPPING = {
    "supermini": "small_car",
    "small_family_car": "medium_car",
    "large_family_car": "large_car",
    "executive": "executive_car",
    "large_mpv": "multi_purpose_car",
    "large_off_road": "sports_utility_car",
}


EPA_MAPPING = {
    "minicompact": "mini_car",
    "subcompact": "small_car",
    "compact": "medium_car",
    "midsize": "large_car",
    "large": "executive_car",
    "two-seater": "sports_coupe",
    "multi_purpose_car": "minivan",
    "standard_suv": "sports_utility_car",
}

VEHICLE_TEMPLATE = {
    "mini_car": {
        # Prototype: Volkswagen Up 3-door
        # Ref 1: https://en.wikipedia.org/wiki/Volkswagen_Up
        # Ref 2: https://www.car.info/en-se/volkswagen/up/up-3-door-34696/specs
        # Ref 3: https://www.volkswagen.co.uk/assets/common/pdf/brochures/up-nf-dimensions.pdf
        "length": 3.540,
        "width": 1.641,
        "height": 1.489,
        "wheel_base": 2.420,
        "front_overhang": 0.585,
        "rear_overhang": 0.535,
        "kerb_weight": 1070,
        "max_speed": 44.44,
        "0_100_km/h": 14.4,
        "max_decel": 10.0,
        "driven_mode": "FWD",
    },
    "small_car": {
        # Prototype: Volkswagen Polo
        # Ref 1: https://en.wikipedia.org/wiki/Volkswagen_Polo#Sixth_generation_(AW/BZ;_2017)
        # Ref 2: https://www.car.info/en-se/volkswagen/polo/polo-5-door-vi-11446713/specs
        # Ref 3: https://www.youtube.com/watch?v=AkOhOJ797yA
        "length": 4.053,
        "width": 1.751,
        "height": 1.461,
        "wheel_base": 2.548,
        "front_overhang": 0.824,
        "rear_overhang": 0.681,
        "kerb_weight": 1565,
        "max_speed": 52.78,
        "0_100_km/h": 11.2,
        "max_decel": 10.0,
        "driven_mode": "FWD",
    },
    "medium_car": {
        # Prototype: Volkswagen Golf
        # Ref 1: https://en.wikipedia.org/wiki/Volkswagen_Golf
        # Ref 2: https://www.car.info/en-se/volkswagen/golf/golf-5-door-50893/specs
        # Ref 3: https://car.autohome.com.cn/config/series/871-10281.html#pvareaid=3454437
        # Ref 4: https://www.youtube.com/watch?v=m8zsh_uPVME
        "length": 4.284,
        "width": 1.799,
        "height": 1.452,
        "wheel_base": 2.637,
        "front_overhang": 0.880,
        "rear_overhang": 0.767,
        "kerb_weight": 1620,
        "max_speed": 69.44,
        "0_100_km/h": 8.9,
        "max_decel": 11.0,
        "driven_mode": "FWD",
    },
    "large_car": {
        # Prototype: Volkswagen Passat (B8) (Magotan)
        # Ref 1: https://en.wikipedia.org/wiki/Volkswagen_Passat_(B8)
        # Ref 2: https://www.car.info/en-se/volkswagen/passat/b8-52441/specs
        "length": 4.866,
        "width": 1.832,
        "height": 1.477,
        "wheel_base": 2.871,
        "front_overhang": 0.955,
        "rear_overhang": 1.040,
        "kerb_weight": 1735,
        "max_speed": 58.33,
        "0_100_km/h": 8.4,
        "max_decel": 11.0,
        "driven_mode": "FWD",
    },
    "executive_car": {
        # Prototype: Audi A6L
        # Ref 1: https://en.wikipedia.org/wiki/Audi_A6#C8_(Typ_4K,_2018%E2%80%93present)
        # Ref 2: https://www.car.info/en-se/audi/a6/specs
        # Ref 3: https://www.youtube.com/watch?v=W67gTiXB3oA
        "length": 5.050,
        "width": 1.886,
        "height": 1.475,
        "wheel_base": 3.024,
        "front_overhang": 0.921,
        "rear_overhang": 1.105,
        "kerb_weight": 2175,
        "max_speed": 63.89,
        "0_100_km/h": 8.1,
        "max_decel": 11.3,
        "driven_mode": "FWD",
    },
    "luxury_car": {
        # Prototype: Audi A8L
        # Ref 1: https://en.wikipedia.org/wiki/Audi_A8#D5_(Typ_4N,_2017%E2%80%93present)
        # Ref 2: https://www.car.info/en-se/audi/a8/d5-11462954/specs
        "length": 5.302,
        "width": 1.945,
        "height": 1.488,
        "wheel_base": 3.128,
        "front_overhang": 0.989,
        "rear_overhang": 1.185,
        "kerb_weight": 2520,
        "max_speed": 69.44,
        "0_100_km/h": 6.7,
        "max_decel": 11.3,
        "driven_mode": "AWD",
    },
    "sports_coupe": {
        # Prototype: Ford Mustang
        # Ref 1: https://en.wikipedia.org/wiki/Ford_Mustang
        # Ref 2: https://www.car.info/en-se/ford/mustang/vi-facelift-10847995/specs
        # Ref 3: https://www.youtube.com/watch?v=zd-QxcJ52YI
        "length": 4.788,
        "width": 1.916,
        "height": 1.381,
        "wheel_base": 2.720,
        "front_overhang": 0.830,
        "rear_overhang": 1.238,
        "kerb_weight": 1740,
        "max_speed": 63.89,
        "0_100_km/h": 5.3,
        "max_decel": 10.4,
        "driven_mode": "AWD",
    },
    "multi_purpose_car": {
        # Prototype: Kia Carnival
        # Ref 1: https://en.wikipedia.org/wiki/Kia_Carnival
        # Ref 2: https://www.car.info/en-se/kia/carnival/carnival-ka4-24989256/specs
        # Ref 3: https://www.youtube.com/watch?v=iLnVC76bMhs
        "length": 5.155,
        "width": 1.995,
        "height": 1.740,
        "wheel_base": 3.090,
        "front_overhang": 0.935,
        "rear_overhang": 1.130,
        "kerb_weight": 2095,
        "max_speed": 66.67,
        "0_100_km/h": 9.4,
        "max_decel": 10.3,
        "driven_mode": "4WD",
    },
    "sports_utility_car": {
        # Jeep Grand Cherokee
        # Ref 1: https://en.wikipedia.org/wiki/Jeep_Grand_Cherokee
        # Ref 2: https://www.car.info/en-se/jeep/grand-cherokee/grand-cherokee-wl-2nd-facelift-17122351/specs
        # Ref 3: https://www.youtube.com/watch?v=SJ-on34a_WM
        "length": 4.828,
        "width": 1.943,
        "height": 1.792,
        "wheel_base": 2.915,
        "front_overhang": 0.959,
        "rear_overhang": 0.954,
        "kerb_weight": 2200,
        "max_speed": 88.89,
        "0_100_km/h": 3.8,
        "max_decel": 10.29,
        "driven_mode": "4WD",
    },
}

CYCLIST_TEMPLATE = {
    "cyclist": {
        "length": 1.80,
        "width": 0.65,
        "height": 1.70,
        "max_steer": 1.05,
        "max_speed": 22.78,
        "max_accel": 5.8,
        "max_decel": 7.8,
    },
    "moped": {
        "length": 2.00,
        "width": 0.70,
        "height": 1.70,
        "max_steer": 0.35,
        "max_speed": 13.89,
        "max_accel": 3.5,
        "max_decel": 7.0,
    },
    "motorcycle": {
        "length": 2.40,
        "width": 0.80,
        "height": 1.70,
        "max_steer": 0.44,
        "max_speed": 75.00,
        "max_accel": 5.0,
        "max_decel": 10.0,
    },
}


PEDESTRIAN_TEMPLATE = {
    "adult_male": {
        "length": 0.24,
        "width": 0.40,
        "height": 1.75,
        "max_speed": 7.0,
        "max_accel": 1.5,
    },
    "adult_female": {
        "length": 0.22,
        "width": 0.37,
        "height": 1.65,
        "max_speed": 6.0,
        "max_accel": 1.5,
    },
    "children_six_year_old": {
        "length": 0.18,
        "width": 0.25,
        "height": 1.16,
        "max_speed": 3.5,
        "max_accel": 1,
    },
    "children_ten_year_old": {
        "length": 0.20,
        "width": 0.35,
        "height": 1.42,
        "max_speed": 4.5,
        "max_accel": 1.0,
    },
}

VEHICLE_MODEL = {
    "sedan": {
        # Prototype: Volkswagen Tiguan (https://en.wikipedia.org/wiki/Volkswagen_Tiguan)
        "length": 4.668,
        "width": 1.825,
        "height": 1.481,
        "wheel_base": 2.604,
        "front_overhang": 0,
        "rear_overhang": 0,
        "max_speed": 41.7,
        "max_accel": 1.98,
        "max_decel": -7.5,
        "0_100_km/h": 14,
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


def list_vehicle_templates():
    """This function prints the default vehicle templates in a table."""
    headers = [
        "Vehicle Type",
        "Length (m)",
        "Width (m)",
        "Height (m)",
        "Wheel Base (m)",
        "Front Overhang (m)",
        "Rear Overhang (m)",
        "Kerb Weight (kg)",
        "Max Speed (km/h)",
        "0-100 km/h (s)",
        "Driven Mode",
    ]
    keys = [
        "length",
        "width",
        "height",
        "wheel_base",
        "front_overhang",
        "rear_overhang",
        "kerb_weight",
        "max_speed",
        "0_100_km/h",
        "driven_mode",
    ]
    table_data = []
    for key, value in VEHICLE_TEMPLATE.items():
        line = [key]
        for key in keys:
            line.append(value[key])
        table_data.append(line)

    print(tabulate(table_data, headers=headers))


def list_cyclist_templates():
    """This function prints the default cyclist templates in a table."""
    headers = [
        "Cyclist Type",
        "Length (m)",
        "Width (m)",
        "Height (m)",
        "Max Speed (km/h)",
        "Max Steer (rad)",
        "Max Acceleration (m/s^2)",
        "Max Deceleration (m/s&2)",
    ]
    keys = ["length", "width", "height", "max_steer", "max_speed", "max_accel", "max_decel"]
    table_data = []
    for key, value in CYCLIST_TEMPLATE.items():
        line = [key]
        for key in keys:
            line.append(value[key])
        table_data.append(line)

    print(tabulate(table_data, headers=headers))


def list_pedestrian_templates():
    """This function prints the default pedestrian templates in a table."""
    headers = [
        "Pedestrian Type",
        "Length (m)",
        "Width (m)",
        "Height (m)",
        "Max Speed (km/h)",
        "Max Acceleration (m/s^2)",
    ]
    keys = ["length", "width", "height", "max_speed", "max_accel"]
    table_data = []
    for key, value in PEDESTRIAN_TEMPLATE.items():
        line = [key]
        for key in keys[1:]:
            line.append(value[key])
        table_data.append(line)

    print(tabulate(table_data, headers=headers))
