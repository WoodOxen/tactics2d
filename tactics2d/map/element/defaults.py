LEGAL_SPEED_UNIT = ["km/h", "mi/h", "m/s"]

LANE_CONFIG = {
    "road": {
        "inferred_participants": [],
        "speed_limit": 20,
        "speed_limit_unit": "km/h",
    },
    "highway": {
        "inferred_participants": [],
        "speed_limit": 35,
        "speed_limit_unit": "km/h",
    },
    "play_street": {
        "inferred_participants": [],
        "speed_limit": 15,
        "speed_limit_unit": "km/h",
    },
    "emergency_lane": {
        "inferred_participants": [],
        "speed_limit": 35,
        "speed_limit_unit": "km/h",
    },
    "bus_lane": {
        "inferred_participants": [],
        "speed_limit": 15,
        "speed_limit_unit": "km/h",
    },
    "bicycle_lane": {
        "inferred_participants": [],
        "speed_limit": 10,
        "speed_limit_unit": "km/h",
    },
    "exit": {
        "inferred_participants": [],
        "speed_limit": 5,
        "speed_limit_unit": "km/h",
    },
    "walkway": {
        "inferred_participants": [],
        "speed_limit": 5,
        "speed_limit_unit": "km/h",
    },
    "shared_walkway": {
        "inferred_participants": [],
        "speed_limit": 10,
        "speed_limit_unit": "km/h",
    },
    "crosswalk": {
        "inferred_participants": [],
        "speed_limit": 5,
        "speed_limit_unit": "km/h",
    },
    "stairs": {
        "inferred_participants": [],
        "speed_limit": 5,
        "speed_limit_unit": "km/h",
    },
}


LANE_CHANGE_MAPPING = {
    "line_thin": {
        "solid": (False, False),
        "solid_solid": (False, False),
        "dashed": (True, True),
        "dashed_solid": (True, False), # left->right: yes
        "solid_dashed": (False, True), # right->left: yes
    },
    "line_thick": {
        "solid": (False, False),
        "solid_solid": (False, False),
        "dashed": (True, True),
        "dashed_solid": (True, False), # left->right: yes
        "solid_dashed": (False, True), # right->left: yes
    },
    "curbstone": {
        "high": (False, False),
        "low":  (False, False),
    }
}