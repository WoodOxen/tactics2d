HIGHD_MAP_CONFIG = {
    "highD_1": {
        "name": "highD location 1",
        "osm_file": "highD_1.osm",
        "sumo_net_file": "highD_1.net.xml",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "highD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.001, 0.0],
        "trajectory_types": ["Car", "Truck"],
        # fmt: off
        "trajectory_files": [
            11, 12, 13, 14, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57
        ]
        # fmt: on
    },
    "highD_2": {
        "name": "highD location 2",
        "osm_file": "highD_2.osm",
        "sumo_net_file": "highD_2.net.xml",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "highD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.001, 0.0],
        "trajectory_types": ["Car", "Truck"],
        "trajectory_files": [1, 2, 3],
    },
    "highD_3": {
        "name": "highD location 3",
        "osm_file": "highD_3.osm",
        "sumo_net_file": "highD_3.net.xml",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "highD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.001, 0.0],
        "trajectory_types": ["Car", "Truck"],
        "trajectory_files": [4, 5, 6],
    },
    "highD_4": {
        "name": "highD location 4",
        "osm_file": "highD_4.osm",
        "sumo_net_file": "highD_4.net.xml",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "highD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.001, 0.0],
        "trajectory_types": ["Car", "Truck"],
        "trajectory_files": [7, 8, 9, 10],
    },
    "highD_5": {
        "name": "highD location 5",
        "osm_file": "highD_5.osm",
        "sumo_net_file": "highD_5.net.xml",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "highD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.001, 0.0],
        "trajectory_types": ["Car", "Truck"],
        "trajectory_files": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    },
    "highD_6": {
        "name": "highD location 6",
        "osm_file": "highD_6.osm",
        "sumo_net_file": "highD_6.net.xml",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "highD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.001, 0.0],
        "trajectory_types": ["Car", "Truck"],
        "trajectory_files": [58, 59, 60],
    },
}


IND_MAP_CONFIG = {
    "inD_1": {
        "name": "inD location 1",
        "osm_file": "inD_1.osm",
        "sumo_net_file": "inD_1.net.xml",
        "country": "DEU",
        "scenario_type": "intersection",
        "dataset": "inD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.07038, 50.782334],
        "trajectory_files": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    },
    "inD_2": {
        "name": "inD location 2",
        "osm_file": "inD_2.osm",
        "sumo_net_file": "inD_2.net.xml",
        "country": "DEU",
        "scenario_type": "intersection",
        "dataset": "inD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.101505, 50.76863],
        "trajectory_files": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    },
    "inD_3": {
        "name": "inD location 3",
        "osm_file": "inD_3.osm",
        "sumo_net_file": "inD_3.net.xml",
        "country": "DEU",
        "scenario_type": "intersection",
        "dataset": "inD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.164783, 50.77908],
        "trajectory_files": [30, 31, 32],
    },
    "inD_4": {
        "name": "inD location 4",
        "osm_file": "inD_4.osm",
        "sumo_net_file": "inD_4.net.xml",
        "country": "DEU",
        "scenario_type": "intersection",
        "dataset": "inD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.12896, 50.785635],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6],
    },
}


ROUND_MAP_CONFIG = {
    "rounD_0": {
        "name": "rounD location 0",
        "osm_file": "rounD_0.osm",
        "sumo_net_file": "rounD_0.net.xml",
        "country": "DEU",
        "scenario_type": "roundabout",
        "dataset": "rounD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.173554, 50.890924],
        # fmt: off
        "trajectory_files": [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ],
        # fmt: on
    },
    "rounD_1": {
        "name": "rounD location 1",
        "osm_file": "rounD_1.osm",
        "sumo_net_file": "rounD_1.net.xml",
        "country": "DEU",
        "scenario_type": "roundabout",
        "dataset": "rounD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.05821, 50.79120],
        "trajectory_files": [0],
    },
    "rounD_2": {
        "name": "rounD location 2",
        "osm_file": "rounD_2.osm",
        "sumo_net_file": "rounD_2.net.xml",
        "country": "DEU",
        "scenario_type": "roundabout",
        "dataset": "rounD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.10473, 50.8744],
        "trajectory_files": [1],
    },
}


EXID_MAP_CONFIG = {
    "exiD_0": {
        "name": "exiD location 0",
        "osm_file": "exiD_0.osm",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "exiD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.89314, 50.99284],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    },
    "exiD_1": {
        "name": "exiD location 1",
        "osm_file": "exiD_1.osm",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "exiD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.88829, 50.99619],
        # fmt: off
        "trajectory_files": [
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        ],
        # fmt: on
    },
    "exiD_2": {
        "name": "exiD location 2",
        "osm_file": "exiD_2.osm",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "exiD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.13736, 50.75477],
        "trajectory_files": [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
    },
    "exiD_3": {
        "name": "exiD location 3",
        "osm_file": "exiD_3.osm",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "exiD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.65025, 50.93705],
        "trajectory_files": [53, 54, 55, 56, 57, 58, 59, 60],
    },
    "exiD_4": {
        "name": "exiD location 4",
        "osm_file": "exiD_4.osm",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "exiD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.90978, 50.89751],
        "trajectory_files": [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
    },
    "exiD_5": {
        "name": "exiD location 5",
        "osm_file": "exiD_5.osm",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "exiD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.07323, 50.80543],
        "trajectory_files": [73, 74, 75, 76, 77],
    },
    "exiD_6": {
        "name": "exiD location 6",
        "osm_file": "exiD_6.osm",
        "country": "DEU",
        "scenario_type": "highway",
        "dataset": "exiD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [6.51825, 50.84766],
        "trajectory_files": [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92],
    },
}


UNID_MAP_CONFIG = {
    "uniD_0": {
        "name": "uniD location 0",
        "country": "DEU",
        "scenario_type": "intersection",
        "dataset": "uniD",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 32, "datum": "WGS84"},
        "gps_origin": [0, 0],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
}


INTERACTION_MAP_CONFIG = {
    "DR_DEU_Merging_MT": {
        "name": "DR_DEU_Merging_MT",
        "osm_file": "DR_DEU_Merging_MT.osm",
        "country": "DEU",
        "scenario_type": "",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    },
    "DR_CHN_Merging_ZS": {
        "name": "DR_CHN_Merging_ZS",
        "osm_file": "DR_CHN_Merging_ZS.osm",
        "country": "CHN",
        "scenario_type": "",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
    "DR_USA_Roundabout_SR": {
        "name": "DR_USA_Roundabout_SR",
        "osm_file": "DR_USA_Roundabout_SR.osm",
        "country": "USA",
        "scenario_type": "roundabout",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "DR_CHN_Roundabout_LN": {
        "name": "DR_CHN_Roundabout_LN",
        "osm_file": "DR_CHN_Roundabout_LN.osm",
        "country": "CHN",
        "scenario_type": "roundabout",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3, 4],
    },
    "DR_DEU_Roundabout_OF": {
        "name": "DR_DEU_Roundabout_OF",
        "osm_file": "DR_DEU_Roundabout_OF.osm",
        "country": "DEU",
        "scenario_type": "roundabout",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    "DR_USA_Roundabout_FT": {
        "name": "DR_USA_Roundabout_FT",
        "osm_file": "DR_USA_Roundabout_FT.osm",
        "country": "USA",
        "scenario_type": "roundabout",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        # fmt: off
        "trajectory_files": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45,
        ],
        # fmt: on
    },
    "DR_USA_Roundabout_EP": {
        "name": "DR_USA_Roundabout_EP",
        "osm_file": "DR_USA_Roundabout_EP.osm",
        "country": "USA",
        "scenario_type": "roundabout",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    "DR_USA_Intersection_EP0": {
        "name": "DR_USA_Intersection_EP",
        "osm_file": "DR_USA_Intersection_EP0.osm",
        "country": "USA",
        "scenario_type": "intersection",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    "DR_USA_Intersection_EP1": {
        "name": "DR_USA_Intersection_EP",
        "osm_file": "DR_USA_Intersection_EP1.osm",
        "country": "USA",
        "scenario_type": "intersection",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3, 4, 5],
    },
    "DR_USA_Intersection_MA": {
        "name": "DR_USA_Intersection_MA",
        "osm_file": "DR_USA_Intersection_MA.osm",
        "country": "USA",
        "scenario_type": "intersection",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        # fmt: off
        "trajectory_files": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        ],
        # fmt: on
    },
    "DR_USA_Intersection_GL": {
        "name": "DR_USA_Intersection_GL",
        "osm_file": "DR_USA_Intersection_GL.osm",
        "country": "USA",
        "scenario_type": "intersection",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        # fmt: off
        "trajectory_files": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        ],
        # fmt: on
    },
    "TC_BGR_Intersection_VA": {
        "name": "TC_BGR_Intersection_VA",
        "osm_file": "TC_BGR_Intersection_VA.osm",
        "country": "BGR",
        "scenario_type": "intersection",
        "dataset": "INTERACTION",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [0.0, 0.0],
        "trajectory_files": [0, 1, 2, 3],
    },
}

DLP_MAP_CONFIG = {
    "DLP": {
        "name": "Dragon Lake Parking Lot",
        "osm_file": "DLP.osm",
        "country": "USA",
        "scenario_type": "parking",
        "dataset": "DLP",
        "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
        "gps_origin": [-1.4887438843872076, 0],
        # fmt: off
        "trajectory_files": [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ],
        # fmt: on
    }
}

NUPLAN_MAP_CONFIG = {
    "sg-one-north": {
        "name": "sg-one-north",
        "gpkg_file": "sg-one-north.gpkg",
        "country": "SGP",
        "scenario_type": "urban",
        "dataset": "NuPlan",
        "project_rule": {},
        "gps_origin": [0, 0],
    },
    "us-ma-boston": {
        "name": "us-ma-boston",
        "osm_file": "",
        "gpkg_file": "us-ma-boston.gpkg",
        "country": "USA",
        "scenario_type": "urban",
        "dataset": "NuPlan",
        "project_rule": {},
        "gps_origin": [0, 0],
    },
    "las_vegas": {
        "name": "us-nv-las-vegas-strip",
        "osm_file": "",
        "gpkg_file": "us-nv-las-vegas-strip.gpkg",
        "country": "USA",
        "scenario_type": "urban",
        "dataset": "NuPlan",
        "project_rule": {},
        "gps_origin": [0, 0],
    },
    "us-pa-pittsburgh-hazelwood": {
        "name": "us-pa-pittsburgh-hazelwood",
        "osm_file": "",
        "gpkg_file": "us-pa-pittsburgh-hazelwood.gpkg",
        "country": "USA",
        "scenario_type": "urban",
        "dataset": "NuPlan",
        "project_rule": {},
        "gps_origin": [0, 0],
    },
}
