# Tactics2D

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/11004258066441c3a2cd47e45ca6f8f0)](https://app.codacy.com/gh/WoodOxen/tactics2d?utm_source=github.com&utm_medium=referral&utm_content=WoodOxen/tactics2d&utm_campaign=Badge_Grade)

## About

Tactics2d provides 

## Quick Start



## Development

### Folder Structure

```shell
.
├── data
│   └── maps
│       ├── configs
│       └── defaults
├── tactics2d
│   ├── common
│   ├── envs
│   ├── env_wrapper
│   ├── interface
│   ├── map_base
│   ├── map_converter
│   ├── map_parser
│   ├── object_base
│   ├── render
│   ├── utils
│   └── vehicle_kinematics
│       └── base
└── tests
```

### Map Naming

`<scenario type>_<data source>_<index>_<country>.osm/xodr`

The available scenario types include

| Notation | Scenario |
| ---------- | ---------- |
| C | Racing track, racing circuit |
| H | Highway |
| I | Intersection without traffic signs |
| P | Parking lot |
| R | Roundabout |
| SI | Intersection with traffic signs |
| U | Unknown scenario or custom scenario |

Some modified maps from the following track datasets are available in `Tactics2D`. They are provided in the Lanelet2 format.

-  [HighD](https://www.highd-dataset.com/)
-  [InD](https://www.ind-dataset.com/)
-  [RounD](https://www.round-dataset.com/)
-  [INTERACTION](https://interaction-dataset.com/)
