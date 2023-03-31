# Tactics2D

[![Codacy](https://app.codacy.com/project/badge/Grade/2bb48186b56d4e3ab963121a5923d6b5)](https://app.codacy.com/gh/WoodOxen/tactics2d/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codecov](https://codecov.io/gh/WoodOxen/tactics2d/branch/stable/graph/badge.svg?token=X81Z6AOIMV)](https://codecov.io/gh/WoodOxen/tactics2d)

![Test Modules](https://github.com/WoodOxen/tactics2d/actions/workflows/test_modules.yml/badge.svg?branch=feature-1)
[![Read the Docs](https://img.shields.io/readthedocs/tactics2d)](https://tactics2d.readthedocs.io/en/latest/)
[![Github license](https://img.shields.io/github/license/WoodOxen/tactics2d)](https://github.com/WoodOxen/tactics2d/blob/dev/LICENSE)

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

-   [HighD](https://www.highd-dataset.com/)
-   [InD](https://www.ind-dataset.com/)
-   [RounD](https://www.round-dataset.com/)
-   [INTERACTION](https://interaction-dataset.com/)
