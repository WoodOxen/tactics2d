# Change Log

## [Unreleased]

### Added

### Changed

- Split `tactics2d.math` into `tactics2d.interpolator` and `tactics2d.geometry`.
- Parameter changes in `tactics2d.geometry`:
  - `Circle`: Remove `get_circle_by_three_points` and `get_circle_by_tangent_vector`. Keep their functionality in `Circle.get_circle(**kwargs)`.
- Parameter changes in `tactics2d.interpolator`:
  - `Bezier`: Change `get_curve` method to static method. Move `order` parameter from `__init__` to `get_curve` method.
  - `BSpline`: Change `get_curve` method to static method. Move `degree` parameter from `__init__` to `get_curve` method.
  - `CubicSpline`: Change `get_curve` method to static method. Move `boundary_type` parameter from `__init__` to `get_curve` method.
  - `Spiral`: Rename `get_spiral` method to `get_curve`.
  - `tactics2d.traffic.scenario_display` to `tactics2d.sensor.matplotlib.renderer`.
  - `tactics2d.sensor.camera` returns dictionary for frontend rendering.
  - `tactics2d.sensor.lidar` returns dictionary for frontend rendering.

### Fixed

- Improve `NuPlanParser.map_parser()` at my best, still it is a mess of shit.

### Deprecated

### Removed

### TODO

- `tactics2d.interpolator.cubic_spline`: Improve efficiency of CubicSpline by ThomasSolver.
- `tactics2d.dataset_parser.WOMDParser`: Identify the boundaries of a lane element.
- `tactics2d.dataset_parser.womd_proto`: Add compatibility to protobuf 3.x.x and 4.x.x.
- `tactics2d.map.parser.OSMParser`: Handle the tag `highway` in `load_way` for the original [OSM label style](https://wiki.openstreetmap.org/wiki/Key:lanes).
- `tactics2d.dataset_parser`: Improve the efficiency.

## [0.1.8] - 2025-05-22

### Added

- Add data parser for NGSIM.
- Add data parser for CitySim.
- Add carla sensor base class.
- Add a new tutorial for pure pursuit controller in racing environment.
- Add controller class for pure pursuit controller.
- Add Chinese README.
- Add data analysis for LevelX datasets (highD, inD, rounD, exiD, uniD) and CitySim.

### Changed

- Boost LevelX datasets by polars, 10 times faster than before.
- Move `test` to `tests` in the root directory.
- Improve map rendering speed.
- Improve the running efficiency of Bezier and b_spline interpolators by adding c++ implementation.
- Change the interface of `tactics2d.map.parser.OSMParser` and `tactics2d.map.parser.XODRParser`.

### Fixed

- Fix `type_node is none` bug
- Fix bugs in test_b_spline.py.
- Fix pygame window unresponsive where events aren't handled.

## [0.1.7] - 2024-05-22

### Added

- [#109] Tutorial for training an agent in the parking lot environment.

### Changed

- [#101] `tactics2d/.github/workflows/tag_on_PR.yml`: Change the tag trigger to `workflow_dispatch` instead of `pull_request`.
- [#109] Adjust some configurations within the parking environment.
- [#109] Improve the point generation process in Dubin's interpolator and Reeds-Shepp interpolator.

### Fixed

- [#101] `tactics2d/map/parser/parse_xodr.py`: Fix lane parsing error.
- [#101] `tactics2d/map/parser/parse_osm.py`: Remove "height" tag when parsing OSM map with Lanelet2 tag style.
- [#109] Fix the checking condition of the NoAction scenario event detection.

### Removed

- [#109] Remove `action_mask.py`, `rs_planner.py`, `train_parking_agent.py` files in the tutorial folder.

## [0.1.6] - 2024-04-01

The first release of the project.
