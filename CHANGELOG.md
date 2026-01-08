# Change Log

## [Unreleased]

### Added

### Changed

- Split `tactics2d.math` into `tactics2d.interpolator` and `tactics2d.geometry`.
- Refactor geometry module:
  - `Circle`: Remove `get_circle_by_three_points` and `get_circle_by_tangent_vector`. Keep functionality in `Circle.get_circle(**kwargs)`.
- Refactor interpolator module:
  - `Bezier`: Change `get_curve` method to static. Move `order` parameter from `__init__` to `get_curve`.
  - `BSpline`: Change `get_curve` method to static. Move `degree` parameter from `__init__` to `get_curve`.
  - `CubicSpline`: Change `get_curve` method to static. Move `boundary_type` parameter from `__init__` to `get_curve`.
  - `Spiral`: Rename `get_spiral` method to `get_curve`.
- Rename `tactics2d.traffic.scenario_display` to `tactics2d.sensor.matplotlib.renderer`.
- Update sensor interfaces:
  - `tactics2d.sensor.camera`: Now returns dictionary for frontend rendering.
  - `tactics2d.sensor.lidar`: Now returns dictionary for frontend rendering.
- Replace TensorFlow dependency with tfrecord for WOMD parsing:
  - Remove `tensorflow-cpu` dependency, add `tfrecord>=0.2.0`.
  - Update `WOMDParser` to use `tfrecord.tfrecord_iterator` instead of `tf.data.TFRecordDataset`.
  - Cache scenario data to avoid generator exhaustion.

### Fixed

- Improve `NuPlanParser.map_parser()` (ongoing improvements).

### Deprecated

### Removed


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
