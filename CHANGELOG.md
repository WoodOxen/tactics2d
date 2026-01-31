# Change Log

## [0.1.9rc3] - 2026-01-29

### Added

- Added classic search algorithms: Dijkstra, A*, D*, Hybrid A*, RRT, RRT*, RRTConnect.
- Added tests for classic search algorithms.
- Added tutorial for classic search algorithms in grid world environment.
- Added claude documents and ruls to boost development efficiency.

### Changed

- Fixed dependency vulnerability issue of protobuf.

## [0.1.9rc1] - 2026-01-09

### Added

- Add `defusedxml` dependency for enhanced XML parsing safety.
- Add CI tests for multiple operating systems.

### Changed

- Split `tactics2d.math` module into `tactics2d.interpolator` and `tactics2d.geometry`.
- Refactor geometry module:
  - Remove `get_circle_by_three_points` and `get_circle_by_tangent_vector` methods from `Circle` class.
  - Consolidate functionality into `Circle.get_circle(**kwargs)` method.
- Refactor interpolator module:
  - Change `get_curve` method to static for `Bezier`, `BSpline`, and `CubicSpline` classes.
  - Move `order` parameter from `__init__` to `get_curve` method in `Bezier` class.
  - Move `degree` parameter from `__init__` to `get_curve` method in `BSpline` class.
  - Move `boundary_type` parameter from `__init__` to `get_curve` method in `CubicSpline` class.
  - Rename `get_spiral` method to `get_curve` in `Spiral` class.
- Rename `tactics2d.traffic.scenario_display` module to `tactics2d.sensor.matplotlib.renderer`.
- Update sensor interfaces to return dictionaries for frontend rendering:
  - Update `tactics2d.sensor.camera` interface.
  - Update `tactics2d.sensor.lidar` interface.
- Replace TensorFlow dependency with `tfrecord` for WOMD parsing:
  - Remove `tensorflow-cpu` dependency.
  - Add `tfrecord>=0.2.0` dependency.
  - Update `WOMDParser` to use `tfrecord.tfrecord_iterator` instead of `tf.data.TFRecordDataset`.
  - Cache scenario data to avoid generator exhaustion.
- Change headers to follow PEP format.
- Improve code formatting and remove unused imports.
- Correct header version and formats.
- Change matplotlib backend to Agg for non-interactive environments.
- Update version to 0.1.9rc1.
- Normalize Python file header descriptions.
- Update copyright years based on git history.

### Fixed

- Improve `NuPlanParser.map_parser()` method (ongoing improvements).

### Removed

- Remove test dependency on bezier library.

## [0.1.8] - 2025-05-22

### Added

- Add NGSIM data parser.
- Add CitySim data parser.
- Add Carla sensor base class.
- Add pure pursuit controller class.
- Add tutorial for pure pursuit controller in racing environment.
- Add Chinese README documentation.
- Add data analysis for LevelX datasets (highD, inD, rounD, exiD, uniD) and CitySim.

### Changed

- Improve performance of LevelX datasets processing using polars (10x faster).
- Move `test` directory to `tests` in root directory.
- Improve map rendering speed.
- Improve performance of Bezier and B-spline interpolators with C++ implementation.
- Change interface of `tactics2d.map.parser.OSMParser` and `tactics2d.map.parser.XODRParser`.

### Fixed

- Fix `type_node is None` bug.
- Fix bugs in `test_b_spline.py`.
- Fix pygame window unresponsiveness when events are not handled.

## [0.1.7] - 2024-05-22

### Added

- Add tutorial for training an agent in parking lot environment.

### Changed

- Change tag trigger from `pull_request` to `workflow_dispatch` in `tactics2d/.github/workflows/tag_on_PR.yml`.
- Adjust configurations in parking environment.
- Improve point generation process in Dubins and Reeds-Shepp interpolators.

### Fixed

- Fix lane parsing error in `tactics2d/map/parser/parse_xodr.py`.
- Remove "height" tag when parsing OSM map with Lanelet2 tag style in `tactics2d/map/parser/parse_osm.py`.
- Fix checking condition of NoAction scenario event detection.

### Removed

- Remove `action_mask.py`, `rs_planner.py`, and `train_parking_agent.py` files from tutorial folder.

## [0.1.6] - 2024-04-01

The first release of the project.
