````markdown
# Change Log

## [0.1.9] - 2026-09-24

### Added

- Added BITS, SMART, InterSim, and LimSim behavior models with shared rollout interfaces, training and inference utilities, tests, and tutorials.
- Added a structured road-segment generation framework covering intersections, ramps, fork/merge sections, roundabouts, and asymmetric junction approaches, with configurable lane-marking rules.
- Added native SUMO `.net.xml` parsing, standalone OSM/OpenDRIVE/SUMO writers, and bidirectional conversion between SUMO, OpenDRIVE, and Lanelet2 map formats.
- Added lane-level routing with topology construction and configurable costs, plus Dijkstra, A*, D*, Hybrid A*, MCTS, PRM, and RRT-family search algorithms.
- Added the DriveInsightD parser and expanded NuPlan and WOMD support with route extraction, official-shard compatibility, richer map semantics, and dataset tutorials.
- Added `ControllerBase`, `PIDController`, and `IDMController` with aligned controller interfaces.
- Added a unified display system with Matplotlib, Pygame, and browser backends, including live previews and broad dataset visualization support.

### Changed

- Split `tactics2d.math` into the public `tactics2d.geometry` and `tactics2d.interpolator` packages, with updated curve APIs and expanded C++ acceleration.
- Refactored display and sensor APIs around reusable renderers and dictionary-based frontend data, replacing the legacy scenario-display path.
- Refactored map conversion around the shared `Map` representation and public writers, folded connection data into `Junction`, and moved routing costs behind `CostBuilder` without changing preset names.
- Improved simulation performance and single-track vehicle-model fidelity, while updating LimSim's MCTS decision process to use chained receding-horizon searches.
- Replaced TensorFlow with `tfrecord` for WOMD parsing and added protobuf 3.x/4.x compatibility.

### Fixed

- Fixed map parsing and round-trip conversion issues involving curved-road geometry, junctions, lane widths, speed units, U-turns, and missing map bounds.
- Fixed CitySim rendering, WOMD edge cases, NuPlan map parsing, and routing parameter forwarding.
- Fixed protobuf dependency vulnerabilities and hardened XML parsing with `defusedxml`.

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
````
