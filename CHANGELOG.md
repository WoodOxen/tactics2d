# Change Log

## [Unreleased]

### Added

- Added a parser and corresponding tests, documentations for DriveInsightD dataset.
- Added native SUMO `.net.xml` map parser (`NetXMLParser`) with junction geometry parsing, connection attachment, and junction shape auto-completion via convex hull.
- Merged `Connection` class into `Junction` by flattening its properties directly into `Junction` with default values.
- Added `Net2XodrConverter` for converting SUMO `.net.xml` maps to OpenDRIVE `.xodr` format.
- Added `Xodr2NetConverter` for converting OpenDRIVE `.xodr` maps to SUMO `.net.xml` format.
- Added lane-level routing module with topology-graph construction, search adapter integration, route containers, and WOMD tutorial notebook.
- Added `Osm2XodrConverter` for converting Lanelet2 `.osm` maps to OpenDRIVE `.xodr` format, with topology-aware predecessor/successor link generation and junction detection.
- Added configurable routing cost presets and custom cost-function injection for lane-level routing, including classic distance/time baselines and source-inspired Lanelet2/Apollo variants.
- Added `Xodr2OsmConverter` for converting OpenDRIVE `.xodr` maps to Lanelet2-annotated `.osm` format via the `XODRParser` → `Map` → `OsmWriter` pipeline, with roadMark-to-subtype mapping and speed limit regulatory element export.
- Added `OsmWriter` as a standalone public class in `tactics2d/map/writer/` for writing a Tactics2D `Map` to Lanelet2 OSM XML, with public `write_nodes`, `write_way`, `write_boundary_ways`, `write_lanelet_relation`, and `write_speed_regulatory` methods.
- Added `XodrWriter` as a standalone public class in `tactics2d/map/writer/` for writing a Tactics2D `Map` to OpenDRIVE `.xodr` XML, with topology inference via lane endpoint proximity and lane width fitted as a cubic polynomial over real arc-length.
- Added `SumoWriter` as a standalone public class in `tactics2d/map/writer/` for writing a Tactics2D `Map` to SUMO `.net.xml` XML, grouping lanes by `sumo_id` edge prefix and supporting lossless centre-line export via `custom_tags["centerline"]`.

### Fixed

- Fixed lane boundary direction misalignment in `Xodr2NetConverter` and `NetXMLParser` on curved roads.
- Fixed backtrack points in lane boundary geometry produced by `XODRParser` on tight curves via direction-change filtering in `_sanitise_linestring`.
- Fixed self-intersecting offset curves in `NetXMLParser` caused by narrow lane offsets on sharp bends.
- Fixed routing tutorial notebook execution flow and route visualization output for WOMD examples.
- Fixed unified routing cost parameter forwarding so `lane_change_penalty` consistently reaches Lanelet2-style and Apollo-inspired presets through `Router`.
- Fixed `XODRParser` offset geometry on curved roads: all `_sample_*` methods now return analytic curvature alongside sampled points (`line` → 0, `arc` → constant, `spiral` → linear, `poly3`/`paramPoly3` → Frenet-Serret formula), eliminating finite-difference estimation noise at segment boundaries that caused offset points to deviate by hundreds of metres on roundabout geometries.
- Fixed `_build_offset_polyline` curvature-aware clamping: corrected `0.99 / kappa_abs * sign(t)` to `0.99 / kappa`, ensuring the collapse boundary is computed with the correct sign for both left and right offsets.
- Fixed `XodrWriter._fit_width` width polynomial fitted over normalised `[0, 1]` instead of real arc-length, causing `XODRParser` to evaluate the polynomial far outside its valid domain and produce lane widths of ±700 m on roads longer than ~10 m.
- Fixed `_sanitise_linestring` direction-change filter threshold from `dots > -0.5` to `dots > 0.0`, retaining all geometrically valid curved segments while still removing U-turn backtrack artefacts.

### Changed

- Extracted `OsmWriter` from `Xodr2OsmConverter` into `tactics2d/map/writer/osm_writer.py` as a standalone public class with full Google-style docstrings and type annotations.
- Refactored `Xodr2OsmConverter` to reuse `XODRParser` and `OsmWriter` via the `Map` intermediate representation, removing the duplicate `_XodrReader` XML parser, the `_LaneGeom` intermediary struct, and the redundant geometry helper functions.
- Refactored `Net2XodrConverter` and `Osm2XodrConverter` to delegate XML construction to `XodrWriter`, removing duplicated `_write_plan_view`, `_write_lanes`, and related private methods.
- Refactored `Xodr2NetConverter` to delegate XML construction to `SumoWriter`, removing inline XML construction logic.
- Stored original SUMO lane `shape` in `NetXMLParser` `custom_tags["centerline"]` for lossless centre-line export to xodr and net.xml without re-deriving from offset boundaries.
- Updated docstring `Example` sections across converter and writer classes to Google-style Markdown code blocks.
- Improved WOMD parser support for official Motion Dataset shards:
  - reconstruct lane sides from WOMD boundary metadata,
  - expose driveway polygons as `drivable_area`,
  - parse dynamic lane signal states as time-indexed `traffic_light` regulations,
  - harden map parsing against single-point road-edge features,
  - add official-shard parser tests and dataset support documentation.
- Fixed speed unit handling in `Net2XodrConverter` and `Xodr2NetConverter` to correctly convert between m/s internal storage and km/h xodr output.

## [0.1.9rc3] - 2026-01-29

### Added

- Added ControlBase, PIDController, and IDMController
- Added tests for controllers.
- Added classic search algorithms: Dijkstra, A*, D*, Hybrid A*, RRT, RRT*, RRTConnect.
- Added tests for classic search algorithms.
- Added tutorial for classic search algorithms in grid world environment.
- Added claude documents and ruls to boost development efficiency.

### Changed

- Aligned interface within controller module.
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
