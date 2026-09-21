````markdown
# Change Log

## [Unreleased]

### Added

- Added road-segment map generation framework with structured generators (fork/merge, ramp, intersection, two-way road), shared geometry and RoadLine utilities (polyline sampling, cutting, offsetting, curvature checks), and lane-marking foundations (MUTCD/GB rule tables, lane-change permissions, rendering metadata).
- Added parser, tests, and documentation for the DriveInsightD dataset.
- Added a BITS behavior-model reproduction path for NuPlan, including the core BITS API, torch inference/training utilities, a cached planner training script, tests, and a tutorial.
- Added native SUMO `.net.xml` map parser (`NetXMLParser`) with junction geometry parsing and shape auto-completion.
- Added bidirectional converters between SUMO `.net.xml`, OpenDRIVE `.xodr`, and Lanelet2 `.osm` formats: `Net2XodrConverter`, `Xodr2NetConverter`, `Osm2XodrConverter`, `Xodr2OsmConverter`, `Net2OsmConverter`, `Osm2NetConverter`.
- Added standalone map-writer classes (`OsmWriter`, `XodrWriter`, `SumoWriter`) in `tactics2d/map/writer/`.
- Added lane-level routing module with topology-graph construction, search adapter integration, configurable cost presets, and custom cost-function injection.

### Changed

- LimSim MCTS now chains searches per decision step with a decaying iteration budget, matching the original paper's receding-horizon-within-MCTS design.
  <details><summary>Budget decay formula</summary>

  The budget decays as `budget = base / (depth / 2 + 1)`, so the immediate next action receives the most computation while distant steps get progressively fewer iterations.
  </details>
- Extracted `OsmWriter` from `Xodr2OsmConverter` as a standalone public class.
- Refactored converters (`Xodr2OsmConverter`, `Net2XodrConverter`, `Osm2XodrConverter`, `Xodr2NetConverter`) to delegate XML construction to shared writer classes, removing duplicated logic.
  <details><summary>Converter changes</summary>

  - `Xodr2OsmConverter` now reuses `XODRParser` and `OsmWriter` via the `Map` intermediate representation, removing a duplicate XML parser.
  - `Net2XodrConverter` and `Osm2XodrConverter` delegate to `XodrWriter`, removing duplicated road/lane writing methods.
  - `Xodr2NetConverter` delegates to `SumoWriter`, removing inline XML construction logic.
  </details>
- Merged `Connection` class into `Junction` by flattening its properties with default values.
- Stored original SUMO lane shapes in `NetXMLParser` `custom_tags["centerline"]` for lossless round-trip export.
- Refactored `NetXMLParser` into modular pipeline stages for improved maintainability.
- Refactored routing cost presets behind a `CostBuilder` abstraction while preserving public preset names.
- Renamed `SumoWriter` private methods to public `write_xxx` interface, consistent with `OsmWriter`.
- Improved WOMD parser support for official Motion Dataset shards.
  <details><summary>Improvements</summary>

  - Reconstruct lane sides from WOMD boundary metadata.
  - Expose driveway polygons as `drivable_area`.
  - Parse dynamic lane signal states as time-indexed `traffic_light` regulations.
  - Harden map parsing against single-point road-edge features.
  - Add official-shard parser tests and dataset support documentation.
  </details>
- Updated docstring `Example` sections across converters and writers to Google-style Markdown code blocks.

### Fixed
- Fixed CitySim OSM rendering failure by adding standard highway type color mappings.
- Fixed XML round-trip conversion bugs in the converter pipeline.
  <details><summary>Affected components</summary>

  - Fixed lane boundary direction misalignment in `Xodr2NetConverter` and `NetXMLParser` on curved roads.
  - Fixed `NetXMLParser` not reading the lane element's `width` attribute, falling back to inaccurate heuristic estimation.
  - Fixed self-intersecting offset curves in `NetXMLParser` caused by narrow lane offsets on sharp bends.
  - Fixed U-turn internal lanes (dir="T") rendering as dots due to extreme curvature collapsing the inner offset boundary.
  - Fixed `NetXMLParser` filtering out junctions without shape, causing junction count mismatch in round-trip tests.
  - Fixed `XodrWriter._get_centerline` using lane center instead of left boundary, causing gaps between adjacent lanes after round-trip conversion.
  </details>
- Fixed `XODRParser` geometry artifacts on curved roads and tight corners.
  <details><summary>Technical details</summary>

  - All `_sample_*` methods now return analytic curvature (line → 0, arc → constant, spiral → linear, poly3/paramPoly3 → Frenet-Serret formula), eliminating finite-difference estimation noise at segment boundaries.
  - Fixed `_build_offset_polyline` curvature-aware clamping from `0.99 / kappa_abs * sign(t)` to `0.99 / kappa`.
  - Fixed `_sanitise_linestring` direction-change filter threshold from `dots > -0.5` to `dots > 0.0`.
  </details>
- Fixed `XodrWriter` width polynomial domain error that caused extreme lane width values (±700 m) on roads over ~10 m.
  <details><summary>Root cause</summary>

  `XodrWriter._fit_width` fitted the width polynomial over normalized `[0, 1]` instead of real arc-length, causing `XODRParser` to evaluate the polynomial far outside its valid domain.
  </details>
- Fixed `SumoWriter` crash when `Map.boundary` is unset.
- Fixed `NetXMLParser` missing lane width attribute parsing and incorrect static method declaration.
- Fixed speed unit handling in `Net2XodrConverter` and `Xodr2NetConverter` to correctly convert between m/s and km/h.
- Fixed routing tutorial execution flow and `lane_change_penalty` parameter forwarding through `Router`.

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

### Fixed

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
````
