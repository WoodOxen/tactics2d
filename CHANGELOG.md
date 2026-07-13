````markdown
# Change Log

## [Unreleased]

### Added

- Added NuPlan support to the browser frontend: NuPlan sqlite logs are discovered from the data root (`nuPlan/data/cache/<split>/*.db` or the nuplan-devkit layout) and appear in the dataset dropdown grouped by split, scenes stream through the same preview pipeline (`NuPlanParser` trajectories, city map auto-resolved from the log via `NUPLAN_MAP_CONFIG`, camera following the longest-lived vehicle), registered geopackage city maps appear in the map form with a capped city-center preview window, and `tactics2d preview dataset --dataset NuPlan` works from the CLI. UTM-scale NuPlan coordinates are shifted to a local origin at the map center before rendering, since float32 resolution at that magnitude (0.5 m at 4.7e6 m) would visibly deform geometry in the browser.
- Added NGSIM, INTERACTION, DLP, CitySim, Argoverse 2, and DriveInsightD to the browser frontend dataset preview, covering every trajectory parser except WOMD: each family is discovered from its official layout under the data root, streams through a shared stamped-scene pipeline (observed frame stamps, automatic origin shift for global coordinates, string participant ids renumbered for the renderer with the original kept as `source_id`), and resolves its map when one ships with the dataset (INTERACTION scenario OSM, DLP parking-lot OSM, Argoverse 2 per-scenario map json, DriveInsightD sibling OpenDRIVE); CitySim renders trajectory-only scenes; NGSIM draws the street-centerline shapefiles shipped in its per-location `gis-files` folders as a base map (state-plane feet converted like the trajectories, clipped to the recording's extent), with headings derived from motion since the source data carries none.
- Added WOMD to the browser frontend dataset preview, completing coverage of every trajectory parser: tfrecord shards are discovered under the data root, shards up to 32 MB are enumerated per scenario as `<shard>/<scenario_id>` dropdown entries (bigger shards appear as one entry and preview their first scenario), and scenes stream the per-scenario vector map with the trajectories through the shared stamped-scene pipeline; `tactics2d preview dataset --dataset WOMD` works from the CLI.
- Added rendering of `Other` participants (traffic cones, barriers, construction-zone signs, generic objects) to the BEV camera and browser frontend: they draw as small type-colored rectangles when the dataset provides dimensions and as dots otherwise, so nuPlan construction zones now show their cones and barriers instead of skipping them.
- Added support for multiple environments streaming concurrently to one browser frontend server: frame acknowledgements are keyed by a server-wide publish sequence instead of publisher frame ids (which collide when every environment numbers from zero), the browser coalesces pending frames per stream so a fast publisher cannot starve a slower one, and `BrowserBackend` accepts `sensor_id`/`exclusive` so each environment renders into its own view and removes only that view on close. Verified end to end with two 20 fps publisher processes rendering side by side with zero drops.
- Added sustained 100 Hz live streaming to the browser frontend: `FrontendRenderer` now reuses one keep-alive HTTP connection (a fresh TCP handshake per frame cost tens of milliseconds), encodes frames with orjson when available, and paces against the previous send start instead of its completion (the old anchor added the request time to every interval, capping the rate below `max_fps`). Measured on loopback: 99.6 fps sustained with 100 participants and 400 road elements, sub-millisecond HTTP round-trip; `wait_ack=True` intentionally settles at the browser's display refresh rate.
- Added a remote-mode section to the browser frontend tutorial (EN/ZH): binding with `--host 0.0.0.0`, connecting publishers and browsers by server address, and a security note recommending trusted intranets or SSH tunnels since the server has no authentication. Verified end to end over a LAN address (server on `0.0.0.0`, browser and a 60-frame publisher both connecting via the machine's LAN IP).
- Added `imageio-ffmpeg` as a core dependency so the screen-recording compatibility mode works out of the box without a system ffmpeg.
- Added recording capabilities to the browser frontend: a screen-recording button that captures the composited sensor views into a downloadable video via MediaRecorder (MP4/H.264 when the browser supports it, WebM fallback), and server-side frame recording/replay (`POST /api/record/start|stop`, `GET /api/recordings`, `POST /api/preview/replay`) that stores published frame payloads as JSONL (seeded with the current scene snapshot) under `~/.cache/tactics2d/recordings/` (`TACTICS2D_RECORD_DIR` to override) and replays them through the preview pipeline; documented the parallel Python-side `GifRecorder` offline rendering pattern.
- Added dataset and map discovery to the browser frontend: a configurable data root (`tactics2d start --data-root`, the `TACTICS2D_DATA_ROOT` environment variable, or the `./data` repository convention) is scanned for LevelX recordings in the official layout and for available OSM maps, and the browser forms now use two-level selection-based interaction (dataset → recording with recordings grouped by registered map location, and dataset → map) with automatic folder and map resolution instead of manual path input; manual path fields remain available under the advanced section.
- Refined structured road-segment generators, including fork/merge, ramp, intersection, two-way road, and related test coverage.
- Updated generator rules and geometry helpers to improve roadline metadata handling, reference-line construction, and module connection consistency.
- Added shared geometry and RoadLine utilities for road-segment map generation, including polyline sampling, cutting, offsetting, intersection extraction, curvature checks, and marking metadata support.
- Added lane-marking foundations for map generation, including marking token specifications, MUTCD/GB rule tables, lane-change permissions, rendering metadata, and shared road module socket/result types.
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
- Added `Net2OsmConverter` for converting SUMO `.net.xml` maps to Lanelet2-annotated `.osm` format via the `NetXMLParser` → `Map` → `OsmWriter` pipeline.
- Added `Osm2NetConverter` for converting Lanelet2-annotated `.osm` maps to SUMO `.net.xml` format via the `OSMParser` → `Map` → `SumoWriter` pipeline.

### Fixed

- Fixed `Argoverse2Parser` crashing with `KeyError: 'construction'` on real motion-forecasting scenarios: the `construction` and `unknown` object categories were missing from the type mappings, and unseen future categories now degrade to `Other` instead of aborting the parse. Found by loading official validation scenarios from the public Argoverse S3 bucket.
- Fixed screen recordings failing to play in strict players (e.g. GNOME Videos): with the compatibility-mode toggle enabled (default), captures are finalized server-side via ffmpeg (`POST /api/record/export`) into constant-frame-rate H.264 MP4, since raw MediaRecorder output declares a variable (0/1) frame rate; unticking the toggle (or a server without ffmpeg) downloads the raw recording unchanged.
- Fixed compatibility-mode recordings crashing hardware-accelerated players (GStreamer VA-API heap corruption, observed in GNOME Videos) when the capture width is not a multiple of 4: the ffmpeg finalize now pads output to 4-pixel-aligned dimensions, the capture canvas is aligned client-side, and a failed transcode surfaces a status message instead of silently downloading the raw file.
- Fixed dataset previews aborting with "Type is not JSON serializable: numpy.float64" once a pedestrian enters the scene (e.g. inD): pedestrian positions from the BEV camera carried numpy scalars, and the frontend server serialization now tolerates numpy scalars/arrays in any payload (matching the client-side renderer). The map-preview camera position (derived from the map boundary) is cast for the same reason.
- Fixed exiD dataset previews failing to resolve their maps: the official levelXdata layout (`<dataset>/maps/lanelet2/<location>_<site>.osm`) is now searched, matching registered configs by location number.
- Fixed ghost trails behind moving vehicles in screen recordings: the browser's realtime H.264 encoder leaves uncorrected residual blocks on flat scenes, so the capture now prefers VP9 with an explicit resolution-scaled bitrate (raw downloads are WebM on most browsers, MP4 on Safari; compatibility mode still produces H.264 MP4), and the server transcode quality was raised (CRF 20 → 18).
- Fixed browser frontend layout selection being overridden on every frame during demo/dataset/live streaming; a manual layout choice (toolbar button or `POST /api/layout`) now takes precedence over the per-frame payload default.
- Fixed `POST /api/preview/map` raising an unhandled HTTP 500 with no UI feedback when the OSM path is missing or invalid; it now returns HTTP 400 with an error message that persists in the status bar.
- Fixed browser frontend favicon 404 console error by adding an inline SVG favicon.
- Removed a machine-specific hardcoded dataset folder from the browser frontend preview defaults; defaults are now derived from data-root discovery.
- Fixed `NetXMLParser._get_lane_subtype` incorrectly declared as `@staticmethod` with a `self` parameter, causing all lane parsing to fail silently.
- Fixed `NetXMLParser._offset_line` referencing undefined normal vector variables when consecutive points have zero distance.
- Fixed `NetXMLParser` not reading the lane element's `width` attribute, falling back to inaccurate heuristic estimation.
- Fixed `SumoWriter` failing when `Map.boundary` is None by adding automatic boundary computation from lane geometries.
- Fixed `XodrWriter._get_centerline` using lane center as XODR reference line instead of left boundary, causing gaps between adjacent lanes after round-trip conversion.
- Fixed `NetXMLParser` filtering out junctions without shape, causing junction count mismatch in round-trip conversion tests.
- Fixed U-turn internal lanes (dir="T") rendering as dots due to extreme curvature collapsing the inner offset boundary; these lanes are now excluded during parsing.
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
- Refactored `NetXMLParser` into modular pipeline stages (`_parse_location`, `_build_edge_junction_map`, `_parse_edges`, `_parse_junctions`, `_parse_connections`, `_compute_junction_shapes`) for improved readability and maintainability.
- Changed `SumoWriter` method naming from `_write_xxx` private pattern to direct public `write_xxx` methods, consistent with `OsmWriter` style.
- Refactored routing cost presets behind a `CostBuilder` abstraction while preserving the public preset names and custom cost-function support.
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
````
