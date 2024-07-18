# Change Log

## [Unreleased]

### Added

### Changed

- Move `test` to `tests` in the root directory.

### Fixed

### Deprecated

### Removed

### TODO

- `tactics2d.dataset_parser.NuPlanParser`: Identify the boundaries of a lane element.
- `tactics2d.dataset_parser.WOMDParser`: Identify the boundaries of a lane element.
- `tactics2d.dataset_parser.womd_proto`: Add compatibility to protobuf 3.x.x and 4.x.x.
- `tactics2d.map.parser.OSMParser`: Handle the tag `highway` in `load_way` for the original [OSM label style](https://wiki.openstreetmap.org/wiki/Key:lanes).
- `tactics2d.dataset_parser`: Improve the efficiency.

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
