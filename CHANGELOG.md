# Change Log

## [Unreleased]

> [Date]

### Added

### Changed

- [#101] `tactics2d/.github/workflows/tag_on_PR.yml`: Change the tag trigger to `workflow_dispatch` instead of `pull_request`.

### Fixed

- [#101] `tactics2d/map/parser/parse_xodr.py`: Fix lane parsing error.
- [#101] `tactics2d/map/parser/parse_osm.py`: Remove "height" tag when parsing OSM map with Lanelet2 tag style.

### Deprecated

### Removed

### TODO

- `tactics2d.dataset_parser.NuPlanParser`: Identify the boundaries of a lane element.
- `tactics2d.dataset_parser.WOMDParser`: Identify the boundaries of a lane element.
- `tactics2d.dataset_parser.womd_proto`: Add compatibility to protobuf 3.x.x and 4.x.x.
- `tactics2d.map.parser.OSMParser`: Handle the tag `highway` in `load_way` for the original [OSM label style](https://wiki.openstreetmap.org/wiki/Key:lanes).

## Version 0.1.6

> April 1st, 2024

The first release of the project.
