# Release Notes

## Known Issues

### Unfinished Features

- `tactics2d.dataset_parser.NuPlanParser`: Identify the boundaries of a lane element.
- `tactics2d.dataset_parser.WOMDParser`: Identify the boundaries of a lane element.
- `tactics2d.dataset_parser.womd_proto`: Add compatibility to protobuf 3.x.x and 4.x.x.
- `tactics2d.map.parser.OSMParser`: Handle the tag `highway` in `load_way` for the original [OSM label style](https://wiki.openstreetmap.org/wiki/Key:lanes).

### Bugs

- `test_math_interpolate.test_b_spline`: Random ValueError raised due to wrong input to `scipy.interpolate.BSpline`.

## Version 1.0.0 -- 2024-03-14

### Overview

### Features
