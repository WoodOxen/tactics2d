# Release Notes

## Unreleased

### Breaking Changes

### New Features

### Bug Fixes

### Improvements

---

## Version 0.1.8 - 2025-05-22

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
- Fix pygame window unresponsive where events aren't handled

## Version 0.1.7 - 2024-05-22

### Bug Fixes

1. Traffic Event Detection
    - Fix the checking condition of the NoAction scenario event detection.
2. Map Parser
    - Fix lane parsing error in the XODR parser.
    - Remove "height" tag when parsing OSM map with Lanelet2 tag style.

### Improvements

1. Environment
    - Test the parking environment is feasible for training an agent.
2. Documentation
    - Add a tutorial for training an agent in the parking lot environment.
    - Fix the display issue of the tutorial in the GitHub page.

## Version 0.1.6 - 2024-04-01

The first release of the project.

### New Features

1. Dataset Parser

    Support parsing maps and trajectories from the following datasets:

    - HighD
    - InD
    - RounD
    - ExiD
    - Argoverse
    - Dragon Lake Parking (DLP)
    - INTERACTION
    - NuPlan
    - WOMD

2. Map Parser

    Support parsing maps in the following formats:

    - OpenStreetMap (OSM)
    - OpenStreetMap annotated in Lanelet2
    - OpenDRIVE (XODR)

3. Math Interpolation Algorithms

    Support the following interpolation algorithms:

    - B-Spline
    - Bezier
    - Cubic
    - Spiral
    - Dubins
    - Reeds Shepp

4. Traffic Participant

    The following traffic participants are implemented:

    - Vehicle
    - Cyclist
    - Pedestrian

    For each traffic participants, a set of parameters are available to configure the behavior.

5. Physics Model

    The following physics model of traffic participants are supported:

    - Bicycle model (Kinematic): recommended for cyclists and low-speed vehicles
    - Bicycle model (Dynamic): recommended for cyclists and high-speed vehicles
    - Point mass (Kinematic): recommended for pedestrians
    - Single-track drift model (Dynamic): recommended for vehicles

6. Road Element

    The following road elements are implemented:

    - Lane
    - Area
    - Junction
    - Road line
    - Base class of traffic regulations

7. Traffic Event Detection
    - Static collision detection
    - Dynamic collision detection
    - Arrival event detection

8. Sensor

    - Bird-eye-view (BEV) semantic segmentation RGB image
    - Single-line LiDAR point cloud

---

## Template

```markdown
## Version X.Y.Z - Year-Month-Date

A concise description that overviews the changes in this release.

### Breaking Changes

A complete list of breaking changes (preferably there are none, unless this is a major version).

### New Features

Describe the new feature and when/why to use it. Add some pictures! Call out any caveats/warnings? Is it a beta feature?

### Bug Fixes
Call out any existing feature/functionality that now works as intended or expected.

### Improvements
Improvements/enhancements to a workflow, performance, logging, error messaging, or user experience
```
