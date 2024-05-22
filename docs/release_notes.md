# Release Notes

## Unreleased

### Breaking Changes

### New Features

### Bug Fixes

**Map Parser**

Fixed bugs in XODR and OSM parsers.

### Improvements

## Template

```markdown
## Version X.Y.Z

[Date]

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

## Version 0.1.6

> April 1st, 2024

The first release of the project.

### New Features

**Dataset Parser**

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

**Map Parser**

Support parsing maps in the following formats:

- OpenStreetMap (OSM)
- OpenStreetMap annotated in Lanelet2
- OpenDRIVE (XODR)

**Math Interpolation Algorithms**

Support the following interpolation algorithms:

- B-Spline
- Bezier
- Cubic
- Spiral
- Dubins
- Reeds Shepp

**Traffic Participant**

The following traffic participants are implemented:

- Vehicle
- Cyclist
- Pedestrian

For each traffic participants, a set of parameters are available to configure the behavior.

**Physics Model**

The following physics model of traffic participants are supported:

- Bicycle model (Kinematic): recommended for cyclists and low-speed vehicles
- Bicycle model (Dynamic): recommended for cyclists and high-speed vehicles
- Point mass (Kinematic): recommended for pedestrians
- Single-track drift model (Dynamic): recommended for vehicles

**Road Element**

The following road elements are implemented:

- Lane
- Area
- Junction
- Road line
- Base class of traffic regulations

**Traffic Event Detection**

- Static collision detection
- Dynamic collision detection
- Arrival event detection

**Sensor**

- Bird-eye-view (BEV) semantic segmentation RGB image
- Single-line LiDAR point cloud
