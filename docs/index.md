![Tactics2D LOGO](https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/Tactics_LOGO_long.jpg)

[![Codacy](https://app.codacy.com/project/badge/Grade/2bb48186b56d4e3ab963121a5923d6b5)](https://app.codacy.com/gh/WoodOxen/tactics2d/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codecov](https://codecov.io/gh/WoodOxen/tactics2d/graph/badge.svg?token=X81Z6AOIMV)](https://codecov.io/gh/WoodOxen/tactics2d)
![Test Modules](https://github.com/WoodOxen/tactics2d/actions/workflows/test_modules.yml/badge.svg?)
[![Read the Docs](https://img.shields.io/readthedocs/tactics2d)](https://tactics2d.readthedocs.io/en/latest/)
[![Downloads](https://img.shields.io/pypi/dm/tactics2d)](https://pypi.org/project/tactics2d/)

![python-version](https://camo.githubusercontent.com/2b53588bcdf5ca9bcfc10921eb80d43a1e2d52e5a4ede24273800a5074a0916d/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f67796d6e617369756d2e737667)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Github license](https://img.shields.io/github/license/WoodOxen/tactics2d)](https://github.com/WoodOxen/tactics2d/blob/dev/LICENSE)

## About

Welcome to the official documentation of Python Library tactics2d!

`tactics2d` is an open-source Python library that provides diverse and challenging traffic scenarios for the development and evaluation of reinforcement learning-based decision-making models in autonomous driving. `tactics2d` stands out with the following key features:

- **Compatibility**
  - 📦 Trajectory dataset -- Enables seamless importation of various real-world trajectory datasets, including Argoverse, Dragon Lake Parking (DLP), INTERACTION, LevelX Series (highD, inD, rounD, ExiD), NuPlan, and Waymo Open Motion Dataset (WOMD), encompassing both trajectory parsing and map information.
  - 📄 Map format -- Enables parsing and conversion of commonly used open map formats like OpenDRIVE, Lanelet2-style OpenStreetMap (OSM), and SUMO roadnet.
- **Customizability**
  - 🚗 Traffic participants -- Supports the creation of new traffic participant classes with customizable physical attributes, physics dynamics/kinematics models, and behavior models.
  - 🚧 Road elements -- Support the definition of new road elements, with a focus on regulatory aspects.
- **Diversity**
  - 🛣️ Traffic scenarios -- Features an extensive range of built-in Gym-style traffic scenarios, including highway, lane-merging, unsignalized/signalized intersection, roundabout, parking, and racing.
  - 🚲 Traffic participants -- Features a variety of built-in traffic participants with realistic physics parameters, detailed further [here](https://tactics2d.readthedocs.io/en/latest/api/participant/#templates-for-traffic-participants).
  - 📷 Sensors -- Provides bird-eye-view (BEV) semantic segmentation RGB image and single-line LiDAR point cloud for model input.
- **Visualization** -- Offers a user-friendly visualization tool for real-time rendering of traffic scenarios and participants, along with the capability to record and replay traffic scenarios.
- **Reliability** -- Over 85\% code is covered by unit tests and integration tests.

## What can you do with `tactics2d`?

[Description]

### Features

> Updated on April 1, 2024.
>
> Corresponds to version 0.1.6.

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

## Why `tactics2d`?

### Similar Works

`tactics2d` is crafted to offer a robust and intuitive environment tailored for the development and evaluation of autonomous driving decision-making models. As a third-party library, `tactics2d` does not cater to any specific dataset; instead, its focus lies in facilitating parsing, visualization, log replaying, and interactive simulation across a diverse array of datasets and map formats. The table below provides a comparative overview of `tactics2d` alongside other open-source simulators under active maintenance.

> These tables are updated on 2024-04-01.
> Notations:
> :white_check_mark: = Implemented and tested
> :construction: = Under development
> `-` = Not implemented and not planned

### Functionality

| Simulator | Built-in RL Environment | Custom Trajectory | Custom Map | Dataset Compatibility | Interactive NPCs | Multi-agent |
| --- | --- | --- | --- | --- | --- | --- |
| [SUMO](https://eclipse.dev/sumo/) | - | :white_check_mark: | :white_check_mark: | - | :white_check_mark: | - |
| [CarRacing](https://gymnasium.farama.org/environments/box2d/car_racing/) | :white_check_mark: | - | - | - | - | - |
| [CARLA](https://carla.org) | - | :white_check_mark: | :white_check_mark: | - | :white_check_mark: | :white_check_mark: |
| [CommonRoad](https://commonroad.in.tum.de/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | - |
| [highway-env](https://github.com/Farama-Foundation/HighwayEnv) | :white_check_mark: | - | - | - | :white_check_mark: | - |
| [SMARTS](https://github.com/huawei-noah/SMARTS) | :white_check_mark: | - | - | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [MetaDrive](https://github.com/metadriverse/metadrive) | :white_check_mark: | :white_check_mark: | :white_check_mark: | - | :white_check_mark: | :white_check_mark: |
| [NuPlan](https://github.com/motional/nuplan-devkit) | - | - | - | - | :white_check_mark: | - |
| [InterSim](https://github.com/Tsinghua-MARS-Lab/InterSim) | - | - | - | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [TBSim](https://github.com/NVlabs/traffic-behavior-simulation) | - | - | - | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Waymax](https://github.com/waymo-research/waymax) | :white_check_mark: | - | - | - | :white_check_mark: | :white_check_mark: |
| **Tactics2D** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :construction: | :construction: |

### Dataset Compatibility

`tactics2d` excels in parsing various datasets into unified map and traffic participant data structures, enabling seamless integration for both log replay and interactive simulations.

Below is a comparison of dataset support between `tactics2d` and other simulators. `tactics2d` strives to accommodate a wide range of datasets, and we commit to keeping the table updated on a regular basis.

!!! note "TODO"
    We have a plan to add support to the following datasets in the future:

    - NGSIM
    - Lyft 5

Feel free to suggest additional trajectory datasets to be incorporated into our support list by either [opening an issue](https://github.com/WoodOxen/tactics2d/issues) or submitting a pull request. We value community input and are committed to expanding our dataset coverage to better serve our users.

| Simulators | [Argoverse](https://www.argoverse.org/av2.html#forecasting-link) | [DLP](https://sites.google.com/berkeley.edu/dlp-dataset) | [INTERACTION](https://interaction-dataset.com/) | [LevelX](https://www.highd-dataset.com/) | [NuPlan](https://www.nuscenes.org/nuplan) | [WOMD](https://waymo.com/open/about/) |
| --- | --- | --- | --- | --- | --- | --- |
| [SUMO](https://eclipse.dev/sumo/) | - | - | - | - | - | - |
| [CarRacing](https://gymnasium.farama.org/environments/box2d/car_racing/) | - | - | - | - | - | - |
| [CARLA](https://carla.org) | - | - | - | - | - | - |
| [CommonRoad](https://commonroad.in.tum.de/) | - | - | :white_check_mark: | :white_check_mark: | - | - |
| [highway-env](https://github.com/Farama-Foundation/HighwayEnv) | - | - | - | - | - | - |
| [SMARTS](https://github.com/huawei-noah/SMARTS) | - | - | - | - | - | - |
| [MetaDrive](https://github.com/metadriverse/metadrive) | - | - | - | - | - | - |
| [NuPlan](https://github.com/motional/nuplan-devkit) | - | - | - | - | :white_check_mark: | - |
| [InterSim](https://github.com/Tsinghua-MARS-Lab/InterSim) | - | - | - | - | :white_check_mark: | :white_check_mark: |
| [TBSim](https://github.com/NVlabs/traffic-behavior-simulation) | - | - | - | - | :white_check_mark: | :white_check_mark: |
| [Waymax](https://github.com/waymo-research/waymax) | - | - | - | - | - | :white_check_mark: |
| **Tactics2D** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

### Map Format Compatibility

| Simulators | OpenDRIVE | OpenStreetMap | SUMO Roadnet |
| --- | --- | --- | --- |
| [SUMO](https://eclipse.dev/sumo/) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [CarRacing](https://gymnasium.farama.org/environments/box2d/car_racing/) | - | - | - |
| [CARLA](https://carla.org) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [CommonRoad](https://commonroad.in.tum.de/) | :white_check_mark: | :white_check_mark: | - |
| [highway-env](https://github.com/Farama-Foundation/HighwayEnv) | - | - | - |
| [SMARTS](https://github.com/huawei-noah/SMARTS) | :white_check_mark: | - | :white_check_mark: |
| [MetaDrive](https://github.com/metadriverse/metadrive) | :white_check_mark: | - | :white_check_mark: |
| [NuPlan](https://github.com/motional/nuplan-devkit) | - | - | - |
| [InterSim](https://github.com/Tsinghua-MARS-Lab/InterSim) | - | - | - |
| [TBSim](https://github.com/NVlabs/traffic-behavior-simulation) | - | - | - |
| [Waymax](https://github.com/waymo-research/waymax) | - | - | - |
| **Tactics2D** | :white_check_mark: | :white_check_mark: | :construction: |
