![Tactics2D LOGO](https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/Tactics_LOGO_long.jpg)

# Tactics2D: A Reinforcement Learning Environment Library for Driving Decision-making

[![Codacy](https://app.codacy.com/project/badge/Grade/2bb48186b56d4e3ab963121a5923d6b5)](https://app.codacy.com/gh/WoodOxen/tactics2d/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codecov](https://codecov.io/gh/WoodOxen/tactics2d/graph/badge.svg?token=X81Z6AOIMV)](https://codecov.io/gh/WoodOxen/tactics2d)
![Test Modules](https://github.com/WoodOxen/tactics2d/actions/workflows/test_modules.yml/badge.svg?)
[![Read the Docs](https://img.shields.io/readthedocs/tactics2d)](https://tactics2d.readthedocs.io/en/latest/)
[![Downloads](https://img.shields.io/pypi/dm/tactics2d)](https://pypi.org/project/tactics2d/)

![python-version](https://camo.githubusercontent.com/2b53588bcdf5ca9bcfc10921eb80d43a1e2d52e5a4ede24273800a5074a0916d/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f67796d6e617369756d2e737667)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Github license](https://img.shields.io/github/license/WoodOxen/tactics2d)](https://github.com/WoodOxen/tactics2d/blob/dev/LICENSE)

## About

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
- **Reliability** -- Over [FILL LATER]\% code is covered by unit tests and integration tests.

For further information on `tactics2d`, please refer to our comprehensive [documentation](https://tactics2d.readthedocs.io/en/latest/), and a detailed comparison with other similar libraries is available [here](https://tactics2d.readthedocs.io/en/latest/#why-tactics2d).

### Community

We have a Discord server for the community to discuss, ask questions and make contribution. You can join the server by clicking the following link.

<div>
  <iframe src="https://discordapp.com/widget?id=1209363816912126003&theme=system" width="700" height="350" allowtransparency="true" frameborder="0" sandbox="allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts"></iframe>
</div>

## Installation

### System Requirements

We have conducted testing for the execution and construction of `tactics2d` on the following platforms:

| System | 3.8 | 3.9 | 3.10 | 3.11 |
| --- | --- | --- | --- | --- |
| Ubuntu 18.04 | $\surd$ | - | - | - |
| Ubuntu 20.04 | $\surd$ | $\surd$ | $\surd$ | $\surd$ |
| Ubuntu 22.04 | $\surd$ | $\surd$ | $\surd$ | $\surd$ |
| Windows | $\surd$ | $\surd$ | $\surd$ | $\surd$ |
| MacOS | $\surd$ | $\surd$ | $\surd$ | $\surd$ |

### Install from PyPI

You can simply install `tactics2d` from PyPI with the following command.

```shell
pip install tactics2d
```

### Install from Github

You can also install `tactics2d` from from its source on GitHub. This way is recommended if you want to run the sample code or contribute to the development of `tactics2d`.

```shell
git clone --recurse-submodules git@github.com:WoodOxen/tactics2d.git
cd tactics2d

conda create -n tactics2d python=3.8
# if you are a developer, you can install the development dependencies by
pip install -e .
# if you are a user, you can install tactics2d by
pip install .

# test the installation by
[TODO]
```

If no errors occurs, you should have installed `tactics2d` successfully.

## Citation

If you find `tactics2d` useful, please cite this in your publication.

```bibtex
@article{li2023tactics2d,
  title={Tactics2D: A Multi-agent Reinforcement Learning Environment for Driving Decision-making},
  author={Li, Yueyuan and Zhang, Songan and Jiang, Mingyang and Chen, Xingyuan and Yang, Ming},
  journal={arXiv preprint arXiv:2311.11058},
  year={2023}
}
```
