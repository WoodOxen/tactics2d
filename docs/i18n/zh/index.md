# Tactics2D 中文文档

[![Codacy](https://app.codacy.com/project/badge/Grade/2bb48186b56d4e3ab963121a5923d6b5)](https://app.codacy.com/gh/WoodOxen/tactics2d/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codecov](https://codecov.io/gh/WoodOxen/tactics2d/graph/badge.svg?token=X81Z6AOIMV)](https://codecov.io/gh/WoodOxen/tactics2d)
![Test Modules](https://github.com/WoodOxen/tactics2d/actions/workflows/test_modules.yml/badge.svg?)
[![Read the Docs](https://img.shields.io/readthedocs/tactics2d)](https://tactics2d.readthedocs.io/en/latest/)
[![Downloads](https://img.shields.io/pypi/dm/tactics2d)](https://pypi.org/project/tactics2d/)

![python-version](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Github license](https://img.shields.io/github/license/WoodOxen/tactics2d)](https://github.com/WoodOxen/tactics2d/blob/dev/LICENSE)

> [!note]
> 这是上海交通大学课程 AU7043 的官方代码库。
> **选修本课程的同学，请使用 `git pull` 命令下载此仓库！**

## 关于

欢迎阅读 Python 库 tactics2d 的官方文档！

`tactics2d` 是一个开源 Python 库，为自动驾驶中基于强化学习的决策模型提供多样化和具有挑战性的交通场景。`tactics2d` 具有以下主要特性：

- **兼容性**
  - 📦 轨迹数据集 -- 支持无缝导入多种真实世界轨迹数据集，包括 Argoverse、Dragon Lake Parking (DLP)、INTERACTION、LevelX 系列 (highD、inD、rounD、ExiD)、NuPlan 和 Waymo Open Motion Dataset (WOMD)，涵盖轨迹解析和地图信息。
  - 📄 地图格式 -- 支持解析和转换常用开放地图格式，如 OpenDRIVE、Lanelet2 风格 OpenStreetMap (OSM) 和 SUMO roadnet。
- **可定制性**
  - 🚗 交通参与者 -- 支持创建具有可定制物理属性、物理动力学/运动学模型和行为模型的新交通参与者类。
  - 🚧 道路元素 -- 支持定义新的道路元素，重点关注监管方面。
- **多样性**
  - 🛣️ 交通场景 -- 提供广泛的 Gym 风格交通场景，包括高速公路、车道合并、无信号/有信号交叉口、环岛、停车和赛车场景。

## 快速开始

请参阅[安装指南](installation.md)开始使用 tactics2d。
