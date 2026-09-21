::: tactics2d.participant

::: tactics2d.participant.trajectory
    options:
        heading_level: 2
        members:
            - State
            - Trajectory

::: tactics2d.participant.element
    options:
        heading_level: 2
        members:
            - ParticipantBase
            - Vehicle
            - Cyclist
            - Pedestrian
            - Other
            - list_vehicle_templates
            - list_cyclist_templates
            - list_pedestrian_templates

## 交通参与者模板

### 四轮车辆模型

我们对车辆类型的分类遵循[欧洲排放标准](https://en.wikipedia.org/wiki/Vehicle_size_class#EEC)（EEC），因其分类清晰明了。为了确定参数，我们基于常见销售型号和在线可获取的数据，为每个类型选择了一款代表性车辆。这些选择经过仔细考虑，以确保所使用的数据既具有代表性又准确。

由于获取每种车辆类型的精确最大转向值具有挑战性，我们假设所有车辆的统一最大转向值为 $\pi/6$ 弧度。这一假设基于以下理解：由于我们的车辆物理模型基于自行车模型，转向范围的细微变化不太可能对仿真结果产生显著影响。

默认车辆参数可通过调用 [`tactics2d.participant.element.list_vehicle_templates()`](#tactics2d.participant.element.list_vehicle_templates) 查看。

| EEC 类别 | 原型 | 长度 (m) | 宽度 (m) | 高度 (m) | 轴距 (m) | 前悬 (m) | 后悬 (m) | 整备质量 (kg) | 最高速度 (m/s) | 0-100 km/h (s) | 驱动模式 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: 微型车 | [Volkswagen Up](https://en.wikipedia.org/wiki/Volkswagen_Up) | 3.540 | 1.641 | 1.489 | 2.420 | 0.585 | 0.535 | 1070 | 44.44 | 14.4 | FWD |
| B: 小型车 | [Volkswagen Polo](https://en.wikipedia.org/wiki/Volkswagen_Polo) | 4.053 | 1.751 | 1.461 | 2.548 | 0.824 | 0.681 | 1565 | 52.78 | 11.2 | FWD |
| C: 中型车 | [Volkswagen Golf](https://en.wikipedia.org/wiki/Volkswagen_Golf_Mk8) | 4.284 | 1.799 | 1.452 | 2.637 | 0.880 | 0.767 | 1620 | 69.44 | 8.9 | FWD |
| D: 大型车 | [Volkswagen Passat](https://en.wikipedia.org/wiki/Volkswagen_Passat_(B8)) | 4.866 | 1.832 | 1.477 | 2.871 | 0.955 | 1.040 | 1735 | 58.33 | 8.4 | FWD |
| E: 行政级车 | [Audi A6](https://en.wikipedia.org/wiki/Audi_A6) | 5.050 | 1.886 | 1.475 | 3.024 | 0.921 | 1.105  | 2175 | 63.89 | 8.1 | FWD |
| F: 豪华车 | [Audi A8](https://en.wikipedia.org/wiki/Audi_A8#Fourth_generation_(D5;_2018%E2%80%93present)) | 5.302 | 1.945 | 1.488 | 3.128 | 0.989 | 1.185 | 2520 | 69.44 | 6.7 | AWD |
| S: 运动轿跑 | [Ford Mustang](https://en.wikipedia.org/wiki/Ford_Mustang) | 4.788 | 1.916 | 1.381 | 2.720 | 0.830 | 1.238 | 1740 | 63.89 | 5.3 | AWD |
| M: MPV | [Kia Carnival](https://en.wikipedia.org/wiki/Kia_Carnival) | 5.155 | 1.995 | 1.740 | 3.090 | 0.935 | 1.130 | 2095 | 66.67 | 9.4 | 4WD |
| J: SUV | [Jeep Grand Cherokee](https://en.wikipedia.org/wiki/Jeep_Grand_Cherokee) | 4.828 | 1.943 | 1.792 | 2.915 | 0.959 | 0.954 | 2200 | 88.89 | 3.8 | 4WD |

### 骑行者模型

骑行者模型基于平均参数设计。要访问默认骑行者参数，您可以调用 [`tactics2d.participant.element.list_cyclist_templates()`](#tactics2d.participant.element.list_cyclist_templates)。

| 名称 | 长度 (m) | 宽度 (m) | 高度 (m) | 最大转向 (rad) | 最高速度 (m/s) | 最大加速度 (m/s$^2$) | 最大减速度 (m/s$^2$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 骑行者 | 1.80 | 0.65 | 1.70 | 1.05 | 22.78 | 5.8 | 7.8 |
| 轻便摩托车 | 2.00 | 0.70 | 1.70 | 0.35 | 13.89 | 3.5 | 7.0 |
| 摩托车 | 2.40 | 0.80 | 1.70 | 0.44 | 75.00 | 5.0 | 10.0 |

### 行人模型

行人模型基于平均参数设计。要访问默认行人参数，您可以调用 [`tactics2d.participant.element.list_pedestrian_templates()`](#tactics2d.participant.element.list_pedestrian_templates)。

| 名称 | 长度 (m) | 宽度 (m) | 高度 (m) | 最高速度 (m/s) | 最大加速度 (m/s$^2$) |
| --- | --- | --- | --- | --- | --- |
| 成人/男性 | 0.24 | 0.40 | 1.75 | 7.0 | 1.5 |
| 成人/女性 | 0.22 | 0.37 | 1.65 | 6.0 | 1.5 |
| 儿童 (六岁) | 0.18 | 0.25 | 1.16 | 3.5 | 1.0 |
| 儿童 (十岁) | 0.20 | 0.35 | 1.42 | 4.5 | 1.0 |
