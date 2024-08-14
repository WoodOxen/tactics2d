# Interactive Traffic Participants' Behavior Models

## InterSim

The `intersim` module is a third-party algorithm that provides interactive traffic behavior model. The `intersim` module is refactored from the interaction model proposed by

> Sun, Qiao, et al. "InterSim: Interactive traffic simulation via explicit relation modeling." *2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2022.

The original implementation is available at [Tsinghua-MARS-Lab/InterSim](https://github.com/Tsinghua-MARS-Lab/InterSim/tree/main). We refactor the code to suit the `tactics2d` framework. The simulation efficiency is improved by using the `tactics2d` data structure and the `tactics2d` simulation engine. We also improve the code quality and add more comments to help users understand the code.

```latex
@inproceedings{sun2022intersim,
    title={{InterSim}: Interactive Traffic Simulation via Explicit Relation Modeling},
    author={Sun, Qiao and Huang, Xin and Williams, Brian and Zhao, Hang},
    booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2022},
    organization={IEEE}
}
```

---

TODO:

- 所有函数的调用形式都是从最高路径 (tactics2d) 开始
- 所有纯数学、无物理意义的utils函数整理到tactics2d/utils文件夹中，不要分别放在planning和prediction里面
- 用tactics2d里面的agent，如果interface有缺失的属性，先跟我讨论如何补充的方案
- 目前看起来并不能跑通，抓紧时间debug
- setup.py不要了，等到重构完了结合到tactics2d里面去
- 每个同名文件中只应该包含它的同名文件的函数，不应该包含其他文件的函数，在一级类/函数的注释中标明你的基准文件（网页链接），如果有些函数来自非基准的同名文件，那么在函数注释中标明来源。
- 如果是不需要被用户调用的函数，写成私有类/函数，即在类/函数名前加一个下划线
- 注释格式按 Google的来，每个函数的注释都要标注输入输出的类型和含义

## BITS

## LitSim
