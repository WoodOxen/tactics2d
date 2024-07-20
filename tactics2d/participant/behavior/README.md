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

- setup.py不要了，等到重构完了结合到tactics2d里面去
- 那个html的用途是什么？是不是可以删掉？
- utils_cython.c好像是靠setup.py生成的，你看下如何设置生成过程，确保如果本地运行setup.py可以覆写
- 每个同名文件中只应该包含它的同名文件的函数，不应该包含其他文件的函数，在一级类/函数的注释中标明你的基准文件（网页链接），如果有些函数来自非基准的同名文件，那么在函数注释中标明来源。
- predictor.py是干什么的？我们这边不需要吗？还是其实更好写成的形式？
- 如果是不需要被用户调用的函数，写成私有类/函数，即在类/函数名前加一个下划线
- 注释格式按 Google的来，每个函数的注释都要标注输入输出的类型和含义

## BITS

## LitSim
