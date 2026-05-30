# 安装指南

## 系统要求

我们已在以下平台测试 `tactics2d` 的执行和构建：

| 系统 | 3.8 | 3.9 | 3.10 | 3.11 |
| --- | --- | --- | --- | --- |
| Ubuntu 18.04 | :white_check_mark: | - | - | - |
| Ubuntu 20.04 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Ubuntu 22.04 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Windows | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| MacOS | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Linux 系统安装

!!! note
    建议使用虚拟环境安装 `tactics2d`，以避免与其他 Python 包冲突。

!!! info
    我们已在 Ubuntu 18.04、20.04 和 22.04 的 Docker 环境中测试安装过程。

### 通过 PyPI 安装

您可以通过以下命令从 PyPI 安装 `tactics2d`：

```bash
pip install tactics2d
```

### 通过源码安装

1. 克隆仓库：
```bash
git clone https://github.com/WoodOxen/tactics2d.git
cd tactics2d
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 构建 C++ 扩展：
```bash
python setup.py install
```

## Windows 系统安装

Windows 用户需要安装 Microsoft Visual C++ Build Tools。

## 验证安装

安装完成后，您可以通过以下命令验证安装：

```bash
python -c "import tactics2d; print(tactics2d.__version__)"
```

## 常见问题

- **问题：** 缺少 C++ 编译器
  **解决方案：** 安装 gcc/g++ (Linux) 或 Microsoft Visual C++ Build Tools (Windows)

- **问题：** 依赖冲突
  **解决方案：** 使用虚拟环境或 Conda 环境隔离依赖
