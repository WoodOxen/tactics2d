# Installation

## System Requirements

We have conducted testing for the execution and construction of `tactics2d` on the following platforms:

| System | 3.8 | 3.9 | 3.10 | 3.11 |
| --- | --- | --- | --- | --- |
| Ubuntu 18.04 | :white_check_mark: | - | - | - |
| Ubuntu 20.04 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Ubuntu 22.04 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Windows | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| MacOS | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Installation on Linux

!!! note
    We recommend using a virtual environment to install `tactics2d` to avoid conflicts with other Python packages.

!!! info
    We have tested the installation process on Dockers of Ubuntu 18.04, 20.04, and 22.04.

### Install with PyPI

You can simply install `tactics2d` from PyPI with the following command.

```bash
pip install tactics2d
```

### Install from Source

You can also install `tactics2d` from from its source on GitHub. This way is recommended if you want to run the sample code or contribute to the development of `tactics2d`. Please note that you should have GCC installed on your operating system before installing tactics2d.

```bash
# clone the repository with submodules but ignore the large files (mainly the NuPlan's map data)
# please download NuPlan's map data from its official website and put it in the `tactics2d/data/map/NuPlan` directory
git clone --recurse-submodules git@github.com:WoodOxen/tactics2d.git
cd tactics2d
pip install -v .
```

If no errors occurs, you should have installed `tactics2d` successfully.

## Configure Datasets
