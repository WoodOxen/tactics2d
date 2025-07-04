##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: cli.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9

import sys

sys.path.append("..")

import logging

logging.basicConfig(level=logging.INFO)
import argparse
import threading

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Tactics2D Command Line Tool")
    parser.add_argument("--version", action="version", version="tactics2d 0.1.9")

    args = parser.parse_args()


def main():
    renderer = WebGLRenderer(max_fps=25)
    renderer.run()


if __name__ == "__main__":
    main()
