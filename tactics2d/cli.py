# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cli implementation."""


import sys

sys.path.append("..")

import logging

logging.basicConfig(level=logging.INFO)
import argparse

import tactics2d


def parse_args():
    parser = argparse.ArgumentParser(description="Tactics2D Command Line Tool")
    parser.add_argument("--version", action="version", version=f"tactics2d {tactics2d.__version__}")

    args = parser.parse_args()


def main():
    renderer = WebGLRenderer(max_fps=25)
    renderer.run()


if __name__ == "__main__":
    main()
