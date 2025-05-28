##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: cli.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9

import argparse


def main():
    parser = argparse.ArgumentParser(description="Tactics2D Command Line Tool")
    parser.add_argument("--version", action="version", version="tactics2d 0.1.9")

    args = parser.parse_args()
