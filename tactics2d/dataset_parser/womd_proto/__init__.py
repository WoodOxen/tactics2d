##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Initialize the womd_proto module.
# @Author: Yueyuan Li
# @Version: 1.0.0

import logging

import google.protobuf
from packaging import version

if version.parse(google.protobuf.__version__) <= version.parse("3.20.3"):
    from .pb2 import scenario_pb2 as scenario_pb2

    logging.info("Using Protocol Buffers v2")
else:
    from .pb3 import scenario_pb2 as scenario_pb2

    logging.info("Using Protocol Buffers v3")

__all__ = ["scenario_pb2"]
