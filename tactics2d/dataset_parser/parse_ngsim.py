##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_ngsim.py
# @Description: This file implements a parser for NGSIM dataset.
# @Author: Yueyuan Li
# @Version: 1.0.0

import os
from typing import Tuple, Union

import numpy as np
import pandas as pd

from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class NGSIMParser:
    """
    TODO: The support of NGSIM dataset is planned to be added before version 1.1.0.
    """

    def parse_trajectory(self, file: str, folder: str, stamp_range: Tuple[int, int] = None):
        if stamp_range is None:
            stamp_range = (-np.inf, np.inf)

        # load the vehicles that have frame in the arbitrary range
        participants = dict()
        actual_stamp_range = (np.inf, -np.inf)

        df_track_chunk = pd.read_csv(os.path.join(folder, file), iterator=True, chunksize=10000)

        for _, info in df_track_chunk.iterrows():
            return
        return
