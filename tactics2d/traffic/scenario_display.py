##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: scenario_display.py
# @Description:
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Tuple, List, Union
import time
import queue

from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

from tactics2d.sensor.defaults import DEFAULT_COLOR
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle, Cyclist, Pedestrian


class ScenarioDisplay:
    def __init__(self):
        self.map_patches = dict()
        self.participant_patches = dict()

    def reset(self):
        for patch in self.map_patches.values():
            patch.remove()

        for patch in self.participant_patches.values():
            patch.remove()

    def export(self, fps, **kwargs):
        pass

    def convert_color(self, color):
        normalize_color = (color[0] / 255, color[1] / 255, color[2] / 255)
        return normalize_color

    def display_map(self, map_, ax):
        patches = []
        lines = []
        line_widths = []
        line_colors = []

        for area in map_.areas.values():
            print(list(area.geometry.exterior.coords))
            area = Polygon(area.geometry.exterior.coords, True, color="black")
            patches.append(area)

        for lane in map_.lanes.values():
            lane = Polygon(lane.geometry.coords, True, color="gray")
            patches.append(lane)

        for roadline in map_.roadlines.values():
            lines.append(roadline.shape)
            line_widths.append(0.5 if roadline.type_ == "line_thin" else 1)
            line_colors.append(roadline.color)

        p = PatchCollection(patches, match_original=True)
        l = LineCollection(lines)

        ax.add_collection(p)
        ax.add_collection(l)

    def update_participants(self, frame, participants, ax):
        print(frame)

        for participant in participants.values():
            try:
                participant.get_state(frame)
            except:
                if participant.id_ in self.participant_patches:
                    self.participant_patches[participant.id_].remove()
                    self.participant_patches.pop(participant.id_)

                continue

            if isinstance(participant, Vehicle):
                if participant.id_ not in self.participant_patches:
                    self.participant_patches[participant.id_] = ax.add_patch(
                        Polygon(participant.get_pose(frame).coords, True, color="green")
                    )
                else:
                    self.participant_patches[participant.id_].set_xy(
                        participant.get_pose(frame).coords
                    )

        return list(self.participant_patches.values())

    def display(
        self,
        participants,
        map_,
        trajectory_fps: int = 10,
        time_range: Union[Tuple[int, int], List[int]] = (0, 10000),
        axis_range=None,
        export: bool = False,
        **export_args
    ):
        interval_ms = int(1000 / trajectory_fps)
        average_time = queue.Queue(maxsize=10)

        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        # ax.set_xlim([331200, 331800])
        # ax.set_ylim([4689000, 4690000])
        self.display_map(map_, ax)
        ax.plot()


        print(time_range[0], time_range[1], interval_ms)

        if len(time_range) == 2:
            ani = FuncAnimation(
                fig,
                self.update_participants,
                frames=np.arange(time_range[0], time_range[1], interval_ms),
                fargs=(participants, ax),
                interval=interval_ms,
                repeat=False,
            )
        else:
            ani = FuncAnimation(
                fig,
                self.update_participants,
                frames=time_range,
                fargs=(participants, ax),
                interval=interval_ms,
                repeat=False,
            )
        # ax.autoscale()
        plt.show()
