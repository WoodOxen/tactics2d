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

from tactics2d.map.element import Lane, Area, RoadLine, Map
from tactics2d.participant.element import Vehicle, Cyclist, Pedestrian
from tactics2d.sensor.defaults import COLOR_PALETTE, DEFAULT_COLOR


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

    def _get_color(self, element):
        if element.color in COLOR_PALETTE:
            return COLOR_PALETTE[element.color]
        
        if element.color is None:
            if hasattr(element, "subtype") and element.subtype in DEFAULT_COLOR:
                return DEFAULT_COLOR[element.subtype]
            if hasattr(element, "type_") and element.type_ in DEFAULT_COLOR:
                return DEFAULT_COLOR[element.type_]
            elif isinstance(element, Area):
                return DEFAULT_COLOR["area"]
            elif isinstance(element, Lane):
                return DEFAULT_COLOR["lane"]
            elif isinstance(element, RoadLine):
                return DEFAULT_COLOR["roadline"]
            elif isinstance(element, Vehicle):
                return DEFAULT_COLOR["vehicle"]
            elif isinstance(element, Cyclist):
                return DEFAULT_COLOR["cyclist"]
            elif isinstance(element, Pedestrian):
                return DEFAULT_COLOR["pedestrian"]
            
        if len(element.color) == 3 or len(element.color) == 4:
            if 1 < np.max(element.color) <= 255:
                return tuple([color / 255 for color in element.color])

    def display_map(self, map_, ax):
        patches = []
        lines = []
        line_widths = []
        line_colors = []

        for area in map_.areas.values():
            area = Polygon(area.geometry.exterior.coords, True, color=self._get_color(area), edgecolor=None)
            patches.append(area)

        for lane in map_.lanes.values():
            lane = Polygon(lane.geometry.coords, True, color=self._get_color(lane), edgecolor=None)
            patches.append(lane)

        for roadline in map_.roadlines.values():
            if roadline.type_ == "virtual":
                continue
            lines.append(roadline.shape)
            line_widths.append(0.5 if roadline.type_ == "line_thin" else 1)
            line_colors.append(self._get_color(roadline))

        p = PatchCollection(patches, match_original=True)
        l = LineCollection(lines, colors = line_colors, linewidths = line_widths)

        ax.add_collection(p)
        ax.add_collection(l)

    def update_participants(self, frame, participants, ax):

        for participant in participants.values():
            try:
                participant.get_state(frame)
            except:
                if participant.id_ in self.participant_patches:
                    self.participant_patches[participant.id_].remove()
                    self.participant_patches.pop(participant.id_)

                continue

            if isinstance(participant, Vehicle):
                print(list(participant.get_pose(frame).coords))
                if participant.id_ not in self.participant_patches:
                    self.participant_patches[participant.id_] = ax.add_patch(
                        Polygon(participant.get_pose(frame).coords, True, color=self._get_color(participant), edgecolor=None)
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
        ax.autoscale()
        plt.show()
