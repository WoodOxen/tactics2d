##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: scenario_display.py
# @Description:
# @Author: Yueyuan Li
# @Version: 1.0.0

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from tactics2d.map.element import Area, Lane, RoadLine
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.sensor.render_template import COLOR_PALETTE, DEFAULT_COLOR, DEFAULT_ORDER


class ScenarioDisplay:
    """This class implements a matplotlib-based scenario visualizer."""

    def __init__(self):
        self.participant_patches = dict()

    def _get_color(self, element):
        if element.color in COLOR_PALETTE:
            return COLOR_PALETTE[element.color]

        if element.color is None:
            if hasattr(element, "subtype") and element.subtype in DEFAULT_COLOR:
                return DEFAULT_COLOR[element.subtype]
            if hasattr(element, "type_") and element.type_ in DEFAULT_COLOR:
                return DEFAULT_COLOR[element.type_]
            if isinstance(element, Area):
                return DEFAULT_COLOR["area"]
            if isinstance(element, Lane):
                return DEFAULT_COLOR["lane"]
            if isinstance(element, RoadLine):
                return DEFAULT_COLOR["roadline"]
            if isinstance(element, Vehicle):
                return DEFAULT_COLOR["vehicle"]
            if isinstance(element, Cyclist):
                return DEFAULT_COLOR["cyclist"]
            if isinstance(element, Pedestrian):
                return DEFAULT_COLOR["pedestrian"]

        if len(element.color) == 3 or len(element.color) == 4:
            if 1 < np.max(element.color) <= 255:
                return tuple([color / 255 for color in element.color])

    def _get_order(self, element):
        if hasattr(element, "subtype") and element.subtype in DEFAULT_ORDER:
            return DEFAULT_ORDER[element.subtype]
        if hasattr(element, "type_") and element.type_ in DEFAULT_ORDER:
            return DEFAULT_ORDER[element.type_]
        if isinstance(element, Area):
            return DEFAULT_ORDER["area"]
        if isinstance(element, Lane):
            return DEFAULT_ORDER["lane"]
        if isinstance(element, RoadLine):
            return DEFAULT_ORDER["roadline"]
        if isinstance(element, Vehicle):
            return DEFAULT_ORDER["vehicle"]
        if isinstance(element, Cyclist):
            return DEFAULT_ORDER["cyclist"]
        if isinstance(element, Pedestrian):
            return DEFAULT_ORDER["pedestrian"]

    def _get_line(self, roadline):
        if roadline.type_ == "virtual":
            return None

        line_shape = np.array(roadline.shape)
        line_width = 0.5 if roadline.type_ in ["line_thin", "curbstone"] else 1

        if roadline.subtype == "solid_solid":
            line1 = Line2D(
                line_shape[:, 0] + 0.05,
                line_shape[:, 1],
                linewidth=line_width,
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )

            line2 = Line2D(
                line_shape[:, 0] - 0.05,
                line_shape[:, 1],
                linewidth=line_width,
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )

            return [line1, line2]

        if roadline.subtype == "dashed":
            return [
                Line2D(
                    line_shape[:, 0],
                    line_shape[:, 1],
                    linewidth=line_width,
                    linestyle=(0, (5, 5)),
                    color=self._get_color(roadline),
                    zorder=self._get_order(roadline),
                )
            ]

        if roadline.subtype == "dashed_dashed":
            line1 = Line2D(
                line_shape[:, 0] + 0.05,
                line_shape[:, 1],
                linewidth=line_width,
                linestyle=(0, (5, 5)),
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )

            line2 = Line2D(
                line_shape[:, 0] - 0.05,
                line_shape[:, 1],
                linewidth=line_width,
                linestyle=(0, (5, 5)),
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )

        return [
            Line2D(
                line_shape[:, 0],
                line_shape[:, 1],
                linewidth=line_width,
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )
        ]

    def display_map(self, map_, ax):
        for area in map_.areas.values():
            area = Polygon(
                area.geometry.exterior.coords,
                True,
                facecolor=self._get_color(area),
                edgecolor=None,
                zorder=self._get_order(area),
            )
            ax.add_patch(area)

        for lane in map_.lanes.values():
            lane = Polygon(
                lane.geometry.coords,
                True,
                facecolor=self._get_color(lane),
                edgecolor=None,
                zorder=self._get_order(lane),
            )
            ax.add_patch(lane)

        for roadline in map_.roadlines.values():
            lines = self._get_line(roadline)
            if lines is None:
                continue

            for line in lines:
                ax.add_line(line)

    def update_participants(self, frame, participants, ax):
        for participant in participants.values():
            try:
                participant.get_state(frame)
            except:
                if (
                    participant.id_ in self.participant_patches
                    and participant.trajectory.last_frame < frame
                ):
                    self.participant_patches[participant.id_].remove()
                    self.participant_patches.pop(participant.id_)

                continue

            if isinstance(participant, Vehicle):
                if participant.id_ not in self.participant_patches:
                    self.participant_patches[participant.id_] = ax.add_patch(
                        Polygon(
                            participant.get_pose(frame).coords,
                            True,
                            facecolor=self._get_color(participant),
                            edgecolor=None,
                            zorder=self._get_order(participant),
                        )
                    )
                else:
                    self.participant_patches[participant.id_].set_xy(
                        participant.get_pose(frame).coords
                    )
            elif isinstance(participant, Cyclist):
                if participant.id_ not in self.participant_patches:
                    self.participant_patches[participant.id_] = ax.add_patch(
                        Polygon(
                            participant.get_pose(frame).coords,
                            True,
                            facecolor=self._get_color(participant),
                            edgecolor=None,
                            zorder=self._get_order(participant),
                        )
                    )
                else:
                    self.participant_patches[participant.id_].set_xy(
                        participant.get_pose(frame).coords
                    )
            elif isinstance(participant, Pedestrian):
                if participant.id_ not in self.participant_patches:
                    self.participant_patches[participant.id_] = ax.add_patch(
                        Circle(
                            participant.get_state(frame).location,
                            radius=0.5,
                            facecolor=self._get_color(participant),
                            edgecolor=None,
                            zorder=self._get_order(participant),
                        )
                    )
                else:
                    self.participant_patches[participant.id_].set(
                        center=participant.get_state(frame).location
                    )

        return list(self.participant_patches.values())

    def display(self, participants, map_, interval, frames, fig_size, **ax_kwargs):
        fig, ax = plt.subplots()
        fig.set_size_inches(fig_size)
        fig.set_layout_engine("none")
        ax.set(**ax_kwargs)
        ax.set_axis_off()

        if "womd" in map_.name:
            fig.set_facecolor(COLOR_PALETTE["black"])

        self.display_map(map_, ax)
        ax.plot()

        animation = FuncAnimation(
            fig,
            self.update_participants,
            frames=frames,
            fargs=(participants, ax),
            interval=interval,
        )

        return animation

    def reset(self):
        for patch in self.participant_patches.values():
            patch.remove()

        self.participant_patches.clear()
