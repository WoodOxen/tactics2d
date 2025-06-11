import logging
import time

logging.basicConfig(level=logging.INFO)

import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO

from tactics2d.dataset_parser import LevelXParser
from tactics2d.map.map_config import HIGHD_MAP_CONFIG
from tactics2d.map.parser import OSMParser
from tactics2d.renderer import BEVCamera


class WebGLRenderer:
    _layout = "grid"
    _max_fps = 60

    def __init__(
        self, host: str = "127.0.0.1", port: int = 5000, max_fps: int = 60, layout: str = "grid"
    ):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode="threading")

        self.last_send_time = 0
        self.data_buffer = None
        self.client_ready = True

        self._setup_routes()
        self._setup_socket_events()
        self.update_layout(layout)
        self.update_max_fps(max_fps)

    @property
    def max_fps(self):
        return self._max_fps

    @property
    def layout(self):
        return self._layout

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            logging.info("Loading index page")
            return render_template("index.html", ws_url=f"http://{self.host}:{self.port}")

    def _setup_socket_events(self):
        @self.socketio.on("connect")
        def on_connect():
            logging.info("Client connected")
            self.socketio.start_background_task(self.generate_data)

        @self.socketio.on("render_complete")
        def on_render_complete():
            self.client_ready = True

    def _send_data_with_throttle(self):
        now = time.time()
        min_interval = 1.0 / self.max_fps

        # TODO: If a performance bottleneck is frequently reported, we may try to send the signal of different sensors separately.
        if self.client_ready:
            time.sleep(max(0, min_interval - (now - self.last_send_time)))
            self.socketio.emit("sensor_data", self.data_buffer)
            self.last_send_time = now
            self.client_ready = False

    def update_layout(self, layout: str):
        if layout is None:
            return

        if layout in ["grid", "hierarchical"]:
            if layout != self._layout:
                self._layout = layout
                self.socketio.emit("layout", self._layout)
                logging.info(f"Change the layout to {self._layout}")
        else:
            logging.warning(f"Invalid layout type. Reverting to the current layout {self._layout}.")

    def update_max_fps(self, max_fps: int):
        max_fps = int(max_fps)
        if max_fps > 100 or max_fps <= 0:
            logging.debug(f"Max FPS out range. Reverting to {self._max_fps} Hz.")
        else:
            self._max_fps = max_fps
            logging.debug(f"Change the maximum render FPS to {self._max_fps} Hz.")

    def generate_data(self):
        dataset = "highD"
        file_id = 1
        stamp_range = None
        file_path = f"/home/rowena/Documents/tactics2d/data/{dataset}/data"
        map_config = f"/home/rowena/Documents/tactics2d/tactics2d/dataset_parser/map.config"

        dataset_parser = LevelXParser(dataset)
        participants, actual_stamp_range = dataset_parser.parse_trajectory(
            file_id, file_path, stamp_range
        )

        map_parser = OSMParser(lanelet2=True)
        map_ = map_parser.parse(
            "/home/rowena/Documents/tactics2d/tactics2d/data/map/highD/highD_2.osm",
            HIGHD_MAP_CONFIG["highD_2"],
        )

        camera = BEVCamera(id_=1, map_=map_, perception_range=50)

        boundary = map_.boundary
        camera_position = np.array(
            [(boundary[0] + boundary[1]) / 2, (boundary[2] + boundary[3]) / 2]
        )

        prev_road_id_set = set()
        prev_participant_id_set = set()

        while True:
            for frame in range(actual_stamp_range[0], actual_stamp_range[1], 40):
                participant_ids = [pid for pid, p in participants.items() if p.is_active(frame)]

                geometry_data, prev_road_id_set, prev_participant_id_set = camera.update(
                    frame,
                    participants,
                    participant_ids,
                    prev_road_id_set,
                    prev_participant_id_set,
                    camera_position,
                )

                logging.debug("Sending geometry data for frame %d", frame)

                sensor_data = [
                    {
                        "id": "camera_1",
                        "perception_range": np.max(camera.perception_range),
                        "position": camera_position.tolist(),
                        "yaw": 0,
                        "map_data": geometry_data["map_data"],
                        "participant_data": geometry_data["participant_data"],
                    }
                ]
                self.update(sensor_data)

    def update(self, sensor_data: dict, layout: str = None):
        self.data_buffer = sensor_data
        self.update_layout(layout)
        self._send_data_with_throttle()

    def run(self):
        self.socketio.run(self.app, debug=True)
