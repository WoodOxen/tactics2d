import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("...")

import logging
import random
import time

logging.basicConfig(level=logging.INFO)

import eventlet

eventlet.monkey_patch()
import json

from camera import BEVCamera
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from shapely.geometry import Point

from tactics2d.dataset_parser import LevelXParser
from tactics2d.map.parser import OSMParser

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")


@app.route("/")
def index():
    host = request.args.get("host", "127.0.0.1")
    port = request.args.get("port", "5000")
    ws_url = f"http://{host}:{port}"
    return render_template("index.html", ws_url=ws_url)


@socketio.on("connect")
def on_connect():
    print("Client connected")
    socketio.start_background_task(generate_data)


def generate_data():
    dataset = "highD"
    file_id = 1
    stamp_range = None
    file_path = f"/home/rowena/Documents/tactics2d/data/{dataset}/data"
    map_config = f"/home/rowena/Documents/tactics2d/tactics2d/dataset_parser/map.config"
    with open(map_config) as f:
        configs = json.load(f)

    dataset_parser = LevelXParser(dataset)
    participants, actual_stamp_range = dataset_parser.parse_trajectory(
        file_id, file_path, stamp_range
    )

    map_parser = OSMParser(lanelet2=True)
    map_ = map_parser.parse(
        "/home/rowena/Documents/tactics2d/tactics2d/data/map/highD/highD_2.osm", configs["highD_2"]
    )

    camera = BEVCamera(id_=1, map_=map_, perception_range=50)

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
                Point(200, -10),
            )

            socketio.emit("geometry_data", geometry_data)

            time.sleep(0.04)


if __name__ == "__main__":
    socketio.run(app, debug=True)
