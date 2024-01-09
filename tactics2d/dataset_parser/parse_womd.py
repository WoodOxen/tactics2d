import os

import tensorflow as tf

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist, Other
from tactics2d.trajectory.element import State, Trajectory
from tactics2d.map.element import Map
from tactics2d.dataset_parser.womd_proto import scenario_pb2


class WOMDParser:
    """This class implements a parser for Waymo Open Motion Dataset.

    Ettinger, Scott, et al. "Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    """

    TYPE_MAPPING = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}

    CLASS_MAPPING = {0: Other, 1: Vehicle, 2: Pedestrian, 3: Cyclist, 4: Other}

    def get_scenario_ids(self, file_name: str, folder_path: str):
        """This function get the list of scenario ids from the given tfrecord file.

        Args:
            file_name (str): The name of the tfrecord file.
            folder_path (str): The path to the folder containing the tfrecord file.
        """
        id_list = []
        file_path = os.path.join(folder_path, file_name)
        dataset = tf.data.TFRecordDataset(file_path, compression_type="")

        for data in dataset:
            proto_string = data.numpy()
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(proto_string)
            id_list.append(scenario.scenario_id)

        return id_list

    def parse_trajectory(self, scenario_id=None, **kwargs):
        """This function parses trajectories from a single tfrecord file with a given scenario id. Because the duration of the scenario is well articulated, the parser will not provide an option to parse a subset of time range of the scenario.

        Args:
            scenario_id (str, optional): The id of the scenario to parse. If the scenario id is not
                given, the first scenario in the file will be parsed. Defaults to None.
            dataset (tf.data.TFRecordDataset, optional): The dataset to parse.
            file_name (str): The name of the tfrecord file.
            folder_path (str): The path to the folder containing the tfrecord file.

        Returns:
            dict: A dictionary of participants. If the scenario id is not found, return None.
        """
        if "dataset" in kwargs:
            dataset = kwargs["dataset"]
        elif "file_name" in kwargs and "folder_path" in kwargs:
            file_path = os.path.join(kwargs["folder_path"], kwargs["file_name"])
            dataset = tf.data.TFRecordDataset(file_path, compression_type="")
        else:
            raise KeyError(
                "Either dataset or file_name and folder_path should be given as keyword arguments."
            )

        for data in dataset:
            proto_string = data.numpy()
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(proto_string)

            if scenario_id is None:
                scenario_id = scenario.scenario_id

            if scenario_id == scenario.scenario_id:
                participants = dict()
                timestamps = scenario.timestamps_seconds
                for track in scenario.tracks:
                    trajectory = Trajectory(id_=track.id, fps=10, stable_freq=False)
                    width = 0
                    length = 0
                    height = 0
                    cnt = 0
                    for i, state_ in enumerate(track.states):
                        if not state_.valid:
                            continue
                        state = State(
                            frame=timestamps[i],
                            x=state_.center_x,
                            y=state_.center_y,
                            heading=state_.heading,
                            vx=state_.velocity_x,
                            vy=state_.velocity_y,
                        )

                        trajectory.append_state(state)

                        width += state_.width
                        length += state_.length
                        height += state_.height
                        cnt += 1

                    participant = self.CLASS_MAPPING[track.object_type](
                        id_=track.id,
                        type_=self.TYPE_MAPPING[track.object_type],
                        length=length / cnt,
                        width=width / cnt,
                        height=height / cnt,
                        trajectory=trajectory,
                    )
                    participants[track.id] = participant

                return participants

        return None

    def _parse_map_features(self, map_feature):
        return

    def parse_map(self, scenario_id=None, **kwargs):
        """_summary_

        Args:
            scenario_id (_type_, optional): _description_. Defaults to None.

        Raises:
            KeyError: _description_
        """
        if "dataset" in kwargs:
            dataset = kwargs["dataset"]
        elif "file_name" in kwargs and "folder_path" in kwargs:
            file_path = os.path.join(kwargs["folder_path"], kwargs["file_name"])
            dataset = tf.data.TFRecordDataset(file_path, compression_type="")
        else:
            raise KeyError(
                "Either dataset or file_name and folder_path should be given as keyword arguments."
            )

        for data in dataset:
            proto_string = data.numpy()
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(proto_string)

            if scenario_id is None:
                scenario_id = scenario.scenario_id

            if scenario_id == scenario.scenario_id:
                map_ = Map(name="nuplan_" + scenario.scenario_id)
                for map_feature in scenario.map_features:
                    return
