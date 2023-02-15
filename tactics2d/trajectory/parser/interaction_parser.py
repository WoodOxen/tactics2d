import os
import pandas as pd

from tactics2d.object.element.vehicle import Vehicle
from tactics2d.object.element.vehicle import Pedestrian
from tactics2d.object.element.cyclist import Cyclist
from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


class InteractionParser(object):
    def _get_vehicle_type(self, length: float, width: float) -> str:
        return 

    def _get_pedestrian_type(self, trajectory: Trajectory) -> str:
        return

    @staticmethod
    def parse_vehicle(file_id, folder_path, stamp_range):
        df_vehicle = pd.read_csv(os.path.join(folder_path, "vehicle_tracks_%03d.csv" % file_id))

        vehicles = dict()
        trajectories = dict()

        for _, state_info in df_vehicle.iterrows():
            if state_info["frame_id"] < stamp_range[0] or state_info["frame_id"] > stamp_range[1]:
                continue
            
            vehicle_id =  state_info["track_id"]
            if vehicle_id not in vehicles:
                vehicle_type = InteractionParser()._get_vehicle_type(state_info["length"], state_info["width"])
                vehicle = Vehicle(
                    id_=vehicle_id, type_=vehicle_type,
                    length=state_info["length"], width=state_info["width"]
                )
                vehicles[vehicle_id] = vehicle

            if vehicle_id not in trajectories:
                trajectories[vehicle_id] = Trajectory(vehicle_id)

            state = State(
                frame=state_info["frame_id"], x=state_info["x"], y=state_info["y"],
                heading=state_info["psi_rad"]
            )
            state.set_velocity(state_info["vx"], state_info["vy"])
            trajectories[vehicle_id].add_state(state)
        
        for vehicle_id, vehicle in vehicles.items():
            vehicles[vehicle_id].bind_trajectory(trajectories[vehicle_id])
        
        return vehicles

    @staticmethod
    def parse_pedestrians(file_id, folder_path, stamp_range):

        pedestrian_path = os.path.join(folder_path, "pedestrian_tracks_%03d.csv" % file_id)
        if os.path.exists(pedestrian_path):
            df_pedestrian = pd.read_csv(pedestrian_path)
        else:
            return dict(), dict()

        pedestrians = dict()
        cyclists = dict()
        trajectories = dict()

        for _, state_info in df_pedestrian.iterrows():
            if state_info["frame_id"] < stamp_range[0] or state_info["frame_id"] > stamp_range[1]:
                continue

            trajectory_id = int(state_info["frame_id"][1:])
            if trajectory_id not in trajectories:
                trajectories[trajectory_id] = Trajectory(trajectory_id)

            state = State(frame=state_info["frame_id"], x=state_info["x"], y=state_info["y"])
            state.set_velocity(state_info["vx"], state_info["vy"])
            trajectories[trajectory_id].add_state(state)

        for trajectory_id, trajectory in trajectories.items():
            if InteractionParser()._get_pedestrian_type(trajectory) == "pedestrians":
                pedestrians[trajectory_id] = Pedestrian(trajectory_id)
                pedestrians[trajectory_id].set_trajectory(trajectory)
            else:
                cyclists[trajectory_id] = Cyclist(trajectory_id)
                cyclists[trajectory_id].set_trajectory(trajectory)

        return pedestrians, cyclists

    @staticmethod
    def parse(file_id, folder_path, stamp_range):
        InteractionParser.parse_vehicle(file_id, folder_path, stamp_range)
        InteractionParser.parse_pedestrians(file_id, folder_path, stamp_range)
        return