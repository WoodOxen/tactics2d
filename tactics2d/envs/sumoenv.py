from mimetypes import init
import os, sys

SUMO_TOOLS_DIR = "/home/rowena/sumo/tools"
try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
  
import traci
import sumolib

from random import randint, sample
import xml.dom.minidom as minidom
import math
import json
import pandas as pd
import numpy as np
import gym

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ
     
class SumoEnv(gym.Env):
    """SUMO environment for RL training
    
    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param begin_time: (int) The time step (in seconds) the simulation starts
    """
    CONNECTION_LABEL = 0 # TODO: multi-client support
    # :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    def __init__(self,
                 scenario: str,
                 file_list: list,
                 data_dir: str,
                 init_only: bool=False,
                 use_gui: bool=False,
                 delay: float=400,
                 sumo_warnings: bool=True):
        
        if not os.path.exists(self.data_dir):
            raise ValueError("Cannot find the directory that contains the dataset!")
        
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')
        
        self.scenario=scenario
        self.file_list = file_list
        self.data_dir=data_dir
        self.init_only = init_only
        self.delay = delay
        self.sumo_warnings = sumo_warnings
        self.label = str(SumoEnv.CONNECTION_LABEL)
        SumoEnv.CONNECTION_LABEL += 1
            
        with open("./config/%s.config" % scenario) as f_config:
            self.data_config = json.load(f_config)
        self.range_file = self.data_config["num"]
        
        self._route = None
        self.df = None
        self.df_init = None
        self._net = None
        self._route = None
        self._step = 0
        self._begin_time = 0
        self.vehicle_NPC = set()
        self.done = False
        self.agent_id = None
        self.agent_state = {}
        
        self.perception_range = 50

    def _clear_env(self):
        """Clear cache of the last run"""
        self._route = None
        self.df = None
        self.df_init = None
        self._net = None
        self._route = None
        self._step = 0
        self._begin_time = 0
        self.vehicle_NPC = self.vehicle_NPC.clear()
        
    def _start_simulation(self):
        """Initialize simulation environment"""
        # Randomly choose a train file, load its data, and find its corresponding map.
        file_id = sample(self.file_list)
        for key, value in self.data_config.items():
            if self.scenario in key:
                if value in value["ids"]:
                    map_name = key
                    break
        self._net = "./net/%s.net.xml" % map_name
        self._route = "./net/%s.rou.xml" % map_name
        
        self.df = pd.read_csv("%s/%s/traci/%02d_tracks.csv" % (self.data_dir, self.scenario, file_id), dtype={"edge": str, "id": str})
        self.df_init = pd.read_csv("%s/%s/traci/%02d_tracksMeta.csv" % (self.data_dir, self.scenario, file_id), dtype={"route": str, "id": str})
        
        # Because the NPC vehicles are initialized (and updated) based on real trajectories, we need to guarantee that the agent car runs on a feasible route. Therefore, this environment will create the agent by replacing an arbitrary NPC vehicle.
        # To make the route as complete as possible (allow more steps for simulation), the vehicles starting at frame 0 or stop at the last frame are filtered.
        last_frame = np.max(self.df["frame"])
        while True:
            agent = self.df.sample().iloc[0]
            if agent["initialFrame"] > 0 and agent["finalFrame"] < last_frame:
                self.vehicle_agent = agent["id"]
                self._begin_time = agent["initialFrame"]
                self._step = self._begin_time
                break

        # Generate sumo commands and start the simulation.
        sumo_cmd = [self._sumo_binary,
                    "-n", self._net,
                    "-r", self._route,
                    "-a", "./config/vehicles.xml",
                    "-b", self._begin_time
                    ]
        if self.use_gui:
            sumo_cmd.append([
                "--start","--quit-on-end",
                "-g", "./config/visualization.settings",
                "-d", self.delay])

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
            
    def _initialize_agent(self):
        veh_meta = self.df_init[self.df_init["id"]==self.vehicle_agent].iloc[0]
        veh_info = self.df[(self.df["id"]==self.vehicle_agent) & (self.df["frame"]==self._begin_time)].iloc[0]
        traci.vehicle.add(
            vehID=self.vehicle_agent,
            routeID=veh_meta["route"],
            typeID=veh_meta["vtype"],
            departSpeed=veh_meta["departSpeed"]
        )
        traci.vehicle.setWidth(veh_meta["width"])
        traci.vehicle.setHeight(veh_meta["height"])
        traci.vehicle.moveToXY(
            vehID=self.vehicle_agent,
            edgeID=veh_info["edge"],
            lane=veh_info["lane"],
            x=veh_info["x"],
            y=veh_info["y"],
            angle=veh_info["angle"]
        )

    def _add_vehicles(self):
        """Initialize the NPC vehicles."""
        df_sub = self.df[(self.df["frame"]==self._step) & (self.df["id"]!=self.vehicle_agent)]
        for _, line in df_sub.iterrows():
            if line["id"] not in self.vehicles:
                line_init = self.df_init[self.df_init["id"]==line["id"]].iloc[0]
                traci.vehicle.add(
                    vehID=line["id"],
                    routeID=line_init["route"],
                    typeID=line_init["vtype"],
                    departSpeed=line["speed"],
                    arrivalSpeed=line_init["arrivalSpeed"]
                )
                traci.vehicle.setLength(vehID=line["id"], length=line_init["length"])
                traci.vehicle.setWidth(vehID=line["id"], width=line_init["width"])
                self.vehicle_NPC.add(line["id"])
    
    def _remove_vehicles(self):
        """Remove arrived NPC vehicles"""
        vehicle_to_remove = self.df_init[self.df_init["finalFrame"]==self._step-1]["id"].tolist()
        vehicle_existing = traci.vehicle.getIDList()
        for id in vehicle_to_remove:
            if id in vehicle_existing:
                traci.vehicle.remove(vehID=id)
            self.vehicle_NPC.remove(id)
    
    def _update_states(self):
        """Update NPC vehicles' state
        According to the document of traci, the priority of moveToXY is higher than slowDown. Calling slowDown is to make the trajectory smoother.
        TODO: 
        possible improvement 1: merge these two functions.
        possible improvement 2: allow update of state over unstable time step. 
        """
        if not self.init_only:
            df_sub = self.df[self.df["frame"] == self._step]
            for id in self.vehicle_NPC:
                veh_info = df_sub[df_sub["id"]==id].iloc[0]
                traci.vehicle.slowDown(
                    vehID=id,
                    speed=veh_info["speed"],
                    duration=0.04
                )
                traci.vehicle.moveToXY(
                    vehID=id,
                    edgeID=veh_info["edge"],
                    lane=veh_info["lane"],
                    x=veh_info["x"],
                    y=veh_info["y"],
                    angle=veh_info["angle"]
                )
                
    def _obtain_observation(self):
        """Obtain observation as input to the RL models.
        Three section of observation:
        1. agent's own state:
        2. the neighbored NPCs' state:
        3. the route and lane information #TODO
        """
        observation = {}
        
        # get agent's state
        agent_state = {
            "speed": traci.vehicle.getSpeed(self.vehicle_agent),
            "angle": traci.vehicle.getAngle(self.vehicle_agent),
            "width": traci.vehicle.getHeight(self.vehicle_agent),
            "length": traci.vehicle.getLength(self.vehicle_agent),
        }
        pos_agent = traci.vehicle.getPosition(self.vehicle_agent)
        observation["agent"] = agent_state
        
        # get neighbors' state
        vehicle_existing = traci.vehicle.getIDList()
        observation["npc"] = []
        for id in self.vehicle_NPC:
            if id in vehicle_existing:
                pos_npc = traci.vehicle.getPosition(id)
                pos_relative = [pos_agent[0]-pos_npc[0], pos_agent[1]-pos_npc[1]]
                distance = math.sqrt(pos_relative[0]**2 + pos_relative[1]**2)
                # if the npc vehicle is a neighbor of the agent, record it into observation
                if distance <= self.perception_range:
                    npc_state = {
                        "id": id,
                        "vtype": traci.vehicle.getVehicleClass(id),
                        "speed": traci.vehicle.getSpeed(id),
                        "angle": traci.vehicle.getAngle(id),
                        "width": traci.vehicle.getHeight(id),
                        "length": traci.vehicle.getLength(id),
                        "distance": distance,
                        "pos_relative": pos_relative
                    }
                    observation["npc"].append(npc_state)
                    
        # get road information
        
        return observation
    
    def _apply_actions(self, action):
        pass
    
    def reset(self):
        self._clear_env()
        self._start_simulation()
        # add agent and NPCs into the environment
        self._initialize_agent()
        self._add_vehicles()
        self._update_states()
        observation = self._obtain_observation()
        # update simulation step
        self._step += 1
        return observation
    
    def step(self, action):
        observations = self._obtain_observation(self.agent)
        stop = self._to_stop()
        if stop:
            return observations, stop
        self._add_vehicles()
        self._remove_vehicles()
        self._apply_actions(action)
        self._update_states()

        return observations, stop
    
    def render(self):
        pass
    
    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        # try:
        #     self.disp.stop()
        # except AttributeError:
        #     pass
        self.sumo = None


class SumoMultiAgentEnv(SumoEnv):
    def __init__(self):
        pass