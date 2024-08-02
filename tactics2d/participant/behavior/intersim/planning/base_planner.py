import copy
import logging
import math
import random
import time

import interactive_sim.envs.util as utils
import numpy as np
import plan.utils as plan_helper
from plan.env_planner import Agent, EnvPlanner, SudoInterpolator

S0 = 3
T = 0.25  # 1.5  # reaction time when following
DELTA = 4  # the power term in IDM
PLANNING_HORIZON = 5  # in frames
PREDICTION_HTZ = 10  # prediction_htz
T_HEADWAY = 0.2
A_SPEEDUP_DESIRE = 0.3  # A
A_SLOWDOWN_DESIRE = 1.5  # B
XPT_SHRESHOLD = 0.7
MINIMAL_DISTANCE_PER_STEP = 0.05
MINIMAL_DISTANCE_TO_TRAVEL = 4
# MINIMAL_DISTANCE_TO_RESCALE = -999 #0.1
REACTION_AFTER = 200  # in frames
MINIMAL_SCALE = 0.3
MAX_DEVIATION_FOR_PREDICTION = 4
TRAFFIC_LIGHT_COLLISION_SIZE = 2

MINIMAL_SPEED_TO_TRACK_ORG_GOAL = 5
MINIMAL_DISTANCE_TO_GOAL = 15
PRINT_TIMER = False

DEFAULT_SPEED = 75  # in mph


class BasePlanner(EnvPlanner):
    """
    BasePlanner should not use the gt future trajectory information for planning.
    Using the gt trajectory for collision avoidance is not safe.
    The future trajectory might be changed after the ego planner's planning by the env planner.
    The BasePlanner has its own predictor which does not share information with the predictor of the EnvPlanner
    The BasePlanner is used to control the ego agent only.
    The BasePlanner is derived from the EnvPlanner, change predefined functions to build/test your own planner.
    """

    def plan_marginal_trajectories(
        self, current_state, current_frame_idx, ego_agent_id, my_current_pose, my_current_v_per_step
    ):
        current_state["agent"][ego_agent_id]["action"] = "follow"
        if my_current_v_per_step > 1 * self.frame_rate:
            my_current_v_per_step = 0.1 * self.frame_rate
        elif my_current_v_per_step < 0.001 * self.frame_rate:
            my_current_v_per_step = 0

        if PRINT_TIMER:
            last_tic = time.perf_counter()

        current_routes = (
            current_state["predicting"]["route"][ego_agent_id]
            if ego_agent_id in current_state["predicting"]["route"]
            else []
        )
        my_trajs, current_routes = self.get_reroute_traj(
            current_state=current_state,
            agent_id=ego_agent_id,
            current_frame_idx=current_frame_idx,
            dynamic_turnings=True,
            current_route=current_routes,
            is_ego=True,
        )
        if ego_agent_id not in current_state["predicting"]["route"]:
            current_state["predicting"]["route"][ego_agent_id] = current_routes
            # draw goal point
            goal_pt, goal_yaw = self.online_predictor.data["predicting"]["goal_pts"][ego_agent_id]
            current_state["predicting"]["mark_pts"] = [goal_pt]

        if not self.current_on_road:
            print("TEST OFF ROAD!!!!!!!!")

        if PRINT_TIMER:
            print(f"Time spent on ego reroute:  {time.perf_counter() - last_tic:04f}s")
            last_tic = time.perf_counter()

        my_interpolators = []
        my_interpolated_marginal_trajectories = []
        for my_traj in my_trajs:
            my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)
            my_interpolated_trajectory = self.get_trajectory_from_interpolator(
                my_interpolator=my_interpolator,
                my_current_speed=my_current_v_per_step,
                agent_id=ego_agent_id,
            )
            my_traj = my_interpolated_trajectory[:, :2]
            my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)
            my_interpolators.append(my_interpolator)
            my_interpolated_marginal_trajectories.append(my_interpolated_trajectory)
        return my_interpolators, my_interpolated_marginal_trajectories, current_routes

    def make_predictions(self, current_state, current_frame_idx, ego_agent_id):
        other_agent_traj = []
        other_agent_ids = []
        # prior for ped and cyclist
        prior_agent_traj = []
        prior_agent_ids = []

        if self.predict_env_for_ego_collisions:
            # use constant velocity
            for each_agent_id in current_state["agent"]:
                if each_agent_id == ego_agent_id:
                    continue
                varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                # varies = [1]
                for v in varies:
                    delta_x = (
                        current_state["agent"][each_agent_id]["pose"][current_frame_idx - 1, 0]
                        - current_state["agent"][each_agent_id]["pose"][current_frame_idx - 6, 0]
                    ) / 5
                    delta_y = (
                        current_state["agent"][each_agent_id]["pose"][current_frame_idx - 1, 1]
                        - current_state["agent"][each_agent_id]["pose"][current_frame_idx - 6, 1]
                    ) / 5
                    pred_traj_with_yaw = np.ones((self.planning_horizon, 4)) * -1
                    pred_traj_with_yaw[:, 3] = current_state["agent"][each_agent_id]["pose"][
                        current_frame_idx - 1, 3
                    ]
                    for t in range(self.planning_horizon):
                        pred_traj_with_yaw[t, 0] = (
                            current_state["agent"][each_agent_id]["pose"][current_frame_idx - 1, 0]
                            + t * delta_x * v
                        )
                        pred_traj_with_yaw[t, 1] = (
                            current_state["agent"][each_agent_id]["pose"][current_frame_idx - 1, 1]
                            + t * delta_y * v
                        )
                    # always yield with constant v
                    prior_agent_traj.append(pred_traj_with_yaw)
                    prior_agent_ids.append(each_agent_id)
        else:
            for each_agent in current_state["agent"]:
                if each_agent == ego_agent_id:
                    continue
                each_agent_pose = current_state["agent"][each_agent]["pose"]
                # check distance
                if (
                    utils.euclidean_distance(
                        current_state["agent"][ego_agent_id]["pose"][current_frame_idx - 1, :2],
                        each_agent_pose[current_frame_idx - 1, :2],
                    )
                    > 500
                    and current_state["agent"][ego_agent_id]["pose"][current_frame_idx - 1, 0] != -1
                ):  # 20m for 1 second on 70km/h
                    continue

                # 'predict' its trajectory by following lanes
                if int(current_state["agent"][each_agent]["type"]) not in self.vehicle_types:
                    # for pedestrians or bicycles
                    if (
                        each_agent_pose[current_frame_idx - 1, 0] == -1.0
                        or each_agent_pose[current_frame_idx - 6, 0] == -1.0
                    ):
                        continue
                    # for non-vehicle types agent
                    delta_x = (
                        each_agent_pose[current_frame_idx - 1, 0]
                        - each_agent_pose[current_frame_idx - 6, 0]
                    ) / 5
                    delta_y = (
                        each_agent_pose[current_frame_idx - 1, 1]
                        - each_agent_pose[current_frame_idx - 6, 1]
                    ) / 5
                    varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                    predict_horizon = 39  # in frames
                    for mul in varies:
                        traj_with_yaw = np.ones((self.planning_horizon, 4)) * -1
                        traj_with_yaw[:, 3] = each_agent_pose[current_frame_idx - 1, 3]
                        traj_with_yaw[0, :] = each_agent_pose[current_frame_idx, :]
                        for i in range(predict_horizon):
                            # constant v with variations
                            traj_with_yaw[i + 1, 0] = traj_with_yaw[i, 0] + min(0.5, delta_x * mul)
                            traj_with_yaw[i + 1, 1] = traj_with_yaw[i, 1] + min(0.5, delta_y * mul)
                        other_agent_ids.append(each_agent)
                        other_agent_traj.append(traj_with_yaw)
                else:
                    # for vehicles
                    if (
                        each_agent_pose[current_frame_idx - 1, 0] == -1.0
                        or each_agent_pose[current_frame_idx - 6, 0] == -1.0
                        or each_agent_pose[current_frame_idx - 11, 0] == -1.0
                    ):
                        continue

                    # for vehicle types agents
                    each_agent_current_pose = each_agent_pose[current_frame_idx - 1]
                    each_agent_current_v_per_step = (
                        utils.euclidean_distance(
                            each_agent_pose[current_frame_idx - 1, :2],
                            each_agent_pose[current_frame_idx - 6, :2],
                        )
                        / 5
                    )
                    each_agent_current_a_per_step = (
                        utils.euclidean_distance(
                            each_agent_pose[current_frame_idx - 1, :2],
                            each_agent_pose[current_frame_idx - 6, :2],
                        )
                        / 5
                        - utils.euclidean_distance(
                            each_agent_pose[current_frame_idx - 6, :2],
                            each_agent_pose[current_frame_idx - 11, :2],
                        )
                        / 5
                    ) / 5
                    if each_agent_current_v_per_step > 1 * self.frame_rate:
                        each_agent_current_v_per_step = 0.1 * self.frame_rate
                    # get the route for each agent, you can use your prediction model here
                    if each_agent_current_v_per_step < 0.025 * self.frame_rate:
                        each_agent_current_v_per_step = 0

                    if each_agent_current_a_per_step > 0.05 * self.frame_rate:
                        each_agent_current_a_per_step = 0.03 * self.frame_rate

                    current_lane, current_closest_pt_idx, dist_to_lane = self.find_closest_lane(
                        current_state=current_state,
                        agent_id=each_agent,
                        my_current_v_per_step=each_agent_current_v_per_step,
                        my_current_pose=each_agent_current_pose,
                    )

                    # detect parking vehicles
                    assert current_frame_idx >= self.frame_rate, current_frame_idx
                    steady_in_past = (
                        utils.euclidean_distance(
                            each_agent_pose[current_frame_idx - 1, :2],
                            each_agent_pose[current_frame_idx - self.frame_rate, :2],
                        )
                        < 3
                    )
                    if (
                        each_agent_current_v_per_step < 0.05
                        and (dist_to_lane is None or dist_to_lane > 2)
                        and steady_in_past
                    ):
                        dummy_steady = np.repeat(
                            each_agent_pose[current_frame_idx - 1, :][np.newaxis, :],
                            self.planning_horizon,
                            axis=0,
                        )
                        prior_agent_ids.append(each_agent)
                        prior_agent_traj.append(dummy_steady)
                        # current_state['agent'][each_agent]['marking'] = "Parking"
                        continue

                    # 2. search all possible route from this lane and add trajectory from the lane following model
                    # random shooting for all possible routes
                    if (
                        current_lane in current_state["road"]
                        and "speed_limit" in current_state["road"][current_lane]
                    ):
                        speed_limit = current_state["road"][current_lane]["speed_limit"]
                        my_target_speed = (
                            speed_limit
                            if speed_limit is not None
                            else plan_helper.mph_to_meterpersecond(DEFAULT_SPEED) / self.frame_rate
                        )
                    else:
                        my_target_speed = (
                            plan_helper.mph_to_meterpersecond(DEFAULT_SPEED) / self.frame_rate
                        )

                    routes = []
                    for _ in range(self.frame_rate):
                        lanes_in_a_route = [current_lane]
                        current_looping = current_lane
                        route_traj_left = np.array(
                            current_state["road"][current_looping]["xyz"][
                                current_closest_pt_idx + self.frame_rate :, :2
                            ],
                            ndmin=2,
                        )
                        next_lanes = current_state["road"][current_looping]["next_lanes"]
                        while len(next_lanes) > 0 and len(lanes_in_a_route) < 5:
                            lanes_in_a_route.append(current_looping)
                            current_looping = random.choice(next_lanes)
                            if current_looping not in current_state["road"]:
                                continue
                            next_lanes = current_state["road"][current_looping]["next_lanes"]
                            route_traj_left = np.concatenate(
                                (
                                    route_traj_left,
                                    current_state["road"][current_looping]["xyz"][:, :2],
                                )
                            )
                        if lanes_in_a_route not in routes:
                            routes.append(lanes_in_a_route)
                            varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                            for mul in varies:
                                other_interpolator = SudoInterpolator(
                                    route_traj_left.copy(), each_agent_current_pose
                                )
                                traj_with_yaw = self.get_trajectory_from_interpolator(
                                    my_interpolator=other_interpolator,
                                    my_current_speed=each_agent_current_v_per_step * mul,
                                    a_per_step=each_agent_current_a_per_step,
                                    desired_speed=my_target_speed,
                                    check_turning_dynamics=False,
                                )
                                other_agent_traj.append(traj_with_yaw)
                                other_agent_ids.append(each_agent)
        return other_agent_traj, other_agent_ids, prior_agent_traj, prior_agent_ids

    def plan_ego(self, current_state, current_frame_idx):
        current_state["predicting"]["emergency_stopping"] = False
        if self.is_planning(current_frame_idx) and current_state is not None:
            # load scenario data
            planner_tic = time.perf_counter()
            if "planner_timer" not in current_state:
                current_state["planner_timer"] = []
                current_state["predict_timer"] = []

            ego_agent_id = current_state["predicting"]["ego_id"][1]
            my_current_pose = current_state["agent"][ego_agent_id]["pose"][current_frame_idx - 1]
            my_current_v_per_step = (
                utils.euclidean_distance(
                    current_state["agent"][ego_agent_id]["pose"][current_frame_idx - 1, :2],
                    current_state["agent"][ego_agent_id]["pose"][current_frame_idx - 6, :2],
                )
                / 5
            )
            total_time_frame = current_state["agent"][ego_agent_id]["pose"].shape[0]
            goal_point = current_state["predicting"]["goal_pts"][ego_agent_id]
            my_target_speed = (
                DEFAULT_SPEED / self.frame_rate
            )  # change this to the speed limit of the current lane

            goal_pt, goal_yaw = self.online_predictor.get_goal(
                current_data=current_state, agent_id=ego_agent_id, dataset=self.dataset
            )
            goal_lane, _, _ = self.find_closest_lane(
                current_state=current_state,
                my_current_pose=[goal_pt[0], goal_pt[1], -1, goal_yaw],
                valid_lane_types=self.valid_lane_types,
            )

            (
                my_interpolators,
                my_interpolated_marginal_trajectories,
                current_routes,
            ) = self.plan_marginal_trajectories(
                current_state=current_state,
                current_frame_idx=current_frame_idx,
                ego_agent_id=ego_agent_id,
                my_current_pose=my_current_pose,
                my_current_v_per_step=my_current_v_per_step,
            )

            # deal with interactions
            # 1. make predictions
            (
                other_agent_traj,
                other_agent_ids,
                prior_agent_traj,
                prior_agent_ids,
            ) = self.make_predictions(
                current_state=current_state,
                current_frame_idx=current_frame_idx,
                ego_agent_id=ego_agent_id,
            )

            if PRINT_TIMER:
                print(
                    f"Time spent on ego planning other agents:  {time.perf_counter() - last_tic:04f}s"
                )
                last_tic = time.perf_counter()

            # check collisions with ego
            prior_collisions = []  # [[collision_frame, target_id], ..] from idx small to large
            collisions = []  # [[collision_frame, target_id], ..] from idx small to large
            # ego_org_traj = my_interpolated_trajectory

            def check_traffic_light(ego_org_traj):
                total_time_frame = self.planning_horizon
                for current_time in range(total_time_frame):
                    if current_frame_idx + current_time < 90:
                        traffic_light_ending_pts = self.get_traffic_light_collision_pts(
                            current_state=current_state,
                            current_frame_idx=current_frame_idx + min(5, current_time),
                        )
                    else:
                        traffic_light_ending_pts = []
                    ego_pose = ego_org_traj[current_time]
                    if abs(ego_pose[0]) < 1.1 and abs(ego_pose[1]) < 1.1:
                        continue
                    ego_agent = Agent(
                        x=ego_pose[0],
                        y=ego_pose[1],
                        yaw=ego_pose[3],
                        length=current_state["agent"][ego_agent_id]["shape"][0][1],
                        width=current_state["agent"][ego_agent_id]["shape"][0][0],
                        agent_id=ego_agent_id,
                    )

                    # check if ego agent is running a red light
                    if abs(ego_org_traj[-1, 0] + 1) < 0.01 or abs(ego_org_traj[0, 0] + 1) < 0.01:
                        ego_dist = 0
                    else:
                        ego_dist = utils.euclidean_distance(
                            ego_org_traj[-1, :2], ego_org_traj[0, :2]
                        )
                    if abs(ego_org_traj[60, 3] + 1) < 0.01:
                        ego_turning_right = False
                    else:
                        ego_yaw_diff = -utils.normalize_angle(
                            ego_org_traj[60, 3] - ego_org_traj[0, 3]
                        )
                        ego_running_red_light = False
                        if (
                            math.pi / 180 * 15 < ego_yaw_diff
                            and abs(ego_org_traj[60, 3] + 1) > 0.01
                        ):
                            ego_turning_right = True
                        else:
                            ego_turning_right = False

                    if not ego_turning_right and ego_dist > 10:
                        for tl_pt in traffic_light_ending_pts:
                            dummy_tf_agent = Agent(
                                x=tl_pt[0],
                                y=tl_pt[1],
                                yaw=0,
                                length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                agent_id=99999,
                            )
                            running = utils.check_collision(
                                checking_agent=ego_agent, target_agent=dummy_tf_agent
                            )
                            if running:
                                ego_running_red_light = True
                                return current_time
                return None

            def detect_conflicts_and_solve(
                others_trajectory,
                target_agent_ids,
                always_yield=False,
                ego_org_traj=my_interpolated_marginal_trajectories[0],
            ):
                total_time_frame = self.planning_horizon
                my_reactors = []
                for current_time in range(total_time_frame):
                    if current_frame_idx + current_time < self.planning_to:
                        traffic_light_ending_pts = self.get_traffic_light_collision_pts(
                            current_state=current_state,
                            current_frame_idx=current_frame_idx + min(5, current_time),
                        )
                    else:
                        traffic_light_ending_pts = []
                    ego_running_red_light = False
                    ego_time_length = ego_org_traj.shape[0]

                    if current_time >= ego_time_length:
                        print("break ego length: ", current_time, ego_time_length)
                        break
                    ego_pose = ego_org_traj[current_time]
                    if ego_pose[0] == -1.0 and ego_pose[1] == -1.0:
                        continue
                    ego_agent = Agent(
                        x=ego_pose[0],
                        y=ego_pose[1],
                        yaw=ego_pose[3],
                        length=current_state["agent"][ego_agent_id]["shape"][0][1],
                        width=current_state["agent"][ego_agent_id]["shape"][0][0],
                        agent_id=ego_agent_id,
                    )
                    # check if ego agent is running a red light
                    if ego_org_traj[-1, 0] == -1.0 or ego_org_traj[0, 0] == -1.0:
                        ego_dist = 0
                    else:
                        ego_dist = utils.euclidean_distance(
                            ego_org_traj[-1, :2], ego_org_traj[0, :2]
                        )
                    if ego_org_traj[20, 3] == -1.0:
                        ego_turning_right = False
                    else:
                        ego_yaw_diff = -utils.normalize_angle(
                            ego_org_traj[20, 3] - ego_org_traj[0, 3]
                        )
                        ego_running_red_light = False
                        if math.pi / 180 * 15 < ego_yaw_diff and ego_org_traj[20, 3] != -1.0:
                            ego_turning_right = True
                        else:
                            ego_turning_right = False
                    if not ego_turning_right and ego_dist > 10:
                        for tl_pt in traffic_light_ending_pts:
                            dummy_tf_agent = Agent(
                                x=tl_pt[0],
                                y=tl_pt[1],
                                yaw=0,
                                length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                agent_id=99999,
                            )
                            running = utils.check_collision(
                                checking_agent=ego_agent, target_agent=dummy_tf_agent
                            )
                            if running:
                                ego_running_red_light = True
                                break

                    if ego_running_red_light:
                        earliest_collision_idx = current_time
                        collision_point = ego_org_traj[current_time, :2]
                        earliest_conflict_agent = 99999
                        target_speed = 0
                        each_other_traj, detected_relation = None, None
                        return [
                            earliest_collision_idx,
                            collision_point,
                            earliest_conflict_agent,
                            target_speed,
                            None,
                            None,
                        ]

                    for j, each_other_traj in enumerate(others_trajectory):
                        target_agent_id = target_agent_ids[j]

                        # Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
                        target_type = int(current_state["agent"][target_agent_id]["type"])
                        if target_type not in self.vehicle_types:
                            target_shape = [
                                max(2, current_state["agent"][target_agent_id]["shape"][0][0]),
                                max(6, current_state["agent"][target_agent_id]["shape"][0][1]),
                            ]
                        else:
                            target_shape = [
                                max(1, current_state["agent"][target_agent_id]["shape"][0][0]),
                                max(1, current_state["agent"][target_agent_id]["shape"][0][1]),
                            ]

                        if target_agent_id in my_reactors:
                            continue
                        total_frame_in_target = each_other_traj.shape[0]
                        if current_time > total_frame_in_target - 1:
                            continue
                        target_pose = each_other_traj[current_time]
                        if target_pose[0] == -1.0 or target_pose[1] == -1.0:
                            continue

                        # check if target agent is running a red light
                        yaw_diff = utils.normalize_angle(
                            each_other_traj[-1, 3] - each_other_traj[0, 3]
                        )
                        dist = utils.euclidean_distance(
                            each_other_traj[-1, :2], each_other_traj[0, :2]
                        )
                        target_agent = Agent(
                            x=target_pose[0],
                            y=target_pose[1],
                            yaw=target_pose[3],
                            length=target_shape[1],
                            width=target_shape[0],
                            agent_id=target_agent_id,
                        )
                        # check target agent is stopping for a red light
                        running_red_light = False
                        if dist > 10:
                            for tl_pt in traffic_light_ending_pts:
                                dummy_tf_agent = Agent(
                                    x=tl_pt[0],
                                    y=tl_pt[1],
                                    yaw=0,
                                    length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                    width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                    agent_id=99999,
                                )
                                running = utils.check_collision(
                                    checking_agent=target_agent, target_agent=dummy_tf_agent
                                )
                                if running:
                                    # check if they are on two sides of the red light
                                    ego_tf_yaw = plan_helper.get_angle_of_a_line(
                                        pt1=ego_pose[:2], pt2=tl_pt[:2]
                                    )
                                    target_tf_yae = plan_helper.get_angle_of_a_line(
                                        pt1=target_pose[:2], pt2=tl_pt[:2]
                                    )
                                    if (
                                        abs(utils.normalize_angle(ego_tf_yaw - target_tf_yae))
                                        < math.pi / 2
                                    ):
                                        running_red_light = True
                                        break

                        if running_red_light:
                            continue

                        # check collision with ego vehicle
                        has_collision = utils.check_collision(
                            checking_agent=ego_agent, target_agent=target_agent
                        )

                        if current_time < ego_time_length - 1:
                            ego_pose2 = ego_org_traj[current_time + 1]
                            ego_agent2 = Agent(
                                x=(ego_pose2[0] + ego_pose[0]) / 2,
                                y=(ego_pose2[1] + ego_pose[1]) / 2,
                                yaw=plan_helper.get_angle_of_a_line(ego_pose[:2], ego_pose2[:2]),
                                length=max(
                                    2, utils.euclidean_distance(ego_pose2[:2], ego_pose[:2])
                                ),
                                width=current_state["agent"][ego_agent_id]["shape"][0][0],
                                agent_id=ego_agent_id,
                            )
                            if current_time < total_time_frame - 1:
                                target_pose2 = each_other_traj[current_time + 1]
                                target_agent2 = Agent(
                                    x=target_pose2[0],
                                    y=target_pose2[1],
                                    yaw=target_pose2[3],
                                    length=target_shape[1],
                                    width=target_shape[0],
                                    agent_id=target_agent_id,
                                )
                                has_collision |= utils.check_collision(
                                    checking_agent=ego_agent2, target_agent=target_agent2
                                )
                            else:
                                has_collision |= utils.check_collision(
                                    checking_agent=ego_agent2, target_agent=target_agent
                                )

                        if has_collision:
                            if not always_yield:
                                # FORWARD COLLISION CHECKINGS
                                target_pose_0 = each_other_traj[0]
                                target_agent_0 = Agent(
                                    x=target_pose_0[0],
                                    y=target_pose_0[1],
                                    yaw=target_pose_0[3],
                                    length=target_shape[1],
                                    width=target_shape[0],
                                    agent_id=target_agent_id,
                                )
                                collision_0 = False
                                for fcc_time in range(total_time_frame):
                                    ego_pose = ego_org_traj[fcc_time]
                                    if ego_pose[0] == -1.0 and ego_pose[1] == -1.0:
                                        continue
                                    ego_agent = Agent(
                                        x=ego_pose[0],
                                        y=ego_pose[1],
                                        yaw=ego_pose[3],
                                        length=current_state["agent"][ego_agent_id]["shape"][0][1],
                                        width=current_state["agent"][ego_agent_id]["shape"][0][0],
                                        agent_id=ego_agent_id,
                                    )

                                    collision_0 |= utils.check_collision(ego_agent, target_agent_0)
                                    if collision_0:
                                        break

                                ego_pose_0 = ego_org_traj[0]
                                ego_agent_0 = Agent(
                                    x=ego_pose_0[0],
                                    y=ego_pose_0[1],
                                    yaw=ego_pose_0[3],
                                    length=current_state["agent"][ego_agent_id]["shape"][0][1],
                                    width=current_state["agent"][ego_agent_id]["shape"][0][0],
                                    agent_id=ego_agent_id,
                                )
                                collision_1 = False
                                for fcc_time in range(total_time_frame):
                                    target_pose = each_other_traj[fcc_time]
                                    if target_pose[0] == -1.0 or target_pose[1] == -1.0:
                                        continue
                                    target_agent = Agent(
                                        x=target_pose[0],
                                        y=target_pose[1],
                                        yaw=target_pose[3],
                                        length=target_shape[1],
                                        width=target_shape[0],
                                        agent_id=target_agent_id,
                                    )

                                    collision_1 |= utils.check_collision(target_agent, ego_agent_0)
                                    if collision_1:
                                        break
                                # collision_1 = utils.check_collision(target_agent, ego_agent_0)
                                # collision_1 |= utils.check_collision(target_agent2, ego_agent_0)

                                if collision_0 and self.predict_with_rules:
                                    # yield
                                    detected_relation = [[target_agent_id, ego_agent_id, "FCC"]]
                                elif collision_1 and self.predict_with_rules:
                                    # pass
                                    my_reactors.append(target_agent_id)
                                    continue
                                else:
                                    # check relation
                                    # if collision, solve conflict
                                    predict_tic = time.perf_counter()
                                    self.online_predictor.predict_one_time(
                                        each_pair=[ego_agent_id, target_agent_id],
                                        current_frame=current_frame_idx,
                                        clear_history=True,
                                        with_rules=self.predict_with_rules,
                                        current_data=current_state,
                                    )
                                    detected_relation = self.online_predictor.data["predicting"][
                                        "relation"
                                    ]
                                    predict_time = time.perf_counter() - predict_tic
                                    current_state["predict_timer"].append(predict_time)

                                    if [ego_agent_id, target_agent_id] in detected_relation:
                                        if [target_agent_id, ego_agent_id] in detected_relation:
                                            # bi-directional relations, still yield
                                            pass
                                        else:
                                            # not to yield, and skip conflict
                                            my_reactors.append(target_agent_id)
                                            continue
                            else:
                                detected_relation = [[target_agent_id, ego_agent_id, "Prior"]]

                            copy = []
                            for each_r in detected_relation:
                                if len(each_r) == 2:
                                    copy.append([each_r[0], each_r[1], "predict"])
                                else:
                                    copy.append(each_r)
                            detected_relation = copy

                            earliest_collision_idx = current_time
                            collision_point = ego_org_traj[current_time, :2]
                            earliest_conflict_agent = target_agent_id

                            if total_frame_in_target - current_time > 5:
                                target_speed = (
                                    utils.euclidean_distance(
                                        each_other_traj[current_time, :2],
                                        each_other_traj[current_time + 5, :2],
                                    )
                                    / 5
                                )
                            elif current_time > 5:
                                target_speed = (
                                    utils.euclidean_distance(
                                        each_other_traj[current_time - 5, :2],
                                        each_other_traj[current_time, :2],
                                    )
                                    / 5
                                )
                            else:
                                target_speed = 0
                            return [
                                earliest_collision_idx,
                                collision_point,
                                earliest_conflict_agent,
                                target_speed,
                                each_other_traj,
                                detected_relation,
                            ]

                return None

            earliest_collision_idx = None
            collision_point = None
            earliest_conflict_agent = None
            target_speed = None
            detected_relation = None

            tf_light_frame_idx = check_traffic_light(my_interpolated_marginal_trajectories[0])

            # save trajectories
            # for p, each_traj in enumerate(other_agent_traj):
            #     this_agent_id = other_agent_ids[p]
            #     if this_agent_id not in current_state['predicting']['guilded_trajectory']:
            #         current_state['predicting']['guilded_trajectory'][this_agent_id] = []
            #     current_state['predicting']['guilded_trajectory'][this_agent_id] += [each_traj]

            # process prior collision pairs
            progress_for_all_traj = []
            trajectory_to_mark = []
            interpolators = []
            rsts_to_yield = []
            routes_to_yield = []
            target_length = []
            current_state["predicting"]["all_relations_last_step"] = []
            to_yield = []
            has_goal_index = None

            for i, each_route in enumerate(current_routes):
                if goal_lane in each_route:
                    has_goal_index = i
                    break

            for i, each_traj in enumerate(my_interpolated_marginal_trajectories):
                my_interpolator = my_interpolators[i]
                selected_route = current_routes[i]
                traj_to_mark_this_traj = None
                to_yield_this_traj = False
                rst = detect_conflicts_and_solve(
                    others_trajectory=prior_agent_traj,
                    target_agent_ids=prior_agent_ids,
                    always_yield=True,
                    ego_org_traj=each_traj,
                )
                if rst is not None and rst[5] is not None:
                    current_state["predicting"]["all_relations_last_step"] += rst[5]
                    (
                        earliest_collision_idx,
                        collision_point,
                        earliest_conflict_agent,
                        target_speed,
                        each_other_traj,
                        detected_relation,
                    ) = rst
                    if each_other_traj is not None:
                        traj_to_mark_this_traj = each_other_traj
                        # current_state['predicting']['trajectory_to_mark'].append(each_other_traj)
                    to_yield_this_traj = True
                # check collisions with not prior collisions
                rst = detect_conflicts_and_solve(
                    other_agent_traj,
                    other_agent_ids,
                    always_yield=(not self.predict_relations_for_ego),
                    ego_org_traj=each_traj,
                )
                if rst is not None and rst[5] is not None:
                    current_state["predicting"]["all_relations_last_step"] += rst[5]
                if rst is not None and len(rst) == 6:
                    if to_yield_this_traj:
                        if rst[0] < earliest_collision_idx:
                            consider = True
                        else:
                            consider = False
                    else:
                        consider = True
                        to_yield_this_traj = True
                    if consider:
                        (
                            earliest_collision_idx,
                            collision_point,
                            earliest_conflict_agent,
                            target_speed,
                            each_other_traj,
                            detected_relation,
                        ) = rst
                        if each_other_traj is not None:
                            traj_to_mark_this_traj = each_other_traj

                if not to_yield_this_traj:
                    # then take current trajectory
                    progress_for_all_traj.append(-1)
                    trajectory_to_mark.append(None)
                    rsts_to_yield.append(None)
                    interpolators.append(None)
                    routes_to_yield.append(None)
                    target_length.append(None)
                    to_yield.append(False)
                else:
                    progress_for_all_traj.append(earliest_collision_idx)
                    # mark other's trajectory to yield
                    trajectory_to_mark.append(traj_to_mark_this_traj)
                    rsts_to_yield.append(
                        [
                            earliest_collision_idx,
                            collision_point,
                            earliest_conflict_agent,
                            target_speed,
                            each_other_traj,
                            detected_relation,
                        ]
                    )
                    interpolators.append(my_interpolator)
                    routes_to_yield.append(selected_route)
                    if earliest_conflict_agent == 99999:
                        # shape for traffic light
                        target_length.append(2)
                    else:
                        target_shapes = current_state["agent"][earliest_conflict_agent]["shape"]
                        if len(target_shapes.shape) == 2:
                            if target_shapes.shape[0] == 1:
                                target_length.append(target_shapes[0, 1])
                            else:
                                try:
                                    target_length.append(target_shapes[earliest_collision_idx, 1])
                                except:
                                    print("Unknown shape size: ", target_shapes.shape)
                                    target_length.append(target_shapes[0, 1])
                        else:
                            target_length.append(target_shapes[1])
                    to_yield.append(True)

            ego_to_yield = False
            if has_goal_index is not None:
                # choose goal route
                index_to_select = has_goal_index
                ego_to_yield |= to_yield[has_goal_index]
            else:
                any_not_yield = False
                not_yield_index = None
                for i, each_yield in enumerate(to_yield):
                    any_not_yield |= not each_yield
                    if any_not_yield:
                        not_yield_index = i
                        break
                if any_not_yield:
                    ego_to_yield = False
                    index_to_select = not_yield_index
                else:
                    # all yields, choose furtheest one
                    ego_to_yield = True
                    index_to_select = progress_for_all_traj.index(max(progress_for_all_traj))

            if ego_to_yield:
                if trajectory_to_mark[index_to_select] is not None:
                    # for traffic light, this each other traj will be None
                    current_state["predicting"]["trajectory_to_mark"].append(
                        trajectory_to_mark[index_to_select]
                    )
                (
                    earliest_collision_idx,
                    collision_point,
                    earliest_conflict_agent,
                    target_speed,
                    each_other_traj,
                    detected_relation,
                ) = rsts_to_yield[index_to_select]
                my_interpolator = interpolators[index_to_select]
                my_traj = my_interpolated_marginal_trajectories[index_to_select]
                selected_route = routes_to_yield[index_to_select]
                S0 = target_length[index_to_select] / 2 * 1.5
            else:
                current_state["predicting"]["trajectory_to_mark"].append(
                    my_interpolated_marginal_trajectories[index_to_select]
                )
                # my_interpolators = interpolators[index_to_select]
                my_traj = my_interpolated_marginal_trajectories[index_to_select]
                selected_route = current_routes[index_to_select]

            if len(selected_route) > 0:
                current_lane = selected_route[0]
                current_lane_speed_limit = (
                    current_state["road"][current_lane]["speed_limit"]
                    if current_lane in current_state["road"]
                    and "speed_limit" in current_state["road"][current_lane]
                    else None
                )
                if current_lane_speed_limit is not None:
                    my_target_speed = (
                        plan_helper.mph_to_meterpersecond(current_lane_speed_limit)
                        / self.frame_rate
                    )

            if earliest_collision_idx is not None and (
                tf_light_frame_idx is None or earliest_collision_idx < tf_light_frame_idx
            ):
                # data to save
                if detected_relation is not None:
                    if "relations_per_frame_ego" not in current_state["predicting"]:
                        current_state["predicting"]["relations_per_frame_ego"] = {}
                    for dt in range(self.planning_interval):
                        if (current_frame_idx + dt) not in current_state["predicting"][
                            "relations_per_frame_ego"
                        ]:
                            current_state["predicting"]["relations_per_frame_ego"][
                                current_frame_idx + dt
                            ] = []
                        current_state["predicting"]["relations_per_frame_ego"][
                            current_frame_idx + dt
                        ] += detected_relation
            elif tf_light_frame_idx is not None:
                earliest_collision_idx = tf_light_frame_idx
                # collision_point = ego_org_traj[earliest_collision_idx, :2]
                earliest_conflict_agent = 99999
                target_speed = 0
                detected_relation = None
                ego_to_yield = True

            if ego_to_yield:
                distance_to_minuse = S0
                if earliest_conflict_agent == 99999:
                    distance_to_minuse = 0.1
                distance_to_travel = (
                    my_interpolator.get_distance_with_index(earliest_collision_idx)
                    - distance_to_minuse
                )
                stopping_point = my_interpolator.interpolate(
                    distance_to_travel - distance_to_minuse
                )[:2]

                # current_state['predicting']['mark_pts'] = [stopping_point]
                if distance_to_travel < MINIMAL_DISTANCE_PER_STEP:
                    planed_traj = self.get_trajectory_from_interpolator(
                        my_interpolator=my_interpolator,
                        my_current_speed=my_current_v_per_step,
                        desired_speed=my_target_speed,
                        emergency_stop=True,
                    )
                    current_state["predicting"]["emergency_stopping"] = True
                    # planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                    #                                           reactor_traj=my_traj,
                    #                                           reactor_interpolator=my_interpolator,
                    #                                           scale=scale,
                    #                                           current_v_per_step=my_current_v_per_step,
                    #                                           target_speed=my_target_speed)
                elif (
                    my_current_v_per_step < 1 / self.frame_rate
                    and utils.euclidean_distance(my_traj[0, :2], my_traj[-1, :2])
                    < MINIMAL_DISTANCE_TO_TRAVEL
                ):
                    planed_traj = self.get_trajectory_from_interpolator(
                        my_interpolator=my_interpolator,
                        my_current_speed=my_current_v_per_step,
                        desired_speed=my_target_speed,
                        hold_still=True,
                    )
                else:
                    planed_traj = self.adjust_speed_for_collision(
                        interpolator=my_interpolator,
                        distance_to_end=distance_to_travel,
                        current_v=my_current_v_per_step,
                        end_point_v=min(my_current_v_per_step * 0.8, target_speed),
                    )
                    assert len(planed_traj.shape) > 1, planed_traj.shape
                    # my_interpolator = SudoInterpolator(my_traj, my_current_pose)
                    # planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                    #                                           reactor_traj=my_traj,
                    #                                           reactor_interpolator=my_interpolator,
                    #                                           scale=1,
                    #                                           current_v_per_step=my_current_v_per_step,
                    #                                           target_speed=my_target_speed)
            else:
                if (
                    utils.euclidean_distance(my_traj[0, :2], my_traj[-1, :2])
                    < MINIMAL_DISTANCE_TO_TRAVEL
                ):
                    planed_traj = self.get_trajectory_from_interpolator(
                        my_interpolator=my_interpolator,
                        my_current_speed=my_current_v_per_step,
                        desired_speed=my_target_speed,
                        hold_still=True,
                    )
                else:
                    planed_traj = my_traj

            if PRINT_TIMER:
                print(f"Time spent on adjust speed:  {time.perf_counter() - last_tic:04f}s")
                last_tic = time.perf_counter()

            current_state["predicting"]["trajectory_to_mark"].append(planed_traj)

            if planed_traj.shape[0] < self.planning_horizon:
                planed_traj = self.get_trajectory_from_interpolator(
                    my_interpolator=my_interpolator,
                    my_current_speed=my_current_v_per_step,
                    desired_speed=my_target_speed,
                    hold_still=True,
                )
            assert planed_traj.shape[0] >= self.planning_horizon, planed_traj.shape

            planning_rst_horizon, _ = planed_traj.shape
            jumping_idx = self.assert_traj(planed_traj[: total_time_frame - current_frame_idx, :2])
            if jumping_idx != -1:
                if jumping_idx >= 6:
                    planed_traj[jumping_idx:, :] = -1
                else:
                    print(f"Early jumping {jumping_idx} {ego_agent_id}")
                    # assert False, f'Jumping early: {jumping_idx} {ego_agent_id}'
            planned_shape = planed_traj[: total_time_frame - current_frame_idx, :].shape
            if (
                total_time_frame - current_frame_idx > 1
                and len(planned_shape) > 1
                and planned_shape[1] == 4
            ):
                current_state["agent"][ego_agent_id]["pose"][
                    current_frame_idx : planning_rst_horizon + current_frame_idx, :
                ] = planed_traj[: total_time_frame - current_frame_idx, :]
                current_state["agent"][ego_agent_id]["pose"][
                    planning_rst_horizon + current_frame_idx :, :
                ] = -1
            else:
                print(
                    "WARNING: No planning trajectory replacing!! ",
                    total_time_frame,
                    current_frame_idx,
                    planned_shape,
                    planed_traj,
                )

            planner_time = time.perf_counter() - planner_tic
            current_state["planner_timer"].append(planner_time)

        return current_state
