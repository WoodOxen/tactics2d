import math

import interactive_sim.envs.util as utils


def mph_to_meterpersecond(mph):
    return mph * 0.4472222222


def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle


def get_current_pose_and_v(current_state, agent_id, current_frame_idx):
    my_current_pose = current_state["agent"][agent_id]["pose"][current_frame_idx - 1]
    if (
        current_state["agent"][agent_id]["pose"][current_frame_idx - 1, 0] == -1
        or current_state["agent"][agent_id]["pose"][current_frame_idx - 6, 0] == -1
    ):
        my_current_v_per_step = 0
        print("Past invalid for ", agent_id, " and setting v to 0")
    else:
        my_current_v_per_step = (
            utils.euclidean_distance(
                current_state["agent"][agent_id]["pose"][current_frame_idx - 1, :2],
                current_state["agent"][agent_id]["pose"][current_frame_idx - 6, :2],
            )
            / 5
        )
    return my_current_pose, my_current_v_per_step
