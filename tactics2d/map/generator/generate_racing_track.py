import numpy as np
from shapely.geometry import Point, LineString, LinearRing
from shapely.affinity import affine_transform

from tactics2d.utils.get_circle import get_circle
from tactics2d.map.element import Lane, Map


# Track related configurations
N_CHECKPOINT = (10, 20) # the number of turns is ranging in 10-20
TRACK_WIDTH = 15 # the width of the track is ranging in 15m
TRACK_RAD = 800 # the maximum curvature radius
CURVE_RAD = (10, 150) # the curvature radius is ranging in 10-150m
TILE_LENGTH = 10 # the length of each tile
STEP_LIMIT = 20000 # steps
TIME_STEP = 0.01 # state update time step: 0.01 s/step


class RacingTrackGenerator:
    def _create_checkpoints(self):
        n_checkpoint = np.random.randint(*N_CHECKPOINT)
        noise = np.random.uniform(0, 2 * np.pi / n_checkpoint, n_checkpoint)
        alpha = 2 * np.pi * np.arange(n_checkpoint) / n_checkpoint + noise
        rad = np.random.uniform(TRACK_RAD / 5, TRACK_RAD, n_checkpoint)

        checkpoints = np.array([rad * np.cos(alpha), rad * np.sin(alpha)])
        success = False
        control_points = []

        for _ in range(100):
            glued_cnt = 0
            control_points.clear()

            for i in range(n_checkpoint):
                pt1 = checkpoints[:, i-1]
                pt2 = checkpoints[:, i]
                next_i = 0 if i+1 == n_checkpoint else i+1
                pt3 = checkpoints[:, next_i]

                t1 = np.random.uniform(low=1/4, high=1/2)
                t2 = np.random.uniform(low=1/4, high=1/2)
                pt1_ = (1-t1) * pt2 + t1 * pt1
                pt3_ = (1-t2) * pt2 + t2 * pt3
                _, radius = get_circle(pt1_, pt2, pt3_)
                if radius < CURVE_RAD[0]:
                    if rad[i] > rad[next_i]:
                        rad[next_i] += np.random.uniform(0., 10.)
                    else:
                        rad[next_i] -= np.random.uniform(0., 10.)
                    alpha[next_i] += np.random.uniform(0., 0.05)
                    checkpoints[:, next_i] = \
                        [rad[next_i]*np.cos(alpha[next_i]), rad[next_i]*np.sin(alpha[next_i])]
                elif radius > CURVE_RAD[1]:
                    if rad[i] > rad[next_i]:
                        rad[next_i] -= np.random.uniform(0., 10.)
                    else:
                        rad[next_i] += np.random.uniform(0., 10.)
                    alpha[next_i] -= np.random.uniform(0., 0.05)
                    checkpoints[:, next_i] = \
                        [rad[next_i]*np.cos(alpha[next_i]), rad[next_i]*np.sin(alpha[next_i])]
                else:
                    glued_cnt += 1
                    control_points.append([pt1_, pt3_])

            if glued_cnt == n_checkpoint:
                success = True
                break
    
        success = success and all(alpha == sorted(alpha))
        return checkpoints, control_points, success

    def _create_map(self, center: LineString):
        self.map.reset()

        # set map boundary
        bbox = center.bounds
        boundary = [bbox[0]-50, bbox[2]+50, bbox[1]-50,  bbox[3]+50]
        shift_matrix = [1,0,0,1,-boundary[0], -boundary[2]]
        center = affine_transform(center, shift_matrix)
        boundary = [
            boundary[0]-boundary[0], boundary[1]-boundary[0],
            boundary[2] - boundary[2], boundary[3]-boundary[2]
        ]
        self.map.boundary = boundary

        # split the track into tiles
        
        distance = center.length
        n_tile = int(np.ceil(distance/TILE_LENGTH))

        if self.verbose:
            print("The track is %dm long and has %d tiles." % (int(distance), n_tile), end=" ")

        center_points = []
        left_points = []
        right_points = []

        for i in range(n_tile+1):
            center_points.append(center.interpolate(TILE_LENGTH * i))
            if i > 0:
                pt1 = center_points[i-1]
                pt2 = center_points[i]
                x_diff = pt2.x - pt1.x
                y_diff = pt2.y - pt1.y
                k = TRACK_WIDTH / 2 / Point(pt1).distance(Point(pt2))
                left_points.append([pt1.x - k*y_diff, pt1.y + k*x_diff])
                right_points.append([pt1.x + k*y_diff, pt1.y - k*x_diff])
        left_points.append(left_points[0])
        right_points.append(right_points[0])
        
        # generate map
        for i in range(n_tile):
            tile = Lane(
                id_="%04d" % i,
                left_side=LineString([left_points[i], left_points[i+1]]),
                right_side=LineString([right_points[i], right_points[i+1]]),
                subtype="road", inferred_participants=["vehicle:car", "vehicle:racing"]
            )
            self.map.add_lane(tile)

        return n_tile

