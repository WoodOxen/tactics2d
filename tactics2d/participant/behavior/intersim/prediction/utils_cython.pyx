# cython -a src/utils_cython.pyx && python src/setup.py build_ext --inplace
# cd src/ && cython -a utils_cython.pyx && python setup.py build_ext --inplace && cd ..
# cython: language_level=3, boundscheck=False, wraparound=False

language_level = 3
import numpy as np

cimport cython
cimport numpy as np
from libc.stdlib cimport RAND_MAX, rand
from libc.time cimport CLOCKS_PER_SEC, clock


cdef extern from "math.h":
    double sin(double x)
    double sqrt(double x)
    double cos(double x)
    double exp(double x)
    double fabs(double x)

cdef:
    float M_PI = 3.14159265358979323846
    int pixel_num_1m = 4

cdef int get_round(float a):
    if a > 0:
        return int(a + 0.5)
    else:
        return -int(fabs(a) + 0.5)

cdef np.float32_t get_dis_point(np.float32_t a, np.float32_t b):
    return sqrt(a * a + b * b)

cdef np.float32_t get_point_for_ratio(np.ndarray[np.float32_t, ndim=1] point, np.ndarray[np.float32_t, ndim=1] end,
                                      np.float32_t ratio, int c):
    return point[c] * (1.0 - ratio) + end[c] * ratio

def _normalize(np.ndarray[np.float32_t, ndim=2] polygon, np.float32_t angle, np.float32_t center_point_y):
    cdef np.float32_t cos_, sin_, min_sqr_dis, temp
    min_sqr_dis = 10000.0
    cdef int i, n
    cos_ = cos(angle)
    sin_ = sin(angle)
    n = polygon.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] new_points = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        new_points[i, 0] = polygon[i, 0] * cos_ - polygon[i, 1] * sin_
        new_points[i, 1] = polygon[i, 0] * sin_ + polygon[i, 1] * cos_
        temp = center_point_y - new_points[i, 1]
        min_sqr_dis = min(min_sqr_dis, new_points[i, 0] * new_points[i, 0] + temp * temp)
    return new_points, min_sqr_dis

def normalize(polygon, cent_x, cent_y, angle, center_point):
    polygon[:, 0] -= cent_x
    polygon[:, 1] -= cent_y
    return _normalize(polygon, angle, center_point[1])

cdef np.float32_t get_sqr_dis_point(np.float32_t a, np.float32_t b):
    return a * a + b * b

def get_rotate_lane_matrix(lane_matrix, x, y, angle):
    return _get_rotate_lane_matrix(lane_matrix, x, y, angle)

def _get_rotate_lane_matrix(np.ndarray[np.float32_t, ndim=2] lane_matrix, np.float32_t x, np.float32_t y, np.float32_t angle):
    cdef np.float32_t sin_, cos_, dx, dy
    cdef int i, n
    cos_ = cos(angle)
    sin_ = sin(angle)
    n = lane_matrix.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros((n, 20), dtype=np.float32)
    for r in range(n):
        for i in range(10):
            dx = lane_matrix[r, i * 2] - x
            dy = lane_matrix[r, i * 2 + 1] - y
            res[r, i * 2] = dx * cos_ - dy * sin_
            res[r, i * 2 + 1] = dx * sin_ + dy * cos_
    return res

cdef np.float32_t get_rand(np.float32_t l, np.float32_t r):
    cdef np.float32_t t = rand()
    return l + t / RAND_MAX * (r - l)

cdef int get_rand_int(int l, int r):
    return l + rand() % (r - l + 1)

args = None

# def _get_roads(xyzs, vector_dirs, types, road_ids, normalizer, args):
def _get_roads(decoded_example, normalizer, args):
    cdef:
        np.float32_t x = normalizer.x, y = normalizer.y, yaw = normalizer.yaw
        np.float32_t cos_ = cos(yaw), sin_ = sin(yaw)
        int i, j, k, this_id, now, lst, len_vectors = 0, max_point_num = 2500, polyline_num = 0, max_vector_num = 10000, max_lane_num = 1000
        int lane_id

        # np.ndarray[np.float32_t, ndim=2] xyz = xyzs
        # # warning: dir need normalize
        # np.ndarray[np.float32_t, ndim=2] dir = vector_dirs
        # np.ndarray[int, ndim=1] type = types.reshape(-1).astype(np.int32)
        # np.ndarray[int, ndim=1] id = road_ids.reshape(-1).astype(np.int32)

        np.ndarray[np.float32_t, ndim=2] xyz = decoded_example['roadgraph_samples/xyz']
        # warning: dir need normalize
        np.ndarray[np.float32_t, ndim=2] dir = decoded_example['roadgraph_samples/dir']
        np.ndarray[int, ndim=1] type = decoded_example['roadgraph_samples/type'].reshape(-1).astype(np.int32)
        np.ndarray[int, ndim=1] valid = decoded_example['roadgraph_samples/valid'].reshape(-1).astype(np.int32)
        np.ndarray[int, ndim=1] id = decoded_example['roadgraph_samples/id'].reshape(-1).astype(np.int32)

        np.ndarray[int, ndim=1] lane_id_2_length = np.zeros(max_lane_num, dtype=np.int32)
        np.ndarray[int, ndim=2] lane_id_2_point_ids = np.zeros([max_lane_num, max_point_num], dtype=np.int32)

        int point_size = 2, num_points = xyz.shape[0]

        np.ndarray[np.float32_t, ndim=2] vectors = np.zeros([max_vector_num, 128], dtype=np.float32)
        np.ndarray[int, ndim=2] polyline_spans = np.zeros([max_vector_num, 2], dtype=np.int32)

        int max_goals_2D_num = 100000
        np.ndarray[np.float32_t, ndim=2] goals_2D = np.zeros([max_goals_2D_num, 2], dtype=np.float32)
        int goals_2D_len = 0

        np.ndarray[np.float32_t, ndim=2] lane

        np.float32_t visible_y = 30.0, max_dis = 80.0

        # raster
        int do_raster = 0
        np.ndarray[np.int8_t, ndim=3] image
        int x_int, y_int, raster_scale = 1
        float x_float, y_float

        # early_fuse
        int do_early_fuse = 0


    # if 'raster' in args.other_params:
    if True:
        do_raster = 1
        image = args.image
        # raster_scale = args.other_params['raster_scale']
    for i in range(num_points):
        xyz[i, 0] -= x
        xyz[i, 1] -= y
        xyz[i, 0], xyz[i, 1] = xyz[i, 0] * cos_ - xyz[i, 1] * sin_, xyz[i, 0] * sin_ + xyz[i, 1] * cos_

    # cdef int use_lane_point = args.agent_type != 'pedestrian'
    cdef int use_lane_point = 1

    for i in range(num_points):
        if get_dis_point(xyz[i, 0], xyz[i, 1] - visible_y) < max_dis:
            if use_lane_point:
                if rand() % 10 == 0:
                    if goals_2D_len < max_goals_2D_num:
                        goals_2D[goals_2D_len, 0], goals_2D[goals_2D_len, 1] = xyz[i, 0], xyz[i, 1]
                        goals_2D_len += 1

            lane_id = id[i]

            if 0 > lane_id or lane_id >= max_lane_num:
                continue
            if lane_id_2_length[lane_id] >= max_point_num:
                continue


            # assert 0 <= lane_id and lane_id < max_lane_num
            # assert lane_id_2_length[lane_id] < max_point_num

            lane_id_2_point_ids[lane_id, lane_id_2_length[lane_id]] = i
            lane_id_2_length[lane_id] += 1

    cdef:
        int length, stride, start, cur, t, c
        np.float32_t t_float, scale = 1.0, t2_float

    # if 'raster' in args.other_params:
    if True:
        goals_2D_len = 0
        i = 0
        while i < 224:
            x_float = _raster_int_to_float(i, 0, raster_scale)
            j = 0
            while j < 224:
                y_float = _raster_int_to_float(j, 1, raster_scale)
                goals_2D[goals_2D_len, 0], goals_2D[goals_2D_len, 1] = x_float, y_float
                goals_2D_len += 1
                j += raster_scale

            i += raster_scale

    if True:
        t = 500
        while goals_2D_len < t:
            t_float = get_rand(-M_PI, M_PI)
            t2_float = get_rand(0.0, max_dis)
            goals_2D[goals_2D_len, 0], goals_2D[goals_2D_len, 1] = t2_float * cos(t_float), t2_float * sin(t_float) + visible_y
            goals_2D_len += 1

    lanes = []

    stride = 5
    # if 'stride_10_2' in args.other_params:
    if True:
        stride = 10
        scale = 0.03
    for i in range(max_lane_num):
        length = lane_id_2_length[i]
        start = len_vectors
        if length > 0:

            # if length < 10:
            #     stride = 1
            # elif length < 100:
            #     stride = int(length / 25)
            # else:
            #     stride = min(int(length / 50), 10)

            if do_raster:
                for j in range(length):
                    now = lane_id_2_point_ids[i, j]
                    x_int = _raster_float_to_int(xyz[now, 0], 0, raster_scale)
                    y_int = _raster_float_to_int(xyz[now, 1], 1, raster_scale)
                    assert 0 <= type[now] < 20
                    if _in_image(x_int, y_int):
                        image[x_int, y_int, 40 + type[now]] = 1

            for c in range((length + stride - 1) // stride):
                j = c * stride

                now = lane_id_2_point_ids[i, j]

                cur = 0
                for k in range(stride + 2):
                    t = lane_id_2_point_ids[i, min(j + k, length - 1)]
                    vectors[len_vectors, cur + 2 * k + 0] = xyz[t, 0] * scale
                    vectors[len_vectors, cur + 2 * k + 1] = xyz[t, 1] * scale

                cur = 30
                if type[now] != -1:
                    assert type[now] < 20
                    vectors[len_vectors, cur + type[now]] = 1.0

                cur = 40
                vectors[len_vectors, cur + 0] = j
                t_float = j
                vectors[len_vectors, cur + 1] = t_float / length

                len_vectors += 1
                assert len_vectors < max_vector_num

            lane = np.zeros([length, 2], dtype=np.float32)
            for j in range(length):
                t = lane_id_2_point_ids[i, j]
                lane[j, 0], lane[j, 1] = xyz[t, 0], xyz[t, 1]
            lanes.append(lane)

            polyline_spans[polyline_num, 0], polyline_spans[polyline_num, 1] = start, len_vectors
            polyline_num += 1

    if polyline_num == 0:
        start = len_vectors
        len_vectors += 1
        polyline_spans[polyline_num, 0], polyline_spans[polyline_num, 1] = start, len_vectors
        polyline_num += 1

        lane = np.zeros([1, 2], dtype=np.float32)
        lanes.append(lane)

    return vectors, polyline_spans, len_vectors, polyline_num, goals_2D, goals_2D_len, lanes

def get_roads(road_raw, normalizer, args):
    vectors, polyline_spans, len_vectors, polyline_num, goals_2D, goals_2D_len, lanes = \
        _get_roads(road_raw, normalizer, args)
    return vectors[:len_vectors].copy(), polyline_spans[:polyline_num].copy(), goals_2D[:goals_2D_len].copy(), lanes

cdef int _raster_float_to_int(float a, int is_y, float scale):
    if not is_y:
        return int(a * scale + 10000 + 0.5) - 10000 + 112
    else:
        return int(a * scale + 10000 + 0.5) - 10000 + 56

def raster_float_to_int(a, is_y, scale):
    return _raster_float_to_int(a, is_y, scale)

cdef int _in_image(int x, int y):
    return 0 <= x < 224 and 0 <= y < 224

def in_image(x, y):
    return _in_image(x, y)

cdef float _raster_int_to_float(int a, int is_y, float scale):
    if not is_y:
        return (a - 112) / scale
    else:
        return (a - 56) / scale

def raster_int_to_float(a, is_y, scale):
    return _raster_int_to_float(a, is_y, scale)

def _get_agents(gt_trajectory_, gt_future_is_valid_, tracks_type_, visualize, args, gt_influencer_traj_, prediction_scores_):
    cdef:
        int i, j, k, this_id, now, lst, len_vectors = 0, max_point_num = 2500, polyline_num = 0, agent_num = gt_trajectory_.shape[0]
        int history_frame_num = 11, max_vector_num = 10000

        np.ndarray[np.float32_t, ndim=3] gt_trajectory = gt_trajectory_
        np.ndarray[int, ndim=2] gt_future_is_valid = gt_future_is_valid_.astype(np.int32)
        np.ndarray[int, ndim=1] tracks_type = tracks_type_.astype(np.int32)

        np.ndarray[np.float32_t, ndim=2] vectors = np.zeros([max_vector_num, 128], dtype=np.float32)
        np.ndarray[int, ndim=2] polyline_spans = np.zeros([max_vector_num, 2], dtype=np.int32)

        int start, cur, nxt
        np.float32_t scale = 1.0

        # filter_stationary
        int valid_num
        np.float32_t vx, vy

        # interactive-single_traj
        np.ndarray[np.float32_t, ndim=2] oppo_agent_labels
        int oppo_agent_idx = -1
        int t_int

        # raster
        int do_raster = 0
        np.ndarray[np.int8_t, ndim=3] image
        int x, y, raster_scale = 1

        int do_early_fuse = 0

    trajs = []

    # if 'raster' in args.other_params:
    if True:
        # assert 'interactive' not in args.other_params
        do_raster = 1
        image = args.image
        # raster_scale = args.other_params['raster_scale']

    for i in range(agent_num):
        start = len_vectors

        for j in range(history_frame_num):
            if not gt_future_is_valid[i, j]:
                for k in range(7):
                    gt_trajectory[i, j, k] = 0.0

        if visualize:
            yaw = None
            trajectory = np.zeros([gt_future_is_valid[i, :history_frame_num].sum(), 2])
            cur = 0
            for j in range(history_frame_num):
                if gt_future_is_valid[i, j]:
                    trajectory[cur, 0], trajectory[cur, 1] = gt_trajectory[i, j, 0], gt_trajectory[i, j, 1]
                    yaw = gt_trajectory[i, j, 4]
                    cur += 1
            trajs.append(trajectory)

        if do_raster:
            for j in range(history_frame_num):
                if gt_future_is_valid[i, j]:
                    x = _raster_float_to_int(gt_trajectory[i, j, 0], 0, raster_scale)
                    y = _raster_float_to_int(gt_trajectory[i, j, 1], 1, raster_scale)
                    if _in_image(x, y):
                        if i == 0:
                            image[x, y, j] = 1
                        else:
                            image[x, y, j + 20] = 1

        # if i == 0:
        #     if gt_influencer_traj_ is not None:
        #         # add a new influencer info
        #         # has one pred(on eval)/gt(on training)
        #         assert len(gt_influencer_traj_.shape) == 2, gt_influencer_traj_.shape
        #         time_frames, _ = gt_influencer_traj_[history_frame_num:, :].shape
        #         for j in range(time_frames):
        #             current_time_frame = j + history_frame_num
        #             # if args.do_eval or gt_future_is_valid[i, current_time_frame]:
        #             x = _raster_float_to_int(gt_influencer_traj_[current_time_frame, 0], 0, raster_scale)
        #             y = _raster_float_to_int(gt_influencer_traj_[current_time_frame, 1], 1, raster_scale)
        #             if _in_image(x, y):
        #                 image[x, y, j + 60] = 1

        # history_frame_num = 11
        for j in range(history_frame_num - 1):
            cur = 0
            for k in range(2):
                vectors[len_vectors, cur + k] = gt_trajectory[i, j, k] * scale
            for k in range(2, 7):
                vectors[len_vectors, cur + k] = gt_trajectory[i, j, k]

            cur = 20
            for k in range(2):
                vectors[len_vectors, cur + k] = gt_trajectory[i, j + 1, k] * scale
            for k in range(2, 7):
                vectors[len_vectors, cur + k] = gt_trajectory[i, j + 1, k]

            cur = 30
            vectors[len_vectors, cur + 0] = j
            vectors[len_vectors, cur + 1 + j] = 1

            cur = 50
            vectors[len_vectors, cur + 0] = tracks_type[i]
            vectors[len_vectors, cur + 1 + tracks_type[i]] = 1

            len_vectors += 1
            assert len_vectors < max_vector_num

        # add the intention
        # if i == 0:
        #     # for j in range(gt_influencer_traj_.shape[0] - history_frame_num - 1):
        #     for j in range(gt_influencer_traj_.shape[0] - 1):
        #         cur = 0
        #         for k in range(2):
        #             vectors[len_vectors, cur + k] = gt_influencer_traj_[j, k] * scale
        #
        #         cur = 5
        #         for k in range(2):
        #             vectors[len_vectors, cur + k] = gt_influencer_traj_[j + 1, k] * scale
        #
        #         len_vectors += 1
        #         assert len_vectors < max_vector_num

        polyline_spans[polyline_num, 0], polyline_spans[polyline_num, 1] = start, len_vectors # len_vectors=10*agent_num
        polyline_num += 1

    assert len_vectors <= max_vector_num

    return vectors, polyline_spans, len_vectors, polyline_num, trajs

def get_agents(gt_trajectory, gt_future_is_valid, tracks_type, visualize, args, gt_influencer_traj=None, prediction_scores=None):
    vectors, polyline_spans, len_vectors, polyline_num, trajs = _get_agents(gt_trajectory, gt_future_is_valid, tracks_type, visualize, args, gt_influencer_traj, prediction_scores)
    return vectors[:len_vectors].copy(), polyline_spans[:polyline_num].copy(), trajs

def _get_normalized(np.ndarray[np.float32_t, ndim=3] polygons, np.float32_t x, np.float32_t y, np.float32_t angle):
    cdef:
        np.float32_t cos_, sin_, min_sqr_dis, temp
        int i, n, polygon_idx
    cos_ = cos(angle)
    sin_ = sin(angle)
    n = polygons.shape[1]
    # cdef np.ndarray[np.float32_t, ndim=3] new_polygons = np.zeros((polygons.shape[0], n, 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] new_polygons = np.zeros_like(polygons, dtype=np.float32)
    # cdef np.ndarray[np.float32_t, ndim=3] new_polygons = np.zeros((polygons.shape[0], n, polygons.shape[2]), dtype=np.float32)
    for polygon_idx in range(polygons.shape[0]):
        for i in range(n):
            polygons[polygon_idx, i, 0] -= x
            polygons[polygon_idx, i, 1] -= y
            new_polygons[polygon_idx, i, 0] = polygons[polygon_idx, i, 0] * cos_ - polygons[polygon_idx, i, 1] * sin_
            new_polygons[polygon_idx, i, 1] = polygons[polygon_idx, i, 0] * sin_ + polygons[polygon_idx, i, 1] * cos_
            # normalize yaw angles
            # new_polygons[polygon_idx, i, 4] = (polygons[polygon_idx, i, 4] - angle) % M_PI * 2
            # while not -M_PI <= new_polygons[polygon_idx, i, 4] < M_PI:
            #     new_polygons[polygon_idx, i, 4] -= M_PI * 2

    return new_polygons

def get_normalized(trajectorys, normalizer, reverse=False):
    if trajectorys.dtype is not np.float32:
        trajectorys = trajectorys.astype(np.float32)

    if reverse:
        return _get_normalized(trajectorys, normalizer.origin[0], normalizer.origin[1], -normalizer.yaw)
    return _get_normalized(trajectorys, normalizer.x, normalizer.y, normalizer.yaw)

def get_normalized_points(points: np.ndarray, normalizer, reverse=False):
    if points.dtype is not np.float32:
        points = points.astype(np.float32)

    trajectorys = points[np.newaxis, :]
    if reverse:
        return _get_normalized(trajectorys, normalizer.origin[0], normalizer.origin[1], -normalizer.yaw)[0]
    return _get_normalized(trajectorys, normalizer.x, normalizer.y, normalizer.yaw)[0]

cdef:
    np.float32_t speed_lower_bound = 1.4, speed_upper_bound = 11.0, speed_scale_lower = 0.5, speed_scale_upper = 1.0

    np.float32_t lateral_miss_threshold_3 = 1.0, longitudinal_miss_threshold_3 = 2.0
    np.float32_t lateral_miss_threshold_5 = 1.8, longitudinal_miss_threshold_5 = 3.6
    np.float32_t lateral_miss_threshold_8 = 3.0, longitudinal_miss_threshold_8 = 6.0

cdef np.float32_t _speed_scale_factor(np.float32_t speed):
    cdef:
        np.float32_t fraction

    if speed < speed_lower_bound:
        return speed_scale_lower
    elif speed > speed_upper_bound:
        return speed_scale_upper
    else:
        fraction = (speed - speed_lower_bound) / (speed_upper_bound - speed_lower_bound)
        return speed_scale_lower + (speed_scale_upper - speed_scale_lower) * fraction

def speed_scale_factor(speed):
    return _speed_scale_factor(speed)

cdef int _is_true_positive(np.ndarray[np.float32_t, ndim=2] disps, np.float32_t speed, int time, np.float32_t delta_heading):
    cdef:
        np.float32_t scale, lateral_miss_threshold, longitudinal_miss_threshold, cos_, sin_, longitudinal, lateral
        int i, n = disps.shape[0], miss

    scale = _speed_scale_factor(speed)

    if time == 30:
        lateral_miss_threshold = lateral_miss_threshold_3
        longitudinal_miss_threshold = longitudinal_miss_threshold_3
    elif time == 50:
        lateral_miss_threshold = lateral_miss_threshold_5
        longitudinal_miss_threshold = longitudinal_miss_threshold_5
    elif time == 80:
        lateral_miss_threshold = lateral_miss_threshold_8
        longitudinal_miss_threshold = longitudinal_miss_threshold_8
    else:
        assert False

    lateral_miss_threshold *= scale
    longitudinal_miss_threshold *= scale

    cos_ = cos(delta_heading)
    sin_ = sin(delta_heading)

    miss = 1
    for i in range(n):
        longitudinal, lateral = disps[i, 0] * cos_ - disps[i, 1] * sin_, disps[i, 0] * sin_ + disps[i, 1] * cos_
        if fabs(lateral) <= lateral_miss_threshold and fabs(longitudinal) <= longitudinal_miss_threshold:
            miss = 0
            break

    return miss == 0

def is_true_positive(points, gt_point, speed, time, headings, normalizer):
    heading = headings[time - 1]
    delta_heading = -normalizer.yaw - heading
    disps = (points - gt_point[np.newaxis, :]).astype(np.float32)
    return _is_true_positive(disps, speed, time, delta_heading)

def _classify_track(np.ndarray[int, ndim=1] gt_future_is_valid, np.ndarray[np.float32_t, ndim=2] gt_trajectory):
    # warning: gt_trajectory must not norm

    cdef:
        int end_state = -1, c, i, start_state
        int history_frame_num = 11
        float kMaxSpeedForStationary = 2.0
        float kMaxDisplacementForStationary = 5.0
        float kMaxLateralDisplacementForStraight = 5.0
        float kMinLongitudinalDisplacementForUTurn = -5.0
        float kMaxAbsHeadingDiffForStraight = 3.14159265358979323846 / 6.0
        float x_delta, y_delta, final_displacement, heading_diff, dx, dy, cos_, sin_, x, y
        float start_speed, end_speed, max_speed

    start_state = history_frame_num - 1

    i = history_frame_num + 80
    while True:
        i -= 1
        if i <= start_state:
            break

        if gt_future_is_valid[i]:
            end_state = i
            break

    if end_state == -1 or not gt_future_is_valid[start_state]:
        return 'STATIONARY'
        # return None

    x_delta = gt_trajectory[end_state, 0] - gt_trajectory[start_state, 0]
    y_delta = gt_trajectory[end_state, 1] - gt_trajectory[start_state, 1]
    final_displacement = get_dis_point(x_delta, y_delta)
    heading_diff = gt_trajectory[end_state, 4] - gt_trajectory[start_state, 4]

    cos_ = cos(-gt_trajectory[start_state, 4])
    sin_ = sin(-gt_trajectory[start_state, 4])
    x = x_delta
    y = y_delta
    dx, dy = x * cos_ - y * sin_, x * sin_ + y * cos_
    start_speed = get_dis_point(gt_trajectory[start_state, 5], gt_trajectory[start_state, 6])
    end_speed = get_dis_point(gt_trajectory[end_state, 5], gt_trajectory[end_state, 6])
    max_speed = max(start_speed, end_speed)

    # if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
    #     return 'STATIONARY'
    if final_displacement < kMaxDisplacementForStationary:
        return 'STATIONARY'
    if fabs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if fabs(dy) < kMaxLateralDisplacementForStraight:
            return 'STRAIGHT'
        return 'STRAIGHT_RIGHT' if dy < 0 else 'STRAIGHT_LEFT'
    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return 'RIGHT_TURN'
        # return 'RIGHT_U_TURN' if dx < kMinLongitudinalDisplacementForUTurn else 'RIGHT_TURN'
    if dx < kMinLongitudinalDisplacementForUTurn:
        return 'LEFT_U_TURN'
    return 'LEFT_TURN'

def _tnt_square_filter(np.ndarray[np.float32_t, ndim=2] goals_2D,
                       np.ndarray[np.float32_t, ndim=1] scores,
                       args):
    cdef:
        int n = goals_2D.shape[0]
        int i, j, width = 100, x, y, m = 0
        float cent_x = 0.0, cent_y = 30.0, eps = 1e-6
        np.ndarray[float, ndim = 2] heatmap = np.ones((width, width), dtype=np.float32)

    for i in range(n):
        x = get_round(goals_2D[i, 0] - cent_x) + width // 2
        y = get_round(goals_2D[i, 1] - cent_y) + width // 2
        if 0 <= x < width and 0 <= y < width:
            heatmap[x, y] = scores[i]

    for i in range(0, width, 1):
        for j in range(0, width, 1):
            if heatmap[i, j] < 1.0 - eps:
                goals_2D[m, 0] = i - width // 2 + cent_x
                goals_2D[m, 1] = j - width // 2 + cent_y
                scores[m] = heatmap[i, j]
                m += 1

    m = max(m, 10)

    assert m > 0, (n, goals_2D, scores)
    goals_2D = goals_2D[:m]
    scores = scores[:m]

    ids = np.argsort(-scores)[:args.joint_target_each]
    scores = scores[ids].copy()
    goals_2D = goals_2D[ids].copy()

    return goals_2D, scores

def tnt_square_filter(goals_2D, scores, args):
    return _tnt_square_filter(goals_2D, scores, args)

@cython.boundscheck(True)
def _tnt_square_combine(np.ndarray[np.float32_t, ndim=2] goals_2D,
                        np.ndarray[np.float32_t, ndim=1] scores,
                        np.ndarray[np.float32_t, ndim=2] goals_2D_oppo,
                        np.ndarray[np.float32_t, ndim=1] scores_oppo,
                        int is_oppo):
    cdef:
        int i, j, k = 0, n = goals_2D.shape[0], m = goals_2D_oppo.shape[0]
        np.ndarray[np.float32_t, ndim=2] goals_4D = np.zeros((n * m, 4), dtype=np.float32)
        np.ndarray[np.float32_t, ndim = 1] product_scores = np.zeros((n * m), dtype=np.float32)

    if is_oppo:
        for j in range(m):
            for i in range(n):
                goals_4D[k, 0], goals_4D[k, 1] = goals_2D[i, 0], goals_2D[i, 1]
                goals_4D[k, 2], goals_4D[k, 3] = goals_2D_oppo[j, 0], goals_2D_oppo[j, 1]
                product_scores[k] = scores[i] * scores_oppo[j]
                k += 1
    else:
        for i in range(n):
            for j in range(m):
                goals_4D[k, 0], goals_4D[k, 1] = goals_2D[i, 0], goals_2D[i, 1]
                goals_4D[k, 2], goals_4D[k, 3] = goals_2D_oppo[j, 0], goals_2D_oppo[j, 1]
                product_scores[k] = scores[i] * scores_oppo[j]
                k += 1

    return goals_4D, product_scores

def tnt_square_combine(goals_2D, scores, goals_2D_oppo, scores_oppo, is_oppo=False, max_each=None):
    if max_each is not None:
        def get(goals_2D, scores):
            argsort = np.argsort(-scores)
            goals_2D = goals_2D[argsort][:max_each]
            scores = scores[argsort][:max_each]
            return goals_2D, scores
        goals_2D, scores = get(goals_2D, scores)
        goals_2D_oppo, scores_oppo = get(goals_2D_oppo, scores_oppo)

    assert scores[0] < 0 and scores_oppo[0] < 0
    scores = np.exp(scores)
    scores_oppo = np.exp(scores_oppo)
    return _tnt_square_combine(goals_2D, scores, goals_2D_oppo, scores_oppo, is_oppo)

def classify_track(gt_future_is_valid, gt_trajectory_not_norm):
    return _classify_track(gt_future_is_valid.astype(np.int32), gt_trajectory_not_norm)

cdef int may_collide(np.ndarray[np.float32_t, ndim=1] a, np.ndarray[np.float32_t, ndim=1] b):
    if get_dis_point(a[0] - b[0], a[1] - b[1]) < 3.0:
        return True
    else:
        return False

cdef int may_collide_traj(np.ndarray[np.float32_t, ndim=2] a, np.ndarray[np.float32_t, ndim=2] b):
    cdef int i
    assert len(a) == len(b) == 80
    for i in range(len(a) - 5, len(a)):
        if get_dis_point(a[i, 0] - b[i, 0], a[i, 1] - b[i, 1]) < 1.0:
            return True

    return False

@cython.boundscheck(True)
def _colli_det(np.ndarray[np.float32_t, ndim=2] goals_4D, np.ndarray[np.float32_t, ndim=1] joint_scores, normalizer, normalizer_o,
               np.ndarray[np.float32_t, ndim=3] trajs, np.ndarray[np.float32_t, ndim=3] trajs_o):
    cdef:
        int i, n, x, y, final_idx = trajs.shape[1] - 1

    goals_4D = goals_4D.copy()
    joint_scores = joint_scores.copy()

    goals_4D[:, 0:2] = get_normalized_points(goals_4D[:, 0:2], normalizer, reverse=True)
    goals_4D[:, 2:4] = get_normalized_points(goals_4D[:, 2:4], normalizer_o, reverse=True)

    assert np.all(joint_scores >= 0)
    n = len(goals_4D)

    for i in range(n):
        x = i // len(trajs_o)
        y = i % len(trajs_o)
        assert x < len(trajs)
        # print(goals_4D[i, 0])
        # print(x, final_idx)
        # print(trajs[x, final_idx, 0], goals_4D[i, 0])
        # print(trajs_o[y, final_idx, 0], goals_4D[i, 2])
        assert fabs(goals_4D[i, 0] - trajs[x, final_idx, 0]) < 1e-2
        assert fabs(goals_4D[i, 2] - trajs_o[y, final_idx, 0]) < 1e-2

        # if get_rand_int(0, 500 * 50) == 0:
        #     print(may_collide_traj(trajs[x], trajs_o[y]))
        #     print(trajs[x, ::5], trajs_o[y, ::5])

        if may_collide_traj(trajs[x], trajs_o[y]):
            joint_scores[i] = 0
        # if may_collide(goals_4D[i, 0:2], goals_4D[i, 2:4]):
        #     joint_scores[i] = 0

    return joint_scores

def colli_det(goals_4D, joint_scores, normalizer, normalizer_o, trajs, trajs_o):
    return _colli_det(goals_4D, joint_scores, normalizer, normalizer_o, trajs, trajs_o)

# def _get_early_fuse_patches(np.ndarray[np.float32_t, ndim=2] early_fuse_points, np.ndarray[np.int8_t, ndim=3] image, args):
#     cdef:
#         np.ndarray[np.int8_t, ndim=4] early_fuse_patches = \
#             np.zeros((early_fuse_points.shape[0], 9, 9, image.shape[2]), dtype=image.dtype)
#         int i, x, y, r = 4
#
#     for i in range(len(early_fuse_points)):
#         x = get_round(early_fuse_points[i, 0])
#         y = get_round(early_fuse_points[i, 1])
#         early_fuse_patches[i] = image[x - r:x + r + 1, y - r:y + r + 1, :]
#     pass

def get_early_fuse_patches(early_fuse_points, image, args):
    # print('in get_early_fuse_patches')
    r = 4
    early_fuse_patches = np.zeros((early_fuse_points.shape[0], 9, 9, image.shape[2]), dtype=image.dtype)
    raster_scale = 1

    for i in range(len(early_fuse_points)):
        x = raster_float_to_int(early_fuse_points[i, 0], 0, raster_scale)
        y = raster_float_to_int(early_fuse_points[i, 1], 1, raster_scale)
        if 0 <= x - r and x + r < 224 and 0 <= y - r and y + r < 224:
            early_fuse_patches[i] = image[x - r:x + r + 1, y - r:y + r + 1, :]

    # print(' out get_early_fuse_patches')

    return early_fuse_patches

    # return _get_early_fuse_patches(early_fuse_points, image, args)
