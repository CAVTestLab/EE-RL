import math

import numpy as np

from config import CONFIG


min_speed = CONFIG.reward_params.min_speed
max_speed = CONFIG.reward_params.max_speed
target_speed = CONFIG.reward_params.target_speed
max_distance = CONFIG.reward_params.max_distance
max_std_center_lane = CONFIG.reward_params.max_std_center_lane
max_angle_center_lane = CONFIG.reward_params.max_angle_center_lane
penalty_reward = CONFIG.reward_params.penalty_reward
early_stop = CONFIG.reward_params.early_stop
reward_functions = {}


def create_reward_fn(reward_fn):
    """Wrapper applying early stopping conditions and penalty rewards"""
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            speed = env.vehicle.get_speed()
            if speed < 1.0:
                env.low_speed_timer += 1
            else:
                env.low_speed_timer = 0.0

            if env.low_speed_timer >= 90 * env.fps:
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"

            if env.distance_from_center > max_distance and not env.eval:
                env.terminal_state = True
                terminal_reason = "Off-track"

        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            env.low_speed_timer = 0.0
            if reward_fn in {reward_fn5}:
                reward += penalty_reward

        if env.success_state:
            pass

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func

def reward_fn_revolve(env):
    """
    Revolve reward function (https://arxiv.org/pdf/2406.01309)
    Combines collision penalty, inactivity penalty, speed, position, and smoothness rewards.
    """
    collision_penalty = -100
    inactivity_penalty = -10
    speed_reward_weight = 2.0
    position_reward_weight = 1.0
    smoothness_reward_weight = 0.5
    speed_temp = 0.5
    position_temp = 0.1
    smoothness_temp = 0.1
    
    reward_components = {
        "collision_penalty": 0,
        "inactivity_penalty": 0,
        "speed_reward": 0,
        "position_reward": 0,
        "smoothness_reward": 0
    }
    collision = env.collision_state
    speed = env.vehicle.get_speed() / 3.6
    action_list = env.action_list
    min_pos = env.distance_from_center

    if collision:
        reward_components["collision_penalty"] = collision_penalty
    if speed < 1.5:
        reward_components["inactivity_penalty"] = inactivity_penalty
    if 4.0 <= speed <= 6.0:
        speed_score = 1 - np.abs(speed - 5) / 1.75
    else:
        speed_score = -1
    reward_components["speed_reward"] = speed_reward_weight * (1 / (1 + np.exp(
        -speed_score / speed_temp)))
    position_score = np.exp(-min_pos / position_temp)
    reward_components["position_reward"] = position_reward_weight * position_score
    steering_smoothness = -np.std(action_list)
    reward_components["smoothness_reward"] = smoothness_reward_weight * np.exp(
        steering_smoothness / smoothness_temp)
    
    total_reward = 0
    if reward_components["collision_penalty"] < 0:
        total_reward = reward_components["collision_penalty"]
    elif reward_components["inactivity_penalty"] < 0:
        total_reward = reward_components["inactivity_penalty"]
    else:
        total_reward = sum(reward_components.values())
    total_reward = np.clip(total_reward, -1, 1)
    return total_reward

reward_functions["reward_fn_revolve"] = create_reward_fn(reward_fn_revolve)

def reward_fn_revolve_auto(env):
    """
    Revolve variant with automatic feedback mechanism (inspired by Eureka).
    Rewards collision avoidance, lane centering, speed regulation, and smooth driving.
    """
    def calculate_distance(point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

    def get_distance_to_nearest_obstacle(vehicle, max_distance_view=20):
        vehicle_transform = vehicle.get_transform()
        start_location = vehicle_transform.location
        forward_vector = vehicle_transform.get_forward_vector()
        end_location = start_location + forward_vector * max_distance_view

        world = vehicle.get_world()
        hit_result = world.cast_ray(start_location, end_location)

        if hit_result:
            min_distance = 9999
            for hit_res in hit_result:
                distance = calculate_distance(start_location, hit_res.location)
                if distance < min_distance:
                    min_distance = distance
            return min_distance
        else:
            return max_distance_view

    temp_collision = 5.0
    temp_inactivity = 10.0
    temp_centering = 2.0
    temp_speed = 2.0
    temp_smoothness = 1.0
    
    reward_components = {
        'collision_penalty': 0,
        'inactivity_penalty': 0,
        'lane_centering_bonus': 0,
        'speed_regulation_bonus': 0,
        'smooth_driving_bonus': 0,
        'front_distance_bonus': 0
    }
    collision = env.collision_state
    speed = env.vehicle.get_speed() / 3.6
    action_list = env.action_list
    min_pos = env.distance_from_center

    reward_components['collision_penalty'] = np.exp(-temp_collision) if collision else 0
    inactivity_threshold = 1.5
    if speed < inactivity_threshold and not collision:
        reward_components['inactivity_penalty'] = np.exp(-temp_inactivity * (
                inactivity_threshold - speed))
    reward_components['lane_centering_bonus'] = np.exp(-temp_centering * min_pos)
    ideal_speed = (4.0 + 6.0) / 2
    speed_range = 6.0 - 4.0
    reward_components['speed_regulation_bonus'] = np.exp(-temp_speed * (abs(speed - ideal_speed) / speed_range))
    max_steering_variance = 0.1
    smoothness_factor = np.std(action_list)
    reward_components['smooth_driving_bonus'] = np.exp(-temp_smoothness * (
            smoothness_factor / max_steering_variance))
    max_distance_view = 20
    distance = get_distance_to_nearest_obstacle(env.vehicle, max_distance_view)
    reward_components['front_distance_bonus'] = np.clip(distance / max_distance_view, 0, 1)
    
    total_reward = sum(reward_components.values())
    if collision or speed < inactivity_threshold:
        total_reward = -1
    total_reward = np.clip(total_reward, -1, 1)
    return total_reward

reward_functions["reward_fn_revolve_auto"] = create_reward_fn(reward_fn_revolve_auto)

def reward_fn_chatscene(env):
    """Calculate composite reward from collision, steering, lane adherence, and speed."""
    out_lane_thres = 4
    desired_speed = 5.5

    def get_lane_dis(waypoints, x, y):
        """Calculate lateral distance from (x,y) to waypoint path."""
        eps = 1e-5
        dis_min = 99999
        waypt = waypoints[0]
        for pt in waypoints:
            pt_loc = pt.transform.location
            d = np.sqrt((x - pt_loc.x) ** 2 + (y - pt_loc.y) ** 2)
            if d < dis_min:
                dis_min = d
                waypt = pt
        vec = np.array([x - waypt.transform.location.x, y - waypt.transform.location.y])
        lv = np.linalg.norm(np.array(vec)) + eps
        w = np.array([np.cos(waypt.transform.rotation.yaw / 180 * np.pi),
                      np.sin(waypt.transform.rotation.yaw / 180 * np.pi)])
        cross = np.cross(w, vec / lv)
        dis = - lv * cross
        return dis, w

    collision = env.collision_state
    waypoints = [i[0] for i in env.route_waypoints[env.current_waypoint_index % len(env.route_waypoints): ]]
    r_collision = -1 if collision else 0
    r_steer = -env.vehicle.get_control().steer ** 2

    trans = env.vehicle.get_transform()
    ego_x = trans.location.x
    ego_y = trans.location.y
    dis, w = get_lane_dis(waypoints, ego_x, ego_y)
    r_out = -1 if abs(dis) > out_lane_thres else 0

    v = env.vehicle.get_velocity()
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)
    r_fast = -1 if lspeed_lon > desired_speed else 0
    r_lat = -abs(env.vehicle.get_control().steer) * lspeed_lon ** 2

    total_reward = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
    return total_reward

reward_functions["reward_fn_chatscene"] = create_reward_fn(reward_fn_chatscene)

def reward_fn_simple(env):
    """
    Trustworthy safety improvement for autonomous driving using reinforcement learning, 2022.
    Returns -1 on collision, 0 otherwise.
    """
    collision = env.collision_state
    if collision:
        return -1
    else:
        return 0

reward_functions["reward_fn_simple"] = create_reward_fn(reward_fn_simple)

def reward_fn5(env):
    """
    Multiply speed reward, centering factor, angle factor, and distance std factor.
    Speed reward: linear [0,1] for speeds in [0, min_speed], 1.0 for [min_speed, target_speed],
    then decreases linearly for speeds > target_speed.
    """
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    
    if speed_kmh < min_speed:
        speed_reward = speed_kmh / min_speed
    elif speed_kmh > target_speed:
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:
        speed_reward = 1.0

    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)
    
    reward = speed_reward * centering_factor * angle_factor * distance_std_factor
    return reward

reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)
