import torch
from torchvision import transforms

import numpy as np
import gym

from carla_env.wrappers import vector, get_displacement_vector
import config

torch.cuda.empty_cache()


def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    frame = preprocess(frame).unsqueeze(0)
    return frame


_TL_STATE_MAP = {
    'Red':    0.0,
    'Yellow': 0.5,
    'Green':  1.0,
    'Unknown': -1.0,
    'Off':    -1.0,
}

_TL_MAX_DISTANCE = 50.0


def create_encode_state_fn(measurements_to_include, CONFIG, vae=None):
    """
    Returns a function that encodes the current state of 
    the environment into some feature vector.
    """

    has_steer           = "steer"            in measurements_to_include
    has_throttle        = "throttle"         in measurements_to_include
    has_speed           = "speed"            in measurements_to_include
    has_waypoints       = "waypoints"        in measurements_to_include
    has_front_rgb       = "front_rgb_camera" in measurements_to_include
    has_tl_info         = "traffic_light_info" in measurements_to_include

    def create_observation_space():
        observation_space = {}

        # vehicle_measures: [steer, throttle, speed, tl_distance, tl_state]
        low, high = [], []
        if has_steer:    low.append(-1.0);  high.append(1.0)
        if has_throttle: low.append(0.0);   high.append(1.0)
        if has_speed:    low.append(0.0);   high.append(120.0)
        if has_tl_info:
            low.append(0.0);   high.append(1.0)
            low.append(-1.0);  high.append(1.0)

        if low:
            observation_space['vehicle_measures'] = gym.spaces.Box(
                low=np.array(low, dtype=np.float32),
                high=np.array(high, dtype=np.float32),
                dtype=np.float32
            )

        if has_waypoints:
            observation_space['waypoints'] = gym.spaces.Box(
                low=-50, high=50, shape=(15, 2), dtype=np.float32
            )

        if has_front_rgb:
            observation_space['front_rgb_camera'] = gym.spaces.Box(
                low=0, high=255,
                shape=(config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3),
                dtype=np.uint8
            )

        return gym.spaces.Dict(observation_space)

    def encode_state(env):
        encoded_state = {}
        vehicle_measures = []
        if has_steer:    vehicle_measures.append(env.vehicle.control.steer)
        if has_throttle: vehicle_measures.append(env.vehicle.control.throttle)
        if has_speed:    vehicle_measures.append(env.vehicle.get_speed())

        if has_tl_info:
            tl_info = getattr(env, 'current_traffic_light_info', {})
            raw_dist = tl_info.get('distance', float('inf'))
            if raw_dist == float('inf') or raw_dist > _TL_MAX_DISTANCE:
                tl_dist_norm = 1.0
            else:
                tl_dist_norm = float(raw_dist) / _TL_MAX_DISTANCE

            tl_state_str = tl_info.get('state', 'Unknown')
            tl_state_enc = _TL_STATE_MAP.get(tl_state_str, -1.0)
            vehicle_measures.append(tl_dist_norm)
            vehicle_measures.append(tl_state_enc)

        if vehicle_measures:
            encoded_state['vehicle_measures'] = vehicle_measures

        if has_waypoints:
            next_waypoints_state = env.route_waypoints[
                env.current_waypoint_index: env.current_waypoint_index + 15
            ]
            waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]

            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)

            relative_waypoints = np.zeros((15, 2))
            for i, w_location in enumerate(waypoints):
                relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
            if len(waypoints) < 15:
                start_index = len(waypoints)
                reference_vector = (
                    relative_waypoints[start_index - 1] - relative_waypoints[start_index - 2]
                    if start_index >= 2 else np.array([0.0, 1.0])
                )
                for i in range(start_index, 15):
                    relative_waypoints[i] = relative_waypoints[i - 1] + reference_vector

            encoded_state['waypoints'] = relative_waypoints

        if has_front_rgb:
            encoded_state['front_rgb_camera'] = env.front_rgb_data

        return encoded_state

    return create_observation_space(), encode_state
