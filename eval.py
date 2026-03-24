import os
import argparse
import pandas as pd
import numpy as np
import config

parser = argparse.ArgumentParser(description="Evaluate a CARLA agent")

# Model/config selection
model_group = parser.add_argument_group("Model Selection")
model_group.add_argument(
    "--model",
    type=str,
    default="",
    help="Path to a model checkpoint (.zip)",
)
model_group.add_argument(
    "--config",
    type=str,
    default="VLM-SAC",
    help="Training config name used for loading and evaluation",
)

# CARLA runtime settings
carla_group = parser.add_argument_group("CARLA Runtime")
carla_group.add_argument("--host", default="localhost", type=str, help="CARLA server host")
carla_group.add_argument("--port", default=2000, type=int, help="CARLA server TCP port")
carla_group.add_argument("--start_carla", default=True, help="Start a CARLA server process")
carla_group.add_argument("--no_render", default=True, help="Enable spectator/render window")
carla_group.add_argument("--fps", type=int, default=15, help="Simulation FPS")
carla_group.add_argument("--town", choices=['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06'], default="Town02",
                         help="Town map")
carla_group.add_argument("--density", choices=['empty', 'regular', 'dense'], default="regular",
                         help="Background traffic density")

# Evaluation settings
eval_group = parser.add_argument_group("Evaluation")
eval_group.add_argument("--no_record_video", action="store_false", help="Record evaluation video")
eval_group.add_argument("--seed", type=int, default=101, help="Random seed")
eval_group.add_argument("--device", type=str, default="cuda:0", help="Inference device")

args = vars(parser.parse_args())

config_name = args["config"]
CONFIG = config.set_config(config_name)
CONFIG.seed = args["seed"]
CONFIG.algorithm_params.device = args["device"]

from stable_baselines3 import PPO, DDPG, SAC
from vlm_system.algorithms import VLMRewardedSAC, VLMRewardedTD3, VLMRewardedDDPG

from utils import VideoRecorder, parse_wrapper_class
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions

from carla_env.wrappers import vector, get_displacement_vector
from carla_env.envs.env_utils import patch_env
patch_env()
from carla_env.envs.carla_route_env import CarlaRouteEnv
from eval_plots import plot_eval, summary_eval


def resolve_model_checkpoint(cli_args):
    model_path = cli_args["model"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not model_path.endswith(".zip"):
        raise ValueError(f"Model checkpoint must be a .zip file: {model_path}")
    return model_path


def build_eval_env(cfg, cli_args, activate_traffic_flow, tf_num):
    observation_space, encode_state_fn = create_encode_state_fn(cfg.state, cfg)
    env = CarlaRouteEnv(
        obs_res=cfg.obs_res,
        host=cli_args["host"],
        port=cli_args["port"],
        reward_fn=reward_functions[cfg.reward_fn],
        observation_space=observation_space,
        encode_state_fn=encode_state_fn,
        fps=cli_args["fps"],
        action_smoothing=cfg.action_smoothing,
        activate_spectator=cli_args["no_render"],
        activate_render=cli_args["no_render"],
        activate_traffic_flow=activate_traffic_flow,
        tf_num=tf_num,
        activate_pedestrians=True,
        start_carla=cli_args["start_carla"],
        activate_front_rgb=True,
        use_vlm=True,
        save_vlm_image=False,
        anticipation_distance=30,
        town=cli_args["town"],
    )

    for wrapper_class_str in cfg.wrappers:
        wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
        env = wrap_class(env, *wrap_params)

    return env


def convert_state(state):
    c_state = dict()
    c_state['seg_camera'] = np.transpose(state['seg_camera'], (2, 0, 1))
    c_state['seg_camera'] = np.array([c_state['seg_camera']])
    c_state['waypoints'] = np.array([state['waypoints']])
    c_state['vehicle_measures'] = np.array([state['vehicle_measures']])
    return c_state


def run_eval(env, model, model_path=None, record_video=False, eval_suffix=''):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), 'eval{}'.format(eval_suffix))
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    model_id = f"{model_path.split('/')[-2]}-{model_name.split('_')[-2]}"
    state = env.reset()

    columns = ["model_id", "episode", "step", "throttle", "steer", "vehicle_location_x", "vehicle_location_y",
               "reward", "distance", "speed", "center_dev", "angle_next_waypoint", "waypoint_x", "waypoint_y",
               "route_x", "route_y", "routes_completed", "collision_speed", "collision_interval", "CPS", "CPM"
               ]
    df = pd.DataFrame(columns=columns)

    # Initialize video recording
    if record_video:
        rendered_frame = env.render(mode="rgb_array")
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape,
                                                              int(env.fps)))
        video_recorder = VideoRecorder(video_path,
                                       frame_size=rendered_frame.shape,
                                       fps=env.fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = 0
    print("Episode ", episode_idx)
    saved_route = False
    while episode_idx < 10:
        env.extra_info.append("Evaluation")
        action, _states = model.predict(state, deterministic=True)
        next_state, reward, dones, info = env.step(action)

        state = next_state
        if env.step_count >= 150 and env.current_waypoint_index == 0:
            dones = True

        if not saved_route:
            initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            initial_vehicle_location = vector(env.vehicle.get_location())
            for way in env.route_waypoints:
                route_relative = get_displacement_vector(initial_vehicle_location,
                                                         vector(way[0].transform.location),
                                                         initial_heading)
                new_row = pd.DataFrame([['route', env.episode_idx, route_relative[0], route_relative[1]]],
                                       columns=["model_id", "episode", "route_x", "route_y"])
                df = pd.concat([df, new_row], ignore_index=True)
            saved_route = True

        vehicle_relative = get_displacement_vector(initial_vehicle_location, vector(env.vehicle.get_location()),
                                                   initial_heading)
        waypoint_relative = get_displacement_vector(initial_vehicle_location,
                                                    vector(env.current_waypoint.transform.location), initial_heading)

        if env.collision_state:
            collision_speed, collision_interval, cps, cpm = env.collision_speed, env.collision_interval, env.cps, env.cpm
        else:
            collision_speed, collision_interval, cps, cpm = 0, None, 0, 0
        new_row = pd.DataFrame(
            [[model_id, env.episode_idx, env.step_count, env.vehicle.control.throttle, env.vehicle.control.steer,
              vehicle_relative[0], vehicle_relative[1], reward,
              env.distance_traveled,
              env.vehicle.get_speed(), env.distance_from_center,
              np.rad2deg(env.vehicle.get_angle(env.current_waypoint)),
              waypoint_relative[0], waypoint_relative[1], None, None,
              env.routes_completed, collision_speed, collision_interval, cps, cpm
              ]], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

        if record_video:
            rendered_frame = env.render(mode="rgb_array")
            video_recorder.add_frame(rendered_frame)
        if dones:
            state = env.reset()
            episode_idx += 1
            saved_route = False
            print("Episode ", episode_idx)

    if record_video:
        video_recorder.release()

    df.to_csv(csv_path, index=False)
    plot_eval([csv_path])
    summary_eval(csv_path)


if __name__ == "__main__":
    algorithm_dict = {
        "VLM-SAC": VLMRewardedSAC,
        "VLM-TD3": VLMRewardedTD3,
        "VLM-DDPG": VLMRewardedDDPG,
    }

    if CONFIG.algorithm not in algorithm_dict:
        raise ValueError(f"Invalid algorithm name in config '{config_name}': {CONFIG.algorithm}")

    AlgorithmRL = algorithm_dict[CONFIG.algorithm]
    model_ckpt = resolve_model_checkpoint(args)
    print(f"Evaluating checkpoint: {model_ckpt}")

    eval_suffix = ''
    if args['density'] == 'empty':
        activate_traffic_flow = False
        tf_num = 0
        eval_suffix += 'empty'
    else:
        activate_traffic_flow = True
        if args['density'] == 'regular':
            tf_num = 20
        else:
            tf_num = 40
            eval_suffix += 'dense'
    if args['town'] != 'Town02':
        eval_suffix += args['town']

    print(f"Using config: {config_name}, algorithm: {CONFIG.algorithm}")

    env = build_eval_env(CONFIG, args, activate_traffic_flow, tf_num)

    try:
        if CONFIG.algorithm in {"VLM-SAC", "VLM-TD3", "VLM-DDPG"}:
            model = AlgorithmRL.load(model_ckpt, env=env, config=CONFIG, device=args["device"])
            model.inference_only = True
        else:
            model = AlgorithmRL.load(model_ckpt, env=env, device=args["device"])

        print("Model loaded successfully...")
        run_eval(env, model, model_ckpt, record_video=args["no_record_video"], eval_suffix=eval_suffix)
    finally:
        env.close()
