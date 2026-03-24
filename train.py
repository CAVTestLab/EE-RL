import warnings
import os
from datetime import datetime
import json
import torch

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import argparse
import config

parser = argparse.ArgumentParser(description="Trains a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--total_timesteps", type=int, default=800000, help="Total timestep to train for")
parser.add_argument("--start_carla", default=False, help="If True, start a CARLA server")
parser.add_argument("--no_render", default=False, help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--log_dir", type=str, default="tensorboard", help="Directory to save logs")
parser.add_argument("--device", type=str, default="cuda:0", help="cpu, cuda:0, cuda:1, cuda:2")
parser.add_argument("--config", type=str, default="VLM-SAC", help="Config to use (default: VLM-SAC)")

parser.add_argument("--model_dir", type=str, default="", help="Directory of the model to load for resuming training")
parser.add_argument("--enable_vlm_env", action="store_true", help="Enable env-side VLM image processing")

args = vars(parser.parse_args())
CONFIG = config.set_config(args["config"])
CONFIG.algorithm_params.device = args["device"]

from stable_baselines3 import PPO, DDPG, SAC
from vlm_system.models import create_vlm_model, VLMModelConfig
from vlm_system.algorithms import VLMRewardedSAC, VLMRewardedDDPG, VLMRewardedTD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from carla_env.envs.env_utils import patch_env
patch_env()
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, CurriculumWarmupCallback, write_json, parse_wrapper_class, analyze_and_save_model_parameters

os.makedirs(args["log_dir"], exist_ok=True)

algorithm_dict = {"VLM-SAC": VLMRewardedSAC, "VLM-DDPG": VLMRewardedDDPG, "VLM-TD3": VLMRewardedTD3} 
if CONFIG.algorithm not in algorithm_dict:
    raise ValueError("Invalid algorithm name")

vlm_config_local = VLMModelConfig( # Local VLM
    model_name="0",
    api_key="0",
    base_url="http://localhost:8000/v1",
    reward_scale=1.0,
    enable_monitor=False,
    image_size=config.RGB_CAMERA_RESOLUTION,
)

vlm_config = VLMModelConfig( # Online VLM
    model_name="",
    api_key="",
    base_url="",
    reward_scale=1.0,
    enable_monitor=False,
    image_size=config.RGB_CAMERA_RESOLUTION,
)

vlm_model = create_vlm_model("qwen", config=vlm_config_local)
AlgorithmRL = algorithm_dict[CONFIG.algorithm]
observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)

env = CarlaRouteEnv(obs_res=CONFIG.obs_res, host=args["host"], port=args["port"],
                    reward_fn=reward_functions[CONFIG.reward_fn], observation_space=observation_space,
                    encode_state_fn=encode_state_fn, fps=args["fps"],
                    action_smoothing=CONFIG.action_smoothing,activate_spectator=args["no_render"], 
                    activate_render=args["no_render"],
                    activate_traffic_flow=False, activate_pedestrians = False,
                    start_carla=args["start_carla"], activate_front_rgb=True,
                    use_vlm=args["enable_vlm_env"], save_vlm_image=False,
                    )

for wrapper_class_str in CONFIG.wrappers:
    wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
    env = wrap_class(env, *wrap_params)

if args["model_dir"]:
    model_dir = args["model_dir"]
    model_files = glob.glob(os.path.join(model_dir, "*.zip"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in directory: {model_dir}")
    latest_model_file = max(model_files, key=os.path.getctime)
    print(f"Loading model from: {latest_model_file}")
    
    import re
    match = re.search(r"model_(\d+)_steps\.zip", latest_model_file)
    if match:
        trained_steps = int(match.group(1))
        print(f"Resuming training from step: {trained_steps}")
    else:
        raise ValueError(f"Could not extract step count from model file name: {latest_model_file}")
    
    if AlgorithmRL.__name__ in ("VLMRewardedSAC", "VLMRewardedDDPG", "VLMRewardedTD3"):
        model = AlgorithmRL.load(latest_model_file, 
                                env=env, 
                                config=CONFIG, 
                                device=args["device"],
                                vlm_model=vlm_model,
                                use_dual_buffer=False, # Enable dual replay buffer
                                aux_buffer_size=5000,
                                sampling_ratio=8.0,
                                )
    else:
        model = AlgorithmRL.load(latest_model_file, env=env, device=args["device"])

else:
    # Initialize the new model
    trained_steps = 0
    if AlgorithmRL.__name__ in ("VLMRewardedSAC", "VLMRewardedDDPG", "VLMRewardedTD3"):
        model = AlgorithmRL(
                env=env,
                config=CONFIG,
                vlm_model=vlm_model,
                use_dual_buffer=False, # Enable dual replay buffer
                aux_buffer_size=5000,
                sampling_ratio=11.0,
                )
    else:
        model = AlgorithmRL('MultiInputPolicy', env, verbose=1, seed=CONFIG.seed,
                            tensorboard_log=args["log_dir"], **CONFIG.algorithm_params)

    model_suffix = "{}_id{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), args['config'])
    model_name = f'{model.__class__.__name__}_{model_suffix}'
    model_dir = os.path.join(args["log_dir"], model_name)

new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))


analysis_results = analyze_and_save_model_parameters(model, model_dir)
remaining_timesteps = args["total_timesteps"] - trained_steps
if remaining_timesteps <= 0:
    raise ValueError("Total timesteps must be greater than the already trained steps.")

model.learn(
    total_timesteps=remaining_timesteps,
    callback=[
        HParamCallback(CONFIG),
        TensorboardCallback(300),
        # CurriculumWarmupCallback(
        #     warmup_steps=20000,
        #     phase_thresholds=[200000, 500000],
        #     verbose=1
        # ), # Adapt to the tasks of the new stage
        CheckpointCallback(
            save_freq=100000,
            save_path=model_dir,
            name_prefix="model"
        )
    ],
    reset_num_timesteps=False,
    progress_bar=True,
    log_interval=10
)