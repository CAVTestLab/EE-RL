import os
import cv2
import math
import json

import gym
import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


def write_json(data, path):
    config_dict = {}
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            if isinstance(v, str) and v.isnumeric():
                config_dict[k] = int(v)
            elif isinstance(v, dict):
                config_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    config_dict[k][k_inner] = v_inner.__str__()
                config_dict[k] = str(config_dict[k])
            else:
                config_dict[k] = v.__str__()
        json.dump(config_dict, f, indent=4)


class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, int(fps), (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def add_frame_with_reward(self, frame, reward):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        reward_text = f"Reward: {reward:.2f}"

        (text_width, text_height), _ = cv2.getTextSize(
            reward_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )
        position = (frame.shape[1] - text_width - 10,
                    frame.shape[0] - 10)
        cv2.putText(
            frame, reward_text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
        )
        self.video_writer.write(frame)

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


class HParamCallback(BaseCallback):
    def __init__(self, config):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()
        self.config = config

    def _on_training_start(self) -> None:
        hparam_dict = {}
        for k, v in self.config.items():
            if isinstance(v, str) and v.isnumeric():
                hparam_dict[k] = int(v)
            elif isinstance(v, dict):
                hparam_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    hparam_dict[k][k_inner] = v_inner.__str__()
                hparam_dict[k] = str(hparam_dict[k])
            else:
                hparam_dict[k] = v.__str__()
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class TensorboardCallback(BaseCallback):
    """
    Log training and episode metrics to TensorBoard.
    
    Args:
        log_interval: Log interval in environment steps. Set 0 to disable interval logs.
        verbose: Callback verbosity.
    """

    def __init__(self, log_interval=10, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # Track cumulative reward for current episode.
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        
        # Periodic logging.
        if self.log_interval > 0 and self.num_timesteps % self.log_interval == 0:
            if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer.pos > 0:
                recent_rewards = self.model.replay_buffer.rewards[max(0, self.model.replay_buffer.pos-500):self.model.replay_buffer.pos]
                if len(recent_rewards) > 0:
                    self.logger.record("replay_buffer/mean_recent_rewards", np.mean(recent_rewards), exclude="stdout")
                    self.logger.record("replay_buffer/sum_recent_rewards", np.sum(recent_rewards), exclude="stdout")
            
            if len(self.episode_rewards) > 0:
                recent_ep_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                recent_ep_lengths = self.episode_lengths[-10:] if len(self.episode_lengths) >= 10 else self.episode_lengths
                self.logger.record("interval/mean_episode_reward", np.mean(recent_ep_rewards), exclude="stdout")
                self.logger.record("interval/mean_episode_length", np.mean(recent_ep_lengths), exclude="stdout")
            
            self.logger.dump(self.num_timesteps)
        
        # Episode-end logging.
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(info.get('episode_length', 0))
            
            self.logger.record("time/num_timesteps", self.num_timesteps)
            self.logger.record("episode/total_reward", info.get('total_reward', 0))
            self.logger.record("episode/routes_completed", info.get('routes_completed', 0))
            self.logger.record("episode/total_distance", info.get('total_distance', 0))
            self.logger.record("episode/avg_center_dev", info.get('avg_center_dev', 0))
            self.logger.record("episode/avg_speed", info.get('avg_speed', 0))
            self.logger.record("episode/mean_reward", info.get('mean_reward', 0))
            self.logger.record("episode/collision_rate", info.get('collision_rate', 0))
            self.logger.record("episode/collision_num", info.get('collision_num', 0))
            self.logger.record("episode/episode_length", info.get('episode_length', 0))
            
            if info.get('collision_state', False):
                self.logger.record("episode/CPS", info.get('CPS', 0))
                self.logger.record("episode/CPM", info.get('CPM', 0))
                self.logger.record("episode/collision_interval", info.get('collision_interval', 0))
                self.logger.record("episode/collision_speed", info.get('collision_speed', 0))

            self.logger.dump(self.num_timesteps)
            
            self.current_episode_reward = 0.0
            
            # Keep a bounded rolling window.
            if len(self.episode_rewards) > 100:
                self.episode_rewards = self.episode_rewards[-100:]
                self.episode_lengths = self.episode_lengths[-100:]

        return True


class VideoRecorderCallback(BaseCallback):
    def __init__(self, video_path, frame_size, video_length=-1, fps=30, skip_frame=1, verbose=0):
        super().__init__(verbose)
        self.video_recorder = VideoRecorder(video_path, frame_size, fps)
        self.max_length = video_length
        self.skip_frame = skip_frame

    def _on_step(self) -> bool:
        # Stop when reaching max recording length.
        if self.max_length != -1 and self.num_timesteps > self.max_length:
            self.video_recorder.release()
            return False
        if self.num_timesteps % self.skip_frame != 0:
            return True
        display = self.training_env.unwrapped.envs[0].env.display
        frame = np.array(pygame.surfarray.array3d(display), dtype=np.uint8).transpose([1, 0, 2])

        self.video_recorder.add_frame(frame)
        return True

    def _on_training_end(self) -> None:
        self.video_recorder.release()

class CurriculumWarmupCallback(BaseCallback):
    """
    Warmup callback for curriculum phase transitions.
    At each phase threshold, pause updates and collect warmup data.
    """
    def __init__(self, warmup_steps=10000, phase_thresholds=[200000, 500000], verbose=0):
        super(CurriculumWarmupCallback, self).__init__(verbose)
        self.warmup_steps = warmup_steps
        self.phase_thresholds = phase_thresholds
        self.current_phase = 0
        self.warmup_mode = False
        self.warmup_start_step = 0
        self.original_learning_starts = None
        
    def _on_step(self) -> bool:
        """Run phase-switch warmup logic on each step."""
        current_step = self.num_timesteps
        
        if not self.warmup_mode:
            for i, threshold in enumerate(self.phase_thresholds):
                if current_step >= threshold and self.current_phase == i:
                    # Enter warmup mode.
                    self.warmup_mode = True
                    self.warmup_start_step = current_step
                    self.current_phase = i + 1
                    
                    # Temporarily delay learning updates.
                    if hasattr(self.model, 'learning_starts'):
                        self.original_learning_starts = self.model.learning_starts
                        self.model.learning_starts = current_step + self.warmup_steps + 1000
                    
                    # Disable training mode during warmup exploration.
                    if hasattr(self.model.policy, 'set_training_mode'):
                        self.model.policy.set_training_mode(False)
                    break
        
        elif self.warmup_mode:
            warmup_progress = current_step - self.warmup_start_step
            
            if warmup_progress >= self.warmup_steps:
                # Exit warmup mode.
                self.warmup_mode = False
                
                # Restore learning and training state.
                if hasattr(self.model, 'learning_starts') and self.original_learning_starts is not None:
                    self.model.learning_starts = self.original_learning_starts
                
                if hasattr(self.model.policy, 'set_training_mode'):
                    self.model.policy.set_training_mode(True)
        
        return True
    
    def _on_training_start(self) -> None:
        """Initialize phase index from current timestep."""
        current_step = self.num_timesteps
        for i, threshold in enumerate(self.phase_thresholds):
            if current_step >= threshold:
                self.current_phase = i + 1

def lr_schedule(initial_value: float, end_value: float, rate: float):
    """
    Learning rate schedule:
        Exponential decay by factors of 10 from initial_value to end_value.

    :param initial_value: Initial learning rate.
    :param rate: Exponential rate of decay. High values mean fast early drop in LR
    :param end_value: The final value of the learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: A float value between 0 and 1 that represents the remaining progress.
        :return: The current learning rate.
        """
        if progress_remaining <= 0:
            return end_value

        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))

    func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    lr_schedule.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"

    return func


class HistoryWrapperObsDict(gym.Wrapper):
    # History Wrapper from rl-baselines3-zoo
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/10de3a8804b14b4ea605b487ae7d8117c52901c4/rl_zoo3/wrappers.py
    """
    History Wrapper for dict observation.
    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2, obs_key: str = 'vae_latent') -> object:
        self.obs_key = obs_key
        assert isinstance(env.observation_space.spaces[obs_key], gym.spaces.Box)
        wrapped_obs_space = env.observation_space.spaces[self.obs_key]
        wrapped_action_space = env.action_space

        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Replace observation space with history-augmented space.
        env.observation_space.spaces[obs_key] = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Clear history buffers on reset.
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs_dict = self.env.reset()
        obs = obs_dict[self.obs_key]
        self.obs_history[..., -obs.shape[-1]:] = obs

        obs_dict[self.obs_key] = self._create_obs_from_history()

        return obs_dict

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = obs_dict[self.obs_key]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action

        obs_dict[self.obs_key] = self._create_obs_from_history()

        return obs_dict, reward, done, info


class FrameSkip(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action: np.ndarray):
        """
        Step the environment with the given action
        Repeat action, sum reward.
        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()


def parse_wrapper_class(wrapper_class_str: str):
    """
    Parse a string to a wrapper class.

    :param wrapper_class_str: (str) The string to parse.
    :return: (type) The wrapper class and its parameters.
    """
    wrap_class, wrap_params = wrapper_class_str.split("_", 1)
    wrap_params = wrap_params.split("_")
    wrap_params = [int(param) if param.isnumeric() else param for param in wrap_params]

    if wrap_class == "HistoryWrapperObsDict":
        return HistoryWrapperObsDict, wrap_params
    elif wrap_class == "FrameSkip":
        return FrameSkip, wrap_params


def analyze_and_save_model_parameters(model, save_dir):
    """
    Analyze model parameters and save results to local files.
    """
    # Structured analysis output.
    analysis_results = {
        "model_info": {
            "model_class": model.__class__.__name__,
            "total_parameters": 0,
            "trainable_parameters": 0,
            "has_custom_cnn": False,
            "has_self_attention": False,
            "has_multi_input_extractor": False
        },
        "layer_details": {},
        "parameter_groups": {},
        "attention_layers": []
    }
    
    # Collect parameters from model or policy.
    try:
        if hasattr(model, 'parameters'):
            model_parameters = list(model.parameters())
        elif hasattr(model, 'policy') and hasattr(model.policy, 'parameters'):
            model_parameters = list(model.policy.parameters())
        else:
            model_parameters = []
    except Exception:
        model_parameters = []
    
    # Basic parameter statistics.
    total_params = sum(p.numel() for p in model_parameters)
    trainable_params = sum(p.numel() for p in model_parameters if p.requires_grad)
    
    analysis_results["model_info"]["total_parameters"] = total_params
    analysis_results["model_info"]["trainable_parameters"] = trainable_params

    # Collect analyzable submodules.
    analyzable_components = []

    if hasattr(model, 'policy') and model.policy is not None:
        analyzable_components.append(('policy', model.policy))

    if hasattr(model, 'critic') and model.critic is not None:
        analyzable_components.append(('critic', model.critic))
    if hasattr(model, 'critic_target') and model.critic_target is not None:
        analyzable_components.append(('critic_target', model.critic_target))

    if hasattr(model, 'actor') and model.actor is not None:
        analyzable_components.append(('actor', model.actor))
    if hasattr(model, 'actor_target') and model.actor_target is not None:
        analyzable_components.append(('actor_target', model.actor_target))

    if not analyzable_components:
        analyzable_components.append(('model', model))
    
    layer_count = 0
    for component_name, component in analyzable_components:
        try:
            if component is None:
                continue
                
            for name, module in component.named_modules():
                if len(list(module.children())) == 0:
                    layer_count += 1
                    try:
                        param_count = sum(p.numel() for p in module.parameters())
                    except:
                        param_count = 0
                    
                    full_name = f"{component_name}.{name}" if name else component_name
                    
                    if "CustomCNN" in str(type(module)) or "custom" in name.lower():
                        analysis_results["model_info"]["has_custom_cnn"] = True
                        
                    if "SelfAttention" in str(type(module)) or "self_attention" in name.lower():
                        analysis_results["model_info"]["has_self_attention"] = True
                        analysis_results["attention_layers"].append({
                            "name": full_name,
                            "type": str(type(module)),
                            "parameters": param_count
                        })
                        
                    if "CustomMultiInputExtractor" in str(type(module)) or "multi_input" in name.lower():
                        analysis_results["model_info"]["has_multi_input_extractor"] = True
                    
                    analysis_results["layer_details"][full_name] = {
                        "type": str(type(module)),
                        "parameters": param_count,
                        "shape": str(getattr(module, 'weight', 'N/A')).replace('tensor', '').replace('Parameter containing:', '') if hasattr(module, 'weight') else 'N/A'
                    }
        except Exception:
            continue
    
    extractor_found = False
    for component_name, component in analyzable_components:
        if component is None:
            continue
            
        if hasattr(component, 'features_extractor'):
            extractor = component.features_extractor
            if extractor is None:
                continue
                
            extractor_found = True
            
            attention_found = False
            try:
                for name, module in extractor.named_modules():
                    if "attention" in name.lower() or "SelfAttention" in str(type(module)):
                        attention_found = True
            except Exception:
                pass
            
            if attention_found:
                analysis_results["model_info"]["has_self_attention"] = True
            
            cnn_found = False
            try:
                for name, module in extractor.named_modules():
                    if "cnn" in name.lower() or "CustomCNN" in str(type(module)):
                        cnn_found = True
            except Exception:
                pass
            
            if cnn_found:
                analysis_results["model_info"]["has_custom_cnn"] = True

    if not extractor_found:
        pass
    
    param_groups = {}
    for component_name, component in analyzable_components:
        if component is None:
            continue
            
        try:
            for name, param in component.named_parameters():
                full_name = f"{component_name}.{name}"
                group_name = full_name.split('.')[0] if '.' in full_name else 'root'
                if group_name not in param_groups:
                    param_groups[group_name] = 0
                param_groups[group_name] += param.numel()
        except Exception:
            continue
    
    analysis_results["parameter_groups"] = param_groups

    # Save all artifacts locally.
    os.makedirs(save_dir, exist_ok=True)

    analysis_file = os.path.join(save_dir, "model_analysis.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    params_file = os.path.join(save_dir, "model_parameters.txt")
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPLETE MODEL PARAMETER LIST\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write("-"*80 + "\n\n")
        
        param_index = 1
        for component_name, component in analyzable_components:
            if component is None:
                f.write(f"\n--- {component_name.upper()} COMPONENT ---\n")
                f.write("Component is None\n")
                continue
                
            f.write(f"\n--- {component_name.upper()} COMPONENT ---\n")
            try:
                for name, param in component.named_parameters():
                    full_name = f"{component_name}.{name}"
                    f.write(f"{param_index:4d}. {full_name}\n")
                    f.write(f"      Shape: {list(param.shape)}\n")
                    f.write(f"      Parameters: {param.numel():,}\n")
                    f.write(f"      Requires Grad: {param.requires_grad}\n")
                    f.write(f"      Device: {param.device}\n")
                    f.write(f"      Data Type: {param.dtype}\n")
                    if param.numel() <= 10:
                        f.write(f"      Values: {param.data.flatten().tolist()}\n")
                    f.write("\n")
                    param_index += 1
            except Exception as e:
                f.write(f"Error analyzing component {component_name}: {e}\n")
    
    structure_file = os.path.join(save_dir, "model_structure.txt")
    with open(structure_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL STRUCTURE\n")
        f.write("="*80 + "\n")
        f.write(f"Model Class: {model.__class__.__name__}\n")
        f.write(f"Model Type: {type(model)}\n\n")
        
        for component_name, component in analyzable_components:
            f.write(f"\n--- {component_name.upper()} COMPONENT STRUCTURE ---\n")
            if component is None:
                f.write("Component is None\n")
            else:
                f.write(str(component))
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FEATURE EXTRACTOR STRUCTURE\n")
        f.write("="*80 + "\n")
        
        extractor_found = False
        for component_name, component in analyzable_components:
            if component is None:
                continue
                
            if hasattr(component, 'features_extractor'):
                extractor = component.features_extractor
                f.write(f"\n--- {component_name.upper()} FEATURE EXTRACTOR ---\n")
                if extractor is None:
                    f.write("Feature extractor is None\n")
                else:
                    f.write(str(extractor))
                    extractor_found = True
                f.write("\n")
        
        if not extractor_found:
            f.write("No feature extractors found")
    
    return analysis_results