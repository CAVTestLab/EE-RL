import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Optional, Tuple, TypeVar, Type, Union, Dict, Any, List

import numpy as np
import torch as th
import torch
from gymnasium import spaces
from box import Box
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_setattr, load_from_zip_file
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import safe_mean, check_for_correct_spaces, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..buffers.main_buffer import VLMReplayBuffer
    from ..models.model_interface import VLMModelInterface

SelfVLMRewardedTD3 = TypeVar("SelfVLMRewardedTD3", bound="VLMRewardedTD3")


class VLMRewardedTD3(TD3):
    replay_buffer: "VLMReplayBuffer"

    def __init__(
        self,
        *,
        env: VecEnv,
        config: Box,
        vlm_model: Optional["VLMModelInterface"] = None,
        inference_only: bool = False,
        use_dual_buffer: bool = True,
        aux_buffer_size: int = 3000,
        sampling_ratio: float = 10.0,
    ):
        self.config = config
        self.vlm_model = vlm_model
        self.use_dual_buffer = use_dual_buffer
        self.aux_buffer_size = aux_buffer_size
        self.sampling_ratio = sampling_ratio
        self.inference_only = inference_only

        super().__init__(
            env=env,
            policy="MultiInputPolicy",
            tensorboard_log="tensorboard",
            seed=config.seed,
            _init_setup_model=False,
            **self.config.algorithm_params,
        )
        
        self.ep_vlm_info_buffer = None
        
        if not self.inference_only:
            self._setup_model()

    def _setup_model(self):
        super()._setup_model()
        from ..buffers.main_buffer import VLMReplayBuffer
        
        self.replay_buffer = VLMReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            use_dual_buffer=self.use_dual_buffer,
            aux_buffer_size=self.aux_buffer_size,
            sampling_ratio=self.sampling_ratio,
        )
        
        if self.vlm_model and hasattr(self.replay_buffer, 'async_vlm_processor'):
            self.replay_buffer.async_vlm_processor.vlm_model = self.vlm_model

    def set_vlm_model(self, vlm_model: "VLMModelInterface"):
        self.vlm_model = vlm_model
        if hasattr(self.replay_buffer, 'async_vlm_processor'):
            self.replay_buffer.async_vlm_processor.vlm_model = vlm_model

    def trigger_vlm_correction(self, num_experiences: int = 50) -> None:
        if not hasattr(self.replay_buffer, 'vlm_processing_enabled') or not self.replay_buffer.vlm_processing_enabled:
            return

        self.replay_buffer.force_vlm_correction(num_experiences)

    def get_vlm_statistics(self) -> Dict[str, Any]:
        stats = {}
        
        if hasattr(self.replay_buffer, 'get_async_vlm_status'):
            stats['vlm_processor'] = self.replay_buffer.get_async_vlm_status()
        
        if hasattr(self.replay_buffer, 'get_vlm_queue_info'):
            stats['vlm_queue'] = self.replay_buffer.get_vlm_queue_info()
        
        if self.vlm_model and hasattr(self.vlm_model, 'get_model_info'):
            stats['vlm_model'] = self.vlm_model.get_model_info()
        
        return stats

    def enable_vlm_processing(self):
        if hasattr(self.replay_buffer, 'enable_vlm_processing'):
            self.replay_buffer.enable_vlm_processing()

    def disable_vlm_processing(self):
        if hasattr(self.replay_buffer, 'disable_vlm_processing'):
            self.replay_buffer.disable_vlm_processing()

    def set_vlm_correction_interval(self, interval: int):
        if hasattr(self.replay_buffer, 'set_vlm_correction_interval'):
            self.replay_buffer.set_vlm_correction_interval(interval)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: "VLMReplayBuffer",
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "A VecEnv instance is required."
        assert train_freq.frequency > 0, "At least one step or one episode must be collected."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "Episode-based training supports only a single environment."

        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                self.actor.reset_noise(env.num_envs)

            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            new_obs, rewards, dones, infos = env.step(actions)
            for env_idx, info in enumerate(infos):
                executed_action = info.get('executed_action', None)
                if executed_action is None:
                    continue
                executed_action = np.asarray(executed_action, dtype=np.float32)
                if executed_action.ndim == 2 and executed_action.shape[0] == 1:
                    executed_action = executed_action[0]
                if executed_action.ndim == 1 and env_idx < buffer_actions.shape[0]:
                    buffer_actions[env_idx] = executed_action
                    
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            self._update_info_buffer(infos, dones)

            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        
        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def save(self, *args, **kwargs) -> None:
        super().save(*args, exclude=["vlm_model"], **kwargs)

    @classmethod
    def load(
            cls: Type[SelfVLMRewardedTD3],
            path: Union[str, pathlib.Path],
            *,
            env: Optional[VecEnv] = None,
            device: Union[torch.device, str] = "cuda:0",
            custom_objects: Optional[Dict[str, Any]] = None,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfVLMRewardedTD3:
        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
        )

        assert data is not None, "No data found in the saved archive."
        assert params is not None, "No parameters found in the saved archive."

        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("Observation space and action space are required to validate the environment.")

        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            env = cls._wrap_env(env, data["verbose"])
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            if force_reset and data is not None:
                data["_last_obs"] = None
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            if "env" in data:
                env = data["env"]

        if "config" not in data:
            data["config"] = Box(default_box=True)

        data["config"].algorithm_params.device = device
        model = cls(
            env=env,
            config=data["config"],
        )

        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "A model saved with SB3 < 1.7.0 may be loaded. "
                    "exact_match has been disabled; consider re-saving the model to avoid future issues."
                )
            else:
                raise e

        if pytorch_variables is not None:
            for name in pytorch_variables:
                if pytorch_variables[name] is None:
                    continue
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        if model.use_sde:
            model.policy.reset_noise()

        return model

    def cleanup(self):
        if hasattr(self.replay_buffer, 'cleanup'):
            self.replay_buffer.cleanup()
        
        if self.vlm_model and hasattr(self.vlm_model, 'cleanup'):
            self.vlm_model.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass
