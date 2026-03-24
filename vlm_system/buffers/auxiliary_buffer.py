import torch as th
import numpy as np
import threading
import time
from typing import Dict, Any, Union, Optional
from gym import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from .buffer_config import AuxiliaryBufferConfig
import config


class AuxiliaryVLMReplayBuffer(DictReplayBuffer):
    """Replay buffer dedicated to VLM-corrected transitions with fixed-capacity FIFO overwrite."""
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        # Allocate extra fields required by the VLM correction pipeline.
        self.speeds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.vehicle_yaws = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.throttles = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.steers = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.current_maneuvers = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.images = np.zeros((self.buffer_size, self.n_envs, config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
        
        # Store traffic-light context for reward correction analysis.
        self.traffic_light_in_range = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.traffic_light_distances = np.full((self.buffer_size, self.n_envs), np.inf, dtype=np.float32)
        self.traffic_light_states = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.traffic_light_affects_lane = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        # Keep both original/corrected reward values and queue metadata.
        self.vlm_corrected_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.original_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.correction_timestamps = np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
        self.queue_head = 0
        self.queue_tail = 0
        self.is_queue_full = False
        
        # Use separate locks for write and read paths in async processing.
        self._write_lock = threading.RLock()
        self._read_lock = threading.RLock()
    
    def add_vlm_corrected_experience(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        original_reward: np.ndarray,
        vlm_corrected_reward: np.ndarray,
        done: np.ndarray,
        speed: float,
        vehicle_yaw: float,
        throttle: float,
        steer: float,
        current_maneuver: str,
        image: np.ndarray,
        timestamp: float,
        env_idx: int = 0,
        # Traffic-light metadata
        traffic_light_in_range: bool = False,
        traffic_light_distance: float = np.inf,
        traffic_light_state: str = "Unknown",
        traffic_light_affects_lane: bool = False,
    ) -> None:
        """Append one VLM-corrected transition in a thread-safe FIFO manner."""
        with self._write_lock:
            # Always write at the logical queue tail.
            pos = self.queue_tail
            
            # Persist observation dictionaries.
            for key in obs.keys():
                self.observations[key][pos, env_idx] = obs[key]
                self.next_observations[key][pos, env_idx] = next_obs[key]
            
            # Persist core transition fields; reward uses VLM-corrected value.
            self.actions[pos, env_idx] = action
            self.rewards[pos, env_idx] = vlm_corrected_reward
            self.dones[pos, env_idx] = done
            
            # Persist extended driving context.
            self.speeds[pos, env_idx] = speed
            self.vehicle_yaws[pos, env_idx] = vehicle_yaw
            self.throttles[pos, env_idx] = throttle
            self.steers[pos, env_idx] = steer
            self.current_maneuvers[pos, env_idx] = current_maneuver
            
            # Persist traffic-light context.
            self.traffic_light_in_range[pos, env_idx] = traffic_light_in_range
            self.traffic_light_distances[pos, env_idx] = traffic_light_distance
            self.traffic_light_states[pos, env_idx] = traffic_light_state
            self.traffic_light_affects_lane[pos, env_idx] = traffic_light_affects_lane
            
            # Keep image shape consistent with configured camera resolution.
            if isinstance(image, np.ndarray):
                if image.shape == config.RGB_CAMERA_RESOLUTION + (3,):
                    self.images[pos, env_idx] = image
                else:
                    # Resize incoming image when shape does not match target size.
                    import cv2
                    resized_image = cv2.resize(image, config.RGB_CAMERA_RESOLUTION)
                    self.images[pos, env_idx] = resized_image
        
            # Store reward-correction bookkeeping fields.
            self.vlm_corrected_rewards[pos, env_idx] = vlm_corrected_reward
            self.original_rewards[pos, env_idx] = original_reward
            self.correction_timestamps[pos, env_idx] = timestamp
            
            # Advance queue pointers with FIFO overwrite when capacity is reached.
            next_tail = (self.queue_tail + 1) % self.buffer_size
            
            if next_tail == self.queue_head and not self.is_queue_full:
                self.is_queue_full = True
            
            self.queue_tail = next_tail
            
            if self.is_queue_full:
                self.queue_head = (self.queue_head + 1) % self.buffer_size
            
            # Mirror queue state to parent buffer fields used by SB3 internals.
            self.pos = self.queue_tail
            self.full = self.is_queue_full
    
    def get_queue_size(self) -> int:
        with self._read_lock:
            if self.is_queue_full:
                return self.buffer_size
            else:
                return (self.queue_tail - self.queue_head) % self.buffer_size
    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        with self._read_lock:
            return super().sample(batch_size, env)
    
    def reset(self) -> None:
        """Reset buffer contents and queue metadata to the initial state."""
        super().reset()
        self.speeds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.vehicle_yaws = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.throttles = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.steers = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.current_maneuvers = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.images = np.zeros((self.buffer_size, self.n_envs, config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
        
        # Clear traffic-light context arrays.
        self.traffic_light_in_range = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.traffic_light_distances = np.full((self.buffer_size, self.n_envs), np.inf, dtype=np.float32)
        self.traffic_light_states = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.traffic_light_affects_lane = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        self.vlm_corrected_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.original_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.correction_timestamps = np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
        self.queue_head = 0
        self.queue_tail = 0
        self.is_queue_full = False
    
    def get_queue_info(self) -> Dict[str, Any]:
        with self._read_lock:
            return {
                'queue_size': self.get_queue_size(),
                'queue_capacity': self.buffer_size,
                'queue_head': self.queue_head,
                'queue_tail': self.queue_tail,
                'is_full': self.is_queue_full,
            }
