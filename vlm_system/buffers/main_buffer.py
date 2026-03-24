import torch as th
import numpy as np
import time
from typing import Dict, Any, List, Union, Optional
from gym import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from .auxiliary_buffer import AuxiliaryVLMReplayBuffer
from .buffer_config import MainBufferConfig
from ..processors.async_processor import AsyncVLMProcessor
import config


class VLMReplayBuffer(DictReplayBuffer):
    """Main replay buffer with optional dual-buffer VLM correction flow."""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        use_dual_buffer: bool = False,
        aux_buffer_size: int = 500,
        sampling_ratio: float = 10.0,
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

        # Allocate additional fields needed by VLM processing.
        self.speeds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.vehicle_yaws = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.throttles = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.steers = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.current_maneuvers = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.images = np.zeros((self.buffer_size, self.n_envs, config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
        
        self.traffic_light_in_range = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.traffic_light_distances = np.full((self.buffer_size, self.n_envs), np.inf, dtype=np.float32)
        self.traffic_light_states = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.traffic_light_affects_lane = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        self.use_dual_buffer = use_dual_buffer
        self.sampling_ratio = sampling_ratio
        
        if self.use_dual_buffer:
            self.vlm_queue = AuxiliaryVLMReplayBuffer(
                aux_buffer_size,
                observation_space,
                action_space,
                device,
                n_envs,
                optimize_memory_usage,
                handle_timeout_termination,
            )
            self.pending_vlm_corrections = []
            self.vlm_processing_enabled = True
            
            self.async_vlm_processor = AsyncVLMProcessor()
            self.async_vlm_processor.start()
            
            self.vlm_correction_interval = 50
            self.experience_count_since_last_vlm = 0
            self.traffic_light_collect_times = 0

            self.traffic_light_collection_tracker = {}
            self.collection_frame_counter = 0
            self.collection_interval = 4
            self.max_samples_per_state = 5
            self.collection_time_window = 30.0

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Add one transition batch and persist extended environment context."""
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Use the same index just written by the parent replay buffer.
        pos = (self.pos - 1) % self.buffer_size
        
        for i, info in enumerate(infos):
            if i < self.n_envs:
                self.speeds[pos, i] = info.get("vehicle_speed", 0.0)
                self.vehicle_yaws[pos, i] = info.get("vehicle_yaw", 0.0)
                self.current_maneuvers[pos, i] = info.get("current_maneuver", "UNKNOWN")
                
                self.traffic_light_in_range[pos, i] = info.get("traffic_light_in_range", False)
                self.traffic_light_distances[pos, i] = info.get("traffic_light_distance", np.inf)
                self.traffic_light_states[pos, i] = info.get("traffic_light_state", "Unknown")
                self.traffic_light_affects_lane[pos, i] = info.get("traffic_light_affects_lane", False)
                
                # Extract steering/throttle from action values.
                if i < len(action):
                    if len(action[i]) >= 2:
                        self.steers[pos, i] = action[i][0]
                        self.throttles[pos, i] = action[i][1]
                    else:
                        self.steers[pos, i] = 0.0
                        self.throttles[pos, i] = 0.0

                image_stored = False
                
                # Prefer image from observation and normalize shape to HWC.
                if "front_rgb_camera" in obs and i < obs["front_rgb_camera"].shape[0]:
                    image_data = obs["front_rgb_camera"][i]
                    if len(image_data.shape) == 3:
                        if image_data.shape[0] == 3:
                            image_data = np.transpose(image_data, (1, 2, 0))
                        elif image_data.shape[2] == 3:
                            pass
                        else:
                            image_data = np.zeros((config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
                    else:
                        image_data = np.zeros((config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
                    
                    self.images[pos, i] = image_data.copy()
                    image_stored = True
                
                # Fall back to image provided in info payload.
                elif "front_rgb_image" in info and not image_stored:
                    if isinstance(info["front_rgb_image"], np.ndarray):
                        image_data = info["front_rgb_image"]
                        if len(image_data.shape) == 3:
                            if image_data.shape[0] == 3:
                                image_data = np.transpose(image_data, (1, 2, 0))
                            elif image_data.shape[2] == 3:
                                pass
                            else:
                                image_data = np.zeros((config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
                        else:
                            image_data = np.zeros((config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
                        
                        self.images[pos, i] = image_data.copy()
                        image_stored = True
                
                # Ensure image storage is always valid.
                if not image_stored:
                    self.images[pos, i] = np.zeros((config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
        
        # Trigger traffic-light-driven collection in dual-buffer mode.
        if self.use_dual_buffer and self.vlm_processing_enabled:
            self._process_traffic_light_collection(pos, infos)
    
    def _process_traffic_light_collection(self, pos: int, infos: List[Dict[str, Any]]):
        """Collect candidate samples keyed by traffic-light identity and state."""
        self.collection_frame_counter += 1
        current_time = time.time()
        
        # Remove stale tracking entries outside the configured time window.
        self._cleanup_expired_traffic_lights(current_time)
        
        for i, info in enumerate(infos):
            if not (info.get("traffic_light_in_range", False) and 
                    info.get("traffic_light_affects_lane", False)):
                continue
            
            traffic_light_id = info.get("traffic_light_id", None)
            traffic_light_state = info.get("traffic_light_state", "Unknown")
            
            if not traffic_light_id or traffic_light_state == "Unknown":
                continue
            
            if self._should_collect_traffic_light_sample(
                traffic_light_id, traffic_light_state, current_time
            ):
                if self.collection_frame_counter % self.collection_interval == 0:
                    self._add_traffic_light_sample(
                        pos, i, traffic_light_id, traffic_light_state, 
                        info.get("traffic_light_distance", np.inf), current_time
                    )

    def _cleanup_expired_traffic_lights(self, current_time: float):
        """Drop expired traffic-light trackers."""
        expired_lights = []
        for light_id, collection_info in self.traffic_light_collection_tracker.items():
            if current_time - collection_info['first_seen_time'] > self.collection_time_window:
                expired_lights.append(light_id)
        
        for light_id in expired_lights:
            del self.traffic_light_collection_tracker[light_id]

    def _should_collect_traffic_light_sample(
        self, traffic_light_id: int, state: str, current_time: float
    ) -> bool:
        """Return whether a sample should be collected for this light/state pair."""
        if traffic_light_id not in self.traffic_light_collection_tracker:
            self.traffic_light_collection_tracker[traffic_light_id] = {
                'first_seen_time': current_time,
                'state_counts': {'Red': 0, 'Yellow': 0, 'Green': 0, 'Unknown': 0},
                'total_samples': 0,
                'pending_samples': []
            }
        
        collection_info = self.traffic_light_collection_tracker[traffic_light_id]
        
        if current_time - collection_info['first_seen_time'] > self.collection_time_window:
            return False
        
        if collection_info['state_counts'].get(state, 0) >= self.max_samples_per_state:
            return False
        
        return True

    def _add_traffic_light_sample(
        self, pos: int, env_idx: int, traffic_light_id: int, 
        state: str, distance: float, current_time: float
    ):
        """Append one sampled traffic-light experience to pending queue."""
        collection_info = self.traffic_light_collection_tracker[traffic_light_id]
        
        collection_info['state_counts'][state] += 1
        collection_info['total_samples'] += 1
        
        sample = {
            'buffer_idx': pos,
            'env_idx': env_idx,
            'traffic_light_id': traffic_light_id,
            'traffic_light_state': state,
            'traffic_light_distance': distance,
            'timestamp': current_time
        }
        
        collection_info['pending_samples'].append(sample)

        # Submit in small batches to amortize async dispatch overhead.
        if len(collection_info['pending_samples']) >= 5:
            self._submit_traffic_light_samples_to_vlm(traffic_light_id)

    def _submit_traffic_light_samples_to_vlm(self, traffic_light_id: int):
        """Submit pending traffic-light samples to the async VLM processor."""
        if traffic_light_id not in self.traffic_light_collection_tracker:
            return
        
        collection_info = self.traffic_light_collection_tracker[traffic_light_id]
        pending_samples = collection_info['pending_samples']
        
        if not pending_samples:
            return
        
        experiences_to_process = []
        
        for sample in pending_samples:
            pos = sample['buffer_idx']
            env_idx = sample['env_idx']
            
            experience = {
                'obs': {key: self.observations[key][pos, env_idx].copy() 
                       for key in self.observations.keys()},
                'next_obs': {key: self.next_observations[key][pos, env_idx].copy() 
                            for key in self.next_observations.keys()},
                'action': self.actions[pos, env_idx].copy(),
                'reward': self.rewards[pos, env_idx],
                'done': self.dones[pos, env_idx],
                'speed': self.speeds[pos, env_idx],
                'vehicle_yaw': self.vehicle_yaws[pos, env_idx],
                'throttle': self.throttles[pos, env_idx],
                'steer': self.steers[pos, env_idx],
                'current_maneuver': self.current_maneuvers[pos, env_idx],
                'image': self.images[pos, env_idx].copy(),
                'main_buffer_idx': pos,
                'traffic_light_id': sample['traffic_light_id'],
                'traffic_light_state': sample['traffic_light_state'],
                'traffic_light_distance': sample['traffic_light_distance'],
                'is_traffic_light_experience': True,
                'collection_timestamp': sample['timestamp'],
            }
            experiences_to_process.append(experience)
        
        submitted_count = 0
        for experience in experiences_to_process:
            if self.async_vlm_processor.submit_experience(experience, self.vlm_queue):
                submitted_count += 1

        collection_info['pending_samples'].clear()

    def get_traffic_light_collection_status(self) -> Dict[str, Any]:
        """Return current traffic-light collection status."""
        current_time = time.time()
        status = {
            'total_tracked_lights': len(self.traffic_light_collection_tracker),
            'collection_frame_counter': self.collection_frame_counter,
            'lights_detail': {}
        }
        
        for light_id, info in self.traffic_light_collection_tracker.items():
            elapsed_time = current_time - info['first_seen_time']
            remaining_time = max(0, self.collection_time_window - elapsed_time)
            
            status['lights_detail'][light_id] = {
                'state_counts': info['state_counts'].copy(),
                'total_samples': info['total_samples'],
                'pending_samples': len(info['pending_samples']),
                'elapsed_time': elapsed_time,
                'remaining_time': remaining_time,
            }
        
        return status

    def sample(
        self, 
        batch_size: int, 
        env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """Sample from main buffer or mixed main/auxiliary buffers when ready."""
        # Enable mixed sampling only after the auxiliary queue reaches a warm-up threshold.
        vlm_queue_size = self.get_vlm_queue_size()
        use_mixed_sampling = (
            self.use_dual_buffer and 
            hasattr(self, 'vlm_queue') and 
            vlm_queue_size >= 2000 # Manual setting. Please don't make it too small. ！！！
        )
        
        if not use_mixed_sampling:
            return super().sample(batch_size, env)
        
        # Compute sample split from configured main-to-auxiliary ratio.
        total_ratio = 1.0 + 1.0 / self.sampling_ratio
        main_samples = int(batch_size * (1.0 / total_ratio))
        vlm_samples = batch_size - main_samples
        
        main_samples = max(1, min(main_samples, batch_size - 1))
        vlm_samples = batch_size - main_samples
        
        if vlm_samples > vlm_queue_size:
            return super().sample(batch_size, env)
        
        main_samples_data = self._sample_from_main(main_samples, env)
        vlm_samples_data = self._sample_from_vlm_queue(vlm_samples, env)
        combined_samples = self._combine_samples(main_samples_data, vlm_samples_data)
        
        return combined_samples

    def _sample_from_main(
        self, 
        batch_size: int, 
        env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """Sample from the main replay buffer."""
        return super().sample(batch_size, env)

    def _sample_from_vlm_queue(
        self, 
        batch_size: int, 
        env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """Sample from the auxiliary VLM-corrected queue."""
        available_samples = self.vlm_queue.get_queue_size()
        if batch_size > available_samples:
            batch_size = available_samples
        
        return self.vlm_queue.sample(batch_size, env)

    def _combine_samples(
        self,
        main_samples: DictReplayBufferSamples,
        vlm_samples: DictReplayBufferSamples
    ) -> DictReplayBufferSamples:
        """Concatenate samples from main and auxiliary buffers."""
        combined_observations = {}
        combined_next_observations = {}
        
        for key in main_samples.observations.keys():
            combined_observations[key] = th.cat([
                main_samples.observations[key],
                vlm_samples.observations[key]
            ], dim=0)
            
            combined_next_observations[key] = th.cat([
                main_samples.next_observations[key],
                vlm_samples.next_observations[key]
            ], dim=0)
        
        combined_actions = th.cat([main_samples.actions, vlm_samples.actions], dim=0)
        combined_dones = th.cat([main_samples.dones, vlm_samples.dones], dim=0)
        combined_rewards = th.cat([main_samples.rewards, vlm_samples.rewards], dim=0)
        
        return DictReplayBufferSamples(
            observations=combined_observations,
            actions=combined_actions,
            next_observations=combined_next_observations,
            dones=combined_dones,
            rewards=combined_rewards,
        )
    
    def get_vlm_queue_size(self) -> int:
        if not self.use_dual_buffer or not hasattr(self, 'vlm_queue'):
            return 0
        
        return self.vlm_queue.get_queue_size()
    
    def get_vlm_queue_info(self) -> Dict[str, Any]:
        if not self.use_dual_buffer or not hasattr(self, 'vlm_queue'):
            return {}
        
        return {
            'queue_size': self.vlm_queue.get_queue_size(),
            'queue_capacity': self.vlm_queue.buffer_size,
            'queue_head': self.vlm_queue.queue_head,
            'queue_tail': self.vlm_queue.queue_tail,
            'is_full': self.vlm_queue.is_queue_full,
        }

    def _get_samples(
        self, 
        batch_inds: np.ndarray, 
        env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        samples = super()._get_samples(batch_inds, env)
        
        return samples
    
    def reset(self) -> None:
        super().reset()
        self.speeds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.vehicle_yaws = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.throttles = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.steers = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.current_maneuvers = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.images = np.zeros((self.buffer_size, self.n_envs, config.RGB_CAMERA_HEIGHT, config.RGB_CAMERA_WIDTH, 3), dtype=np.uint8)
        
        self.traffic_light_in_range = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.traffic_light_distances = np.full((self.buffer_size, self.n_envs), np.inf, dtype=np.float32)
        self.traffic_light_states = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.traffic_light_affects_lane = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.traffic_light_vlm_queue.clear()
        
        if self.use_dual_buffer and hasattr(self, 'vlm_queue'):
            self.vlm_queue.reset()
            self.pending_vlm_corrections = []

    def get_extra_data(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> Dict[str, th.Tensor]:
        return {
            "speeds": self.to_torch(self.speeds[batch_inds, env_indices]),
            "vehicle_yaws": self.to_torch(self.vehicle_yaws[batch_inds, env_indices]),
            "throttles": self.to_torch(self.throttles[batch_inds, env_indices]),
            "steers": self.to_torch(self.steers[batch_inds, env_indices]),
            "current_maneuvers": self.current_maneuvers[batch_inds, env_indices],
            "images": self.to_torch(self.images[batch_inds, env_indices]),
        }

    def enable_vlm_processing(self):
        self.vlm_processing_enabled = True
        if hasattr(self, 'async_vlm_processor'):
            self.async_vlm_processor.start()
    
    def disable_vlm_processing(self):
        self.vlm_processing_enabled = False
    
    def set_vlm_correction_interval(self, interval: int):
        self.vlm_correction_interval = interval
    
    def get_async_vlm_status(self) -> Dict:
        if not hasattr(self, 'async_vlm_processor'):
            return {'error': 'async_vlm_processor not initialized'}
        
        status = self.async_vlm_processor.get_status()
        status.update({
            'vlm_processing_enabled': self.vlm_processing_enabled,
            'correction_interval': self.vlm_correction_interval,
            'experiences_since_last_vlm': self.experience_count_since_last_vlm,
            'vlm_queue_size': self.get_vlm_queue_size(),
        })
        return status
    
    def force_vlm_correction(self, num_experiences: int = 50):
        self._trigger_async_vlm_correction(num_experiences)
    
    def cleanup(self):
        if hasattr(self, 'async_vlm_processor'):
            self.async_vlm_processor.stop()
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass
