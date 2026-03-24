import torch as th
import numpy as np
import time
import threading
import queue
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
from ..models.model_interface import VLMModelInterface
from .processor_config import ProcessorConfig
if TYPE_CHECKING:
    from ..buffers.auxiliary_buffer import AuxiliaryVLMReplayBuffer


class AsyncVLMProcessor:
    """Process VLM reward correction in a background worker thread."""
    def __init__(
        self, 
        vlm_model: Optional[VLMModelInterface] = None,
        config: Optional[ProcessorConfig] = None
    ):
        """Initialize processor state, queues, and thread-safe counters."""
        self.config = config or ProcessorConfig()
        self.vlm_model = vlm_model
        self.task_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        self.processed_count = 0
        self.error_count = 0
        self._stats_lock = threading.Lock()
        
    def start(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
    
    def submit_experience(self, experience: Dict, vlm_buffer: "AuxiliaryVLMReplayBuffer") -> bool:
        """Submit one experience for asynchronous VLM post-processing."""
        try:
            task = {
                'experience': experience,
                'vlm_buffer': vlm_buffer,
                'timestamp': time.time()
            }
            self.task_queue.put_nowait(task)
            return True
        except queue.Full:
            return False
    
    def _worker_loop(self):
        """Consume queued tasks and process them until stopped."""
        while self.is_running:
            try:
                # Poll with timeout so stop() can be observed quickly.
                task = self.task_queue.get(timeout=1.0)
                
                self._process_task(task)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                with self._stats_lock:
                    self.error_count += 1
    
    def _process_task(self, task: Dict):
        """Run VLM correction and push corrected experience to buffer."""
        try:
            experience = task['experience']
            vlm_buffer = task['vlm_buffer']
            
            # Build model input from experience fields.
            vlm_input = self._prepare_vlm_input(experience)
            
            # Predict reward delta and combine with original reward.
            vlm_reward_delta = self.vlm_model.predict_reward_correction(vlm_input)
            
            corrected_total_reward = experience['reward'] + vlm_reward_delta

            # Normalize corrected reward before buffer insertion.
            vlm_corrected_reward = self._normalize_vlm_corrected_reward(corrected_total_reward)
            vlm_buffer.add_vlm_corrected_experience(
                obs=experience['obs'],
                next_obs=experience['next_obs'],
                action=experience['action'],
                original_reward=experience['reward'],
                vlm_corrected_reward=vlm_corrected_reward,
                done=experience['done'],
                speed=experience['speed'],
                vehicle_yaw=experience['vehicle_yaw'],
                throttle=experience['throttle'],
                steer=experience['steer'],
                current_maneuver=experience['current_maneuver'],
                image=experience['image'],
                timestamp=time.time(),
                env_idx=0,
            )
            
            with self._stats_lock:
                self.processed_count += 1
            
        except Exception as e:
            with self._stats_lock:
                self.error_count += 1

    def _normalize_vlm_corrected_reward(self, corrected_reward: float) -> float:
        """Map corrected reward from [-2, 3] into [0, 1] via min-max scaling."""
        
        min_possible = -2.0
        max_possible = 3.0

        normalized = (corrected_reward - min_possible) / (max_possible - min_possible)
        return np.clip(normalized, 0.0, 1.0)
    
    def _prepare_vlm_input(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Extract required model inputs and fill missing fields with defaults."""
        vlm_input = {}
        
        # Enforce all required keys expected by the VLM model.
        for key in self.config.required_input_keys:
            if key in experience:
                vlm_input[key] = experience[key]
            else:
                # Fallback defaults for incomplete experiences.
                if key == 'speed':
                    vlm_input[key] = 0.0
                elif key == 'vehicle_yaw':
                    vlm_input[key] = 0.0
                elif key == 'throttle':
                    vlm_input[key] = 0.0
                elif key == 'steer':
                    vlm_input[key] = 0.0
                elif key == 'current_maneuver':
                    vlm_input[key] = 'UNKNOWN'
                elif key == 'traffic_light_state':
                    vlm_input[key] = 'Unknown'
                elif key == 'traffic_light_distance':
                    vlm_input[key] = float('inf')
                elif key == 'image':
                    vlm_input[key] = None
        
        return vlm_input
    
    def get_status(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                'is_running': self.is_running,
                'task_queue_size': self.task_queue.qsize(),
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config),
            }
    
    def clear_queue(self):
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
    
    def wait_for_completion(self, timeout: float = 30.0):
        self.task_queue.join()
    
    def __del__(self):
        try:
            self.stop()
        except:
            pass
        self.task_queue.join()
    
    def __del__(self):
        try:
            self.stop()
        except:
            pass
