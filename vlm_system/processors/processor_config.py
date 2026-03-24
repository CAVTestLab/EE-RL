from dataclasses import dataclass, field
from typing import Callable, Optional, Any


@dataclass
class VLMProcessorConfig:
    max_queue_size: int = 50
    processing_timeout: float = 1.0
    
    vlm_model_fn: Optional[Callable] = None
    inference_device: str = "cpu"
    
    required_input_keys: list = field(default_factory=lambda: [
        'image', 'speed', 'vehicle_yaw', 'throttle', 'steer', 'current_maneuver',
        'traffic_light_state', 'traffic_light_distance'
    ])
    optional_input_keys: list = field(default_factory=lambda: [
        'reward', 'done', 'obs', 'next_obs', 'action'
    ])
    
    batch_processing: bool = False
    batch_size: int = 8
    processing_interval: float = 0.01  # seconds
    
    enable_monitoring: bool = True
    log_errors: bool = True
    
    def __post_init__(self):
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        
        if self.processing_timeout <= 0:
            raise ValueError("processing_timeout must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.processing_interval < 0:
            raise ValueError("processing_interval must be non-negative")

# Backward compatibility
ProcessorConfig = VLMProcessorConfig
