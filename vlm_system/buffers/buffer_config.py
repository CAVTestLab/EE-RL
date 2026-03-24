from dataclasses import dataclass
from typing import Union
import torch as th
import config


@dataclass
class BufferConfig:
    
    buffer_size: int = 10000 # config.py setting
    device: Union[th.device, str] = "auto" 
    n_envs: int = 1
    optimize_memory_usage: bool = False
    handle_timeout_termination: bool = True
    
    use_dual_buffer: bool = True
    aux_buffer_size: int = 3000 # train.py setting
    sampling_ratio: float = 10.0 # train.py setting
    
    vlm_correction_interval: int = 50 # train.py setting
    vlm_processing_enabled: bool = True # train.py setting
    
    image_shape: tuple = config.RGB_CAMERA_RESOLUTION + (3,)
    
    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        
        if self.aux_buffer_size <= 0:
            raise ValueError("aux_buffer_size must be positive")
        
        if self.sampling_ratio <= 0:
            raise ValueError("sampling_ratio must be positive")
        
        if self.vlm_correction_interval <= 0:
            raise ValueError("vlm_correction_interval must be positive")


@dataclass 
class AuxiliaryBufferConfig:
    
    buffer_size: int = 3000
    device: Union[th.device, str] = "auto"
    n_envs: int = 1
    optimize_memory_usage: bool = False
    handle_timeout_termination: bool = True
    image_shape: tuple = config.RGB_CAMERA_RESOLUTION + (3,)
    
    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")


@dataclass
class MainBufferConfig(BufferConfig):
    async_processing: bool = True
    correction_threshold: float = 0.1
    
    def __post_init__(self):
        super().__post_init__()
        if self.correction_threshold < 0:
            raise ValueError("correction_threshold must be non-negative")
