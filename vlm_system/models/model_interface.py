from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import torch as th
import numpy as np

from .model_config import VLMModelConfig


class VLMModelInterface(ABC):
    def __init__(self, config: VLMModelConfig):
        self.config = config
        self.is_loaded = False
        self.device = config.device
    
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def predict_reward(self, input_data: Dict[str, Any]) -> float:
        pass
    
    @abstractmethod
    def predict_batch_rewards(self, batch_input: List[Dict[str, Any]]) -> List[float]:
        pass
    
    @abstractmethod
    def predict_reward_correction(self, input_data: Dict[str, Any]) -> float:
        pass
    
    def preprocess_image(self, image: Union[np.ndarray, th.Tensor]) -> th.Tensor:
        if isinstance(image, np.ndarray):
            image = th.from_numpy(image)
        image = image.to(self.device)
        
        if self.config.normalize_inputs:
            image = image.float() / 255.0
        
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        
        return image
    
    def preprocess_state(self, state_data: Dict[str, Any]) -> Dict[str, th.Tensor]:
        processed = {}
        for key, value in state_data.items():
            if isinstance(value, (int, float)):
                processed[key] = th.tensor([value], dtype=th.float32, device=self.device)
            elif isinstance(value, np.ndarray):
                processed[key] = th.from_numpy(value).float().to(self.device)
            elif isinstance(value, th.Tensor):
                processed[key] = value.float().to(self.device)
            else:
                
                try:
                    processed[key] = th.tensor(value, dtype=th.float32, device=self.device)
                except:
                    print(f"{key}: {type(value)}")
        
        return processed
    
    def postprocess_reward(self, raw_reward: Union[float, th.Tensor]) -> float:
        if isinstance(raw_reward, th.Tensor):
            raw_reward = raw_reward.item()
        
        scaled_reward = raw_reward * self.config.reward_scale
        return float(scaled_reward)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.config.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'config': self.config.__dict__,
        }
    
    def __call__(self, input_data: Dict[str, Any]) -> float:
        return self.predict_reward(input_data)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.config.model_name}, device={self.device})"
