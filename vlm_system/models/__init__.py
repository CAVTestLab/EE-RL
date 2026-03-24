from .model_interface import VLMModelInterface
from .model_config import VLMModelConfig

try:
    from .qwen_vlm_model import QwenVLMModel
    __all__ = [
        "VLMModelInterface",
        "VLMModelConfig", 
        "QwenVLMModel"
    ]
except ImportError as e:
    print(f"Warning: QwenVLMModel could not be imported: {e}")
    __all__ = [
        "VLMModelInterface",
        "VLMModelConfig",
    ]

def create_vlm_model(model_type: str = "qwen", config: VLMModelConfig = None) -> VLMModelInterface:
    if config is None:
        config = VLMModelConfig()
    
    if model_type == "qwen":
        try:
            return QwenVLMModel(config, enable_monitor=config.enable_monitor)
        except NameError:
            print("QwenVLMModel is not available. Please check your installation.")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
