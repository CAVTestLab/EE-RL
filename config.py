import torch as th
from box import Box
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch.nn as nn
import gymnasium as gym
import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import math
import torch.nn.functional as F

# Custom CNN
class CustomCNN(nn.Module):
    def __init__(self, input_shape, features_dim=256):
        super(CustomCNN, self).__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x
# ResNet18
class EfficientCNN(nn.Module):
    def __init__(self, input_shape, features_dim=256):
        super(EfficientCNN, self).__init__()
        n_input_channels = input_shape[0]
        
        self.backbone = models.resnet18(pretrained=True)
        if n_input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
    
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
# EfficientNet-B0
class EfficientNetCNN(nn.Module):
    def __init__(self, input_shape, features_dim=256):
        super(EfficientNetCNN, self).__init__()
        
        n_input_channels = input_shape[0]
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        
        if n_input_channels != 3:
            if n_input_channels == 1:
                self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
                nn.init.constant_(self.input_adapter.weight, 1.0/3.0)
            else:
                self.input_adapter = nn.Conv2d(n_input_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()  
        num_ftrs = self.backbone._fc.in_features

        self.backbone._fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.input_adapter(x)
        x = self.backbone(x)
        return x

# #########################################################
#       Spatial attention + self-attention fusion
# #########################################################

class SpatialAttentionPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attn_map = torch.sigmoid(self.attn_conv(x))
        x = x * attn_map
        x = self.pool(x)
        return x.flatten(start_dim=1)

class SpatialAttentionCNN(nn.Module):
    def __init__(self, input_shape, features_dim=256):
        super(SpatialAttentionCNN, self).__init__()
        n_input_channels = input_shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 7, 2, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, 2, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.spatial_attn_pool = SpatialAttentionPool(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat = self.cnn(x)
        feat = self.spatial_attn_pool(feat)
        return self.classifier(feat)

class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=4, ffn_expand=2):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.head_dim = in_dim // num_heads
        
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        
        self.query = nn.Linear(in_dim, in_dim)
        self.key   = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.out   = nn.Linear(in_dim, in_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(in_dim)
        
        ffn_dim = in_dim * ffn_expand
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, in_dim),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(in_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.in_dim)
        x = self.norm1(x + self.out(attn_output))

        x = self.norm2(x + self.ffn(x))
        return x

class CustomMultiInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 cnn_type: str = "efficient", use_attention: bool = False):
        super().__init__(observation_space, features_dim)
        self.use_attention = use_attention

        if not use_attention:
            # ── Mode 1 ──
            cnn_classes = {
                "custom":       CustomCNN,
                "efficient":    EfficientCNN,
                "efficientnet": EfficientNetCNN,
            }
            selected_cnn = cnn_classes.get(cnn_type, CustomCNN)
            extractors = {}
            total_concat_size = 0

            if isinstance(observation_space, gym.spaces.Dict):
                for key, subspace in observation_space.spaces.items():
                    if key in ("front_rgb_camera"):
                        extractors[key] = selected_cnn((subspace.shape[2], subspace.shape[0], subspace.shape[1]) if len(subspace.shape) == 3 and subspace.shape[2] in (1, 3, 4) else subspace.shape, features_dim=features_dim)
                        total_concat_size += features_dim
                    else:
                        extractors[key] = nn.Flatten()
                        total_concat_size += get_flattened_obs_dim(subspace)
            else:
                raise RuntimeError(
                    f"CustomMultiInputExtractor (mode 1): no matching gym-dict found. "
                )
            self.extractors = nn.ModuleDict(extractors)
            self._features_dim = total_concat_size

        else:
            # ── Mode 2──
            self.modality_keys = []
            token_dim = max(256, ((features_dim + 3) // 4) * 4)
            self.token_dim = token_dim
            feature_extractors = {}
            token_projections  = {}

            if isinstance(observation_space, gym.spaces.Dict):
                for key, subspace in observation_space.spaces.items():
                    if key in ("front_rgb_camera"):
                        feature_extractors[key] = SpatialAttentionCNN((subspace.shape[2], subspace.shape[0], subspace.shape[1]) if len(subspace.shape) == 3 and subspace.shape[2] in (1, 3, 4) else subspace.shape, features_dim=features_dim)
                        raw_dim = features_dim
                    else:
                        flattened_size = get_flattened_obs_dim(subspace)
                        feature_extractors[key] = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(flattened_size, flattened_size),
                            nn.ReLU(inplace=True),
                        )
                        raw_dim = flattened_size
                    token_projections[key] = nn.Sequential(
                        nn.Linear(raw_dim, token_dim),
                        nn.ReLU(inplace=True),
                    )
                    self.modality_keys.append(key)
            else:
                raise RuntimeError(
                    f"CustomMultiInputExtractor (mode 2): no matching gym-dict found. "
                )

            self.feature_extractors = nn.ModuleDict(feature_extractors)
            self.token_projections  = nn.ModuleDict(token_projections)
            num_modalities = len(self.modality_keys)
            self.modality_embedding = nn.Embedding(num_modalities, token_dim)
            self.self_attention = SelfAttention(in_dim=token_dim, num_heads=4)
            self.output_projection = nn.Sequential(
                nn.Linear(token_dim, features_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
            self._features_dim = features_dim

    def forward(self, observations) -> torch.Tensor:
        if not self.use_attention:
            # ── Mode 1 ──
            encoded_tensor_list = []
            if hasattr(observations, "keys"):
                for key, extractor in self.extractors.items():
                    if key in observations:
                        encoded_tensor_list.append(extractor(observations[key]))
            else:
                raise RuntimeError(
                    f"CustomMultiInputExtractor (mode 1): no matching keys found. "
                    )
            return torch.cat(encoded_tensor_list, dim=1)
        else:
            # ── Mode 2 ──
            token_list = []
            if hasattr(observations, "keys"):
                for key in self.modality_keys:
                    if key in observations:
                        feat  = self.feature_extractors[key](observations[key])
                        token = self.token_projections[key](feat)
                        token_list.append(token.unsqueeze(1))
            else:
                 raise RuntimeError(
                    f"CustomMultiInputExtractor (mode 2): token_list is empty. "
                    )
            tokens   = torch.cat(token_list, dim=1)          # [B, N, D]
            mod_ids  = torch.arange(tokens.size(1), device=tokens.device)  # [N] [0, 1, 2, ...]
            tokens   = tokens + self.modality_embedding(mod_ids).unsqueeze(0)  # [B, N, D]
            attended = self.self_attention(tokens)
            pooled   = attended.mean(dim=1)
            return self.output_projection(pooled)


algorithm_params = {
    "VLM-SAC": dict(
        device="cuda:0",
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=40000,
        batch_size=64,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=5,
        gradient_steps=1,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[512, 256], qf=[512, 256]),
            features_extractor_class=CustomMultiInputExtractor,
            features_extractor_kwargs=dict(features_dim=256, 
                                                            cnn_type="custom",
                                                            use_attention=False),
        )
    ),

    "VLM-DDPG": dict(
        device="cuda:0",
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=40000,
        batch_size=64,
        gamma=0.98,
        tau=0.02,
        train_freq=5,
        gradient_steps=1,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.3 * np.ones(2)),
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[512, 256], qf=[512, 256]),
            features_extractor_class=CustomMultiInputExtractor,
            features_extractor_kwargs=dict(features_dim=256, 
                                                            cnn_type="custom",
                                                            use_attention=False),
        )
    ),

    "VLM-TD3": dict(
        device="cuda:0",
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=40000,
        batch_size=64,
        gamma=0.98,
        tau=0.02,
        train_freq=5,
        gradient_steps=1,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2)),
        policy_delay=2,
        target_policy_noise=0.1,
        target_noise_clip=0.3,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[512, 256], qf=[512, 256]),
            features_extractor_class=CustomMultiInputExtractor,
            features_extractor_kwargs=dict(features_dim=256, 
                                                            cnn_type="custom",
                                                            use_attention=False),
        )
    ),
}

state = ["steer", "throttle", "speed", "waypoints", "front_rgb_camera", "traffic_light_info"]

reward_params = {
    "reward_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    )
}

_CONFIG_VLM_SAC = {
    "algorithm": "VLM-SAC",
    "algorithm_params": algorithm_params["VLM-SAC"],
    "state": state,
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
}

_CONFIG_VLM_DDPG = {
    "algorithm": "VLM-DDPG",
    "algorithm_params": algorithm_params["VLM-DDPG"],
    "state": state,
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
}

_CONFIG_VLM_TD3 = {
    "algorithm": "VLM-TD3",
    "algorithm_params": algorithm_params["VLM-TD3"],
    "state": state,
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
}

CONFIGS = {
    "VLM-SAC": _CONFIG_VLM_SAC,
    "VLM-DDPG": _CONFIG_VLM_DDPG,
    "VLM-TD3": _CONFIG_VLM_TD3,
}

CONFIG = None

def set_config(config_name):
    global CONFIG
    CONFIG = Box(CONFIGS[config_name], default_box=True)
    return CONFIG

RGB_CAMERA_WIDTH = 400
RGB_CAMERA_HEIGHT = 400
RGB_CAMERA_RESOLUTION = (RGB_CAMERA_WIDTH, RGB_CAMERA_HEIGHT)
