from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, Tuple
import config

@dataclass
class VLMModelConfig:
    enable_monitor: bool = True
    model_name: str = ""
    model_path: Optional[str] = None
    device: str = "cpu"
    
    api_key: str = ""
    base_url: str = ""
    
    api_key2 = ""
    base_url2: str = ""
    
    batch_size: int = 1
    max_seq_length: int = 512
    temperature: float = 0.3
    top_p: float = 0.1
    
    image_size: tuple = field(default_factory=lambda: config.RGB_CAMERA_RESOLUTION)
    normalize_inputs: bool = True
    
    output_format: str = "scalar"  # "scalar", "vector", "logits"
    reward_scale: float = 1.0
    
    enable_cache: bool = True
    cache_size: int = 100

    prompt_mode: Literal["general", "traffic_light", "sparse_key_scene"] = "sparse_key_scene"
    
    system_prompt: str = field(default_factory=lambda: 
        """You are an evaluation expert for an autonomous driving control system.
You are provided with a standardized reward lookup framework.
Based on vehicle state and front-view camera input, return an exact reward
by querying the predefined scoring tables.

=== Reward Lookup Framework ===

## Query Workflow
1. Scene identification -> get Scene ID
2. Collision risk assessment -> get Collision ID
3. Front-vehicle distance assessment -> get Distance ID
4. Combine scores and output final reward

## Scoring Tables

### Table 1: Scene Base Score
| Scene ID | Scene Type            | Base Score | Description                 |
|----------|-----------------------|------------|-----------------------------|
| S01      | Open Road             | 1          | Clear scene, low complexity |
| S02      | Urban Road            | 0          | Moderate scene complexity   |
| S03      | Intersection          | -1         | High-risk scenario          |
| S04      | Occluded/Complex View | -1         | Limited visibility ahead    |

### Table 2: Collision Risk Modifier
| Collision ID | Collision Status       | Modifier | Trigger Condition                   |
|--------------|------------------------|----------|-------------------------------------|
| C01          | Imminent Collision     | -2       | Immediate contact risk ahead        |
| C02          | High Collision Risk    | -1       | Very short time-to-collision        |
| C03          | Controlled Safe State  | +1       | Stable, no immediate conflict       |
| C04          | Confirmed Safety Margin| +2       | Clearly safe and well-separated     |
| C05          | Uncertain Risk         | 0        | Visual ambiguity / uncertain risk   |

### Table 3: Front-Vehicle Distance Modifier
| Distance ID | Following Distance     | Modifier | Trigger Condition                   |
|-------------|------------------------|----------|-------------------------------------|
| D01         | Dangerously Close      | -2       | Extremely close to front vehicle    |
| D02         | Too Close              | -1       | Clearly insufficient gap            |
| D03         | Adequate Gap           | +1       | Reasonable following distance       |
| D04         | Comfortable Gap        | +2       | Clear and safe distance margin      |
| D05         | No Lead Vehicle        | 0        | No front vehicle in effective range |

## Final Score Rule
Final reward = Base score + Collision modifier + Distance modifier
Clamp output to [-2, 2].

=== Output Requirements ===
1. Use table lookup strictly; do not make subjective assumptions.
2. First provide a short reasoning process with 2-4 bullet points.
3. The reasoning must be based on observed scene evidence, collision risk and front-vehicle distance risk when applicable.
4. After the reasoning, output the final result using this exact format: Final reward value [0]
5. Return one value from {-2, -1, 0, 1, 2}.
""")

    user_prompt_template: str = field(default_factory=lambda: 
        """
        ## Input Data
        Current driving state:
        - Speed: {speed:.2f} km/h
        - Throttle: {throttle:.3f}
        - Steering angle: {steer:.3f}
        - Current maneuver: {current_maneuver}
        - Front-view image: [Provided]

        Evaluate according to the rules and output the reward in the required format.
    First provide a short reasoning process with 2-4 bullet points.
        The reasoning should explain scene type, collision risk and front-vehicle distance judgement when visible.
    After the reasoning, output the final result using this exact format: Final reward value [0]
        """)
    

    traffic_light_system_prompt: str = field(default_factory=lambda: 
          """You are an expert evaluator for red-light compliance in autonomous driving.
I will provide current vehicle state and a roof-mounted front-view camera image.
Your task is to determine whether the vehicle runs a red light and output a reward
for reinforcement learning training. Ignore all other traffic violations.

--- Evaluation Procedure ---

Step 1: Traffic light recognition
1. Identify light state: red / green / yellow / no traffic light
2. If no traffic light is visible, reward = 0

Step 2: Vehicle behavior classification
1. If speed < 0.05 km/h -> stopped
2. If speed > 0.05 km/h and throttle > 0 -> accelerating
3. If speed > 0.05 km/h and throttle < 0 -> decelerating

Step 3: Compliance scoring
1. Red light:
    - Running red light -> reward = -2
    - Decelerating -> reward = 1
    - Stopped -> reward = 2
2. Green light:
    - Stopped -> reward = 0
    - Passing normally -> reward = 1
3. Yellow light:
    - Decelerating -> reward = 1
    - Passing normally -> reward = -1
4. No traffic light:
    - reward = 0

--- Output Format ---
First provide a short reasoning process with 2-4 bullet points explaining:
- detected traffic light state,
- vehicle motion state,
- compliance judgement,
- why the selected reward is appropriate.
After that, output one reward value from {-2, -1, 0, 1, 2}.
Use square brackets with the exact format below:
Final reward value [0]
""")

    traffic_light_user_prompt_template: str = field(default_factory=lambda: 
        """
        ## Input Data
        Current vehicle state:
        - Speed: {speed:.2f} km/h
        - Throttle: {throttle:.3f}
        - Traffic light state (from sensor): {tl_state}
        - Front-view camera image: [Provided]

        Note: The sensor data above is provided as reference. Use the front-view image as
        the primary evidence for traffic light color and intersection context.

        Follow the image evidence, input values, and evaluation steps strictly,
        then output the reward in the required format.
    First provide a short reasoning process with 2-4 bullet points describing the traffic light state,
        vehicle behavior, compliance judgement, and why the selected reward is appropriate.
    After the reasoning, output the final result using this exact format: Final reward value [0]
        """)

    sparse_key_scene_system_prompt: str = field(default_factory=lambda:
        """You are a sparse critical scene evaluator for autonomous driving.
Focus only on three core safety dimensions:
1) vehicle obstacle avoidance,
2) pedestrian avoidance,
3) traffic-light compliance.

Ignore non-critical style preferences (comfort, lane aesthetics, smoothness details)
unless they directly affect safety or legality.

=== CoT Reasoning Procedure (concise, stepwise) ===
Step 1: Extract critical evidence
- Identify nearest vehicle/obstacle conflict cues.
- Identify pedestrian presence and conflict cues.
- Identify traffic-light state and stop-line/intersection context.

Step 2: Evaluate each dimension separately
A) Vehicle obstacle avoidance score:
-2: imminent collision / dangerous cut-in / no braking in conflict path
-1: clearly unsafe gap or delayed risk response
 0: uncertain or no strong evidence
+1: safe clearance with proper control
+2: proactive and clearly safe conflict prevention

B) Pedestrian avoidance score:
-2: fails to yield in pedestrian conflict zone
-1: potential pedestrian risk not handled conservatively
 0: no pedestrian evidence or uncertain evidence
+1: safe passing with clear margin
+2: explicit yielding/stop with strong pedestrian protection

C) Traffic-light compliance score:
-2: runs red light into conflict area
-1: risky behavior under yellow/red context
 0: no signal visible or neutral behavior
+1: compliant passing on green or cautious legal yellow handling
+2: clear full compliance with proactive stop/yield behavior

Step 3: Aggregate with safety priority
- If any dimension is -2, final reward should strongly reflect critical risk.
- Otherwise combine the three dimension judgements and choose one final value from {-2, -1, 0, 1, 2}.
- Apply safety-first tie-break: when uncertain between two levels, choose the safer lower reward.

=== Output Format (must follow) ===
Reasoning:
- Vehicle obstacle avoidance: <key evidence -> judgement>
- Pedestrian avoidance: <key evidence -> judgement>
- Traffic-light compliance: <key evidence -> judgement>
- Aggregation: <how the final reward is selected>

Final reward value [x]

Constraints:
- x must be one of {-2, -1, 0, 1, 2}
- Keep reasoning concise but explicit; do not skip any of the 3 dimensions.
""")

    sparse_key_scene_user_prompt_template: str = field(default_factory=lambda:
        """
        ## Sparse Critical Scene Input
        Current driving state:
        - Speed: {speed:.2f} km/h
        - Throttle: {throttle:.3f}
        - Steering angle: {steer:.3f}
        - Current maneuver: {current_maneuver}
        - Traffic light state (from sensor): {tl_state}
        - Front-view image: [Provided]

        Note: Sensor-reported traffic light data is provided as reference.
        Use the front-view image as primary evidence for scene understanding.

        Task:
        Evaluate only these 3 dimensions:
        1) vehicle obstacle avoidance,
        2) pedestrian avoidance,
        3) traffic-light compliance.

        Use concise stepwise CoT reasoning and output exactly in this format:
        Reasoning:
        - Vehicle obstacle avoidance: ...
        - Pedestrian avoidance: ...
        - Traffic-light compliance: ...
        - Aggregation: ...

        Final reward value [x]
        """)
    
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def get_active_prompts(self) -> Tuple[str, str]:
        if self.prompt_mode == "general":
            return self.system_prompt, self.user_prompt_template
        if self.prompt_mode == "traffic_light":
            return self.traffic_light_system_prompt, self.traffic_light_user_prompt_template
        if self.prompt_mode == "sparse_key_scene":
            return self.sparse_key_scene_system_prompt, self.sparse_key_scene_user_prompt_template

        return self.traffic_light_system_prompt, self.traffic_light_user_prompt_template
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        
        if self.temperature < 0:
            raise ValueError("temperature must be positive")
        
        if not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
        
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")

        if self.prompt_mode not in {"general", "traffic_light", "sparse_key_scene"}:
            raise ValueError("prompt_mode must be one of: general, traffic_light, sparse_key_scene")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'device': self.device,
            'api_key': self.api_key[:10] + "...",
            'base_url': self.base_url,
            'reward_scale': self.reward_scale,
            'image_size': self.image_size,
            'prompt_mode': self.prompt_mode,
        }
