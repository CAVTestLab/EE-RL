import os
import re
import json
import time
import base64
import argparse
from dataclasses import dataclass
from typing import List, Dict

from openai import OpenAI

SPARSE_KEY_SCENE_SYSTEM_PROMPT = """You are a sparse critical scene evaluator for autonomous driving.
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
"""

SPARSE_KEY_SCENE_USER_PROMPT_TEMPLATE = """
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
"""


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    openai_compatible: bool = True


# Common provider presets. Override with CLI args when needed.
PROVIDER_PRESETS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        openai_compatible=True,
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        openai_compatible=True,
    ),
    "qwen": ProviderConfig(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        openai_compatible=True,
    ),
    # Non-compatible providers can be added here with openai_compatible=False.
    "custom": ProviderConfig(
        name="custom",
        base_url="",
        api_key_env="LLM_API_KEY",
        openai_compatible=False,
    ),
}


def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def has_valid_reward_tag(text: str) -> bool:
    match = re.search(r"Final reward value\s*\[\s*(-?\d+)\s*\]", text)
    if not match:
        return False
    return int(match.group(1)) in {-2, -1, 0, 1, 2}


def build_user_prompt(state: dict) -> str:
    speed = float(state.get("vehicle_speed_kmh", 0.0))
    throttle = float(state.get("throttle", 0.0))
    steer = float(state.get("steer", 0.0))
    maneuver = state.get("anticipated_maneuver", "Unknown")
    tl_state = state.get("traffic_light_state", "Unknown")

    return SPARSE_KEY_SCENE_USER_PROMPT_TEMPLATE.format(
        speed=speed,
        throttle=throttle,
        steer=steer,
        current_maneuver=maneuver,
        tl_state=tl_state,
    ).strip()


def call_openai_label(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            },
        ],
    )

    return response.output_text.strip()


def call_custom_provider_label(
    provider_name: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    """
    Reserved hook for non-OpenAI-compatible providers.
    Implement HTTP request payload/response parsing for your target provider here.
    """
    raise NotImplementedError(
        f"Provider '{provider_name}' is marked non-compatible. "
        f"Implement call_custom_provider_label() for endpoint: {base_url}"
    )


def list_state_files(vehicle_states_dir: str) -> List[str]:
    files = []
    for fname in sorted(os.listdir(vehicle_states_dir)):
        if fname.endswith(".json"):
            files.append(os.path.join(vehicle_states_dir, fname))
    return files


def resolve_provider_config(args) -> ProviderConfig:
    preset = PROVIDER_PRESETS.get(args.provider)
    if preset is None:
        raise ValueError(
            f"Unsupported provider '{args.provider}'. "
            f"Supported: {', '.join(PROVIDER_PRESETS.keys())}"
        )

    base_url = args.base_url if args.base_url else preset.base_url
    api_key_env = args.api_key_env if args.api_key_env else preset.api_key_env

    return ProviderConfig(
        name=preset.name,
        base_url=base_url,
        api_key_env=api_key_env,
        openai_compatible=preset.openai_compatible,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate sparse key-scene JSON via OpenAI API")
    parser.add_argument("--provider", default="deepseek", choices=list(PROVIDER_PRESETS.keys()))
    parser.add_argument("--base-url", default="", help="Override provider API base URL")
    parser.add_argument("--api-key-env", default="", help="Env var name storing API key")
    parser.add_argument("--vehicle-states-dir", default="./vlm_lora_data/key_road_data/vehicle_states")
    parser.add_argument("--images-dir", default="./vlm_lora_data/key_road_data/images")
    parser.add_argument("--output-path", default="./vlm_lora_data/key_road_sparse_reward_api.json")
    parser.add_argument("--model", default="", help="Model id required by selected provider")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    args = parser.parse_args()

    provider_cfg = resolve_provider_config(args)

    if not os.path.isdir(args.vehicle_states_dir):
        raise FileNotFoundError(f"vehicle states dir not found: {args.vehicle_states_dir}")
    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"images dir not found: {args.images_dir}")

    if not args.model:
        raise ValueError("--model is required, e.g. deepseek-chat / qwen-vl-max / moonshot-v1-8k")

    api_key = os.getenv(provider_cfg.api_key_env)
    if not api_key:
        raise EnvironmentError(f"API key not found. Please set env var: {provider_cfg.api_key_env}")

    if provider_cfg.openai_compatible and not provider_cfg.base_url:
        raise ValueError("base_url is empty for openai-compatible provider")

    client = None
    if provider_cfg.openai_compatible:
        client = OpenAI(api_key=api_key, base_url=provider_cfg.base_url)

    print(
        f"Provider={provider_cfg.name}, BaseURL={provider_cfg.base_url}, "
        f"API_KEY_ENV={provider_cfg.api_key_env}, Model={args.model}"
    )

    state_files = list_state_files(args.vehicle_states_dir)
    if args.max_samples > 0:
        state_files = state_files[: args.max_samples]

    samples = []
    total = len(state_files)

    for i, fpath in enumerate(state_files, start=1):
        with open(fpath, "r", encoding="utf-8") as f:
            state = json.load(f)

        sample_id = state.get("sample_id", os.path.basename(fpath).split(".")[0])
        image_path_abs = os.path.join(args.images_dir, f"{sample_id}.jpg")
        image_path_rel = image_path_abs.replace("\\", "/")

        if not os.path.exists(image_path_abs):
            print(f"[{i}/{total}] skip missing image: {image_path_abs}")
            continue

        user_prompt = build_user_prompt(state)
        image_data_url = encode_image_to_data_url(image_path_abs)

        try:
            if provider_cfg.openai_compatible:
                output_str = call_openai_label(
                    client=client,
                    model=args.model,
                    system_prompt=SPARSE_KEY_SCENE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    image_data_url=image_data_url,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
            else:
                output_str = call_custom_provider_label(
                    provider_name=provider_cfg.name,
                    base_url=provider_cfg.base_url,
                    api_key=api_key,
                    model=args.model,
                    system_prompt=SPARSE_KEY_SCENE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    image_data_url=image_data_url,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
        except Exception as e:
            print(f"[{i}/{total}] API call failed for sample {sample_id}: {e}")
            continue

        if not has_valid_reward_tag(output_str):
            output_str = (
                output_str
                + "\n\n"
                + "Final reward value [0]"
            )

        sample = {
            "instruction": SPARSE_KEY_SCENE_SYSTEM_PROMPT,
            "input": user_prompt,
            "output": output_str,
            "images": [image_path_rel],
        }
        samples.append(sample)

        print(f"[{i}/{total}] generated sample {sample_id}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(samples)} samples -> {args.output_path}")


if __name__ == "__main__":
    main()
