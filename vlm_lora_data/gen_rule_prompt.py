import os
import json

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

vehicle_states_dir = "./vlm_lora_data/key_road_data/vehicle_states"
images_dir = "./vlm_lora_data/key_road_data/images"
output_path = "./vlm_lora_data/key_road_sparse_reward.json"


def _eval_vehicle_obstacle(state):
    speed = float(state.get("vehicle_speed_kmh", 0.0))
    throttle = float(state.get("throttle", 0.0))
    brake = float(state.get("brake", 0.0))
    leading = state.get("leading_vehicle", {}) or {}

    has_lead = bool(leading.get("has_leading_vehicle", False))
    dist = leading.get("leading_vehicle_distance")
    rel_speed = leading.get("leading_vehicle_relative_speed")

    if not has_lead or dist is None:
        return 0, "no clear leading-vehicle conflict evidence -> neutral"

    dist = float(dist)
    rel_speed = float(rel_speed or 0.0)

    if dist < 6.0 and speed > 10.0 and rel_speed > 2.0 and brake < 0.1:
        return -2, "lead vehicle at very close range, fast closing trend, weak braking -> critical risk"
    if dist < 10.0 and speed > 8.0 and rel_speed > 1.0 and brake < 0.1:
        return -1, "unsafe following gap with delayed response -> risky"
    if dist < 12.0 and (brake > 0.25 or throttle < 0.0):
        return 2, "close leading vehicle handled with explicit slowing/braking -> proactive prevention"
    if dist < 20.0 and (brake > 0.1 or rel_speed <= 0.0):
        return 1, "maintains safe clearance with appropriate control response -> safe"

    return 0, "leading vehicle observed but conflict evidence not decisive -> neutral"


def _eval_pedestrian(state):
    speed = float(state.get("vehicle_speed_kmh", 0.0))
    brake = float(state.get("brake", 0.0))
    ped = state.get("pedestrian", {}) or {}

    has_ped = bool(ped.get("has_pedestrian_in_range", False))
    dist = ped.get("nearest_pedestrian_distance")

    if not has_ped or dist is None:
        return 0, "no clear pedestrian conflict evidence -> neutral"

    dist = float(dist)

    if dist < 8.0 and speed > 8.0 and brake < 0.2:
        return -2, "pedestrian at very close range and vehicle not yielding -> critical risk"
    if dist < 12.0 and speed > 15.0 and brake < 0.2:
        return -1, "pedestrian in potential conflict zone with weak caution -> risky"
    if dist < 8.0 and speed < 2.0 and brake > 0.3:
        return 2, "pedestrian close and vehicle clearly yielding/stopping -> strong protection"
    if dist < 12.0 and speed < 8.0:
        return 1, "pedestrian present with conservative passing speed -> safe"

    return 0, "pedestrian observed but evidence not decisive -> neutral"


def _eval_traffic_light(state):
    speed = float(state.get("vehicle_speed_kmh", 0.0))
    throttle = float(state.get("throttle", 0.0))
    brake = float(state.get("brake", 0.0))
    light = str(state.get("traffic_light_state", "Unknown"))

    tl_dist = state.get("traffic_light_distance")
    if tl_dist is None:
        tl_dist = state.get("reference_distance")
    if tl_dist is None:
        tl_dist = 999.0
    tl_dist = float(tl_dist)

    if light in {"Unknown", "Off", "No Traffic Light", "", "None"}:
        return 0, "no reliable traffic-light evidence -> neutral"

    if light == "Red":
        if tl_dist < 25.0 and speed > 6.0 and throttle > 0.0 and brake < 0.1:
            return -2, "red light context with continued passing tendency -> red-light violation risk"
        if tl_dist < 25.0 and speed > 2.0 and brake < 0.1:
            return -1, "red light context with insufficient deceleration -> risky"
        if tl_dist < 25.0 and speed < 0.5 and brake > 0.2:
            return 2, "red light context with clear stop/yield behavior -> strong compliance"
        if tl_dist < 25.0 and brake > 0.05:
            return 1, "red light context with cautious slowing -> compliant trend"
        return 0, "red light detected but distance/context uncertain -> neutral"

    if light == "Yellow":
        if tl_dist < 20.0 and speed > 20.0 and throttle > 0.0 and brake < 0.05:
            return -1, "yellow light context with aggressive passing tendency -> risky"
        if tl_dist < 20.0 and (brake > 0.1 or speed < 10.0):
            return 1, "yellow light context handled cautiously -> compliant"
        return 0, "yellow context but evidence not decisive -> neutral"

    if light == "Green":
        if tl_dist < 25.0 and speed > 1.0:
            return 1, "green light context with legal passing -> compliant"
        if tl_dist < 25.0 and speed < 0.5 and brake > 0.2:
            return 0, "green light context but vehicle stopped; legality uncertain without extra context -> neutral"
        return 0, "green signal seen but context weak -> neutral"

    return 0, f"unrecognized light state '{light}' -> neutral"


def _aggregate_score(scores):
    values = [scores["vehicle"], scores["pedestrian"], scores["traffic_light"]]
    if -2 in values:
        return -2, "at least one dimension is critical (-2), apply safety-priority override"

    total = sum(values)
    if total >= 4:
        return 2, f"combined score={total}, consistently safe across dimensions"
    if total >= 1:
        return 1, f"combined score={total}, overall safe with limited uncertainty"
    if total <= -2:
        return -1, f"combined score={total}, risk indicators dominate"
    return 0, f"combined score={total}, mixed/uncertain evidence -> neutral"


def build_output_label(state):
    veh_score, veh_reason = _eval_vehicle_obstacle(state)
    ped_score, ped_reason = _eval_pedestrian(state)
    tl_score, tl_reason = _eval_traffic_light(state)

    final_reward, agg_reason = _aggregate_score(
        {
            "vehicle": veh_score,
            "pedestrian": ped_score,
            "traffic_light": tl_score,
        }
    )

    output = (
        "Reasoning:\n"
        f"- Vehicle obstacle avoidance: {veh_reason}; score={veh_score}\n"
        f"- Pedestrian avoidance: {ped_reason}; score={ped_score}\n"
        f"- Traffic-light compliance: {tl_reason}; score={tl_score}\n"
        f"- Aggregation: {agg_reason}\n\n"
        f"Final reward value [{final_reward}]"
    )

    return final_reward, output

def main():
    samples = []
    if not os.path.isdir(vehicle_states_dir):
        raise FileNotFoundError(f"vehicle_states_dir not found: {vehicle_states_dir}")

    for fname in sorted(os.listdir(vehicle_states_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(vehicle_states_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            state = json.load(f)

        speed = float(state.get("vehicle_speed_kmh", 0.0))
        throttle = float(state.get("throttle", 0.0))
        steer = float(state.get("steer", 0.0))
        maneuver = state.get("anticipated_maneuver", "Unknown")
        tl_state = state.get("traffic_light_state", "Unknown")
        sample_id = state.get("sample_id", fname.split(".")[0])
        image_path = os.path.join(images_dir, f"{sample_id}.jpg").replace("\\", "/")

        input_str = SPARSE_KEY_SCENE_USER_PROMPT_TEMPLATE.format(
            speed=speed,
            throttle=throttle,
            steer=steer,
            current_maneuver=maneuver,
            tl_state=tl_state,
        ).strip()

        _, output_str = build_output_label(state)

        sample = {
            "instruction": SPARSE_KEY_SCENE_SYSTEM_PROMPT,
            "input": input_str,
            "output": output_str,
            "images": [image_path]
        }
        samples.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(samples)} samples -> {output_path}")

if __name__ == "__main__":
    main()
