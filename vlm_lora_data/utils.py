import numpy as np
import carla
import cv2
import json
import os
from datetime import datetime

def vector(v):
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])

def calculate_distance_2d(pos1, pos2):
    return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

def calculate_speed_kmh(velocity):
    return 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

def get_maneuver_from_steer(steer, threshold=0.1):
    if abs(steer) < threshold:
        return "STRAIGHT"
    elif steer < -threshold:
        return "LEFT"
    elif steer > threshold:
        return "RIGHT"
    else:
        return "UNKNOWN"

def save_image_with_metadata(image_array, filepath, metadata=None):
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(filepath, image_bgr)
    
    if success and metadata:
        metadata_path = filepath.replace('.jpg', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    return success

def create_output_directories(base_dir):
    directories = {
        'base': base_dir,
        'images': os.path.join(base_dir, 'images'),
        'vehicle_states': os.path.join(base_dir, 'vehicle_states'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def log_collection_event(log_file, event_type, message, additional_data=None):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "message": message
    }
    
    if additional_data:
        log_entry.update(additional_data)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')

def validate_carla_connection(client):
    try:
        client.get_server_version()
        client.get_world()
        return True
    except Exception as e:
        print(f"CARLA connection validation failed: {e}")
        return False

def filter_traffic_lights_by_distance(traffic_lights, vehicle_location, max_distance):
    filtered_lights = []
    
    for light in traffic_lights:
        distance = vehicle_location.distance(light.get_location())
        if distance <= max_distance:
            filtered_lights.append((light, distance))
    
    filtered_lights.sort(key=lambda x: x[1])
    return filtered_lights

def get_traffic_light_state_info(traffic_light):
    state_map = {
        carla.TrafficLightState.Red: "RED",
        carla.TrafficLightState.Yellow: "YELLOW", 
        carla.TrafficLightState.Green: "GREEN",
        carla.TrafficLightState.Off: "OFF",
        carla.TrafficLightState.Unknown: "UNKNOWN"
    }
    
    state = traffic_light.get_state()
    return {
        "state": state_map.get(state, "UNKNOWN"),
        "green_time": traffic_light.get_green_time(),
        "yellow_time": traffic_light.get_yellow_time(),
        "red_time": traffic_light.get_red_time()
    }
