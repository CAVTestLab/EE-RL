import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import re

import carla
import numpy as np
import cv2
import json
import time
import weakref
from datetime import datetime
from collections import deque
import argparse
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, K_ESCAPE

from carla_env.wrappers import *
from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints


class TrafficLightCollector:
    def __init__(self, host="127.0.0.1", port=2000, town='Town02'):
        self.host = host
        self.port = port
        self.town = town
        
        self.image_width = 600
        self.image_height = 600
        self.detection_distance = 10.0
        self.anticipation_distance = 15.0
        
        self.leading_vehicle_detection_distance = 30.0
        self.pedestrian_detection_distance = 30.0
        
        self.processed_traffic_lights = set()
        self.current_collecting_light = None
        self.entity_sample_limit = 20
        self.traffic_light_sample_counts = {}
        self.leading_vehicle_sample_counts = {}
        self.pedestrian_sample_counts = {}
        
        # Output directories
        self.output_dir = "vlm_lora_data/key_road_data"
        self.image_dir = os.path.join(self.output_dir, "images")
        self.json_dir = os.path.join(self.output_dir, "vehicle_states")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        self.sample_count = self._get_max_sample_count()

        # CARLA environment
        self.client = None
        self.world = None
        self.vehicle = None
        self.front_camera = None
        self.front_image = None
        self.use_dynamic_traffic_lights = True
        self.enable_pedestrian_ai = False

        # Traffic manager
        self.traffic_manager = None
        self.tm_port = port + 5000
        self.autopilot_enabled = False
        
        # Background participants
        self.traffic_flow_vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []
        self.tf_num = 50
        self.pedestrian_num = 20
        
        # Rendering
        self.viewer_res = (1120, 560)
        self.activate_render = True
        self.display = None
        self.clock = None
        self.hud = None
        self.camera = None
        self.viewer_image = None
        
        # Navigation
        self.route_waypoints = []
        self.current_waypoint_index = 0
        self.current_waypoint = None
        self.next_waypoint = None
        self.current_road_maneuver = RoadOption.LANEFOLLOW
        self.next_road_maneuver = RoadOption.LANEFOLLOW
        self.anticipated_maneuver = RoadOption.LANEFOLLOW
        
        # Vehicle state
        self.vehicle_speed = 0.0
        self.vehicle_throttle = 0.0
        self.vehicle_steer = 0.0
        self.current_maneuver = "Unknown"
   
    def _get_max_sample_count(self):
        """Get maximum sample index from existing data"""
        max_idx = 0
        if os.path.exists(self.image_dir):
            for fname in os.listdir(self.image_dir):
                match = re.match(r"(\d+)\.jpg", fname)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
        return max_idx
    
    def configure_dynamic_traffic_lights(self):
        """Configure traffic lights to run in dynamic cycle"""
        if not self.use_dynamic_traffic_lights:
            return

        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for light in traffic_lights:
            try:
                light.freeze(False)
                light.set_green_time(12.0)
                light.set_yellow_time(3.0)
                light.set_red_time(10.0)
            except Exception:
                continue

    def initialize_carla(self):
        """Initialize CARLA environment and connect to server"""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            
            self.world = World(self.client, town=self.town)
            
            # Set synchronous mode (20 FPS)
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
            self.world.set_weather(
                carla.WeatherParameters(
                    cloudiness=100.0, 
                    precipitation=0.0, 
                    sun_altitude_angle=45.0
                )
            )
            
            self._setup_traffic_manager()
            self._create_initial_route()
            self._spawn_vehicle()
            self._setup_rendering()
            self._setup_sensors()
            self._spawn_background_vehicles()
            self._spawn_pedestrians()
            self._enable_autopilot()
            self.configure_dynamic_traffic_lights()
            return True
            
        except Exception as e:
            print(f"CARLA initialization failed: {e}")
            return False
    
    def _setup_traffic_manager(self):
        """Setup traffic manager with synchronous mode"""
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(50.0)
        
        self.traffic_manager.global_percentage_speed_difference(0.0)
        self.traffic_manager.set_boundaries_respawn_dormant_vehicles(25.0, 700.0)
    
    def _spawn_background_vehicles(self):
        """Spawn background traffic vehicles"""
        import random
        spawn_points = self.world.map.get_spawn_points()
        number_of_vehicles = min(len(spawn_points), self.tf_num)
        
        ego_location = self.vehicle.get_location()
        safe_spawn_points = []
        
        for spawn_point in spawn_points:
            distance_to_ego = spawn_point.location.distance(ego_location)
            if distance_to_ego > 30.0:
                safe_spawn_points.append(spawn_point)
        
        if not safe_spawn_points:
            return
        
        random.shuffle(safe_spawn_points)

        for i in range(min(number_of_vehicles, len(safe_spawn_points))):
            try:
                blueprint_name = 'vehicle.tesla.model3'
                vehicle_bp = self.world.get_blueprint_library().find(blueprint_name)
                if vehicle_bp.has_attribute('color'):
                    color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                    vehicle_bp.set_attribute('color', color)
                sp = safe_spawn_points[i]
                actor = self.world.try_spawn_actor(vehicle_bp, sp)
                if actor is not None:
                    actor.set_autopilot(True, self.traffic_manager.get_port())
                    self.traffic_flow_vehicles.append(actor)
            except Exception:
                continue
    
    def _spawn_pedestrians(self):
        """Spawn pedestrian actors"""
        import random

        pedestrian_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        for _ in range(self.pedestrian_num):
            try:
                pedestrian_bp = random.choice(pedestrian_blueprints)
                nav_location = self.world.get_random_location_from_navigation()
                if nav_location is None:
                    continue
                spawn_transform = carla.Transform(nav_location)

                pedestrian = self.world.try_spawn_actor(pedestrian_bp, spawn_transform)
                if pedestrian is None:
                    continue
                self.pedestrians.append(pedestrian)

                # Keep pedestrians static by default for runtime stability.
                if self.enable_pedestrian_ai:
                    controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), pedestrian)
                    if controller is None:
                        pedestrian.destroy()
                        self.pedestrians.pop()
                        continue

                    self.pedestrian_controllers.append(controller)
                    self.world.tick()
                    controller.start()
                    target_location = self.world.get_random_location_from_navigation()
                    if target_location is not None:
                        controller.go_to_location(target_location)
            except Exception:
                pass
    
    def _enable_autopilot(self):
        if self.vehicle and self.traffic_manager:
            self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.autopilot_enabled = True
            
            vehicle_actor = self.vehicle.get_carla_actor()
            
            self.traffic_manager.auto_lane_change(vehicle_actor, True)
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle_actor, 0.0)
            self.traffic_manager.distance_to_leading_vehicle(vehicle_actor, 3.0)
            self.traffic_manager.ignore_lights_percentage(vehicle_actor, 100.0)
            self.traffic_manager.ignore_signs_percentage(vehicle_actor, 0.0)
            self.traffic_manager.ignore_vehicles_percentage(vehicle_actor, 0.0)
            self.traffic_manager.ignore_walkers_percentage(vehicle_actor, 0.0)
            
            self.traffic_manager.set_desired_speed(vehicle_actor, 40.0)
    
    def _disable_autopilot(self):
        if self.vehicle and self.autopilot_enabled:
            self.vehicle.set_autopilot(False)
            self.autopilot_enabled = False
    
    def _set_route_for_autopilot(self):
        if self.route_waypoints and self.traffic_manager and self.autopilot_enabled:
            route_locations = []
            for waypoint, _ in self.route_waypoints[::5]:
                if hasattr(waypoint, "transform") and hasattr(waypoint.transform, "location"):
                    loc = waypoint.transform.location
                elif isinstance(waypoint, carla.Location):
                    loc = waypoint
                else:
                    continue
                route_locations.append(loc)
            route_locations = [loc for loc in route_locations if isinstance(loc, carla.Location)]
            if len(route_locations) >= 2:
                vehicle_actor = self.vehicle.get_carla_actor()
                self.traffic_manager.set_route(vehicle_actor, route_locations)
    
    def _spawn_vehicle(self):
        """Spawn ego vehicle at preferred location or random spawn point"""
        max_attempts = 100
        attempt = 0
        success = False
        spawn_points = self.world.map.get_spawn_points()
        tried_indices = set()
        
        if self.route_waypoints:
            spawn_point = self.route_waypoints[0][0].transform
            try:
                self.vehicle = Vehicle(
                    self.world, 
                    spawn_point,
                    on_collision_fn=lambda e: self._on_collision(e),
                    on_invasion_fn=lambda e: self._on_invasion(e),
                    is_ego=True
                )
                return
            except Exception:
                pass
        
        while attempt < max_attempts:
            idx = np.random.randint(0, len(spawn_points))
            if idx in tried_indices:
                attempt += 1
                continue
            tried_indices.add(idx)
            spawn_point = spawn_points[idx]
            try:
                self.vehicle = Vehicle(
                    self.world, 
                    spawn_point,
                    on_collision_fn=lambda e: self._on_collision(e),
                    on_invasion_fn=lambda e: self._on_invasion(e),
                    is_ego=True
                )
                success = True
                break
            except Exception:
                attempt += 1
        
        if not success:
            raise RuntimeError("Unable to spawn vehicle at available locations.")
    
    def _setup_rendering(self):
        if self.activate_render:
            pygame.init()
            pygame.font.init()
            width, height = self.viewer_res
            self.display = pygame.display.set_mode((width, height), HWSURFACE | DOUBLEBUF)
            pygame.display.set_caption("CARLA Traffic Light Data Collector")
            self.clock = pygame.time.Clock()
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)
    
    def _setup_sensors(self):
        """Setup front and spectator cameras"""
        width, height = self.viewer_res
        
        self.camera = Camera(
            self.world, 
            width, 
            height,
            transform=sensor_transforms["spectator"],
            attach_to=self.vehicle, 
            on_recv_image=lambda e: self._set_viewer_image(e)
        )
        
        self.front_camera = Camera(
            self.world,
            width=self.image_width,
            height=self.image_height,
            transform=sensor_transforms["front"],
            attach_to=self.vehicle,
            on_recv_image=lambda e: self._set_front_image(e),
            fov=110
        )
    
    def _create_initial_route(self):
        """Create initial route with random start and end points"""
        spawn_points = self.world.map.get_spawn_points()
        import random
        start_idx = random.randint(0, len(spawn_points) - 1)
        start_wp = self.world.map.get_waypoint(spawn_points[start_idx].location)

        for i in range(10, min(50, len(spawn_points))):
            end_wp = self.world.map.get_waypoint(spawn_points[i].location)
            route_waypoints = compute_route_waypoints(
                self.world.map, start_wp, end_wp, resolution=1.0
            )
            if len(route_waypoints) > 20:
                self.route_waypoints = route_waypoints
                break

        self.current_waypoint_index = 0

        if self.route_waypoints:
            self.current_waypoint, self.current_road_maneuver = self.route_waypoints[0]
            if len(self.route_waypoints) > 1:
                self.next_waypoint, self.next_road_maneuver = self.route_waypoints[1]

        self._set_route_for_autopilot()
    
    def _create_new_route(self):
        """Create new random route for continuous navigation"""
        spawn_points = self.world.map.get_spawn_points()
        import random
        start_idx = random.randint(0, len(spawn_points) - 1)
        end_idx = random.randint(0, len(spawn_points) - 1)
        while end_idx == start_idx:
            end_idx = random.randint(0, len(spawn_points) - 1)
        start_wp = self.world.map.get_waypoint(spawn_points[start_idx].location)
        end_wp = self.world.map.get_waypoint(spawn_points[end_idx].location)
        route_waypoints = compute_route_waypoints(
            self.world.map, start_wp, end_wp, resolution=1.0
        )
        if len(route_waypoints) < 20:
            return self._create_new_route()
        self.route_waypoints = route_waypoints
        self.current_waypoint_index = 0
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[0]
        if len(self.route_waypoints) > 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[1]
        self._set_route_for_autopilot()

    def _set_viewer_image(self, image):
        """Store spectator camera image"""
        self.viewer_image = image
    
    def _set_front_image(self, image):
        """Store front camera image"""
        self.front_image = image
    
    def _on_collision(self, event):
        """Collision event callback"""
        pass
    
    def _on_invasion(self, event):
        """Lane invasion event callback"""
        pass
    
    def get_nearest_traffic_light(self):
        """Get nearest traffic light and determine if it's collectible"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        vehicle_waypoint = self.world.map.get_waypoint(vehicle_location)
        
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        
        if not traffic_lights:
            return None, float('inf'), False
        
        nearest_light = None
        min_distance = float('inf')
        can_collect = False
        
        for light in traffic_lights:
            light_location = light.get_location()
            distance = vehicle_location.distance(light_location)
            
            if distance > self.detection_distance:
                continue
                
            direction_to_light = light_location - vehicle_location
            direction_to_light_normalized = direction_to_light / distance
            
            dot_product = (vehicle_forward.x * direction_to_light_normalized.x + 
                        vehicle_forward.y * direction_to_light_normalized.y)
            
            # Traffic light must be ahead (angle < 60 degrees)
            if dot_product < 0.5:
                continue
                
            if self._is_traffic_light_relevant(light, vehicle_waypoint, vehicle_forward):
                if distance < min_distance:
                    min_distance = distance
                    nearest_light = light
                    can_collect = True
        
        return nearest_light, min_distance, can_collect

    def _is_traffic_light_relevant(self, traffic_light, vehicle_waypoint, vehicle_forward):
        """Check if traffic light affects current vehicle lane"""
        try:
            affected_waypoints = traffic_light.get_affected_lane_waypoints()
            
            if not affected_waypoints:
                return False
            
            for waypoint in affected_waypoints:
                if (waypoint.lane_id == vehicle_waypoint.lane_id and 
                    waypoint.road_id == vehicle_waypoint.road_id):
                    return True
                    
                distance_to_affected_lane = vehicle_waypoint.transform.location.distance(
                    waypoint.transform.location
                )
                
                if distance_to_affected_lane < 50.0:
                    lane_forward = waypoint.transform.get_forward_vector()
                    direction_similarity = (
                        vehicle_forward.x * lane_forward.x + 
                        vehicle_forward.y * lane_forward.y
                    )
                    
                    if direction_similarity > 0.707:
                        return True
                        
            return False
            
        except Exception as e:
            print(f"Error checking traffic light relevance: {e}")
            return False

    def _update_navigation(self):
        """Update navigation state including anticipated maneuver"""
        if not self.route_waypoints:
            return
        
        transform = self.vehicle.get_transform()
        
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            if next_waypoint_index >= len(self.route_waypoints):
                break
            wp, _ = self.route_waypoints[next_waypoint_index]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                        vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
            else:
                break
        
        self.current_waypoint_index = waypoint_index
        
        if self.current_waypoint_index < len(self.route_waypoints):
            self.current_waypoint, self.current_road_maneuver = self.route_waypoints[self.current_waypoint_index]
            
            if self.current_waypoint_index + 1 < len(self.route_waypoints):
                self.next_waypoint, self.next_road_maneuver = self.route_waypoints[self.current_waypoint_index + 1]
        
        self._update_anticipated_maneuver(transform)
    
    def _update_anticipated_maneuver(self, transform):
        """Update anticipated maneuver based on waypoint lookahead"""
        current_location = vector(transform.location)
        max_lookahead = min(15, len(self.route_waypoints) - self.current_waypoint_index - 1)
        
        self.anticipated_maneuver = self.current_road_maneuver
        
        for i in range(1, max_lookahead + 1):
            future_waypoint_index = self.current_waypoint_index + i
            if future_waypoint_index >= len(self.route_waypoints):
                break
            
            future_waypoint, future_maneuver = self.route_waypoints[future_waypoint_index]
            future_location = vector(future_waypoint.transform.location)
            distance_to_future_wp = np.linalg.norm(current_location - future_location)
            
            future_maneuver_name = future_maneuver.name if hasattr(future_maneuver, 'name') else str(future_maneuver)
            
            if (future_maneuver_name in ["LEFT", "RIGHT", "CHANGELANELEFT", "CHANGELANERIGHT"]
                and distance_to_future_wp <= self.anticipation_distance):
                self.anticipated_maneuver = future_maneuver
                break
    
    def _maneuver_to_english(self, maneuver):
        """Convert maneuver enum to English string"""
        if hasattr(maneuver, 'name'):
            maneuver_name = maneuver.name
        else:
            maneuver_name = str(maneuver)
        
        maneuver_map = {
            "LANEFOLLOW": "Lane Follow",
            "LEFT": "Turn Left",
            "RIGHT": "Turn Right", 
            "STRAIGHT": "Go Straight",
            "CHANGELANELEFT": "Change Lane Left",
            "CHANGELANERIGHT": "Change Lane Right",
        }
        return maneuver_map.get(maneuver_name, "Unknown")
    
    def _get_leading_vehicle_info(self):
        """Get leading vehicle distance and speed information"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_forward = vehicle_transform.get_forward_vector()
        vehicle_velocity = self.vehicle.get_velocity()
        vehicle_speed = 3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
        
        leading_vehicle_info = {
            "has_leading_vehicle": False,
            "leading_vehicle_id": None,
            "leading_vehicle_distance": None,
            "leading_vehicle_speed": None,
            "leading_vehicle_relative_speed": None
        }
        
        min_distance = float('inf')
        nearest_vehicle = None
        
        for bg_vehicle in self.traffic_flow_vehicles:
            if not bg_vehicle.is_alive:
                continue
            
            try:
                bg_location = bg_vehicle.get_location()
                distance = vehicle_location.distance(bg_location)
                
                if distance > self.leading_vehicle_detection_distance or distance < 1.0:
                    continue
                
                direction_to_vehicle = bg_location - vehicle_location
                direction_normalized = direction_to_vehicle / distance
                
                dot_product = (vehicle_forward.x * direction_normalized.x + 
                             vehicle_forward.y * direction_normalized.y)
                
                if dot_product > 0.5 and distance < min_distance:
                    min_distance = distance
                    nearest_vehicle = bg_vehicle
            except Exception:
                continue
        
        if nearest_vehicle is not None and nearest_vehicle.is_alive:
            try:
                bg_velocity = nearest_vehicle.get_velocity()
                bg_speed = 3.6 * np.sqrt(bg_velocity.x**2 + bg_velocity.y**2 + bg_velocity.z**2)
                relative_speed = vehicle_speed - bg_speed
                
                leading_vehicle_info.update({
                    "has_leading_vehicle": True,
                    "leading_vehicle_id": nearest_vehicle.id,
                    "leading_vehicle_distance": round(min_distance, 2),
                    "leading_vehicle_speed": round(bg_speed, 2),
                    "leading_vehicle_relative_speed": round(relative_speed, 2)
                })
            except Exception:
                pass
        
        return leading_vehicle_info
    
    def _get_pedestrian_info(self):
        """Get nearest pedestrian distance and lateral offset"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        pedestrian_info = {
            "has_pedestrian_in_range": False,
            "nearest_pedestrian_id": None,
            "nearest_pedestrian_distance": None,
            "nearest_pedestrian_lateral_offset": None,
            "pedestrian_count_in_range": 0
        }
        
        min_distance = float('inf')
        nearest_pedestrian = None
        nearest_pedestrian_id = None
        pedestrian_count = 0
        
        for pedestrian in self.pedestrians:
            if not pedestrian.is_alive:
                continue
            
            try:
                ped_location = pedestrian.get_location()
                distance = vehicle_location.distance(ped_location)
                
                if distance > self.pedestrian_detection_distance or distance < 0.5:
                    continue
                
                pedestrian_count += 1
                
                direction_to_ped = ped_location - vehicle_location
                direction_normalized = direction_to_ped / distance
                
                dot_product = (vehicle_forward.x * direction_normalized.x + 
                             vehicle_forward.y * direction_normalized.y)
                
                if dot_product > 0 and distance < min_distance:
                    min_distance = distance
                    nearest_pedestrian = ped_location
                    nearest_pedestrian_id = pedestrian.id
            except Exception:
                continue
        
        if nearest_pedestrian is not None:
            try:
                direction_to_ped = nearest_pedestrian - vehicle_location
                right_vector = vehicle_transform.get_right_vector()
                lateral_offset = (direction_to_ped.x * right_vector.x + 
                                direction_to_ped.y * right_vector.y)
                
                pedestrian_info.update({
                    "has_pedestrian_in_range": True,
                    "nearest_pedestrian_id": nearest_pedestrian_id,
                    "nearest_pedestrian_distance": round(min_distance, 2),
                    "nearest_pedestrian_lateral_offset": round(lateral_offset, 2),
                    "pedestrian_count_in_range": pedestrian_count
                })
            except Exception:
                pedestrian_info["pedestrian_count_in_range"] = pedestrian_count
        else:
            pedestrian_info["pedestrian_count_in_range"] = pedestrian_count
        
        return pedestrian_info
    
    def get_vehicle_state(self):
        """Collect current vehicle state including all sensors"""
        control = self.vehicle.get_control()
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()
        
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        throttle = control.throttle
        brake = control.brake
        steer = control.steer
        
        current_maneuver = self._maneuver_to_english(self.current_road_maneuver)
        anticipated_maneuver = self._maneuver_to_english(self.anticipated_maneuver)
        
        leading_vehicle_info = self._get_leading_vehicle_info()
        pedestrian_info = self._get_pedestrian_info()
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "vehicle_speed_kmh": round(speed, 2),
            "throttle": round(throttle, 3),
            "brake": round(brake, 3),
            "steer": round(steer, 3),
            "current_maneuver": current_maneuver,
            "anticipated_maneuver": anticipated_maneuver,
            "position": {
                "x": round(transform.location.x, 2),
                "y": round(transform.location.y, 2),
                "z": round(transform.location.z, 2)
            },
            "rotation": {
                "pitch": round(transform.rotation.pitch, 2),
                "yaw": round(transform.rotation.yaw, 2),
                "roll": round(transform.rotation.roll, 2)
            },
            "waypoint_index": self.current_waypoint_index,
            "total_waypoints": len(self.route_waypoints),
            "leading_vehicle": leading_vehicle_info,
            "pedestrian": pedestrian_info
        }
        
        return state
    
    def save_data(self, vehicle_state, reference_distance, traffic_light_state=None):
        if self.front_image is None:
            return False
        
        self.sample_count += 1
        
        image_filename = f"{self.sample_count}.jpg"
        image_path = os.path.join(self.image_dir, image_filename)
        cv2.imwrite(image_path, cv2.cvtColor(self.front_image, cv2.COLOR_RGB2BGR))
        
        vehicle_state["sample_id"] = self.sample_count
        vehicle_state["reference_distance"] = round(reference_distance, 2)
        vehicle_state["image_filename"] = image_filename
        
        if traffic_light_state:
            vehicle_state["traffic_light_state"] = traffic_light_state
        
        json_filename = f"{self.sample_count}.json"
        json_path = os.path.join(self.json_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(vehicle_state, f, indent=2, ensure_ascii=False)
        
        print(f"Saved sample {self.sample_count}")
        return True
    
    def _draw_path(self, camera, viewer_image):
        if not self.route_waypoints or viewer_image is None:
            return viewer_image
        
        return viewer_image
    
    def render(self):
        """Render pygame display with vehicle info and camera feeds"""
        if not self.activate_render or self.display is None:
            return True
        
        self.clock.tick(20)
        new_size = (self.display.get_size()[1] // 2, self.display.get_size()[1] // 2)
        font = pygame.font.Font(None, 24)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == pygame.K_a:
                    if self.autopilot_enabled:
                        self._disable_autopilot()
                    else:
                        self._enable_autopilot()
                elif event.key == pygame.K_r:
                    self.processed_traffic_lights.clear()
                    self.current_collecting_light = None
        
        if self.hud:
            self.hud.tick(self.world, self.clock)
        
        if self.viewer_image is not None:
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        if self.front_image is not None:
            render_image = cv2.resize(self.front_image, (280, 280), interpolation=cv2.INTER_AREA)
            front_rgb_pygame = pygame.surfarray.make_surface(render_image.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(front_rgb_pygame, new_size)
            pos_obs = (self.display.get_size()[0] - self.display.get_size()[1] // 2, 0)
            self.display.blit(scaled_surface, pos_obs)
            front_text = font.render("Front RGB Camera", True, (255, 255, 255))
            self.display.blit(front_text, (pos_obs[0], pos_obs[1]))
      
        extra_info = [
            f"Sample Count: {self.sample_count}",
            f"Expected Maneuver: {self._maneuver_to_english(self.anticipated_maneuver)}",
            f"Speed: {self.vehicle.get_speed():.1f} km/h",
            f"Throttle: {self.vehicle.control.throttle:.2f}",
            f"Steering: {self.vehicle.control.steer:.2f}",
            f"Waypoint: {self.current_waypoint_index}/{len(self.route_waypoints)}",
            "",
            "Controls:", 
            "A - Toggle Autopilot", 
            "R - Reset Traffic Light Record",
            "ESC - Exit Program", 
        ]
        
        if self.hud:
            self.hud.render(self.display, extra_info=extra_info)
        
        pygame.display.flip()
        return True
    
    def run_collection(self, duration_seconds=300):
        start_time = time.time()
        total_samples = self.sample_count
        processed_lights = self.processed_traffic_lights.copy()
        
        try:
            while time.time() - start_time < duration_seconds:
                self.world.tick()
                self._update_navigation()
                
                if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                    total_samples = self.sample_count
                    processed_lights = self.processed_traffic_lights.copy()
                    self.cleanup()

                    if not self.initialize_carla():
                        break
                    
                    self.sample_count = total_samples
                    self.processed_traffic_lights = processed_lights
                    self.current_collecting_light = None
                    continue
                
                if self.autopilot_enabled:
                    vehicle_actor = self.vehicle.get_carla_actor()
                    self.traffic_manager.set_desired_speed(vehicle_actor, 40.0)
                
                nearest_light, distance, can_collect = self.get_nearest_traffic_light()
                leading_vehicle_info = self._get_leading_vehicle_info()
                pedestrian_info = self._get_pedestrian_info()
                trigger_types = []
                trigger_meta = {}
                
                # Trigger 1: traffic light region
                if nearest_light and can_collect:
                    light_id = nearest_light.id
                    light_count = self.traffic_light_sample_counts.get(light_id, 0)
                    if light_count < self.entity_sample_limit:
                        self.traffic_light_sample_counts[light_id] = light_count + 1
                        trigger_types.append("traffic_light")
                        light_state = nearest_light.get_state()
                        state_map = {
                            carla.TrafficLightState.Red: "Red",
                            carla.TrafficLightState.Yellow: "Yellow", 
                            carla.TrafficLightState.Green: "Green",
                            carla.TrafficLightState.Off: "Off",
                            carla.TrafficLightState.Unknown: "Unknown"
                        }
                        light_state_str = state_map.get(light_state, "Unknown")
                        trigger_meta["traffic_light_id"] = light_id
                        trigger_meta["traffic_light_state"] = light_state_str
                        trigger_meta["traffic_light_distance"] = round(distance, 2)

                # Trigger 2: leading background vehicle ahead
                if leading_vehicle_info.get("has_leading_vehicle"):
                    lead_id = leading_vehicle_info.get("leading_vehicle_id")
                    if lead_id is not None:
                        lead_count = self.leading_vehicle_sample_counts.get(lead_id, 0)
                        if lead_count < self.entity_sample_limit:
                            self.leading_vehicle_sample_counts[lead_id] = lead_count + 1
                            trigger_types.append("leading_vehicle")
                            trigger_meta["leading_vehicle_id"] = lead_id
                            trigger_meta["leading_vehicle_distance"] = leading_vehicle_info.get("leading_vehicle_distance")

                # Trigger 3: pedestrian ahead
                if pedestrian_info.get("has_pedestrian_in_range"):
                    ped_id = pedestrian_info.get("nearest_pedestrian_id")
                    if ped_id is not None:
                        ped_count = self.pedestrian_sample_counts.get(ped_id, 0)
                        if ped_count < self.entity_sample_limit:
                            self.pedestrian_sample_counts[ped_id] = ped_count + 1
                            trigger_types.append("pedestrian")
                            trigger_meta["nearest_pedestrian_id"] = ped_id
                            trigger_meta["nearest_pedestrian_distance"] = pedestrian_info.get("nearest_pedestrian_distance")

                if trigger_types:
                    vehicle_state = self.get_vehicle_state()
                    vehicle_state["autopilot_enabled"] = self.autopilot_enabled
                    vehicle_state["traffic_manager_port"] = self.tm_port
                    vehicle_state["ignore_traffic_lights"] = True
                    vehicle_state["collection_triggers"] = trigger_types
                    vehicle_state["primary_collection_type"] = trigger_types[0]
                    vehicle_state.update(trigger_meta)

                    reference_distance = (
                        trigger_meta.get("traffic_light_distance")
                        or trigger_meta.get("nearest_pedestrian_distance")
                        or trigger_meta.get("leading_vehicle_distance")
                        or 0.0
                    )

                    if self.save_data(
                        vehicle_state,
                        reference_distance,
                        trigger_meta.get("traffic_light_state")
                    ):
                        time.sleep(0.2)
                
                if not self.render():
                    break
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")

    def cleanup(self):
        self._disable_autopilot()

        for controller in self.pedestrian_controllers:
            try:
                if controller.is_alive:
                    controller.stop()
                    controller.destroy()
            except Exception:
                pass
        self.pedestrian_controllers.clear()
        
        for bg_vehicle in self.traffic_flow_vehicles:
            if bg_vehicle.is_alive:
                bg_vehicle.destroy()
        self.traffic_flow_vehicles.clear()
        
        for pedestrian in self.pedestrians:
            if pedestrian.is_alive:
                pedestrian.destroy()
        self.pedestrians.clear()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.world.destroy()
        
        if self.activate_render:
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='CARLA Traffic Light Data Collector')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host address')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')     
    parser.add_argument('--town', default='Town02', help='CARLA map name')   
    parser.add_argument('--duration', type=int, default=5000, help='Collection duration (seconds)')  
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--tf-num', type=int, default=50, help='Number of background vehicles')
    parser.add_argument('--pedestrian-num', type=int, default=20, help='Number of pedestrians')
    parser.add_argument('--pedestrian-ai', action='store_true', help='Enable pedestrian AI controllers (less stable)')
    parser.add_argument('--fixed-traffic-lights', action='store_true', help='Disable dynamic traffic light cycling')
    args = parser.parse_args()
    
    collector = TrafficLightCollector(host=args.host, port=args.port, town=args.town)
    collector.tf_num = max(0, args.tf_num)
    collector.pedestrian_num = max(0, args.pedestrian_num)
    collector.enable_pedestrian_ai = args.pedestrian_ai
    collector.use_dynamic_traffic_lights = not args.fixed_traffic_lights
    if args.no_render:
        collector.activate_render = False
    
    try:
        if collector.initialize_carla():
            collector.run_collection(duration_seconds=args.duration)
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
