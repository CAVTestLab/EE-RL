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

class NormalRoadCollector:
    def __init__(self, host="127.0.0.1", port=2000, town='Town02'):
        self.host = host
        self.port = port
        self.town = town
        
        self.image_width = 600
        self.image_height = 600
        self.collection_interval = 0.2
        self.last_collection_time = 0
        self.traffic_light_avoidance_distance = 15.0
        
        self.output_dir = "vlm_lora_data/normal_road_data"
        self.image_dir = os.path.join(self.output_dir, "images")
        self.json_dir = os.path.join(self.output_dir, "vehicle_states")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        self.sample_count = self._get_max_sample_count()

        self.client = None
        self.world = None
        self.vehicle = None
        self.front_camera = None
        self.front_image = None

        self.traffic_manager = None
        self.tm_port = port + 5000
        self.autopilot_enabled = False
        
        self.viewer_res = (1120, 560)
        self.activate_render = True
        self.display = None
        self.clock = None
        self.hud = None
        self.camera = None
        self.viewer_image = None
        
        self.route_waypoints = []
        self.current_waypoint_index = 0
        self.current_waypoint = None
        self.next_waypoint = None
        self.current_road_maneuver = RoadOption.LANEFOLLOW
        self.next_road_maneuver = RoadOption.LANEFOLLOW
        self.anticipated_maneuver = RoadOption.LANEFOLLOW
        
        self.vehicle_speed = 0.0
        self.vehicle_throttle = 0.0
        self.vehicle_steer = 0.0
        self.current_maneuver = "Unknown"
   
    def _get_max_sample_count(self):
        max_idx = 0
        if os.path.exists(self.image_dir):
            for fname in os.listdir(self.image_dir):
                match = re.match(r"(\d+)\.jpg", fname)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
        return max_idx
    
    def initialize_carla(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            
            self.world = World(self.client, town=self.town)
            
            # Keep simulation deterministic for data capture.
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
            self._enable_autopilot()
            return True
            
        except Exception as e:
            print(f"CARLA initialization failed: {e}")
            return False
    
    def _setup_traffic_manager(self):
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)

        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(50.0)

        self.traffic_manager.global_percentage_speed_difference(0.0)
        self.traffic_manager.set_boundaries_respawn_dormant_vehicles(25.0, 700.0)
    
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
            
            self.traffic_manager.set_desired_speed(vehicle_actor, 35.0)
    
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
        """Spawn ego vehicle with retry."""
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
            except Exception as e:
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
            except Exception as e:
                attempt += 1
        
        if not success:
            raise RuntimeError("Unable to spawn ego vehicle on available points.")
    
    def _setup_rendering(self):
        """Setup pygame rendering window."""
        if self.activate_render:
            pygame.init()
            pygame.font.init()
            width, height = self.viewer_res
            self.display = pygame.display.set_mode((width, height), HWSURFACE | DOUBLEBUF)
            pygame.display.set_caption("CARLA Normal Road Data Collector")
            self.clock = pygame.time.Clock()
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)
    
    def _setup_sensors(self):
        """Create spectator and front RGB cameras."""
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
        """Create initial route."""
        spawn_points = self.world.map.get_spawn_points()
        import random
        start_idx = random.randint(0, len(spawn_points) - 1)
        start_wp = self.world.map.get_waypoint(spawn_points[start_idx].location)

        for i in range(10, min(50, len(spawn_points))):
            end_wp = self.world.map.get_waypoint(spawn_points[i].location)
            route_waypoints = compute_route_waypoints(
                self.world.map, start_wp, end_wp, resolution=1.0
            )
            if len(route_waypoints) > 30:
                self.route_waypoints = route_waypoints
                break

        self.current_waypoint_index = 0

        if self.route_waypoints:
            self.current_waypoint, self.current_road_maneuver = self.route_waypoints[0]
            if len(self.route_waypoints) > 1:
                self.next_waypoint, self.next_road_maneuver = self.route_waypoints[1]

        self._set_route_for_autopilot()
    
    def _create_new_route(self):
        """Create a new route when current route ends."""
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
        
        if len(route_waypoints) < 30:
            return self._create_new_route()
        
        self.route_waypoints = route_waypoints
        self.current_waypoint_index = 0
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[0]
        if len(self.route_waypoints) > 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[1]
        self._set_route_for_autopilot()

    def _set_viewer_image(self, image):
        """Update spectator image buffer."""
        self.viewer_image = image
    
    def _set_front_image(self, image):
        """Update front camera image buffer."""
        self.front_image = image
    
    def _on_collision(self, event):
        """Collision callback."""
    
    def _on_invasion(self, event):
        """Lane invasion callback."""

    def _update_navigation(self):
        """Update waypoint tracking and anticipated maneuver."""
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
        """Predict upcoming maneuver from lookahead waypoints."""
        current_location = vector(transform.location)
        max_lookahead = min(15, len(self.route_waypoints) - self.current_waypoint_index - 1)
        
        self.anticipated_maneuver = self.current_road_maneuver
        anticipation_distance = 15.0
        
        for i in range(1, max_lookahead + 1):
            future_waypoint_index = self.current_waypoint_index + i
            if future_waypoint_index >= len(self.route_waypoints):
                break
            
            future_waypoint, future_maneuver = self.route_waypoints[future_waypoint_index]
            future_location = vector(future_waypoint.transform.location)
            distance_to_future_wp = np.linalg.norm(current_location - future_location)
            
            future_maneuver_name = future_maneuver.name if hasattr(future_maneuver, 'name') else str(future_maneuver)
            
            if (future_maneuver_name in ["LEFT", "RIGHT", "CHANGELANELEFT", "CHANGELANERIGHT"]
                and distance_to_future_wp <= anticipation_distance):
                self.anticipated_maneuver = future_maneuver
                break
    
    def _maneuver_to_english(self, maneuver):
        """Convert maneuver enum to readable string."""
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
    
    def get_vehicle_state(self):
        """Build current vehicle state payload."""
        control = self.vehicle.get_control()
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()
        
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        throttle = control.throttle
        brake = control.brake
        steer = control.steer
        
        current_maneuver = self._maneuver_to_english(self.current_road_maneuver)
        anticipated_maneuver = self._maneuver_to_english(self.anticipated_maneuver)
        
        return {
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
            "total_waypoints": len(self.route_waypoints)
        }
    
    def get_nearest_traffic_light(self):
        """Return nearest relevant traffic light and distance."""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        vehicle_waypoint = self.world.map.get_waypoint(vehicle_location)
        
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        
        if not traffic_lights:
            return None, float('inf')
        
        nearest_light = None
        min_distance = float('inf')
        
        for light in traffic_lights:
            light_location = light.get_location()
            distance = vehicle_location.distance(light_location)
            
            if distance > 50.0:
                continue
                
            direction_to_light = light_location - vehicle_location
            direction_to_light_normalized = direction_to_light / distance
            
            dot_product = (vehicle_forward.x * direction_to_light_normalized.x + 
                        vehicle_forward.y * direction_to_light_normalized.y)
            
            if dot_product < 0.0:
                continue
                
            if self._is_traffic_light_relevant(light, vehicle_waypoint, vehicle_forward):
                if distance < min_distance:
                    min_distance = distance
                    nearest_light = light
        
        return nearest_light, min_distance

    def _is_traffic_light_relevant(self, traffic_light, vehicle_waypoint, vehicle_forward):
        """Check whether a traffic light is relevant for ego lane."""
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
                
                if distance_to_affected_lane < 30.0:
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

    def save_data(self, vehicle_state, traffic_light_distance):
        """Persist image and state json for one sample."""
        if self.front_image is None:
            return False
        
        self.sample_count += 1

        # Save paired image and metadata atomically per sample id.
        image_filename = f"{self.sample_count}.jpg"
        image_path = os.path.join(self.image_dir, image_filename)
        cv2.imwrite(image_path, cv2.cvtColor(self.front_image, cv2.COLOR_RGB2BGR))
        
        vehicle_state["sample_id"] = self.sample_count
        vehicle_state["traffic_light_distance"] = round(traffic_light_distance, 2) if traffic_light_distance != float('inf') else None
        vehicle_state["traffic_light_state"] = "No Traffic Light"
        vehicle_state["image_filename"] = image_filename
        vehicle_state["autopilot_enabled"] = self.autopilot_enabled
        vehicle_state["traffic_manager_port"] = self.tm_port
        vehicle_state["ignore_traffic_lights"] = True
        
        json_filename = f"{self.sample_count}.json"
        json_path = os.path.join(self.json_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(vehicle_state, f, indent=2, ensure_ascii=False)
        return True
    
    def render(self):
        """Render spectator/front camera and HUD overlay."""
        if not self.activate_render or self.display is None:
            return True
        
        self.clock.tick(20)
        
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
        
        if self.hud:
            self.hud.tick(self.world, self.clock)
        
        if self.viewer_image is not None:
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
            
        font = pygame.font.Font(None, 24)

        if self.front_image is not None:
            new_size = (self.display.get_size()[1] // 2, self.display.get_size()[1] // 2)
            render_image = cv2.resize(self.front_image, (280, 280), interpolation=cv2.INTER_AREA)
            front_rgb_pygame = pygame.surfarray.make_surface(render_image.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(front_rgb_pygame, new_size)
            pos_obs = (self.display.get_size()[0] - self.display.get_size()[1] // 2, 0)
            self.display.blit(scaled_surface, pos_obs)
            front_text = font.render("Front RGB Camera", True, (255, 255, 255))
            self.display.blit(front_text, (pos_obs[0], pos_obs[1]))

        _, traffic_light_distance = self.get_nearest_traffic_light()
        distance_str = f"{traffic_light_distance:.1f}m" if traffic_light_distance != float('inf') else "No TL"
        
        extra_info = [
            f"Sample Count: {self.sample_count}",
            f"Collection Mode: Normal Road (Avoiding Traffic Lights)",
            f"Nearest Traffic Light: {distance_str}",
            f"Expected Maneuver: {self._maneuver_to_english(self.anticipated_maneuver)}",
            f"Speed: {self.vehicle.get_speed():.1f} km/h",
            f"Throttle: {self.vehicle.control.throttle:.2f}",
            f"Steering: {self.vehicle.control.steer:.2f}",
            f"Waypoint: {self.current_waypoint_index}/{len(self.route_waypoints)}",
            "",
            "Controls:", 
            "A - Toggle Autopilot",
            "ESC - Exit Program", 
        ]
        
        if self.hud:
            self.hud.render(self.display, extra_info=extra_info)
        
        pygame.display.flip()
        return True
    
    def run_collection(self, duration_seconds=300):
        """Run periodic data collection away from traffic lights."""
        
        start_time = time.time()
        self.last_collection_time = start_time
        
        try:
            while time.time() - start_time < duration_seconds:
                self.world.tick()
                
                self._update_navigation()
                
                if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                    self._create_new_route()
                    continue
                
                nearest_light, traffic_light_distance = self.get_nearest_traffic_light()
                
                # Skip samples near intersections to keep normal-road distribution.
                current_time = time.time()
                if current_time - self.last_collection_time >= self.collection_interval:
                    if traffic_light_distance >= self.traffic_light_avoidance_distance:
                        vehicle_state = self.get_vehicle_state()
                        if self.save_data(vehicle_state, traffic_light_distance):
                            self.last_collection_time = current_time
                    else:
                        self.last_collection_time = current_time
                
                if not self.render():
                    break
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")

    def cleanup(self):
        """Release actors and restore async world settings."""
        
        self._disable_autopilot()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            self.world.destroy()
        
        if self.activate_render:
            pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='CARLA Normal Road Data Collector')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host address')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')     
    parser.add_argument('--town', default='Town02', help='CARLA map name')   
    parser.add_argument('--duration', type=int, default=1800, help='Collection duration (seconds)')  
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--interval', type=float, default=1.0, help='Collection interval (seconds)')
    args = parser.parse_args()
    
    collector = NormalRoadCollector(host=args.host, port=args.port, town=args.town)
    collector.collection_interval = args.interval
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
