import glob
import os
import subprocess
import time
import gym
import pygame
import cv2
from pygame.locals import HWSURFACE, DOUBLEBUF, K_ESCAPE
import random
from config import CONFIG
from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.wrappers import *
import carla
from collections import deque
import itertools
from datetime import datetime

import io
from PIL import Image
import tempfile
import config
import numpy as np

CONFIG
intersection_routes = itertools.cycle(
    [(57, 81), (70, 11), (70, 12), (78, 68), (74, 41), (42, 73), (71, 62), (74, 40), (71, 77), (6, 12), (65, 52),
     (63, 80)])
eval_routes = itertools.cycle(
    [(48, 21), (0, 72), (28, 83), (61, 39), (27, 14), (6, 67), (61, 49), (37, 64), (33, 80), (12, 30), ])

class_blueprint = {
    'car': ['vehicle.tesla.model3',
            'vehicle.audi.tt',
            'vehicle.chevrolet.impala',
            'vehicle.audi.a2',
            'vehicle.audi.etron',
            'vehicle.bmw.grandtourer',
            'vehicle.citroen.c3',
            'vehicle.dodge.charger_police',
            'vehicle.ford.mustang',
            'vehicle.jeep.wrangler_rubicon',
            'vehicle.lincoln.mkz_2017',
            'vehicle.mercedes.coupe',
            'vehicle.mini.cooper_s',
            'vehicle.nissan.patrol',
            'vehicle.seat.leon',
            'vehicle.toyota.prius',
            'vehicle.volkswagen.t2',
            'vehicle.nissan.micra',
            'vehicle.mercedes.sprinter',
            'vehicle.ford.ambulance'],
    'pedestrian': ['walker.pedestrian.0001',
                   'walker.pedestrian.0002',
                   'walker.pedestrian.0003',
                   'walker.pedestrian.0004',
                   'walker.pedestrian.0005',
                   'walker.pedestrian.0006',
                   'walker.pedestrian.0007',
                   'walker.pedestrian.0008',
                   'walker.pedestrian.0009',
                   'walker.pedestrian.0010',
                   'walker.pedestrian.0011',
                   'walker.pedestrian.0012',
                   'walker.pedestrian.0013',
                   'walker.pedestrian.0014',
                   'walker.pedestrian.0015',
                   'walker.pedestrian.0016',
                   'walker.pedestrian.0017',
                   'walker.pedestrian.0018',
                   'walker.pedestrian.0019',
                   'walker.pedestrian.0020']
}


def random_choice_from_blueprint(blueprint):
    """Select random vehicle/actor from blueprint pool"""
    all_elements = [item for sublist in blueprint.values() for item in sublist]
    return random.choice(all_elements)


def random_choice_from_category(blueprint, category):
    """Select random actor from specific category"""
    if category in blueprint:
        return random.choice(blueprint[category])
    return None


class CarlaRouteEnv(gym.Env):
    global_step_counter = 0
    PHASE_1_THRESHOLD = 200000 # regular scenarios
    PHASE_2_THRESHOLD = 500000 # sparse-critical scenarios

    def __init__(self, host="127.0.0.1", port=2000,
                viewer_res=(1600, 900), obs_res=(80, 120),
                reward_fn=None,
                observation_space=None,
                encode_state_fn=None,
                fps=15, action_smoothing=0.0,
                activate_spectator=True,
                start_carla=False,
                eval=False,
                activate_render=False,
                activate_traffic_flow=False,
                tf_num=40,
                town='Town02',
                use_vlm=False,
                save_vlm_image=False,
                activate_front_rgb=False,
                anticipation_distance=15.0,
                traffic_light_detection_distance=50.0,
                activate_pedestrians=True,
                pedestrian_num=50,
                 ):
        self.viewer_res = viewer_res
        self.obs_res = obs_res
        self.host = host
        self.port = port
        self.reward_fn = reward_fn
        self.observation_space = observation_space
        self.encode_state_fn = encode_state_fn
        self.fps = fps
        self.action_smoothing = action_smoothing
        self.activate_spectator = activate_spectator
        self.start_carla = start_carla
        self.eval = eval
        self.activate_render = activate_render
        self.activate_traffic_flow = activate_traffic_flow
        self.tf_num = tf_num
        self.town = town
        self.use_vlm = use_vlm
        self.save_vlm_image = save_vlm_image
        self.activate_front_rgb = activate_front_rgb
        self.anticipation_distance = anticipation_distance
        self.traffic_light_detection_distance = traffic_light_detection_distance
        self.activate_pedestrians = activate_pedestrians
        self.pedestrian_num = pedestrian_num
        self.traffic_flow_vehicles = []
        self.pedestrians = []
        
        self.current_traffic_light_info = {
            'in_range': False,
            'distance': float('inf'),
            'state': 'Unknown',
            'affects_current_lane': False,
            'traffic_light': None,
            'traffic_light_id': None,
            'is_opposite_side': False
        }

        self.carla_process = None
        if start_carla:
            carla_path = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
            launch_command = [carla_path, '-quality_level=Low', '-benchmark',
                            f"-fps={fps}", '-prefernvidia', f'-carla-world-port={port}']
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL, 
                                                 stderr=subprocess.STDOUT)
            time.sleep(10)

        width, height = viewer_res
        self.activate_render = activate_render
        self.num_envs = 1

        # Setup continuous action space: [steer, throttle]
        self.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]),
                                           dtype=np.float32)

        self.observation_space = observation_space
        self.episode_idx = -2
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        
        # Episode tracking
        self.max_distance = 3000
        self.low_speed_timer = 0.0
        self.collision_num = 0
        self.cps = 0
        self.cpm = 0
        self.collision_speed = 0.0
        self.collision_interval = 0
        self.last_collision_step = 0
        self.collision_deque = deque(maxlen=100)
        self.total_steps = 0

        self.world = None
        self.last_capture_time = 0
        self.last_vlm_output = None

        # Dynamic weather settings
        self.dynamic_weather_enabled = False
        self.dynamic_weather_transition_steps = max(int(self.fps * 20), 1)
        self.dynamic_weather_step = 0
        self.dynamic_weather_source = None
        self.dynamic_weather_target = None
        self.current_weather = None
        self.weather_presets = [
            carla.WeatherParameters(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0,
                wind_intensity=5.0, sun_altitude_angle=55.0, fog_density=0.0, fog_distance=100.0, wetness=0.0),
            carla.WeatherParameters(cloudiness=65.0, precipitation=0.0, precipitation_deposits=0.0,
                wind_intensity=20.0, sun_altitude_angle=35.0, fog_density=5.0, fog_distance=80.0, wetness=15.0),
            carla.WeatherParameters(cloudiness=90.0, precipitation=45.0, precipitation_deposits=35.0,
                wind_intensity=35.0, sun_altitude_angle=20.0, fog_density=15.0, fog_distance=60.0, wetness=60.0),
            carla.WeatherParameters(cloudiness=100.0, precipitation=90.0, precipitation_deposits=85.0,
                wind_intensity=55.0, sun_altitude_angle=8.0, fog_density=30.0, fog_distance=45.0, wetness=90.0)
        ]

        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(5.0)
            self.world = World(self.client, town=town)

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)
            
            if not self.eval and self.dynamic_weather_enabled:
                self._init_dynamic_weather()

            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e),
                                   is_ego=True)

            if self.activate_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(width, height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            if self.activate_spectator:
                self.camera = Camera(self.world, width, height,
                                     transform=sensor_transforms["spectator"],
                                     attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
            
            if self.activate_front_rgb:
                self.front_rgb_camera = Camera(
                    self.world,
                    width=config.RGB_CAMERA_WIDTH,
                    height=config.RGB_CAMERA_HEIGHT,
                    transform=sensor_transforms["front"],
                    attach_to=self.vehicle,
                    on_recv_image=lambda e: self._set_front_rgb_data(e),
                )
                os.makedirs("output_front_rgb", exist_ok=True)

            if activate_traffic_flow:
                self.traffic_manager = self.client.get_trafficmanager(port + 5000)
                self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
                self.traffic_manager.set_synchronous_mode(True)
                self.traffic_manager.set_hybrid_physics_mode(True)
                self.traffic_manager.set_hybrid_physics_radius(50.0)
            else:
                self.traffic_manager = None
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to initialize CARLA environment: {e}")
        
        self.reset()

    # ============================================================================
    # GYM INTERFACE: Core environment methods (reset, step, render, close)
    # ============================================================================

    def reset(self, is_training=False):
        """Initialize new episode: reset vehicle state and route"""
        self.num_routes_completed = -1
        self.episode_idx += 1
        self.new_route()

        self.terminal_state = False
        self.success_state = False
        self.collision_state = False

        self.closed = False
        self.extra_info = []
        self.viewer_image = self.viewer_image_buffer = None
        self.front_rgb_data = self.front_rgb_image_buffer = None
        self.step_count = 0

        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.low_speed_timer = 0.0
        self.collision = False
        self.action_list = []
        
        self.world.tick()
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)

        return obs

    def step(self, action):
        """Execute action and return next state, reward, done flag, info"""
        if self.closed:
            raise RuntimeError("step() called on closed environment. Check info['closed'].")
        
        executed_action = None

        if action is not None:
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                if not self.eval:
                    self.new_route()
                else:
                    self.success_state = True

            steer, throttle = [float(a) for a in action]
            
            transform = self.vehicle.get_transform()
            current_phase = self._get_current_learning_phase()

            executed_action = np.array([[steer, throttle]], dtype=np.float32)
            self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
            if throttle >= 0:
                self.vehicle.control.throttle = throttle
                self.vehicle.control.brake = 0
            else:
                self.vehicle.control.throttle = 0
                self.vehicle.control.brake = abs(throttle)
            self.action_list.append(self.vehicle.control.steer)
        
        self._update_dynamic_weather()
        self.world.tick()

        if self.activate_spectator:
            self.viewer_image = self._get_viewer_image()
        
        if self.activate_front_rgb:
            self.front_rgb_data = self._get_front_rgb_data()

        if self.use_vlm and self.front_rgb_camera:
            self._process_front_rgb_image(self.front_rgb_data)

        transform = self.vehicle.get_transform()
        self.current_traffic_light_info = self._detect_traffic_lights_in_range(transform)
        
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
            else:
                break
        self.current_waypoint_index = waypoint_index

        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        
        self._update_anticipated_maneuver(transform)
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        self.speed_accum += self.vehicle.get_speed()

        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)
        self._update_global_step_counter()
        
        current_phase = self._get_current_learning_phase()
        additional_reward = 0.0

        self.last_reward = self.reward_fn(self) + additional_reward
        self.total_reward += self.last_reward

        encoded_state = self.encode_state_fn(self)
        self.step_count += 1
        self.total_steps += 1
        
        if self.activate_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self.terminal_state = True
            self.render()

        max_distance = CONFIG.reward_params.max_distance
        max_std_center_lane = CONFIG.reward_params.max_std_center_lane
        max_angle_center_lane = CONFIG.reward_params.max_angle_center_lane
        centering_factor = max(1.0 - self.distance_from_center / max_distance, 0.0)

        angle = self.vehicle.get_angle(self.current_waypoint)
        angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

        std = np.std(self.distance_from_center_history)
        distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

        if self.terminal_state or self.success_state:
            if self.collision_state:
                self.collision_num += 1
                self.collision_deque.append(1)
                self.cps = 1 / self.step_count
                self.collision_interval = self.total_steps - self.last_collision_step
                self.last_collision_step = self.total_steps
                self.collision_speed = self.vehicle.get_speed()
            else:
                self.collision_deque.append(0)
        
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        maneuver_name = self.current_road_maneuver.name if hasattr(self.current_road_maneuver, 'name') else str(self.current_road_maneuver)
        
        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count),
            "centering_factor": centering_factor,
            "angle_factor": angle_factor,
            "distance_std_factor": distance_std_factor,
            "speed": self.vehicle.get_speed(),
            "collision_num": self.collision_num,
            "collision_rate": sum(self.collision_deque) / len(self.collision_deque) if self.collision_deque else 0.0,
            "episode_length": self.step_count,
            "collision_state": self.collision_state,
            "vehicle_speed": self.vehicle.get_speed(),
            "vehicle_yaw": vehicle_yaw,
            "current_maneuver": maneuver_name,
            "traffic_light_in_range": self.current_traffic_light_info['in_range'],
            "traffic_light_distance": self.current_traffic_light_info['distance'],
            "traffic_light_state": self.current_traffic_light_info['state'],
            "traffic_light_affects_lane": self.current_traffic_light_info['affects_current_lane'],
            "traffic_light_id": self.current_traffic_light_info['traffic_light_id'],
            "curriculum_phase": current_phase,
            "global_steps": CarlaRouteEnv.global_step_counter,
            "weather_cloudiness": self.current_weather.cloudiness if self.current_weather is not None else None,
            "weather_precipitation": self.current_weather.precipitation if self.current_weather is not None else None,
            "weather_sun_altitude": self.current_weather.sun_altitude_angle if self.current_weather is not None else None,
            "executed_action": executed_action,
        }

        if self.terminal_state or self.success_state:
            if self.collision_state:
                info.update({"CPS": self.cps,
                             "CPM": self.cpm,
                             "collision_interval": self.collision_interval,
                             "collision_speed": self.collision_speed,
                             })
        return encoded_state, self.last_reward, self.terminal_state or self.success_state, info

    def render(self, mode="human"):
        """Render frame for visualization (pygame)"""
        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation_render

        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        if self.current_road_maneuver == RoadOption.LANEFOLLOW:
            maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:
            maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:
            maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT:
            maneuver = "Straight"
        elif self.current_road_maneuver == RoadOption.CHANGELANELEFT:
            maneuver = "Change Lane Left"
        elif self.current_road_maneuver == RoadOption.CHANGELANERIGHT:
            maneuver = "Change Lane Right"
        else:
            maneuver = "INVALID"

        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "",
            "Maneuver:  % 17s" % maneuver,
            "Throttle:            %7.2f" % self.vehicle.control.throttle,
            "Brake:               %7.2f" % self.vehicle.control.brake,
            "Steer:               %7.2f" % self.vehicle.control.steer,
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
        ])

        if self.activate_spectator:
            self.viewer_image = self._draw_path(self.camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
            new_size = (self.display.get_size()[1] // 2, self.display.get_size()[1] // 2)
            font = pygame.font.Font(None, 24)

            if self.front_rgb_data is not None:
                render_image = cv2.resize(self.front_rgb_data, (400, 400), interpolation=cv2.INTER_AREA)
                front_rgb_pygame = pygame.surfarray.make_surface(render_image.swapaxes(0, 1))
                scaled_surface = pygame.transform.scale(front_rgb_pygame, new_size)
                pos_obs = (self.display.get_size()[0] - self.display.get_size()[1] // 2, 0)
                self.display.blit(scaled_surface, pos_obs)
        
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []

        pygame.display.flip()

    def close(self):
        """Clean up environment: destroy actors, close pygame, terminate CARLA"""
        try:
            if hasattr(self, 'traffic_flow_vehicles'):
                for bg_veh in list(self.traffic_flow_vehicles):
                    if bg_veh.is_alive:
                        bg_veh.destroy()
                self.traffic_flow_vehicles.clear()
            
            if hasattr(self, 'pedestrians'):
                for pedestrian in list(self.pedestrians):
                    if pedestrian.is_alive:
                        pedestrian.destroy()
                self.pedestrians.clear()
            
            if self.world is not None:
                all_vehicles = self.world.get_actors().filter('vehicle.*')
                for vehicle in all_vehicles:
                    try:
                        vehicle.destroy()
                    except:
                        pass
                        
        except Exception as e:
            pass
            
        if self.carla_process:
            self.carla_process.terminate()
            
        try:
            pygame.quit()
        except:
            pass
            
        if self.world is not None:
            self.world.destroy()
            
        self.closed = True

    # ============================================================================
    # WEATHER SYSTEM: Dynamic weather management
    # ============================================================================

    def _interpolate_weather(self, source_weather, target_weather, alpha):
        """Interpolate between two weather states"""
        return carla.WeatherParameters(
            cloudiness=source_weather.cloudiness + (target_weather.cloudiness - source_weather.cloudiness) * alpha,
            precipitation=source_weather.precipitation + (target_weather.precipitation - source_weather.precipitation) * alpha,
            precipitation_deposits=source_weather.precipitation_deposits + (target_weather.precipitation_deposits - source_weather.precipitation_deposits) * alpha,
            wind_intensity=source_weather.wind_intensity + (target_weather.wind_intensity - source_weather.wind_intensity) * alpha,
            sun_azimuth_angle=source_weather.sun_azimuth_angle + (target_weather.sun_azimuth_angle - source_weather.sun_azimuth_angle) * alpha,
            sun_altitude_angle=source_weather.sun_altitude_angle + (target_weather.sun_altitude_angle - source_weather.sun_altitude_angle) * alpha,
            fog_density=source_weather.fog_density + (target_weather.fog_density - source_weather.fog_density) * alpha,
            fog_distance=source_weather.fog_distance + (target_weather.fog_distance - source_weather.fog_distance) * alpha,
            wetness=source_weather.wetness + (target_weather.wetness - source_weather.wetness) * alpha,
        )

    def _init_dynamic_weather(self):
        """Initialize dynamic weather system"""
        self.dynamic_weather_source = random.choice(self.weather_presets)
        self.dynamic_weather_target = random.choice(self.weather_presets)
        while self.dynamic_weather_target == self.dynamic_weather_source and len(self.weather_presets) > 1:
            self.dynamic_weather_target = random.choice(self.weather_presets)

        self.dynamic_weather_step = 0
        self.current_weather = self.dynamic_weather_source
        self.world.set_weather(self.current_weather)

    def _update_dynamic_weather(self):
        """Update weather with smooth transition"""
        if self.eval or not self.dynamic_weather_enabled:
            return

        if self.dynamic_weather_source is None or self.dynamic_weather_target is None:
            self._init_dynamic_weather()
            return

        self.dynamic_weather_step += 1
        alpha = min(self.dynamic_weather_step / float(self.dynamic_weather_transition_steps), 1.0)
        self.current_weather = self._interpolate_weather(self.dynamic_weather_source, self.dynamic_weather_target, alpha)
        self.world.set_weather(self.current_weather)

        if alpha >= 1.0:
            self.dynamic_weather_source = self.dynamic_weather_target
            self.dynamic_weather_target = random.choice(self.weather_presets)
            while self.dynamic_weather_target == self.dynamic_weather_source and len(self.weather_presets) > 1:
                self.dynamic_weather_target = random.choice(self.weather_presets)
            self.dynamic_weather_step = 0

    # ============================================================================
    # TRAFFIC LIGHT DETECTION & REWARD: Traffic signal perception and planning
    # ============================================================================

    def _is_traffic_light_opposite_side(self, traffic_light):
        """Check if traffic light is on opposite side of road (e.g., Town03)"""
        try:
            ego_wp = self.world.map.get_waypoint(self.vehicle.get_location())
            tl_loc = traffic_light.get_location()
            right = ego_wp.transform.get_right_vector()
            to_tl = tl_loc - ego_wp.transform.location
            lateral = abs(right.x * to_tl.x + right.y * to_tl.y)
            return lateral > 6.0
        except Exception:
            return False

    def _detect_traffic_lights_in_range(self, vehicle_transform):
        """Detect traffic lights affecting vehicle route within detection range"""
        vehicle_location = vehicle_transform.location

        result = {
            'in_range': False,
            'distance': float('inf'),
            'state': 'Unknown',
            'affects_current_lane': False,
            'traffic_light': None,
            'traffic_light_id': None,
            'is_opposite_side': False
        }

        try:
            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl is not None:
                    ego_fwd = vehicle_transform.get_forward_vector()
                    has_forward_wp = False
                    try:
                        ref_wps = tl.get_stop_waypoints() or tl.get_affected_lane_waypoints()
                        if ref_wps:
                            for wp in ref_wps:
                                to_wp = wp.transform.location - vehicle_location
                                dot = ego_fwd.x * to_wp.x + ego_fwd.y * to_wp.y
                                if dot > 0:
                                    has_forward_wp = True
                                    break
                        else:
                            has_forward_wp = True
                    except Exception:
                        has_forward_wp = True

                    if has_forward_wp:
                        dist = self._get_distance_to_stop_line(tl, vehicle_location)
                        is_opposite = self._is_traffic_light_opposite_side(tl)
                        result.update({
                            'in_range': True,
                            'distance': dist,
                            'state': str(tl.get_state()),
                            'affects_current_lane': True,
                            'traffic_light': tl,
                            'traffic_light_id': tl.id,
                            'is_opposite_side': is_opposite
                        })
                        return result
        except Exception:
            pass

        projected_route = self._get_projected_route_points(self.traffic_light_detection_distance)
        if not projected_route:
            return result

        traffic_lights = self.world.get_actors().filter('traffic.traffic_light*')

        for tl in traffic_lights:
            try:
                rough_dist = vehicle_location.distance(tl.get_location())
                if rough_dist > self.traffic_light_detection_distance + 40:
                    continue

                affected_wps = tl.get_affected_lane_waypoints()
                if not affected_wps:
                    continue

                stop_wps = []
                try:
                    stop_wps = tl.get_stop_waypoints()
                except Exception:
                    pass
                
                candidate_wps = list(affected_wps) + list(stop_wps)

                matched = False
                for cand_wp in candidate_wps:
                    for route_wp in projected_route:
                        if (route_wp.road_id == cand_wp.road_id and
                                route_wp.lane_id == cand_wp.lane_id):
                            dist = vehicle_location.distance(cand_wp.transform.location)
                            to_cand = cand_wp.transform.location - vehicle_location
                            dot = (vehicle_transform.get_forward_vector().x * to_cand.x +
                                   vehicle_transform.get_forward_vector().y * to_cand.y)
                            if dot > 0 and 0 < dist < result['distance']:
                                is_opposite = self._is_traffic_light_opposite_side(tl)
                                result.update({
                                    'in_range': True,
                                    'distance': dist,
                                    'state': str(tl.get_state()),
                                    'affects_current_lane': True,
                                    'traffic_light': tl,
                                    'traffic_light_id': tl.id,
                                    'is_opposite_side': is_opposite
                                })
                            matched = True
                            break
                    if matched:
                        break
            except Exception:
                continue

        return result

    def _get_distance_to_stop_line(self, traffic_light, vehicle_location):
        """Calculate distance to traffic light stop line"""
        ego_forward = self.vehicle.get_transform().get_forward_vector()
        ego_wp = self.world.map.get_waypoint(vehicle_location)

        def _best_from_waypoints(wps):
            candidates = []
            for wp in wps:
                to_wp = wp.transform.location - vehicle_location
                dist = vehicle_location.distance(wp.transform.location)
                if dist >= 0.5:
                    dot = (ego_forward.x * to_wp.x + ego_forward.y * to_wp.y) / dist
                    if dot <= 0.3:
                        continue
                if wp.road_id == ego_wp.road_id and wp.lane_id == ego_wp.lane_id:
                    candidates.append(dist)
            return min(candidates) if candidates else None

        def _best_from_waypoints_loose(wps):
            candidates = []
            for wp in wps:
                to_wp = wp.transform.location - vehicle_location
                dist = vehicle_location.distance(wp.transform.location)
                if dist < 0.5:
                    candidates.append(dist)
                    continue
                dot = (ego_forward.x * to_wp.x + ego_forward.y * to_wp.y) / dist
                if dot > 0.3:
                    candidates.append(dist)
            return min(candidates) if candidates else None

        try:
            stop_wps = traffic_light.get_stop_waypoints()
            if stop_wps:
                d = _best_from_waypoints(stop_wps)
                if d is not None:
                    return d
                d = _best_from_waypoints_loose(stop_wps)
                if d is not None:
                    return d
        except Exception:
            pass

        try:
            affected_wps = traffic_light.get_affected_lane_waypoints()
            if affected_wps:
                d = _best_from_waypoints(affected_wps)
                if d is not None:
                    return d
                d = _best_from_waypoints_loose(affected_wps)
                if d is not None:
                    return d
        except Exception:
            pass

        return vehicle_location.distance(traffic_light.get_location())
    
    def _is_vehicle_past_traffic_light(self, vehicle_transform, traffic_light):
        """Check if vehicle passed traffic light"""
        vehicle_location = vehicle_transform.location
        light_location = traffic_light.get_location()
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        vehicle_to_light = carla.Vector3D(
            light_location.x - vehicle_location.x,
            light_location.y - vehicle_location.y,
            0
        )
        
        dot_product = (vehicle_forward.x * vehicle_to_light.x + 
                      vehicle_forward.y * vehicle_to_light.y)
        
        return dot_product <= 0

    def _calculate_traffic_light_reward(self, throttle):
        """Compute traffic light compliance reward"""
        if not self.current_traffic_light_info['in_range'] or not self.current_traffic_light_info['affects_current_lane']:
            return 0.0
        
        distance = self.current_traffic_light_info['distance']
        light_state = self.current_traffic_light_info['state']
        current_speed_kmh = self.vehicle.get_speed()
        current_speed_ms = current_speed_kmh / 3.6
        
        if 'Red' in light_state or 'Yellow' in light_state:
            if distance <= 15.0 and distance >= 4.0:
                if current_speed_ms > 1.0:
                    if throttle >= 0:
                        return -2.0
                    else:
                        return -1.0
                else:
                    if throttle <= 0:
                        return 2.0
                    else:
                        return -2.0
            return 0.0
        
        elif 'Green' in light_state:
            if distance <= 15.0 and distance >= 4.0 and current_speed_ms > 1.0:
                return 1.0
            return 0.0
        
        return 0.0

    # ============================================================================
    # COLLISION AVOIDANCE: Time-to-collision (TTC) computation
    # ============================================================================

    def _calculate_ttc_with_actors(self):
        """Compute time-to-collision with nearby vehicles"""
        ttc_values = []
        ego_transform = self.vehicle.get_transform()
        ego_velocity = self.vehicle.get_velocity()
        ego_waypoint = self.world.map.get_waypoint(ego_transform.location)
        
        for bg_vehicle in self.traffic_flow_vehicles:
            if bg_vehicle.is_alive:
                try:
                    bg_transform = bg_vehicle.get_transform()
                    bg_velocity = bg_vehicle.get_velocity()
                    bg_waypoint = self.world.map.get_waypoint(bg_transform.location)
                    
                    if self._is_relevant_vehicle_for_ttc(ego_waypoint, bg_waypoint, ego_transform, bg_transform):
                        ttc = calculate_ttc(ego_transform, ego_velocity, bg_transform, bg_velocity)
                        if ttc != float('inf'):
                            ttc_values.append(ttc)
                except:
                    continue
        
        return ttc_values

    def _is_relevant_vehicle_for_ttc(self, ego_waypoint, bg_waypoint, ego_transform, bg_transform):
        """Determine if background vehicle is relevant for TTC calculation"""
        try:
            distance = ego_transform.location.distance(bg_transform.location)
            if distance > 100.0:
                return False
            
            ego_forward = ego_transform.get_forward_vector()
            to_bg_vehicle = bg_transform.location - ego_transform.location
            
            to_bg_vehicle_normalized = carla.Vector3D(
                to_bg_vehicle.x / distance,
                to_bg_vehicle.y / distance,
                0
            )
            
            dot_product = (ego_forward.x * to_bg_vehicle_normalized.x + 
                        ego_forward.y * to_bg_vehicle_normalized.y)
            
            if dot_product <= 0.3:
                return False
            
            is_ego_in_junction = ego_waypoint.is_junction
            is_bg_in_junction = bg_waypoint.is_junction
            
            if is_ego_in_junction and is_bg_in_junction:
                if distance <= 20.0 and dot_product > 0.2:
                    return True
            
            elif is_ego_in_junction and not is_bg_in_junction:
                if distance <= 20.0 and dot_product > 0.4:
                    return True
            
            elif not is_ego_in_junction and is_bg_in_junction:
                if distance <= 20.0 and dot_product > 0.5:
                    return True
            
            else:
                if ego_waypoint.road_id != bg_waypoint.road_id:
                    return False
                
                ego_lane_id = ego_waypoint.lane_id
                bg_lane_id = bg_waypoint.lane_id
                
                if ego_lane_id == 0 or bg_lane_id == 0:
                    if distance <= 80.0 and dot_product > 0.6:
                        return True
                else:
                    if ((ego_lane_id > 0 and bg_lane_id > 0) or 
                        (ego_lane_id < 0 and bg_lane_id < 0) or
                        abs(ego_lane_id - bg_lane_id) <= 1):
                        
                        if distance <= 80.0 and dot_product > 0.5:
                            return True
            
            ego_forward = ego_transform.get_forward_vector()
            bg_forward = bg_transform.get_forward_vector()
            
            direction_similarity = (ego_forward.x * bg_forward.x + 
                                ego_forward.y * bg_forward.y)
            
            if (direction_similarity > 0.7 and
                dot_product > 0.4 and
                distance <= 60.0):
                return True
            
            return False
            
        except Exception:
            return False

    # ============================================================================
    # REWARD CALCULATION: Curriculum learning with multi-stage rewards
    # ============================================================================

    def _calculate_curriculum_reward(self, base_reward, throttle):
        """Calculate curriculum reward normalized across phases"""
        current_phase = self._get_current_learning_phase()

        total_raw = base_reward
        if current_phase == 1:
            return total_raw

        if current_phase >= 2:
            total_raw += self._calculate_traffic_light_reward(throttle)

        if current_phase >= 3:
            total_raw += 0.10

        bounds = {1: (0.0, 1.0),
                2: (-2.0, 3.0),
                3: (-2.0, 3.0)}

        r_min, r_max = bounds[current_phase]
        r01 = (total_raw - r_min) / (r_max - r_min + 1e-8)
        r01 = np.clip(r01, 0.0, 1.0)

        reward_net = 2.0 * r01 - 1.0

        return total_raw

    # ============================================================================
    # ROUTE MANAGEMENT: Route generation and waypoint handling
    # ============================================================================

    def new_route(self):
        """Generate new navigation route and spawn traffic if needed"""
        current_phase = self._get_current_learning_phase()
        should_have_traffic = self.activate_traffic_flow and current_phase >= 1
        should_have_pedestrians = self.activate_pedestrians and current_phase >= 2
        
        if hasattr(self, 'traffic_flow_vehicles') and self.traffic_flow_vehicles:
            for bg_veh in list(self.traffic_flow_vehicles):
                if bg_veh.is_alive:
                    bg_veh.destroy()
            self.traffic_flow_vehicles.clear()
        
        if should_have_pedestrians:
            self._cleanup_pedestrians()

        self._cleanup_remaining_vehicles()

        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.set_simulate_physics(False)

        if not self.eval:
            if self.episode_idx % 2 == 0 and self.num_routes_completed == -1:
                spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(intersection_routes)]
            else:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
        else:
            spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(eval_routes)]
        
        route_length = 1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)

        self.distance_from_center_history = deque(maxlen=30)
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        
        self.vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)

        if should_have_traffic and hasattr(self, 'traffic_manager') and self.traffic_manager is not None:
            self._spawn_background_vehicles()
        
        if should_have_pedestrians:
            self._spawn_pedestrians()

    def _get_projected_route_points(self, max_distance=50.0):
        """Get waypoints on planned route ahead of vehicle"""
        projected = []
        vehicle_location = self.vehicle.get_transform().location

        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            wp, _ = self.route_waypoints[i]
            if vehicle_location.distance(wp.transform.location) > max_distance:
                break
            projected.append(wp)

        if projected:
            last_wp = projected[-1]
            for _ in range(int(max_distance)):
                next_wps = last_wp.next(2.0)
                if not next_wps:
                    break
                last_wp = next_wps[0]
                if vehicle_location.distance(last_wp.transform.location) > max_distance:
                    break
                projected.append(last_wp)

        return projected

    def _update_anticipated_maneuver(self, transform):
        """Plan maneuver ahead of time at lookahead distance"""
        current_location = vector(transform.location)
        max_lookahead = min(30, len(self.route_waypoints) - self.current_waypoint_index - 1)
        anticipation_distance = max(self.anticipation_distance, 20.0)

        for i in range(1, max_lookahead + 1):
            future_waypoint_index = self.current_waypoint_index + i
            if future_waypoint_index >= len(self.route_waypoints):
                break

            future_waypoint, future_maneuver = self.route_waypoints[future_waypoint_index]
            future_location = vector(future_waypoint.transform.location)
            distance_to_future_wp = np.linalg.norm(current_location - future_location)

            if (future_maneuver in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]
                and distance_to_future_wp <= anticipation_distance):
                self.current_road_maneuver = future_maneuver
                break

    # ============================================================================
    # ACTOR MANAGEMENT: Spawn and cleanup of background traffic and pedestrians
    # ============================================================================

    def _spawn_background_vehicles(self):
        """Spawn background vehicles using batch commands"""
        spawn_points = self.world.get_map().get_spawn_points()
        map_scale_factor = max(1.0, len(spawn_points) / 80.0)
        number_of_vehicles = min(len(spawn_points), int(self.tf_num * map_scale_factor))

        ego_location = self.vehicle.get_location()
        safe_spawn_points = []
        
        for spawn_point in spawn_points:
            distance_to_ego = spawn_point.location.distance(ego_location)
            if distance_to_ego > 30.0:
                safe_spawn_points.append(spawn_point)

        if not safe_spawn_points:
            return

        random.shuffle(safe_spawn_points)

        batch_commands = []
        for i in range(min(number_of_vehicles, len(safe_spawn_points))):
            blueprint_name = random_choice_from_category(class_blueprint, 'car')
            if blueprint_name is None:
                continue
            vehicle_bp = self.world.get_blueprint_library().find(blueprint_name)
            if vehicle_bp.has_attribute('color'):
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)
            sp = safe_spawn_points[i]
            batch_commands.append(
                carla.command.SpawnActor(vehicle_bp, sp)
                    .then(carla.command.SetAutopilot(
                        carla.command.FutureActor, True,
                        self.traffic_manager.get_port()))
            )

        results = self.client.apply_batch_sync(batch_commands, True)
        spawned_vehicles = 0
        for result in results:
            if not result.error:
                actor = self.world.get_actor(result.actor_id)
                if actor is not None:
                    self.traffic_flow_vehicles.append(actor)
                    spawned_vehicles += 1
    
    def _spawn_pedestrians(self):
        """Spawn pedestrians with control"""
        try:
            walker_blueprints = list(self.world.get_blueprint_library().filter('walker.pedestrian.*'))
            if not walker_blueprints:
                return
            
            spawn_points = []
            for i in range(self.pedestrian_num * 5):
                try:
                    spawn_point = carla.Transform()
                    loc = self.world.get_random_location_from_navigation()
                    if loc is None:
                        continue
                    
                    spawn_point.location = loc
                    
                    if self.vehicle:
                        vehicle_location = self.vehicle.get_location()
                        if spawn_point.location.distance(vehicle_location) < 20.0:
                            continue
                    
                    spawn_points.append(spawn_point)
                    
                except Exception:
                    continue
            
            batch_commands = []
            for i in range(min(self.pedestrian_num, len(spawn_points))):
                walker_bp = random.choice(walker_blueprints)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                
                batch_commands.append(carla.command.SpawnActor(walker_bp, spawn_points[i]))
            
            results = self.client.apply_batch_sync(batch_commands, True)
            walker_ids = []
            
            for i, result in enumerate(results):
                if not result.error:
                    walker_ids.append(result.actor_id)
            
            controller_batch = []
            controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            
            for walker_id in walker_ids:
                controller_batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id))
            
            controller_results = self.client.apply_batch_sync(controller_batch, True)
            
            world_actors = self.world.get_actors()
            
            for i, (walker_id, controller_result) in enumerate(zip(walker_ids, controller_results)):
                if controller_result.error:
                    try:
                        walker = world_actors.find(walker_id)
                        if walker:
                            walker.destroy()
                    except:
                        pass
                    continue
                
                try:
                    walker = world_actors.find(walker_id)
                    controller = world_actors.find(controller_result.actor_id)
                    
                    if walker and controller:
                        pedestrian = Pedestrian(
                            world=self.world,
                            walker_actor=walker,
                            controller=controller,
                            speed=np.random.uniform(1.0, 2.0)
                        )
                        
                        if pedestrian.is_alive:
                            self.pedestrians.append(pedestrian)
                        
                except Exception:
                    continue
            
        except Exception:
            self._cleanup_pedestrians()
    
    def _cleanup_remaining_vehicles(self):
        """Destroy all extraneous vehicles except ego"""
        try:
            all_vehicles = self.world.get_actors().filter('vehicle.*')
            ego_id = self.vehicle.id if hasattr(self.vehicle, 'id') else None
            
            for vehicle in all_vehicles:
                if vehicle.id != ego_id:
                    try:
                        vehicle.destroy()
                    except:
                        pass
                        
        except Exception:
            pass
    
    def _cleanup_pedestrians(self):
        """Destroy all pedestrians"""
        try:
            if hasattr(self, 'pedestrians') and self.pedestrians:
                for pedestrian in list(self.pedestrians):
                    try:
                        if hasattr(pedestrian, 'destroy'):
                            pedestrian.destroy()
                    except Exception:
                        pass
                
                self.pedestrians.clear()
            
        except Exception:
            pass
    
    def _restart_carla_actor(self):
        """Restart ego vehicle and sensors"""
        if self.world is not None:
            self.world.destroy()
            time.sleep(0.5)
        
        try:
            spawn_points = self.world.get_map().get_spawn_points()
            for spawn_point in spawn_points:
                try:
                    self.vehicle = Vehicle(self.world, spawn_point,
                                        on_collision_fn=lambda e: self._on_collision(e),
                                        on_invasion_fn=lambda e: self._on_invasion(e),
                                        is_ego=True)
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("Failed spawning vehicle at any spawn point")
            
            if self.activate_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((self.viewer_res[0], self.viewer_res[1]), 
                                                       pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(self.viewer_res[0], self.viewer_res[1])
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            if self.activate_spectator:
                self.camera = Camera(self.world, self.viewer_res[0], self.viewer_res[1],
                                    transform=sensor_transforms["spectator"],
                                    attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
            if self.activate_front_rgb:
                self.front_rgb_camera = Camera(
                    self.world,
                    width=config.RGB_CAMERA_WIDTH,
                    height=config.RGB_CAMERA_HEIGHT,
                    transform=sensor_transforms["front"],
                    attach_to=self.vehicle,
                    on_recv_image=lambda e: self._set_front_rgb_data(e))
                os.makedirs("output_front_rgb", exist_ok=True)

            self.traffic_flow_vehicles = []
            self.pedestrians = []
            time.sleep(0.5)
        
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed restarting CARLA actor: {e}")

    # ============================================================================
    # CURRICULUM LEARNING: Learning phase management and progress tracking
    # ============================================================================

    def _get_current_learning_phase(self):
        """Get curriculum learning phase based on global step counter"""
        if CarlaRouteEnv.global_step_counter < self.PHASE_1_THRESHOLD:
            return 1
        elif CarlaRouteEnv.global_step_counter < self.PHASE_2_THRESHOLD:
            return 2
        else:
            return 3
    
    def _update_global_step_counter(self):
        """Increment global step counter"""
        CarlaRouteEnv.global_step_counter += 1

    # ============================================================================
    # FRONT CAMERA & VLM: Front RGB image processing for VLM integration
    # ============================================================================
    
    def _process_front_rgb_image(self, image):
        """Process front camera image and optionally save to disk"""
        if isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]
        
        if self.save_vlm_image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = f"output_front_rgb/front_rgb_{timestamp}.png"
            cv2.imwrite(file_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        self.front_rgb_image = image_array
        return image_array

    # ============================================================================
    # DEBUG 
    # ============================================================================

    def _debug_vehicle_count(self):
        """Log vehicle count and status for debugging"""
        try:
            all_vehicles = self.world.get_actors().filter('vehicle.*')
            ego_id = self.vehicle.id if hasattr(self.vehicle, 'id') else None
            
            for i, vehicle in enumerate(all_vehicles):
                is_ego = vehicle.id == ego_id
                is_in_list = any(bg.id == vehicle.id for bg in self.traffic_flow_vehicles if bg.is_alive)
                vehicle_type = "ego" if is_ego else ("traffic" if is_in_list else "unknown")
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5 * 3.6
                
        except Exception:
            pass
