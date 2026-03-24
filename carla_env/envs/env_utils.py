import glob
import os
import subprocess
import time
import gym
import pygame
import cv2
from pygame.locals import *
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

def _get_latest_front_rgb_image(self):

        image_folder = "output_front_rgb"
        image_files = glob.glob(os.path.join(image_folder, "*.png"))
        if not image_files:
            return None
        latest_image = max(image_files, key=os.path.getctime)
        return latest_image

def _draw_path(self, camera, image):
    """
        Draw a connected path from start of route to end using homography.
    """
    vehicle_vector = vector(self.vehicle.get_transform().location)
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    image_w = int(camera.actor.attributes['image_size_x'])
    image_h = int(camera.actor.attributes['image_size_y'])
    fov = float(camera.actor.attributes['fov'])
    for i in range(self.current_waypoint_index, len(self.route_waypoints)):
        waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
        waypoint_vector = vector(waypoint_location)
        if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
            continue
        K = build_projection_matrix(image_w, image_h, fov)
        x, y = get_image_point(waypoint_location, K, world_2_camera)
        if i == len(self.route_waypoints) - 1:
            color = (0, 0, 255)
        else:
            color = (144, 238, 144)
        image = cv2.circle(image, (int(x), int(y)), radius=3, color=color, thickness=-1)
    return image

def _set_viewer_image(self, image):
    self.viewer_image_buffer = image

def _get_viewer_image(self):
    while self.viewer_image_buffer is None:
        pass
    image = self.viewer_image_buffer.copy()
    self.viewer_image_buffer = None
    return image

def _set_front_rgb_data(self, image):
    self.front_rgb_image_buffer = image

def _get_front_rgb_data(self):
    while self.front_rgb_image_buffer is None:
        pass
    image = self.front_rgb_image_buffer.copy()
    self.front_rgb_image_buffer = None
    return image

def _on_collision(self, event):
    if get_actor_display_name(event.other_actor) != "Road":
        self.terminal_state = True
        self.collision_state = True
        print("0| Terminal:  Collision with {}".format(event.other_actor.type_id))
    if self.activate_render:
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

def _on_invasion(self, event):
    lane_types = set(x.type for x in event.crossed_lane_markings)
    text = ["%r" % str(x).split()[-1] for x in lane_types]
    if self.activate_render:
        self.hud.notification("Crossed line %s" % " and ".join(text))

def patch_env():
    from carla_env.envs.carla_route_env import CarlaRouteEnv
    CarlaRouteEnv._get_latest_front_rgb_image = _get_latest_front_rgb_image
    CarlaRouteEnv._draw_path = _draw_path
    CarlaRouteEnv._set_viewer_image = _set_viewer_image
    CarlaRouteEnv._get_viewer_image = _get_viewer_image
    CarlaRouteEnv._set_front_rgb_data = _set_front_rgb_data
    CarlaRouteEnv._get_front_rgb_data = _get_front_rgb_data
    CarlaRouteEnv._on_collision = _on_collision
    CarlaRouteEnv._on_invasion = _on_invasion