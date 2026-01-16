import config
import numpy as np
from typing import List

from .fish_angle import fish_solid_angle
from .disc_angle import disc_solid_angle

from world import Fish, Tank, Spot

FOV_HALF_RAD = config.FOV_HALF * np.pi / 180

def perceive(fish: Fish, fishies: List[Fish], spots: List[Spot], tank: Tank):
    perception = {
        "fish": [],
        "spots": [],
        "under_spot": False, 
        "wall_state": {},
    }

    # fish perception
    ox, oy = fish.position
    o_theta = fish.orientation

    cos_theta = np.cos(-o_theta)
    sin_theta = np.sin(-o_theta)

    for other_fish in fishies:
        if tank.ray_intersects_wall(fish.position, other_fish.position):
            continue

        dx = other_fish.position[0] - ox
        dy = other_fish.position[1] - oy

        x_local = cos_theta * dx - sin_theta * dy
        y_local = sin_theta * dx + cos_theta * dy
        distance = np.hypot(x_local, y_local)
        mu = np.arctan2(y_local, x_local)

        if abs(mu) > FOV_HALF_RAD:
            continue

        solid_angle = fish_solid_angle(distance, other_fish.orientation - fish.orientation)

        if solid_angle <= 0.0:
            continue

        perception["fish"].append({
            "id": other_fish.id,
            "mu": mu,
            "A": min(solid_angle, 2*np.pi),
        })

    # spot perception
    eye_pos_world = np.array([fish.position[0], fish.position[1], 0.0], dtype=float)
    fish_orientation = fish.orientation

    for spot in spots:
        
        if tank.ray_intersects_wall(fish.position, spot.position[:2]):
            continue  # Skip this spot if a wall is between fish and spot

        center_vec = spot.position - eye_pos_world

        mu = (np.arctan2(center_vec[1], center_vec[0]) - fish_orientation)
        if np.linalg.norm(center_vec) < spot.radius:
            perception["under_spot"] = True
        elif abs(mu) > config.FOV_HALF * np.pi / 180:
            continue
        
        r = np.linalg.norm(center_vec[:2])
        A = disc_solid_angle(r)

        perception["spots"].append({
            "id": spot.id,
            "mu": mu,
            "A": A,
        })

    # wall perception
    d_list, mu_w = tank.tangent_wall_directions(fish.position, fish.orientation)
    mu_w_relative = [(mu - fish.orientation + np.pi) % (2 * np.pi) - np.pi for mu in mu_w]

    perception["wall_state"] = {
        "mu_w": mu_w_relative,
        "distance": d_list,   
    }
    return perception
