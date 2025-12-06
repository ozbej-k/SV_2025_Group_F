"""
Perception module that:
- transforms world-space stimuli to fish-local space
- filters by FOV
- computes mu (direction)
- computes solid angles via the solid_angle functions
- computes wall proximity, wall directions
- returns structured perception output
"""

import config
import numpy as np
from geometry.transforms import world_to_local, rotation_matrix_z
from geometry.fish_body import make_fish_box_vertices, make_box_tri_faces
from geometry.solid_angle import disc, polyhedron
from world import Fish, Tank, Spot
from typing import List

def vec2_to_vec3(p):
    """Convert a 2D world point to 3D by adding z=0."""
    return np.array([p[0], p[1], 0.0], dtype=float)

def compute_mu(v_local):
    """Return horizontal angle (relative direction)."""
    return np.arctan2(v_local[1], v_local[0])

def in_field_of_view(v_local):
    """
    Check if local-space vector is inside the fish's 270 deg FOV.
    """
    angle = np.arctan2(v_local[1], v_local[0])  # angle in horizontal plane

    return abs(angle) <= config.FOV_HALF * np.pi / 180 # convert to radians

def perceive(fish : Fish, fishies : List[Fish], spots : List[Spot], tank : Tank):
    """
    fish_obj: Fish instance (world position, orientation)
    other_fish_list: list of Fish instances (other agents)
    spot_list: list of Spot instances
    tank: Tank instance

    Returns perception dictionary with keys: 'fish', 'spots', 'wall_state'
    """
    perception = {
        'fish': [],
        'spots': [],
        'wall_state': {},
        'under_spot': False,
    }

    eye_pos_world = vec2_to_vec3(fish.position)  # assume eye at z=0 (same plane)
    fish_orientation = fish.orientation

    # 1) Other fish
    verts_local_other = make_fish_box_vertices(config.FISH_LENGTH, config.FISH_WIDTH, config.FISH_HEIGHT)
    fish_faces = make_box_tri_faces()
    for other in fishies:
        if other is fish:
            continue

        center_world = vec2_to_vec3(other.position)
        local_vec = world_to_local(center_world, eye_pos_world, fish_orientation)

        if not in_field_of_view(local_vec):
            continue

        mu = compute_mu(local_vec)

        # obtain other fish vertices in world coords
        R_other = rotation_matrix_z(other.orientation)
        verts_world = [np.array(vec2_to_vec3(other.position) + np.dot(R_other, v)) for v in verts_local_other]

        # convert world verts to eye-local coordinates
        verts_eye_local = [world_to_local(v, eye_pos_world, fish_orientation) for v in verts_world]

        A = polyhedron.solid_angle(verts_eye_local, fish_faces)
        
        perception['fish'].append({'id': other.id, 'mu': float(mu), 'A': float(A)})

    # 2) Spots
    for spot in spots:
        center_world = np.array([spot.x, spot.y, spot.height])
        local_vec = world_to_local(center_world, eye_pos_world, fish_orientation)

        if not in_field_of_view(local_vec):
            continue
        
        if np.linalg.norm(eye_pos_world[:2] - center_world[:2]) < spot.radius:
            perception["under_spot"] = True
        
        mu = compute_mu(local_vec)

        A = disc.solid_angle(center_world - eye_pos_world, np.array([0.0, 0.0, 1.0]), spot.radius)

        perception['spots'].append({'id': spot.id, 'mu': float(mu), 'A': float(A)})

    # 3) Wall state
    d_nearest, mu_w1_world, mu_w2_world = tank.tangent_wall_directions(fish.position)
    near_wall = d_nearest < config.PDF_DW

    if near_wall:
        mu_w1 = (mu_w1_world - fish_orientation + np.pi) % (2*np.pi) - np.pi
        mu_w2 = (mu_w2_world - fish_orientation + np.pi) % (2*np.pi) - np.pi
    else:
        mu_w1 = None
        mu_w2 = None

    perception['wall_state'] = {
        'near_wall': near_wall,
        'mu_w1': mu_w1,
        'mu_w2': mu_w2,
        'distance': float(d_nearest)
    }

    return perception
