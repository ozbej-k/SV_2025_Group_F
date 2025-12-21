"""
Perception module that:
- transforms world-space stimuli to fish-local space
- filters by FOV
- computes mu (direction)
- computes apparent sizes for fish (fast analytic model)
- computes solid angles for spots
- computes wall proximity and escape directions
- returns structured perception output
"""

import config
import numpy as np
from typing import List

from geometry.transforms import world_to_local
from geometry.solid_angle import disc

from perception.fish_fast import perceive_fish_fast

from world import Fish, Tank, Spot

import time

def perceive_fish_mesh(fish: Fish, fishies: List[Fish]):
    """
    Original mesh-based fish perception.
    Returns list of {'id', 'mu', 'A'} dicts.
    """

    from geometry.transforms import rotation_matrix_z
    from geometry.fish_body import make_fish_box_vertices, make_box_tri_faces
    from geometry.solid_angle import polyhedron

    perceived = []

    eye_pos_world = vec2_to_vec3(fish.position)
    fish_orientation = fish.orientation

    verts_local_other = make_fish_box_vertices(
        config.FISH_LENGTH,
        config.FISH_WIDTH,
        config.FISH_HEIGHT,
    )
    fish_faces = make_box_tri_faces()

    for other in fishies:
        if other is fish:
            continue

        center_world = vec2_to_vec3(other.position)
        local_vec = world_to_local(center_world, eye_pos_world, fish_orientation)

        if not in_field_of_view(local_vec):
            continue

        mu = compute_mu(local_vec)

        R_other = rotation_matrix_z(other.orientation)
        verts_world = [
            vec2_to_vec3(other.position) + R_other @ v
            for v in verts_local_other
        ]

        verts_eye_local = [
            world_to_local(v, eye_pos_world, fish_orientation)
            for v in verts_world
        ]

        A = polyhedron.solid_angle(verts_eye_local, fish_faces)

        perceived.append({
            'id': other.id,
            'mu': float(mu),
            'A': float(A),
        })

    return perceived


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
    angle = np.arctan2(v_local[1], v_local[0])
    return abs(angle) <= config.FOV_HALF * np.pi / 180


def perceive(fish: Fish, fishies: List[Fish], spots: List[Spot], tank: Tank):
    """
    fish: focal Fish instance
    fishies: list of Fish instances (other agents)
    spots: list of Spot instances
    tank: Tank instance

    Returns perception dictionary with keys:
        'fish', 'spots', 'wall_state', 'under_spot'
    """

    perception = {
        'fish': [],
        'spots': [],
        'wall_state': {},
        'under_spot': False,
    }


    # 1 Fish perception, configurable


    mode = config.FISH_PERCEPTION_MODE

    if mode == "fast":
        perception['fish'] = perceive_fish_fast(
            focal_fish=fish,
            other_fish_list=fishies,
            params={
                "fov_half_angle": config.FOV_HALF * np.pi / 180,
                "perception_radius": config.FOV_DEGREES,
                "fish_length": config.FISH_LENGTH,
                "fish_width": config.FISH_WIDTH,
            },
        )


    elif mode == "mesh":
        perception['fish'] = perceive_fish_mesh(fish, fishies)

 


    elif mode == "both":
        start_fast = time.time()
        fish_fast = perceive_fish_fast(
            focal_fish=fish,
            other_fish_list=fishies,
            params={
                "fov_half_angle": config.FOV_HALF * np.pi / 180,
                "perception_radius": config.FOV_DEGREES,
                "fish_length": config.FISH_LENGTH,
                "fish_width": config.FISH_WIDTH,
            },
        )
        fast_time = time.time() - start_fast

        start_mesh = time.time()
        fish_mesh = perceive_fish_mesh(fish, fishies)
        mesh_time = time.time() - start_mesh

        print(f"Fast perception time: {fast_time:.6f} seconds")
        print(f"Mesh perception time: {mesh_time:.6f} seconds")

        perception['fish'] = fish_fast
        perception['fish_mesh_debug'] = fish_mesh

    else:
        raise ValueError(f"Unknown FISH_PERCEPTION_MODE: {mode}")


    # 2 Spot perception

    eye_pos_world = vec2_to_vec3(fish.position)
    fish_orientation = fish.orientation

    for spot in spots:
        center_world = np.array([spot.x, spot.y, spot.height])
        local_vec = world_to_local(center_world, eye_pos_world, fish_orientation)

        if not in_field_of_view(local_vec):
            continue

        # Check if fish is under spot
        if np.linalg.norm(eye_pos_world[:2] - center_world[:2]) < spot.radius:
            perception["under_spot"] = True

        mu = compute_mu(local_vec)

        A = disc.solid_angle(
            center_world - eye_pos_world,
            np.array([0.0, 0.0, 1.0]),
            spot.radius,
        )

        perception['spots'].append({
            'id': spot.id,
            'mu': float(mu),
            'A': float(A),
        })

    # 3 Wall perception

    d_nearest, mu_w1_world, mu_w2_world = tank.tangent_wall_directions(fish.position)
    near_wall = d_nearest < config.PDF_DW

    if near_wall:
        mu_w1 = (mu_w1_world - fish_orientation + np.pi) % (2 * np.pi) - np.pi
        mu_w2 = (mu_w2_world - fish_orientation + np.pi) % (2 * np.pi) - np.pi
    else:
        mu_w1 = None
        mu_w2 = None

    perception['wall_state'] = {
        'near_wall': near_wall,
        'mu_w1': mu_w1,
        'mu_w2': mu_w2,
        'distance': float(d_nearest),
    }

    return perception
