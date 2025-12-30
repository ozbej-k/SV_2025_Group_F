"""
Perception module that:
- transforms world-space stimuli to fish-local space
- filters by FOV
- computes mu (direction)
- computes solid angles via the solid_angle functions
- computes wall proximity, wall directions
- returns structured perception output
"""

import numpy as np
import math
import logging
from geometry.vectors import vec2_to_vec3
from geometry.transforms import world_to_local
from perception.fov import in_field_of_view
from perception.direction import compute_mu
from geometry.fish_body import make_fish_box_vertices, make_box_tri_faces
from perception.solid_angle import polyhedron as poly_mod
from perception.solid_angle import disc as disc_mod
from geometry.wall_geometry import distance_to_walls_2d, wall_tangent_directions
from config.fish_params import FISH_LENGTH, FISH_WIDTH, FISH_HEIGHT, DW
#from config.fish_params import FALLBACK_PLANAR_SAMPLE_POINTS, SPOT_RADIUS, SPOT_HEIGHT

logger = logging.getLogger(__name__)

def perceive(fish_obj, other_fish_list, spot_list, tank):
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
        'wall_state': {}
    }

    eye_pos_world = vec2_to_vec3(fish_obj.position)  # assume eye at z=0 (same plane)
    fish_orientation = fish_obj.orientation

    # 1) Other fish
    for other in other_fish_list:
        if other is fish_obj:
            continue
        # compute reference point(s) for other fish: use center and vertices
        # compute center world position (x,y, z=0)
        center_world = vec2_to_vec3(other.position)

        local_vec = world_to_local(center_world, eye_pos_world, fish_orientation)
        # quick FOV check on center
        if not in_field_of_view(local_vec):
            continue

        mu = compute_mu(local_vec)

        # obtain other fish vertices in world coords
        verts_local_other = make_fish_box_vertices(FISH_LENGTH, FISH_WIDTH, FISH_HEIGHT)
        # translate vertices to the other's world position and rotate by other's orientation
        # first rotate vertices by other's orientation (world oriented)
        # create world positions of vertices
        from geometry.transforms import rotation_matrix_z
        R_other = rotation_matrix_z(other.orientation)
        verts_world = [np.array(vec2_to_vec3(other.position) + np.dot(R_other, v)) for v in verts_local_other]

        # convert world verts to eye-local coordinates
        verts_eye_local = [world_to_local(v, eye_pos_world, fish_orientation) for v in verts_world]

        # faces for box
        faces = make_box_tri_faces()
        # compute solid angle via polyhedron implementation
        A = poly_mod.compute_solid_angle_polyhedron(verts_eye_local, faces)
        
        perception['fish'].append({'id': other.id, 'mu': float(mu), 'A': float(A)})

    # 2) Spots
    for spot in spot_list:
        center_world = np.array([spot.x, spot.y, spot.height])
        local_vec = world_to_local(center_world, eye_pos_world, fish_orientation)
        if not in_field_of_view(local_vec):
            continue
        mu = compute_mu(local_vec)

        A = disc_mod.compute_solid_angle_disc(center_world - eye_pos_world, np.array([0.0, 0.0, 1.0]), spot.radius)

        perception['spots'].append({'id': spot.id, 'mu': float(mu), 'A': float(A)})

    # 3) Wall state
    d_nearest, nearest_wall = distance_to_walls_2d((fish_obj.position[0], fish_obj.position[1]), tank)
    near_wall = d_nearest < DW
    if near_wall:
        mu_w1_world, mu_w2_world = wall_tangent_directions(nearest_wall)
        # convert these world angles to fish-local relative angles:
        # relative_angle = mu_world - fish_orientation, normalized
        def rel(a):
            diff = a - fish_orientation
            # normalize to [-pi,pi]
            return (diff + math.pi) % (2*math.pi) - math.pi
        mu_w1 = rel(mu_w1_world)
        mu_w2 = rel(mu_w2_world)
    else:
        mu_w1 = None
        mu_w2 = None

    perception['wall_state'] = {
        'near_wall': bool(near_wall),
        'mu_w1': mu_w1,
        'mu_w2': mu_w2,
        'distance': float(d_nearest)
    }

    return perception
