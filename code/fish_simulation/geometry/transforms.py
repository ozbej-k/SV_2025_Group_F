import numpy as np
#from .vectors import normalize

def rotation_matrix_z(theta):
    """
    Rotation matrix rotating a vector by +theta around Z (right-hand rule).
    Use R(-theta) to rotate world -> local when fish heading = theta.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def world_to_local(point, fish_position, fish_orientation):
    """
    Convert a 3D point (world) to fish-local coordinates where:
      - origin = fish_position
      - +x local = fish heading (fish_orientation)
      - +y local = left
      - +z local = up (same as world up)
    fish_orientation: angle in radians (world frame). 0 points along +x.
    """
    p = np.asarray(point, dtype=float) - np.asarray(fish_position, dtype=float)
    # rotate by -orientation to bring world forward to local +x
    R = rotation_matrix_z(-fish_orientation)
    return R.dot(p)
