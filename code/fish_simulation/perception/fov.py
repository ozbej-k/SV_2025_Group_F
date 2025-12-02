import numpy as np

FOV_HALF_ANGLE = 135 * np.pi / 180  # convert to radians

def in_field_of_view(v_local):
    """
    Check if local-space vector is inside the fish's 270 deg FOV.
    """
    angle = np.arctan2(v_local[1], v_local[0])  # angle in horizontal plane

    return abs(angle) <= FOV_HALF_ANGLE
