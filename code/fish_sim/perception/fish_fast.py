"""
Fast analytic fish perception model.

This module replaces mesh solid angle computation for fish
with approximation based on distance and relative orientation.
"""

import numpy as np



# Default perception params
DEFAULT_PARAMS = {
    # Field of view (270 degrees total)
    "fov_half_angle": 3 * np.pi / 4,   # -+135 degrees

    # Maximum perception distance for fish
    "perception_radius": 2.0,

    # Fish body dimensions (meters)
    "fish_length": 0.035,
    "fish_width": 0.010,

    # Global scale factor for apparent size, for tunning
    "size_gain": 1.08,
}




def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def fish_apparent_size(
    distance: float,
    rel_orientation: float,
    fish_length: float,
    fish_width: float,
    fish_height: float = 0.0115,
    size_gain: float = 1.0,
) -> float:
    """
    Compute apparent size A of a fish analytically.
    """

    # Effective projected width of an oriented ellipse
    sin_phi = np.sin(rel_orientation)
    cos_phi = np.cos(rel_orientation)

    w_eff = np.sqrt((fish_length * sin_phi) ** 2 +(fish_width * cos_phi) ** 2)
    
    #w_eff = np.abs(fish_length * np.sin(rel_orientation)) + np.abs(fish_width * np.cos(rel_orientation))
    
    # possible improvement
    #A = (w_eff * fish_height) / (distance * distance)
    
    theta_h = 2.0 * np.arctan2(w_eff, 2.0 * distance)

    h_eff = fish_height   # new param
    theta_v = 2.0 * np.arctan2(h_eff, 2.0 * distance)

    A = size_gain * theta_h * theta_v
    
    return A


# Public API
def perceive_fish_fast(
    focal_fish,
    other_fish_list,
    params: dict | None = None,
):
    """
    Fast fish perception function.
    
    Returns
        {
            'id': fish.id,
            'mu': direction angle,
            'A' : apparent size
        }
    """

    if params is None:
        params = DEFAULT_PARAMS
    else:
        # merge defaults with overrides
        merged = DEFAULT_PARAMS.copy()
        merged.update(params)
        params = merged

    fov_half = params["fov_half_angle"]
    max_dist = params["perception_radius"]
    L = params["fish_length"]
    W = params["fish_width"]
    size_gain = params["size_gain"]

    fx, fy = focal_fish.position
    f_theta = focal_fish.orientation

    perceived = []

    for other in other_fish_list:

        # Vector from focal to other
        dx = other.position[0] - fx
        dy = other.position[1] - fy

        # Distance check
        d = np.hypot(dx, dy)
        if d <= 0.0 or d > max_dist:
            continue

        bearing = np.arctan2(dy, dx)

        # Direction in focal fish frame
        cos_t = np.cos(-f_theta)
        sin_t = np.sin(-f_theta)

        x_local = cos_t * dx - sin_t * dy
        y_local = sin_t * dx + cos_t * dy

        mu = np.arctan2(y_local, x_local)

        # FOV check
        if abs(mu) > fov_half:
            continue

        # Relative orientation between fish
#        rel_phi = _wrap_angle(other.orientation - f_theta)
        
        body_view_angle = _wrap_angle(other.orientation - bearing)

        # Apparent size
        A = fish_apparent_size(
            distance=d,
            rel_orientation=body_view_angle,
#            rel_orientation=rel_phi,
            fish_length=L,
            fish_width=W,
            size_gain=size_gain,
        )

        if A <= 0.0:
            continue

        perceived.append({
            "id": other.id,
            "mu": mu,
            "A": A,
        })

    return perceived
