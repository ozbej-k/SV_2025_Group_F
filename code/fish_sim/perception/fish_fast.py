"""Fast analytic fish perception model.

This module approximates mesh based solid angle for fish using
double pyramid fish body mesh.
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
    "fish_height": 0.010,

    # Global scale factor for apparent size, calibrated vs mesh
    "size_gain": 1.0,
}




def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


_INNER_DEG = 11.0
_MID_DEG = 24.0
_BACK_START_DEG = 166.5
_INNER_BUMP_AMP = 0.02      # relative hill height at 0
_INNER_BASE_SCALE = 1    # lower whole inner band (edges and center) a bit

_PLATEAU = 0.0005317378200909325
_COEF_MID = np.array([
    5.97126498e-07,
    4.88609916e-21,
    4.62837630e-04,
])
_COEF_FOURIER_OUTER = np.array([
    1.28685448e-03,
   -7.32868931e-04,
   -9.08284651e-05,
   -1.49532151e-05,
   -1.23054725e-06,
])
_COEF_BACK = np.array([
   -4.89844830e-08,
   -4.69172755e-07,
    5.78462404e-04,
])

# Reference distance used when fitting the above coefficients (meters)
_REFERENCE_DISTANCE = 0.3


def _apparent_size_at_reference_distance(body_view_angle: float) -> float:
    """Full combined A(θ) at the reference distance using fitted model.

    This encodes the plateau, mid quadratic band, outer Fourier band,
    and back-facing quadratic override from the calibration script.
    """

    theta = float(body_view_angle)
    theta_deg = np.degrees(theta)
    abs_theta_deg = abs(theta_deg)

    # Back-facing
    if abs_theta_deg >= _BACK_START_DEG:
        delta = 180.0 - abs_theta_deg
        if delta < 0.0:
            delta = 0.0
        a2, a1, a0 = _COEF_BACK
        A_back = (a2 * delta + a1) * delta + a0
        return max(A_back, 0.0)

    # Reduced angle in [-90, 90] so central structure repeats
    phi_deg = ((theta_deg + 90.0) % 180.0) - 90.0
    abs_phi = abs(phi_deg)

    # Inner plateau <= 11 degrees
    if abs_phi <= _INNER_DEG:
        # Small symmetric hill, matching back region shape
        t = abs_phi / _INNER_DEG
        bump = 1.0 + _INNER_BUMP_AMP * (1.0 - t * t)
        return (_PLATEAU * _INNER_BASE_SCALE) * bump

    # Mid band 11 to 24 degrees
    if abs_phi <= _MID_DEG:
        a2, a1, a0 = _COEF_MID
        td = phi_deg
        return (a2 * td + a1) * td + a0

    # Outer band 
    a0 = _COEF_FOURIER_OUTER[0]
    val = a0
    for n in range(1, len(_COEF_FOURIER_OUTER)):
        val += _COEF_FOURIER_OUTER[n] * np.cos(2.0 * n * theta)
    return max(val, 0.0)


def fish_apparent_size_from_mesh(
    distance: float,
    body_view_angle: float,
    size_gain: float = 1.0,
) -> float:
    """
    Fast analytic apparent size using calibrated full model.
    """

    if distance <= 0.0:
        return 0.0

    A_theta = _apparent_size_at_reference_distance(body_view_angle)
    if A_theta <= 0.0:
        return 0.0

    scale = (_REFERENCE_DISTANCE / float(distance)) ** 2
    return size_gain * A_theta * scale


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

        # Relative orientation between fish (body axis vs line-of-sight)
        body_view_angle = _wrap_angle(other.orientation - bearing)

        # Apparent size using calibrated analytic full model
        A = fish_apparent_size_from_mesh(
            distance=d,
            body_view_angle=body_view_angle,
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
