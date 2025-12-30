import numpy as np

_INNER_DEG = 11.0
_MID_DEG = 24.0
_BACK_START_DEG = 166.5
_INNER_BUMP_AMP = 0.02 # relative hill height at 0
_INNER_BASE_SCALE = 1 # lower whole inner band (edges and center) a bit

_PLATEAU = 0.0005317378200909325
_COEF_MID = np.array([5.97126498e-07, 4.88609916e-21, 4.62837630e-04])
_COEF_FOURIER_OUTER = np.array([1.28685448e-03, -7.32868931e-04, -9.08284651e-05, -1.49532151e-05, -1.23054725e-06])
_COEF_BACK = np.array([-4.89844830e-08, -4.69172755e-07, 5.78462404e-04])
_COEF_SCALE = np.array([-1.90819551e+13, 1.30318296e+12, -3.55587092e+10, 4.92003046e+08, -3.56959358e+06, 1.17554913e+04])
_REFERENCE_DISTANCE = 0.3

def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _apparent_size_at_reference_angle(relative_orientation: float) -> float:
    theta = _wrap_angle(relative_orientation)
    theta_deg = np.degrees(theta)
    abs_theta_deg = abs(theta_deg)

    # Back-facing
    if abs_theta_deg >= _BACK_START_DEG:
        delta = 180.0 - abs_theta_deg
        delta = max(delta, 0.0)
        a2, a1, a0 = _COEF_BACK
        A_back = (a2 * delta + a1) * delta + a0
        return max(A_back, 0.0)

    # Reduce angle to [-90, 90] for central structure
    phi_deg = ((theta_deg + 90.0) % 180.0) - 90.0
    abs_phi = abs(phi_deg)

    # Inner plateau <= 11 degrees
    if abs_phi <= _INNER_DEG:
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

def fish_solid_angle(distance, relative_orientation):
    if distance <= 0.0:
        return 0.0

    A_theta = _apparent_size_at_reference_angle(relative_orientation)
    if A_theta <= 0.0:
        return 0.0
    
    if distance < 0.02:
        scale = np.polyval(_COEF_SCALE, distance)
    else:
        scale = (_REFERENCE_DISTANCE / float(distance)) ** 2
    return min(A_theta * scale, 2*np.pi)

def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi
