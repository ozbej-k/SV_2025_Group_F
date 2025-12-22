import numpy as np


# polynomial approximation of disc solid angle as a function
# of lateral distance r from the disc centre, calibrated for
# current config.SPOT_RADIUS and config.SPOT_HEIGHT.

_SPLIT_R = 0.2

_COEF_INNER = np.array([
	-3.87587831e09,
	 6.93476457e08,
	-4.67048261e07,
	 1.43186134e06,
	-1.62799523e04,
	-1.09626002e02,
	 3.46269359e00,
])

_COEF_OUTER = np.array([
	 8.56212948e03,
	-1.13892397e04,
	 6.10506356e03,
	-1.68781059e03,
	 2.55130684e02,
	-2.03971678e01,
	 7.22321383e-01,
])


def solid_angle_fast(center_vec):
	"""Fast approximation of disc solid angle seen from the origin.

	center_vec: 3D vector from eye to disc centre in eye-local coords.

	Returns approximate solid angle in steradians, fitted to
	mesh implementation for current spot geometry.
	"""

	v = np.asarray(center_vec, dtype=float)
	# Lateral distance in the horizontal plane
	r = float(np.sqrt(v[0] * v[0] + v[1] * v[1]))
	s = r * r

	if r <= _SPLIT_R:
		A = np.polyval(_COEF_INNER, s)
	else:
		A = np.polyval(_COEF_OUTER, s)

	# Guard against tiny negative values from polynomial approximation
	if A < 0.0:
		A = 0.0

	return float(A)

