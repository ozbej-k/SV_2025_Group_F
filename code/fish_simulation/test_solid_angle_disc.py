'''Test script for solid angle computation of discs.'''

import numpy as np
from perception.solid_angle.disc import compute_solid_angle_disc

# analytic forward-facing test
d = 1.0
r = 0.5
center = np.array([d, 0.0, 0.0])         # disc at +x
normal = np.array([-1.0, 0.0, 0.0])     # disc faces the origin
Omega_num = compute_solid_angle_disc(center, normal, r, n_samples=256)
Omega_analytic = 2 * np.pi * (1.0 - d / np.sqrt(d * d + r * r))
print("analytic:", Omega_analytic, "numeric:", Omega_num, "diff:", Omega_num - Omega_analytic)

# floating spot typical case: disc above fish
center2 = np.array([0.3, 0.1, 0.05])     # fish at origin, spot at z=0.05
normal2 = np.array([0.0, 0.0, 1.0])      # normal up
Omega2 = compute_solid_angle_disc(center2, normal2, 0.10, n_samples=256)
print("floating spot Omega:", Omega2)
