import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
"""Fast analytic fish perception model.

This module approximates mesh based solid angle for fish using
double pyramid fish body mesh.
"""

import numpy as np


FOV_HALF_RAD = 135 * np.pi / 180
_INNER_DEG = 11.0
_MID_DEG = 24.0
_BACK_START_DEG = 166.5
_INNER_BUMP_AMP = 0.02      # relative hill height at 0
_INNER_BASE_SCALE = 1    # lower whole inner band (edges and center) a bit

_PLATEAU = 0.0005317378200909325
_COEF_MID = np.array([5.97126498e-07, 4.88609916e-21, 4.62837630e-04])
_COEF_FOURIER_OUTER = np.array([1.28685448e-03, -7.32868931e-04, -9.08284651e-05, -1.49532151e-05, -1.23054725e-06])
_COEF_BACK = np.array([-4.89844830e-08, -4.69172755e-07, 5.78462404e-04])
_COEF_SCALE = np.array([-1.90819551e+13, 1.30318296e+12, -3.55587092e+10, 4.92003046e+08, -3.56959358e+06, 1.17554913e+04])
_REFERENCE_DISTANCE = 0.3

def _apparent_size_at_reference_angle(relative_orientation: float) -> float:
    """
    Compute apparent size for any relative orientation angle (radians),
    automatically handling angles outside [-pi, pi].
    """
    # wrap angle first
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

def fish_apparent_size_from_mesh(distance, relative_orientation):
    if distance <= 0.0:
        return 0.0

    A_theta = _apparent_size_at_reference_angle(relative_orientation)
    if A_theta <= 0.0:
        return 0.0
    
    if distance < 0.02:
        scale = np.polyval(_COEF_SCALE, distance)
    else:
        scale = (_REFERENCE_DISTANCE / float(distance)) ** 2
    return A_theta * scale


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def perceive_fish_fast(observer, others):
    ox, oy, _ = observer.position
    o_theta = observer.orientation

    perceived = []

    cos_theta = np.cos(-o_theta)
    sin_theta = np.sin(-o_theta)

    for fish in others:
        dx = fish.position[0] - ox
        dy = fish.position[1] - oy

        x_local = cos_theta * dx - sin_theta * dy
        y_local = sin_theta * dx + cos_theta * dy
        distance = np.hypot(x_local, y_local)

        mu = np.arctan2(y_local, x_local)
        if abs(mu) > FOV_HALF_RAD:
            continue

        solid_angle = fish_apparent_size_from_mesh(distance, fish.orientation - observer.orientation)

        if solid_angle <= 0.0:
            continue

        perceived.append({
            "mu": mu,
            "A": min(solid_angle, 2*np.pi),
        })

    return perceived




class Fish:
    length = 0.035
    width = 0.01
    height = 0.01
    l, w, h = length / 2, width / 2, height / 2
    base_fish = np.array([
        [0, 0, h],
        [0, 0, -h],
        [l, 0, 0],
        [0, w, 0],
        [0, -w, 0],
        [-l, 0, 0],
    ])

    def __init__(self, x, y, orientation):
        self.position = np.array([x, y, 0])
        self.orientation = orientation
        self.speed = np.zeros(2)

def fish_vertices(fish: Fish):
    c, s = np.cos(fish.orientation), np.sin(fish.orientation)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    vertices = (rz @ Fish.base_fish.T).T + fish.position
    return vertices

def spherical_arc(u, v):
    # angle between two points on unit circle (same as doing it for spherical points)
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

def solid_angle_triangle(u, v, w):
    # solid angle of triangle given 3 points on unit circle
    a = spherical_arc(v, w)
    b = spherical_arc(u, w)
    c = spherical_arc(u, v)
    s = (a + b + c) / 2
    return 4 * np.arctan(np.sqrt(
        np.tan(s / 2) * np.tan((s - a) / 2) * np.tan((s - b) / 2) * np.tan((s - c) / 2)
    ))

def normalize(v):
    return v / np.linalg.norm(v)

def fish_solid_angle(fish_focal: Fish, fish_observed: Fish):
    # only works for fish at z = 0
    vertices = fish_vertices(fish_observed) - fish_focal.position
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    unit = vertices / norms

    direction = unit.sum(axis=0)
    direction /= np.linalg.norm(direction)

    left_right = vertices @ np.array([-direction[1], direction[0], 0])
    L = unit[np.argmin(left_right)]
    R = unit[np.argmax(left_right)]
    T = unit[0]
    B = unit[1]

    solid_angle = (
        solid_angle_triangle(L, T, R) +
        solid_angle_triangle(L, B, R)
    )
    silhouette = np.array([L, T, R, B])
    return solid_angle, silhouette



focal = Fish(0, 0, 0)
distances = np.linspace(0.00001, 0.02, 100)
solid_angles_distance = []
solid_angles_distance1 = []
solid_angles_distance2 = []

# for d in distances:
#     other_fish = Fish(d, 0, 0)
#     omega, _ = fish_solid_angle(focal, other_fish)
#     omega1 = perceive_fish_fast(focal, [other_fish])
#     penis=False
#     omega2 = perceive_fish_fast(focal, [other_fish])
#     penis=True
#     solid_angles_distance.append(omega)
#     solid_angles_distance1.append(omega1[0]["A"])
#     solid_angles_distance2.append(omega2[0]["A"])

# solid_angles_distance = np.array(solid_angles_distance)
# solid_angles_distance1 = np.array(solid_angles_distance1)
# solid_angles_distance2 = np.array(solid_angles_distance2)
# print(solid_angles_distance1)
# print(solid_angles_distance2)
# coefficients = np.polyfit(distances, solid_angles_distance / solid_angles_distance2, 5)
# print(coefficients)

# plt.plot(distances, solid_angles_distance2 * np.polyval(coefficients, distances), color = "orange")
# plt.plot(distances, solid_angles_distance)
# plt.plot(distances, solid_angles_distance1, color="red")
# plt.plot(distances, solid_angles_distance2, color="green")
# plt.xlabel("Distance from focal fish")
# plt.ylabel("Solid angle (sr)")
# plt.title("Solid angle vs distance")
# plt.grid(True)
# plt.show()
# exit()

















focal = Fish(0, 0, 0)

# Define distances and rotation angles
distances = np.linspace(0.0001, 0.03, 50)
# plt.plot(scale)
# plt.show()

angles = np.linspace(0, 2*np.pi, 50)

# Prepare lists to store data
X, Y, Z_s, Z_p, Zy = [], [], [], [], []

# Compute solid angles for all combinations
for theta in angles:
    for d in distances:
        other_fish = Fish(d, 0, theta)
        omega, _ = fish_solid_angle(focal, other_fish)
        omega1 = perceive_fish_fast(focal, [other_fish])[0]["A"]
        penis = False
        omega2 = perceive_fish_fast(focal, [other_fish])[0]["A"]
        penis = True

        X.append(d)
        Y.append(np.degrees(theta))
        Z_s.append(omega)
        Z_p.append(omega1)
        Zy.append(omega2)
Z_s, Z_p, Zy = np.array(Z_s), np.array(Z_p), np.array(Zy)

# 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z_s, color='blue', label='fish_solid_angle', alpha=0.6)
ax.scatter(X, Y, Z_p, color='red', label='perceive_fish_fast', alpha=0.6)
# ax.scatter(X, Y, Zy, color='green', label='perceive_fish_fast', alpha=0.6)

ax.set_xlabel("Distance")
ax.set_ylabel("Rotation angle (degrees)")
ax.set_zlabel("Solid angle (sr)")
ax.set_title("Solid angle vs Distance & Rotation (Scatter)")
ax.legend()
plt.show()

exit()

# import numpy as np
# import matplotlib.pyplot as plt


angles = np.linspace(0, 2*np.pi, 100)
solid_angles_rotation = []
solid_angles_rotation1 = []
fixed_distance = 0.1

for theta in angles:
    other_fish = Fish(fixed_distance, 0, theta)
    omega, _ = fish_solid_angle(focal, other_fish)
    solid_angles_rotation.append(omega)
    omega1 = perceive_fish_fast(focal, [other_fish])
    solid_angles_rotation1.append(omega1[0]["A"])

plt.figure()
plt.plot(np.degrees(angles), solid_angles_rotation)
plt.plot(np.degrees(angles), solid_angles_rotation1, color="red")
plt.xlabel("Rotation angle of observed fish (degrees)")
plt.ylabel("Solid angle (sr)")
plt.title("Solid angle vs rotation")
plt.grid(True)
plt.show()
