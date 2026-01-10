
import matplotlib.pyplot as plt

import numpy as np

import config
from disc_mesh import solid_angle as disc_solid_angle
from scipy.optimize import curve_fit

R = config.SPOT_RADIUS
h = config.SPOT_HEIGHT

distance = np.linspace(0, 4, 2000)
A_mesh = np.zeros_like(distance)
A_fast = np.zeros_like(distance)

for i, r in enumerate(distance):
    center_vec = np.array([r, 0.0, h])
    A_mesh[i] = disc_solid_angle(center_vec, np.array([0.0, 0.0, 1.0]), R)

# print(np.linalg.norm(A_mesh-A_fast))

# coef = np.polyfit(distance, A_mesh, 10)
# print(coef)
# plt.plot(distance, np.polyval(coef, distance), "--", label="disc fast", linewidth=2)

# def model(x, a, b):
#     return (a / x)**b
# params, cov = curve_fit(model, distance, A_mesh)
# a, b = params
# print(a, b)

# plt.plot(distance, model(distance, a, b))

plt.plot(distance, A_mesh, label="disc mesh", linewidth=2)
plt.plot(distance, A_fast, "--", label="disc fast", linewidth=2)
plt.xlabel("Lateral distance r (m)")
plt.ylabel("Apparent size A (solid angle)")
plt.title("Disc apparent size vs lateral distance")
plt.legend()
plt.tight_layout()
plt.show()


