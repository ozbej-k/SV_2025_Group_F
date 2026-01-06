import matplotlib.pyplot as plt

import numpy as np

import config
from perception.perception_stuff.disc_mesh import solid_angle
from perception.disc_angle import disc_solid_angle

R = config.SPOT_RADIUS
h = config.SPOT_HEIGHT

distance = np.linspace(0, 4, 2000)
A_mesh = np.zeros_like(distance)
A_fast = np.zeros_like(distance)

for i, r in enumerate(distance):
    center_vec = np.array([r, 0.0, h])
    A_mesh[i] = solid_angle(center_vec, np.array([0.0, 0.0, 1.0]), R)
    A_fast[i] = disc_solid_angle(r)

plt.plot(distance, A_mesh, label="disc mesh", linewidth=2)
plt.plot(distance, A_fast, "--", label="disc fast", linewidth=2)
plt.xlabel("Lateral distance r (m)")
plt.ylabel("Apparent size A (solid angle)")
plt.title("Disc apparent size vs lateral distance")
plt.legend()
plt.tight_layout()
plt.show()

from perception.fish_angle import fish_solid_angle as fast_fish_solid_angle
from perception.perception_stuff.source_paper_fish import fish_solid_angle

from world.fish import Fish

focal = Fish(0.0, 0.0, 0.0, id_given="focal")
focal.position = np.array([1.0,0.0,0.0])
other = Fish(0.0, 0.0, 0.0, id_given="focal")
other.position = focal.position.copy()

dist = np.linspace(0.035, 0.0001, 100)
thetas = np.linspace(0.111, 2*np.pi-0.111, 60)

D, T = np.meshgrid(dist, thetas)

A_mesh = np.zeros_like(D)
A_fast = np.zeros_like(D)

for i in range(T.shape[0]):
    for j in range(D.shape[1]):
        d = D[i, j]
        theta = T[i, j]

        other.position[0] = focal.position[0] + d
        other.orientation = theta

        A_mesh[i, j] = fish_solid_angle(focal, other)[0]
        A_fast[i, j] = fast_fish_solid_angle(d, theta)

D_flat = D.ravel()
T_flat = T.ravel()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    D_flat, T_flat, A_mesh.ravel(),
    s=3, alpha=0.6, label="mesh"
)

ax.scatter(
    D_flat, T_flat, A_fast.ravel(),
    s=3, alpha=0.6, label="fast"
)

ax.set_xlabel("Distance")
ax.set_ylabel("Theta")
ax.set_zlabel("A")
ax.legend()

plt.tight_layout()
plt.show()

from perception.fish_angle import fish_solid_angle
from perception.perception_stuff.source_paper_fish import fish_solid_angle as source_fish_solid_angle 

from world.fish import Fish

import numpy as np
from matplotlib import pyplot as plt

focal = Fish(0.0, 0.0, 0.0)
focal.position = np.array([0.0,0.0,0.0])
other = Fish(0.1, 0.0, 0.0)
other.position = focal.position.copy()

dist = np.linspace(0.000001, 2, 2000)
dist_mesh = []
dist_fast = []

for d in dist:
  other.position[0] = focal.position[0] + d
  perc_mesh = source_fish_solid_angle(focal, other)[0]
  perc_fast = fish_solid_angle(d, 0)
  dist_mesh.append(perc_mesh)
  dist_fast.append(perc_fast)

plt.plot(dist, dist_mesh, label="mesh")
plt.plot(dist, dist_fast, label="fast", linestyle="--")
plt.ylim(0, max(dist_mesh)*1.2)
plt.legend()
plt.show()

thetas = np.linspace(0, 2*np.pi, 720)
focal.position = np.array([0.0,0.0,0.0])
other.position = focal.position.copy()
other.position[0] = focal.position[0] + 0.1

size_mesh = []
size_fast = []

for theta in thetas:
  other.orientation = theta
  perc_mesh = source_fish_solid_angle(focal, other)[0]
  perc_fast = fish_solid_angle(0.1, theta)
  size_mesh.append(perc_mesh)
  size_fast.append(perc_fast)

plt.plot(thetas, size_mesh, label="mesh")
plt.plot(thetas, size_fast, label="fast")
plt.ylim(0, max(size_fast)*1.2)
plt.legend()
plt.show()




