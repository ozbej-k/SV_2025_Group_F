import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


if __name__ == "__main__":
    focal = Fish(0, 0, 0)
    other_fish = Fish(0.1, 0, 0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.2)

    fish_line, = ax.plot([], [], [], lw=3)
    proj_line, = ax.plot([], [], [], lw=4)
    ray_lines = []

    ax.scatter([0], [0], [0], s=80)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    def update(frame):
        global ray_lines
        for ln in ray_lines:
            ln.remove()
        ray_lines = []

        zero_one = (frame % 200) / 200
        theta = 2 * np.pi * zero_one
        other_fish.orientation = theta

        omega, sil = fish_solid_angle(focal, other_fish)
        fish = fish_vertices(other_fish)

        loop = np.vstack([fish, fish[0]])
        fish_line.set_data(loop[:, 0], loop[:, 1])
        fish_line.set_3d_properties(loop[:, 2])

        sil_loop = np.vstack([sil, sil[0]])
        proj_line.set_data(sil_loop[:, 0], sil_loop[:, 1])
        proj_line.set_3d_properties(sil_loop[:, 2])

        for vtx in sil:
            ln, = ax.plot([0, vtx[0]], [0, vtx[1]], [0, vtx[2]], ls='dashed', color="blue")
            ray_lines.append(ln)

        ax.set_title(f"Silhouette-based solid angle = {omega:.5f} sr")
        return fish_line, proj_line

    a = FuncAnimation(fig, update, frames=200, interval=40)
    plt.show()
