
import matplotlib.pyplot as plt

import numpy as np

import config
from geometry.solid_angle.disc import solid_angle as disc_solid_angle
from perception.disc_fast import solid_angle_fast


def disc_sweep(n_steps: int = 200, r_max: float = 3):
    """Sweep lateral distance under the disc and get apparent size.
    """

    R = config.SPOT_RADIUS
    h = config.SPOT_HEIGHT

    distance = np.linspace(0.0, r_max, n_steps)
    A_mesh = np.zeros_like(distance)
    A_fast = np.zeros_like(distance)

    normal = np.array([0.0, 0.0, 1.0])

    for i, r in enumerate(distance):
        center_vec = np.array([r, 0.0, h])
        A_mesh[i] = disc_solid_angle(center_vec, normal, R)
        A_fast[i] = solid_angle_fast(center_vec)

    return distance, A_mesh, A_fast


def plot_disc(distance, A_mesh, A_fast):
    plt.figure(figsize=(7, 5))
    plt.plot(distance, A_mesh, label="disc mesh", linewidth=2)
    plt.plot(distance, A_fast, "--", label="disc fast", linewidth=2)
    plt.xlabel("Lateral distance r (m)")
    plt.ylabel("Apparent size A (solid angle)")
    plt.title("Disc apparent size vs lateral distance")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    distance, A_mesh, A_fast = disc_sweep()
    plot_disc(distance, A_mesh, A_fast)


if __name__ == "__main__":
    main()
