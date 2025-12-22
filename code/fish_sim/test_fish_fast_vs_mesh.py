"""
Validation test: fast vs mesh model.

This script compares:
    - mu (direction)
    - A  (apparent size)

for identical fish configurations.
"""
import config
import time

from world.fish import Fish
from world.tank import Tank
from perception.perception_model import perceive

import numpy as np

def make_test_scene():

    focal = Fish(0.0, 0.0, 0.0, id_given="focal")

    fishies = [
        focal,
        Fish(0.15, 0.00, 0.0, id_given="f1"),   # directly ahead
        Fish(0.15, 0.05, 0.5, id_given="f2"),   # ahead and left, rotated
        Fish(0.20, -0.10, 1.0, id_given="f3"),  # ahead and right
        Fish(-0.10, 0.00, 0.0, id_given="f4"),  # behind (should be filtered)
        Fish(0.05, 0.15, 0.0, id_given="f5"),   # left side
        Fish(0.05, -0.15, 0.0, id_given="f6"),  # right side
        Fish(0.30, 0.00, np.pi/4, id_given="f7"),  # ahead but rotated
        Fish(0.05, 0.05, np.pi/2, id_given="f8"),  # to the left, rotated
        Fish(0.05, -0.05, -np.pi/2, id_given="f9"),  # to the right, rotated
    ]

    tank = Tank(width=2.0, height=2.0, origin_at_center=True)

    return focal, fishies, tank


def compare_outputs(fast, mesh, atol_mu=1e-6, rtol_A=0.2):
    """
    Compare fast and mesh perception outputs.
    """

    mesh_by_id = {f["id"]: f for f in mesh}

    print("\nComparison results:")
    print("-" * 72)
    print(f"{'id':<8} {'|Δμ|':>12} {'|ΔA|':>12} {'rel ΔA':>12}")
    print("-" * 72)

    for f in fast:
        fid = f["id"]

        if fid not in mesh_by_id:
            print(f"{fid:<8} missing in mesh output")
            continue

        m = mesh_by_id[fid]

        d_mu = abs(f["mu"] - m["mu"])
        d_A = abs(f["A"] - m["A"])
        rel_A = d_A / max(m["A"], 1e-12)

        print(f"{fid:<8} {d_mu:12.4e} {d_A:12.4e} {rel_A:12.4e}")

        if d_mu > atol_mu:
            print(f"  WARNING: mu mismatch for {fid}")

        if rel_A > rtol_A:
            print(f"  WARNING: A mismatch for {fid}")

    print("-" * 72)


def main():
    # Enable dual perception
    config.FISH_PERCEPTION_MODE = "both"

    focal, fishies, tank = make_test_scene()

    perception = perceive(
        fish=focal,
        fishies=fishies,
        spots=[],
        tank=tank,
    )

    fast = perception["fish"]
    mesh = perception["fish_mesh_debug"]

    # Sort by ID to ensure consistent comparison
    fast = sorted(fast, key=lambda x: x["id"])
    mesh = sorted(mesh, key=lambda x: x["id"])

    compare_outputs(fast, mesh)


if __name__ == "__main__":
    main()
