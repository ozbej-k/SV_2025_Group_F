import matplotlib.pyplot as plt

import numpy as np

import config
from world.fish import Fish
from world.tank import Tank
from perception.perception_model import perceive


def rotation_sweep(n_steps: int = 360, radius: float = 0.3):
	config.FISH_PERCEPTION_MODE = "both"

	focal = Fish(0.0, 0.0, 0.0, id_given="focal")
	tank = Tank(width=2.0, height=2.0, origin_at_center=True)

	thetas = np.linspace(-np.pi, np.pi, n_steps, endpoint=False)
	A_fast = []
	A_mesh = []

	other_id = "f1"

	for theta in thetas:
		# Place the other fish at fixed distance from the focal, rotate its body
		other = Fish(radius, 1.0, theta, id_given=other_id)
		fishies = [focal, other]

		perception = perceive(fish=focal, fishies=fishies, spots=[], tank=tank)

		fast = {f["id"]: f for f in perception["fish"]}
		mesh = {f["id"]: f for f in perception["fish_mesh_debug"]}

		A_fast.append(fast[other_id]["A"])
		A_mesh.append(mesh[other_id]["A"])
    
	return thetas, np.array(A_fast), np.array(A_mesh)

def plot_rotation(thetas, A_fast, A_mesh):
	theta_deg = np.degrees(thetas)

	plt.figure(figsize=(7, 5))
	plt.plot(theta_deg, A_mesh, label="fish mesh", linewidth=2)
	plt.plot(theta_deg, A_fast, "--", label="fish fast", linewidth=2)
	plt.xlabel("Body orientation (degrees)")
	plt.ylabel("Apparent size A (solid angle)")
	plt.title("Fish apparent size: mesh vs fast")
	plt.legend()
	plt.tight_layout()
	plt.show()

def main():
	thetas, A_fast, A_mesh = rotation_sweep()

	# RMSE diagnostic
	mask = A_mesh > 0
	if np.any(mask):
		rmse = np.sqrt(np.mean((A_fast[mask] - A_mesh[mask]) ** 2))
		print("Fish fast vs mesh RMSE:", rmse)

	plot_rotation(thetas, A_fast, A_mesh)

if __name__ == "__main__":
	main()

