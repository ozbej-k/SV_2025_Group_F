import numpy as np
import matplotlib.pyplot as plt

# Load 2D histogram from CSV
#hist = np.loadtxt("simulations/histograms/sim_histogram_hetero10_alpha0_9_alphaW_2_beta0_0.9999999999999999_betaaW_0.5.csv", delimiter=",")
hist = np.loadtxt("simulations/histograms/sim_histogram_hetero10_alpha0_9_alphaW_2_beta0_0.9999999999999999_betaaW_0.5.csv", delimiter=",")
#9,2,5,3

#hist = np.loadtxt("simulations/histograms/test_sim_histogram.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(5, 5))

vmin = 0.0
vmax = 0.003


img = ax.imshow(
    np.fliplr(hist),
    origin="lower",
    cmap="Oranges",
    vmin=vmin,
    vmax=vmax
)

ax.set_title("2D Histogram")
ax.set_xlabel("X bin")
ax.set_ylabel("Y bin")

# Colorbar
plt.colorbar(img, ax=ax, shrink=0.75)


plt.tight_layout()
plt.show()
