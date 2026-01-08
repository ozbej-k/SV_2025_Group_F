import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Load CSV
df = pd.read_csv("simulations/histograms/distances.csv")

x = df["alpha_0"].values
y = df["alpha_w"].values
z = df["ssim"].values

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_trisurf(
    x, y, z,
    cmap="viridis",
    linewidth=0.2,
    antialiased=True
)

ax.set_xlabel("alpha_0")
ax.set_ylabel("alpha_w")
ax.set_zlabel("SSIM")

fig.colorbar(surf, ax=ax, label="SSIM")

plt.title("SSIM surface over alpha_0 and alpha_w")
plt.show()
