import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import glob


def get_positions(path):
    def get_pos(path, id):
        df = pd.read_csv(path, sep=' ', header=None, skiprows=1)
        df.rename(columns={0: 'time'}, inplace=True)

        num_objects = (df.shape[1] - 1) // 2

        dfs = []
        for i in range(num_objects):
            x_col, y_col, obj_id = (i*2+1, i*2+2, i+1)
            temp = df[['time', x_col, y_col]].copy()
            temp.columns = ['time', 'x', 'y']
            temp['fish_id'] = str(obj_id) + str(id)
            dfs.append(temp)

        long_df = pd.concat(dfs, ignore_index=True)
        return long_df[['time', 'fish_id', 'x', 'y']]
    
    files = glob.glob(path)
    dfs = [get_pos(f, fi) for fi, f in enumerate(files)]
    return pd.concat(dfs, ignore_index=True)

def plot_presence_probability(ax, df, bins=(30, 30), tank_pos=(0, 0), tank_size=(1.2, 1.2), cmap='Oranges'):
    xs = df['x'].to_numpy()
    ys = df['y'].to_numpy()

    if tank_size is None:
        xmin, xmax = np.nanmin(xs), np.nanmax(xs)
        ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    else:
        width, height = tank_size
        xmin, ymin = tank_pos
        xmax, ymax = xmin + width, ymin + height
        
    H, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H = H / len(xs)

    img = ax.imshow(H.T, origin='upper', extent=(xmin, xmax, ymin, ymax), cmap=cmap, aspect='equal', vmax=0.003)
    return img

for name in ["Homogeneous_1AB","Homogeneous_10AB","Heterogeneous_1AB","Heterogeneous_10AB"]:
    fig, ax = plt.subplots()
    df = pd.read_csv(f"simulations/{name}.csv")
    plot_presence_probability(ax, df, tank_pos=(-0.6, -0.6), tank_size=(1.2, 1.2))
    plt.savefig(f"simulations/sim_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    fig, ax = plt.subplots()
    df = get_positions(f"experimental_data/Zebrafish_Positions_data/{name}/*")
    df["x"] -= 0.6
    df["y"] -= 0.6
    plot_presence_probability(ax, df, tank_pos=(-0.6, -0.6), tank_size=(1.2, 1.2), cmap="Blues")
    fig.savefig(f"simulations/exp_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

for name in ["Drawn_1AB","Drawn_10AB"]:
    fig, ax = plt.subplots()
    df = pd.read_csv(f"simulations/{name}.csv")
    plot_presence_probability(ax, df, bins=(40, 30), tank_pos=(-0.8, -0.6), tank_size=(1.6, 1.2))
    plt.savefig(f"simulations/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

sim_homogeneous_1AB_df = pd.read_csv(f"simulations/Homogeneous_1AB.csv")
sim_homogeneous_10AB_df = pd.read_csv(f"simulations/Homogeneous_10AB.csv")
sim_heterogeneous_1AB_df = pd.read_csv(f"simulations/Heterogeneous_1AB.csv")
sim_heterogeneous_10AB_df = pd.read_csv(f"simulations/Heterogeneous_10AB.csv")

sim_dfs = [
    sim_homogeneous_1AB_df,
    sim_homogeneous_10AB_df,
    sim_heterogeneous_1AB_df,
    sim_heterogeneous_10AB_df,
]

exp_homogeneous_1AB_df = get_positions(f"experimental_data/Zebrafish_Positions_data/Homogeneous_1AB/*")
exp_homogeneous_10AB_df = get_positions(f"experimental_data/Zebrafish_Positions_data/Homogeneous_10AB/*")
exp_heterogeneous_1AB_df = get_positions(f"experimental_data/Zebrafish_Positions_data/Heterogeneous_1AB/*")
exp_heterogeneous_10AB_df = get_positions(f"experimental_data/Zebrafish_Positions_data/Heterogeneous_10AB/*")

exp_dfs = [
    exp_homogeneous_1AB_df,
    exp_homogeneous_10AB_df,
    exp_heterogeneous_1AB_df,
    exp_heterogeneous_10AB_df,
]

for df in exp_dfs:
    df["x"] -= 0.6
    df["y"] -= 0.6

sim_drawn_1AB_df = pd.read_csv(f"simulations/Drawn_1AB.csv")
sim_drawn_10AB_df = pd.read_csv(f"simulations/Drawn_10AB.csv")

sim_drawn_dfs = [
    sim_drawn_1AB_df,
    sim_drawn_10AB_df
]

titles = ["(a)", "(b)", "(c)", "(d)"]

cmaps = ["Blues", "Oranges", "Blues", "Oranges"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, df, title, cmap in zip(axes.ravel(), [exp_dfs[0], sim_dfs[0], exp_dfs[1], sim_dfs[1]], titles, cmaps):
    img = plot_presence_probability(ax, df, tank_pos=(-0.6, -0.6), tank_size=(1.2, 1.2), cmap=cmap)
    ax.set_title(title)
cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.75)
plt.savefig("simulations/homo_presence_probability.png", dpi=300, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, df, title, cmap in zip(axes.ravel(), [exp_dfs[2], sim_dfs[2], exp_dfs[3], sim_dfs[3]], titles, cmaps):
    img = plot_presence_probability(ax, df, tank_pos=(-0.6, -0.6), tank_size=(1.2, 1.2), cmap=cmap)
    circle1 = patches.Circle((-0.35, 0.35), 0.1, edgecolor='red', facecolor='none', linewidth=2)
    circle2 = patches.Circle((0.35, -0.35), 0.1, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_title(title)
cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.75)
plt.savefig("simulations/hetero_presence_probability.png", dpi=300, bbox_inches="tight")
plt.show()

import cv2
walls = cv2.imread('test_tank.png')
walls = cv2.flip(walls, 0)
walls = cv2.cvtColor(walls, cv2.COLOR_BGR2GRAY)
rgba = np.zeros((walls.shape[0], walls.shape[1], 4), dtype=np.float32)
threshold = 1  # adjust if needed
mask = walls >= threshold
rgba[mask, :3] = 0
rgba[mask, 3] = 1

tank_pos=(-0.8, -0.6)
tank_size=(1.6, 1.2)
x0 = - tank_size[0]/2
x1 = + tank_size[0]/2
y0 = - tank_size[1]/2
y1 = + tank_size[1]/2

fig, axes = plt.subplots(2, 1, figsize=(5, 8))
for ax, df, title, cmap in zip(axes.ravel(), [sim_drawn_dfs[0], sim_drawn_dfs[1]], titles[:2], ["Oranges", "Oranges"]):
    df["y"] = -df["y"]
    img = plot_presence_probability(ax, df, bins=(40, 30), tank_pos=tank_pos, tank_size=tank_size, cmap=cmap)
    ax.imshow(rgba, extent=(x0, x1, y0, y1))
    circle1 = patches.Circle((-0.15, 0.4), 0.1, edgecolor='red', facecolor='none', linewidth=2)
    circle2 = patches.Circle((-0.55, -0.4), 0.1, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_title(title)
cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.75)
plt.savefig("simulations/drawn_presence_probability.png", dpi=300, bbox_inches="tight")
plt.show()


