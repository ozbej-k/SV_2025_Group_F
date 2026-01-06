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

def plot_presence_probability(ax, df, bins=(30, 30), tank_pos=(0, 0), tank_size=(1.2, 1.2), cmap='Blues'):
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

    img = ax.imshow(H.T, origin='upper', extent=(xmin, xmax, ymin, ymax), cmap=cmap, aspect='auto', vmax=0.003)
    return img

# ## 3h => i forgot that one fish_loop is 1/3th of a second, so its actually 36000/3 seconds ~ 3h, df["time"] should be divided by 3
# # name = "Homogeneous_1AB_3h"
# # name = "Homogeneous_10AB_3h"
# name = "Heterogeneous_10AB_3h"
# # name = "Heterogeneous_1AB_3h"

# df = pd.read_csv(f"simulations/{name}.csv")
# plot_occupancy_map(df, tank_pos=(-0.6, -0.6), tank_size=(1.2, 1.2))

# plt.savefig(f"simulations/{name}.png", dpi=300, bbox_inches="tight")
# plt.show()

# sim_homogeneous_1AB_df = pd.read_csv(f"simulations/Homogeneous_1AB_3h.csv")
# sim_homogeneous_10AB_df = pd.read_csv(f"simulations/Homogeneous_10AB_3h.csv")
# sim_heterogeneous_1AB_df = pd.read_csv(f"simulations/Heterogeneous_1AB_3h.csv")
# sim_heterogeneous_10AB_df = pd.read_csv(f"simulations/Heterogeneous_10AB_3h.csv")
sim_homogeneous_1AB_df = pd.read_csv(f"simulations/Homogeneous_1AB_fast.csv")
sim_homogeneous_10AB_df = pd.read_csv(f"simulations/Homogeneous_10AB_fast.csv")
sim_heterogeneous_1AB_df = pd.read_csv(f"simulations/Heterogeneous_1AB_fast.csv")
sim_heterogeneous_10AB_df = pd.read_csv(f"simulations/Heterogeneous_10AB_fast.csv")

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


