import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

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
    
def get_speed_pdf(df : pd.DataFrame, homogeneous=True):
    df["dx"] = df.groupby("fish_id")["x"].diff()
    df["dy"] = df.groupby("fish_id")["y"].diff()
    df["dt"] = df.groupby("fish_id")["time"].diff()
    df["speed"] = np.sqrt(df["dx"]**2 + df["dy"]**2) / df["dt"]
    MAX_SPEED = 0.005 # speed higher than 5mm (paper says 1mm, but we get much more stationary fish speeds)
    df = df[df["speed"].notna() & (df["speed"] > MAX_SPEED)].copy()

    if homogeneous:
        bins = np.arange(0, df["speed"].max() + 0.01, 0.01) # 1 cm wide bins
        counts, bins = np.histogram(df["speed"], bins=bins, density=True)
        counts /= np.sum(counts)
        return counts, bins
    else:
        r = 0.10
        center_1 = np.array([0.25, 0.25])
        center_2 = np.array([0.95, 0.95])
        d1 = (df["x"] - center_1[0])**2 + (df["y"] - center_1[1])**2
        d2 = (df["x"] - center_2[0])**2 + (df["y"] - center_2[1])**2
        df["inside_disk"] = (d1 <= r**2) | (d2 <= r**2)

        inside = df.loc[df["inside_disk"], "speed"]
        outside = df.loc[~df["inside_disk"], "speed"]

        bins = np.arange(0, df["speed"].max() + 0.01, 0.01) # 1 cm wide bins
        outside_counts, bins = np.histogram(outside, bins=bins, density=True)
        inside_counts, _ = np.histogram(inside, bins=bins, density=True)
        outside_counts /= np.sum(outside_counts)
        inside_counts /= np.sum(inside_counts)
        return inside_counts, outside_counts, bins

if __name__ == "__main__":
    # df = get_positions("Zebrafish_Positions_data/Heterogeneous_10AB/*01*")
    # plot_occupancy_map(df, bins=(30, 30), tank_size=(1.2, 1.2), cmap='Blues')
    # exit()


    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation

    # # Assuming df is the result of get_positions(path)
    # df = get_positions("Zebrafish_Positions_data/Heterogeneous_10AB/*02*")

    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xlim(df['x'].min(), df['x'].max())
    # ax.set_ylim(df['y'].min(), df['y'].max())
    # ax.set_xlabel('X Position')
    # ax.set_ylabel('Y Position')
    # ax.set_title('Fish Trajectories Over Time')
    # ax.invert_yaxis()  # Optional if coordinates come from images

    # fish_ids = df['fish_id'].unique()
    # colors = plt.cm.tab10.colors  # color palette

    # scatters = {}
    # trails = {}
    # for i, fid in enumerate(fish_ids):
    #     scatters[fid] = ax.plot([], [], 'o', color=colors[i % 10])[0]
    #     trails[fid] = ax.plot([], [], '-', color=colors[i % 10], alpha=0.5)[0]

    # times = sorted(df['time'].unique())
    # def animate(frame):
    #     t = times[frame]
    #     for fid in fish_ids:
    #         # Get last 3 positions (including current)
    #         df_fish = df[(df['fish_id'] == fid) & (df['time'] <= t)].tail(3)
    #         if not df_fish.empty:
    #             # Current position
    #             scatters[fid].set_data(df_fish['x'].values[-1], df_fish['y'].values[-1])
    #             # Trail positions
    #             trails[fid].set_data(df_fish['x'].values, df_fish['y'].values)
    #     return list(scatters.values()) + list(trails.values())

    # ani = FuncAnimation(fig, animate, frames=len(times), interval=1000, blit=True)
    # plt.show()
    # exit()

    script_path = os.path.join(os.path.dirname(__file__))
    homogeneous_1AB_df = get_positions(f"{script_path}/Zebrafish_Positions_data/Homogeneous_1AB/*")
    homogeneous_10AB_df = get_positions(f"{script_path}/Zebrafish_Positions_data/Homogeneous_10AB/*")
    heterogeneous_1AB_df = get_positions(f"{script_path}/Zebrafish_Positions_data/Heterogeneous_1AB/*")
    heterogeneous_10AB_df = get_positions(f"{script_path}/Zebrafish_Positions_data/Heterogeneous_10AB/*")

    homogeneous_1AB_counts, homogeneous_1AB_bins = get_speed_pdf(homogeneous_1AB_df)
    homogeneous_10AB_counts, homogeneous_10AB_bins = get_speed_pdf(homogeneous_10AB_df)
    heterogeneous_1AB_counts_inside, heterogeneous_1AB_counts_outside, heterogeneous_1AB_bins = get_speed_pdf(heterogeneous_1AB_df, False)
    heterogeneous_10AB_counts_inside, heterogeneous_10AB_counts_outside, heterogeneous_10AB_bins = get_speed_pdf(heterogeneous_10AB_df, False)

    data = {
        "homogeneous_1AB": {
            "counts": homogeneous_1AB_counts.tolist(),
            "bins": homogeneous_1AB_bins.tolist()
        },
        "homogeneous_10AB": {
            "counts": homogeneous_10AB_counts.tolist(),
            "bins": homogeneous_10AB_bins.tolist()
        },
        "heterogeneous_1AB": {
            "counts_inside": heterogeneous_1AB_counts_inside.tolist(),
            "counts_outside": heterogeneous_1AB_counts_outside.tolist(),
            "bins": heterogeneous_1AB_bins.tolist()
        },
        "heterogeneous_10AB": {
            "counts_inside": heterogeneous_10AB_counts_inside.tolist(),
            "counts_outside": heterogeneous_10AB_counts_outside.tolist(),
            "bins": heterogeneous_10AB_bins.tolist()
        }
    }

    with open("speed_histograms.json", "w") as f:
        json.dump(data, f, indent=4)

    plt.title("Homogeneous_1AB")
    counts, bins = homogeneous_1AB_counts, homogeneous_1AB_bins
    plt.bar(bins[:-1], counts, width=np.diff(bins), align="edge")
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.show()

    plt.title("Homogeneous_10AB")
    counts, bins = homogeneous_10AB_counts, homogeneous_10AB_bins
    plt.bar(bins[:-1], counts, width=np.diff(bins), align="edge")
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.show()

    plt.title("Heterogeneous_1AB")
    inside_counts, outside_counts, bins = heterogeneous_1AB_counts_inside, heterogeneous_1AB_counts_outside, heterogeneous_1AB_bins
    plt.bar(bins[:-1], outside_counts, width=np.diff(bins), align="edge", label="outside")
    plt.bar(bins[:-1], -inside_counts, width=np.diff(bins), align="edge", label="inside")
    plt.legend()
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.show()

    plt.title("Heterogeneous_10AB")
    inside_counts, outside_counts, bins = heterogeneous_10AB_counts_inside, heterogeneous_10AB_counts_outside, heterogeneous_10AB_bins
    plt.bar(bins[:-1], outside_counts, width=np.diff(bins), align="edge", label="outside")
    plt.bar(bins[:-1], -inside_counts, width=np.diff(bins), align="edge", label="inside")
    plt.legend()
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.show()



