import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.ticker import FuncFormatter

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

def plot_presence_probability(df, bins=(30, 30), tank_pos=(0, 0), tank_size=(1.2, 1.2), cmap='Blues'):
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

    plt.imshow(H.T, origin='upper', extent=(xmin, xmax, ymin, ymax), cmap=cmap, aspect='auto', vmax=0.003)
    return H
    
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
    # # df_real = get_positions("experimental_data/Zebrafish_Positions_data/Heterogeneous_1AB/*07*")
    # df_real = pd.read_csv("simulations/Heterogeneous_1AB1h.csv")
    # # df_real = df_real[df_real["fish_id"] == "10"]
    # # df_real = df_real[df_real["time"] <= 600]    
    # df_real["x"] +=  0.6
    # df_real["y"] +=  0.6
    # hist_real = plot_presence_probability(df_real, bins=(30, 30), tank_size=(1.2, 1.2), cmap='Blues')
    # # figure out how diff they are
    # # diff(real, real) should be 0
    # plt.ylim(0, 1.2)
    # plt.xlim(0, 1.2)
    # plt.axis("equal")
    # plt.show()
    # for fish_id, group in df_real.groupby("fish_id"):
    #     plt.plot(group["x"], group["y"])
    # plt.show()
    # print(df_real)
    # exit()


    # df_real = get_positions("Zebrafish_Positions_data/Heterogeneous_10AB/*01*")
    # df_sim = pd.read_csv("../simulations/Homogeneous_1AB_fast.csv")
    # df_sim["x"] +=  0.6
    # df_sim["y"] +=  0.6
    # hist_real = plot_presence_probability(df_real, bins=(30, 30), tank_size=(1.2, 1.2), cmap='Blues')
    # hist_sim = plot_presence_probability(df_sim, bins=(30, 30), tank_size=(1.2, 1.2), cmap='Blues')
    # # figure out how diff they are
    # # diff(real, real) should be 0
    # plt.ylim(0, 1.2)
    # plt.xlim(0, 1.2)
    # plt.axis("equal")
    # plt.show()
    # exit()


    # ### ANIMATION with trail of last 3 positions ###

    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation

    # # Fixed (corrected) and raw positions
    # df = get_positions("Zebrafish_Positions_data/Homogeneous_10AB_fixed_0_35/*09*")
    # df_raw = get_positions("Zebrafish_Positions_data/Homogeneous_10AB/*09*")

    # # Create side-by-side subplots: left = fixed, right = raw
    # fig, (ax_fixed, ax_raw) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # # Common axis limits based on both datasets
    # xmin = min(df['x'].min(), df_raw['x'].min())
    # xmax = max(df['x'].max(), df_raw['x'].max())
    # ymin = min(df['y'].min(), df_raw['y'].min())
    # ymax = max(df['y'].max(), df_raw['y'].max())

    # for ax in (ax_fixed, ax_raw):
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #     ax.set_xlabel('X Position')
    #     ax.set_ylabel('Y Position')
    #     ax.invert_yaxis()  # Optional if coordinates come from images

    # ax_fixed.set_title('Fixed trajectories')
    # ax_raw.set_title('Raw trajectories')

    # fish_ids = df['fish_id'].unique()
    # colors = plt.cm.tab10.colors  # color palette

    # scatters_fixed = {}
    # trails_fixed = {}
    # scatters_raw = {}
    # trails_raw = {}
    # for i, fid in enumerate(fish_ids):
    #     scatters_fixed[fid] = ax_fixed.plot([], [], 'o', color=colors[i % 10], label=f"fish {fid}")[0]
    #     trails_fixed[fid] = ax_fixed.plot([], [], '-', color=colors[i % 10], alpha=0.5)[0]

    #     # Raw (original) data for the same fish on the right, more transparent
    #     scatters_raw[fid] = ax_raw.plot([], [], 'o', color=colors[i % 10], alpha=0.4, label=f"fish {fid}")[0]
    #     trails_raw[fid] = ax_raw.plot([], [], '-', color=colors[i % 10], alpha=0.4)[0]

    # # Legends showing which color corresponds to which fish_id
    # ax_fixed.legend(loc='upper right', fontsize=6, framealpha=0.6)
    # ax_raw.legend(loc='upper right', fontsize=6, framealpha=0.6)

    # times = sorted(df['time'].unique())

    # # Text annotation to show current frame index (on fixed axis)
    # frame_text = ax_fixed.text(
    #     0.02,
    #     0.98,
    #     '',
    #     transform=ax_fixed.transAxes,
    #     va='top',
    #     ha='left',
    #     fontsize=10,
    #     color='black',
    #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    # )

    # def animate(frame):
    #     t = times[frame]
    #     # Update frame index display
    #     frame_text.set_text(f"Frame {frame + 1}/{len(times)}  (t={t})")
    #     for fid in fish_ids:
    #         # Get last 3 positions (including current) for fixed data
    #         df_fish = df[(df['fish_id'] == fid) & (df['time'] <= t)].tail(3)
    #         if not df_fish.empty:
    #             # Current position (fixed)
    #             scatters_fixed[fid].set_data([df_fish['x'].values[-1]], [df_fish['y'].values[-1]])
    #             # Trail positions (fixed)
    #             trails_fixed[fid].set_data(df_fish['x'].values, df_fish['y'].values)
    #         else:
    #             scatters_fixed[fid].set_data([], [])
    #             trails_fixed[fid].set_data([], [])

    #         # Get last 3 positions for raw data
    #         df_fish_raw = df_raw[(df_raw['fish_id'] == fid) & (df_raw['time'] <= t)].tail(3)
    #         if not df_fish_raw.empty:
    #             scatters_raw[fid].set_data([df_fish_raw['x'].values[-1]], [df_fish_raw['y'].values[-1]])
    #             trails_raw[fid].set_data(df_fish_raw['x'].values, df_fish_raw['y'].values)
    #         else:
    #             scatters_raw[fid].set_data([], [])
    #             trails_raw[fid].set_data([], [])

    #     return (
    #         list(scatters_fixed.values())
    #         + list(trails_fixed.values())
    #         + list(scatters_raw.values())
    #         + list(trails_raw.values())
    #     )

    # frame_index = [0]  # Mutable object to hold the current frame index

    # # Key press event handler
    # def on_key(event):
    #     if event.key == 'up':
    #         frame_index[0] = (frame_index[0] + 1) % len(times)
    #     elif event.key == 'down': 
    #         frame_index[0] = (frame_index[0] - 1) % len(times)  
    #     animate(frame_index[0])  # Update the animation manually
    #     fig.canvas.draw_idle()  
    # # Connect the key press event to the figure
    # fig.canvas.mpl_connect('key_press_event', on_key)

    # # Initial animation setup
    # animate(frame_index[0])  # Draw the first frame
    # plt.show()
    # exit()

    ### SPEED HISTOGRAMS ###

    script_path = os.path.join(os.path.dirname(__file__))
    homogeneous_1AB_df = get_positions(f"{script_path}/Zebrafish_Positions_data/Homogeneous_1AB/*")
    homogeneous_10AB_df = get_positions(f"{script_path}/Zebrafish_Positions_data/Homogeneous_10AB_fixed_0_35/*")
    heterogeneous_1AB_df = get_positions(f"{script_path}/Zebrafish_Positions_data/Heterogeneous_1AB/*")
    heterogeneous_10AB_df = get_positions(f"{script_path}/Zebrafish_Positions_data/Heterogeneous_10AB_fixed_0_35/*")

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

    # with open("speed_histograms.json", "w") as f:
    #     json.dump(data, f, indent=4)

    # plt.title("Homogeneous_1AB")
    # counts, bins = homogeneous_1AB_counts, homogeneous_1AB_bins
    # plt.bar(bins[:-1], counts, width=np.diff(bins), align="edge")
    # plt.xlabel("Speed")
    # plt.ylabel("Frequency")
    # plt.show()

    # plt.title("Homogeneous_10AB")
    # counts, bins = homogeneous_10AB_counts, homogeneous_10AB_bins
    # plt.bar(bins[:-1], counts, width=np.diff(bins), align="edge")
    # plt.xlabel("Speed")
    # plt.ylabel("Frequency")
    # plt.show()

    # plt.title("Heterogeneous_1AB")
    # inside_counts, outside_counts, bins = heterogeneous_1AB_counts_inside, heterogeneous_1AB_counts_outside, heterogeneous_1AB_bins
    # plt.bar(bins[:-1], outside_counts, width=np.diff(bins), align="edge", label="outside")
    # plt.bar(bins[:-1], -inside_counts, width=np.diff(bins), align="edge", label="inside")
    # plt.legend()
    # plt.xlabel("Speed")
    # plt.ylabel("Frequency")
    # plt.show()

    # plt.title("Heterogeneous_10AB")
    # inside_counts, outside_counts, bins = heterogeneous_10AB_counts_inside, heterogeneous_10AB_counts_outside, heterogeneous_10AB_bins
    # plt.bar(bins[:-1], outside_counts, width=np.diff(bins), align="edge", label="outside")
    # plt.bar(bins[:-1], -inside_counts, width=np.diff(bins), align="edge", label="inside")
    # plt.legend()
    # plt.xlabel("Speed")
    # plt.ylabel("Frequency")
    # plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    counts, bins = homogeneous_1AB_counts, homogeneous_1AB_bins
    axes[0, 0].bar(bins[:-1], counts, width=np.diff(bins), align="edge")
    axes[0, 0].set_title("(a)",loc="left")
    axes[0, 0].set_xlabel("Speed")
    axes[0, 0].set_ylabel("Frequency")

    counts, bins = homogeneous_10AB_counts, homogeneous_10AB_bins
    axes[0, 1].bar(bins[:-1], counts, width=np.diff(bins), align="edge")
    axes[0, 1].set_title("(b)",loc="left")
    axes[0, 1].set_xlabel("Speed")
    axes[0, 1].set_ylabel("Frequency")

    inside, outside, bins = (
        heterogeneous_1AB_counts_inside,
        heterogeneous_1AB_counts_outside,
        heterogeneous_1AB_bins
    )
    axes[1, 0].bar(bins[:-1], outside, width=np.diff(bins),
                align="edge", label="outside spot")
    axes[1, 0].bar(bins[:-1], -inside, width=np.diff(bins),
                align="edge", label="under spot")
    axes[1, 0].legend()
    axes[1, 0].set_title("(c)",loc="left")
    axes[1, 0].set_xlabel("Speed")
    axes[1, 0].set_ylabel("Frequency")

    inside, outside, bins = (
        heterogeneous_10AB_counts_inside,
        heterogeneous_10AB_counts_outside,
        heterogeneous_10AB_bins
    )
    axes[1, 1].bar(bins[:-1], outside, width=np.diff(bins),
                align="edge", label="outside spot")
    axes[1, 1].bar(bins[:-1], -inside, width=np.diff(bins),
                align="edge", label="under spot")
    axes[1, 1].legend()
    axes[1, 1].set_title("(d)",loc="left")
    axes[1, 1].set_xlabel("Speed")
    axes[1, 1].set_ylabel("Frequency")
    top_max = max(
        max(homogeneous_1AB_counts),
        max(homogeneous_10AB_counts)
    )
    axes[0, 0].set_ylim(0, top_max * 1.05)
    axes[0, 1].set_ylim(0, top_max * 1.05)

    # --- match y-limits for bottom row (symmetric because of +/- bars) ---    
    inside_max = max(
        max(heterogeneous_1AB_counts_inside),
        max(heterogeneous_10AB_counts_inside),
    )
    outside_max = max(
        max(heterogeneous_1AB_counts_outside),
        max(heterogeneous_10AB_counts_outside)
    )
    abs_formatter = FuncFormatter(lambda y, _: f"{abs(y):.2f}")
    axes[1, 0].yaxis.set_major_formatter(abs_formatter)
    axes[1, 1].yaxis.set_major_formatter(abs_formatter)
    axes[1, 0].set_ylim(-inside_max * 1.05, outside_max * 1.15)
    axes[1, 1].set_ylim(-inside_max * 1.05, outside_max * 1.15)
    plt.savefig("speed_pdfs.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()