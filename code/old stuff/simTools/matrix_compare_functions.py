import numpy as np
import pandas as pd
import glob
from skimage.metrics import structural_similarity as ssim

def eucledian_distance(A, B):
    return np.linalg.norm(A - B, ord='fro')

def l1_distance(A, B):
    return np.sum(np.abs(A - B))

def max_distance(A, B):
    return np.max(np.abs(A - B))

def mse(A, B):
    diff = A - B
    return np.mean(diff * diff)

def ssim_distance(A, B):
    return ssim(A, B, data_range = max(A.max(), B.max()) - min(A.min(), B.min()))

def create_histogram(df, bins=(30, 30), tank_pos=(0, 0), tank_size=(1.2, 1.2)): #plot_presence_probability without img drawing 
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

    return H

def get_positions(path): #Copied from draw_position_probability
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

def compare(sim_path='simulations/Homogeneous_1AB_fast.csv', exp_path='experimental_data/Zebrafish_Positions_data/Homogeneous_1AB/*', name=''):
    sim_homogeneous_1AB_df = pd.read_csv(sim_path)
    exp_homogeneous_1AB_df = get_positions(exp_path)
    sim_histogram = create_histogram(sim_homogeneous_1AB_df, tank_pos=(-0.6, -0.6))
    exp_histogram = create_histogram(exp_homogeneous_1AB_df)

    np.savetxt(f"simulations/histograms/sim_histogram{name}.csv", sim_histogram, delimiter=",")
    #np.savetxt(f"simulations/histograms/test_sim_histogram.csv", exp_histogram, delimiter=",")

    return (eucledian_distance(exp_histogram,sim_histogram), l1_distance(exp_histogram,sim_histogram), mse(exp_histogram,sim_histogram), ssim_distance(exp_histogram,sim_histogram))

#compare(sim_path='simulations/Heterogeneous_10AB_fast.csv', exp_path='experimental_data/Zebrafish_Positions_data/Heterogeneous_10AB/*')

"""
A = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

B = np.array([
    [1.1, 2.0, 2.9],
    [4.0, 4.9, 6.1],
    [7.0, 8.2, 8.8]
])

print("E_d:", eucledian_distance(A, B))
print("L1:", l1_distance(A, B))
print("Mx:", max_distance(A, B))
print("Mse:", mse(A, B))
"""