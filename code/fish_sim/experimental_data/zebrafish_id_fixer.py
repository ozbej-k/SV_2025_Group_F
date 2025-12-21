# %%
import numpy as np

import pandas as pd

import glob

import os

import sys

from scipy.optimize import linear_sum_assignment

PER_SECOND_DISTANCE = 0.5  # default, can be overridden from command line

def load_data_to_dataframe(file_path):
    """
    Load zebrafish positional data into a DataFrame.
    """
    # Read the file
    data = pd.read_csv(file_path, header=None, skiprows=1, sep=r'\s+')

    data = data.drop(data.columns[0], axis=1)

    # Generate column names dynamically
    column_names = [f"X{i//2}" if i % 2 == 0 else f"Y{i//2}" for i in range(data.shape[1])]

    # Assign column names and set the time column asindex
    data.columns = column_names
    data.index.name = "time"

    return data

# %%
def is_suspicious(row_1, row_2, threshold=0.6):
    distances = []
    indexes = []
    for i in range(10):
        pair_1 = (row_1.iloc[2*i], row_1.iloc[2*i+1])
        pair_2 = (row_2.iloc[2*i], row_2.iloc[2*i+1])
        dist = np.sqrt((pair_1[0] - pair_2[0])**2 + (pair_1[1] - pair_2[1])**2)
        if dist > threshold:
            distances.append(dist)
            indexes.append(i)
    
    if not distances:
        return [], []
    
    sorted_pairs = sorted(zip(distances, indexes), key=lambda x: x[0], reverse=True)
    sorted_distances, sorted_indexes = zip(*sorted_pairs)  
    
    return sorted_distances, sorted_indexes

# %%
def build_tracked_dataframe(df, teleport_threshold=None):
    """
    Build a new DataFrame where fish IDs are made consistent over time
    by tracking them frame-by-frame with the Hungarian algorithm.

    This treats each frame as an unordered set of 10 fish positions and
    reconstructs 10 continuous trajectories (X0,Y0,...,X9,Y9) that move
    as smoothly as possible between consecutive frames.
    """
    
    n_fish = df.shape[1] // 2
    if n_fish == 0:
        return df.copy()

    tracked = df.copy()

    prev = df.iloc[0].to_numpy().reshape(n_fish, 2)

    for t in range(1, len(df)):
        curr = df.iloc[t].to_numpy().reshape(n_fish, 2)

        # Cost matrix squared distances between previous tracks and curr
        diff = prev[:, None, :] - curr[None, :, :]
        cost = np.einsum("ijk,ijk->ij", diff, diff)

        # If everything is very far, keep original labeling for frame
        if teleport_threshold is not None:
            min_per_track = np.sqrt(cost.min(axis=1))
            if np.all(min_per_track > teleport_threshold):
                prev = curr
                continue

        row_ind, col_ind = linear_sum_assignment(cost)

        # for each track i, which detection j it gets
        assignment = np.empty(n_fish, dtype=int)
        assignment[row_ind] = col_ind

        # Reorder current detections into track order
        new_curr = curr[assignment]

        # Write back into the tracked DataFrame for this frame.
        tracked.iloc[t, 0::2] = new_curr[:, 0]
        tracked.iloc[t, 1::2] = new_curr[:, 1]

        # Update
        prev = new_curr

    return tracked

# %%
def map_fix_ids(row_1, row_2, problem_fish_id=None):
    """
    Compute a global ID mapping between two consecutive frames using the
    Hungarian algorithm
    """
    # Number of fish inferred from the number of columns
    n_fish = len(row_1) // 2
    if n_fish == 0:
        return {}

    # Extract coordinates as (n_fish, 2) arrays for the two frames.
    coords1 = np.column_stack([row_1.iloc[0::2].values, row_1.iloc[1::2].values])
    coords2 = np.column_stack([row_2.iloc[0::2].values, row_2.iloc[1::2].values])

    # Build squared-distance matrix and euclidean distances
    diff = coords1[:, None, :] - coords2[None, :, :]
    sqdist = np.einsum("ijk,ijk->ij", diff, diff)
    dist = np.sqrt(sqdist)

    # global threshold as hard constraint
    threshold = PER_SECOND_DISTANCE

    # Start from squared distances as cost
    cost = sqdist.copy()
    max_sqdist = sqdist.max() if sqdist.size else 1.0
    big = max_sqdist * 1e6

    # not allow moves longer than the threshold by assigning huge cost
    cost[dist > threshold] = big

    # If fish flagged as problematic and its self move is within the allowed threshold, 
    # slightly penalize keeping its label to
    # encourage new label
    if problem_fish_id is not None and 0 <= problem_fish_id < n_fish:
        if dist[problem_fish_id, problem_fish_id] <= threshold:
            cost[problem_fish_id, problem_fish_id] += max_sqdist * 1e3

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build permutation assignment ids
    assignment = np.empty(n_fish, dtype=int)
    assignment[row_ind] = col_ind

    # Check if assignment keeps within threshold
    assigned_dist = dist[np.arange(n_fish), assignment]
    if not np.all(assigned_dist <= threshold):
        return {}

    # Convert to mapping dict
    mapping = {int(i): int(j) for i, j in enumerate(assignment) if i != j}

    return mapping


# %%
fixed_path = "Zebrafish_Positions_data/Heterogeneous_10AB_fixed/02_fixed.txt"

# %%
def swap_ids_in_dataframe(df, mapping, start_time):
    """
    Swap fish IDs in the DataFrame based on the provided mapping, from time onwards.
    """
    # Convert df to a np array for faster operations
    data = df.to_numpy()
    
    data_orig = data.copy()

    # Iterate over rows starting from the specified time
    for t in range(start_time, len(data)):
        for old_id, new_id in mapping.items():
            # Calculate column indices for X and Y coordinates
            old_x, old_y = 2 * old_id, 2 * old_id + 1
            new_x, new_y = 2 * new_id, 2 * new_id + 1

            # Swap coordinates using np
            data[t, [old_x, old_y]] = data_orig[t, [new_x, new_y]]

    # Update df with the modified np array
    df.iloc[:, :] = data

# %%
def check_for_error(index, problem_fish_id, df_raw):
    """
    Calc a corrected location for a single fish at time t+1 using
    information from the previous and later rows in the data

    This function does not enforce the distance threshold, it only
    returns a candidate, midpoint
    between the raw positions at times t and t+2
    """
    # get later raw row (t+2) for interpolation
    if index >= len(df_raw) or index + 2 >= len(df_raw):
        return None

    # Raw positions at times t and t+2 for the selected fish
    row_1_raw = df_raw.iloc[index]
    row_3_raw = df_raw.iloc[index + 2]

    x_t = row_1_raw.iloc[2 * problem_fish_id]
    y_t = row_1_raw.iloc[2 * problem_fish_id + 1]
    x_t2 = row_3_raw.iloc[2 * problem_fish_id]
    y_t2 = row_3_raw.iloc[2 * problem_fish_id + 1]

    # Proposed position at time t+1 is midpoint between raw positions at t and t+2
    x_mid = (x_t + x_t2) / 2.0
    y_mid = (y_t + y_t2) / 2.0

    return (x_mid, y_mid)

# %%
def fix_ids(df, df_raw):
    index = 0

    while index < (len(df) - 1):
        row_1 = df.iloc[index] # get the current row
        row_2 = df.iloc[index + 1] # get the next row

        # check suspicious distances
        sorted_distances, sorted_indexes = is_suspicious(row_1, row_2, PER_SECOND_DISTANCE)

        if not sorted_distances:
            index += 1
            continue

        # There are one or more suspicious jumps at this frame.
        print(f"At time {index}, suspicious distances: {[f'{dist:.2f}' for dist in sorted_distances]} for fish indexes: {sorted_indexes}")


        # 1. try a single global id mapping
        mapping = map_fix_ids(row_1, row_2, problem_fish_id=None)

        if mapping:
            # apply the mapping from time index+1 onwards
            swap_ids_in_dataframe(df, mapping, index + 1)
            print("Applied mapping:", mapping)
            continue

        # 2) If no ideal mapping exists, correct each suspicious fish at this frame individually using interpolation
        for fish_id in sorted_indexes:
            
            row_1 = df.iloc[index]
            row_2 = df.iloc[index + 1]

            # proposal for the new location at time t+1
            proposal = check_for_error(
                index, fish_id, df_raw
            )

            if proposal is None:
                # fallback, use the current position at t+1
                x_new = row_2.iloc[2 * fish_id]
                y_new = row_2.iloc[2 * fish_id + 1]
            else:
                x_new, y_new = proposal

            # clamp the new position so that the distance from time t to t+1 does not exceed PER_SECOND_DISTANCE in corrected df.
            x_prev = row_1.iloc[2 * fish_id]
            y_prev = row_1.iloc[2 * fish_id + 1]

            dx = x_new - x_prev
            dy = y_new - y_prev
            dist = np.sqrt(dx * dx + dy * dy)

            if dist > PER_SECOND_DISTANCE and dist > 0:
                scale = PER_SECOND_DISTANCE / dist
                x_new = x_prev + dx * scale
                y_new = y_prev + dy * scale

            # write the clamped position into df for time t+1
            df.iloc[index + 1, 2 * fish_id] = x_new
            df.iloc[index + 1, 2 * fish_id + 1] = y_new

        index += 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("use: python zebrafish_id_fixer.py <input_folder> <METERS_PER_SECOND>=0.5")
        sys.exit(1)

    input_folder = sys.argv[1]
    PER_SECOND_DISTANCE = float(sys.argv[2])
    
    for file in glob.glob(os.path.join(input_folder, "*.txt")):
        print(f"Processing file: {file}")

        df_raw = load_data_to_dataframe(file)
        df = df_raw.copy()

        df_tracked = build_tracked_dataframe(df, PER_SECOND_DISTANCE)
        df = df_tracked
        
        fix_ids(df, df_raw)
        
        df = df.reset_index()
        
        # Create output folder with "_fixed" suffix and speed param
        output_folder = input_folder.rstrip(os.sep) + "_fixed_" + str(PER_SECOND_DISTANCE).replace('.', '_')
        os.makedirs(output_folder, exist_ok=True)

        # Generate fixed file path with "_fixed" suffix and speed param
        base_name = os.path.basename(file)
        fixed_file_name = os.path.splitext(base_name)[0] + "_fixed_" + str(PER_SECOND_DISTANCE).replace('.', '_') + ".txt"
        fixed_file_path = os.path.join(output_folder, fixed_file_name)

        df.to_csv(fixed_file_path, sep=' ', header=True, index=False)