import numpy as np

def compute_mu(v_local):
    """Return horizontal angle (relative direction)."""
    return np.arctan2(v_local[1], v_local[0])
