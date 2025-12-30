import numpy as np

def norm(v):
    """Return vector length (Euclidean)."""
    return np.linalg.norm(v)

def normalize(v):
    """Return unit vector. If zero vector, return v unchanged."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def dot(a, b):
    return float(np.dot(a, b))

def cross(a, b):
    return np.cross(a, b)

def angle_between(a, b):
    """Angle between 3D vectors a and b (radians)."""
    a = normalize(a)
    b = normalize(b)
    c = np.clip(dot(a, b), -1.0, 1.0)
    return np.arccos(c)

def vec2_to_vec3(p):
    """Convert a 2D world point to 3D by adding z=0."""
    return np.array([p[0], p[1], 0.0], dtype=float)
