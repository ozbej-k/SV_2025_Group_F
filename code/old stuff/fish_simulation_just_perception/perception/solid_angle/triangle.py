"""
Computes the solid angle subtended at the origin by a triangle whose vertices are given
by position vectors a, b, c (in the coordinate frame where the origin is the eye).
Implements the Van Oosterom & Strackee formula as in original paper.
"""

import numpy as np

def solid_angle_triangle(a, b, c):
    """
    a, b, c : 3-element arrays (vectors from origin to triangle vertices)
    returns scalar solid angle in steradians (between -2pi and 2pi)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    la = np.linalg.norm(a)
    lb = np.linalg.norm(b)
    lc = np.linalg.norm(c)
    if la == 0 or lb == 0 or lc == 0:
        return 0.0

    # scalar triple product
    numerator = np.dot(a, np.cross(b, c))

    # denominator term
    denom = (la * lb * lc +
             np.dot(a, b) * lc +
             np.dot(b, c) * la +
             np.dot(c, a) * lb)

    # handle edge cases
    if denom == 0:
        # degenerate: points nearly collinear or very close to origin
        return 0.0

    omega = 2.0 * np.arctan2(numerator, denom)
    return float(omega)
