import numpy as np
from .triangle import solid_angle_triangle

def solid_angle(vertices, faces):
    """
    Compute solid angle of a convex polyhedron as seen from the origin.
    vertices: list of 3D vectors (already in eye-local coordinates)
    faces: list of (i,j,k) triples indexing vertices
    Returns: solid angle in steradians (positive scalar)
    """

    total = 0.0

    for (i, j, k) in faces:
        a = np.asarray(vertices[i], dtype=float)
        b = np.asarray(vertices[j], dtype=float)
        c = np.asarray(vertices[k], dtype=float)

        # Compute solid angle for this face
        omega = solid_angle_triangle(a, b, c)

        # If triangle is back-facing (negative), ignore it,
        # convex polyhedron should only have forward-facing contributions.
        if omega > 0:
            total += omega

    # To valid range
    if total < 0:
        total = 0.0

    return float(total)
