import numpy as np
from .triangle import solid_angle_triangle  # your existing stable triangle formula


def compute_solid_angle_disc(center, normal, radius, n_samples=64):
    """
    Compute solid angle of a circular disc as seen from the origin.

    center: 3D vector in eye-local coordinates (disc center position)
    normal: unit normal of disc plane (not used for one-sided check)
    radius: radius of disc
    n_samples: number of boundary samples

    Returns: solid angle in steradians.
    """

    center = np.asarray(center, dtype=float)
    d = np.linalg.norm(center)
    if d == 0.0:
        # observer inside disc -- undefined; return full sphere
        return 4*np.pi

    # Unit direction to the disc center on sphere
    u_center = center / d

    # Normalize disc normal
    normal = np.asarray(normal, dtype=float)
    if np.linalg.norm(normal) == 0:
        normal = np.array([0, 0, 1])
    else:
        normal = normal / np.linalg.norm(normal)

    # Build an orthonormal basis (u_dir, v_dir) for the disc plane
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u_dir = np.cross(normal, tmp)
    u_norm = np.linalg.norm(u_dir)
    if u_norm == 0:
        u_dir = np.array([1.0, 0.0, 0.0])
    else:
        u_dir /= u_norm

    v_dir = np.cross(normal, u_dir)

    # Sample boundary points in 3D space
    thetas = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    boundary_points = [
        center + radius*(np.cos(t)*u_dir + np.sin(t)*v_dir)
        for t in thetas
    ]

    # Project boundary to unit sphere
    boundary_dirs = []
    for p in boundary_points:
        nrm = np.linalg.norm(p)
        if nrm == 0:
            continue
        boundary_dirs.append(p / nrm)

    # Must have at least 3 samples
    if len(boundary_dirs) < 3:
        return 0.0

    # Triangulate: center direction + each pair of boundary directions
    Omega = 0.0
    for i in range(len(boundary_dirs)):
        a = u_center
        b = boundary_dirs[i]
        c = boundary_dirs[(i+1) % len(boundary_dirs)]
        tri = solid_angle_triangle(a, b, c)
        if tri > 0:
            Omega += tri

    # To valid range
    if Omega < 0:
        Omega = 0.0
    if Omega > 4*np.pi:
        Omega = 4*np.pi

    return float(Omega)

