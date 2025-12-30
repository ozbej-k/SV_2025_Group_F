"""
Simple fish body generator: box-shaped mesh.

Returns:
    vertices: list of 3D points in fish-local coordinates (centered on fish origin)
    faces: list of triangular faces as triples of vertex indices
Notes:
    the fish-local coordinate frame: origin at fish center (or eye, can later offset if needed)
    forward along +x, left along +y, up +z
"""
import numpy as np

def make_fish_box_vertices(length, width, height):
    # create 8 corners of box centered on origin
    L2 = length / 2.0
    W2 = width / 2.0
    H2 = height / 2.0

    # vertices: x, y, z
    verts = [
        np.array([ L2,  W2,  H2]),
        np.array([ L2, -W2,  H2]),
        np.array([-L2, -W2,  H2]),
        np.array([-L2,  W2,  H2]),
        np.array([ L2,  W2, -H2]),
        np.array([ L2, -W2, -H2]),
        np.array([-L2, -W2, -H2]),
        np.array([-L2,  W2, -H2]),
    ]
    return verts

def make_box_tri_faces():
    # triangulate each face into two triangles (12 triangles total)
    faces = [
        # top (+z)
        (0,1,2), (2,3,0),
        # bottom (-z)
        (4,7,6), (6,5,4),
        # front (+x)
        (0,4,5), (5,1,0),
        # back (-x)
        (3,2,6), (6,7,3),
        # left (+y)
        (0,3,7), (7,4,0),
        # right (-y)
        (1,5,6), (6,2,1)
    ]
    return faces


def get_fish_body_mesh(length=0.03, height=0.01, width=0.006):
    """
    Simple 6-vertex convex fish body mesh (as in Collignon et al.).
    Centered at origin, pointing +x direction.
    """

    L = length
    H = height
    W = width

    vertices = [
        np.array([ L/2, 0,   0  ]),    # nose tip
        np.array([-L/2, 0,   0  ]),    # tail center
        np.array([ 0,   W/2, 0  ]),    # left fin
        np.array([ 0,  -W/2, 0  ]),    # right fin
        np.array([ 0,   0,   H/2]),    # top
        np.array([ 0,   0,  -H/2])     # bottom
    ]

    # Triangulate: convex polyhedron of 6 points
    faces = [
        (0,2,4), (0,4,3), (0,3,5), (0,5,2),
        (1,4,2), (1,2,5), (1,5,3), (1,3,4)
    ]

    return vertices, faces

