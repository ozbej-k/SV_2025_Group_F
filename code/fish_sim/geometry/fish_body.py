"""
Simple fish body generator: double pyramid mesh.
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


def get_fish_body_mesh(length=0.035, height=0.01, width=0.01):
    """
    Simple 6-vertex convex fish body mesh.

    Geometry:
    - Nose at +L/2 on the x-axis
    - Tail at -L/2 on the x-axis
    - A cross-section (shared base) located aprox. 1/3 of the body length back from the nose:
    at that x, -+height/2 in z (up/down)
    at that x, -+width/2  in y (left/right)
    This forms two pyramids sharing this cross-section as a base:
    - front pyramid height L/3
    - tail pyramid height 2L/3
    so the tail pyramid is twice as long along the x-axis.
    """

    L = length          # total length (default 3.5 cm)
    H = height          # total height (default 1.0 cm)
    W = width           # total width  (default 1.0 cm)

    # Apex positions along skeleton line
    x_nose =  L / 2.0
    x_tail = -L / 2.0

    # Cross-section position: 1/3 of body length back from the nose
    x_cross = L / 6.0

    # Vertices:
    # 0: nose tip
    # 1: tail center
    # 2: left  (y +W/2)  at cross-section
    # 3: right (y -W/2)  at cross-section
    # 4: top   (z +H/2)  at cross-section
    # 5: bottom(z -H/2)  at cross-section
    vertices = [
        np.array([x_nose,  0.0,      0.0     ]),  # 0 nose tip
        np.array([x_tail,  0.0,      0.0     ]),  # 1 tail center
        np.array([x_cross, W / 2.0,  0.0     ]),  # 2 left
        np.array([x_cross, -W / 2.0, 0.0     ]),  # 3 right
        np.array([x_cross, 0.0,      H / 2.0]),   # 4 top
        np.array([x_cross, 0.0,     -H / 2.0]),   # 5 bottom
    ]

    # Faces unchanged, two pyramids sharing the (2,3,4,5) base
    faces = [
        (0,2,4), (0,4,3), (0,3,5), (0,5,2),   # nose pyramid
        (1,4,2), (1,2,5), (1,5,3), (1,3,4)    # tail pyramid
    ]

    return vertices, faces

