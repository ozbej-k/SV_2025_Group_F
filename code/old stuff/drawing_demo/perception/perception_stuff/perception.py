import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Fish:
    length = 0.035
    width = 0.01
    height = 0.01
    l, w, h = length/2, width/2, height/2
    base_fish_vert = np.array([
        [ 0,  0,  h],
        [ 0,  0, -h],
        [ l,  0,  0],
        [ 0,  w,  0],
        [ 0, -w,  0],
        [-l,  0,  0],
    ])
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

    def __init__(self, x, y, orientation):
        self.position = np.array([x, y, 0], dtype=float)
        self.orientation = orientation
        self.speed = np.zeros(2)

def fish_vertices(fishes):
    orientations = np.array([f.orientation for f in fishes])
    positions = np.array([f.position for f in fishes])
    
    c, s = np.cos(orientations), np.sin(orientations)
    rz = np.stack([[[ci, -si, 0],[si, ci, 0],[0,0,1]] for ci, si in zip(c, s)])
    
    base = Fish.base_fish_vert.T
    vertices = rz @ base
    vertices = vertices.transpose(0, 2, 1) + positions[:, None, :]
    return vertices

def normalize(v, axis=-1):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(norm == 0, 1.0, norm)

def spherical_arc(u, v): # angle between unit vectors, same as spherical coords
    return np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0))

def solid_angle_triangle(u, v, w):
    a = spherical_arc(v, w)
    b = spherical_arc(u, w)
    c = spherical_arc(u, v)
    s = (a + b + c) / 2
    return 4 * np.arctan(np.sqrt(
        np.tan(s/2) * np.tan((s-a)/2) *
        np.tan((s-b)/2) * np.tan((s-c)/2)
    ))

def fish_solid_angle_all(fishes):
    num_fish = len(fishes)
    vertices_all = fish_vertices(fishes)
    positions = np.array([f.position for f in fishes])
    
    rel_vertices = vertices_all[None, :, :, :] - positions[:, None, None, :]
    norms = np.linalg.norm(rel_vertices, axis=-1, keepdims=True)
    unit_vertices = rel_vertices / np.where(norms == 0, 1.0, norms)
    
    rel_pos = positions[None, :, :] - positions[:, None, :]
    angles_xy = np.arctan2(rel_pos[:, :, 1], rel_pos[:, :, 0])
    
    solid_angles = []
    angles_list = []

    for i in range(num_fish):
        direction = unit_vertices[i].sum(axis=1)
        direction = normalize(direction)
        
        perp = np.array([-direction[:, 1], direction[:, 0]]).T
        vertices_xy = unit_vertices[i, :, :, :2]
        left_right = np.einsum('vjd,vd->vj', vertices_xy, perp)
        L_idx = np.argmin(left_right, axis=1)
        R_idx = np.argmax(left_right, axis=1)
        
        sa_list = []
        angle_list = []
        for j in range(num_fish):
            if i == j:
                continue
            L, R = unit_vertices[i, j, L_idx[j]], unit_vertices[i, j, R_idx[j]]
            T, B = unit_vertices[i, j, Fish.TOP_INDEX], unit_vertices[i, j, Fish.BOTTOM_INDEX]
            sa = solid_angle_triangle(L, T, R) + solid_angle_triangle(L, B, R)
            sa_list.append(sa)
            angle_list.append(angles_xy[i, j])
        
        solid_angles.append(np.array(sa_list))
        angles_list.append(np.array(angle_list))
    
    return solid_angles, angles_list

fishes = [
    Fish(0,0.1,0),
    Fish(0.1,0,0),
    Fish(-0.1,0.1,np.pi/4),
    Fish(0.05,-0.1,np.pi/2),
    Fish(0.15,-0.11,np.pi/2),
    Fish(0.25,-0.13,np.pi/2),
    Fish(0.35,-0.15,np.pi/2),
    Fish(0.45,-0.21,np.pi/2),
    Fish(0.55,-0.41,np.pi/2),
    Fish(0.65,-0.11,np.pi/2),
]

print(fish_solid_angle_all(fishes)[0][0])









def fish_vertices(fish: Fish):
    c, s = np.cos(fish.orientation), np.sin(fish.orientation)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    vertices = (rz @ Fish.base_fish.T).T + fish.position
    return vertices

def spherical_arc(u, v):
    # angle between two points on unit circle (same as doing it for spherical points)
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

def solid_angle_triangle(u, v, w):
    # solid angle of triangle given 3 points on unit circle
    a = spherical_arc(v, w)
    b = spherical_arc(u, w)
    c = spherical_arc(u, v)
    s = (a + b + c) / 2
    return 4 * np.arctan(np.sqrt(
        np.tan(s / 2) * np.tan((s - a) / 2) * np.tan((s - b) / 2) * np.tan((s - c) / 2)
    ))

def normalize(v):
    return v / np.linalg.norm(v)

def fish_solid_angle(fish_focal: Fish, fish_observed: Fish):
    # only works for fish at z = 0
    vertices = fish_vertices(fish_observed) - fish_focal.position
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    unit = vertices / norms

    direction = unit.sum(axis=0)
    direction /= np.linalg.norm(direction)

    left_right = vertices @ np.array([-direction[1], direction[0], 0])
    L = unit[np.argmin(left_right)]
    R = unit[np.argmax(left_right)]
    T = unit[0]
    B = unit[1]

    solid_angle = (
        solid_angle_triangle(L, T, R) +
        solid_angle_triangle(L, B, R)
    )
    silhouette = np.array([L, T, R, B])
    return solid_angle, silhouette