'''Test script to compute angles mu of other fish relative to focal fish orientation.'''

import numpy as np
import matplotlib.pyplot as plt

focal = np.array([0.0, 0.0])
facing = 0.0  # radians; 0 => +x

others = {
    'other1': np.array([0.10, 0.05]),
    'other2': np.array([0.0, -0.6]),
    'behind_example': np.array([-0.6, 0.0])
}

def mu_of(point, focal_=focal, theta=facing):
    rel = point - focal_
    # if focal had a nonzero orientation, rotate rel by -theta:
    R = [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]
    rel = R @ rel
    
    return np.arctan2(rel[1], rel[0])

for name, p in others.items():
    mu = mu_of(p)
    deg = np.degrees(mu)
    print(f"{name}: pos={p}, mu={mu:.3f} deg = {deg:.1f}°")

# quick plot
plt.figure(figsize=(5,5))
plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
plt.scatter([0],[0], c='red', label='focal')
for name, p in others.items():
    plt.scatter(p[0], p[1], label=f"{name} ({np.round(p,2)})")
    plt.text(p[0]+0.02, p[1]+0.02, name)
# draw forward arrow
plt.arrow(0,0, 0.2, 0, head_width=0.03, color='red')
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1.0, 1.0); plt.ylim(-1.0, 1.0)
plt.legend()
plt.title("Top-down view: focal faces +x")
plt.show()
