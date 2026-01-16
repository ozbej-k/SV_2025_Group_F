import numpy as np
import config
from ui_utils import world_to_grid
from PIL import Image

class Tank:
    RAYCAST_N_RAYS = 8     # Number of rays around the fish
    RAYCAST_STEP_FRAC = 0.5    # Step fraction of cell size
    def __init__(self, width=1.20, height=1.20, xmin=0.0, ymin=0.0, origin_at_center=False):
        self.width = width
        self.height = height
        if origin_at_center:
            self.xmin = -width/2.0
            self.xmax = width/2.0
            self.ymin = -height/2.0
            self.ymax = height/2.0
        else:
            self.xmin = xmin
            self.xmax = xmin + width
            self.ymin = ymin
            self.ymax = ymin + height

        # wall grid
        self.wall_grid = np.ones((int(self.height/config.GRID_CELL_SIZE), int(self.width/config.GRID_CELL_SIZE)), dtype=bool)
        self.wall_grid[1:-1, 1:-1] = 0
        self.grid_height, self.grid_width = self.wall_grid.shape

    def save_tank(self, path):
        img_uint8 = self.wall_grid.astype(np.uint8) * 255
        Image.fromarray(img_uint8, mode="L").save(path)

    def load_tank(self, path):
        img = Image.open(path).convert("L")
        img_grid = np.array(img) > 0
        
        self.wall_grid = np.ones((int(self.height/config.GRID_CELL_SIZE), int(self.width/config.GRID_CELL_SIZE)), dtype=bool)
        self.wall_grid[1:-1, 1:-1] = 0

        self.wall_grid[
            :min(self.wall_grid.shape[0], img_grid.shape[0]), 
            :min(self.wall_grid.shape[1], img_grid.shape[1])
        ] |= img_grid[
            :min(self.wall_grid.shape[0], img_grid.shape[0]),
            :min(self.wall_grid.shape[1], img_grid.shape[1])
        ]

    def is_wall_at(self, x, y):
        """Check if there's a wall at world coordinates (x, y)"""
        if self.wall_grid is None:
            return False
        
        gx, gy = world_to_grid(self, x, y)
        if gx is None:
            return False
        return self.wall_grid[gy, gx]
    
    def grid_draw(self, x, y, prev_x, prev_y, brush_radius):
        def brush(gx, gy, radius):
            gx = int(gx)
            gy = int(gy)
            h, w = self.wall_grid.shape

            y_min = max(0, gy - radius)
            y_max = min(h, gy + radius + 1)
            x_min = max(0, gx - radius)
            x_max = min(w, gx + radius + 1)

            y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
            mask = (x_indices - gx)**2 + (y_indices - gy)**2 <= radius**2
            self.wall_grid[y_min:y_max, x_min:x_max][mask] = True

        if prev_x is not None and prev_y is not None:
            dx = x - prev_x
            dy = y - prev_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                steps = max(1, int(dist / 0.005))
                for i in range(steps + 1):
                    ix = prev_x + i * dx / steps
                    iy = prev_y + i * dy / steps
                    gx, gy = world_to_grid(self, ix, iy)
                    brush(gx, gy, brush_radius)
        else:
            gx, gy = world_to_grid(self, x, y)
            brush(gx, gy, brush_radius)

    def ray_intersects_wall(self, start_pos, end_pos, num_samples=20):
        for i in range(num_samples + 1):
            t = i / num_samples
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            if self.is_wall_at(x, y):
                return True
        return False

    def tangent_wall_directions(self, pos, orientation):
        distances, mu_w = [], []
        drawn_wall_segments = self._raycast_drawn_wall_tangents(pos, orientation)
        for s in drawn_wall_segments:
            if s["distance"] <= config.PDF_DW:
                distances.append(s["distance"])
                mu_w.append(s["mu_w1"])
                mu_w.append(s["mu_w2"])

        return distances, mu_w

    def is_wall_near(self, x, y, buffer=0.05):
        """
        Returns True if the fish would intersect the wall considering its size.
        """
        for dx in [-buffer, 0, buffer]:
            for dy in [-buffer, 0, buffer]:
                if self.is_wall_at(x + dx, y + dy):
                    return True
        return False

    def _raycast_drawn_wall_tangents(self, pos, orientation, max_dist=(config.PDF_DW * 2)):
        """
        Cast rays around the fish to detect drawn-wall segments.
        Returns a list of dicts: {"mu_w1", "mu_w2", "distance"}.
        """
        if self.wall_grid is None:
            return []

        cell_size = self.width / self.grid_width
        step = cell_size * self.RAYCAST_STEP_FRAC
        angles = np.linspace(-np.pi, np.pi, self.RAYCAST_N_RAYS, endpoint=False) + orientation
        hits = []

        # 1 Cast rays
        for theta in angles:
            hit_dist = None
            num_steps = int(max_dist / step) + 1
            for i in range(num_steps):
                px = pos[0] + i * step * np.cos(theta)
                py = pos[1] + i * step * np.sin(theta)
                if self.is_wall_at(px, py):
                    hit_dist = i * step
                    break
            hits.append(hit_dist)

        # 2 Identify wall directions
        wall_segments = []
        n = len(hits)

        if hits[0] is not None and hits[-1] is not None:
            # Find where the initial segment ends
            i = 0
            while i < n and hits[i] is not None:
                i += 1
            end_of_start = i - 1
            
            # Find where the final segment begins
            j = n - 1
            while j >= 0 and hits[j] is not None:
                j -= 1
            start_of_end = j + 1
            
            # If they're actually separate segments (there's a gap between)
            if end_of_start < start_of_end:
                # Merge into one wrap-around segment
                distances = []
                for k in range(start_of_end, n):
                    distances.append(hits[k])
                for k in range(0, end_of_start + 1):
                    distances.append(hits[k])
                
                mu_w1 = angles[start_of_end - 1]
                mu_w2 = angles[(end_of_start + 1) % n]
                
                wall_segments.append({
                    "mu_w1": mu_w1,
                    "mu_w2": mu_w2,
                    "distance": min(distances)
                })
                
                # Process the middle part
                i = end_of_start + 1
                while i < start_of_end:
                    if hits[i] is not None:
                        start = i
                        distances = []
                        while i < start_of_end and hits[i] is not None:
                            distances.append(hits[i])
                            i += 1
                        end = i - 1
                        
                        mu_w1 = angles[start - 1]
                        mu_w2 = angles[(end + 1) % n]
                        
                        wall_segments.append({
                            "mu_w1": mu_w1,
                            "mu_w2": mu_w2,
                            "distance": min(distances)
                        })
                    else:
                        i += 1
            else:
                # All hits - full circle
                wall_segments.append({
                    "mu_w1": angles[-1],
                    "mu_w2": angles[0],
                    "distance": min(hits)
                })
        else:
            i = 0
            while i < n:
                if hits[i] is not None:
                    start = i
                    distances = []
                    while hits[i % n] is not None:
                        distances.append(hits[i % n])
                        i += 1
                        if i - start >= n:
                            break
                    end = (i - 1) % n
                    # Compute tangent angles
                    mu_w1 = angles[start-1] if start > 0 else angles[-1]
                    mu_w2 = angles[(end+1) % n]
                    wall_segments.append({
                        "mu_w1": mu_w1,  
                        "mu_w2": mu_w2,
                        "distance": min(distances)
                    })
                else:
                    i += 1

        return wall_segments
    