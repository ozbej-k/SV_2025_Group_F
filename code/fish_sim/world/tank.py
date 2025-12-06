import numpy as np
import config

class Tank:
    def __init__(self, width=1.20, height=1.20, xmin=0.0, ymin=0.0, origin_at_center=False):
        # default experimental tank 1.20 x 1.20 m (from paper)
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

    def tangent_wall_directions(self, pos):
        tank_m = np.array([[self.xmin, self.ymin], [self.xmax, self.ymax]])
        diff = np.abs(np.array([pos - tank_m[0], tank_m[1] - pos]))
        nearest = diff.min(0)
        
        d_nearest = diff.min()
        if d_nearest >= config.PDF_DW:
            return d_nearest, None, None

        if np.abs(nearest[0] - nearest[1]) < config.PDF_DW:  # corner
            ix, iy = np.argmin(diff[:,0]), np.argmin(diff[:,1])
            # ix: 0 = left, 1 = right
            # iy: 0 = bottom, 1 = top

            horiz = 0 if ix == 0 else np.pi
            vert  = 3*np.pi/2 if iy == 1 else np.pi/2

            return d_nearest, vert, horiz

        if nearest[0] < nearest[1]: # left right wall
            return d_nearest, np.pi/2, 3*np.pi/2
        else: # top / bottom wall
            return d_nearest, 0, np.pi
