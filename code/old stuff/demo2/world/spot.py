import numpy as np

class Spot:
    _next_id = 0
    
    def __init__(self, x, y, radius, height, id_given=None):
        self.position = np.array([x, y, height], dtype=float)
        self.radius = float(radius)
        self.dragged = False
        if id_given is not None:
            self.id = id_given
        else:
            self.id = Spot._next_id
            Spot._next_id += 1

    def __repr__(self):
        return f"Spot(position={self.position}, r={self.radius:.3f})"
