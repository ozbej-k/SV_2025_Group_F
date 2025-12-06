import numpy as np

class Fish:
    _next_id = 0
    
    def __init__(self, x, y, orientation, id_given=None):
        """
        id_given: optional unique identifier, if None then assigned automatically
        x,y: position in meters (world coordinates), z assumed 0
        orientation: heading angle in radians (0 along world +x)
        """
        self.position = np.array([float(x), float(y)])
        self.orientation = float(orientation)
        if id_given is not None:
            self.id = id_given
        else:
            self.id = Fish._next_id
            Fish._next_id += 1

    def __repr__(self):
        return f"Fish(id={self.id}, pos=({self.position[0]:.3f},{self.position[1]:.3f}), ori={self.orientation:.3f})"
