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
        self.speed = np.zeros(2)
        if id_given is not None:
            self.id = id_given
        else:
            self.id = Fish._next_id
            Fish._next_id += 1

        # temporary values inbetween updates
        self.next_position = None
        self.next_orientation = None
        self.next_speed = None
        self.dragged = False
        self.selected = False

        # interpolation values
        self.prev_position = self.position.copy()
        self.prev_orientation = self.orientation
    
    def update(self):
        self.position = self.next_position
        self.orientation = self.next_orientation
        self.speed = self.next_speed

    def __repr__(self):
        return f"Fish(id={self.id}, position=({self.position[0]:.3f},{self.position[1]:.3f}), orientation={self.orientation:.3f}, speed=({self.speed[0]:.3f},{self.speed[1]:.3f}))"
