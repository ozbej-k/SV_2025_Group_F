class Spot:
    
    _next_id = 0
    
    def __init__(self, x, y, radius, height, id_given=None):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.height = float(height)
        if id_given is not None:
            self.id = id_given
        else:
            self.id = Spot._next_id
            Spot._next_id += 1

    def __repr__(self):
        return f"Spot(x={self.x:.3f}, y={self.y:.3f}, r={self.radius:.3f}, h={self.height:.3f})"
