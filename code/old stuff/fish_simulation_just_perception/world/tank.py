class Tank:
    def __init__(self, width=1.20, height=1.20, xmin=0.0, ymin=0.0, origin_at_center=False):
        # default experimental tank 1.20 x 1.20 m (from paper)
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
