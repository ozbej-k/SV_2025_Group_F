import math

def distance_to_walls_2d(pos_xy, tank):
    """
    pos_xy: (x,y)
    tank: object with xmin,xmax,ymin,ymax
    returns (distance, nearest_wall) where nearest_wall in {'left','right','bottom','top'}
    """
    x, y = pos_xy
    d_left = x - tank.xmin
    d_right = tank.xmax - x
    d_bottom = y - tank.ymin
    d_top = tank.ymax - y
    distances = {
        'left': d_left,
        'right': d_right,
        'bottom': d_bottom,
        'top': d_top
    }
    nearest = min(distances, key=distances.get)
    return distances[nearest], nearest

def wall_tangent_directions(nearest_wall):
    """
    Returns two tangential directions (unit vectors in world frame) along the wall.
    Represented as angles in radians where 0 = +x world.
    For left/right walls, tangential directions are +y and -y.
    For top/bottom walls, tangential directions are +x and -x.
    """
    
    if nearest_wall in ('left', 'right'):
        # tangential directions along world +y and -y
        return (math.pi/2.0, -math.pi/2.0)
    else:
        # top/bottom -> tangential along +x and -x
        return (0.0, math.pi)
