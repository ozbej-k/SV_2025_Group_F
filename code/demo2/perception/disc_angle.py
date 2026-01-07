import numpy as np

_COEF_INNER = np.array([
    4.05359822e+09, -5.61259822e+09, 3.09872672e+09, -8.98779297e+08, 
    1.49497597e+08, -1.45392685e+07, 8.19266576e+05, -2.62906486e+04, 
    2.93703430e+02, -2.90517239e+00, 3.47612779e+00
])

def disc_solid_angle(distance): # distance x y, not z
    if distance <= 0.2:
        return np.polyval(_COEF_INNER, distance)
    else:
        return (0.12591490890012613 / distance)**3.176085672765036
