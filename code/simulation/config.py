# pygame simulation values
FISH_FPS = 3 # zebrafish beat their tails about 3 times per second on average
FISH_TIME_STEP = 1.0 / FISH_FPS
DISPLAY_FPS = 60
WIDTH, HEIGHT = 1050, 650
SIDEBAR_WIDTH = 0.4
LINE_WIDTH = 0.005
def scale(width, height): return min(width, height) * 0.65
SCALE = scale(WIDTH, HEIGHT)

# Tank geometry (meters)
TANK_WIDTH = 1.6
TANK_HEIGHT = 1.20
GRID_CELL_SIZE = 0.01
BRUSH_RADIUS = 1

# Fish geometry (meters) - values from the paper
FISH_LENGTH = 0.035  # 3.5 cm
FISH_WIDTH = 0.01    # 1 cm
FISH_HEIGHT = 0.01   # 1 cm

# Spot (disc) geometry
SPOT_RADIUS = 0.10   # 10 cm (paper uses 0.1 m)
SPOT_HEIGHT = 0.05   # 5 cm above plane

# Field of view
FOV_DEGREES = 270.0
FOV_HALF = FOV_DEGREES / 2.0  # 135 deg

# Wall interaction distance
PDF_DW = 0.05  # meters

# Dispersion parameters
PDF_K0 = 6.3 # basic-swimming dispersion
PDF_KW = 20  # wall-following dispersion
PDF_KF = 20  # percieved fish dispersion
PDF_KS_0 = 10  # spot of interest dispersion when outside spot
PDF_KS_S = 0.5   # spot of interest dispersion when under spot 

PDF_KWB = 3

# Weights
PDF_ALPHA_0 = 7    # weight of the perceived fish during basic-swimming
PDF_ALPHA_W = 2    # weight of the perceived fish during wall-following
PDF_BETA_0 = 0.25   # weight of the perceived spots during basic-swimming
PDF_BETA_W = 0.125  # weight of the perceived spots during wall-following

PDF_ALPHA_0B = 9    # weight of the perceived fish during basic-swimming when fish and spots are present
PDF_ALPHA_WB = 2    # weight of the perceived fish during wall-following when fish and spots are present
PDF_BETA_0B = 0.25   # weight of the perceived spots during basic-swimming when fish and spots are present
PDF_BETA_WB = 0.125  # weight of the perceived spots during wall-following when fish and spots are present
