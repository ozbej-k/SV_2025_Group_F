# pygame simulation values
FISH_FPS = 3 # zebrafish beat their tails about 3 times per second on average
FISH_TIME_STEP = 1.0 / FISH_FPS
DISPLAY_FPS = 60
WIDTH, HEIGHT = 800, 600
def scale(width, height): return min(width, height) * 0.65
SCALE = scale(WIDTH, HEIGHT)

# Tank geometry (meters)
TANK_WIDTH = TANK_HEIGHT = 1.20

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
PDF_K0 = 6.3  # basic-swimming dispersion
PDF_KW = 20  # wall-following dispersion
PDF_KF = 20  # percieved fish dispersion
PDF_KS = 20  # spot of interest dispersion

PDF_KWB = 3

# Weights
PDF_ALPHA_0 = 27.5  # weight of the perceived fish during basic-swimming
PDF_ALPHA_W = 10  # weight of the perceived fish during wall-following
PDF_BETA_0 = 1  # weight of the perceived spots during basic-swimming
PDF_BETA_W = 0.01  # weight of the perceived spots during wall-following

# Factors for weight
PDF_WF = 1/2  # factor of weight for alpha_0 and alpha_w when fish and spots are present
PDF_WS = 1/9  # factor of weight for beta_0 and beta_w when fish and spots are present

# Fish perception mode: "fast", "mesh", "both"
FISH_PERCEPTION_MODE = "mesh"

# threshold for debugging differences
FISH_DEBUG_EPS = 1e-6