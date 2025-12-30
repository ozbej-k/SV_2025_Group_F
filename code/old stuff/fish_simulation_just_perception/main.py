"""Main script to demonstrate fish perception simulation."""

import logging
from world.tank import Tank
from world.fish_state import Fish
from world.spot import Spot
from perception.perception_model import perceive
from perception.perception_output import build_perception_summary
from config.fish_params import SPOT_RADIUS, SPOT_HEIGHT
#from config.fish_params import FISH_LENGTH, FISH_WIDTH, FISH_HEIGHT

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("fish_sim")

def demo():
    # Create tank and objects
    tank = Tank(width=1.20, height=1.20, origin_at_center=True)

    # create focal fish at center facing +x
    focal = Fish(0.55, 0.5, 0.0, id_given='focal')

    # create another fish slightly ahead-left
    other1 = Fish(0.10, 0.05, 0.5, id_given='other1')
    other2 = Fish(0, -0.6, 0, id_given='other2')
    # create a spot at upper-right quadrant
    spot = Spot(0.05, 0.05, SPOT_RADIUS, SPOT_HEIGHT)

    perception = perceive(focal, [other1, other2], [spot], tank)
    summary = build_perception_summary(perception)

    print("Perception summary:")
    import pprint
    pprint.pprint(summary)

if __name__ == "__main__":
    demo()
