from world import Fish, Spot, Tank
from perception.perception_model import perceive
import config
from pprint import pprint
import pygame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from orientation_PDF import total_f, THETA_GRID, sample_from_pdf
from experimental_data.speed_PDF import SPEED_PDF_MAP
from draw_utils import draw_fish, draw_spot, draw_tank, world_to_screen, screen_to_world

def calculate_next_fish_state(tank: Tank, fish : Fish, perception, time_step):
    d = perception["wall_state"]["distance"] # distance from wall
    mu_w = np.array([perception["wall_state"]["mu_w1"], perception["wall_state"]["mu_w2"]]) # wall tangents
    A_f = np.array([fish["A"] for fish in perception["fish"]]) # sizes of percieved fishies
    mu_f = np.array([fish["mu"] for fish in perception["fish"]]) # directions of percieved fishies
    A_s = np.array([fish["A"] for fish in perception["spots"]]) # sizes of percieved spots
    mu_s = np.array([fish["mu"] for fish in perception["spots"]]) # directions of percieved spots

    # pprint(perception)
    near_wall = d < config.PDF_DW
    pdf_values = total_f(THETA_GRID, near_wall, mu_w, A_f, A_s, mu_f, mu_s)
    sees_spots = A_s.size != 0
    under_spot = perception["under_spot"]
    sees_fish = A_f.size != 0
    speed_bins, speed_pdf = SPEED_PDF_MAP[(sees_fish, sees_spots, under_spot)]
    # print("sees_fish", sees_fish, "sees_spots", sees_spots, "under_spot", under_spot)
    # speed_bins, speed_pdf = SPEED_PDF_MAP[(False, False, False)] # swimming alone

    fish.next_position = fish.position + fish.speed * time_step
    fish.next_position[0] = np.clip(fish.next_position[0], tank.xmin + config.FISH_LENGTH, tank.xmax - config.FISH_LENGTH)
    fish.next_position[1] = np.clip(fish.next_position[1], tank.ymin + config.FISH_LENGTH, tank.ymax - config.FISH_LENGTH)
    
    fish.next_orientation = fish.orientation + sample_from_pdf(THETA_GRID, pdf_values)
    fish.next_orientation = (fish.next_orientation + np.pi) % (2 * np.pi) - np.pi
    fish.next_speed = sample_from_pdf(speed_bins, speed_pdf) * np.array([np.cos(fish.next_orientation), np.sin(fish.next_orientation)])
    
def fish_loop(fishies, spots, tank, time_step):
    fish : Fish
    # save previous fish states for interpolation
    for fish in fishies:
        fish.prev_position = fish.position.copy()
        fish.prev_orientation = fish.orientation

    # calculate new fish states
    for fish in fishies:
        if fish.dragged: continue
        perception = perceive(fish, fishies, spots, tank)
        calculate_next_fish_state(tank, fish, perception, time_step=time_step)

    # update fishies
    for fish in fishies:
        if fish.dragged: continue
        fish.update()


def run_and_save_sim(tank, fishies, spots, duration_s, save_path=None):
    positions = []
    for i in range(duration_s * config.FISH_FPS):
        if i % (60 * config.FISH_FPS) == 0:
            print(f"{i / config.FISH_FPS}/{duration_s}s")
        
        fish_loop(fishies, spots, tank, config.FISH_TIME_STEP)
        for fish in fishies:
            positions.append([float(i / config.FISH_FPS), fish.id, fish.position[0], fish.position[1]])

    positions = pd.DataFrame(positions, columns=["time", "fish_id", "x", "y"])
    if save_path is None:
        for fish_id, group in positions.groupby("fish_id"):
            plt.plot(group["x"], group["y"])

        plt.show()
    else:
        positions.to_csv(f"{save_path}.csv", index=False)


tank = Tank(config.TANK_WIDTH, config.TANK_HEIGHT, origin_at_center=True)

np.random.seed(1)
num_fishies = 1
num_fishies = 10
# num_fishies = 40
# fishies = [Fish(0.40, 0.35, 0)]
# fishies = [Fish(0.40, 0.40, np.pi), Fish(0.30, 0.40, 0)]
fishies = [Fish(x, y, theta) for x, y, theta in zip(
    np.random.uniform(tank.xmin, tank.xmax, num_fishies),
    np.random.uniform(tank.ymin, tank.ymax, num_fishies),
    np.random.uniform(0, 2*np.pi, num_fishies)
)]

spots = [
    Spot(0.35, 0.35, config.SPOT_RADIUS, config.SPOT_HEIGHT),
    Spot(-0.35, -0.35, config.SPOT_RADIUS, config.SPOT_HEIGHT),
]

# pprint(perceive(fishies[0], fishies, spots, tank))
# exit()

# import time
# time_start = time.time()
# # run_and_save_sim(tank, fishies, spots, 60*60*10, "simulations/Homogeneous_1AB_fast")
# # run_and_save_sim(tank, fishies, spots, 60*60*10, "simulations/Homogeneous_10AB_fast")
# # run_and_save_sim(tank, fishies, spots, 60*60*10, "simulations/Heterogeneous_1AB_fast")
# run_and_save_sim(tank, fishies, spots, 60*60*10, "simulations/Heterogeneous_10AB_fast")
# # run_and_save_sim(tank, fishies, spots, 60*10, None)
# time_end = time.time() - time_start
# print(f"Time: {time_end:.3f} seconds")
# exit()

pygame.init()
screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()

elapsed, running = 0.0, True
dragged_fish = None
while running:
    dt = clock.tick(config.DISPLAY_FPS) / 1000.0  # seconds since last frame
    elapsed += dt

    for e in pygame.event.get():
        if e.type == pygame.QUIT: 
            running = False
        elif e.type == pygame.VIDEORESIZE:
            config.WIDTH, config.HEIGHT = e.w, e.h
            config.SCALE = config.scale(config.WIDTH, config.HEIGHT)
            screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.RESIZABLE)
        elif e.type == pygame.MOUSEBUTTONDOWN:
            p = np.array(e.pos)
            for fish in fishies:
                if np.linalg.norm(world_to_screen(*fish.position) - p) < 20: 
                    fish.dragged, dragged_fish = True, fish 
                    break
        elif e.type == pygame.MOUSEBUTTONUP:
            if dragged_fish: 
                dragged_fish.dragged, dragged_fish = False, None
        elif e.type == pygame.MOUSEMOTION and dragged_fish:
            x, y = screen_to_world(*e.pos)
            dragged_fish.position[0] = np.clip(x, tank.xmin + config.FISH_LENGTH, tank.xmax - config.FISH_LENGTH)
            dragged_fish.position[1] = np.clip(y, tank.ymin + config.FISH_LENGTH, tank.ymax - config.FISH_LENGTH)

    # update every FISH_TIME_STEP seconds
    if elapsed >= config.FISH_TIME_STEP:
        elapsed = 0
        fish_loop(fishies, spots, tank, config.FISH_TIME_STEP)

    # draw everything
    screen.fill((15, 15, 30))  # background
    draw_tank(screen, tank)

    # interpolate fish position and orientation
    t = elapsed / config.FISH_TIME_STEP
    for fish in fishies:
        draw_fish(screen, fish, t)

    for spot in spots:
        draw_spot(screen, spot)

    pygame.display.flip()

pygame.quit()
