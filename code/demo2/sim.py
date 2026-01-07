import pygame

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
from ui_utils import *

def calculate_next_fish_state(tank: Tank, fish: Fish, perception, time_step):
    #d = perception["wall_state"]["distance"]  # distance from wall
    #mu_w = np.array([perception["wall_state"]["mu_w1"], perception["wall_state"]["mu_w2"]])  # wall tangents
    d = perception["wall_state"]["distance"]
    mu_w_list = perception["wall_state"]["mu_w1"] + perception["wall_state"]["mu_w2"]
    mu_w = np.array([mu for mu in mu_w_list if mu is not None])
    A_f = np.array([fish["A"] for fish in perception["fish"]]) # sizes of percieved fishies
    mu_f = np.array([fish["mu"] for fish in perception["fish"]])  # directions of percieved fishies
    A_s = np.array([fish["A"] for fish in perception["spots"]])  # sizes of percieved spots
    mu_s = np.array([fish["mu"] for fish in perception["spots"]])  # directions of percieved spots

    #near_wall = d < config.PDF_DW
    near_wall = d < config.PDF_DW if isinstance(d, float) else (len(d) > 0 and min(d) < config.PDF_DW)
    pdf_values = total_f(THETA_GRID, near_wall, mu_w, A_f, A_s, mu_f, mu_s)
    sees_spots = A_s.size != 0
    under_spot = perception["under_spot"]
    sees_fish = A_f.size != 0
    speed_bins, speed_pdf = SPEED_PDF_MAP[(sees_fish, sees_spots, under_spot)]
    # print("sees_fish", sees_fish, "sees_spots", sees_spots, "under_spot", under_spot)
    # speed_bins, speed_pdf = SPEED_PDF_MAP[(False, False, False)] # swimming alone 

    # new orientation and speed
    new_orientation = fish.orientation + sample_from_pdf(THETA_GRID, pdf_values)
    new_orientation = (new_orientation + np.pi) % (2 * np.pi) - np.pi
    sampled_speed = sample_from_pdf(speed_bins, speed_pdf)
    new_speed_vec = sampled_speed * np.array([np.cos(new_orientation), np.sin(new_orientation)])

    intended_position = fish.position + new_speed_vec * time_step

    # wall handling
    if tank.ray_intersects_wall(fish.position, intended_position) or tank.is_wall_near(*intended_position, buffer=config.FISH_LENGTH/2):
        intended_position = fish.position.copy()
        new_speed_vec = np.array([0.0, 0.0])

    # Update fish state
    fish.next_position = np.clip(intended_position, 
                                 [tank.xmin + config.FISH_LENGTH, tank.ymin + config.FISH_LENGTH],
                                 [tank.xmax - config.FISH_LENGTH, tank.ymax - config.FISH_LENGTH])
    fish.next_orientation = new_orientation
    fish.next_speed = new_speed_vec
    return pdf_values
    
def fish_loop(fishies, spots, tank, time_step, paused=False):
    fish : Fish
    # save previous fish states for interpolation
    for fish in fishies:
        fish.prev_position = fish.position.copy()
        fish.prev_orientation = fish.orientation

    # calculate new fish states
    pdf_values = None
    for fish in fishies:
        if fish.dragged: continue
        perception = perceive(fish, fishies, spots, tank)
        
        if fish is selected_fish:
            pdf_values = calculate_next_fish_state(tank, fish, perception, time_step=time_step)
        else:
            calculate_next_fish_state(tank, fish, perception, time_step=time_step)

    # update fishies
    if not paused:
        for fish in fishies:
            if fish.dragged: continue
            fish.update()
    return pdf_values

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

num_fishies = 10
fishies = [Fish(x, y, theta) for x, y, theta in zip(
    np.random.uniform(tank.xmin, tank.xmax, num_fishies),
    np.random.uniform(tank.ymin, tank.ymax, num_fishies),
    np.random.uniform(0, 2*np.pi, num_fishies)
)]

spots = [
    Spot(0.35, 0.35, config.SPOT_RADIUS, config.SPOT_HEIGHT),
    Spot(-0.35, -0.35, config.SPOT_RADIUS, config.SPOT_HEIGHT),
]

prev_x, prev_y = None, None  
tank.load_tank("test_tank.png")

# ---- fast simulation ----

# run_and_save_sim(tank, fishies, spots, 600, None)
# run_and_save_sim(tank, fishies, spots, 600, "simulations/Heterogeneous_10AB")
# exit()

# ---- interactive simulation ----

def add_fish(): fishies.append(Fish(0.0, 0.0, np.random.uniform(0, 2*np.pi)))
def add_spot(): spots.append(Spot(0.0, 0.0, config.SPOT_RADIUS, config.SPOT_HEIGHT))
def remove_fish(): fishies.pop() if len(fishies) > 0 else None 
def remove_spot(): spots.pop() if len(spots) > 0 else None 

pygame.init()
clock = pygame.time.Clock()
screen, font = recalculate_scale_and_positions(config.WIDTH, config.HEIGHT, tank, add_fish, add_spot, remove_fish, remove_spot)
Button.select_button(Button.MOVE_FISH)

elapsed, running = 0.0, True
dragged_fish = selected_fish = dragged_spot = None
selected_fish = fishies[0]
selected_fish.selected = True
selected_fish_orientation_pdf = None
paused = False
while running:
    dt = clock.tick(config.DISPLAY_FPS) / 1000.0  # seconds since last frame
    elapsed += dt

    for e in pygame.event.get():
        for btn in UI.buttons:
            if btn.handle_event(e):
                for other in UI.buttons:
                    if other != btn:
                        other.selected = False
            
        if e.type == pygame.QUIT: 
            running = False
        elif e.type == pygame.VIDEORESIZE:
            screen, font = recalculate_scale_and_positions(e.w, e.h, tank, add_fish, add_spot, remove_fish, remove_spot)
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_s:
                Button.select_button(Button.MOVE_SPOT)
            if e.key == pygame.K_f:
                Button.select_button(Button.MOVE_FISH)
            if e.key == pygame.K_d:
                Button.select_button(Button.DRAW_BUTTON)
                prev_x, prev_y = None, None
            elif e.key == pygame.K_SPACE: # pause
                paused = not paused
            elif e.key == pygame.K_i: # save tank grid
                tank.save_tank("current_tank.png")
                print("saved tank to current_tank.png")
            elif e.key == pygame.K_c: # clear tank grid
                UI.buttons[Button.CLEAR_TANK].function()
        elif e.type == pygame.MOUSEBUTTONDOWN: # select / drag fish or spot begin
            p = np.array(e.pos)
            if not UI.tank_rect.collidepoint(e.pos):
                continue
            if Button.BUTTON_IS_SELECTED[Button.MOVE_FISH]:
                selected_fish = None
                for fish in fishies:
                    fish.selected = False
                    if selected_fish is None and np.linalg.norm(world_to_screen(*fish.position) - p) < 20: 
                        fish.dragged, dragged_fish = True, fish
                        selected_fish = fish
                        selected_fish.selected = True
            elif Button.BUTTON_IS_SELECTED[Button.MOVE_SPOT]:
                for spot in spots:
                    if dragged_spot is None and np.linalg.norm(world_to_screen(*spot.position[:2]) - p) < spot.radius * config.SCALE: 
                        spot.dragged, dragged_spot = True, spot
        elif e.type == pygame.MOUSEBUTTONUP: # drag fish or spot end
            if Button.BUTTON_IS_SELECTED[Button.MOVE_FISH] and dragged_fish:
                dragged_fish.dragged, dragged_fish = False, None
            elif Button.BUTTON_IS_SELECTED[Button.MOVE_SPOT] and dragged_spot:
                dragged_spot.dragged, dragged_spot = False, None
        elif e.type == pygame.MOUSEMOTION and (dragged_fish or dragged_spot): # drag fish or spot
            dragged_object, object_size = None, None
            if Button.BUTTON_IS_SELECTED[Button.MOVE_FISH]:
                dragged_object = dragged_fish
                object_size = config.FISH_LENGTH
            elif Button.BUTTON_IS_SELECTED[Button.MOVE_SPOT]:
                dragged_object = dragged_spot
                object_size = config.SPOT_RADIUS
            if dragged_object is not None:
                x, y = screen_to_world(*e.pos)
                dragged_object.position[0] = np.clip(x, tank.xmin + object_size, tank.xmax - object_size)
                dragged_object.position[1] = np.clip(y, tank.ymin + object_size, tank.ymax - object_size)
        elif e.type == pygame.MOUSEMOTION and Button.BUTTON_IS_SELECTED[Button.DRAW_BUTTON] and e.buttons[0]: # draw tank grid
            x, y = screen_to_world(*e.pos)
            tank.grid_draw(x, y, prev_x, prev_y, config.BRUSH_RADIUS)
            prev_x, prev_y = x, y
        
    if Button.BUTTON_IS_SELECTED[Button.DRAW_BUTTON] and not pygame.mouse.get_pressed()[0]:
        prev_x, prev_y = None, None 

    # update every FISH_TIME_STEP seconds
    if elapsed >= config.FISH_TIME_STEP:
        elapsed = 0
        selected_fish_orientation_pdf = fish_loop(fishies, spots, tank, config.FISH_TIME_STEP, paused)

    # draw everything
    draw_frame(screen, font, THETA_GRID, elapsed, tank, fishies, spots, selected_fish, selected_fish_orientation_pdf)
    
    pygame.display.flip()

pygame.quit()
