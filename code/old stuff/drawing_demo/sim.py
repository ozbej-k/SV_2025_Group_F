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
from PIL import Image

def save_tank(tank : Tank, path):
    img_uint8 = tank.wall_grid.astype(np.uint8) * 255
    Image.fromarray(img_uint8, mode="L").save(path)

def load_tank(tank : Tank, path):
    img = Image.open(path).convert("L")
    tank.wall_grid = np.array(img) > 0

def calculate_next_fish_state(tank: Tank, fish: Fish, perception, time_step):
    d = perception["wall_state"]["distance"]  # distance from wall
    mu_w = np.array([perception["wall_state"]["mu_w1"], perception["wall_state"]["mu_w2"]])  # wall tangents
    A_f = np.array([fish["A"] for fish in perception["fish"]]) # sizes of percieved fishies
    mu_f = np.array([fish["mu"] for fish in perception["fish"]])  # directions of percieved fishies
    A_s = np.array([fish["A"] for fish in perception["spots"]])  # sizes of percieved spots
    mu_s = np.array([fish["mu"] for fish in perception["spots"]])  # directions of percieved spots

    near_wall = d < config.PDF_DW
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
    if tank.ray_intersects_wall(fish.position, intended_position) or tank.is_wall_near(*intended_position, buffer=0.02):
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

def world_to_grid(x, y):
    # Map world coords to grid indices
    gx = int((x - tank.xmin) / (tank.xmax - tank.xmin) * GRID_WIDTH)
    gy = int((y - tank.ymin) / (tank.ymax - tank.ymin) * GRID_HEIGHT)
    return np.clip(gx, 0, GRID_WIDTH-1), np.clip(gy, 0, GRID_HEIGHT-1)

def grid_to_world(gx, gy):
    # Map grid to world (for rendering)
    x = tank.xmin + (gx / GRID_WIDTH) * (tank.xmax - tank.xmin)
    y = tank.ymin + (gy / GRID_HEIGHT) * (tank.ymax - tank.ymin)
    return x, y 


tank = Tank(config.TANK_WIDTH, config.TANK_HEIGHT, origin_at_center=True)


num_fishies = 1
# fishies = [Fish(x, y, theta) for x, y, theta in zip(
#     np.random.uniform(tank.xmin, tank.xmax, num_fishies),
#     np.random.uniform(tank.ymin, tank.ymax, num_fishies),
#     np.random.uniform(0, 2*np.pi, num_fishies)
# )]
fishies = [Fish(0, 0, 0)]

spots = [
    Spot(0.35, 0.35, config.SPOT_RADIUS, config.SPOT_HEIGHT),
    Spot(-0.35, -0.35, config.SPOT_RADIUS, config.SPOT_HEIGHT),
]

# Pixel grid for drawn walls (True = wall)
GRID_WIDTH, GRID_HEIGHT = 120, 120 
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
cell_screen_size = int((tank.width / GRID_WIDTH) * config.SCALE) 
draw_mode = False
prev_x, prev_y = None, None  
brush_radius = 1 
tank.set_wall_grid(grid)
load_tank(tank, "test_tank.png")

# grid[65:80, :] = True
# fish_loop(fishies, spots, tank, config.FISH_FPS)
# plt.imshow(grid)
# plt.show()

# run_and_save_sim(tank, fishies, spots, 600, None)
# run_and_save_sim(tank, fishies, spots, 600, "simulations/Heterogeneous_10AB")
# exit()


pygame.init()
font = pygame.font.SysFont("Arial", 12)
screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()

elapsed, running = 0.0, True
dragged_fish = selected_fish = None
selected_fish = fishies[0]
selected_fish.selected = True
selected_fish_orientation_pdf = None
paused = False
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
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_d:  #draw mode with 'D' key
                draw_mode = not draw_mode
                prev_x, prev_y = None, None
                print(f"Draw mode: {draw_mode}")
            elif e.key == pygame.K_SPACE:
                paused = not paused
                print("paused", paused)
            elif e.key == pygame.K_s:
                save_tank(tank, "current_tank.png")
                print("saved tank to current_tank.png")
            elif e.key == pygame.K_c:
                tank.wall_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
                print("cleared drawn tank")
        elif e.type == pygame.MOUSEBUTTONDOWN:
            p = np.array(e.pos)
            _selected = None if not draw_mode else selected_fish
            for fish in fishies:
                fish.selected = False
                if np.linalg.norm(world_to_screen(*fish.position) - p) < 20: 
                    fish.dragged, dragged_fish = True, fish
                    _selected = fish
                    break
            
            if _selected is not None:
                selected_fish = _selected
                selected_fish.selected = True
            elif not draw_mode:
                selected_fish = None
                selected_fish_orientation_pdf = None
        elif e.type == pygame.MOUSEBUTTONUP:
            if dragged_fish: 
                dragged_fish.dragged, dragged_fish = False, None
        elif e.type == pygame.MOUSEMOTION and dragged_fish:
            x, y = screen_to_world(*e.pos)
            dragged_fish.position[0] = np.clip(x, tank.xmin + config.FISH_LENGTH, tank.xmax - config.FISH_LENGTH)
            dragged_fish.position[1] = np.clip(y, tank.ymin + config.FISH_LENGTH, tank.ymax - config.FISH_LENGTH)
        elif e.type == pygame.MOUSEMOTION and draw_mode and e.buttons[0] and not dragged_fish:
            x, y = screen_to_world(*e.pos)
            if prev_x is not None:
                # Draw line from prev to current position
                dx = x - prev_x
                dy = y - prev_y
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    steps = max(1, int(dist / 0.005)) 
                    for i in range(steps + 1):
                        ix = prev_x + i * dx / steps
                        iy = prev_y + i * dy / steps
                        gx, gy = world_to_grid(ix, iy)
                        y_indices, x_indices = np.ogrid[:GRID_HEIGHT, :GRID_WIDTH]
                        mask = (x_indices - gx)**2 + (y_indices - gy)**2 <= brush_radius**2
                        tank.wall_grid[mask] = True
            else:
                #First point
                gx, gy = world_to_grid(x, y)
                y_indices, x_indices = np.ogrid[:GRID_HEIGHT, :GRID_WIDTH]
                mask = (x_indices - gx)**2 + (y_indices - gy)**2 <= brush_radius**2
                tank.wall_grid[mask] = True
            prev_x, prev_y = x, y
        
    if draw_mode and not pygame.mouse.get_pressed()[0]:
        prev_x, prev_y = None, None 

    # update every FISH_TIME_STEP seconds
    if elapsed >= config.FISH_TIME_STEP:
        elapsed = 0
        selected_fish_orientation_pdf = fish_loop(fishies, spots, tank, config.FISH_TIME_STEP, paused)

    # draw everything
    screen.fill((15, 15, 30))  #background
    draw_tank(screen, tank)
    for gy in range(GRID_HEIGHT):
        for gx in range(GRID_WIDTH):
            if tank.wall_grid[gy, gx]:
                # Draw one grid cell
                x1, y1 = grid_to_world(gx, gy)
                x2, y2 = grid_to_world(gx + 1, gy + 1)

                sx1, sy1 = world_to_screen(x1, y1)
                sx2, sy2 = world_to_screen(x2, y2)

                left   = min(sx1, sx2)
                right  = max(sx1, sx2)
                top    = min(sy1, sy2)
                bottom = max(sy1, sy2)

                pygame.draw.rect(screen, (255, 255, 255), (left, top, right - left + 1, bottom - top + 1))
    
    if selected_fish_orientation_pdf is not None:
        # pdf might be out of sync with actual fish by 1 fish frame
        x = (0.1 + selected_fish_orientation_pdf) * 0.3 * np.cos(THETA_GRID + selected_fish.orientation)
        y = (0.1 + selected_fish_orientation_pdf) * 0.3 * np.sin(THETA_GRID + selected_fish.orientation)

        screen_x, screen_y = world_to_screen(x, y)
        points = np.column_stack((screen_x, screen_y)).astype(int)
        if len(points) > 1:
            pygame.draw.lines(screen, (0, 200, 255), False, points, 2)

    if draw_mode:
        message = "Drag and press mouse to draw"
    else:
        message = "Press D to draw"

    text_surface = font.render(message, True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))
    # interpolate fish position and orientation
    t = np.clip(elapsed / config.FISH_TIME_STEP, 0, 1)
    for fish in fishies:
        draw_fish(screen, fish, t)

    for spot in spots:
        draw_spot(screen, spot)

    pygame.display.flip()

pygame.quit()
