import pygame
import config
import numpy as np

WHITE = (255, 255, 255)
DARK_BLUE = (20, 120, 255)
BLUE = (70, 170, 255)
LIGHT_BLUE = (100, 200, 255)
PINK = (255, 150, 255)
ORANGE = (255, 150, 100)

class UI:
    sidebar_rect = None
    tank_rect = None
    buttons = None
    line_width_float = None
    line_width_int = None
    button_height = None

def recalculate_scale_and_positions(new_width, new_height, tank, add_fish_action, add_spot_action, remove_fish_action, remove_spot_action):
    config.WIDTH, config.HEIGHT = new_width, new_height
    config.SCALE = config.scale(config.WIDTH, config.HEIGHT)
    UI.line_width_float = config.LINE_WIDTH * config.SCALE
    UI.line_width_int = int(UI.line_width_float)
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.RESIZABLE)
    font = pygame.font.SysFont("Arial", int(0.04 * config.SCALE))

    tank_screen_x, tank_screen_y = world_to_screen(tank.xmin, tank.ymax)
    UI.tank_rect = pygame.Rect(tank_screen_x, tank_screen_y, tank.width * config.SCALE, tank.height * config.SCALE)

    sidebar_screen_x, sidebar_screen_y = world_to_screen(tank.xmin + tank.width - config.LINE_WIDTH, tank.ymax)
    UI.sidebar_rect = pygame.Rect(sidebar_screen_x, sidebar_screen_y, config.SIDEBAR_WIDTH * config.SCALE, tank.height * config.SCALE)
    UI.button_height = 0.1 * config.SCALE

    def clear_tank_action():
        tank.wall_grid = np.ones((int(tank.height/config.GRID_CELL_SIZE), int(tank.width/config.GRID_CELL_SIZE)), dtype=bool)
        tank.wall_grid[1:-1, 1:-1] = 0

    small_button_width = (UI.sidebar_rect.width * 0.40)
    sidebar_right = UI.sidebar_rect.x + UI.sidebar_rect.width
    UI.buttons = [
        Button(Button.MOVE_FISH,   Button.TOGGLE, "Move Fish",   UI.sidebar_rect.x + UI.line_width_float, UI.sidebar_rect.y + UI.sidebar_rect.height - 7 * UI.button_height,    UI.sidebar_rect.width - UI.line_width_float, UI.button_height, BLUE, LIGHT_BLUE, DARK_BLUE),
        Button(Button.REMOVE_FISH, Button.ACTION, "-1 Fish",     UI.sidebar_rect.x + UI.line_width_float, UI.sidebar_rect.y + UI.sidebar_rect.height - 6 * UI.button_height,    small_button_width - UI.line_width_float,    UI.button_height, BLUE, LIGHT_BLUE, DARK_BLUE, remove_fish_action),
        Button(Button.ADD_FISH,    Button.ACTION, "+1 Fish",     sidebar_right - small_button_width,      UI.sidebar_rect.y + UI.sidebar_rect.height - 6 * UI.button_height,    small_button_width,                          UI.button_height, BLUE, LIGHT_BLUE, DARK_BLUE, add_fish_action),

        Button(Button.MOVE_SPOT,   Button.TOGGLE, "Move spots",  UI.sidebar_rect.x + UI.line_width_float, UI.sidebar_rect.y + UI.sidebar_rect.height - 4.48 * UI.button_height, UI.sidebar_rect.width - UI.line_width_float, UI.button_height, BLUE, LIGHT_BLUE, DARK_BLUE),
        Button(Button.REMOVE_SPOT, Button.ACTION, "-1 Spot",     UI.sidebar_rect.x + UI.line_width_float, UI.sidebar_rect.y + UI.sidebar_rect.height - 3.5  * UI.button_height, small_button_width - UI.line_width_float,    UI.button_height, BLUE, LIGHT_BLUE, DARK_BLUE, remove_spot_action),
        Button(Button.ADD_SPOT,    Button.ACTION, "+1 Spot",     sidebar_right - small_button_width,      UI.sidebar_rect.y + UI.sidebar_rect.height - 3.5  * UI.button_height, small_button_width,                          UI.button_height, BLUE, LIGHT_BLUE, DARK_BLUE, add_spot_action),
        
        Button(Button.DRAW_BUTTON, Button.TOGGLE, "Draw Walls",  UI.sidebar_rect.x + UI.line_width_float, UI.sidebar_rect.y + UI.sidebar_rect.height - 2 * UI.button_height,    UI.sidebar_rect.width - UI.line_width_float, UI.button_height, BLUE, LIGHT_BLUE, DARK_BLUE),
        Button(Button.CLEAR_TANK,  Button.ACTION, "Clear Walls", UI.sidebar_rect.x + UI.line_width_float, UI.sidebar_rect.y + UI.sidebar_rect.height - 1 * UI.button_height,    UI.sidebar_rect.width - UI.line_width_float, UI.button_height, BLUE, LIGHT_BLUE, DARK_BLUE, clear_tank_action),
    ]
    return screen, font

class Button:
    # Button IDs
    MOVE_FISH = 0
    REMOVE_FISH = 1
    ADD_FISH = 2
    MOVE_SPOT = 3
    REMOVE_SPOT = 4
    ADD_SPOT = 5
    DRAW_BUTTON = 6
    CLEAR_TANK = 7
    NUM_BUTTONS = 8

    # Button types
    TOGGLE = 1
    ACTION = 2

    BUTTON_IS_SELECTED = [False for _ in range(NUM_BUTTONS)]

    def __init__(self, button_id, button_type, text, x, y, width, height, color, hover_color, selected_color, callback=None):
        self.button_id = button_id
        self.button_type = button_type
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.selected_color = selected_color
        self.function = callback  # Only used for ACTION buttons

    def draw(self, screen, font):
        mouse_pos = pygame.mouse.get_pos()
        current_color = self.color

        is_selected = (self.button_type == Button.TOGGLE and Button.BUTTON_IS_SELECTED[self.button_id])

        if is_selected:
            current_color = self.selected_color

        if self.rect.collidepoint(mouse_pos):
            current_color = (self.hover_color if not is_selected else self.color)

        pygame.draw.rect(screen, current_color, self.rect)

        if is_selected:
            pygame.draw.rect(
                screen,
                WHITE,
                self.rect,
                int(config.LINE_WIDTH * 2 * config.SCALE),
            )

        text_surface = font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                if self.button_type == Button.TOGGLE:
                    Button.select_button(self.button_id)

                elif self.button_type == Button.ACTION and self.function:
                    self.function()

                return True
        return False
    
    @staticmethod
    def select_button(button_id):
        # deselect all buttons first
        for i in range(len(Button.BUTTON_IS_SELECTED)):
            Button.BUTTON_IS_SELECTED[i] = False

        # select this one
        Button.BUTTON_IS_SELECTED[button_id] = True

def draw_frame(screen, font, THETA_GRID, elapsed, tank, fishies, spots, selected_fish, selected_fish_orientation_pdf):
    interpolation_t = np.clip(elapsed / config.FISH_TIME_STEP, 0, 1)
    
    screen.fill((15, 15, 30)) # background
    draw_tank(screen, tank)

    # text_surface = font.render("Press D to draw", True, WHITE)
    # screen.blit(text_surface, (10, 10))
    
    # draw buttons
    for btn in UI.buttons:
        btn.draw(screen, font)

    # draw sidebar stuff
    pygame.draw.rect(screen, WHITE, UI.sidebar_rect, UI.line_width_int) # sidebar border
    draw_orientation_pdf(screen, selected_fish_orientation_pdf, THETA_GRID, selected_fish, interpolation_t)

    # draw fish and spots
    for fish in fishies:
        draw_fish(screen, fish, interpolation_t)

    for spot in spots:
        draw_spot(screen, spot)






def lerp_angle(a, b, t):
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi
    return a + diff * t

def world_to_screen(x, y):
    x -= config.SIDEBAR_WIDTH / 2
    x_px = config.WIDTH  / 2 + x * config.SCALE
    y_px = config.HEIGHT / 2 - y * config.SCALE
    return x_px, y_px

def screen_to_world(x_px, y_px):
    x = (x_px - config.WIDTH  / 2) / config.SCALE
    y = (config.HEIGHT / 2 - y_px) / config.SCALE
    x += config.SIDEBAR_WIDTH / 2
    return x, y

def draw_tank(screen, tank):
    pygame.draw.rect(screen, WHITE, UI.tank_rect, UI.line_width_int)
    
    for gy in range(tank.grid_height):
        for gx in range(tank.grid_width):
            if tank.wall_grid[gy, gx]:
                # Draw one grid cell
                x1, y1 = grid_to_world(tank, gx, gy)
                x2, y2 = grid_to_world(tank, gx + 1, gy + 1)

                sx1, sy1 = world_to_screen(x1, y1)
                sx2, sy2 = world_to_screen(x2, y2)

                left   = min(sx1, sx2)
                right  = max(sx1, sx2)
                top    = min(sy1, sy2)
                bottom = max(sy1, sy2)

                pygame.draw.rect(screen, WHITE, (left, top, right - left + 1, bottom - top + 1))

def draw_orientation_pdf(screen, selected_fish_orientation_pdf, THETA_GRID, selected_fish, interpolation_t):
    pdf_radius = 0.175 * config.SCALE
    middle = (UI.sidebar_rect.x + UI.sidebar_rect.width / 2, UI.sidebar_rect.y + pdf_radius * 1.3)
    if selected_fish is not None and selected_fish_orientation_pdf is not None:
        cutoff = 0.8
        selected_fish_orientation_pdf[selected_fish_orientation_pdf > cutoff] = cutoff
        x = (0.22 + selected_fish_orientation_pdf) * np.cos(THETA_GRID + selected_fish.prev_orientation)
        y = (0.22 + selected_fish_orientation_pdf) * np.sin(THETA_GRID + selected_fish.prev_orientation)

        # convert to screen
        x = middle[0] + x * 0.15 * config.SCALE
        y = middle[1] + y * 0.15 * -config.SCALE
        points = np.column_stack((x, y)).astype(int)
        if len(points) > 1:
            pygame.draw.lines(screen, (0, 200, 255), False, points, 2)
            orientation = lerp_angle(selected_fish.prev_orientation, selected_fish.orientation, interpolation_t)
            _draw_fish(screen, screen_to_world(*middle), orientation, PINK)

    pygame.draw.circle(screen, WHITE, middle, pdf_radius, max(1, int(0.007 * config.SCALE)))

def world_to_grid(tank, x, y): # copied in _raycast_drawn_wall_tangents
    # Map world coords to grid indices
    gx = int((x - tank.xmin) / (tank.xmax - tank.xmin) * tank.grid_width)
    gy = int((y - tank.ymin) / (tank.ymax - tank.ymin) * tank.grid_height)
    return np.clip(gx, 0, tank.grid_width-1), np.clip(gy, 0, tank.grid_height-1)

def grid_to_world(tank, gx, gy):
    # Map grid to world (for rendering)
    x = tank.xmin + (gx / tank.grid_width) * (tank.xmax - tank.xmin)
    y = tank.ymin + (gy / tank.grid_height) * (tank.ymax - tank.ymin)
    return x, y 

def _draw_fish(screen, position, orientation, color):
    length = config.FISH_LENGTH
    width = config.FISH_WIDTH

    # # fish rectangle
    # points = [
    #     pygame.Vector2(-length/2, -width/2),
    #     pygame.Vector2( length/2, -width/2),
    #     pygame.Vector2( length/2,  width/2),
    #     pygame.Vector2(-length/2,  width/2)
    # ]

    # diamond fish
    points = [
        pygame.Vector2( length/2,  0),       # nose
        pygame.Vector2( length/4, -width/2), # upper mid-body
        pygame.Vector2(-length/4, -width/6), # tail top tip
        pygame.Vector2(-length/2, -width/3), # upper tail taper
        pygame.Vector2(-length/2,  width/3), # lower tail taper
        pygame.Vector2(-length/4,  width/6), # tail bottom tip
        pygame.Vector2( length/4,  width/2), # lower mid-body
    ]

    rotated = [corner.rotate_rad(orientation) + pygame.Vector2(*position) for corner in points]
    points = [world_to_screen(p.x, p.y) for p in rotated]

    pygame.draw.polygon(screen, color, points)

def draw_fish(screen, fish, interpolation_t):
    if fish.dragged:
        color = ORANGE
        position = fish.position
        orientation = fish.orientation
    else:
        if fish.selected:
            color = PINK
        else:
            color = LIGHT_BLUE
        position = (1 - interpolation_t) * fish.prev_position + interpolation_t * fish.position
        orientation = lerp_angle(fish.prev_orientation, fish.orientation, interpolation_t)

    _draw_fish(screen, position, orientation, color)

def draw_spot(screen, spot):
    color = ORANGE if spot.dragged else WHITE
    pygame.draw.circle(screen, color, world_to_screen(*spot.position[:2]), spot.radius * config.SCALE, max(1, int(0.007 * config.SCALE)))
