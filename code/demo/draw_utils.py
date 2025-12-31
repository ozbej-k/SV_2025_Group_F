import pygame
import config
import numpy as np
from world import Tank, Spot, Fish

def lerp_angle(a, b, t):
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi
    return a + diff * t

def world_to_screen(x, y):
    x_px = config.WIDTH  / 2 + x * config.SCALE
    y_px = config.HEIGHT / 2 - y * config.SCALE
    return x_px, y_px

def screen_to_world(x_px, y_px):
    x = (x_px - config.WIDTH  / 2) / config.SCALE
    y = (config.HEIGHT / 2 - y_px) / config.SCALE
    return x, y

def draw_tank(screen, tank : Tank):
    top_left = pygame.Vector2(tank.xmin, tank.ymax)
    size = pygame.Vector2(tank.width, tank.height)

    x_px, y_px = world_to_screen(top_left.x, top_left.y)
    rect = pygame.Rect(x_px, y_px, size.x * config.SCALE, size.y * config.SCALE)
    pygame.draw.rect(screen, (255, 255, 255), rect, 2)

def draw_fish(screen, fish : Fish, interpolation_t):
    if fish.dragged:
        color = (255, 150, 100)
        position = fish.position
        orientation = fish.orientation
    else:
        if fish.selected:
            color = (255, 150, 255)
        else:
            color = (100, 200, 255)
        position = (1 - interpolation_t) * fish.prev_position + interpolation_t * fish.position
        orientation = lerp_angle(fish.prev_orientation, fish.orientation, interpolation_t)


    length = config.FISH_LENGTH
    width = config.FISH_WIDTH

    # # fish rectangle
    # points = [
    #     pygame.Vector2(-length/2, -width/2),
    #     pygame.Vector2( length/2, -width/2),
    #     pygame.Vector2( length/2,  width/2),
    #     pygame.Vector2(-length/2,  width/2)
    # ]

    # fish fish
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

def draw_spot(screen, spot : Spot):
    pygame.draw.circle(screen, (255, 255, 255), world_to_screen(spot.x, spot.y), spot.radius * config.SCALE, max(1, int(0.007 * config.SCALE)))
