import pygame
import math

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

CENTER = (WIDTH // 2, HEIGHT // 2)
SCALE = 150  # controls size

def polar_function(theta):
    a = 1
    k = 4
    return 0.2 + abs(a * math.cos(k * theta))

running = True
points = []

theta = 0
while theta <= 2 * math.pi:
    r = polar_function(theta)
    print(r)
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    screen_x = CENTER[0] + x * SCALE
    screen_y = CENTER[1] - y * SCALE

    points.append((screen_x, screen_y))
    theta += 0.01

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((20, 20, 20))

    # draw axes
    pygame.draw.line(screen, (80, 80, 80), (CENTER[0], 0), (CENTER[0], HEIGHT))
    pygame.draw.line(screen, (80, 80, 80), (0, CENTER[1]), (WIDTH, CENTER[1]))

    # draw polar curve
    pygame.draw.lines(screen, (0, 200, 255), False, points, 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
