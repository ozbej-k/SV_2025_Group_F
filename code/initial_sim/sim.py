import pygame, random
pygame.init()

W, H = 800, 600
N = 30
MAX_SPEED, VISION, MARGIN, EDGE_FORCE = 3, 50, 50, 0.3
BORDER = pygame.Rect(MARGIN, MARGIN, W - 2 * MARGIN, H - 2 * MARGIN)
FISH_RADIUS = 5

screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()

class Boid:
  def __init__(self):
    self.pos = pygame.Vector2(random.uniform(MARGIN, W - MARGIN), random.uniform(MARGIN, H - MARGIN))
    self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
    self.drag = False

  def get_next_state(self, boids, dragged_fish):
    if self.drag: return None
    alignment = cohesion = seperation = pygame.Vector2()
    neighbors = [b for b in boids if b != self and self.pos.distance_to(b.pos) < VISION]

    if neighbors:
      alignment = sum((b.vel for b in neighbors), pygame.Vector2()) / len(neighbors) - self.vel
      cohesion = sum((b.pos for b in neighbors), pygame.Vector2()) / len(neighbors) - self.pos
      for b in neighbors:
        distance = self.pos.distance_to(b.pos)
        if distance < 20: 
          seperation += (self.pos - b.pos) / (distance or 1)
  
    edge_reflection = pygame.Vector2(
      EDGE_FORCE if self.pos.x < BORDER.left + 10 else -EDGE_FORCE if self.pos.x > BORDER.right - 10 else 0,
      EDGE_FORCE if self.pos.y < BORDER.top + 10 else -EDGE_FORCE if self.pos.y > BORDER.bottom - 10 else 0
    )

    dragged_fish_pull = pygame.Vector2()
    if dragged_fish and dragged_fish != self:
      d = self.pos.distance_to(dragged_fish.pos)
      if d < 200:
        dragged_fish_pull = (dragged_fish.pos - self.pos) * (0.02 * (1 - d / 200))

    class state_vars:
      def __init__(self, alignment, cohesion, seperation, edge_reflection, dragged_fish_pull):
        self.alignment = alignment
        self.cohesion = cohesion
        self.seperation = seperation
        self.edge_reflection = edge_reflection
        self.dragged_fish_pull = dragged_fish_pull

    return state_vars(alignment, cohesion, seperation, edge_reflection, dragged_fish_pull)
  
  def update_state(self, state):
    if state is None: return

    self.vel += 0.01*state.alignment + 0.005*state.cohesion + 0.05*state.seperation + state.edge_reflection + state.dragged_fish_pull
    if self.vel.length() > MAX_SPEED:
      self.vel.scale_to_length(MAX_SPEED)
    self.pos += self.vel
    
    self.pos.x = min(max(self.pos.x, BORDER.left + FISH_RADIUS), BORDER.right - FISH_RADIUS)
    self.pos.y = min(max(self.pos.y, BORDER.top + FISH_RADIUS), BORDER.bottom - FISH_RADIUS)

  # def draw(self, surf):
  #   color = (255,150,100) if self.drag else (100,200,255)
  #   pygame.draw.circle(surf, color, self.pos, FISH_RADIUS)
  
  def draw(self, surf):
    color = (255,150,100) if self.drag else (100,200,255)
    if self.vel.length_squared() == 0:  # avoid zero division
      forward = pygame.Vector2(1, 0)
    else:
      forward = self.vel.normalize()

    left = forward.rotate(150) * FISH_RADIUS
    right = forward.rotate(-150) * FISH_RADIUS
    tip = forward * FISH_RADIUS
    points = [self.pos + tip, self.pos + left, self.pos + right]
    pygame.draw.polygon(surf, color, points)

boids = [Boid() for _ in range(N)]
dragged_fish = None

running = True
while running:
  for e in pygame.event.get():
    if e.type == pygame.QUIT: 
      running = False
    elif e.type == pygame.MOUSEBUTTONDOWN:
      p = pygame.Vector2(e.pos)
      for b in boids:
        if p.distance_to(b.pos) < 10: 
          b.drag, dragged_fish = True, b 
          break
    elif e.type == pygame.MOUSEBUTTONUP:
      if dragged_fish: 
        dragged_fish.drag, dragged_fish = False, None
    elif e.type == pygame.MOUSEMOTION and dragged_fish:
      x, y = e.pos
      dragged_fish.pos.xy = (min(max(x, BORDER.left), BORDER.right), min(max(y, BORDER.top), BORDER.bottom))

  screen.fill((15,15,30))
  pygame.draw.rect(screen, (255,255,255), BORDER, 2)
  
  # calculate next state for each boid at once
  next_states = []
  for b in boids:
    next_states.append(b.get_next_state(boids, dragged_fish))

  # update state after calculation
  for b, s in zip(boids, next_states):
    b.update_state(s)

  # draw boids
  for b in boids:
    b.draw(screen)

  pygame.display.flip(); clock.tick(60)
pygame.quit()
