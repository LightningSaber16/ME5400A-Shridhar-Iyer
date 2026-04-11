# ─────────────────────────────────────────────
#  intruder.py  —  intruder / target agent
# ─────────────────────────────────────────────

import math
import random
import config as C
from geometry import distance


class Intruder:
    """
    Simple intruder agent.

    Modes
    -----
    'bounce'      : constant velocity, reflects off walls and obstacles
    'random_walk' : periodic random heading changes, reflects off walls
    'waypoints'   : cycles through a list of (x, y) waypoints

    Default: 'bounce'

    To switch mode at runtime:
        intruder.set_mode('random_walk')
    """

    def __init__(self, x=None, y=None, mode='bounce'):
        self.x      = float(x if x is not None else C.INTRUDER_START[0])
        self.y      = float(y if y is not None else C.INTRUDER_START[1])
        self.radius = C.INTRUDER_RADIUS
        self.speed  = C.INTRUDER_SPEED

        # Initial heading (radians)
        self.angle = random.uniform(-math.pi, math.pi)

        self.mode = mode

        # ── random walk state
        self._rw_timer    = 0
        self._rw_interval = 60   # steps between heading changes

        # ── waypoint state (populated by set_waypoints)
        self._waypoints   = []
        self._wp_index    = 0
        self._wp_thresh   = 20   # px — distance to count a waypoint as reached

    # ── Public API ───────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        assert mode in ('bounce', 'random_walk', 'waypoints'), \
            f"Unknown mode '{mode}'. Choose: bounce | random_walk | waypoints"
        self.mode = mode

    def set_waypoints(self, points: list):
        """Set list of (x, y) tuples for waypoint mode."""
        self._waypoints = list(points)
        self._wp_index  = 0

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, obstacles=None):
        if obstacles is None:
            obstacles = []

        if self.mode == 'bounce':
            self._update_bounce(obstacles)
        elif self.mode == 'random_walk':
            self._update_random_walk(obstacles)
        elif self.mode == 'waypoints':
            self._update_waypoints(obstacles)

        self._clamp_to_arena()

    # ── Mode implementations ─────────────────────────────────────────────────

    def _update_bounce(self, obstacles):
        dx = math.cos(self.angle) * self.speed
        dy = math.sin(self.angle) * self.speed
        nx, ny = self.x + dx, self.y + dy

        # Wall reflection
        reflect_x = reflect_y = False
        r = self.radius
        if nx < r or nx > C.ARENA_W - r:
            reflect_x = True
        if ny < r or ny > C.ARENA_H - r:
            reflect_y = True

        # Obstacle reflection (simple: reflect on contact)
        for (cx, cy, obs_r) in obstacles:
            if distance(nx, ny, cx, cy) < r + obs_r:
                # Reflect based on collision normal
                norm_angle = math.atan2(self.y - cy, self.x - cx)
                self.angle = 2 * norm_angle - self.angle + math.pi
                return

        if reflect_x:
            self.angle = math.pi - self.angle
        if reflect_y:
            self.angle = -self.angle

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    def _update_random_walk(self, obstacles):
        self._rw_timer += 1
        if self._rw_timer >= self._rw_interval:
            self._rw_timer = 0
            self._rw_interval = random.randint(30, 120)
            self.angle += random.uniform(-math.pi / 2, math.pi / 2)

        self._update_bounce(obstacles)   # walls/obstacle reflection still applies

    def _update_waypoints(self, obstacles):
        if not self._waypoints:
            return
        tx, ty = self._waypoints[self._wp_index]
        dist = distance(self.x, self.y, tx, ty)
        if dist < self._wp_thresh:
            self._wp_index = (self._wp_index + 1) % len(self._waypoints)
            tx, ty = self._waypoints[self._wp_index]

        self.angle = math.atan2(ty - self.y, tx - self.x)
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    # ── Utility ───────────────────────────────────────────────────────────────

    def _clamp_to_arena(self):
        r = self.radius
        self.x = max(r, min(C.ARENA_W - r, self.x))
        self.y = max(r, min(C.ARENA_H - r, self.y))
