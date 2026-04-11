# ─────────────────────────────────────────────
#  robot.py  —  Braitenberg vehicle
# ─────────────────────────────────────────────

import math
from geometry import (
    angle_wrap, angle_to, distance,
    ray_circle_intersection, ray_wall_intersection,
)
import config as C


class BraitenbergRobot:
    """
    Differential-drive robot controlled by two superimposed
    Braitenberg channels:

      Pursuit   (vehicle 2b / excitatory-crossed)
        Target sensor LEFT  → RIGHT motor  (steer toward)
        Target sensor RIGHT → LEFT  motor

      Avoidance (vehicle 3a / inhibitory-crossed)
        Proximity sensor LEFT  → RIGHT motor (steer away)
        Proximity sensor RIGHT → LEFT  motor
        (inhibitory: high reading → reduce speed on opposite side)

    Motor outputs are clamped to [0, MAX_SPEED].
    """

    def __init__(self, x=None, y=None, angle=None):
        self.x     = float(x     if x     is not None else C.ROBOT_START[0])
        self.y     = float(y     if y     is not None else C.ROBOT_START[1])
        self.angle = float(angle if angle is not None else C.ROBOT_START_ANGLE)

        self.radius = C.ROBOT_RADIUS
        self.max_speed = C.ROBOT_MAX_SPEED

        # Build proximity sensor angles (evenly around body, in robot frame)
        self.n_prox = C.N_PROX_SENSORS
        self.prox_angles = [
            2 * math.pi * i / self.n_prox for i in range(self.n_prox)
        ]

        # Last sensor readings (for visualisation)
        self.prox_readings   = [0.0] * self.n_prox   # 0=clear, 1=max contact
        self.target_left     = 0.0
        self.target_right    = 0.0

        # Last motor commands (for visualisation / logging)
        self.v_left  = 0.0
        self.v_right = 0.0

        self.collided = False   # set True if a wall/obstacle overlap is detected

    # ── Sensing ──────────────────────────────────────────────────────────────

    def _cast_proximity_sensors(self, obstacles):
        """
        For each proximity sensor, cast a ray and compute a normalised
        reading in [0, 1]:  0 = nothing at range,  1 = contact.
        """
        readings = []
        for rel_angle in self.prox_angles:
            world_angle = self.angle + rel_angle
            dx = math.cos(world_angle)
            dy = math.sin(world_angle)

            # Nearest hit distance
            nearest = ray_wall_intersection(
                self.x, self.y, dx, dy,
                C.ARENA_W, C.ARENA_H, C.PROX_RANGE
            )
            for (cx, cy, r) in obstacles:
                d = ray_circle_intersection(
                    self.x, self.y, dx, dy,
                    cx, cy, r, nearest
                )
                nearest = min(nearest, d)

            # Normalise: 1 when touching (dist≈0), 0 at max range
            readings.append(1.0 - nearest / C.PROX_RANGE)
        self.prox_readings = readings
        return readings

    def _sense_target(self, tx, ty):
        """
        Two wide-angle target sensors (left hemisphere / right hemisphere).
        Return normalised intensity in [0, 1] based on bearing and distance.
        """
        dist = distance(self.x, self.y, tx, ty)
        if dist > C.TARGET_SENSOR_RANGE:
            self.target_left = self.target_right = 0.0
            return 0.0, 0.0

        bearing = angle_wrap(angle_to(self.x, self.y, tx, ty) - self.angle)

        intensity = 1.0 - dist / C.TARGET_SENSOR_RANGE

        # Left sensor covers bearings > 0 (target is to the left of heading)
        # Right sensor covers bearings < 0 (target is to the right of heading)
        # Smooth cosine weighting within FOV half-angle
        fov = C.TARGET_SENSOR_FOV
        abs_b = abs(bearing)
        if abs_b <= fov:
            w = intensity * math.cos(abs_b / fov * (math.pi / 2))
            if bearing >= 0:
                left, right = w, 0.0
            else:
                left, right = 0.0, w
        else:
            left = right = 0.0

        self.target_left  = left
        self.target_right = right
        return left, right

    # ── Control law ──────────────────────────────────────────────────────────

    def _braitenberg_control(self, prox_readings, target_left, target_right):
        """
        Superimpose pursuit and avoidance channels.

        Pursuit (vehicle 2b — excitatory, crossed):
          left_motor  += GAIN * target_right
          right_motor += GAIN * target_left

        Avoidance (vehicle 3a — inhibitory):
          Compute aggregate left/right proximity loads.
          Reduce the opposite motor.
        """
        # ── Pursuit channel ──────────────────────────────
        v_l = C.TARGET_GAIN * C.PURSUIT_WEIGHT * target_right
        v_r = C.TARGET_GAIN * C.PURSUIT_WEIGHT * target_left

        # ── Avoidance channel ────────────────────────────
        # Split proximity sensors into left-side and right-side groups
        n = self.n_prox
        left_half  = prox_readings[:n // 2]
        right_half = prox_readings[n // 2:]

        prox_left  = max(left_half)   if left_half  else 0.0
        prox_right = max(right_half)  if right_half else 0.0

        # Inhibitory crossed: strong left obstacle → reduce right motor
        v_l -= C.PROX_GAIN * C.AVOIDANCE_WEIGHT * prox_right
        v_r -= C.PROX_GAIN * C.AVOIDANCE_WEIGHT * prox_left

        return v_l, v_r

    # ── Kinematics ───────────────────────────────────────────────────────────

    def _differential_drive(self, v_left, v_right):
        """
        Differential-drive update.
        Wheelbase assumed equal to 2 * robot radius.
        """
        v_left  = max(0.0, min(self.max_speed, v_left))
        v_right = max(0.0, min(self.max_speed, v_right))
        self.v_left  = v_left
        self.v_right = v_right

        v_centre = (v_left + v_right) / 2.0
        wheelbase = 2 * self.radius
        omega = (v_right - v_left) / wheelbase

        self.angle = angle_wrap(self.angle + omega)
        self.x += v_centre * math.cos(self.angle)
        self.y += v_centre * math.sin(self.angle)

    def _clamp_to_arena(self):
        r = self.radius
        self.x = max(r, min(C.ARENA_W - r, self.x))
        self.y = max(r, min(C.ARENA_H - r, self.y))

    def _check_collision(self, obstacles):
        self.collided = False
        for (cx, cy, r) in obstacles:
            if distance(self.x, self.y, cx, cy) < self.radius + r:
                self.collided = True
                # Simple push-out
                d = distance(self.x, self.y, cx, cy)
                if d > 0:
                    overlap = (self.radius + r) - d
                    nx = (self.x - cx) / d
                    ny = (self.y - cy) / d
                    self.x += nx * overlap
                    self.y += ny * overlap

    # ── Main update ──────────────────────────────────────────────────────────

    def update(self, target_x, target_y, obstacles):
        """
        Called once per simulation step.
        1. Sense
        2. Control
        3. Move
        4. Resolve collisions
        """
        prox = self._cast_proximity_sensors(obstacles)
        tl, tr = self._sense_target(target_x, target_y)
        vl, vr = self._braitenberg_control(prox, tl, tr)
        self._differential_drive(vl, vr)
        self._check_collision(obstacles)
        self._clamp_to_arena()
