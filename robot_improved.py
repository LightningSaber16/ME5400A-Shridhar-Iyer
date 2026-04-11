# robot_improved.py
#
# Improved Braitenberg vehicle — extends the original with:
#   - Cosine-weighted proximity blending (smoother avoidance)
#   - Forward bias (robot never fully stops when target unseen)
#   - Motor floor (avoidance cannot reduce speed to zero)
#   - Stuck detection and recovery (escape from deadlocks)
#
# This is Stage 3 of the pipeline:
#   Stage 1: BraitenbergRobot    (robot.py)          — pure reactive baseline
#   Stage 2: HybridRobot         (hybrid_robot.py)   — RL weights on Stage 1
#   Stage 3: HybridRobotImproved (hybrid_robot_improved.py) — RL weights on this

import math
from geometry import (
    angle_wrap, angle_to, distance,
    ray_circle_intersection, ray_wall_intersection,
)
import config as C


class BraitenbergRobotImproved:
    """
    Improved differential-drive Braitenberg vehicle.

    Changes from the original BraitenbergRobot:
      1. Cosine-weighted proximity aggregation instead of max per hemisphere.
         Front-facing sensors have stronger influence; rear sensors still
         contribute but with lower weight. Produces smoother avoidance.
      2. Forward bias: FORWARD_BIAS px/step added to both motors every step.
         Keeps the robot patrolling when the target is outside sensor range.
      3. Motor floor: motor speed never drops below MOTOR_FLOOR.
         Avoidance channel cannot pin the robot completely.
      4. Stuck recovery: if displacement < STUCK_DIST_THRESH over
         STUCK_PATIENCE steps, triggers a timed reverse-turn to escape.
    """

    def __init__(self, x=None, y=None, angle=None):
        self.x     = float(x     if x     is not None else C.ROBOT_START[0])
        self.y     = float(y     if y     is not None else C.ROBOT_START[1])
        self.angle = float(angle if angle is not None else C.ROBOT_START_ANGLE)

        self.radius    = C.ROBOT_RADIUS
        self.max_speed = C.ROBOT_MAX_SPEED

        self.n_prox = C.N_PROX_SENSORS
        self.prox_angles = [
            2 * math.pi * i / self.n_prox for i in range(self.n_prox)
        ]

        # Exposed for visualisation / logging
        self.prox_readings = [0.0] * self.n_prox
        self.target_left   = 0.0
        self.target_right  = 0.0
        self.v_left        = 0.0
        self.v_right       = 0.0
        self.collided      = False

        # Stuck-detection state
        self._stuck_timer    = 0
        self._stuck_ref_x    = self.x
        self._stuck_ref_y    = self.y
        self._recovery_steps = 0
        self._recovery_turn  = 1   # alternates +1 / -1

    # ── Sensing ──────────────────────────────────────────────────────────────

    def _cast_proximity_sensors(self, obstacles):
        readings = []
        for rel_angle in self.prox_angles:
            world_angle = self.angle + rel_angle
            dx = math.cos(world_angle)
            dy = math.sin(world_angle)
            nearest = ray_wall_intersection(
                self.x, self.y, dx, dy,
                C.ARENA_W, C.ARENA_H, C.PROX_RANGE
            )
            for (cx, cy, r) in obstacles:
                d = ray_circle_intersection(
                    self.x, self.y, dx, dy, cx, cy, r, nearest
                )
                nearest = min(nearest, d)
            readings.append(1.0 - nearest / C.PROX_RANGE)
        self.prox_readings = readings
        return readings

    def _sense_target(self, tx, ty):
        dist = distance(self.x, self.y, tx, ty)
        if dist > C.TARGET_SENSOR_RANGE:
            self.target_left = self.target_right = 0.0
            return 0.0, 0.0
        bearing   = angle_wrap(angle_to(self.x, self.y, tx, ty) - self.angle)
        intensity = 1.0 - dist / C.TARGET_SENSOR_RANGE
        fov       = C.TARGET_SENSOR_FOV
        abs_b     = abs(bearing)
        if abs_b <= fov:
            w = intensity * math.cos(abs_b / fov * (math.pi / 2))
            left, right = (w, 0.0) if bearing >= 0 else (0.0, w)
        else:
            left = right = 0.0
        self.target_left  = left
        self.target_right = right
        return left, right

    # ── Control law (improved) ────────────────────────────────────────────────

    def _braitenberg_control(self, prox_readings, target_left, target_right):
        # Pursuit channel (unchanged from original)
        v_l = C.TARGET_GAIN * C.PURSUIT_WEIGHT * target_right
        v_r = C.TARGET_GAIN * C.PURSUIT_WEIGHT * target_left

        # Avoidance — cosine-weighted across all sensors
        # Each sensor at relative angle rel_a contributes proportionally to
        # its forward-facing component. Left-side sensors suppress right motor;
        # right-side sensors suppress left motor.
        prox_left_load  = 0.0
        prox_right_load = 0.0
        for i, rel_a in enumerate(self.prox_angles):
            reading    = prox_readings[i]
            fwd_weight = max(0.0, math.cos(rel_a)) + 0.3  # 0.3 keeps rear active
            weighted   = reading * fwd_weight
            norm_a     = rel_a % (2 * math.pi)
            if 0 < norm_a <= math.pi:
                prox_left_load  += weighted
            else:
                prox_right_load += weighted

        n   = max(1, self.n_prox)
        v_l -= C.PROX_GAIN * C.AVOIDANCE_WEIGHT * prox_right_load / n
        v_r -= C.PROX_GAIN * C.AVOIDANCE_WEIGHT * prox_left_load  / n

        # Forward bias — keep robot moving when target unseen
        v_l += C.FORWARD_BIAS
        v_r += C.FORWARD_BIAS
 
        # Motor floor — applied to mean speed so asymmetric avoidance turns
        # are still permitted, but net forward momentum is preserved
        v_centre = (v_l + v_r) / 2.0
        if v_centre < C.MOTOR_FLOOR:
            deficit = C.MOTOR_FLOOR - v_centre
            v_l += deficit
            v_r += deficit
 
        return v_l, v_r

       
        return v_l, v_r

    # ── Stuck recovery ────────────────────────────────────────────────────────

    def _update_stuck(self, v_left, v_right):
        if self._recovery_steps > 0:
            self._recovery_steps -= 1
            rv_l = -C.ROBOT_MAX_SPEED * 0.3 * (1 + self._recovery_turn * 0.4)
            rv_r = -C.ROBOT_MAX_SPEED * 0.3 * (1 - self._recovery_turn * 0.4)
            return rv_l, rv_r

        self._stuck_timer += 1
        if self._stuck_timer >= C.STUCK_PATIENCE:
            moved = distance(self.x, self.y, self._stuck_ref_x, self._stuck_ref_y)
            self._stuck_timer = 0
            self._stuck_ref_x = self.x
            self._stuck_ref_y = self.y
            if moved < C.STUCK_DIST_THRESH:
                self._recovery_steps  = C.RECOVERY_DURATION
                self._recovery_turn  *= -1
                rv_l = -C.ROBOT_MAX_SPEED * 0.5 * (1 + self._recovery_turn * 0.6)
                rv_r = -C.ROBOT_MAX_SPEED * 0.5 * (1 - self._recovery_turn * 0.6)
                return rv_l, rv_r
        return v_left, v_right

    # ── Kinematics ───────────────────────────────────────────────────────────

    def _differential_drive(self, v_left, v_right):
        v_left  = max(-self.max_speed, min(self.max_speed, v_left))
        v_right = max(-self.max_speed, min(self.max_speed, v_right))
        self.v_left  = v_left
        self.v_right = v_right
        v_centre  = (v_left + v_right) / 2.0
        omega     = (v_right - v_left) / (2 * self.radius)
        self.angle = angle_wrap(self.angle + omega)
        self.x    += v_centre * math.cos(self.angle)
        self.y    += v_centre * math.sin(self.angle)

    def _clamp_to_arena(self):
        r = self.radius
        self.x = max(r, min(C.ARENA_W - r, self.x))
        self.y = max(r, min(C.ARENA_H - r, self.y))

    def _check_collision(self, obstacles):
        self.collided = False
        for (cx, cy, r) in obstacles:
            if distance(self.x, self.y, cx, cy) < self.radius + r:
                self.collided = True
                d = distance(self.x, self.y, cx, cy)
                if d > 0:
                    overlap = (self.radius + r) - d
                    self.x += (self.x - cx) / d * overlap
                    self.y += (self.y - cy) / d * overlap

    # ── Main update ──────────────────────────────────────────────────────────

    def update(self, target_x, target_y, obstacles):
        prox   = self._cast_proximity_sensors(obstacles)
        tl, tr = self._sense_target(target_x, target_y)
        vl, vr = self._braitenberg_control(prox, tl, tr)
        vl, vr = self._update_stuck(vl, vr)
        self._differential_drive(vl, vr)
        self._check_collision(obstacles)
        self._clamp_to_arena()
