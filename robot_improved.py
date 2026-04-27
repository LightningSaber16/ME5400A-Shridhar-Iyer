# robot_improved.py
#
# BraitenbergRobotImproved — principled kinematic extensions to the pure
# reactive baseline.
#
# Two research-grounded improvements over the original controller:
#
# 1. INVERSE-SQUARE DANGER WEIGHTING (replaces cosine weighting)
#    Motivation: potential field theory (Khatib, 1986) defines repulsive
#    forces as inversely proportional to the square of distance to an
#    obstacle. Applied at the sensor level, sensors reporting nearby
#    obstacles contribute quadratically more to the avoidance channel than
#    sensors reporting distant ones. This produces a controller whose
#    avoidance response is proportional to actual collision danger rather
#    than sensor geometry, and reduces collisions caused by insufficient
#    reaction to close obstacles.
#
#    Weight for sensor i:
#      w_i = 1 / (d_i^2 + epsilon)
#    where d_i is the raw distance to the nearest obstacle on that ray
#    (not the normalised reading), and epsilon prevents division by zero.
#
# 2. GAP NAVIGATION RECOVERY (replaces blind alternating-turn recovery)
#    Motivation: gap navigation (Minguez & Montano, 2004) identifies
#    navigable gaps in the sensor ring and steers toward the widest free
#    passage. Rather than a timed reverse-turn in a fixed direction (which
#    may steer the robot further into obstacles), the robot identifies the
#    sector with the most free space and steers toward it during recovery.
#
#    Implementation: the 8 proximity sensors are treated as 8 candidate
#    escape directions. The sensor reporting the lowest reading (most free
#    space) defines the escape gap. The robot steers differentially toward
#    that direction for RECOVERY_DURATION steps.
#
# Retained from previous version:
#   - Forward bias (prevents stalling when target unseen)
#   - Centre-mean motor floor (avoidance cannot fully suppress movement)
#   - Stuck detection (unchanged threshold logic)
#
# Pipeline:
#   Stage 1: BraitenbergRobot         (robot.py)        — pure reactive baseline
#   Stage 2: HybridRobot              (hybrid_robot.py) — RL weights on Stage 1
#   Stage 4: BraitenbergRobotImproved (this file)       — principled kinematics

import math
from geometry import (
    angle_wrap, angle_to, distance,
    ray_circle_intersection, ray_wall_intersection,
)
import config as C

# Inverse-square weighting: epsilon in px^2.
# d=1px  -> weight ~1.0  (imminent collision)
# d=60px -> weight ~0.00028 (far obstacle, negligible)
_DANGER_EPS = 1.0


class BraitenbergRobotImproved:
    """
    Braitenberg pursuit controller with inverse-square danger weighting
    and gap navigation recovery.

    The avoidance channel uses sensor weights derived from potential field
    theory: w_i = 1/(d_i^2 + eps), making avoidance response proportional
    to collision danger rather than sensor geometry.

    Stuck recovery uses gap navigation: the robot steers toward the most
    open sector in its proximity sensor ring rather than using a fixed
    alternating turn direction.
    """

    def __init__(self, x=None, y=None, angle=None):
        self.x     = float(x     if x     is not None else C.ROBOT_START[0])
        self.y     = float(y     if y     is not None else C.ROBOT_START[1])
        self.angle = float(angle if angle is not None else C.ROBOT_START_ANGLE)

        self.radius    = C.ROBOT_RADIUS
        self.max_speed = C.ROBOT_MAX_SPEED

        self.n_prox      = C.N_PROX_SENSORS
        self.prox_angles = [
            2 * math.pi * i / self.n_prox for i in range(self.n_prox)
        ]

        # Exposed for visualisation / logging
        self.prox_readings  = [0.0] * self.n_prox
        self._prox_raw_dist = [float(C.PROX_RANGE)] * self.n_prox
        self.target_left    = 0.0
        self.target_right   = 0.0
        self.v_left         = 0.0
        self.v_right        = 0.0
        self.collided       = False

        # Stuck detection state
        self._stuck_timer    = 0
        self._stuck_ref_x    = self.x
        self._stuck_ref_y    = self.y
        self._recovery_steps = 0
        self._escape_turn    = 0.0   # set by gap navigation, range [-1, +1]

    # ── Sensing ───────────────────────────────────────────────────────────────

    def _cast_proximity_sensors(self, obstacles):
        """
        Cast all 8 proximity rays. Stores both normalised readings in
        self.prox_readings and raw distances in self._prox_raw_dist.
        Raw distances feed the inverse-square danger weights.
        """
        readings  = []
        raw_dists = []
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
            raw_dists.append(nearest)
            readings.append(1.0 - nearest / C.PROX_RANGE)

        self.prox_readings  = readings
        self._prox_raw_dist = raw_dists
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

    # ── Control law ───────────────────────────────────────────────────────────

    def _braitenberg_control(self, prox_readings, target_left, target_right):
        """
        Compute motor commands combining pursuit (Vehicle 2b) and avoidance
        (Vehicle 3a) channels.

        Avoidance uses inverse-square danger weighting grounded in potential
        field theory (Khatib, 1986):

          danger_i = 1 / (d_i^2 + epsilon)

        where d_i is the raw ray distance in pixels. Sensors detecting
        obstacles within a few pixels contribute quadratically more than
        sensors detecting distant obstacles, producing a response that
        scales with actual collision risk.
        """
        # ── Pursuit channel (Vehicle 2b) ──────────────────────────────────────
        v_l = C.TARGET_GAIN * C.PURSUIT_WEIGHT * target_right
        v_r = C.TARGET_GAIN * C.PURSUIT_WEIGHT * target_left

        # ── Avoidance channel — inverse-square danger weighting ───────────────
        prox_left_load  = 0.0
        prox_right_load = 0.0

        for i, rel_a in enumerate(self.prox_angles):
            reading  = prox_readings[i]
            raw_dist = self._prox_raw_dist[i]
            danger   = 1.0 / (raw_dist ** 2 + _DANGER_EPS)
            contribution = reading * danger
            norm_a = rel_a % (2 * math.pi)
            if 0 < norm_a <= math.pi:
                prox_left_load  += contribution
            else:
                prox_right_load += contribution

        # Normalise so avoidance signal stays in a comparable range to pursuit
        max_danger = (self.n_prox / 2.0) / (1.0 + _DANGER_EPS)
        safe_max   = max(max_danger, 1e-6)
        v_l -= C.PROX_GAIN * C.AVOIDANCE_WEIGHT * prox_right_load / safe_max
        v_r -= C.PROX_GAIN * C.AVOIDANCE_WEIGHT * prox_left_load  / safe_max

        # ── Forward bias ──────────────────────────────────────────────────────
        v_l += C.FORWARD_BIAS
        v_r += C.FORWARD_BIAS

        # ── Centre-mean motor floor ───────────────────────────────────────────
        v_centre = (v_l + v_r) / 2.0
        if v_centre < C.MOTOR_FLOOR:
            deficit = C.MOTOR_FLOOR - v_centre
            v_l += deficit
            v_r += deficit

        return v_l, v_r

    # ── Gap navigation ────────────────────────────────────────────────────────

    def _find_escape_direction(self):
        """
        Identify the widest free gap in the proximity sensor ring.

        Implements the gap navigation principle (Minguez & Montano, 2004):
        the sensor with the lowest reading points toward the most navigable
        free space. The signed angular position of that sensor relative to
        the robot heading determines the escape turn direction.

        Returns a turn bias in [-1, +1]:
          +1  -> turn hard left (gap is to the left)
          -1  -> turn hard right (gap is to the right)
           0  -> gap is ahead (go straight)
        """
        # Find sensor pointing toward most free space
        min_reading = min(self.prox_readings)
        gap_idx     = self.prox_readings.index(min_reading)
        gap_angle   = self.prox_angles[gap_idx]

        # Wrap to [-pi, pi]: positive = left of heading, negative = right
        signed_gap = angle_wrap(gap_angle)

        # Normalise to [-1, +1]
        return float(signed_gap / math.pi)

    def _update_stuck(self, v_left, v_right):
        """
        Stuck detection and gap-directed recovery.

        If the robot has displaced less than STUCK_DIST_THRESH pixels over
        STUCK_PATIENCE steps, gap navigation identifies the most open escape
        direction and the robot executes a targeted reverse-turn toward that
        gap for RECOVERY_DURATION steps.

        This replaces the previous blind alternating-turn with a principled
        direction derived from the current sensor configuration.
        """
        if self._recovery_steps > 0:
            self._recovery_steps -= 1
            spd      = C.ROBOT_MAX_SPEED * 0.5
            turn_mag = abs(self._escape_turn) * C.ROBOT_MAX_SPEED * 0.4
            if self._escape_turn >= 0:
                rv_l = -spd + turn_mag    # pivot left toward gap
                rv_r = -spd - turn_mag
            else:
                rv_l = -spd - turn_mag    # pivot right toward gap
                rv_r = -spd + turn_mag
            return rv_l, rv_r

        self._stuck_timer += 1
        if self._stuck_timer >= C.STUCK_PATIENCE:
            moved = distance(self.x, self.y, self._stuck_ref_x, self._stuck_ref_y)
            self._stuck_timer = 0
            self._stuck_ref_x = self.x
            self._stuck_ref_y = self.y
            if moved < C.STUCK_DIST_THRESH:
                # Gap navigation: compute escape direction from sensor ring
                self._escape_turn    = self._find_escape_direction()
                self._recovery_steps = C.RECOVERY_DURATION
                spd      = C.ROBOT_MAX_SPEED * 0.5
                turn_mag = abs(self._escape_turn) * C.ROBOT_MAX_SPEED * 0.4
                if self._escape_turn >= 0:
                    rv_l = -spd + turn_mag
                    rv_r = -spd - turn_mag
                else:
                    rv_l = -spd - turn_mag
                    rv_r = -spd + turn_mag
                return rv_l, rv_r

        return v_left, v_right

    # ── Kinematics ────────────────────────────────────────────────────────────

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

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, target_x, target_y, obstacles):
        prox   = self._cast_proximity_sensors(obstacles)
        tl, tr = self._sense_target(target_x, target_y)
        vl, vr = self._braitenberg_control(prox, tl, tr)
        vl, vr = self._update_stuck(vl, vr)
        self._differential_drive(vl, vr)
        self._check_collision(obstacles)
        self._clamp_to_arena()
