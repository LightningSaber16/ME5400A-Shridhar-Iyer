# rl_env.py  —  Stage 2 & Stage 3 (v2): unified obs, reward, and history
#
# Both Stage 2 and Stage 3 now share:
#   Observation : 64-dim stacked obs
#                 [frame_t-4 .. frame_t] (5 × 12 = 60) + [v_norm, omega_norm,
#                  prox_mean, stuck_flag] (4) = 64 total
#   Reward      : progress + near-collision + stalling penalties (v2)
#
# They differ only in:
#   Stage 2 action: (Δα, Δβ) weight tuning on original Braitenberg base
#   Stage 3 action: (Δv_L, Δv_R, floor_scale) residual on improved base
#
# This makes the comparison clean: same obs, same network layout,
# different base controller and action space.

import math, random
from collections import deque
import numpy as np

import config as C
from hybrid_robot          import HybridRobot, build_observation
from hybrid_robot_improved import HybridRobotImproved, build_frame
from intruder              import Intruder
from geometry              import distance
from rl_policy             import FRAME_DIM, FLOOR_MAX

# ── Reward hyperparameters ────────────────────────────────────────────────────
LAMBDA_PROG  = 0.05    # progress reward: reward for closing distance each step
LAMBDA_COLL  = 0.10    # collision penalty (reduced — near-collision takes early warning)
LAMBDA_NEAR  = 0.02    # near-collision penalty: penalise max proximity reading
LAMBDA_STUCK = 0.05    # stalling penalty: fired when stuck recovery is active
LAMBDA_TIME  = 0.001   # per-step time penalty (reduced)
LAMBDA_CAPT  = 20.0    # capture bonus (increased — this is the objective)

MAX_STEPS          = 4000
DENSITY_RANGE      = (0.0, 0.35)
OBS_R_MIN          = 15
OBS_R_MAX          = 30
MIN_CLEAR_ROBOT    = 60
MIN_CLEAR_INTRUDER = 50
HISTORY_LEN        = 5   # must match rl_policy.HISTORY_LEN


def generate_obstacles(density, rng):
    if density < 0.005:
        return []
    target_area = density * C.ARENA_W * C.ARENA_H
    obs, placed = [], 0.0
    for _ in range(5000):
        if placed >= target_area:
            break
        r  = rng.randint(OBS_R_MIN, OBS_R_MAX)
        cx = rng.randint(r + 5, C.ARENA_W - r - 5)
        cy = rng.randint(r + 5, C.ARENA_H - r - 5)
        if distance(cx, cy, *C.ROBOT_START)    < r + MIN_CLEAR_ROBOT:    continue
        if distance(cx, cy, *C.INTRUDER_START) < r + MIN_CLEAR_INTRUDER: continue
        if any(distance(cx, cy, ex, ey) < r + er + 8 for ex,ey,er in obs): continue
        obs.append((cx, cy, r))
        placed += math.pi * r * r
    return obs


class SurveillanceEnv:
    """
    Training/evaluation environment for Stage 3 (v2).

    Key differences from v1:
      - _obs_history  : rolling deque of the last HISTORY_LEN raw 12-dim frames
      - _get_stacked_obs() : assembles the 74-dim observation from history + kinematics
      - _apply_action() : applies residual motor corrections, not weight patches
      - reward         : progress + near-collision + stalling penalties

    Stage 2 (use_improved=False) still uses the original 12-dim obs and weight
    patching so that existing Stage 2 results remain valid.
    """

    def __init__(self, policy, training=True, seed=42, use_improved=False):
        self.policy       = policy
        self.training     = training
        self.use_improved = use_improved
        self._rng         = random.Random(seed)
        self.robot        = None
        self.intruder     = None
        self.obstacles    = []
        self.step_count   = 0
        self.done         = False
        self.collision_count = 0
        self.captured     = False
        self._prev_dist   = 0.0

        # Frame history — only used for improved (Stage 3) training
        self._obs_history = deque(
            [np.zeros(FRAME_DIM, dtype=np.float32)] * HISTORY_LEN,
            maxlen=HISTORY_LEN
        )

        if use_improved:
            self._robot_cls = HybridRobotImproved
        else:
            self._robot_cls = HybridRobot

        # Both stages use frame stacking — same history buffer
        # Stage 2: 12-dim frames × 5 + 4 kinematic = 64-dim obs
        # Stage 3: same layout, same dims, different robot and action

    def reset(self, density=None, obstacles=None):
        if obstacles is not None:
            self.obstacles = list(obstacles)
        else:
            d = density if density is not None else self._rng.uniform(*DENSITY_RANGE)
            self.obstacles = generate_obstacles(d, self._rng)

        random.seed(self._rng.randint(0, 2**31))
        self.robot    = self._robot_cls(self.policy, training=self.training)
        self.intruder = Intruder(mode="bounce")
        self.step_count      = 0
        self.done            = False
        self.collision_count = 0
        self.captured        = False

        # Initialise history with the first real frame repeated
        self.robot._cast_proximity_sensors(self.obstacles)
        self.robot._sense_target(self.intruder.x, self.intruder.y)
        first_frame = self._get_frame()
        for _ in range(HISTORY_LEN):
            self._obs_history.append(first_frame.copy())

        self._prev_dist = distance(self.robot.x, self.robot.y,
                                   self.intruder.x, self.intruder.y)
        return self._get_obs()

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action=None):
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")

        if self.training and action is not None:
            if self.use_improved:
                self._apply_action_residual(action)
            else:
                self._apply_action_weights(action)
        else:
            self.robot.update(self.intruder.x, self.intruder.y, self.obstacles)

        self.intruder.update(self.obstacles)
        self.step_count += 1
        if self.robot.collided:
            self.collision_count += 1

        dist     = distance(self.robot.x, self.robot.y,
                            self.intruder.x, self.intruder.y)
        captured = dist < C.CAPTURE_RADIUS + self.intruder.radius
        timeout  = self.step_count >= MAX_STEPS

        # ── Reward ────────────────────────────────────────────────────────────
        # Both Stage 2 and Stage 3 use the same improved reward function.
        # The only difference is Stage 2 has no _recovery_steps so
        # stuck_flag will always be 0 — that's fine, it just won't fire.
        reward = self._compute_reward_v2(dist, captured)

        self._prev_dist = dist
        if captured:
            self.captured = True

        self.done = captured or timeout

        # Update env obs history for next call to _get_obs
        self.robot._cast_proximity_sensors(self.obstacles)
        self.robot._sense_target(self.intruder.x, self.intruder.y)
        self._obs_history.append(self._get_frame())

        info = {
            "captured"  : captured,
            "timeout"   : timeout,
            "step"      : self.step_count,
            "dist"      : dist,
            "collisions": self.collision_count,
        }
        return self._get_obs(), reward, self.done, info

    # ── Reward v2 ─────────────────────────────────────────────────────────────

    def _compute_reward_v2(self, dist, captured):
        # Progress: positive when robot closed distance to intruder this step
        progress   = (self._prev_dist - dist) / C.TARGET_SENSOR_RANGE
        # Near-collision: proportional to worst proximity reading
        prox_max   = max(self.robot.prox_readings) if self.robot.prox_readings else 0.0
        # Stalling: active when stuck recovery is running
        stuck_flag = float(getattr(self.robot, '_recovery_steps', 0) > 0)

        reward = (  LAMBDA_PROG  * progress
                  - LAMBDA_COLL  * float(self.robot.collided)
                  - LAMBDA_NEAR  * prox_max
                  - LAMBDA_STUCK * stuck_flag
                  - LAMBDA_TIME)
        if captured:
            reward += LAMBDA_CAPT
        return reward

    # ── Observation builders ──────────────────────────────────────────────────

    def _get_frame(self):
        """
        Single 12-dim frame — same layout for both Stage 2 and Stage 3.
        build_frame reads prox_readings, target_left/right, distance, bearing
        which are all present on both HybridRobot and HybridRobotImproved.
        """
        return build_frame(self.robot, self.intruder.x, self.intruder.y)

    def _get_obs(self):
        """
        Return the 64-dim stacked observation for both Stage 2 and Stage 3.
        Layout: [frame_t-4, frame_t-3, frame_t-2, frame_t-1, frame_t] (60 dims)
                + [v_norm, omega_norm, prox_mean, stuck_flag]           (4 dims)

        Stage 2 and Stage 3 use identical obs layout — the only differences
        are the robot class (original vs improved) and the action applied.
        This means the policy network architecture is identical for both stages,
        making the comparison clean: same obs, same network, different base controller.
        """
        # Ensure sensors are up to date
        if not self.use_improved:
            self.robot._cast_proximity_sensors(self.obstacles)
            self.robot._sense_target(self.intruder.x, self.intruder.y)

        # Stack history
        stacked = np.concatenate(list(self._obs_history))  # (60,)

        v_norm     = float((self.robot.v_left + self.robot.v_right) / 2.0
                           ) / self.robot.max_speed
        omega_norm = float((self.robot.v_right - self.robot.v_left)
                           / (2.0 * self.robot.max_speed))
        prox_mean  = float(sum(self.robot.prox_readings)
                           / len(self.robot.prox_readings))
        stuck_flag = float(getattr(self.robot, '_recovery_steps', 0) > 0)

        kine = np.array([v_norm, omega_norm, prox_mean, stuck_flag],
                        dtype=np.float32)
        return np.concatenate([stacked, kine])             # (64,)

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action_residual(self, action):
        """
        Stage 3 (v2): apply (Δv_L, Δv_R, floor_scale) as residual on top of
        the Braitenberg base command. No weight-patching.
        """
        delta_vl    = float(action[0])
        delta_vr    = float(action[1])
        floor_scale = float(action[2])

        prox   = self.robot._cast_proximity_sensors(self.obstacles)
        tl, tr = self.robot._sense_target(self.intruder.x, self.intruder.y)

        # Braitenberg base (fixed nominal weights)
        vl_base, vr_base = self.robot._braitenberg_control(prox, tl, tr)

        # Apply residuals
        vl = vl_base + delta_vl
        vr = vr_base + delta_vr

        # Dynamic motor floor
        effective_floor = max(0.0, C.MOTOR_FLOOR + floor_scale * FLOOR_MAX)
        v_centre = (vl + vr) / 2.0
        if v_centre < effective_floor:
            deficit = effective_floor - v_centre
            vl += deficit
            vr += deficit

        # Stuck recovery + physics
        if hasattr(self.robot, "_update_stuck"):
            vl, vr = self.robot._update_stuck(vl, vr)
        self.robot._differential_drive(vl, vr)
        self.robot._check_collision(self.obstacles)
        self.robot._clamp_to_arena()

    def _apply_action_weights(self, action):
        """
        Stage 2 (v1 unchanged): apply (Δα, Δβ) weight adjustments.
        Kept so Stage 2 training is unaffected by this file.
        """
        from hybrid_robot import ALPHA_MIN, ALPHA_MAX, BETA_MIN, BETA_MAX
        da, db = float(action[0]), float(action[1])
        self.robot.alpha_eff = float(np.clip(C.PURSUIT_WEIGHT   + da, ALPHA_MIN, ALPHA_MAX))
        self.robot.beta_eff  = float(np.clip(C.AVOIDANCE_WEIGHT + db, BETA_MIN,  BETA_MAX))

        orig_alpha, orig_beta = C.PURSUIT_WEIGHT, C.AVOIDANCE_WEIGHT
        C.PURSUIT_WEIGHT   = self.robot.alpha_eff
        C.AVOIDANCE_WEIGHT = self.robot.beta_eff

        try:
            prox   = self.robot._cast_proximity_sensors(self.obstacles)
            tl, tr = self.robot._sense_target(self.intruder.x, self.intruder.y)
            vl, vr = self.robot._braitenberg_control(prox, tl, tr)
            if hasattr(self.robot, "_update_stuck"):
                vl, vr = self.robot._update_stuck(vl, vr)
            self.robot._differential_drive(vl, vr)
            self.robot._check_collision(self.obstacles)
            self.robot._clamp_to_arena()
        finally:
            C.PURSUIT_WEIGHT   = orig_alpha
            C.AVOIDANCE_WEIGHT = orig_beta
