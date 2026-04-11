# hybrid_robot_improved.py  —  Stage 3 (v2): residual motor control
#
# Changes from v1:
#   - Policy outputs (Δv_L, Δv_R, floor_scale) instead of (Δα, Δβ)
#   - Braitenberg controller runs first with FIXED nominal weights
#   - RL adds a residual correction to the computed motor commands
#   - floor_scale adjusts C.MOTOR_FLOOR dynamically each step
#   - build_observation_improved now returns a single 12-dim FRAME
#     (stacking into 74-dim obs is done by SurveillanceEnv)
#
# Pipeline:
#   Stage 1: BraitenbergRobot         — pure reactive baseline
#   Stage 2: HybridRobot              — RL weight tuning on original base
#   Stage 3: HybridRobotImproved      — RL residual corrections on improved base
#   Stage 4: BraitenbergRobotImproved — improved base, fixed weights (ablation)

import math
import numpy as np

from robot_improved import BraitenbergRobotImproved
from rl_policy      import PolicyNetwork, DELTA_MAX, FLOOR_MAX, OBS_DIM, FRAME_DIM
from geometry       import distance, angle_wrap, angle_to
import config as C


def build_frame(robot, intruder_x, intruder_y):
    """
    Build a single 12-dim observation frame.
    Identical layout to the original build_observation_improved so that
    the history buffer in SurveillanceEnv can stack frames cleanly.

    Layout: [p0..p7, s_L, s_R, d_norm, th_norm]
    """
    prox    = list(robot.prox_readings)            # 8 × [0,1]
    s_l     = float(robot.target_left)             # [0,1]
    s_r     = float(robot.target_right)            # [0,1]
    dist    = distance(robot.x, robot.y, intruder_x, intruder_y)
    d_norm  = min(dist / C.TARGET_SENSOR_RANGE, 1.0)
    bearing = angle_wrap(
        angle_to(robot.x, robot.y, intruder_x, intruder_y) - robot.angle)
    th_norm = bearing / math.pi
    return np.array(prox + [s_l, s_r, d_norm, th_norm], dtype=np.float32)


# Keep the old name as an alias so run_improved_experiments.py imports work
build_observation_improved = build_frame


class HybridRobotImproved(BraitenbergRobotImproved):
    """
    Stage 3 (v2): improved Braitenberg base + RL residual motor corrections.

    At each step:
      1. Braitenberg computes base commands (v_L_base, v_R_base) with
         FIXED nominal weights — no weight-patching needed.
      2. Policy observes the stacked temporal history (built by the env
         during training, or maintained internally during evaluation).
      3. Policy outputs (Δv_L, Δv_R, floor_scale):
           v_L = v_L_base + Δv_L   (clamped to [-v_max, v_max])
           v_R = v_R_base + Δv_R
           effective_floor = C.MOTOR_FLOOR + floor_scale
      4. Improved kinematics (stuck recovery, differential drive, collision)
         run on the corrected commands.

    The residual formulation means the Braitenberg controller handles
    normal pursuit correctly and the RL only needs to learn corrections
    for the cases where the base controller fails (deadlocks, tight gaps).
    """

    def __init__(self, policy, training=False, x=None, y=None, angle=None):
        super().__init__(x=x, y=y, angle=angle)
        self.policy   = policy
        self.training = training

        # These are kept for logging / compatibility with run_improved_experiments
        self.alpha_eff   = float(C.PURSUIT_WEIGHT)
        self.beta_eff    = float(C.AVOIDANCE_WEIGHT)
        self.last_obs    = np.zeros(OBS_DIM, dtype=np.float32)
        self.last_action = np.zeros(3,       dtype=np.float32)
        self.last_logp   = 0.0
        self.last_value  = 0.0

        # Internal frame history for evaluation mode (env manages it in training)
        from collections import deque
        self._frame_history = deque(
            [np.zeros(FRAME_DIM, dtype=np.float32)] * 5, maxlen=5
        )

    def _build_stacked_obs(self, frame, intruder_x, intruder_y):
        """
        Stack the last 5 frames + 4 kinematic dims into the 74-dim obs vector.

        Kinematic dims (appended once, not per-frame):
          [v_norm, omega_norm, prox_mean, stuck_flag]
          v_norm     = mean motor speed / v_max            ∈ [−1, 1]
          omega_norm = (v_R − v_L) / (2 × v_max)          ∈ [−1, 1]
          prox_mean  = mean proximity reading              ∈ [0, 1]
          stuck_flag = 1 if recovery active, else 0        ∈ {0, 1}
        """
        self._frame_history.append(frame)
        stacked = np.concatenate(list(self._frame_history))   # (60,)

        v_norm    = float((self.v_left + self.v_right) / 2.0) / self.max_speed
        omega_norm = float((self.v_right - self.v_left) / (2.0 * self.max_speed))
        prox_mean  = float(sum(self.prox_readings) / len(self.prox_readings))
        stuck_flag = float(self._recovery_steps > 0)

        kine = np.array([v_norm, omega_norm, prox_mean, stuck_flag],
                        dtype=np.float32)
        return np.concatenate([stacked, kine])                # (64,)

    def update(self, target_x, target_y, obstacles):
        # 1. Sense
        prox   = self._cast_proximity_sensors(obstacles)
        tl, tr = self._sense_target(target_x, target_y)

        # 2. Build current frame and stacked observation
        frame = build_frame(self, target_x, target_y)
        obs   = self._build_stacked_obs(frame, target_x, target_y)
        self.last_obs = obs

        # 3. Braitenberg base command (FIXED nominal weights — no patching)
        vl_base, vr_base = self._braitenberg_control(prox, tl, tr)

        # 4. Policy query → residual corrections
        if self.training:
            action, logp, value = self.policy.sample_action(obs)
            self.last_logp  = logp
            self.last_value = value
        else:
            action = self.policy.greedy_action(obs)

        self.last_action = action
        delta_vl    = float(action[0])          # residual for left motor
        delta_vr    = float(action[1])          # residual for right motor
        floor_scale = float(action[2])          # adjustment to motor floor

        # 5. Apply residual corrections
        vl = vl_base + delta_vl
        vr = vr_base + delta_vr

        # 6. Apply dynamic motor floor (RL can raise or lower it)
        effective_floor = C.MOTOR_FLOOR + floor_scale * FLOOR_MAX
        effective_floor = max(0.0, effective_floor)   # never negative
        v_centre = (vl + vr) / 2.0
        if v_centre < effective_floor:
            deficit = effective_floor - v_centre
            vl += deficit
            vr += deficit

        # 7. Stuck recovery + kinematics + collision (from improved base)
        vl, vr = self._update_stuck(vl, vr)
        self._differential_drive(vl, vr)
        self._check_collision(obstacles)
        self._clamp_to_arena()
