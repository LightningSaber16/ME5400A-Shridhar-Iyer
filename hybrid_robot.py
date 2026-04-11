# hybrid_robot.py  (v2)
#
# HybridRobot — Stage 2: RL weight tuning on original Braitenberg base.
#
# Changes from v1:
#   - update() now builds a 64-dim stacked obs (5-frame history + kinematics)
#     to match the policy network's expected input dimension.
#   - Internal _frame_history deque mirrors HybridRobotImproved's approach.
#   - build_observation() kept for backward compatibility with run_hybrid_experiments.py
#   - Action remains (Δα, Δβ) — Stage 2 definition is unchanged.

import math
from collections import deque

import numpy as np

from robot     import BraitenbergRobot
from rl_policy import (PolicyNetwork, DELTA_MAX,
                       STAGE2_OBS_DIM, FRAME_DIM, HISTORY_LEN, KINE_DIM)
from geometry  import distance, angle_wrap, angle_to
import config as C

ALPHA_MIN, ALPHA_MAX = 0.1, 2.5
BETA_MIN,  BETA_MAX  = 0.1, 2.5

# Keep OBS_DIM pointing at the per-frame dim for any legacy imports
OBS_DIM = FRAME_DIM


def build_observation(robot, intruder_x, intruder_y):
    """
    Build the 12-dim single-frame observation vector.
    Kept for backward compatibility with run_hybrid_experiments.py.

    Layout: [p0..p7, s_L, s_R, d_norm, theta_norm]
    """
    prox    = list(robot.prox_readings)
    s_l     = float(robot.target_left)
    s_r     = float(robot.target_right)
    dist    = distance(robot.x, robot.y, intruder_x, intruder_y)
    d_norm  = min(dist / C.TARGET_SENSOR_RANGE, 1.0)
    bearing = angle_wrap(
        angle_to(robot.x, robot.y, intruder_x, intruder_y) - robot.angle)
    th_norm = bearing / math.pi
    return np.array(prox + [s_l, s_r, d_norm, th_norm], dtype=np.float32)


class HybridRobot(BraitenbergRobot):
    """
    Stage 2: original Braitenberg base + RL-adapted weights (Δα, Δβ).

    Maintains an internal 5-frame history so that greedy_action() during
    evaluation receives the same 64-dim stacked obs the policy was trained on.
    During training, the env builds the stacked obs and passes it in via
    _apply_action_weights() — robot.update() is not called in that path.
    """

    def __init__(self, policy, training=False, x=None, y=None, angle=None):
        super().__init__(x=x, y=y, angle=angle)
        self.policy   = policy
        self.training = training

        self.alpha_eff   = float(C.PURSUIT_WEIGHT)
        self.beta_eff    = float(C.AVOIDANCE_WEIGHT)
        self.last_obs    = np.zeros(STAGE2_OBS_DIM, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_logp   = 0.0
        self.last_value  = 0.0

        # Internal frame history for evaluation mode
        self._frame_history = deque(
            [np.zeros(FRAME_DIM, dtype=np.float32)] * HISTORY_LEN,
            maxlen=HISTORY_LEN
        )

    def _build_stacked_obs(self, frame):
        """
        Append frame to history and return 64-dim stacked obs.
        Layout matches rl_env._get_obs() exactly.
        """
        self._frame_history.append(frame)
        stacked = np.concatenate(list(self._frame_history))  # (60,)

        v_norm     = float((self.v_left + self.v_right) / 2.0) / self.max_speed
        omega_norm = float((self.v_right - self.v_left) / (2.0 * self.max_speed))
        prox_mean  = float(sum(self.prox_readings) / len(self.prox_readings))
        stuck_flag = 0.0   # original robot has no stuck recovery

        kine = np.array([v_norm, omega_norm, prox_mean, stuck_flag],
                        dtype=np.float32)
        return np.concatenate([stacked, kine])   # (64,)

    def update(self, target_x, target_y, obstacles):
        # 1. Sense
        prox   = self._cast_proximity_sensors(obstacles)
        tl, tr = self._sense_target(target_x, target_y)

        # 2. Build 64-dim stacked obs
        frame = build_observation(self, target_x, target_y)   # 12-dim frame
        obs   = self._build_stacked_obs(frame)                 # 64-dim stacked
        self.last_obs = obs

        # 3. Policy query
        if self.training:
            action, logp, value = self.policy.sample_action(obs)
            self.last_logp  = logp
            self.last_value = value
        else:
            action = self.policy.greedy_action(obs)

        self.last_action = action
        da, db = float(action[0]), float(action[1])

        self.alpha_eff = float(np.clip(C.PURSUIT_WEIGHT   + da, ALPHA_MIN, ALPHA_MAX))
        self.beta_eff  = float(np.clip(C.AVOIDANCE_WEIGHT + db, BETA_MIN,  BETA_MAX))

        # 4. Patch config, run Braitenberg, restore
        orig_alpha, orig_beta = C.PURSUIT_WEIGHT, C.AVOIDANCE_WEIGHT
        C.PURSUIT_WEIGHT      = self.alpha_eff
        C.AVOIDANCE_WEIGHT    = self.beta_eff

        try:
            vl, vr = self._braitenberg_control(prox, tl, tr)
        finally:
            C.PURSUIT_WEIGHT   = orig_alpha
            C.AVOIDANCE_WEIGHT = orig_beta

        # 5. Kinematics + collision
        self._differential_drive(vl, vr)
        self._check_collision(obstacles)
        self._clamp_to_arena()
