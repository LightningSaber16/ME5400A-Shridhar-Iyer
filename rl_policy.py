# rl_policy.py  —  Stage 3 (v2): residual motor control + temporal history
#
# Changes from v1:
#   OBS_DIM : 12 → 74  (12-dim single frame × 5 history steps + 4 kinematic dims)
#              Layout per frame: [p0..p7, s_L, s_R, d_norm, th_norm]  (12)
#              Kinematic dims appended once (not stacked): [v_norm, omega_norm,
#                                                           prox_mean, stuck_flag]
#   ACT_DIM : 2  → 3   (Δv_L, Δv_R residual corrections + floor_scale)
#   HIDDEN  : 32 → 64  (wider trunk to handle richer input)
#   DELTA_MAX: 0.5 → 1.0  (residuals in motor-speed units, not weight units)
#   FLOOR_MAX: 0.5        (floor_scale ∈ [-0.5, +0.5] added to C.MOTOR_FLOOR)

import math, os
import numpy as np
import config as C

HISTORY_LEN = 5           # number of past frames stacked
FRAME_DIM   = 12          # dims per frame: 8 prox + 2 target + d_norm + th_norm
KINE_DIM    = 4           # extra kinematic dims appended once
# Both stages use the same obs layout (stacked frames + kinematics = 64 dims)
# and the same two-layer network. The only difference is:
#   Stage 2: act_dim=2 (Δα, Δβ weight tuning), hidden=32 (smaller — original base)
#   Stage 3: act_dim=3 (Δv_L, Δv_R, floor_scale residuals), hidden=64 (larger)
STAGE2_OBS_DIM = FRAME_DIM * HISTORY_LEN + KINE_DIM   # 64 — same as Stage 3
STAGE2_ACT_DIM = 2
STAGE2_HIDDEN  = 32   # smaller network for original base

STAGE3_OBS_DIM = FRAME_DIM * HISTORY_LEN + KINE_DIM   # 64
STAGE3_ACT_DIM = 3
STAGE3_HIDDEN  = 64

# Module-level defaults (Stage 3)
OBS_DIM  = STAGE3_OBS_DIM
ACT_DIM  = STAGE3_ACT_DIM
HIDDEN   = STAGE3_HIDDEN
DELTA_MAX   = 1.0         # max residual motor correction (px/step)
FLOOR_MAX   = 0.5         # max adjustment to motor floor


def _relu(x):
    return np.maximum(0.0, x)


def _xavier(fan_in, fan_out, rng):
    s = math.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, s, (fan_out, fan_in)).astype(np.float32)


class PolicyNetwork:
    """
    Shared-trunk actor-critic (pure numpy).

    Trunk  : Linear(OBS_DIM, 64) -> ReLU -> Linear(64, 64) -> ReLU
    Actor  : Linear(64, 3)  -> tanh * DELTA_MAX  (Δv_L, Δv_R, floor_scale)
    Critic : Linear(64, 1)                        (state value)

    Two hidden layers give the network enough capacity to reason over
    the stacked temporal history without overfitting on single-step patterns.
    """

    def __init__(self, seed=0, obs_dim=None, act_dim=None, hidden=None):
        # Default to Stage 3 dimensions; Stage 2 passes its own values explicitly
        obs_dim = obs_dim if obs_dim is not None else STAGE3_OBS_DIM
        act_dim = act_dim if act_dim is not None else STAGE3_ACT_DIM
        hidden  = hidden  if hidden  is not None else STAGE3_HIDDEN

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden  = hidden

        rng = np.random.default_rng(seed)
        # Two-layer trunk
        self.W1      = _xavier(obs_dim, hidden, rng)
        self.b1      = np.zeros(hidden,   dtype=np.float32)
        self.W2      = _xavier(hidden,  hidden, rng)
        self.b2      = np.zeros(hidden,   dtype=np.float32)
        # Heads
        self.W_a     = _xavier(hidden, act_dim, rng)
        self.b_a     = np.zeros(act_dim,  dtype=np.float32)
        self.W_v     = _xavier(hidden, 1, rng)
        self.b_v     = np.zeros(1,        dtype=np.float32)
        self.log_std = np.full(act_dim, -0.5, dtype=np.float32)

    def _trunk(self, obs):
        h1 = _relu(self.W1 @ obs + self.b1)
        return _relu(self.W2 @ h1 + self.b2)

    def actor(self, obs):
        mu  = np.tanh(self.W_a @ self._trunk(obs) + self.b_a) * DELTA_MAX
        std = np.exp(np.clip(self.log_std, -4.0, 1.0))
        return mu, std

    def critic(self, obs):
        return float((self.W_v @ self._trunk(obs) + self.b_v)[0])

    def sample_action(self, obs):
        mu, std  = self.actor(obs)
        action   = np.clip(mu + np.random.randn(self.act_dim) * std,
                           -DELTA_MAX, DELTA_MAX)
        log_prob = self._log_prob(action, mu, std)
        return action, log_prob, self.critic(obs)

    def greedy_action(self, obs):
        mu, _ = self.actor(obs)
        return mu

    def log_prob_of(self, obs, action):
        mu, std = self.actor(obs)
        return self._log_prob(action, mu, std)

    def _log_prob(self, action, mu, std):
        var = std ** 2 + 1e-8
        return float(-0.5 * np.sum((action - mu)**2 / var + np.log(2 * math.pi * var)))

    def _param_list(self):
        return [self.W1, self.b1, self.W2, self.b2,
                self.W_a, self.b_a,
                self.W_v, self.b_v, self.log_std]

    def flat_params(self):
        return np.concatenate([p.ravel() for p in self._param_list()])

    def set_flat_params(self, flat):
        idx = 0
        for p in self._param_list():
            n = p.size
            p.flat[:] = flat[idx: idx + n]
            idx += n

    def get_params_copy(self):
        return [p.copy() for p in self._param_list()]

    def set_params(self, params):
        for dst, src in zip(self._param_list(), params):
            dst[:] = src

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W_a=self.W_a, b_a=self.b_a,
                 W_v=self.W_v, b_v=self.b_v,
                 log_std=self.log_std)
        print(f"  [policy] saved -> {path}.npz")

    def load(self, path):
        if not path.endswith(".npz"):
            path += ".npz"
        d = np.load(path)
        self.W1, self.b1   = d["W1"],  d["b1"]
        self.W2, self.b2   = d["W2"],  d["b2"]
        self.W_a, self.b_a = d["W_a"], d["b_a"]
        self.W_v, self.b_v = d["W_v"], d["b_v"]
        self.log_std       = d["log_std"]
        print(f"  [policy] loaded <- {path}")
