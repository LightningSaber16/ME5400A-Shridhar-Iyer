#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  train_rl.py  —  REINFORCE with baseline (v2)
#
#  Changes from v1:
#    - compute_gradients updated for 2-layer trunk (W1/b1 + W2/b2)
#    - gradient update loop updated for W2/b2 parameters
#    - captured detection updated: reward > LAMBDA_CAPT/2 (threshold=10.0)
#      since reward scale changed (LAMBDA_CAPT = 20.0 now)
#    - print header updated to reflect new architecture
#
#  Run (Stage 3 only — Stage 2 is unaffected):
#    python train_rl.py --improved --episodes 2000
#    python train_rl.py --improved --episodes 2000 --out models_v2/
# ─────────────────────────────────────────────

import argparse
import csv
import math
import os
import random
import time

import numpy as np

import config as C
from rl_policy  import (PolicyNetwork, OBS_DIM, ACT_DIM, DELTA_MAX,
                        STAGE2_OBS_DIM, STAGE2_ACT_DIM, STAGE2_HIDDEN,
                        STAGE3_OBS_DIM, STAGE3_ACT_DIM, STAGE3_HIDDEN)
from rl_env     import SurveillanceEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EPISODES      = 2000
MAX_STEPS_EP    = 2000
LR_ACTOR        = 5e-4
LR_CRITIC       = 1e-3
LR_LOG_STD      = 1e-4
GAMMA           = 0.99
ENTROPY_COEF    = 0.01
GRAD_CLIP       = 1.0
CURRICULUM_END  = 800
SAVE_INTERVAL   = 250
EVAL_INTERVAL   = 100
N_EVAL_EPS      = 20
OUT_DIR         = "models"
SEED            = 42

# #comment this out when training fully
# N_EPISODES     = 200    # was 2000 — enough to see if reward is trending up
# MAX_STEPS_EP   = 500    # was 2000 — shorter episodes, faster iteration
# CURRICULUM_END = 40     # was 100 — keep same proportion (20% of episodes)
# EVAL_INTERVAL  = 50     # was 100 — evaluate more frequently on short run
# SAVE_INTERVAL  = 100    # was 250 — checkpoint at halfway and end
# N_EVAL_EPS     = 10     # was 20 — faster eval

DENSITY_EASY = [0.00, 0.05, 0.10]
DENSITY_FULL = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]



# ── Analytic policy gradient (updated for 2-layer trunk) ──────────────────────

def compute_gradients(policy, episode_obs, episode_actions,
                      episode_advantages, episode_returns):
    """
    Compute actor and critic gradients analytically.
    Works for both Stage 2 (obs=12, act=2, hidden=32) and
    Stage 3 (obs=64, act=3, hidden=64) by reading shape from policy.
    Both stages use the two-layer trunk (W1->W2).
    """
    dW1   = np.zeros_like(policy.W1)
    db1   = np.zeros_like(policy.b1)
    dW2   = np.zeros_like(policy.W2)
    db2   = np.zeros_like(policy.b2)
    dW_a  = np.zeros_like(policy.W_a)
    db_a  = np.zeros_like(policy.b_a)
    dW_v  = np.zeros_like(policy.W_v)
    db_v  = np.zeros_like(policy.b_v)
    d_log_std = np.zeros_like(policy.log_std)

    T = len(episode_obs)

    for t in range(T):
        obs    = episode_obs[t]
        action = episode_actions[t]
        adv    = float(episode_advantages[t])
        ret    = float(episode_returns[t])

        # ── Forward pass ──────────────────────────────────────────────────────
        z1 = policy.W1 @ obs + policy.b1          # (hidden,)
        h1 = np.maximum(0.0, z1)
        z2 = policy.W2 @ h1  + policy.b2          # (hidden,)
        h2 = np.maximum(0.0, z2)

        raw_a   = policy.W_a @ h2 + policy.b_a    # (act_dim,)
        mu      = np.tanh(raw_a) * DELTA_MAX
        std     = np.exp(np.clip(policy.log_std, -4.0, 1.0))
        var     = std ** 2 + 1e-8
        val_raw = policy.W_v @ h2 + policy.b_v    # (1,)

        # ── Actor gradients ───────────────────────────────────────────────────
        d_logp_mu    = (action - mu) / var
        d_mu_raw     = DELTA_MAX * (1.0 - np.tanh(raw_a) ** 2)
        actor_signal = adv * (d_logp_mu * d_mu_raw)

        dW_a += np.outer(actor_signal, h2)
        db_a += actor_signal
        d_log_std += adv * ((action - mu)**2 / var - 1.0) + ENTROPY_COEF

        # ── Critic gradients ──────────────────────────────────────────────────
        critic_err = float(val_raw[0]) - ret
        dW_v += critic_err * h2.reshape(1, -1)
        db_v += np.array([critic_err])

        # ── Backprop layer 2 ──────────────────────────────────────────────────
        d_h2 = (policy.W_a.T @ actor_signal
                + (policy.W_v.T * critic_err).reshape(-1))
        d_z2 = d_h2 * (z2 > 0).astype(np.float32)
        dW2 += np.outer(d_z2, h1)
        db2 += d_z2

        # ── Backprop layer 1 ──────────────────────────────────────────────────
        d_z1 = (policy.W2.T @ d_z2) * (z1 > 0).astype(np.float32)
        dW1 += np.outer(d_z1, obs)
        db1 += d_z1

    scale = 1.0 / max(1, T)
    return (dW1*scale, db1*scale, dW2*scale, db2*scale,
            dW_a*scale, db_a*scale,
            dW_v*scale, db_v*scale,
            d_log_std*scale)


def _clip_grad(g, max_norm):
    total_norm = math.sqrt(sum(float(np.sum(gi**2)) for gi in g))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        return [gi * scale for gi in g]
    return list(g)


# ── Episode rollout ───────────────────────────────────────────────────────────

def run_episode(env, policy, density=None, max_steps=MAX_STEPS_EP):
    obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
    obs  = env.reset(density=density)
    done = False
    for _ in range(max_steps):
        action, logp, value = policy.sample_action(obs)
        next_obs, reward, done, _ = env.step(action)
        obs_buf.append(obs.copy())
        act_buf.append(action.copy())
        rew_buf.append(reward)
        val_buf.append(value)
        logp_buf.append(logp)
        obs = next_obs
        if done:
            break
    return (np.array(obs_buf), np.array(act_buf),
            np.array(rew_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(logp_buf, dtype=np.float32))


def compute_returns_and_advantages(rewards, values, gamma=GAMMA):
    T       = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G       = 0.0
    for t in reversed(range(T)):
        G          = rewards[t] + gamma * G
        returns[t] = G
    advantages = returns - values
    adv_std = advantages.std()
    if adv_std > 1e-6:
        advantages = (advantages - advantages.mean()) / adv_std
    return returns, advantages


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(policy, density_levels, n_episodes, seed=0, use_improved=False):
    # policy is already instantiated with the right dims — just use it
    rng = random.Random(seed)
    env = SurveillanceEnv(policy, training=False, seed=seed + 1,
                          use_improved=use_improved)
    total_reward = 0.0
    captures     = 0
    for ep in range(n_episodes):
        density = rng.choice(density_levels)
        obs     = env.reset(density=density)
        done    = False
        ep_r    = 0.0
        for _ in range(4000):
            action             = policy.greedy_action(obs)
            obs, r, done, info = env.step(action)
            ep_r              += r
            if done:
                if info["captured"]:
                    captures += 1
                break
        total_reward += ep_r
    return total_reward / max(1, n_episodes), captures / n_episodes * 100.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train(n_episodes=N_EPISODES, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
          out_dir=OUT_DIR, base_seed=SEED, use_improved=False):

    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(base_seed)
    random.seed(base_seed)
    rng = random.Random(base_seed)

    policy = PolicyNetwork(
        seed=base_seed,
        obs_dim=STAGE3_OBS_DIM if use_improved else STAGE2_OBS_DIM,
        act_dim=STAGE3_ACT_DIM if use_improved else STAGE2_ACT_DIM,
        hidden =STAGE3_HIDDEN  if use_improved else STAGE2_HIDDEN,
    )
    env    = SurveillanceEnv(policy, training=True, seed=base_seed + 1,
                             use_improved=use_improved)

    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as lf:
        writer = csv.DictWriter(lf, fieldnames=[
            "episode", "density", "ep_reward", "ep_steps", "captured",
            "mean_reward_50", "eval_capture_pct", "elapsed_s"
        ])
        writer.writeheader()

        arch = "Residual motor (v2)" if use_improved else "Weight-tuning (v1)"
        print(f"\n{'─'*66}")
        print(f"  REINFORCE with baseline")
        print(f"  Architecture : {arch}")
        print(f"  OBS_DIM      : {OBS_DIM}  |  ACT_DIM : {ACT_DIM}")
        print(f"  Episodes     : {n_episodes}")
        print(f"  lr_actor     : {lr_actor}   lr_critic : {lr_critic}")
        print(f"  Curriculum   : easy densities for first {CURRICULUM_END} episodes")
        print(f"  Output       : {os.path.abspath(out_dir)}")
        print(f"{'─'*66}\n")

        t_start       = time.time()
        reward_window = []
        eval_cap_pct  = 0.0

        for ep in range(1, n_episodes + 1):
            levels  = DENSITY_EASY if ep <= CURRICULUM_END else DENSITY_FULL
            density = rng.choice(levels)

            obs_b, act_b, rew_b, val_b, logp_b = run_episode(
                env, policy, density=density)

            # Capture detection: last reward > half the capture bonus
            from rl_env import LAMBDA_CAPT
            captured = bool(rew_b[-1] > LAMBDA_CAPT / 2.0)
            ep_steps = len(rew_b)

            ret_b, adv_b = compute_returns_and_advantages(rew_b, val_b)

            # ── Gradients ─────────────────────────────────────────────────────
            (dW1, db1, dW2, db2,
             dW_a, db_a,
             dW_v, db_v,
             d_log_std) = compute_gradients(policy, obs_b, act_b, adv_b, ret_b)

            actor_grads  = _clip_grad(
                [dW1, db1, dW2, db2, dW_a, db_a, d_log_std], GRAD_CLIP)
            critic_grads = _clip_grad([dW_v, db_v], GRAD_CLIP)

            # ── Parameter update ──────────────────────────────────────────────
            policy.W1      += lr_actor   * actor_grads[0]
            policy.b1      += lr_actor   * actor_grads[1]
            policy.W2      += lr_actor   * actor_grads[2]
            policy.b2      += lr_actor   * actor_grads[3]
            policy.W_a     += lr_actor   * actor_grads[4]
            policy.b_a     += lr_actor   * actor_grads[5]
            policy.log_std += LR_LOG_STD * actor_grads[6]
            policy.log_std  = np.clip(policy.log_std, -4.0, 1.0)

            policy.W_v     -= lr_critic  * critic_grads[0]
            policy.b_v     -= lr_critic  * critic_grads[1]

            # ── Logging ───────────────────────────────────────────────────────
            ep_reward = float(rew_b.sum())
            reward_window.append(ep_reward)
            if len(reward_window) > 50:
                reward_window.pop(0)
            mean_r50 = sum(reward_window) / len(reward_window)

            if ep % EVAL_INTERVAL == 0:
                _, eval_cap_pct = evaluate(policy, DENSITY_FULL,
                                           N_EVAL_EPS, seed=base_seed + ep,
                                           use_improved=use_improved)
                elapsed = time.time() - t_start
                eta     = elapsed / ep * (n_episodes - ep)
                print(f"  ep {ep:5d}/{n_episodes}  "
                      f"d={density:.0%}  "
                      f"r={ep_reward:+7.2f}  "
                      f"mean50={mean_r50:+6.2f}  "
                      f"eval_cap={eval_cap_pct:5.1f}%  "
                      f"ETA {eta:.0f}s")

            writer.writerow({
                "episode"         : ep,
                "density"         : round(density, 3),
                "ep_reward"       : round(ep_reward, 4),
                "ep_steps"        : ep_steps,
                "captured"        : int(captured),
                "mean_reward_50"  : round(mean_r50, 4),
                "eval_capture_pct": round(eval_cap_pct, 2),
                "elapsed_s"       : round(time.time() - t_start, 1),
            })
            lf.flush()

            if ep % SAVE_INTERVAL == 0:
                policy.save(os.path.join(out_dir, f"policy_ep{ep:05d}"))

    final_name = "policy_improved_final" if use_improved else "policy_final"
    final = os.path.join(out_dir, final_name)
    policy.save(final)
    print(f"\n  Training complete in {time.time()-t_start:.1f}s")
    print(f"  Final policy : {final}.npz")
    print(f"  Training log : {log_path}\n")
    return policy


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Braitenberg RL policy — REINFORCE with baseline (v2)"
    )
    p.add_argument("--episodes",   type=int,   default=N_EPISODES)
    p.add_argument("--lr-actor",   type=float, default=LR_ACTOR)
    p.add_argument("--lr-critic",  type=float, default=LR_CRITIC)
    p.add_argument("--out",        type=str,   default=OUT_DIR)
    p.add_argument("--seed",       type=int,   default=SEED)
    p.add_argument("--improved",   action="store_true", default=False,
                   help="Stage 3: residual motor control on improved base")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(n_episodes=args.episodes,
          lr_actor=args.lr_actor,
          lr_critic=args.lr_critic,
          out_dir=args.out,
          base_seed=args.seed,
          use_improved=args.improved)
