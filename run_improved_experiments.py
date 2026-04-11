#!/usr/bin/env python3
# run_improved_experiments.py  —  Stage 3 evaluation
#
# Evaluates the trained policy running on BraitenbergRobotImproved
# (cosine blending + forward bias + motor floor + stuck recovery).
#
# Run:
#   python run_improved_experiments.py --policy models/policy_improved_final
#   python run_improved_experiments.py --policy models/policy_improved_final --out results_improved/
#
# Uses identical density levels, layouts, and seeds as run_experiments.py
# and run_hybrid_experiments.py so all three sets of results are comparable.

import argparse
import os
import random
import time

import config as C
from geometry                import distance
from intruder                import Intruder
from hybrid_robot_improved   import HybridRobotImproved
from rl_policy               import PolicyNetwork
from logger                  import Logger, TRIAL_FIELDS
from run_experiments         import (generate_obstacles, DENSITY_LEVELS,
                                      N_LAYOUTS, N_TRIALS, MAX_STEPS)
import csv as _csv


def run_improved_trial(density, layout_id, trial_idx, obstacles,
                       max_steps, out_dir, trial_seed, policy):
    random.seed(trial_seed)

    robot    = HybridRobotImproved(policy, training=False)
    intruder = Intruder(mode="bounce")
    log      = Logger(out_dir=out_dir, density=density,
                      layout_id=layout_id, trial=trial_idx)

    captured     = False
    capture_step = max_steps

    for step in range(1, max_steps + 1):
        robot.update(intruder.x, intruder.y, obstacles)
        intruder.update(obstacles)
        log.step(step, robot, intruder)

        if distance(robot.x, robot.y, intruder.x, intruder.y)                 < C.CAPTURE_RADIUS + intruder.radius:
            captured     = True
            capture_step = step
            break

    log.close(captured, capture_step, step,
               intruder_x=intruder.x, intruder_y=intruder.y)
    random.seed(None)

    return {"density": density, "layout_id": layout_id,
            "trial": trial_idx, "captured": captured,
            "capture_step": capture_step}


def run_all_improved(policy_path, density_levels, n_layouts, n_trials,
                     max_steps, out_dir, base_seed):
    policy = PolicyNetwork()
    policy.load(policy_path)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "trials.csv"), "w", newline="") as f:
        _csv.DictWriter(f, fieldnames=TRIAL_FIELDS).writeheader()

    grand_total = len(density_levels) * n_layouts * n_trials
    done        = 0
    t_start     = time.time()
    summary     = {d: {"success": 0, "steps": []} for d in density_levels}

    print(f"\n{chr(8212)*66}")
    print(f"  Stage 3 — Improved hybrid policy evaluation")
    print(f"  Policy     : {policy_path}.npz")
    print(f"  Densities  : {[f for d in density_levels]}")
    print(f"  Layouts    : {n_layouts}  |  Trials/layout : {n_trials}")
    print(f"  Total      : {grand_total} trials")
    print(f"{chr(8212)*66}\n")

    for d_idx, density in enumerate(density_levels):
        print(f"  Density {density:.0%}  ", end="", flush=True)

        for layout_id in range(n_layouts):
            layout_seed = base_seed + d_idx * 1000 + layout_id
            obstacles   = generate_obstacles(density, random.Random(layout_seed))

            for trial_idx in range(n_trials):
                trial_seed = layout_seed * 100 + trial_idx
                result = run_improved_trial(
                    density, layout_id, trial_idx, obstacles,
                    max_steps, out_dir, trial_seed, policy
                )
                if result["captured"]:
                    summary[density]["success"] += 1
                summary[density]["steps"].append(result["capture_step"])
                done += 1
                print(".", end="", flush=True)

        s   = summary[density]
        n   = len(s["steps"])
        sr  = s["success"] / n * 100
        ms  = sum(s["steps"]) / n
        eta = (time.time()-t_start)/done*(grand_total-done) if done < grand_total else 0
        print(f"  success={sr:5.1f}%  mean_step={ms:5.0f}  ETA {eta:.0f}s")

    print(f"\n{chr(8212)*66}")
    print(f"  {'Density':>8}  {'Trials':>7}  {'Success%':>9}  "
          f"{'Mean steps':>11}  {'Min steps':>10}")
    print(f"{chr(8212)*66}")
    for density in density_levels:
        s     = summary[density]
        steps = s["steps"]
        n     = len(steps)
        print(f"  {density:>8.0%}  {n:>7}  "
              f"{s['success']/n*100:>9.1f}%  "
              f"{sum(steps)/n:>11.0f}  {min(steps):>10}")
    print(f"{chr(8212)*66}")
    print(f"\n  Total time : {time.time()-t_start:.1f}s")
    print(f"  Summary    : {out_dir}/trials.csv\n")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Stage 3 improved hybrid policy")
    p.add_argument("--policy",    default=os.path.join("models", "policy_improved_final"),
                   help="Path to .npz policy (without extension)")
    p.add_argument("--out",       default="results_improved")
    p.add_argument("--layouts",   type=int,   default=N_LAYOUTS)
    p.add_argument("--trials",    type=int,   default=N_TRIALS)
    p.add_argument("--max-steps", type=int,   default=MAX_STEPS)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--densities", type=float, nargs="+", default=DENSITY_LEVELS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_improved(
        policy_path    = args.policy,
        density_levels = args.densities,
        n_layouts      = args.layouts,
        n_trials       = args.trials,
        max_steps      = args.max_steps,
        out_dir        = args.out,
        base_seed      = args.seed,
    )
