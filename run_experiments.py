#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  run_experiments.py  —  headless batch runner
#
#  Run:
#    python run_experiments.py
#    python run_experiments.py --layouts 4 --trials 5 --out results/
#
#  Structure:
#    For each density level:
#      Generate N_LAYOUTS fixed obstacle layouts (reproducible via seed)
#      For each layout:
#        Run N_TRIALS independent trials (each with its own intruder RNG seed)
#        Log every step  -> results/steps/density{d}_layout{l}_trial{t}.csv
#        Log summary row -> results/trials.csv
#
#  Total trials = len(DENSITY_LEVELS) * N_LAYOUTS * N_TRIALS
#  Default: 8 densities * 4 layouts * 5 trials = 160 trials
# ─────────────────────────────────────────────

import argparse
import math
import os
import random
import time

import config as C
from geometry  import distance
from robot     import BraitenbergRobot
from intruder  import Intruder
from logger    import Logger


# ── Experiment parameters ─────────────────────────────────────────────────────

DENSITY_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_LAYOUTS      = 4        # fixed obstacle layouts per density level
N_TRIALS       = 5        # independent trials per layout
MAX_STEPS      = 4000     # timeout per trial
OUT_DIR        = "results"
BASE_SEED      = 42       # root seed — all layouts and trials are derived from this

OBS_RADIUS_MIN         = 15
OBS_RADIUS_MAX         = 30
MIN_CLEARANCE_ROBOT    = 60
MIN_CLEARANCE_INTRUDER = 50


# ── Obstacle generation ───────────────────────────────────────────────────────

def _arena_area() -> float:
    return C.ARENA_W * C.ARENA_H


def generate_obstacles(density: float, rng: random.Random) -> list:
    """
    Place circular obstacles until their combined area >= density * arena_area.
    Uses the provided rng so each layout seed produces a unique, reproducible layout.
    """
    if density == 0.0:
        return []

    target_area = density * _arena_area()
    obstacles   = []
    placed_area = 0.0

    for _ in range(5000):
        if placed_area >= target_area:
            break

        r  = rng.randint(OBS_RADIUS_MIN, OBS_RADIUS_MAX)
        cx = rng.randint(r + 5, C.ARENA_W - r - 5)
        cy = rng.randint(r + 5, C.ARENA_H - r - 5)

        if distance(cx, cy, *C.ROBOT_START)    < r + MIN_CLEARANCE_ROBOT:
            continue
        if distance(cx, cy, *C.INTRUDER_START) < r + MIN_CLEARANCE_INTRUDER:
            continue

        overlap = any(
            distance(cx, cy, ex, ey) < r + er + 8
            for (ex, ey, er) in obstacles
        )
        if overlap:
            continue

        obstacles.append((cx, cy, r))
        placed_area += math.pi * r * r

    return obstacles


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(
    density: float,
    layout_id: int,
    trial_idx: int,
    obstacles: list,
    max_steps: int,
    out_dir: str,
    trial_seed: int,
    intruder_mode: str = "bounce",
) -> dict:
    """
    Run one trial headlessly. trial_seed controls the intruder's initial
    heading so each trial is independently reproducible.
    """
    # Seed intruder RNG (affects initial heading and random-walk decisions)
    random.seed(trial_seed)

    robot    = BraitenbergRobot()
    intruder = Intruder(mode=intruder_mode)
    log      = Logger(
        out_dir=out_dir, density=density,
        layout_id=layout_id, trial=trial_idx
    )

    captured     = False
    capture_step = max_steps

    for step in range(1, max_steps + 1):
        robot.update(intruder.x, intruder.y, obstacles)
        intruder.update(obstacles)
        log.step(step, robot, intruder)

        if distance(robot.x, robot.y, intruder.x, intruder.y) \
                < C.CAPTURE_RADIUS + intruder.radius:
            captured     = True
            capture_step = step
            break

    log.close(captured, capture_step, step, intruder_x=intruder.x, intruder_y=intruder.y)

    # Restore global RNG state to avoid contamination between trials
    random.seed(None)

    return {
        "density"     : density,
        "layout_id"   : layout_id,
        "trial"       : trial_idx,
        "captured"    : captured,
        "capture_step": capture_step,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_all(density_levels, n_layouts, n_trials, max_steps, out_dir, base_seed):
    os.makedirs(out_dir, exist_ok=True)

    # Always start with a fresh trials.csv so reruns don't accumulate rows
    from logger import TRIAL_FIELDS
    import csv as _csv
    trial_path = os.path.join(out_dir, "trials.csv")
    with open(trial_path, "w", newline="") as _f:
        _csv.DictWriter(_f, fieldnames=TRIAL_FIELDS).writeheader()

    grand_total = len(density_levels) * n_layouts * n_trials
    done        = 0
    t_start     = time.time()

    print(f"\n{'─'*66}")
    print(f"  Braitenberg pursuit — batch experiment")
    print(f"  Densities  : {[f'{d:.0%}' for d in density_levels]}")
    print(f"  Layouts    : {n_layouts} per density")
    print(f"  Trials     : {n_trials} per layout  ({grand_total} total)")
    print(f"  Max steps  : {max_steps}")
    print(f"  Base seed  : {base_seed}")
    print(f"  Output     : {os.path.abspath(out_dir)}")
    print(f"{'─'*66}\n")

    # Per-density summary accumulators
    summary = {d: {"success": 0, "steps": []} for d in density_levels}

    for d_idx, density in enumerate(density_levels):
        print(f"  Density {density:.0%}  ", end="", flush=True)

        for layout_id in range(n_layouts):
            # Each layout gets a unique, deterministic seed derived from base_seed
            layout_seed = base_seed + d_idx * 1000 + layout_id
            layout_rng  = random.Random(layout_seed)
            obstacles   = generate_obstacles(density, layout_rng)

            for trial_idx in range(n_trials):
                # Each trial gets its own seed for intruder behaviour
                trial_seed = layout_seed * 100 + trial_idx

                result = run_trial(
                    density=density,
                    layout_id=layout_id,
                    trial_idx=trial_idx,
                    obstacles=obstacles,
                    max_steps=max_steps,
                    out_dir=out_dir,
                    trial_seed=trial_seed,
                )

                if result["captured"]:
                    summary[density]["success"] += 1
                summary[density]["steps"].append(result["capture_step"])

                done += 1
                print(".", end="", flush=True)

        s      = summary[density]
        steps  = s["steps"]
        n_done = len(steps)
        sr     = s["success"] / n_done * 100
        ms     = sum(steps) / n_done
        elapsed = time.time() - t_start
        eta     = elapsed / done * (grand_total - done) if done < grand_total else 0
        print(f"  success={sr:5.1f}%  mean_step={ms:5.0f}  ETA {eta:.0f}s")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*66}")
    print(f"  {'Density':>8}  {'Trials':>7}  {'Success%':>9}  "
          f"{'Mean steps':>11}  {'Min steps':>10}")
    print(f"{'─'*66}")
    for density in density_levels:
        s     = summary[density]
        steps = s["steps"]
        n     = len(steps)
        sr    = s["success"] / n * 100
        ms    = sum(steps) / n
        mn    = min(steps)
        print(f"  {density:>8.0%}  {n:>7}  {sr:>9.1f}%  {ms:>11.0f}  {mn:>10}")
    print(f"{'─'*66}")
    print(f"\n  Total time : {time.time() - t_start:.1f}s")
    print(f"  Step CSVs  : {out_dir}/steps/")
    print(f"  Summary    : {out_dir}/trials.csv\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Headless batch runner for Braitenberg surveillance sim"
    )
    p.add_argument("--layouts",   type=int,   default=N_LAYOUTS,
                   help=f"Fixed layouts per density (default {N_LAYOUTS})")
    p.add_argument("--trials",    type=int,   default=N_TRIALS,
                   help=f"Trials per layout (default {N_TRIALS})")
    p.add_argument("--max-steps", type=int,   default=MAX_STEPS,
                   help=f"Timeout per trial (default {MAX_STEPS})")
    p.add_argument("--out",       type=str,   default=OUT_DIR,
                   help=f"Output directory (default '{OUT_DIR}')")
    p.add_argument("--seed",      type=int,   default=BASE_SEED,
                   help=f"Base RNG seed (default {BASE_SEED})")
    p.add_argument("--densities", type=float, nargs="+",
                   default=DENSITY_LEVELS,
                   help="Space-separated density fractions e.g. 0.0 0.1 0.25")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        density_levels=args.densities,
        n_layouts=args.layouts,
        n_trials=args.trials,
        max_steps=args.max_steps,
        out_dir=args.out,
        base_seed=args.seed,
    )
