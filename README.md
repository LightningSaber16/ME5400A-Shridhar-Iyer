# ME5400 Project — Braitenberg Robot Surveillance Simulation

A four-stage study of autonomous intruder-pursuit using Braitenberg vehicles, reinforcement learning (REINFORCE with baseline), and hybrid controllers. Built with Pygame and NumPy.

---

## Project Overview

The simulation places a differential-drive robot in an 800×800 px arena. The robot must locate and capture a moving intruder while avoiding circular obstacles. Four controller architectures are compared:

| Stage | Controller | Description |
|-------|-----------|-------------|
| 1 | Pure Braitenberg | Fixed excitatory/inhibitory sensorimotor wiring |
| 2 | RL Adaptive Weights | REINFORCE tunes pursuit/avoidance gain scalars on top of Stage 1 |
| 3 | Improved + RL | Enhanced kinematics with RL residual motor corrections |
| 4 | Improved, No RL | Same enhanced kinematics as Stage 3, without RL |

---

## Repository Structure

```
.
├── config.py                        # All tunable parameters (arena, sensors, gains, colours)
├── geometry.py                      # Ray casting, angle utilities, distance functions
├── robot.py                         # Stage 1: pure Braitenberg robot
├── robot_improved.py                # Stage 4: improved Braitenberg (stuck detection, motor floor)
├── hybrid_robot.py                  # Stage 2: Braitenberg + RL weight adaptation
├── hybrid_robot_improved.py         # Stage 3: improved base + RL residual motor control
├── intruder.py                      # Moving intruder (bounce / random-walk / waypoints modes)
├── rl_env.py                        # Gym-style environment wrapper for RL training
├── rl_policy.py                     # Policy network (NumPy, two-layer MLP, actor-critic)
├── train_rl.py                      # REINFORCE training loop with curriculum
├── logger.py                        # Per-step CSV logger + trial summary
├── renderer.py                      # Pygame renderer for single-panel simulation
│
├── main.py                          # Interactive single-robot simulation (Stage 1)
├── main_fourway.py                  # Side-by-side 2×2 comparison of all four stages
│
├── run_experiments.py               # Headless batch runner (Stage 1 baseline)
├── run_improved_experiments.py      # Headless batch runner (Stage 4)
├── run_hybrid_experiments.py        # Headless batch runner (Stage 2)
├── run_improved_baseline_experiments.py  # Headless batch runner (Stage 3)
│
├── analyse.py                       # Plot and summarise results from trials.csv
├── compare.py                       # Cross-stage comparison plots
│
├── models/
│   ├── policy_final.npz             # Trained Stage 2 policy
│   ├── policy_improved_final.npz    # Trained Stage 3 policy
│   ├── policy_ep00250.npz           # Checkpoints saved every 250 episodes
│   └── training_log.csv             # Episode-by-episode training metrics
│
└── results/
    ├── trials.csv                   # Per-trial summary (all densities)
    └── steps/                       # Per-step CSVs (one file per trial)
```

---

## Quick Start

### Requirements

```bash
pip install pygame numpy matplotlib
```

Python 3.9+ recommended.

### Interactive simulation (Stage 1)

```bash
python main.py
```

Keys: `S` toggle sensors · `R` reset · `Space` pause · `Q` quit

### Four-stage side-by-side comparison

```bash
python main_fourway.py
python main_fourway.py --density 0.15 --seed 7
python main_fourway.py --policy-s2 models/policy_final --policy-s3 models/policy_improved_final
```

Keys: same as above, plus `+`/`-` to change simulation speed.

---

## Training the RL Policy

```bash
# Stage 2 — adaptive weight tuning
python train_rl.py --episodes 2000 --out models/

# Stage 3 — residual motor control on improved base
python train_rl.py --improved --episodes 2000 --out models/
```

Checkpoints are saved every 250 episodes. Training uses a curriculum: easy obstacle densities (0–10%) for the first 800 episodes, then the full range (0–30%).

---

## Running Batch Experiments

```bash
# Stage 1 baseline
python run_experiments.py

# Stage 2 hybrid
python run_hybrid_experiments.py

# Stage 3 improved + RL
python run_improved_baseline_experiments.py

# Stage 4 improved, no RL
python run_improved_experiments.py

# Custom parameters
python run_experiments.py --layouts 4 --trials 5 --densities 0.0 0.1 0.2 0.3 --out results/
```

Default: 7 density levels × 4 layouts × 5 trials = 140 trials, up to 4000 steps each. Results are written to `results/trials.csv` and `results/steps/`.

---

## Analysis

```bash
# Edit ANALYSIS_TARGET at the top of analyse.py to select the run, then:
python analyse.py

# Or specify paths directly:
python analyse.py --csv results/trials.csv --out plots/

# Cross-stage comparison:
python compare.py
```

Produces box plots for capture time, success rate, path efficiency, pursuit efficiency, and heading change rate.

---

## Key Parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ARENA_W / ARENA_H` | 800 px | Arena dimensions |
| `ROBOT_MAX_SPEED` | 2.0 px/step | Maximum wheel speed |
| `N_PROX_SENSORS` | 8 | Proximity sensors (evenly spaced) |
| `PROX_RANGE` | 60 px | Proximity sensor range |
| `TARGET_SENSOR_FOV` | 1.5 rad | Half-angle of target sensor |
| `TARGET_SENSOR_RANGE` | 600 px | Target sensor range |
| `PURSUIT_WEIGHT` | 1.0 | Braitenberg pursuit gain |
| `AVOIDANCE_WEIGHT` | 1.5 | Braitenberg avoidance gain |
| `FORWARD_BIAS` | 0.15 px/step | Constant forward drive for exploration |
| `CAPTURE_RADIUS` | 25 px | Distance threshold for capture |
| `INTRUDER_SPEED` | 1.5 px/step | Intruder movement speed |

---

## Architecture Notes

- **Sensing**: 8 proximity rays (obstacle avoidance) + 2 wide-angle target sensors (left/right hemisphere, cosine-weighted).
- **Control**: Superimposed Braitenberg channels — vehicle 2b (excitatory/crossed) for pursuit, vehicle 3a (inhibitory/crossed) for avoidance.
- **Kinematics**: Differential drive; wheelbase = 2 × robot radius.
- **RL policy**: Two-layer MLP (NumPy only). Actor outputs Gaussian action (mean via tanh); critic outputs scalar value. Trained with REINFORCE + value-function baseline, gradient clipping, and entropy regularisation.
- **Stuck recovery**: Stage 3/4 robots detect low displacement over a window and execute a reverse-turn recovery manoeuvre.
