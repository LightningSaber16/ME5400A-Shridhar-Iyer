# ─────────────────────────────────────────────
#  logger.py  —  per-step and per-trial logging
#
#  Two outputs per trial:
#    results/steps/  density{d}_layout{l}_trial{t}.csv  — one row per step
#    results/trials.csv                                  — one row per trial
#
#  Usage:
#    log = Logger(out_dir="results", density=0.10, layout_id=2, trial=0)
#    log.step(step, robot, intruder)
#    log.close(captured, capture_step, total_steps)
# ─────────────────────────────────────────────

import csv
import math
import os
from geometry import distance


STEP_FIELDS = [
    "step",
    "robot_x", "robot_y", "robot_angle",
    "robot_vl", "robot_vr",
    "intruder_x", "intruder_y",
    "dist_to_intruder",
    "target_left", "target_right",
    "prox_max",              # max proximity reading this step
    "collided",              # 1 if overlapping obstacle
    "heading_change",        # |angle_delta| from previous step (radians)
    "target_detected",       # 1 if either target sensor > 0 this step
]

TRIAL_FIELDS = [
    "density",
    "layout_id",             # which fixed obstacle layout (0-indexed)
    "trial",                 # trial index within this layout
    "captured",
    "capture_step",
    "total_steps",
    "collision_count",
    "path_length",
    "straight_line_dist",
    "path_ratio",
    "pursuit_efficiency",
    "mean_dist_to_intruder",
    "min_dist_to_intruder",
    "time_to_first_detection",   # step at which target was first sensed; -1 if never
    "mean_heading_change_rate",  # mean |angle_delta| per step (radians/step)
]


class Logger:
    def __init__(self, out_dir: str, density: float, layout_id: int, trial: int):
        self.density   = density
        self.layout_id = layout_id
        self.trial     = trial
        self.out_dir   = out_dir

        # ── per-step CSV ──────────────────────────────────────────────────────
        steps_dir = os.path.join(out_dir, "steps")
        os.makedirs(steps_dir, exist_ok=True)
        step_path = os.path.join(
            steps_dir,
            f"density{density:.2f}_layout{layout_id:02d}_trial{trial:03d}.csv"
        )
        self._step_fh  = open(step_path, "w", newline="")
        self._step_csv = csv.DictWriter(self._step_fh, fieldnames=STEP_FIELDS)
        self._step_csv.writeheader()

        # ── trial summary CSV (append-only; header written by experiment runner) ──
        os.makedirs(out_dir, exist_ok=True)
        trial_path      = os.path.join(out_dir, "trials.csv")
        self._trial_fh  = open(trial_path, "a", newline="")
        self._trial_csv = csv.DictWriter(self._trial_fh, fieldnames=TRIAL_FIELDS)

        # ── accumulators ─────────────────────────────────────────────────────
        self._collision_count = 0
        self._path_length     = 0.0
        self._prev_x          = None
        self._prev_y          = None
        self._prev_angle      = None
        self._dist_samples    = []
        self._heading_changes = []          # |angle_delta| each step
        self._robot_start_x   = None
        self._robot_start_y   = None
        self._last_robot_x    = None
        self._last_robot_y    = None

        # Time-to-first-detection: step when target was first sensed
        self._first_detection_step = -1    # -1 = never detected

    # ── Per-step call ─────────────────────────────────────────────────────────

    def step(self, step: int, robot, intruder):
        dist = distance(robot.x, robot.y, intruder.x, intruder.y)

        # Robot start position (first call only)
        if self._robot_start_x is None:
            self._robot_start_x = robot.x
            self._robot_start_y = robot.y
            self._prev_angle    = robot.angle

        # Path length
        if self._prev_x is not None:
            self._path_length += distance(robot.x, robot.y, self._prev_x, self._prev_y)
        self._prev_x       = robot.x
        self._prev_y       = robot.y
        self._last_robot_x = robot.x
        self._last_robot_y = robot.y

        # Heading change (angular displacement from previous step)
        raw_delta     = robot.angle - self._prev_angle
        # Wrap to [-pi, pi] to avoid 2π jumps at the ±π boundary
        delta         = math.atan2(math.sin(raw_delta), math.cos(raw_delta))
        heading_change = abs(delta)
        self._heading_changes.append(heading_change)
        self._prev_angle = robot.angle

        # Collisions
        if robot.collided:
            self._collision_count += 1

        # Distance samples
        self._dist_samples.append(dist)

        # Time to first detection
        target_detected = int(
            robot.target_left > 0.0 or robot.target_right > 0.0
        )
        if target_detected and self._first_detection_step == -1:
            self._first_detection_step = step

        self._step_csv.writerow({
            "step"            : step,
            "robot_x"         : round(robot.x, 2),
            "robot_y"         : round(robot.y, 2),
            "robot_angle"     : round(robot.angle, 4),
            "robot_vl"        : round(robot.v_left, 4),
            "robot_vr"        : round(robot.v_right, 4),
            "intruder_x"      : round(intruder.x, 2),
            "intruder_y"      : round(intruder.y, 2),
            "dist_to_intruder": round(dist, 2),
            "target_left"     : round(robot.target_left, 4),
            "target_right"    : round(robot.target_right, 4),
            "prox_max"        : round(max(robot.prox_readings), 4),
            "collided"        : int(robot.collided),
            "heading_change"  : round(heading_change, 6),
            "target_detected" : target_detected,
        })

    # ── Trial close ───────────────────────────────────────────────────────────

    def close(self, captured: bool, capture_step: int, total_steps: int, intruder_x: float = None, intruder_y: float = None):
        self._step_fh.close()

        # Straight-line distance: robot start → intruder position at end of trial
        # This is the meaningful denominator for path_ratio — how direct was the route
        # relative to where the target actually was when caught (or last seen).
        if intruder_x is not None and self._robot_start_x is not None:
            straight = distance(self._robot_start_x, self._robot_start_y, intruder_x, intruder_y)
        elif self._last_robot_x is not None:
            straight = distance(self._robot_start_x, self._robot_start_y, self._last_robot_x, self._last_robot_y)
        else:
            straight = 0.0

        path_ratio = (
            self._path_length / straight
            if straight > 1.0 else float("nan")
        )
        pursuit_eff = 1.0 / (capture_step * (1 + self._collision_count) + 1e-6)
        mean_dist   = (
            sum(self._dist_samples) / len(self._dist_samples)
            if self._dist_samples else float("nan")
        )
        min_dist = min(self._dist_samples) if self._dist_samples else float("nan")
        mean_hcr = (
            sum(self._heading_changes) / len(self._heading_changes)
            if self._heading_changes else float("nan")
        )

        def fmt(v):
            return round(v, 4) if not (isinstance(v, float) and math.isnan(v)) else "nan"

        self._trial_csv.writerow({
            "density"                : self.density,
            "layout_id"              : self.layout_id,
            "trial"                  : self.trial,
            "captured"               : int(captured),
            "capture_step"           : capture_step,
            "total_steps"            : total_steps,
            "collision_count"        : self._collision_count,
            "path_length"            : round(self._path_length, 2),
            "straight_line_dist"     : round(straight, 2),
            "path_ratio"             : fmt(path_ratio),
            "pursuit_efficiency"     : f"{pursuit_eff:.8f}",
            "mean_dist_to_intruder"  : fmt(mean_dist),
            "min_dist_to_intruder"   : fmt(min_dist),
            "time_to_first_detection": self._first_detection_step,
            "mean_heading_change_rate": fmt(mean_hcr),
        })
        self._trial_fh.close()
