#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  analyse.py  —  plot results from trials.csv
#
#  Run:
#    python analyse.py
#    python analyse.py --csv results/trials.csv --out plots/
# ─────────────────────────────────────────────

import argparse
import os
import csv
import math
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Quick switch: choose which experiment folder to analyse by default.
# Valid options: "results", "results_hybrid", "results_improved"
ANALYSIS_TARGET = "results_hybrid"

TARGET_DEFAULTS = {
    "results": {
        "csv": os.path.join("results", "trials.csv"),
        "out": "plots",
        "id": "baseline",
    },
    "results_hybrid": {
        "csv": os.path.join("results_hybrid", "trials.csv"),
        "out": "plots_hybrid",
        "id": "hybrid",
    },
    "results_improved": {
        "csv": os.path.join("results_improved", "trials.csv"),
        "out": "plots_improved",
        "id": "improved",
    },
}


# ── Load ──────────────────────────────────────────────────────────────────────

FLOAT_FIELDS = [
    "density", "capture_step", "total_steps", "collision_count",
    "path_length", "straight_line_dist", "pursuit_efficiency",
    "mean_dist_to_intruder", "min_dist_to_intruder",
    "time_to_first_detection", "mean_heading_change_rate",
]

def load_trials(csv_path: str) -> list:
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            # Strip whitespace from all values (catches \r artifacts on Windows)
            row = {k: v.strip() for k, v in row.items()}
            for field in FLOAT_FIELDS:
                try:
                    row[field] = float(row[field])
                except (ValueError, KeyError):
                    row[field] = float("nan")
            row["captured"] = int(float(row.get("captured", 0)))
            try:
                row["path_ratio"] = float(row["path_ratio"])
            except (ValueError, KeyError):
                row["path_ratio"] = float("nan")
            # Round density to 4 dp to avoid float grouping splits (0.1 vs 0.10000000001)
            row["density"] = round(row["density"], 4)
            rows.append(row)
    return rows


def group_by_density(rows: list) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["density"]].append(row)
    return dict(sorted(groups.items()))


# ── Text summary ──────────────────────────────────────────────────────────────

def _mean(vals):
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")

def _fmt(v, decimals=1):
    return f"{v:.{decimals}f}" if not (isinstance(v, float) and math.isnan(v)) else "n/a"


def print_summary(groups: dict):
    print(f"\n{'-'*90}")
    print(f"  {'Density':>8}  {'N':>4}  {'Success%':>9}  {'Mean steps':>11}  "
          f"{'Collisions':>11}  {'1st detect':>11}  {'Heading chg':>12}")
    print(f"{'-'*90}")
    for density, rows in groups.items():
        n         = len(rows)
        # Success: count rows where captured==1
        successes = sum(1 for r in rows if r["captured"] == 1)
        sr        = successes / n * 100
        ms        = _mean([r["capture_step"] for r in rows])
        mc        = _mean([r["collision_count"] for r in rows])
        # time_to_first_detection: -1 means never detected; exclude those from mean
        ttd_vals  = [r["time_to_first_detection"] for r in rows
                     if not math.isnan(r.get("time_to_first_detection", float("nan")))
                     and r.get("time_to_first_detection", -1) != -1]
        ttd       = _mean(ttd_vals)
        hcr       = _mean([r["mean_heading_change_rate"] for r in rows])
        valid_pr  = [r["path_ratio"] for r in rows
                     if not math.isnan(r["path_ratio"]) and r["path_ratio"] < 100]
        pr_str    = f"{_mean(valid_pr):.2f}" if valid_pr else "n/a"
        print(f"  {density:>8.0%}  {n:>4}  {sr:>9.1f}%  {_fmt(ms,0):>11}  "
              f"{_fmt(mc,1):>11}  {_fmt(ttd,0):>11}  {_fmt(hcr,4):>12}  "
              f"(path_ratio={pr_str})")
    print(f"{'-'*90}\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

PALETTE = ["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30",
           "#D4537E", "#7F77DD", "#639922", "#E24B4A"]

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def _next_plot_path(out_dir, stem, run_id):
    """Create a unique filename so repeated analyses do not overwrite old plots."""
    idx = 1
    while True:
        filename = f"{stem}_{run_id}_{idx:02d}.png"
        path = os.path.join(out_dir, filename)
        if not os.path.exists(path):
            return path
        idx += 1


def plot_capture_step(groups, out_dir, run_id):
    densities = list(groups.keys())
    data      = [[r["capture_step"] for r in rows] for rows in groups.values()]
    fig, ax   = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "white", "lw": 2})
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(densities) + 1))
    ax.set_xticklabels([f"{d:.0%}" for d in densities])
    _style(ax, "Capture time vs obstacle density",
           "Obstacle density", "Steps to capture (lower = faster)")
    fig.tight_layout()
    path = _next_plot_path(out_dir, "capture_step_vs_density", run_id)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_success_rate(groups, out_dir, run_id):
    densities = list(groups.keys())
    rates     = [sum(1 for r in rows if r["captured"] == 1) / len(rows) * 100
                 for rows in groups.values()]
    fig, ax   = plt.subplots(figsize=(7, 4))
    ax.plot([f"{d:.0%}" for d in densities], rates, "o-",
            color=PALETTE[0], lw=2, ms=7)
    ax.fill_between(range(len(densities)), rates, alpha=0.15, color=PALETTE[0])
    ax.set_ylim(0, 105)
    _style(ax, "Success rate vs obstacle density",
           "Obstacle density", "Capture success (%)")
    fig.tight_layout()
    path = _next_plot_path(out_dir, "success_rate_vs_density", run_id)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_path_ratio(groups, out_dir, run_id):
    densities = list(groups.keys())
    # Exclude outlier path ratios > 50 (these are trapped/stationary trials)
    data      = [[r["path_ratio"] for r in rows
                  if not math.isnan(r["path_ratio"]) and r["path_ratio"] < 50]
                 for rows in groups.values()]
    data_f    = [d if d else [float("nan")] for d in data]
    fig, ax   = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data_f, patch_artist=True, medianprops={"color": "white", "lw": 2})
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(densities) + 1))
    ax.set_xticklabels([f"{d:.0%}" for d in densities])
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="ideal (ratio=1)")
    ax.legend(fontsize=10)
    _style(ax, "Path efficiency vs obstacle density",
           "Obstacle density", "Path ratio (actual / straight-line, successful trials)")
    fig.tight_layout()
    path = _next_plot_path(out_dir, "path_ratio_vs_density", run_id)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_pursuit_efficiency(groups, out_dir, run_id):
    densities = list(groups.keys())
    means, stds = [], []
    for rows in groups.values():
        vals = [r["pursuit_efficiency"] for r in rows
                if not math.isnan(r["pursuit_efficiency"])]
        m = sum(vals) / len(vals) if vals else 0
        means.append(m)
        var = sum((v - m)**2 for v in vals) / len(vals) if vals else 0
        stds.append(var ** 0.5)
    xs = list(range(len(densities)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xs, means, color=PALETTE[2], alpha=0.8, zorder=2)
    ax.errorbar(xs, means, yerr=stds, fmt="none",
                color="gray", capsize=5, lw=1.5, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d:.0%}" for d in densities])
    _style(ax, "Pursuit efficiency vs obstacle density",
           "Obstacle density", "Pursuit efficiency index (higher = better)")
    fig.tight_layout()
    path = _next_plot_path(out_dir, "pursuit_efficiency_vs_density", run_id)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_heading_change(groups, out_dir, run_id):
    """Mean heading change rate per density — indicates avoidance thrashing."""
    densities = list(groups.keys())
    data      = [[r["mean_heading_change_rate"] for r in rows
                  if not math.isnan(r["mean_heading_change_rate"])]
                 for rows in groups.values()]
    data_f    = [d if d else [float("nan")] for d in data]
    fig, ax   = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data_f, patch_artist=True, medianprops={"color": "white", "lw": 2})
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(densities) + 1))
    ax.set_xticklabels([f"{d:.0%}" for d in densities])
    _style(ax, "Heading change rate vs obstacle density",
           "Obstacle density", "Mean |Δangle| per step (rad) — higher = more thrashing")
    fig.tight_layout()
    path = _next_plot_path(out_dir, "heading_change_vs_density", run_id)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# def plot_time_to_first_detection(groups, out_dir):
#     """How many steps before the target entered sensor range."""
#     densities = list(groups.keys())
#     data      = [[r["time_to_first_detection"] for r in rows
#                   if not math.isnan(r.get("time_to_first_detection", float("nan")))
#                   and r.get("time_to_first_detection", -1) != -1]
#                  for rows in groups.values()]
#     data_f    = [d if d else [float("nan")] for d in data]
#     fig, ax   = plt.subplots(figsize=(8, 5))
#     bp = ax.boxplot(data_f, patch_artist=True, medianprops={"color": "white", "lw": 2})
#     for patch, col in zip(bp["boxes"], PALETTE):
#         patch.set_facecolor(col); patch.set_alpha(0.75)
#     ax.set_xticks(range(1, len(densities) + 1))
#     ax.set_xticklabels([f"{d:.0%}" for d in densities])
#     _style(ax, "Time to first detection vs obstacle density",
#            "Obstacle density", "Steps until target first sensed")
#     fig.tight_layout()
#     path = os.path.join(out_dir, "time_to_first_detection_vs_density.png")
#     fig.savefig(path, dpi=150); plt.close(fig)
#     print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if ANALYSIS_TARGET not in TARGET_DEFAULTS:
        print(f"[ERROR] Unknown ANALYSIS_TARGET '{ANALYSIS_TARGET}'.")
        print(f"  Choose one of: {', '.join(TARGET_DEFAULTS)}")
        return

    defaults = TARGET_DEFAULTS[ANALYSIS_TARGET]

    p = argparse.ArgumentParser(description="Analyse experiment results for one controller")
    p.add_argument("--csv", default=defaults["csv"])
    p.add_argument("--out", default=defaults["out"])
    p.add_argument("--id", default=defaults["id"],
                   help="Identifier appended to plot filenames")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] trials.csv not found at '{args.csv}'")
        print(f"  Run the experiment for '{ANALYSIS_TARGET}' first.")
        return

    rows = load_trials(args.csv)
    groups = group_by_density(rows)

    print(f"\nAnalysing target '{ANALYSIS_TARGET}' from '{args.csv}'")
    print(f"Loaded {len(rows)} trials across {len(groups)} density levels.")
    print_summary(groups)

    if not HAS_MPL:
        print("[INFO] matplotlib/numpy not installed — pip install matplotlib numpy")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving plots to '{args.out}/'...")
    plot_capture_step(groups, args.out, args.id)
    plot_success_rate(groups, args.out, args.id)
    plot_path_ratio(groups, args.out, args.id)
    plot_pursuit_efficiency(groups, args.out, args.id)
    plot_heading_change(groups, args.out, args.id)
    #plot_time_to_first_detection(groups, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
