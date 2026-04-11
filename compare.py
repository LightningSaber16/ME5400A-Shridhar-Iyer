#!/usr/bin/env python3
# compare.py  —  side-by-side plots of all four pipeline stages
#
# Run:
#   python compare.py
#   python compare.py \
#       --baseline           results/trials.csv \
#       --hybrid             results_hybrid/trials.csv \
#       --improved           results_improved/trials.csv \
#       --improved-baseline  results_improved_baseline/trials.csv \
#       --out                plots_compare/
#
# Produces 4 comparison plots, each showing all four controllers
# on the same axes with a shared legend:
#   1. success_rate_comparison.png
#   2. capture_step_comparison.png
#   3. collision_count_comparison.png
#   4. pursuit_efficiency_comparison.png
#
# Pipeline:
#   Stage 1 — Pure Braitenberg          (original base, fixed weights)
#   Stage 2 — RL Adaptive Weights       (original base, RL weights)
#   Stage 3 — Improved + RL             (improved base, RL weights)
#   Stage 4 — Improved Baseline (no RL) (improved base, fixed weights)

import argparse
import csv
import math
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FLOAT_FIELDS = [
    "density", "capture_step", "total_steps", "collision_count",
    "path_length", "straight_line_dist", "pursuit_efficiency",
    "mean_dist_to_intruder", "min_dist_to_intruder",
    "time_to_first_detection", "mean_heading_change_rate",
]

# Controller colours, markers, and labels — four stages
CONTROLLERS = [
    ("baseline",          "#E24B4A", "o", "Stage 1 — Pure Braitenberg"),
    ("hybrid",            "#3B8BD4", "o", "Stage 2 — RL Adaptive Weights"),
    ("improved",          "#1D9E75", "o", "Stage 3 — Improved + RL"),
    ("improved_baseline", "#EF9F27", "D", "Stage 4 — Improved, No RL"),
]


# ── Load ──────────────────────────────────────────────────────────────────────

def load_trials(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k: v.strip() for k, v in row.items()}
            for field in FLOAT_FIELDS:
                try:
                    row[field] = float(row[field])
                except (ValueError, KeyError):
                    row[field] = float("nan")
            row["captured"]   = int(float(row.get("captured", 0)))
            try:
                row["path_ratio"] = float(row.get("path_ratio", "nan"))
            except (ValueError, TypeError):
                row["path_ratio"] = float("nan")
            row["density"] = round(row["density"], 4)
            rows.append(row)
    return rows


def group_by_density(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["density"]].append(r)
    return dict(sorted(g.items()))


def _mean(vals):
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def _std(vals):
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if len(clean) < 2:
        return 0.0
    m = sum(clean) / len(clean)
    return math.sqrt(sum((v-m)**2 for v in clean) / len(clean))


# ── Shared style ──────────────────────────────────────────────────────────────

def _style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    if legend:
        ax.legend(fontsize=9, framealpha=0.85)


def _density_labels(densities):
    return [f"{d:.0%}" for d in densities]


# ── Text summary ──────────────────────────────────────────────────────────────

def print_comparison(all_groups):
    densities = sorted(set(d for g in all_groups.values() for d in g.keys()))
    sep = "─" * 110
    print(f"\n{sep}")
    print(f"  Four-way comparison — capture rate (%)")
    print(f"  {'Density':>8}  ", end="")
    for key, _, _, label in CONTROLLERS:
        if key in all_groups:
            print(f"  {label[:30]:>30}", end="")
    print()
    print(sep)
    for d in densities:
        print(f"  {d:>8.0%}", end="")
        for key, _, _, _ in CONTROLLERS:
            if key not in all_groups or d not in all_groups[key]:
                print(f"  {'—':>30}", end="")
                continue
            rows = all_groups[key][d]
            n    = len(rows)
            sr   = sum(1 for r in rows if r["captured"] == 1) / n * 100
            ms   = _mean([r["capture_step"] for r in rows])
            mc   = _mean([r["collision_count"] for r in rows])
            print(f"  {sr:5.1f}%  {ms:6.0f}  {mc:5.1f}col{'':<4}", end="")
        print()
    print(f"{sep}\n")


# ── Plot 1: Success rate ───────────────────────────────────────────────────────

def plot_success_rate(all_groups, out_dir):
    # Determine a consistent x-axis from whichever controller has the most densities
    all_densities = sorted(set(d for g in all_groups.values() for d in g.keys()))
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, colour, marker, label in CONTROLLERS:
        if key not in all_groups:
            continue
        groups    = all_groups[key]
        densities = sorted(groups.keys())
        rates     = [sum(1 for r in groups[d] if r["captured"] == 1)
                     / len(groups[d]) * 100
                     for d in densities]
        xs = [all_densities.index(d) for d in densities]
        # Stage 4 uses dashed line to distinguish it visually as ablation
        ls = "--" if key == "improved_baseline" else "-"
        ax.plot(xs, rates, marker=marker, linestyle=ls,
                color=colour, lw=2.2, ms=7, label=label)
        ax.fill_between(xs, rates, alpha=0.07, color=colour)

    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(all_densities)))
    ax.set_xticklabels(_density_labels(all_densities))
    _style(ax, "Capture success rate — all four controllers",
           "Obstacle density", "Capture success (%)")
    fig.tight_layout()
    path = os.path.join(out_dir, "success_rate_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: Capture step ───────────────────────────────────────────────────────

def plot_capture_step(all_groups, out_dir):
    all_densities = sorted(set(d for g in all_groups.values() for d in g.keys()))
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, colour, marker, label in CONTROLLERS:
        if key not in all_groups:
            continue
        groups    = all_groups[key]
        densities = sorted(groups.keys())
        means = [_mean([r["capture_step"] for r in groups[d]]) for d in densities]
        stds  = [_std( [r["capture_step"] for r in groups[d]]) for d in densities]
        xs    = [all_densities.index(d) for d in densities]
        ls    = "--" if key == "improved_baseline" else "-"
        ax.plot(xs, means, marker=marker, linestyle=ls,
                color=colour, lw=2.2, ms=7, label=label)
        ax.fill_between(xs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.10, color=colour)

    ax.set_xticks(range(len(all_densities)))
    ax.set_xticklabels(_density_labels(all_densities))
    ax.axhline(4000, color="gray", linestyle=":", alpha=0.5, lw=1.2, label="timeout")
    _style(ax, "Capture time — mean ± 1 std",
           "Obstacle density", "Steps to capture")
    fig.tight_layout()
    path = os.path.join(out_dir, "capture_step_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: Collision count ────────────────────────────────────────────────────

def plot_collision_count(all_groups, out_dir):
    all_densities = sorted(set(d for g in all_groups.values() for d in g.keys()))
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, colour, marker, label in CONTROLLERS:
        if key not in all_groups:
            continue
        groups    = all_groups[key]
        densities = sorted(groups.keys())
        means = [_mean([r["collision_count"] for r in groups[d]]) for d in densities]
        stds  = [_std( [r["collision_count"] for r in groups[d]]) for d in densities]
        xs    = [all_densities.index(d) for d in densities]
        ls    = "--" if key == "improved_baseline" else "-"
        ax.plot(xs, means, marker=marker, linestyle=ls,
                color=colour, lw=2.2, ms=7, label=label)
        ax.fill_between(xs,
                        [max(0, m - s) for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.10, color=colour)

    ax.set_xticks(range(len(all_densities)))
    ax.set_xticklabels(_density_labels(all_densities))
    _style(ax, "Collision count — mean ± 1 std",
           "Obstacle density", "Collision steps per trial")
    fig.tight_layout()
    path = os.path.join(out_dir, "collision_count_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4: Pursuit efficiency ─────────────────────────────────────────────────

def plot_pursuit_efficiency(all_groups, out_dir):
    all_densities = sorted(set(d for g in all_groups.values() for d in g.keys()))
    n_d   = len(all_densities)
    n_c   = sum(1 for key, _, _, _ in CONTROLLERS if key in all_groups)
    width = 0.8 / n_c
    fig, ax = plt.subplots(figsize=(11, 5))

    for c_idx, (key, colour, _, label) in enumerate(CONTROLLERS):
        if key not in all_groups:
            continue
        groups = all_groups[key]
        means, stds, xs = [], [], []
        for d_idx, d in enumerate(all_densities):
            if d not in groups:
                continue
            vals = [r["pursuit_efficiency"] for r in groups[d]
                    if not math.isnan(r["pursuit_efficiency"])]
            means.append(_mean(vals))
            stds.append(_std(vals))
            xs.append(d_idx + (c_idx - n_c / 2 + 0.5) * width)
        # Stage 4 uses hatching to distinguish as ablation condition
        hatch = "//" if key == "improved_baseline" else None
        ax.bar(xs, means, width=width * 0.9, color=colour,
               alpha=0.8, label=label, zorder=2, hatch=hatch)
        ax.errorbar(xs, means, yerr=stds, fmt="none",
                    color="gray", capsize=3, lw=1.2, zorder=3)

    ax.set_xticks(range(n_d))
    ax.set_xticklabels(_density_labels(all_densities))
    _style(ax, "Pursuit efficiency index — all controllers",
           "Obstacle density", "Efficiency η (higher = better)")
    fig.tight_layout()
    path = os.path.join(out_dir, "pursuit_efficiency_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare all four pipeline stages on the same plots"
    )
    p.add_argument("--baseline",
                   default=os.path.join("results", "trials.csv"),
                   help="Stage 1 trials.csv")
    p.add_argument("--hybrid",
                   default=os.path.join("results_hybrid", "trials.csv"),
                   help="Stage 2 trials.csv")
    p.add_argument("--improved",
                   default=os.path.join("results_improved", "trials.csv"),
                   help="Stage 3 trials.csv")
    p.add_argument("--improved-baseline",
                   default=os.path.join("results_improved_baseline", "trials.csv"),
                   help="Stage 4 trials.csv")
    p.add_argument("--out", default="plots_compare")
    args = p.parse_args()

    paths = {
        "baseline"          : args.baseline,
        "hybrid"            : args.hybrid,
        "improved"          : args.improved,
        "improved_baseline" : args.improved_baseline,
    }

    all_groups = {}
    for key, path in paths.items():
        if os.path.exists(path):
            rows = load_trials(path)
            all_groups[key] = group_by_density(rows)
            print(f"  Loaded {len(rows):4d} trials from {path}")
        else:
            print(f"  [SKIP] {path} not found")

    if not all_groups:
        print("No results found. Run experiments first.")
        return

    print_comparison(all_groups)

    if not HAS_MPL:
        print("[INFO] matplotlib not installed — pip install matplotlib numpy")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving comparison plots to '{args.out}/'...")
    plot_success_rate(all_groups, args.out)
    plot_capture_step(all_groups, args.out)
    plot_collision_count(all_groups, args.out)
    plot_pursuit_efficiency(all_groups, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
