#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  main_fourway.py  —  run all four stages simultaneously on the same layout
#
#  Layout:  2 × 2 grid, each panel 800 × 800 px, window 1600 × 1600 px
#
#  ┌──────────────┬──────────────┐
#  │  Stage 1     │  Stage 2     │
#  │  Pure        │  RL Weights  │
#  │  Braitenberg │  (Neg Ctrl)  │
#  ├──────────────┼──────────────┤
#  │  Stage 3     │  Stage 4     │
#  │  Improved    │  Improved    │
#  │  + RL        │  No RL       │
#  └──────────────┴──────────────┘
#
#  Keys:
#    S      — toggle sensor visualisation
#    R      — reset all four to initial state
#    SPACE  — pause / unpause
#    Q/ESC  — quit
#    +/-    — speed up / slow down simulation
#
#  Usage:
#    python main_fourway.py
#    python main_fourway.py --density 0.15        # set obstacle density
#    python main_fourway.py --seed 7              # different layout seed
#    python main_fourway.py --policy-s2 models/policy_final
#    python main_fourway.py --policy-s3 models/policy_improved_final
# ─────────────────────────────────────────────────────────────────────────────

import sys
import math
import random
import argparse

import pygame

import config as C
from robot          import BraitenbergRobot
from robot_improved import BraitenbergRobotImproved
from hybrid_robot   import HybridRobot,         build_observation
from hybrid_robot_improved import HybridRobotImproved, build_frame
from intruder       import Intruder
from rl_policy      import PolicyNetwork, STAGE2_OBS_DIM, STAGE2_ACT_DIM, STAGE2_HIDDEN
from rl_policy      import STAGE3_OBS_DIM, STAGE3_ACT_DIM, STAGE3_HIDDEN
from geometry       import distance, ray_circle_intersection, ray_wall_intersection

# ── Panel layout ──────────────────────────────────────────────────────────────
PANEL_W = 300
PANEL_H = 300
WIN_W   = PANEL_W * 2      # 900
WIN_H   = PANEL_H * 2      # 900

# Scale: arena coords (0–800) → panel pixels (0–PANEL_W/H)
SX = PANEL_W / C.ARENA_W
SY = PANEL_H / C.ARENA_H

def sx(v): return int(v * SX)
def sy(v): return int(v * SY)
def sr(v): return max(1, int(v * (SX + SY) / 2))

# Panel offsets: (col, row) → pixel offset (ox, oy)
PANELS = {
    0: (0,       0),        # top-left     Stage 1
    1: (PANEL_W, 0),        # top-right    Stage 2
    2: (0,       PANEL_H),  # bottom-left  Stage 3
    3: (PANEL_W, PANEL_H),  # bottom-right Stage 4
}

STAGE_LABELS = [
    "Stage 1 — Pure Braitenberg",
    "Stage 2 — RL Adaptive Weights",
    "Stage 3 — Improved + RL",
    "Stage 4 — Improved, No RL",
]

STAGE_COLOURS = [
    (226,  75,  74),    # S1 red
    ( 59, 139, 212),    # S2 blue
    ( 29, 158, 117),    # S3 green
    (239, 159,  39),    # S4 orange
]

# ── Obstacle generation ───────────────────────────────────────────────────────

def generate_obstacles(density, seed):
    """
    Generate a random obstacle layout at the given fractional density.
    Uses `seed` for reproducibility — all four panels get the same layout.
    """
    if density < 0.005:
        return []
    rng = random.Random(seed)
    target_area = density * C.ARENA_W * C.ARENA_H
    obstacles, placed = [], 0.0
    for _ in range(10000):
        if placed >= target_area:
            break
        r  = rng.randint(15, 30)
        cx = rng.randint(r + 10, C.ARENA_W - r - 10)
        cy = rng.randint(r + 10, C.ARENA_H - r - 10)
        if distance(cx, cy, *C.ROBOT_START)    < r + 70: continue
        if distance(cx, cy, *C.INTRUDER_START) < r + 60: continue
        if any(distance(cx, cy, ex, ey) < r + er + 8
               for ex, ey, er in obstacles):           continue
        obstacles.append((cx, cy, r))
        placed += math.pi * r * r
    return obstacles

# ── Robot factory ─────────────────────────────────────────────────────────────

def make_robots(policy_s2, policy_s3):
    """Create one fresh robot per stage."""
    r1 = BraitenbergRobot()
    r2 = HybridRobot(policy_s2, training=False)
    r3 = HybridRobotImproved(policy_s3, training=False)
    r4 = BraitenbergRobotImproved()
    return [r1, r2, r3, r4]

# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_panel(screen, overlay, font_s, font_m,
               robot, intruder, obstacles,
               stage_idx, step, captured, show_sensors, ox, oy):
    """Draw one 800×800 panel at pixel offset (ox, oy)."""

    bg_col   = (15, 15, 20)
    obs_col  = (80, 80, 100)
    label_bg = STAGE_COLOURS[stage_idx]

    # Background
    panel_rect = pygame.Rect(ox, oy, PANEL_W, PANEL_H)
    pygame.draw.rect(screen, bg_col, panel_rect)

    # Obstacles
    for (cx, cy, r) in obstacles:
        pygame.draw.circle(screen, obs_col,
                           (ox + sx(cx), oy + sy(cy)), sr(r))
        pygame.draw.circle(screen, (120, 120, 140),
                           (ox + sx(cx), oy + sy(cy)), sr(r), 1)

    # Sensor rays (on overlay)
    if show_sensors:
        _draw_prox_sensors(overlay, robot, ox, oy)
        _draw_target_sensor(overlay, robot, ox, oy)

    # Capture ring
    pygame.draw.circle(overlay,
                       (*C.COL_CAPTURE, 40),
                       (ox + sx(robot.x), oy + sy(robot.y)),
                       sr(C.CAPTURE_RADIUS))

    # Intruder
    pygame.draw.circle(screen, C.COL_INTRUDER,
                       (ox + sx(intruder.x), oy + sy(intruder.y)),
                       sr(intruder.radius))
    pygame.draw.circle(screen, (255, 180, 180),
                       (ox + sx(intruder.x), oy + sy(intruder.y)),
                       sr(intruder.radius), 1)
    iex = intruder.x + math.cos(intruder.angle) * (intruder.radius + 4)
    iey = intruder.y + math.sin(intruder.angle) * (intruder.radius + 4)
    pygame.draw.line(screen, (255, 200, 200),
                     (ox + sx(intruder.x), oy + sy(intruder.y)),
                     (ox + sx(iex), oy + sy(iey)), 2)

    # Robot
    robot_col = C.COL_CAPTURE if robot.collided else C.COL_ROBOT
    pygame.draw.circle(screen, robot_col,
                       (ox + sx(robot.x), oy + sy(robot.y)),
                       sr(robot.radius))
    pygame.draw.circle(screen, (255, 255, 255),
                       (ox + sx(robot.x), oy + sy(robot.y)),
                       sr(robot.radius), 1)
    hex_ = robot.x + math.cos(robot.angle) * (robot.radius + 6)
    hey_ = robot.y + math.sin(robot.angle) * (robot.radius + 6)
    pygame.draw.line(screen, (255, 255, 255),
                     (ox + sx(robot.x), oy + sy(robot.y)),
                     (ox + sx(hex_), oy + sy(hey_)), 2)

    # Stage label bar at top of panel
    bar_rect = pygame.Rect(ox, oy, PANEL_W, 28)
    pygame.draw.rect(screen, label_bg, bar_rect)
    label_surf = font_m.render(STAGE_LABELS[stage_idx], True, (255, 255, 255))
    screen.blit(label_surf, (ox + 8, oy + 5))

    # Mini HUD (bottom-left of panel)
    hud_lines = [
        f"Step: {step}",
        f"v_L/R: {robot.v_left:.2f}/{robot.v_right:.2f}",
    ]
    for i, line in enumerate(hud_lines):
        surf = font_s.render(line, True, (200, 200, 210))
        screen.blit(surf, (ox + 6, oy + PANEL_H - 36 + i * 16))

    # CAPTURED banner
    if captured:
        cap_bg = pygame.Surface((200, 28), pygame.SRCALPHA)
        cap_bg.fill((*label_bg, 200))
        screen.blit(cap_bg, (ox + PANEL_W // 2 - 100,
                              oy + PANEL_H // 2 - 14))
        cap_surf = font_m.render("TARGET CAPTURED", True, (255, 255, 255))
        screen.blit(cap_surf, (ox + PANEL_W // 2 - 90,
                                oy + PANEL_H // 2 - 9))

    # Panel border
    pygame.draw.rect(screen, (50, 50, 70), panel_rect, 2)


def _draw_prox_sensors(overlay, robot, ox, oy):
    for i, rel_angle in enumerate(robot.prox_angles):
        world_angle = robot.angle + rel_angle
        dx = math.cos(world_angle)
        dy = math.sin(world_angle)
        reading = robot.prox_readings[i]
        ray_len = C.PROX_RANGE * (1.0 - reading)
        ex = robot.x + dx * ray_len
        ey = robot.y + dy * ray_len
        intensity = int(40 + reading * 160)
        colour = (intensity, intensity // 2, 200, 80)
        pygame.draw.line(overlay, colour,
                         (ox + sx(robot.x), oy + sy(robot.y)),
                         (ox + sx(ex), oy + sy(ey)), 1)


def _draw_target_sensor(overlay, robot, ox, oy):
    fov = C.TARGET_SENSOR_FOV
    arc_colour = (220, 180, 60, 20)
    for sign in (-1, 1):
        start_a = robot.angle + sign * fov
        end_a   = robot.angle
        steps = 20
        pts = [(ox + sx(robot.x), oy + sy(robot.y))]
        for k in range(steps + 1):
            a  = start_a + (end_a - start_a) * k / steps
            px = robot.x + math.cos(a) * min(C.TARGET_SENSOR_RANGE, C.ARENA_W)
            py = robot.y + math.sin(a) * min(C.TARGET_SENSOR_RANGE, C.ARENA_H)
            pts.append((ox + sx(px), oy + sy(py)))
        if len(pts) >= 3:
            pygame.draw.polygon(overlay, arc_colour, pts)


def draw_dividers(screen):
    """Draw cross-hair dividers between the four panels."""
    pygame.draw.line(screen, (60, 60, 80),
                     (PANEL_W, 0), (PANEL_W, WIN_H), 3)
    pygame.draw.line(screen, (60, 60, 80),
                     (0, PANEL_H), (WIN_W, PANEL_H), 3)


def draw_global_hud(screen, font_s, step, paused, speed_mult, show_sensors):
    """Small overlay in the very centre of the window."""
    lines = [
        f"Step: {step}  {'[PAUSED]' if paused else ''}",
        f"Speed: {speed_mult}x   [S] sensors {'ON' if show_sensors else 'OFF'}",
        "[R] reset  [SPACE] pause  [+/-] speed  [Q] quit",
    ]
    for i, line in enumerate(lines):
        surf = font_s.render(line, True, (180, 180, 200))
        tw = surf.get_width()
        screen.blit(surf, (WIN_W // 2 - tw // 2, WIN_H // 2 - 22 + i * 15))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Four-way stage comparison")
    parser.add_argument("--density",   type=float, default=0.15,
                        help="Obstacle density 0.0–0.35 (default 0.15)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Layout seed (default 42)")
    parser.add_argument("--policy-s2", type=str,
                        default="models/policy_final",
                        help="Path to Stage 2 policy .npz")
    parser.add_argument("--policy-s3", type=str,
                        default="models/policy_improved_final",
                        help="Path to Stage 3 policy .npz")
    args = parser.parse_args()

    # ── Load policies ─────────────────────────────────────────────────────────
    policy_s2 = PolicyNetwork(
        obs_dim=STAGE2_OBS_DIM, act_dim=STAGE2_ACT_DIM, hidden=STAGE2_HIDDEN
    )
    try:
        policy_s2.load(args.policy_s2)
        print(f"[S2] Policy loaded from {args.policy_s2}")
    except Exception as e:
        print(f"[S2] Could not load policy ({e}) — using random weights")

    policy_s3 = PolicyNetwork(
        obs_dim=STAGE3_OBS_DIM, act_dim=STAGE3_ACT_DIM, hidden=STAGE3_HIDDEN
    )
    try:
        policy_s3.load(args.policy_s3)
        print(f"[S3] Policy loaded from {args.policy_s3}")
    except Exception as e:
        print(f"[S3] Could not load policy ({e}) — using random weights")

    # ── Pygame setup ──────────────────────────────────────────────────────────
    pygame.init()
    screen  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        f"Four-Stage Comparison  |  density={args.density:.0%}  seed={args.seed}"
    )
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    clock   = pygame.time.Clock()
    font_s  = pygame.font.SysFont("monospace", 13)
    font_m  = pygame.font.SysFont("monospace", 15)

    # ── Simulation state ──────────────────────────────────────────────────────
    obstacles = generate_obstacles(args.density, args.seed)
    print(f"[SIM] {len(obstacles)} obstacles at density={args.density:.0%} seed={args.seed}")

    def reset():
        robots    = make_robots(policy_s2, policy_s3)
        # All four intruders start at same position/angle so comparison is fair
        intruders = [Intruder(mode="bounce") for _ in range(4)]
        # Sync intruder start state — same initial angle for all
        for intr in intruders:
            intr.x     = intruders[0].x
            intr.y     = intruders[0].y
            intr.angle = intruders[0].angle
        captured  = [False] * 4
        cap_step  = [None]  * 4
        return robots, intruders, captured, cap_step

    robots, intruders, captured, cap_step = reset()

    step         = 0
    paused       = False
    show_sensors = True
    speed_mult   = 1      # 1, 2, 4, 8
    MAX_STEPS    = 4000

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_r:
                    robots, intruders, captured, cap_step = reset()
                    step = 0
                    print("[SIM] Reset")

                if event.key == pygame.K_s:
                    show_sensors = not show_sensors

                if event.key == pygame.K_SPACE:
                    paused = not paused

                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    speed_mult = min(speed_mult * 2, 16)
                    print(f"[SIM] Speed: {speed_mult}x")

                if event.key == pygame.K_MINUS:
                    speed_mult = max(speed_mult // 2, 1)
                    print(f"[SIM] Speed: {speed_mult}x")

        if paused:
            # Still render but don't advance
            _render(screen, overlay, font_s, font_m,
                    robots, intruders, obstacles, captured,
                    step, show_sensors, paused, speed_mult)
            clock.tick(C.FPS)
            continue

        # ── Simulation steps (speed_mult steps per frame) ─────────────────────
        for _ in range(speed_mult):
            if step >= MAX_STEPS:
                break

            step += 1

            for i in range(4):
                if captured[i]:
                    continue

                # Advance robot
                robots[i].update(
                    intruders[i].x, intruders[i].y, obstacles
                )
                # Advance intruder — all four intruders follow identical
                # trajectories because they start with the same state and
                # update identically each step
                intruders[i].update(obstacles)

                # Capture check
                if distance(robots[i].x, robots[i].y,
                            intruders[i].x, intruders[i].y) \
                        < C.CAPTURE_RADIUS + intruders[i].radius:
                    captured[i] = True
                    cap_step[i] = step
                    print(f"[STAGE {i+1}] Captured at step {step}")

        # ── Render ────────────────────────────────────────────────────────────
        _render(screen, overlay, font_s, font_m,
                robots, intruders, obstacles, captured,
                step, show_sensors, paused, speed_mult)

        clock.tick(C.FPS)


def _render(screen, overlay, font_s, font_m,
            robots, intruders, obstacles, captured,
            step, show_sensors, paused, speed_mult):
    screen.fill((10, 10, 15))
    overlay.fill((0, 0, 0, 0))

    for i in range(4):
        ox, oy = PANELS[i]
        draw_panel(
            screen, overlay, font_s, font_m,
            robots[i], intruders[i], obstacles,
            stage_idx=i,
            step=step,
            captured=captured[i],
            show_sensors=show_sensors,
            ox=ox, oy=oy,
        )

    screen.blit(overlay, (0, 0))
    draw_dividers(screen)
    draw_global_hud(screen, font_s, step, paused, speed_mult, show_sensors)
    pygame.display.flip()


if __name__ == "__main__":
    main()
