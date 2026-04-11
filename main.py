#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  main.py  —  simulation entry point
#
#  Run:
#    python main.py
#
#  Keys:
#    S  — toggle sensor visualisation
#    R  — reset simulation
#    Q  — quit
# ─────────────────────────────────────────────

import sys
import pygame

import config as C
from robot    import BraitenbergRobot
from intruder import Intruder
from renderer import Renderer
from geometry import distance
from logger   import Logger


# ── Obstacle generation ───────────────────────────────────────────────────────

def load_obstacles():
    """
    Returns the obstacle list from config.
    Swap this function out later to generate random densities for experiments.

    Example — random density:
        import random
        n = 15   # number of obstacles
        obstacles = []
        while len(obstacles) < n:
            cx = random.randint(60, C.ARENA_W - 60)
            cy = random.randint(60, C.ARENA_H - 60)
            r  = random.randint(15, 35)
            # reject if too close to robot or intruder start
            if distance(cx, cy, *C.ROBOT_START) > r + 60 and \
               distance(cx, cy, *C.INTRUDER_START) > r + 40:
                obstacles.append((cx, cy, r))
        return obstacles
    """
    return list(C.OBSTACLES)


# ── Simulation state factory ──────────────────────────────────────────────────

def make_sim(obstacles):
    robot    = BraitenbergRobot()
    intruder = Intruder(mode='bounce')   # swap to 'random_walk' or 'waypoints'
    return robot, intruder


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    renderer  = Renderer()
    clock     = pygame.time.Clock()
    obstacles = load_obstacles()
    robot, intruder = make_sim(obstacles)

    step     = 0
    captured = False
    paused   = False
    log      = Logger(out_dir="results", density=0.0, layout_id=0, trial=0)

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

                if event.key == pygame.K_s:
                    renderer.show_sensors = not renderer.show_sensors

                if event.key == pygame.K_r:
                    # Reset everything
                    robot, intruder = make_sim(obstacles)
                    step     = 0
                    captured = False
                    paused   = False
                    log      = Logger(out_dir="results", density=0.0, layout_id=0, trial=0)

                if event.key == pygame.K_SPACE:
                    paused = not paused

        if paused:
            renderer.draw(robot, intruder, obstacles, step, captured)
            clock.tick(C.FPS)
            continue

        # ── Simulation step ───────────────────────────────────────────────────
        if not captured:
            robot.update(intruder.x, intruder.y, obstacles)
            intruder.update(obstacles)
            step += 1
            log.step(step, robot, intruder)

            if distance(robot.x, robot.y, intruder.x, intruder.y) \
                    < C.CAPTURE_RADIUS + intruder.radius:
                captured = True
                log.close(True, step, step)
                print(f"[SIM] Target captured at step {step}  "
                      f"robot=({robot.x:.1f},{robot.y:.1f})  "
                      f"intruder=({intruder.x:.1f},{intruder.y:.1f})")

        # ── Render ────────────────────────────────────────────────────────────
        renderer.draw(robot, intruder, obstacles, step, captured)
        clock.tick(C.FPS)


if __name__ == "__main__":
    main()
