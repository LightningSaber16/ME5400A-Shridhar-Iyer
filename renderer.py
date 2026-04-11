# ─────────────────────────────────────────────
#  renderer.py  —  Pygame visualisation layer
# ─────────────────────────────────────────────

import math
import pygame
import config as C


class Renderer:
    """
    Handles all Pygame drawing.  The simulation logic is completely separate —
    pass the current state into draw() each frame.
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((C.ARENA_W, C.ARENA_H))
        pygame.display.set_caption("Braitenberg Surveillance Sim")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 14)
        self.font_m = pygame.font.SysFont("monospace", 18)

        # Transparent overlay surface for sensor arcs
        self.overlay = pygame.Surface((C.ARENA_W, C.ARENA_H), pygame.SRCALPHA)

        self.show_sensors = True   # toggle with S key

    # ── Main draw call ────────────────────────────────────────────────────────

    def draw(self, robot, intruder, obstacles, step, captured):
        self.screen.fill(C.COL_BG)
        self.overlay.fill((0, 0, 0, 0))

        self._draw_obstacles(obstacles)

        if self.show_sensors:
            self._draw_proximity_sensors(robot, obstacles)
            self._draw_target_sensor(robot, intruder)

        self._draw_capture_ring(robot)
        self._draw_intruder(intruder)
        self._draw_robot(robot)
        self._draw_heading(robot)

        self.screen.blit(self.overlay, (0, 0))
        self._draw_hud(robot, step, captured)

        pygame.display.flip()

    # ── Sub-draw helpers ──────────────────────────────────────────────────────

    def _draw_obstacles(self, obstacles):
        for (cx, cy, r) in obstacles:
            pygame.draw.circle(self.screen, C.COL_OBSTACLE, (int(cx), int(cy)), r)
            pygame.draw.circle(self.screen, (120, 120, 140), (int(cx), int(cy)), r, 1)

    def _draw_proximity_sensors(self, robot, obstacles):
        """Draw each proximity ray, coloured by reading intensity."""
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
            pygame.draw.line(
                self.overlay,
                colour,
                (int(robot.x), int(robot.y)),
                (int(ex), int(ey)),
                1
            )

    def _draw_target_sensor(self, robot, intruder):
        """Draw the target sensor FOV arc as a transparent wedge."""
        fov = C.TARGET_SENSOR_FOV
        arc_colour = (220, 180, 60, 25)

        # Draw left and right sensor arcs
        for sign in (-1, 1):
            start_a = robot.angle + sign * fov
            end_a   = robot.angle

            steps = 20
            pts = [(int(robot.x), int(robot.y))]
            for k in range(steps + 1):
                a = start_a + (end_a - start_a) * k / steps
                px = robot.x + math.cos(a) * C.TARGET_SENSOR_RANGE
                py = robot.y + math.sin(a) * C.TARGET_SENSOR_RANGE
                pts.append((int(px), int(py)))
            if len(pts) >= 3:
                pygame.draw.polygon(self.overlay, arc_colour, pts)

    def _draw_capture_ring(self, robot):
        pygame.draw.circle(
            self.overlay,
            (*C.COL_CAPTURE, 40),
            (int(robot.x), int(robot.y)),
            C.CAPTURE_RADIUS
        )

    def _draw_robot(self, robot):
        col = C.COL_CAPTURE if robot.collided else C.COL_ROBOT
        pygame.draw.circle(self.screen, col,
                           (int(robot.x), int(robot.y)), robot.radius)
        pygame.draw.circle(self.screen, (255, 255, 255),
                           (int(robot.x), int(robot.y)), robot.radius, 1)

    def _draw_heading(self, robot):
        """Small line showing current heading direction."""
        ex = robot.x + math.cos(robot.angle) * (robot.radius + 6)
        ey = robot.y + math.sin(robot.angle) * (robot.radius + 6)
        pygame.draw.line(self.screen, (255, 255, 255),
                         (int(robot.x), int(robot.y)),
                         (int(ex), int(ey)), 2)

    def _draw_intruder(self, intruder):
        pygame.draw.circle(self.screen, C.COL_INTRUDER,
                           (int(intruder.x), int(intruder.y)), intruder.radius)
        pygame.draw.circle(self.screen, (255, 180, 180),
                           (int(intruder.x), int(intruder.y)), intruder.radius, 1)
        # Small heading indicator
        ex = intruder.x + math.cos(intruder.angle) * (intruder.radius + 4)
        ey = intruder.y + math.sin(intruder.angle) * (intruder.radius + 4)
        pygame.draw.line(self.screen, (255, 200, 200),
                         (int(intruder.x), int(intruder.y)),
                         (int(ex), int(ey)), 2)

    def _draw_hud(self, robot, step, captured):
        lines = [
            f"Step : {step}",
            f"Pos  : ({robot.x:.0f}, {robot.y:.0f})",
            f"L/R  : {robot.v_left:.2f} / {robot.v_right:.2f}",
            f"TGT L/R: {robot.target_left:.2f} / {robot.target_right:.2f}",
            f"[S] sensors {'ON' if self.show_sensors else 'OFF'}",
            f"[R] reset   [Q] quit",
        ]
        if captured:
            cap_surf = self.font_m.render("TARGET CAPTURED", True, C.COL_CAPTURE)
            self.screen.blit(cap_surf, (C.ARENA_W // 2 - 90, C.ARENA_H // 2 - 12))

        for i, line in enumerate(lines):
            surf = self.font_s.render(line, True, C.COL_TEXT)
            self.screen.blit(surf, (8, 8 + i * 18))
