# ─────────────────────────────────────────────
#  geometry.py  —  pure math utilities
# ─────────────────────────────────────────────

import math


def angle_wrap(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def distance(ax, ay, bx, by) -> float:
    return math.hypot(bx - ax, by - ay)


def angle_to(fx, fy, tx, ty) -> float:
    """Bearing FROM (fx,fy) TO (tx,ty), in radians."""
    return math.atan2(ty - fy, tx - fx)


def ray_circle_intersection(
    ox, oy,       # ray origin
    dx, dy,       # ray direction (unit vector)
    cx, cy, r,    # circle centre + radius
    max_dist,     # maximum ray length
) -> float:
    """
    Returns distance along ray to nearest intersection with circle,
    or max_dist if no hit within range.
    """
    fx, fy = ox - cx, oy - cy
    a = dx * dx + dy * dy          # == 1 if unit vector
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return max_dist
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    for t in (t1, t2):
        if 0 < t < max_dist:
            return t
    return max_dist


def ray_wall_intersection(ox, oy, dx, dy, arena_w, arena_h, max_dist) -> float:
    """Closest wall hit along ray, within max_dist."""
    best = max_dist
    # left wall x=0
    if dx < 0:
        t = -ox / dx
        if 0 < t < best:
            hy = oy + t * dy
            if 0 <= hy <= arena_h:
                best = t
    # right wall x=W
    if dx > 0:
        t = (arena_w - ox) / dx
        if 0 < t < best:
            hy = oy + t * dy
            if 0 <= hy <= arena_h:
                best = t
    # top wall y=0
    if dy < 0:
        t = -oy / dy
        if 0 < t < best:
            hx = ox + t * dx
            if 0 <= hx <= arena_w:
                best = t
    # bottom wall y=H
    if dy > 0:
        t = (arena_h - oy) / dy
        if 0 < t < best:
            hx = ox + t * dx
            if 0 <= hx <= arena_w:
                best = t
    return best
