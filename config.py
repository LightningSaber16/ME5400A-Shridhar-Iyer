# ─────────────────────────────────────────────
#  config.py  —  all tunable parameters
# ─────────────────────────────────────────────

# Arena
ARENA_W = 800          # pixels
ARENA_H = 800
FPS     = 60

# Robot
ROBOT_RADIUS      = 10          # px
ROBOT_MAX_SPEED   = 2.0         # px / step
ROBOT_START       = (100, 400)  # (x, y)
ROBOT_START_ANGLE = 0.0         # radians

# Sensors  ── proximity (obstacle avoidance)
N_PROX_SENSORS   = 8            # evenly spaced around body
PROX_RANGE       = 60           # px  — max detection distance
PROX_GAIN        = 2.0          # scales avoidance response

# Sensors  ── target (pursuit)
TARGET_SENSOR_FOV   = 1.5       # radians, half-angle each side (≈ 103° total)
TARGET_SENSOR_RANGE = 600       # px  — max detection distance (covers full arena)
TARGET_GAIN         = 2.5       # scales pursuit response

# Braitenberg weights
PURSUIT_WEIGHT   = 1.0          # α  — how strongly to pursue
AVOIDANCE_WEIGHT = 1.5          # β  — how strongly to avoid obstacles

# Forward bias — added to both motors every step so the robot keeps searching
# even when the target is outside sensor range
FORWARD_BIAS = 0.15              # px/step

# Motor floor — minimum speed after all channel contributions.
# Prevents avoidance from reducing speed to zero and stalling the robot.
MOTOR_FLOOR  = 0.05              # px/step

# Stuck detection and recovery
STUCK_PATIENCE    = 50          # steps between movement checkpoints
STUCK_DIST_THRESH = 6           # px — less than this = declared stuck
RECOVERY_DURATION = 15          # steps of reverse-turn before resuming

# Capture threshold
CAPTURE_RADIUS = 25             # px — "caught" when robot centre within this of intruder

# Intruder
INTRUDER_RADIUS = 15
INTRUDER_SPEED  = 1.5           # px / step
INTRUDER_START  = (700, 400)

# Obstacles  (list of (cx, cy, radius))
# Edit freely — or generate programmatically in main.py
OBSTACLES = [
    (300, 200, 25),
    (500, 350, 30),
    (250, 550, 20),
    (600, 200, 22),
    (400, 450, 28),
    (150, 300, 18),
    (650, 550, 24),
]

# Colours  (R, G, B)
COL_BG         = (15,  15,  20)
COL_ARENA      = (25,  25,  35)
COL_OBSTACLE   = (80,  80, 100)
COL_ROBOT      = (60, 200, 120)
COL_INTRUDER   = (220,  80,  80)
COL_SENSOR_RAY = (60, 100, 180, 60)   # RGBA — proximity rays
COL_TARGET_RAY = (220, 180,  60, 40)  # RGBA — target sensor arc
COL_TEXT       = (200, 200, 210)
COL_CAPTURE    = (255, 220,  80)
