"""Authoritative constants for the current five-IMU realtime product path.

The runtime follows the frozen SparseIMU baseline (``pnp5_60a``): 60 fps data,
PNP surface-sensor placement, Joints -> Poser cascade, ``win46`` online window
(40 past + 5 future frames at 30 fps = 80 + 10 frames at 60 fps).
"""

REALTIME_LAYOUT = "pnp5_forearm_shank_pelvis"
# Model input slot order. ``pelvis`` is the pelvis bone (SMPL joint 0, dorsal
# sacrum, PNP vertex 3021) -- NOT the old spine1/waist sensor.
REALTIME_LABELS = ("lw", "rw", "ls", "rs", "pelvis")
FPS = 60
PAST_FRAMES = 80
FUTURE_FRAMES = 10
WINDOW = PAST_FRAMES + FUTURE_FRAMES + 1
OUTPUT_INDEX = PAST_FRAMES
OUTPUT_LAG_FRAMES = WINDOW - 1 - OUTPUT_INDEX
ACCELERATION_SCALE = 30.0
PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)
# SMPL joints whose bones carry a sensor (pelvis 0, shanks 4/5, forearms 18/19)
# mapped to the input slot of that sensor; used by identity-anchored ("sc") models.
ANCHOR_SENSOR_SLOTS = {0: 4, 4: 2, 5: 3, 18: 0, 19: 1}
