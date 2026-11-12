"""Tracker layouts supported by the realtime pose bridge.

Keep this module dependency-free.  Runtime layout detection must not import
the historical dataset/surface-IMU generation code.

The legacy PNP model uses forearms, shanks and HIP. The 2026 waist-surface
baselines use forearms, shanks and WAIST, in that input order.
"""
from runtime_contract import REALTIME_LABELS, REALTIME_LAYOUT

LAYOUTS = {
    REALTIME_LAYOUT: {
        "labels": list(REALTIME_LABELS),
    },
    "wrists_shanks_waist": {
        "labels": ["lw", "rw", "ls", "rs", "waist"],
    },
}
