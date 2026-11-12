"""Map live SlimeVR tracker bindings to the current five-IMU layout.

SlimeVR-Server assigns each tracker a ``BodyPart`` (see
``SlimeVR-Server/server/core/.../tracking/trackers/TrackerPosition.kt`` on the
Kotlin side, or ``solarxr_protocol.datatypes.BodyPart`` on the wire). The
current production checkpoints are trained for one fixed 5-sensor placement
(PNP surface vertices): both forearms, both shanks and the pelvis. This module
bridges the live tracker set to that layout definition.

Only an *exact* match is accepted: silently falling back to a "closest" layout
would produce a plausible-looking but wrong pose.

Physical placement (SMPL T-pose geometry of the training sensors):
  forearms  ~85-87 % of the way from elbow to wrist, dorsal side (back of the forearm)
  shanks    ~25 % of the way from knee to ankle (just below the knee), anteromedial
            (inner front surface of the tibia)
  legacy pelvis model: posterior pelvis / sacrum, bind to HIP
  waist-surface models: torso waist attachment, bind to WAIST
"""
from __future__ import annotations

import dataclasses
from typing import Dict, FrozenSet, List, Optional

from production.layouts import LAYOUTS
from production.solarxr_client import TrackerSample
from solarxr_protocol.datatypes.BodyPart import BodyPart

# BodyPart -> production model input label. Everything else (hands, head, chest,
# thighs, feet, upper arms, ...) is irrelevant to this checkpoint family.
# HIP and WAIST are distinct model labels; neither is substituted for the other.
BODY_PART_TO_LABEL: Dict[int, str] = {
    BodyPart.LEFT_LOWER_ARM: "lw",    # forearm-mounted input slot
    BodyPart.RIGHT_LOWER_ARM: "rw",
    BodyPart.LEFT_LOWER_LEG: "ls",    # shank
    BodyPart.RIGHT_LOWER_LEG: "rs",
    BodyPart.HIP: "pelvis",
    BodyPart.WAIST: "waist",
}
# Preferred body part per label (what to bind in SlimeVR-Server).
LABEL_TO_BODY_PART: Dict[str, int] = {
    "lw": BodyPart.LEFT_LOWER_ARM,
    "rw": BodyPart.RIGHT_LOWER_ARM,
    "ls": BodyPart.LEFT_LOWER_LEG,
    "rs": BodyPart.RIGHT_LOWER_LEG,
    "pelvis": BodyPart.HIP,
    "waist": BodyPart.WAIST,
}

# name -> frozenset(labels).
_LAYOUT_LABEL_SETS: Dict[str, FrozenSet[str]] = {
    name: frozenset(spec["labels"]) for name, spec in LAYOUTS.items()
}


class LayoutMismatchError(RuntimeError):
    """Raised when the live tracker set does not exactly match the trained layout."""


@dataclasses.dataclass(frozen=True)
class ResolvedLayout:
    name: str
    labels: List[str]                      # ordered, defines the model's input slot order
    label_to_tracker_key: Dict[str, str]    # label -> SolarXR "device:trackerNum" key


def online_labels(samples: Dict[str, TrackerSample]) -> Dict[str, str]:
    """{mobileposer label: tracker_key} for every currently-online, mappable tracker.

    Raises LayoutMismatchError if two online trackers claim the same label
    (ambiguous binding -- this must be fixed in SlimeVR-Server, not guessed here).
    """
    result: Dict[str, str] = {}
    for sample in samples.values():
        if not sample.online:
            continue
        label = BODY_PART_TO_LABEL.get(sample.body_part)
        if label is None:
            continue  # tracker bound to a body part this checkpoint family doesn't use
        if label in result and result[label] != sample.tracker_key:
            raise LayoutMismatchError(
                f"Two online trackers are both bound to body part '{label}': "
                f"{result[label]} and {sample.tracker_key}. Fix the binding in SlimeVR-Server."
            )
        result[label] = sample.tracker_key
    return result


def detect_layout(samples: Dict[str, TrackerSample]) -> ResolvedLayout:
    """Resolve the current online tracker set to the one supported layout.

    Raises LayoutMismatchError if the online set does not match it exactly.
    """
    labels = online_labels(samples)
    current = frozenset(labels)

    matches = [name for name, label_set in _LAYOUT_LABEL_SETS.items() if label_set == current]

    if not matches:
        known = ", ".join(f"{name}={sorted(s)}" for name, s in _LAYOUT_LABEL_SETS.items())
        raise LayoutMismatchError(
            f"Online tracker body parts {sorted(current)} do not exactly match the supported "
            f"layout. Expected: {known} (bind LEFT/RIGHT_LOWER_ARM, LEFT/RIGHT_LOWER_LEG and either HIP or WAIST)."
        )
    if len(matches) > 1:
        raise LayoutMismatchError(f"Ambiguous layout match for {sorted(current)}: {matches}")

    name = matches[0]
    ordered_labels = LAYOUTS[name]["labels"]
    return ResolvedLayout(
        name=name,
        labels=ordered_labels,
        label_to_tracker_key={label: labels[label] for label in ordered_labels},
    )
