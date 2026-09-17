"""Joint-unit conversion between MuJoCo (radians) and the controller's units.

Controller units: arm joints in degrees; gripper on LeRobot's 0-100 opening scale (0 = closed, 100 = open).

This mapping is defined for the SIMULATED arm only. It is NOT the physical calibration of the real arm
(that lives in LeRobot's calibration JSON and stays separate). The sim zero for every arm joint is the
MJCF zero: arm fully extended horizontally along +x with the gripper pointing away from the base.

    controller_value = sign * degrees(mujoco_radians) + zero_offset_deg        (arm joints)
    gripper_percent  = 100 * (mujoco_radians - closed_rad) / (open_rad - closed_rad)

Positive directions (sim): shoulder_pan +ve = turns toward -y (clockwise seen from above);
shoulder_lift, elbow_flex, wrist_flex +ve = lowers the gripper (rotates it toward the table);
wrist_roll +ve = right-hand rotation about the gripper's long axis.
"""
from __future__ import annotations

import math

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_JOINTS = JOINTS[:-1]

# sign and zero offset (deg) per arm joint. Kept explicit so a future re-mapping is a one-line change.
ARM_CONVERSION = {
    "shoulder_pan": {"sign": 1.0, "zero_offset_deg": 0.0},
    "shoulder_lift": {"sign": 1.0, "zero_offset_deg": 0.0},
    "elbow_flex": {"sign": 1.0, "zero_offset_deg": 0.0},
    "wrist_flex": {"sign": 1.0, "zero_offset_deg": 0.0},
    "wrist_roll": {"sign": 1.0, "zero_offset_deg": 0.0},
}

# Gripper hinge (moving jaw) range in the MJCF: -10 deg (jaw pads touching) .. +100 deg (fully open).
GRIPPER_CLOSED_RAD = -0.174533
GRIPPER_OPEN_RAD = 1.7453292

# Measured fingertip-sphere gap at a few openings (mm) for reference; see README table.
GRIPPER_OPENING_TABLE_MM = {0: 4.1, 9.1: 16.3, 18.2: 30.1, 27.3: 43.7, 36.4: 57.1, 45.5: 70.0, 54.5: 82.4, 63.6: 94.2, 72.7: 105.2, 81.8: 115.5, 90.9: 124.9, 100: 133.4}


def to_controller(joint: str, radians: float) -> float:
    if joint == "gripper":
        return 100.0 * (radians - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    conv = ARM_CONVERSION[joint]
    return conv["sign"] * math.degrees(radians) + conv["zero_offset_deg"]


def from_controller(joint: str, value: float) -> float:
    if joint == "gripper":
        return GRIPPER_CLOSED_RAD + (value / 100.0) * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    conv = ARM_CONVERSION[joint]
    return math.radians((value - conv["zero_offset_deg"]) / conv["sign"])


def dict_to_controller(radians: dict[str, float]) -> dict[str, float]:
    return {j: to_controller(j, radians[j]) for j in JOINTS}


def dict_from_controller(values: dict[str, float]) -> dict[str, float]:
    return {j: from_controller(j, values[j]) for j in JOINTS}


def gripper_gap_mm(percent: float) -> float:
    """Approximate fingertip gap for a commanded opening (linear interpolation of the measured table)."""
    keys = sorted(GRIPPER_OPENING_TABLE_MM)
    percent = min(max(percent, keys[0]), keys[-1])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= percent <= hi:
            t = (percent - lo) / (hi - lo)
            return GRIPPER_OPENING_TABLE_MM[lo] + t * (GRIPPER_OPENING_TABLE_MM[hi] - GRIPPER_OPENING_TABLE_MM[lo])
    return GRIPPER_OPENING_TABLE_MM[keys[-1]]
