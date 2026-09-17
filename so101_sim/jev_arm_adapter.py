"""Adapters that present the simulator through the same call shapes as jev_arm's robot and camera classes.

    SimArm     ~ jev_arm.robot.LeRobotArm   (connect / read / send / hold / close)
    SimCameras ~ jev_arm.cameras.Cameras    (start / snapshot / close / errors) -> JPEG bytes + monotonic timestamps

See examples/run_jev_arm_in_sim.py for wiring these into runtime.run_simulation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .sim import SO101Sim
from .units import JOINTS

try:  # use jev_arm's Frame type when available so runtime code sees the exact same objects
    from jev_arm.cameras import Frame as JpegFrame
except Exception:  # pragma: no cover
    @dataclass(frozen=True)
    class JpegFrame:  # type: ignore[no-redef]
        jpeg: bytes
        captured_at: float


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    import cv2
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


class SimArm:
    """Position interface in controller units. `read_only` mirrors LeRobotArm's record-pose mode."""

    def __init__(self, sim: SO101Sim, read_only: bool = False):
        self.sim = sim
        self.read_only = read_only

    def connect(self):
        return None

    def read(self) -> dict[str, float]:
        return self.sim.read_joint_positions()

    def send(self, target: dict[str, float]):
        if self.read_only:
            raise RuntimeError("Read-only adapter cannot send motor commands")
        self.sim.send_joint_targets({j: float(target[j]) for j in JOINTS})

    def hold(self):
        if not self.read_only:
            self.sim.send_joint_targets(self.read())

    def close(self):
        return None


class SimCameras:
    """Latest-frame-per-camera snapshot with the same freshness checks as jev_arm.cameras.Cameras."""

    def __init__(self, sim: SO101Sim, max_age_s: float = 0.75, max_skew_s: float = 0.25):
        self.sim = sim
        self.max_age_s = max_age_s
        self.max_skew_s = max_skew_s
        self.errors: dict[str, str] = {}

    def start(self):
        self.sim.wait_for_frames()

    def snapshot(self):
        try:
            frames = self.sim.get_camera_frames()
        except RuntimeError as exc:
            self.errors["render"] = type(exc).__name__
            return None
        if set(frames) != {"overhead", "wrist"}:
            return None
        times = [f.captured_at for f in frames.values()]
        now = time.monotonic()
        if now - min(times) > self.max_age_s or max(times) - min(times) > self.max_skew_s:
            return None
        return {name: JpegFrame(encode_jpeg(f.rgb), f.captured_at) for name, f in frames.items()}

    def close(self):
        return None
