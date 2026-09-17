"""Minimal interface example: reset the scene, read both images, move one joint, read ground truth.

Run:  .venv/bin/python -m so101_sim.examples.minimal
"""
import time
from pathlib import Path

import numpy as np

from so101_sim import SO101Sim, SceneConfig


def main():
    sim = SO101Sim(SceneConfig(), realtime=True)          # physics runs in wall-clock time from here on
    joints = sim.reset(seed=0)
    print("start joints (deg, gripper 0-100):", {k: round(v, 1) for k, v in joints.items()})

    frames = sim.get_camera_frames()
    for name, frame in frames.items():
        print(f"{name}: {frame.rgb.shape} RGB uint8, sim_time={frame.sim_time:.3f}s, captured_at={frame.captured_at:.3f} (monotonic)")
        print(f"        intrinsics: {frame.intrinsics}")
    try:
        import cv2
        out = Path("captures/sim_minimal")
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        print("saved", sorted(p.name for p in out.iterdir()))
    except ImportError:
        pass

    # Move one joint: pan 30 degrees to the left over ~1.5 s while physics keeps running.
    target = dict(joints)
    target["shoulder_pan"] = -30.0
    sim.send_joint_targets(target)
    for _ in range(6):
        time.sleep(0.25)
        print(f"t={sim.sim_time:.2f}s pan={sim.read_joint_positions()['shoulder_pan']:.1f} deg")

    gt = sim.get_ground_truth()
    print("ground truth flags:", gt["flags"])
    print("block position (m):", np.round(gt["block"]["position"], 4).tolist(), "cup tilt (deg):", round(gt["cup"]["tilt_deg"], 2))
    sim.close()


if __name__ == "__main__":
    main()
