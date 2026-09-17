"""Scripted pick-and-place through contact physics, with optional video and jev_arm config export.

Run (real-time, ~25 s):   .venv/bin/python -m so101_sim.examples.scripted_pick_place
Fast (stepped physics):   .venv/bin/python -m so101_sim.examples.scripted_pick_place --fast
Record video:             .venv/bin/python -m so101_sim.examples.scripted_pick_place --fast --video so101_sim/media/pick_place.mp4
Export controller config: .venv/bin/python -m so101_sim.examples.scripted_pick_place --fast --export-config config.sim.json --reference captures/sim/overhead_reference.jpg
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from so101_sim import SO101Sim, SceneConfig
from so101_sim.task import TaskParameters, evaluate, export_jev_arm_config, plan_skills, run_skills


class VideoRecorder:
    """Side-by-side overhead + wrist video captured from the simulator's camera stream."""

    def __init__(self, sim: SO101Sim, path: Path, fps: float = 15.0):
        import cv2
        self.sim, self.cv2, self.path, self.fps = sim, cv2, path, fps
        path.parent.mkdir(parents=True, exist_ok=True)
        w = sim.config.overhead_camera.width + sim.config.wrist_camera.width
        h = max(sim.config.overhead_camera.height, sim.config.wrist_camera.height)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.last_sim_time = -1.0
        self.frames = 0

    def capture(self):
        """Append one frame per 1/fps of simulation time. In stepped mode frames are rendered on demand."""
        if self.sim.sim_time - self.last_sim_time < 1.0 / self.fps - 1e-6:
            return
        frames = self.sim.get_camera_frames() if self.sim.running else self.sim.render_now()
        if len(frames) < 2 or frames["overhead"].sim_time == self.last_sim_time:
            return
        self.last_sim_time = frames["overhead"].sim_time
        tile = np.hstack([frames["overhead"].rgb, frames["wrist"].rgb])
        tile = self.cv2.cvtColor(tile, self.cv2.COLOR_RGB2BGR)
        self.cv2.putText(tile, f"sim t={self.last_sim_time:6.2f}s", (10, 25), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.writer.write(tile)
        self.frames += 1

    def close(self):
        self.writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="step physics as fast as possible instead of real time")
    ap.add_argument("--video", type=Path, help="write an mp4 of both cameras")
    ap.add_argument("--export-config", type=Path, help="write a jev_arm-compatible config with the planned waypoints")
    ap.add_argument("--reference", type=Path, help="also save the initial overhead frame as the taught-layout reference")
    ap.add_argument("--distractor", action="store_true", help="add the second block seen in the reference photo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SceneConfig(distractor_block=args.distractor)
    sim = SO101Sim(cfg, realtime=not args.fast, render_hz=15)
    sim.reset(seed=args.seed)
    recorder = VideoRecorder(sim, args.video) if args.video else None
    on_tick = recorder.capture if recorder else None

    if args.reference:
        import cv2
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        frame = sim.wait_for_frames()["overhead"]
        cv2.imwrite(str(args.reference), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        print("saved overhead reference", args.reference)

    params = TaskParameters()
    skills = plan_skills(sim.model, cfg, params)
    if args.export_config:
        export_jev_arm_config(cfg, skills, sim.model, args.export_config, str(args.reference) if args.reference else None)
        print("wrote", args.export_config)

    def on_stage(r):
        f = r.flags
        print(f"[{r.sim_time:6.2f}s] {r.stage:8s} wp{r.waypoint}: held={f['block_held']} lifted={f['block_lifted']} inside={f['block_inside_cup']} "
              f"cup_tilt={r.cup_tilt_deg:4.1f} arm/table={f['arm_touching_table']} arm/cup={f['arm_touching_cup']}")

    t0 = time.monotonic()
    run_skills(sim, skills, params, on_stage, on_tick=on_tick)
    result = evaluate(sim, on_tick=on_tick)
    if recorder:
        recorder.close()
        print(f"video: {args.video} ({recorder.frames} frames)")
    print(json.dumps(result, indent=2))
    print(f"wall time {time.monotonic() - t0:.1f}s, sim stats {sim.stats}")
    sim.close()
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
