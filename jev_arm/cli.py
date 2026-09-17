from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .cameras import Cameras
from .clients import AstraVision, JevJudge, require_keys
from .demo import run_demo
from .models import STAGES, load_config
from .robot import LeRobotArm
from .runtime import run_live, wait_for_cameras


def main():
    parser = argparse.ArgumentParser(description="Astra + Jev: put a block in a glass")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run offline synthetic end-to-end exercise; no API keys or robot")
    for name in ("check", "capture", "observe", "record-pose", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=Path("config.local.json"))
        if name == "capture":
            sub.add_argument("--output-dir", type=Path, default=Path("captures"))
        if name == "observe":
            sub.add_argument("--overhead", type=Path)
            sub.add_argument("--wrist", type=Path)
            sub.add_argument("--stage", choices=STAGES, default="approach")
            sub.add_argument("--output", type=Path, default=Path("logs/observation.json"))
        if name == "record-pose":
            sub.add_argument("--target", choices=("start",) + STAGES[:-1], required=True)
            sub.add_argument("--duration", type=float, default=2.0)
            sub.add_argument("--gripper-only", action="store_true", help="For grasp closure: retain the prior taught arm pose")
        if name == "run":
            sub.add_argument("--enable-motion", action="store_true", help="Enable actual robot commands")
            sub.add_argument("--log", type=Path, default=Path("logs/run.jsonl"))
    args = parser.parse_args()
    try:
        if args.command == "demo":
            print(json.dumps(run_demo(), indent=2))
            return
        config = load_config(args.config)
        load_dotenv(args.config.resolve().parent / ".env", override=False)
        if args.command == "check":
            config.validate_motion(hardware=True)
            require_keys()
            print("Configuration and key presence validated; no API calls or hardware connections made.")
        elif args.command == "capture":
            cameras = Cameras(config)
            try:
                cameras.start()
                frames = wait_for_cameras(cameras)
                args.output_dir.mkdir(parents=True, exist_ok=True)
                for name, frame in frames.items():
                    path = args.output_dir / f"{name}.jpg"
                    path.write_bytes(frame.jpeg)
                    print(path.resolve())
            finally:
                cameras.close()
        elif args.command == "observe":
            require_keys()
            if bool(args.overhead) != bool(args.wrist):
                raise ValueError("Supply both --overhead and --wrist, or neither to use live cameras")
            if args.overhead:
                import cv2
                frames = {}
                for name in ("overhead", "wrist"):
                    image = cv2.imread(str(getattr(args, name)))
                    if image is None:
                        raise ValueError(f"Cannot decode {name} image")
                    ok, encoded = cv2.imencode(".jpg", image)
                    if not ok:
                        raise ValueError(f"Cannot encode {name} image")
                    frames[name] = encoded.tobytes()
            else:
                cameras = Cameras(config)
                try:
                    cameras.start()
                    frames = {name: f.jpeg for name, f in wait_for_cameras(cameras).items()}
                finally:
                    cameras.close()
            state = {"expected_skill": args.stage, "joints": {}, "mode": "observation only; joint state unavailable"}
            vision = AstraVision(config)
            judge = JevJudge(config)
            try:
                scene = vision.describe(frames, state)
                answers = judge.decide(scene, state)
            finally:
                vision.close()
                judge.close()
            result = {"scene": scene.model_dump(), "answers": answers.model_dump(), "motion_enabled": False}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        elif args.command == "record-pose":
            if args.duration <= 0:
                raise ValueError("Waypoint duration must be positive")
            if args.gripper_only and args.target != "grasp":
                raise ValueError("--gripper-only is for the grasp closure waypoint")
            data = json.loads(args.config.read_text())
            # Derive nonmoving joints from the prior taught pose so encoder noise cannot
            # accidentally open the gripper or move the arm during release.
            previous = None
            if args.target != "start":
                skill_index = STAGES.index(args.target)
                existing = data.get("skills", {}).get(args.target, [])
                if existing:
                    previous = existing[-1]["joints"]
                elif skill_index == 0:
                    previous = data.get("start_pose")
                else:
                    points = data.get("skills", {}).get(STAGES[skill_index - 1], [])
                    previous = points[-1]["joints"] if points else None
                if previous is None:
                    raise ValueError("Record poses in order: start, approach, grasp, lift, place, release, retract")
            # Capture an already positioned arm; this command does not enable torque or send goals.
            robot = LeRobotArm(config, read_only=True)
            try:
                robot.connect()
                pose = robot.read()
            finally:
                robot.close()
            if args.target == "start":
                data["start_pose"] = pose
            else:
                if args.gripper_only or args.target == "release":
                    pose = {**previous, "gripper": pose["gripper"]}
                else:
                    pose["gripper"] = previous["gripper"]
                data.setdefault("skills", {}).setdefault(args.target, []).append({"joints": pose, "duration_s": args.duration})
            from .models import Config
            Config.model_validate(data)
            temporary = args.config.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n")
            temporary.replace(args.config)
            print(f"Recorded {args.target} pose in {args.config.resolve()}")
        elif args.command == "run":
            if not args.enable_motion:
                raise ValueError("run requires --enable-motion; use observe for camera/API checks without motion")
            require_keys()
            run_live(config, args.log)
    except KeyboardInterrupt:
        print("Interrupted. Hardware hold was attempted before disconnect.")
        raise SystemExit(130)
    except Exception as exc:
        # Provider exceptions can contain sensitive bodies; keep CLI failures concise.
        if isinstance(exc, (ValueError, RuntimeError)):
            print(f"Error: {exc}")
        else:
            print(f"Error: {type(exc).__name__}. Check configuration, credentials, and connectivity.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
