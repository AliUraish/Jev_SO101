"""Synthetic decision fixtures exercise orchestration; they do not model robot physics."""
from __future__ import annotations

from .controller import Controller
from .models import Answers, CameraConfig, Config, Decision, JOINTS, JointLimit, Scene, STAGES, Waypoint
from .robot import SimRobot


def demo_config() -> Config:
    start = {j: 0.0 for j in JOINTS}
    start["gripper"] = 60.0
    poses = {}
    pose = dict(start)
    for i, skill in enumerate(STAGES[:-1]):
        if skill not in ("grasp", "release"):
            pose["shoulder_pan"] = float(i + 1)
        pose["gripper"] = 20.0 if skill in ("grasp", "lift", "place") else 60.0
        poses[skill] = [Waypoint(joints=dict(pose), duration_s=0.2)]
    return Config(
        simulation_only=True,
        cameras={"overhead": CameraConfig(source=0), "wrist": CameraConfig(source=1)},
        start_pose=start,
        joint_limits={j: JointLimit(minimum=-180 if j != "gripper" else 0, maximum=180 if j != "gripper" else 100, speed_per_s=90, tracking_tolerance=10, arrival_tolerance=0.5) for j in JOINTS},
        skills=poses,
    )


def scene_for(stage: str) -> Scene:
    return Scene(
        objects=[], glass_upright="yes", glass_opening_clear="yes", block_fits_opening="yes",
        glass_at_reference="yes", block_at_reference="yes" if stage in ("approach", "grasp") else "no",
        block_in_gripper="yes" if stage in ("lift", "place", "release") else "no",
        block_inside_glass="yes" if stage in ("retract", "verify") else "no",
        gripper_above_glass="yes" if stage == "release" else "no",
        human_in_workspace="no", hazards=[], uncertainties=[],
    )


def answers_for(stage: str) -> Answers:
    choices = set(STAGES) | {"observe", "hold"}
    return Answers.model_validate({
        "task": {"type": "choice", "choice": stage, "confidence": 0.99, "probabilities": {s: 1.0 if s == stage else 0.0 for s in choices}},
        "unsafe": {"type": "noul", "noul": 0.01},
        "done": {"type": "noul", "noul": 0.99 if stage == "verify" else 0.01},
    })


def run_demo() -> dict:
    config = demo_config()
    robot = SimRobot(config.start_pose)
    now = 0.0
    controller = Controller(config, robot.read(), now)
    accepted = []
    for _ in range(3000):
        now += 1 / config.control_hz
        if not controller.active and not controller.done:
            stage = controller.expected_skill
            controller.accept(Decision(revision=controller.revision, captured_at=now, scene=scene_for(stage), answers=answers_for(stage)), robot.read(), now)
            accepted.append(stage)
        robot.send(controller.tick(robot.read(), now))
        if controller.fault:
            raise RuntimeError(controller.fault)
        if controller.done:
            return {"mode": "offline synthetic demo; no cameras, APIs, or motors", "skills": accepted, "done": True, "simulated_seconds": round(now, 2)}
    raise RuntimeError("Demo failed to finish")
