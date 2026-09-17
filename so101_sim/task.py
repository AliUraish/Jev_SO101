"""Block-into-cup task: IK-planned skill waypoints, a scripted executor, and export to the jev_arm config schema.

The waypoints are computed from the scene geometry (block/cup positions) with the IK solver, then stored in
controller units so the same "taught path" semantics apply as on the real arm: approach -> grasp -> lift ->
place -> release -> retract. Stages follow the jev_arm controller rules (approach keeps the start opening,
grasp closes at a stationary arm pose, lift/place keep the grasp opening, release only opens the gripper at the
final place pose, retract keeps the released opening).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .ik import IKSolver
from .scene import SceneConfig
from .sim import SO101Sim
from .units import JOINTS, from_controller, to_controller

STAGES = ("approach", "grasp", "lift", "place", "release", "retract")


@dataclass(frozen=True)
class TaskParameters:
    open_percent: float = 22.0           # start / approach / release opening (~36 mm fingertip gap)
    grasp_percent: float = 8.0           # commanded closure; the jaws stall on the 25 mm block near 14-15
    approach_height: float = 0.06        # TCP height above the block center before descending (m)
    grasp_offset: float = 0.004          # TCP height above block center at grasp (m)
    lift_height: float = 0.13            # TCP z after lifting (m)
    lift_tilt_deg: float = 20.0          # approach-axis tilt from vertical, leaning radially outward
    transport_height: float = 0.125      # TCP z while moving over the cup (m); block bottom ~1.4 cm above the rim
    transport_tilt_deg: float = 20.0     # tilt while carrying over the cup (pure top-down is out of reach here)
    pre_release_height: float = 0.105    # TCP z with the block entering the opening (m)
    pre_release_tilt_deg: float = 12.0
    release_z: float = 0.085             # TCP z at release; block center ~4 mm below the TCP, ~1 cm below the rim
    release_tilt_deg: float = 10.0       # keep the gripper close to vertical inside the cup to stay off the wall
    lift_out_height: float = 0.105       # straight lift-out at the release tilt before re-orienting
    clear_height: float = 0.16           # high clearing pose between the cup and home, so the joint-space path to home stays above the rim
    clear_tilt_deg: float = 35.0
    grasp_lateral_offset: float = -0.005   # TCP shift for approach/descend along the closing axis (m). The jaw faces at a 22% opening sit at
                                            # -13.6 mm (fixed) and +23.9 mm (moving) from the TCP; moving the TCP -5 mm puts the block at +5 mm, centered
    release_lateral_offset: float = -0.005  # shift of the cup-side targets along the jaw closing axis (m); negative = toward the fixed jaw. The open gripper spans -24..+34 mm about the TCP, so -5 mm centers it in the cup
    ik_orientation_weight: float = 0.08  # IK trade-off between position accuracy and approach/closing orientation
    waypoint_duration_s: float = 2.0
    settle_s: float = 0.8


def radial_dirs(xy, tilt_deg: float):
    """Approach direction tilted `tilt_deg` from vertical toward the target's radial direction, and a
    tangential closing direction (jaws close perpendicular to the radial line)."""
    phi = float(np.arctan2(xy[1], xy[0]))
    t = np.radians(tilt_deg)
    approach = np.array([np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), -np.cos(t)])
    closing = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return approach, closing


def plan_skills(model: mujoco.MjModel, cfg: SceneConfig, params: TaskParameters | None = None) -> dict[str, list[dict]]:
    """Return {stage: [{"joints": {controller units}, "duration_s": float}, ...]} for the six motion stages."""
    p = params or TaskParameters()
    ik = IKSolver(model)
    block = np.array(cfg.block_position)
    cup = np.array(cfg.cup_position)
    start_rad = np.array([from_controller(j, cfg.start_pose[j]) for j in JOINTS[:-1]])

    def solve(pos, tilt, grip, q0, lateral=0.0):
        a, c = radial_dirs(pos[:2], tilt)
        target = np.asarray(pos, dtype=float) + lateral * c
        q, err = ik.solve(target, approach_dir=a, closing_dir=c, q_init=q0, gripper_percent=grip, orientation_weight=p.ik_orientation_weight)
        if err > 0.005:
            raise ValueError(f"IK could not reach {np.round(pos, 3)} (error {err * 1000:.1f} mm); adjust scene or task parameters")
        return q, ik.to_controller(q, grip)

    def wp(joints, duration=p.waypoint_duration_s):
        return {"joints": {j: float(joints[j]) for j in JOINTS}, "duration_s": float(duration)}

    O, G = p.open_percent, p.grasp_percent
    over = np.array([cup[0], cup[1], p.transport_height])
    q, approach = solve(block + [0, 0, p.approach_height], 0.0, O, start_rad, p.grasp_lateral_offset)
    q, descend = solve(block + [0, 0, p.grasp_offset], 0.0, O, q, p.grasp_lateral_offset)
    closure = dict(descend, gripper=G)
    q, lift = solve(block + [0, 0, p.lift_height], p.lift_tilt_deg, G, q)
    q, over_cup = solve(over, p.transport_tilt_deg, G, q)
    off = p.release_lateral_offset
    q, pre_release = solve(np.array([cup[0], cup[1], p.pre_release_height]), p.pre_release_tilt_deg, G, q, off)
    q, place = solve(np.array([cup[0], cup[1], p.release_z]), p.release_tilt_deg, G, q, off)
    release = dict(place, gripper=O)
    q, lift_out = solve(np.array([cup[0], cup[1], p.lift_out_height]), p.release_tilt_deg, O, q, off)
    q, lift_out2 = solve(over, p.transport_tilt_deg, O, q)
    q, clear = solve(np.array([0.5 * (cup[0] + 0.26), 0.5 * cup[1], p.clear_height]), p.clear_tilt_deg, O, q)
    home = {j: float(cfg.start_pose[j]) for j in JOINTS}
    home["gripper"] = O
    return {
        "approach": [wp(approach)],
        "grasp": [wp(descend), wp(closure, 1.5)],
        "lift": [wp(lift)],
        "place": [wp(over_cup, 2.5), wp(pre_release, 1.5), wp(place, 1.5)],
        "release": [wp(release, 1.5)],
        "retract": [wp(lift_out, 1.5), wp(lift_out2, 1.5), wp(clear, 1.5), wp(home, 2.5)],
    }


def joint_limits_for_jev_arm(model: mujoco.MjModel) -> dict[str, dict]:
    """Controller-unit joint limits derived from the MJCF ranges, with prototype speed/tolerance settings."""
    limits = {}
    for j in JOINTS[:-1]:
        lo, hi = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        limits[j] = {"minimum": round(to_controller(j, lo), 2), "maximum": round(to_controller(j, hi), 2),
                     "speed_per_s": 60.0, "tracking_tolerance": 8.0, "arrival_tolerance": 2.0}
    # The force-limited jaw stalls ~8 units above the 8% closure target on the block, so arrival must allow that.
    limits["gripper"] = {"minimum": 0.0, "maximum": 100.0, "speed_per_s": 60.0, "tracking_tolerance": 15.0, "arrival_tolerance": 12.0}
    return limits


def export_jev_arm_config(cfg: SceneConfig, skills: dict, model: mujoco.MjModel, path: Path,
                          overhead_reference: str | None = None) -> dict:
    """Write a config.local.json-compatible file for the jev_arm controller (simulation poses)."""
    start = {j: float(cfg.start_pose[j]) for j in JOINTS}
    start["gripper"] = skills["approach"][0]["joints"]["gripper"]
    data = {
        "simulation_only": True,
        # Measured Astra observations take 13–18 seconds. This allowance is for
        # the controlled simulator only; hardware retains the stricter default.
        "decision_max_age_s": 30.0,
        # Experimental simulation cutoff; local visual gates still apply. Not hardware calibration.
        "max_unsafe_probability": 0.2,
        "robot_id": "so101_sim",
        "robot_port": "simulation",
        "calibration_file": None,
        "overhead_reference": overhead_reference,
        "cameras": {"overhead": {"source": "sim:overhead", "width": cfg.overhead_camera.width, "height": cfg.overhead_camera.height, "fps": 15},
                    "wrist": {"source": "sim:wrist", "width": cfg.wrist_camera.width, "height": cfg.wrist_camera.height, "fps": 15}},
        "start_pose": start,
        "joint_limits": joint_limits_for_jev_arm(model),
        "skills": skills,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


@dataclass
class StageReport:
    stage: str
    waypoint: int
    sim_time: float
    ik_targets: dict
    measured: dict
    flags: dict
    cup_tilt_deg: float


def run_skills(sim: SO101Sim, skills: dict, params: TaskParameters | None = None, on_stage=None, ramp: bool = True,
               on_tick=None) -> list[StageReport]:
    """Execute the planned waypoints. With `ramp` (default) targets follow a smoothstep from the previous target
    over each waypoint's duration, like the jev_arm controller; otherwise targets are stepped and the servos
    interpolate. Uses wall-clock waits when the sim runs in real time, otherwise steps physics."""
    p = params or TaskParameters()
    reports = []
    previous = dict(sim.last_targets() or sim.read_joint_positions())
    for stage in STAGES:
        for i, point in enumerate(skills[stage]):
            goal = point["joints"]
            if ramp:
                _ramp_to(sim, previous, goal, point["duration_s"], on_tick=on_tick)
            else:
                sim.send_joint_targets(goal)
                _advance(sim, point["duration_s"], on_tick)
            _advance(sim, p.settle_s, on_tick)
            previous = dict(goal)
            gt = sim.get_ground_truth()
            report = StageReport(stage, i, gt["sim_time"], goal, sim.read_joint_positions(), gt["flags"], gt["cup"]["tilt_deg"])
            reports.append(report)
            if on_stage:
                on_stage(report)
    return reports


def _ramp_to(sim: SO101Sim, start: dict, goal: dict, duration_s: float, rate_hz: float = 30.0, on_tick=None):
    steps = max(1, int(round(duration_s * rate_hz)))
    for k in range(1, steps + 1):
        t = k / steps
        blend = t * t * (3.0 - 2.0 * t)
        sim.send_joint_targets({j: start[j] + blend * (goal[j] - start[j]) for j in JOINTS})
        _advance(sim, duration_s / steps, on_tick)


def _advance(sim: SO101Sim, seconds: float, on_tick=None):
    """Advance `seconds` of simulation time in slices of at most 1/30 s, calling on_tick after each slice."""
    remaining = seconds
    while remaining > 1e-9:
        slice_s = min(remaining, 1.0 / 30.0)
        if sim.running:
            end = sim.sim_time + slice_s
            while sim.sim_time < end:
                time.sleep(0.005)
        else:
            sim.step(slice_s)
        remaining -= slice_s
        if on_tick:
            on_tick()


def evaluate(sim: SO101Sim, settle_wait_s: float = 2.0, on_tick=None) -> dict:
    """Final judgement: block settled inside the upright cup and not held. Waits for motion to stop first."""
    _advance(sim, settle_wait_s, on_tick)
    gt = sim.get_ground_truth()
    f = gt["flags"]
    return {
        "success": f["success"],
        "block_inside_cup": f["block_inside_cup"],
        "block_settled": f["block_settled"],
        "block_held": f["block_held"],
        "cup_tipped": f["cup_tipped"],
        "block_on_floor": f["block_on_floor"],
        "cup_tilt_deg": gt["cup"]["tilt_deg"],
        "block_position": gt["block"]["position"],
        "sim_time": gt["sim_time"],
    }


def task_parameters_dict(p: TaskParameters) -> dict:
    return asdict(p)
