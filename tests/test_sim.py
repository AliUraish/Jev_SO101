"""Simulator tests: unit mapping, deterministic reset, camera frames, real-time physics, and the scripted task
through contact physics (both directly and through the jev_arm controller)."""
import time
import threading

import mujoco
import numpy as np
import pytest

from so101_sim import SO101Sim, SceneConfig
from so101_sim.units import JOINTS, dict_from_controller, dict_to_controller, from_controller, to_controller
from so101_sim.task import TaskParameters, evaluate, joint_limits_for_jev_arm, plan_skills, run_skills


@pytest.fixture(scope="module")
def sim():
    s = SO101Sim(SceneConfig(), realtime=False, render_hz=5)
    yield s
    s.close()


def test_unit_conversion_roundtrip_and_gripper_endpoints():
    for j in JOINTS:
        for v in (-50.0, 0.0, 37.5):
            assert to_controller(j, from_controller(j, v)) == pytest.approx(v)
    assert from_controller("gripper", 0) == pytest.approx(-0.174533)
    assert from_controller("gripper", 100) == pytest.approx(1.7453292)
    assert to_controller("shoulder_lift", np.radians(30)) == pytest.approx(30)
    d = {j: 10.0 for j in JOINTS}
    assert dict_to_controller(dict_from_controller(d)) == pytest.approx(d)


def test_reset_is_deterministic_and_seeded_jitter_reproducible(sim):
    a = sim.reset(seed=1)
    ga = sim.get_ground_truth()
    sim.step(0.5)
    b = sim.reset(seed=1)
    gb = sim.get_ground_truth()
    assert a == pytest.approx(b)
    assert ga["block"]["position"] == pytest.approx(gb["block"]["position"])
    assert ga["cup"]["position"] == pytest.approx(gb["cup"]["position"])
    jitter = SceneConfig(block_jitter=0.02)
    sim.reset(seed=7, scene_config=jitter)
    p1 = sim.get_ground_truth()["block"]["position"]
    sim.reset(seed=7)
    p2 = sim.get_ground_truth()["block"]["position"]
    sim.reset(seed=8)
    p3 = sim.get_ground_truth()["block"]["position"]
    assert p1 == pytest.approx(p2)
    assert abs(p1[0] - p3[0]) + abs(p1[1] - p3[1]) > 1e-4
    sim.reset(seed=0, scene_config=SceneConfig())


def test_camera_frames_are_rgb_with_timestamps(sim):
    sim.reset()
    sim.step(0.1)
    frames = sim.render_now()
    assert set(frames) == {"overhead", "wrist"}
    for name, f in frames.items():
        assert f.rgb.shape == (480, 640, 3) and f.rgb.dtype == np.uint8
        assert f.captured_at <= time.monotonic()
        assert f.intrinsics["fx"] > 0
    over = frames["overhead"].rgb.astype(int)
    blue = (over[..., 2] > 140) & (over[..., 0] < 90) & (over[..., 1] < 150)
    red = (over[..., 0] > 90) & (over[..., 0] > 1.6 * over[..., 1]) & (over[..., 0] > 1.6 * over[..., 2])
    # The steeper cup-inspection camera partly occludes the pickup block behind the fingers.
    assert blue.sum() > 25, "blue block must be visible in RGB order"
    assert red.sum() > 200, "red cup must be visible in RGB order"


def test_physics_advances_in_wall_clock_time_without_stepping():
    s = SO101Sim(SceneConfig(), realtime=True, render_hz=5)
    try:
        t0 = s.sim_time
        time.sleep(0.6)
        assert s.sim_time - t0 == pytest.approx(0.6, abs=0.15)
        s.pause()
        t1 = s.sim_time
        time.sleep(0.2)
        assert s.sim_time == pytest.approx(t1, abs=0.03)
    finally:
        s.close()


def test_slow_render_does_not_block_joint_reads_or_commands(sim, monkeypatch):
    entered, release, accessed = threading.Event(), threading.Event(), threading.Event()
    original = mujoco.Renderer.render
    def delayed(renderer, *args, **kwargs):
        entered.set()
        release.wait(timeout=2)
        return original(renderer, *args, **kwargs)
    monkeypatch.setattr(mujoco.Renderer, "render", delayed)
    sim._render_request.set()
    assert entered.wait(timeout=2)
    def access():
        sim.send_joint_targets(sim.read_joint_positions())
        accessed.set()
    reader = threading.Thread(target=access)
    reader.start()
    try:
        assert accessed.wait(timeout=0.2), "GPU rendering blocked controller access"
    finally:
        release.set()
        reader.join(timeout=2)
    frames = sim.render_now()
    assert frames["overhead"].captured_at == frames["wrist"].captured_at
    assert frames["overhead"].sim_time == frames["wrist"].sim_time


def test_ground_truth_hides_nothing_needed_by_evaluator(sim):
    sim.reset()
    gt = sim.get_ground_truth()
    for key in ("block_lifted", "block_held", "block_inside_cup", "block_settled", "cup_tipped", "block_on_floor",
                "arm_touching_table", "arm_touching_cup", "success"):
        assert key in gt["flags"]
    assert gt["flags"]["block_touching_table"] and not gt["flags"]["block_held"]
    assert gt["cup"]["tilt_deg"] < 1


def test_scripted_pick_and_place_succeeds_through_contact_physics(sim):
    sim.reset(seed=0)
    skills = plan_skills(sim.model, sim.config, TaskParameters())
    reports = run_skills(sim, skills)
    by_stage = {(r.stage, r.waypoint): r for r in reports}
    assert by_stage[("grasp", 1)].flags["block_held"]
    assert by_stage[("lift", 0)].flags["block_lifted"] and by_stage[("lift", 0)].flags["block_held"]
    assert by_stage[("place", 1)].flags["block_held"]
    assert not any(r.flags["arm_touching_table"] for r in reports)
    result = evaluate(sim)
    assert result["success"], result
    assert result["cup_tilt_deg"] < 5


def test_jev_arm_controller_executes_planned_waypoints_in_sim(sim):
    """The existing jev_arm Controller (real code) drives the simulator with synthetic decisions."""
    from jev_arm.controller import Controller
    from jev_arm.demo import answers_for, scene_for
    from jev_arm.models import Config, CameraConfig, Decision, JointLimit, Waypoint

    sim.reset(seed=0)
    skills = plan_skills(sim.model, sim.config, TaskParameters())
    start = dict(sim.config.start_pose)
    start["gripper"] = skills["approach"][0]["joints"]["gripper"]
    sim.send_joint_targets(start)
    sim.step(1.5)
    config = Config(
        simulation_only=True,
        cameras={"overhead": CameraConfig(source=0), "wrist": CameraConfig(source=1)},
        start_pose=start,
        joint_limits={j: JointLimit(**v) for j, v in joint_limits_for_jev_arm(sim.model).items()},
        skills={k: [Waypoint(**p) for p in v] for k, v in skills.items()},
        settle_timeout_s=4.0,
    )
    dt = 1 / config.control_hz
    now = 0.0
    controller = Controller(config, sim.read_joint_positions(), now)
    accepted = []
    for _ in range(3000):
        now += dt
        positions = sim.read_joint_positions()
        if not controller.active and not controller.done:
            stage = controller.expected_skill
            controller.accept(Decision(revision=controller.revision, captured_at=now, scene=scene_for(stage), answers=answers_for(stage)), positions, now)
            accepted.append(stage)
        target = controller.tick(positions, now)
        assert not controller.fault, controller.fault
        sim.send_joint_targets(target)
        sim.step(dt)
        if controller.done:
            break
    assert controller.done, controller.reason
    result = evaluate(sim)
    assert result["success"], result
