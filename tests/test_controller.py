import math

import pytest
from pydantic import ValidationError

from jev_arm.controller import Controller
from jev_arm.clients import required_scene
from jev_arm.demo import answers_for, demo_config, run_demo, scene_for
from jev_arm.models import Config, Decision, JOINTS, STAGES


def setup_controller():
    config = demo_config()
    return config, Controller(config, config.start_pose, 0.0)


def decision(stage="approach", now=0.1, revision=0, **scene_changes):
    scene = scene_for(stage).model_copy(update=scene_changes)
    return Decision(revision=revision, captured_at=now, scene=scene, answers=answers_for(stage))


def test_complete_offline_cycle():
    result = run_demo()
    assert result["done"]
    assert result["skills"] == list(STAGES) + ["verify"]


@pytest.mark.parametrize("mutate", [
    lambda d: d.model_copy(update={"captured_at": -1.0}),
    lambda d: d.model_copy(update={"captured_at": 2.0}),
    lambda d: d.model_copy(update={"revision": 20}),
])
def test_stale_future_and_wrong_revision_rejected(mutate):
    config, control = setup_controller()
    control.accept(mutate(decision()), config.start_pose, 1.0)
    assert not control.active


def test_expired_evidence_rejected():
    config, control = setup_controller()
    control.accept(decision(), config.start_pose, config.decision_max_age_s + 1)
    assert not control.active


def test_release_cannot_skip_stages():
    config, control = setup_controller()
    control.accept(decision("release"), config.start_pose, 0.1)
    assert not control.active
    assert control.expected_skill == "approach"


@pytest.mark.parametrize("changes", [
    {"glass_at_reference": "unknown"},
    {"human_in_workspace": "unknown"},
    {"block_fits_opening": "no"},
    {"uncertainties": ["Wrist view occluded"]},
])
def test_uncertain_scene_holds(changes):
    config, control = setup_controller()
    control.accept(decision(**changes), config.start_pose, 0.1)
    assert not control.active
    assert not control.done


@pytest.mark.parametrize("changes", [{"human_in_workspace": "yes"}, {"hazards": ["Obstruction"]}])
def test_visible_hazard_latches(changes):
    config, control = setup_controller()
    control.accept(decision(**changes), config.start_pose, 0.1)
    assert control.fault
    control.accept(decision(now=0.2), config.start_pose, 0.2)
    assert not control.active


def test_unsafe_probability_latches_before_motion():
    config, control = setup_controller()
    d = decision()
    d.answers.unsafe.noul = 0.5
    control.accept(d, config.start_pose, 0.1)
    assert control.fault


@pytest.mark.parametrize("field", ["confidence", "probabilities"])
def test_low_confidence_or_probability_holds(field):
    config, control = setup_controller()
    d = decision()
    if field == "confidence":
        d.answers.task.confidence = 0.2
    else:
        d.answers.task.probabilities["approach"] = 0.6
        d.answers.task.probabilities["hold"] = 0.4
    control.accept(d, config.start_pose, 0.1)
    assert not control.active


@pytest.mark.parametrize("kwargs", [{"cameras_ok": False}, {"stop_requested": True}])
def test_local_stop_overrides_active_command(kwargs):
    config, control = setup_controller()
    control.accept(decision(now=0.0), config.start_pose, 0.0)
    target = control.tick(config.start_pose, 1 / 30, **kwargs)
    assert control.fault
    assert target == config.start_pose


def test_missed_deadline_latches():
    config, control = setup_controller()
    control.accept(decision(now=0.0), config.start_pose, 0.0)
    control.tick(config.start_pose, 1.0)
    assert control.fault


def test_tracking_failure_latches():
    config, control = setup_controller()
    positions = dict(config.start_pose)
    positions["shoulder_pan"] = 20.0
    control.tick(positions, 0.03)
    assert control.fault


def test_duplicate_decision_does_not_restart_motion():
    config, control = setup_controller()
    d = decision(now=0.0)
    control.accept(d, config.start_pose, 0.0)
    control.accept(d, config.start_pose, 0.03)
    assert control.active_since == 0.0


def test_speed_bounded_at_every_tick():
    config, control = setup_controller()
    control.accept(decision(now=0.0), config.start_pose, 0.0)
    positions = dict(config.start_pose)
    now = 0.0
    for _ in range(100):
        now += 1 / 30
        target = control.tick(positions, now)
        for joint in JOINTS:
            assert abs(target[joint] - positions[joint]) <= config.joint_limits[joint].speed_per_s / 30 + 1e-8
        positions = target
    assert not control.fault


def test_verification_needs_two_distinct_frames_and_visible_success():
    config, control = setup_controller()
    control.index = len(STAGES) - 1
    d = decision("verify")
    control.accept(d, config.start_pose, 0.1)
    assert not control.done
    control.accept(d, config.start_pose, 0.1)
    assert not control.done
    control.accept(decision("verify", now=0.2, block_inside_glass="unknown"), config.start_pose, 0.2)
    assert control.confirmations == 0
    control.accept(decision("verify", now=0.3), config.start_pose, 0.3)
    control.accept(decision("verify", now=0.4), config.start_pose, 0.4)
    assert control.done


def test_release_requires_visual_alignment():
    config, control = setup_controller()
    control.index = STAGES.index("release")
    control.accept(decision("release", gripper_above_glass="unknown"), config.start_pose, 0.1)
    assert not control.active


def test_lift_does_not_require_future_container_fit_evidence():
    config, control = setup_controller()
    control.index = STAGES.index("lift")
    control.accept(decision("lift", block_fits_opening="unknown"), config.start_pose, 0.1)
    assert control.active


def test_place_still_requires_fit_evidence():
    config, control = setup_controller()
    control.index = STAGES.index("place")
    control.accept(decision("place", block_fits_opening="unknown"), config.start_pose, 0.1)
    assert not control.active


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("cutoff", [0.1, 0.2])
def test_jev_prerequisites_match_local_controller(stage, cutoff):
    config, control = setup_controller()
    config.max_unsafe_probability = cutoff
    control.index = STAGES.index(stage)
    assert control._precondition(decision(stage)) is None
    for field in required_scene(stage):
        assert control._precondition(decision(stage, **{field: "unknown"})) is not None


@pytest.mark.parametrize("changes", [
    {"hazards": ["Object in motion path"]},
    {"human_in_workspace": "yes"},
    {"glass_at_reference": "no"},
    {"block_in_gripper": "unknown"},
    {"uncertainties": ["Cannot establish clearance"]},
])
def test_experimental_sim_cutoff_cannot_override_visual_gates(changes):
    config, control = setup_controller()
    config.max_unsafe_probability = 0.2
    control.index = STAGES.index("lift")
    d = decision("lift", **changes)
    d.answers.unsafe.noul = 0.01
    control.accept(d, config.start_pose, 0.1)
    assert not control.active


def test_no_simulation_poses_on_hardware():
    with pytest.raises(ValueError, match="Simulation"):
        demo_config().validate_motion(hardware=True)


def test_no_early_gripper_opening_in_place_path():
    config = demo_config()
    config.skills["place"][0].joints["gripper"] = 80
    with pytest.raises(ValueError, match="preserve the grasp"):
        config.validate_motion()


def test_release_cannot_move_arm_joints():
    config = demo_config()
    config.skills["release"][0].joints["wrist_roll"] = 10
    with pytest.raises(ValueError, match="keep all arm joints"):
        config.validate_motion()


def test_invalid_joint_values_and_incomplete_setup():
    config = demo_config()
    config.start_pose["shoulder_pan"] = math.nan
    with pytest.raises(ValueError):
        config.validate_motion()
    with pytest.raises(ValueError, match="Record start_pose"):
        Config.model_validate_json(open("config.example.json").read()).validate_motion()


def test_duplicate_cameras_rejected():
    data = demo_config().model_dump()
    data["cameras"]["wrist"]["source"] = 0
    with pytest.raises(ValidationError, match="different camera"):
        Config.model_validate(data)
