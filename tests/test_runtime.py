import threading
import time

import pytest

from jev_arm.cameras import Cameras, Frame
from jev_arm.controller import Controller
from jev_arm.demo import answers_for, demo_config, scene_for
from jev_arm.runtime import DecisionWorker


def test_missing_stale_and_unsynchronized_cameras_rejected(monkeypatch):
    monkeypatch.setattr("jev_arm.cameras.time.monotonic", lambda: 10.0)
    cameras = Cameras(demo_config())
    cameras.frames = {"overhead": Frame(b"x", 10.0)}
    assert cameras.snapshot() is None
    cameras.frames["wrist"] = Frame(b"x", 9.0)
    assert cameras.snapshot() is None
    cameras.frames["wrist"] = Frame(b"x", 9.5)
    assert cameras.snapshot() is None
    cameras.frames["wrist"] = Frame(b"x", 9.9)
    assert cameras.snapshot() is not None


def test_blocked_model_does_not_block_controller_and_late_evidence_is_rejected():
    config = demo_config()
    config.decision_interval_s = 0.01
    config.decision_max_age_s = 0.02
    started = threading.Event()
    unblock = threading.Event()

    class Camera:
        def snapshot(self):
            return {name: Frame(b"x", time.monotonic()) for name in ("overhead", "wrist")}

    class Vision:
        def describe(self, frames, state):
            started.set()
            assert unblock.wait(2)
            return scene_for("approach")

        def close(self):
            pass

    class Judge:
        called = False

        def decide(self, scene, state):
            self.called = True
            return answers_for("approach")

        def close(self):
            pass

    judge = Judge()
    controller = Controller(config, config.start_pose, time.monotonic())
    worker = DecisionWorker(config, Camera(), Vision(), judge)
    worker.publish(controller.state(config.start_pose))
    worker.thread.start()
    try:
        assert started.wait(1)
        # The API is intentionally blocked while the main-thread controller remains usable.
        for _ in range(3):
            assert controller.tick(config.start_pose, time.monotonic()) == config.start_pose
            time.sleep(0.01)
        unblock.set()
        result = worker.results.get(timeout=1)
        assert result["event"] == "stale_vision"
        assert not judge.called
        assert not controller.active
    finally:
        unblock.set()
        worker.close()


@pytest.mark.parametrize("backend,expire_session", [
    ("hardware", False), ("hardware", True), ("simulation", False),
    ("simulation", True), ("simulation", "viewer_stop"),
])
def test_full_runtime_with_fake_hardware_and_models(monkeypatch, tmp_path, expire_session, backend):
    from jev_arm import runtime
    from jev_arm.robot import SimRobot
    import json

    config = demo_config()
    config.simulation_only = backend == "simulation"
    config.robot_id = "fake-arm"
    config.robot_port = "/fake/serial"
    config.calibration_file = tmp_path / "fake-arm.json"
    config.calibration_file.write_text("{}")
    config.overhead_reference = tmp_path / "reference.jpg"
    config.overhead_reference.write_bytes(b"fake reference; mocked vision only")
    config.stop_file = tmp_path / "STOP"
    config.decision_interval_s = 0.01
    if expire_session is True:
        config.max_session_s = 0.02
    events = []

    class Camera:
        errors = {}

        def __init__(self, config):
            pass

        def start(self):
            pass

        def snapshot(self):
            return {name: Frame(b"x", time.monotonic()) for name in ("overhead", "wrist")}

        def close(self):
            events.append("cameras_closed")

    class Robot(SimRobot):
        def __init__(self, config):
            super().__init__(config.start_pose)

        def hold(self):
            events.append("hold")

        def close(self):
            events.append("robot_closed")

    class Vision:
        def __init__(self, config):
            pass

        def describe(self, frames, state):
            return scene_for(state["expected_skill"])

        def close(self):
            pass

    class Judge:
        def __init__(self, config):
            pass

        def decide(self, scene, state):
            return answers_for(state["expected_skill"])

        def close(self):
            pass

    monkeypatch.setattr(runtime, "Cameras", Camera)
    monkeypatch.setattr(runtime, "LeRobotArm", Robot)
    monkeypatch.setattr(runtime, "AstraVision", Vision)
    monkeypatch.setattr(runtime, "JevJudge", Judge)
    log = tmp_path / "run.jsonl"
    import queue
    updates = queue.SimpleQueue()
    stop = threading.Event()
    if expire_session == "viewer_stop":
        stop.set()
    def run():
        if backend == "simulation":
            runtime.run_simulation(config, log, Robot(config), Camera(config), status_queue=updates, stop_event=stop)
        else:
            runtime.run_live(config, log)
    if expire_session == "viewer_stop":
        with pytest.raises(RuntimeError, match="Local stop requested"):
            run()
    elif expire_session:
        with pytest.raises(RuntimeError, match="Session timed out after 0.02s"):
            run()
    else:
        run()
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    if expire_session == "viewer_stop":
        assert any(entry.get("reason") == "Local stop requested" for entry in entries)
        assert not any(entry.get("reason", "").startswith("Executing") for entry in entries)
    elif expire_session:
        assert any(entry.get("reason", "").startswith("Session timed out") for entry in entries)
        assert not any(entry.get("reason") == "Local stop requested" for entry in entries)
    else:
        assert any(entry.get("reason") == "Goal visually confirmed" for entry in entries)
    assert events.index("hold") < events.index("robot_closed") < events.index("cameras_closed")
    if backend == "simulation":
        assert not updates.empty()


def test_simulation_runner_rejects_hardware_config(tmp_path):
    from jev_arm.runtime import run_simulation
    config = demo_config()
    config.simulation_only = False
    with pytest.raises(ValueError, match="simulation_only=true"):
        run_simulation(config, tmp_path / "run.jsonl", None, None)
