import sys
from types import ModuleType, SimpleNamespace

import pytest

from jev_arm.demo import demo_config
from jev_arm.robot import LeRobotArm


def adapter(monkeypatch, tmp_path, read_only=False, calibrated=True, mode=0):
    config = demo_config()
    config.robot_id = "test-arm"
    config.robot_port = "/test/serial"
    config.calibration_file = tmp_path / "test-arm.json"
    config.calibration_file.write_text("{}")
    config.stop_file = tmp_path / "STOP"
    events = []

    class Bus:
        is_connected = False

        def connect(self):
            events.append("connect")
            self.is_connected = True

        def sync_read(self, name, **kwargs):
            events.append(("read", name))
            if name == "Operating_Mode":
                return {j: mode for j in config.start_pose}
            return dict(config.start_pose)

        def sync_write(self, name, values):
            events.append(("write", name, values))

        def enable_torque(self):
            events.append("enable_torque")

        def disconnect(self, disable_torque=True):
            events.append(("disconnect", disable_torque))
            self.is_connected = False

    bus = Bus()
    module = ModuleType("lerobot.robots.so_follower")
    module.SO101FollowerConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    module.SO101Follower = lambda cfg: SimpleNamespace(bus=bus, is_calibrated=calibrated)
    for name in ("lerobot", "lerobot.robots"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", module)
    return LeRobotArm(config, read_only=read_only), events, config


def test_read_only_capture_never_writes_or_enables_torque(monkeypatch, tmp_path):
    robot, events, config = adapter(monkeypatch, tmp_path, read_only=True)
    robot.connect()
    assert robot.read() == config.start_pose
    robot.close()
    assert "enable_torque" not in events
    assert not any(isinstance(e, tuple) and e[0] == "write" for e in events)
    assert events[-1] == ("disconnect", False)


def test_startup_seeds_current_position_before_enabling_torque(monkeypatch, tmp_path):
    robot, events, _ = adapter(monkeypatch, tmp_path)
    robot.connect()
    assert events[-1] == "enable_torque"
    assert events[-2][0:2] == ("write", "Goal_Position")
    robot.close()


@pytest.mark.parametrize("kwargs", [{"calibrated": False}, {"mode": 1}])
def test_calibration_or_mode_mismatch_never_enables_torque(monkeypatch, tmp_path, kwargs):
    robot, events, _ = adapter(monkeypatch, tmp_path, **kwargs)
    with pytest.raises(ValueError):
        robot.connect()
    assert "enable_torque" not in events
    assert events[-1] == ("disconnect", False)


def test_startup_respects_local_stop_before_writing(monkeypatch, tmp_path):
    robot, events, config = adapter(monkeypatch, tmp_path)
    config.stop_file.touch()
    with pytest.raises(ValueError, match="stop"):
        robot.connect()
    assert "enable_torque" not in events
    assert not any(isinstance(e, tuple) and e[0] == "write" for e in events)
