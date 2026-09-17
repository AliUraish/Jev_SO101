from __future__ import annotations

from .models import Config, JOINTS, validate_joints


class SimRobot:
    def __init__(self, start):
        self.positions = dict(start)

    def connect(self):
        pass

    def read(self):
        return dict(self.positions)

    def send(self, target):
        validate_joints(target)
        self.positions = dict(target)

    def hold(self):
        pass

    def close(self):
        pass


class LeRobotArm:
    """LeRobot adapter. All bus access belongs to the controller's main thread."""

    def __init__(self, config: Config, read_only: bool = False):
        config.validate_connection()
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            try:
                from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
            except ImportError as exc:
                raise RuntimeError("Install LeRobot with Feetech support in this environment; see README") from exc
        self.robot = SO101Follower(SO101FollowerConfig(
            id=config.robot_id,
            port=config.robot_port,
            calibration_dir=config.calibration_file.parent,
            cameras={},
            use_degrees=True,
            disable_torque_on_disconnect=False,
            max_relative_target=2.0,
        ))
        self.read_only = read_only
        self.config = config

    def connect(self):
        # Read-only connection also enables waypoint capture alongside an already taught pose.
        # Never invoke automatic recalibration or overwrite the supplied calibration.
        self.robot.bus.connect()
        try:
            if not self.robot.is_calibrated:
                raise ValueError("Motor calibration differs from supplied file; reconcile it with LeRobot first")
            if not self.read_only:
                # Refuse startup unless position mode is already configured. Preserve current torque
                # and PID configuration from LeRobot setup, and replace stale goals before enabling.
                modes = self.robot.bus.sync_read("Operating_Mode", normalize=False)
                if any(mode != 0 for mode in modes.values()):
                    raise ValueError("Configure all motors in position mode using LeRobot first")
                current = self.robot.bus.sync_read("Present_Position")
                self.config.check_pose(current)
                if any(abs(current[j] - self.config.start_pose[j]) > self.config.joint_limits[j].arrival_tolerance for j in JOINTS):
                    raise ValueError("Position the arm at the recorded start pose before enabling motion")
                if self.config.stop_file.exists():
                    raise ValueError("Local stop is active")
                self.robot.bus.sync_write("Goal_Position", current)
                self.robot.bus.enable_torque()
        except Exception:
            self.close()
            raise

    def read(self):
        values = self.robot.bus.sync_read("Present_Position")
        return validate_joints({joint: float(values[joint]) for joint in JOINTS})

    def send(self, target):
        if self.read_only:
            raise RuntimeError("Read-only robot adapter cannot send motor commands")
        self.robot.send_action({f"{joint}.pos": value for joint, value in target.items()})

    def hold(self):
        if not self.read_only and self.robot.bus.is_connected:
            # Freeze at measured position. If serial communication fails this may fail too.
            self.send(self.read())

    def close(self):
        if self.robot.bus.is_connected:
            # Keep torque enabled: automatically dropping a loaded arm is not a safe stop.
            self.robot.bus.disconnect(disable_torque=False)
