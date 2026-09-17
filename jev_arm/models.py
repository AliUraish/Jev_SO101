from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STAGES = ("approach", "grasp", "lift", "place", "release", "retract", "verify")
PRIMITIVES = {
    "approach": "Move the open gripper to a taught pose ABOVE the block, without contact. Alignment is the result of this motion, not a required starting condition.",
    "grasp": "From the completed above-block approach pose, DESCEND to the taught pickup pose, then close the fingers at that stationary pose. The block need not be between or touching the fingertips BEFORE this descent-and-close primitive.",
    "lift": "Raise the visibly held block away from the table along the taught path. This primitive does not travel to or enter the cup.",
    "place": "Carry the held block along the taught clearance path to the cup, then lower it slightly through the opening. Keep the gripper closed. Alignment with the cup is the result of this primitive, not its starting condition.",
    "release": "Open only the gripper while keeping arm joints fixed at the taught aligned release pose. The block may already be slightly inside the rim; require clear alignment and no unintended obstruction.",
    "retract": "After opening, withdraw the gripper up from the cup and return along the taught clearance path. This clears the view for verification; visible success inside the cup is not required before withdrawal.",
    "verify": "Observe without motion whether the block is visibly settled inside the upright receiving vessel and no longer in the gripper.",
}
Skill = Literal["observe", "approach", "grasp", "lift", "place", "release", "retract", "verify", "hold"]
Truth = Literal["yes", "no", "unknown"]
Probability = Annotated[float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Region(Model):
    view: Literal["overhead", "wrist"]
    # Normalized image coordinates, never robot coordinates.
    x_min: Probability
    y_min: Probability
    x_max: Probability
    y_max: Probability

    @model_validator(mode="after")
    def ordered(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Image region must have positive area")
        return self


class SceneObject(Model):
    object_id: str
    kind: Literal["block", "glass", "gripper", "other"]
    description: str
    regions: list[Region]


class Scene(Model):
    objects: list[SceneObject]
    glass_upright: Truth
    glass_opening_clear: Truth
    block_fits_opening: Truth
    glass_at_reference: Truth
    block_at_reference: Truth
    block_in_gripper: Truth
    block_inside_glass: Truth
    gripper_above_glass: Truth
    human_in_workspace: Truth
    hazards: list[str]
    uncertainties: list[str]


class ChoiceAnswer(Model):
    type: Literal["choice"]
    choice: Skill
    probabilities: dict[str, Probability]
    confidence: Probability

    @model_validator(mode="after")
    def distribution(self):
        allowed = set(STAGES) | {"observe", "hold"}
        supplied = set(self.probabilities)
        if not supplied <= allowed or not {"observe", "hold", self.choice} <= supplied or not supplied & set(STAGES):
            raise ValueError("Invalid skill probability keys")
        if abs(sum(self.probabilities.values()) - 1) > 0.03:
            raise ValueError("Jev probabilities do not sum to one")
        if self.probabilities[self.choice] + 1e-6 < max(self.probabilities.values()):
            raise ValueError("Selected choice is not a highest-probability option")
        return self


class NoulAnswer(Model):
    type: Literal["noul"]
    noul: Probability


class Answers(Model):
    task: ChoiceAnswer
    unsafe: NoulAnswer
    done: NoulAnswer


class Decision(Model):
    # Metadata is attached locally, never supplied by a model.
    revision: int
    captured_at: float
    scene: Scene
    answers: Answers


def validate_joints(joints: dict[str, float]) -> dict[str, float]:
    if set(joints) != set(JOINTS):
        raise ValueError(f"Expected all six joint names: {JOINTS}")
    if any(isinstance(v, bool) or not math.isfinite(v) for v in joints.values()):
        raise ValueError("Joint positions must be finite numbers")
    return joints


class Waypoint(Model):
    joints: dict[str, float]
    duration_s: Positive = 2.0

    @model_validator(mode="after")
    def complete_pose(self):
        validate_joints(self.joints)
        return self


class JointLimit(Model):
    minimum: float
    maximum: float
    speed_per_s: Positive
    tracking_tolerance: Positive
    arrival_tolerance: Positive

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum >= self.maximum:
            raise ValueError("Joint limit minimum must be below maximum")
        if self.arrival_tolerance > self.tracking_tolerance:
            raise ValueError("Arrival tolerance must be no larger than tracking tolerance")
        return self


class CameraConfig(Model):
    source: int | str
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    fps: int = Field(default=30, gt=0)


class Config(Model):
    simulation_only: bool = False
    robot_id: str = ""
    robot_port: str = ""
    calibration_file: Path | None = None
    overhead_reference: Path | None = None
    cameras: dict[str, CameraConfig]
    start_pose: dict[str, float] | None = None
    joint_limits: dict[str, JointLimit] = Field(default_factory=dict)
    skills: dict[str, list[Waypoint]] = Field(default_factory=dict)
    astra_model: str = "gpt-6-astra"
    jev_model: str = "jev-latest"
    control_hz: Positive = 30.0
    api_timeout_s: Positive = 20.0
    decision_max_age_s: Positive = 8.0
    camera_max_age_s: Positive = 0.75
    camera_max_skew_s: Positive = 0.25
    decision_interval_s: Positive = 1.0
    min_confidence: Probability = 0.7
    min_task_probability: Probability = 0.8
    max_unsafe_probability: Probability = 0.1
    min_done_probability: Probability = 0.9
    command_timeout_s: Positive = 20.0
    settle_timeout_s: Positive = 3.0
    max_tick_gap_s: Positive = 0.25
    max_session_s: Positive = 180.0
    stop_file: Path = Path("STOP")

    @model_validator(mode="after")
    def camera_pair(self):
        if set(self.cameras) != {"overhead", "wrist"}:
            raise ValueError("Configure overhead and wrist cameras")
        if self.cameras["overhead"].source == self.cameras["wrist"].source:
            raise ValueError("Overhead and wrist must use different camera sources")
        return self

    def validate_motion(self, hardware: bool = False):
        if hardware and self.simulation_only:
            raise ValueError("Simulation poses cannot be used on hardware")
        if self.start_pose is None or set(self.joint_limits) != set(JOINTS):
            raise ValueError("Record start_pose and configure all six joint_limits first")
        self.check_pose(self.start_pose)
        if set(self.skills) != set(STAGES[:-1]) or any(not p for p in self.skills.values()):
            raise ValueError("Record waypoints for approach, grasp, lift, place, release, retract")
        for path in self.skills.values():
            for point in path:
                self.check_pose(point.joints)
        open_gripper = self.start_pose["gripper"]
        if any(abs(p.joints["gripper"] - open_gripper) > 1e-6 for p in self.skills["approach"]):
            raise ValueError("Approach must preserve the start gripper opening")
        previous = self.skills["approach"][-1].joints
        for point in self.skills["grasp"]:
            if point.joints["gripper"] > previous["gripper"] + 1e-6:
                raise ValueError("Grasp may close, but never open, the gripper")
            if point.joints["gripper"] < previous["gripper"] - 1e-6:
                if any(abs(point.joints[j] - previous[j]) > 1e-6 for j in JOINTS[:-1]):
                    raise ValueError("Record a separate stationary grasp closure after descent")
            previous = point.joints
        grip = previous["gripper"]
        if grip >= open_gripper:
            raise ValueError("Grasp must close the gripper")
        for skill in ("lift", "place"):
            if any(abs(p.joints["gripper"] - grip) > 1e-6 for p in self.skills[skill]):
                raise ValueError("Lift and place must preserve the grasp opening")
        # Release opens only the gripper, at the taught place pose above the glass.
        previous = self.skills["place"][-1].joints
        for point in self.skills["release"]:
            if any(abs(point.joints[j] - previous[j]) > 1e-6 for j in JOINTS[:-1]):
                raise ValueError("Release waypoints must keep all arm joints at the place pose")
            if point.joints["gripper"] < previous["gripper"]:
                raise ValueError("Release may only open the gripper")
            previous = point.joints
        if self.skills["release"][-1].joints["gripper"] <= grip:
            raise ValueError("Release must open the gripper (larger 0–100 position)")
        released = self.skills["release"][-1].joints["gripper"]
        if any(abs(p.joints["gripper"] - released) > 1e-6 for p in self.skills["retract"]):
            raise ValueError("Retract must preserve the released gripper opening")
        if hardware:
            self.validate_connection()
            if not self.overhead_reference or not self.overhead_reference.is_file():
                raise ValueError("Capture an overhead reference image for the taught layout")

    def validate_connection(self):
        if not self.robot_id or not self.robot_port or not self.calibration_file:
            raise ValueError("Set robot_id, robot_port, and calibration_file")
        if not self.calibration_file.is_file():
            raise ValueError(f"Calibration file not found: {self.calibration_file}")
        if self.calibration_file.stem != self.robot_id:
            raise ValueError("Calibration filename must be <robot_id>.json")

    def check_pose(self, pose: dict[str, float]):
        validate_joints(pose)
        for joint, value in pose.items():
            limit = self.joint_limits[joint]
            if not limit.minimum <= value <= limit.maximum:
                raise ValueError(f"{joint} is outside configured limits")


def load_config(path: Path) -> Config:
    config = Config.model_validate_json(path.read_text())
    for name in ("calibration_file", "overhead_reference", "stop_file"):
        value = getattr(config, name)
        if value is not None:
            value = value.expanduser()
            setattr(config, name, value if value.is_absolute() else path.resolve().parent / value)
    return config
