"""Tabletop scene description and MJCF generation for the SO-101 block-into-cup task.

All geometry is in meters, world frame: origin at the robot base mount, +z up, +x pointing away
from the robot toward the operator side (the overhead camera looks back toward the arm), table top at z = 0.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
ROBOT_XML = ASSETS / "so101" / "so101.xml"
ROBOT_MESHDIR = ASSETS / "so101" / "assets"
GENERATED = ASSETS / "generated"

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class CameraSpec:
    """A fixed pinhole camera. MuJoCo cameras look along their -z axis; `lookat` defines it here."""

    width: int = 640
    height: int = 480
    fovy_deg: float = 50.0            # vertical field of view
    # Steeper view exposes the cup interior after release for visual verification.
    position: Vec3 = (0.35, 0.0, 0.65)
    lookat: Vec3 = (0.18, 0.0, 0.0)

    @property
    def intrinsics(self) -> dict:
        fy = 0.5 * self.height / math.tan(math.radians(self.fovy_deg) / 2)
        return {"fx": fy, "fy": fy, "cx": self.width / 2, "cy": self.height / 2, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class WristCameraSpec:
    """Wrist camera is defined inside the robot XML (body `camera_mount`, child of `gripper`):
    pos (0, 0.055, -0.045) m, euler (-0.57, 0, 0) rad relative to the gripper body, fovy 60 deg."""

    width: int = 640
    height: int = 480
    fovy_deg: float = 60.0

    @property
    def intrinsics(self) -> dict:
        fy = 0.5 * self.height / math.tan(math.radians(self.fovy_deg) / 2)
        return {"fx": fy, "fy": fy, "cx": self.width / 2, "cy": self.height / 2, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class SceneConfig:
    # Table: top surface at z = 0. Center and half-extents (m).
    table_center: Vec3 = (0.25, 0.0, -0.02)
    table_half_extents: Vec3 = (0.40, 0.35, 0.02)
    floor_z: float = -0.75

    # Task block: cube edge (m), start position of its center (m), mass (kg).
    block_size: float = 0.025
    block_position: Vec3 = (0.22, 0.06, 0.0125)
    block_mass: float = 0.015
    block_rgba: tuple[float, float, float, float] = (0.16, 0.45, 0.85, 1.0)
    # Optional second, identical distractor block (as in the reference photo). Not part of the task.
    distractor_block: bool = False
    distractor_position: Vec3 = (0.30, 0.11, 0.0125)

    # Cup ("glass"): open-topped tapered cup standing on the table. Base center position (m).
    cup_position: Vec3 = (0.22, -0.09, 0.0)
    cup_height: float = 0.095             # 9 oz party-cup proportions (95 mm tall, 80 mm rim, 56 mm base), as in the photo
    cup_radius_top: float = 0.040       # outer radius at the rim
    cup_radius_bottom: float = 0.028    # outer radius at the base
    cup_wall: float = 0.0015            # wall thickness used for the visual meshes
    cup_collision_wall: float = 0.003   # collision wall thickness (thicker than visual to avoid tunneling)
    cup_floor: float = 0.003
    cup_mass: float = 0.030
    cup_segments: int = 24              # number of wall panels in the collision ring
    cup_floor_shape: str = "box"         # 'box' (box-box contacts, most robust for a falling cube) or 'cylinder'
    cup_outer_rgba: tuple[float, float, float, float] = (0.80, 0.07, 0.10, 1.0)
    cup_inner_rgba: tuple[float, float, float, float] = (0.95, 0.93, 0.90, 1.0)
    cup_transparent_alpha: float | None = None  # e.g. 0.35 to render a translucent glass instead

    # Arm start pose in controller units (degrees; gripper 0-100). TCP ~ (0.29, 0, 0.08) m, clear of the table.
    # Gripper 22 = ~36 mm fingertip gap, the opening used for approach and release (see task.TaskParameters).
    start_pose: dict = field(default_factory=lambda: {
        "shoulder_pan": 0.0, "shoulder_lift": -35.0, "elbow_flex": 60.0,
        "wrist_flex": 10.0, "wrist_roll": 0.0, "gripper": 22.0,
    })

    overhead_camera: CameraSpec = field(default_factory=CameraSpec)
    wrist_camera: WristCameraSpec = field(default_factory=WristCameraSpec)

    # Reset jitter (m), applied to block and cup x/y with the reset seed. 0 = fully deterministic.
    block_jitter: float = 0.0
    cup_jitter: float = 0.0

    physics_timestep: float = 0.002
    # Optional visual stress case; baseline must not depict a non-colliding hazard.
    show_cable: bool = False

    # Gripper servo torque limit (N m). The menagerie model uses the STS3215 stall torque (2.94 N m), which squeezes
    # a 25 mm block with ~39 N and launches it sideways on release. 0.5 N m (~6.5 N at the pads) matches a
    # current-limited gripper; adjust to the real arm's configured limit.
    gripper_force_limit_nm: float = 0.5
    # Block contact time constant (s); must be >= 2 * physics_timestep. Given priority over the pads/cup. Values near
    # the 2*timestep limit make corner impacts inside the cup rebound unrealistically; 0.010 measured best.
    block_solref: float = 0.010

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneConfig":
        data = dict(data)
        if "overhead_camera" in data and isinstance(data["overhead_camera"], dict):
            data["overhead_camera"] = CameraSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in data["overhead_camera"].items()})
        if "wrist_camera" in data and isinstance(data["wrist_camera"], dict):
            data["wrist_camera"] = WristCameraSpec(**data["wrist_camera"])
        for key, value in list(data.items()):
            if isinstance(value, list):
                data[key] = tuple(value)
        return cls(**data)

    def with_(self, **changes) -> "SceneConfig":
        return replace(self, **changes)

    NON_GEOMETRY_FIELDS = ("block_jitter", "cup_jitter", "start_pose")

    def geometry_key(self) -> tuple:
        """Everything that changes the compiled model; reset() rebuilds only when this differs."""
        return tuple((k, repr(v)) for k, v in sorted(self.to_dict().items()) if k not in self.NON_GEOMETRY_FIELDS)

    @property
    def cup_interior_radius_top(self) -> float:
        return self.cup_radius_top - self.cup_collision_wall

    def cup_interior_radius_at(self, z_above_base: float) -> float:
        t = min(max(z_above_base / self.cup_height, 0.0), 1.0)
        return (self.cup_radius_bottom + (self.cup_radius_top - self.cup_radius_bottom) * t) - self.cup_collision_wall


# --------------------------------------------------------------------------------------------
# Quaternion helpers (MuJoCo convention: w x y z)

def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_axis(axis: str, angle: float):
    s, c = math.sin(angle / 2), math.cos(angle / 2)
    return {"x": (c, s, 0, 0), "y": (c, 0, s, 0), "z": (c, 0, 0, s)}[axis]


def fmt(*values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def camera_xyaxes(position: Vec3, lookat: Vec3) -> str:
    """MuJoCo camera `xyaxes`: camera looks along -z; x is image right, y is image up."""
    px, py, pz = position
    lx, ly, lz = lookat
    f = (lx - px, ly - py, lz - pz)
    n = math.sqrt(sum(v * v for v in f))
    f = tuple(v / n for v in f)
    up = (0.0, 0.0, 1.0)
    right = (f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0])
    n = math.sqrt(sum(v * v for v in right))
    right = tuple(v / n for v in right)
    cam_up = (right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2], right[0] * f[1] - right[1] * f[0])
    return fmt(*right, *cam_up)


def cup_collision_geoms(cfg: SceneConfig) -> list[str]:
    """Ring of tilted thin boxes plus a floor disc: a real opening and a hollow interior."""
    geoms = []
    h, rt, rb = cfg.cup_height, cfg.cup_radius_top, cfg.cup_radius_bottom
    wall = cfg.cup_collision_wall
    tilt = math.atan2(rt - rb, h)
    slant = math.hypot(rt - rb, h)
    n = cfg.cup_segments
    r_mid = 0.5 * (rt + rb) - wall / 2
    # Chord half-length at the rim (largest radius) with a small overlap so panels leave no gaps.
    chord_half = (rt * math.pi / n) * 1.08
    for i in range(n):
        theta = 2 * math.pi * (i + 0.5) / n
        q = quat_mul(quat_axis("z", theta), quat_axis("y", tilt))
        cx = r_mid * math.cos(theta) + (h / 2) * math.sin(tilt) * math.cos(theta) * 0  # centered on slant midpoint
        pos = (r_mid * math.cos(theta), r_mid * math.sin(theta), h / 2)
        geoms.append(
            f'<geom name="cup_wall_{i}" type="box" size="{fmt(wall / 2, chord_half, slant / 2)}" pos="{fmt(*pos)}" '
            f'quat="{fmt(*q)}" class="cup_collision"/>'
        )
    if cfg.cup_floor_shape == "box":
        # Square floor inscribed in the base circle; the wall ring closes the corners, and a 25 mm block cannot
        # wedge into the few-millimetre gaps at the ring. Box-box contacts avoid the box-cylinder convex solver.
        half = (rb - wall) / math.sqrt(2) * 1.02
        geoms.append(
            f'<geom name="cup_floor" type="box" size="{fmt(half, half, cfg.cup_floor / 2)}" '
            f'pos="{fmt(0, 0, cfg.cup_floor / 2)}" class="cup_collision"/>'
        )
    else:
        geoms.append(
            f'<geom name="cup_floor" type="cylinder" size="{fmt(rb - wall / 2, cfg.cup_floor / 2)}" '
            f'pos="{fmt(0, 0, cfg.cup_floor / 2)}" class="cup_collision"/>'
        )
    return geoms


def cup_mesh_files(cfg: SceneConfig) -> tuple[Path, Path]:
    """Visual cup meshes for these dimensions, generated on demand into assets/generated (cached by size)."""
    from .build_assets import cup_meshes, write_stl
    key = f"{cfg.cup_height:.4f}_{cfg.cup_radius_top:.4f}_{cfg.cup_radius_bottom:.4f}_{cfg.cup_wall:.4f}_{cfg.cup_floor:.4f}"
    outer = GENERATED / f"cup_outer_{key}.stl"
    inner = GENERATED / f"cup_inner_{key}.stl"
    if not outer.is_file() or not inner.is_file():
        GENERATED.mkdir(parents=True, exist_ok=True)
        tri_outer, tri_inner = cup_meshes(cfg.cup_height, cfg.cup_radius_top, cfg.cup_radius_bottom, cfg.cup_wall, cfg.cup_floor)
        write_stl(outer, tri_outer)
        write_stl(inner, tri_inner)
    return outer, inner


def _robot_body_xml() -> tuple[str, str, str]:
    """Split the menagerie robot XML into (defaults+assets, worldbody content, actuators) for inlining."""
    text = ROBOT_XML.read_text()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<option[^>]*/>", "", text)
    text = re.sub(r"<compiler[^>]*/>", "", text)
    text = re.sub(r"<visual>.*?</visual>", "", text, flags=re.S)
    defaults = re.search(r"<default>.*?</default>\s*(?=<asset>)", text, flags=re.S).group(0)
    assets = re.search(r"<asset>.*?</asset>", text, flags=re.S).group(0)
    world = re.search(r"<worldbody>(.*?)</worldbody>", text, flags=re.S).group(1)
    actuators = re.search(r"<actuator>.*?</actuator>", text, flags=re.S).group(0)
    return defaults + "\n" + assets, world, actuators


def build_scene_xml(cfg: SceneConfig) -> str:
    """Return a complete MJCF string. Robot XML is inlined; mesh/texture paths are absolute."""
    robot_defaults_assets, robot_world, robot_actuators = _robot_body_xml()
    robot_actuators = re.sub(r'(<position class="sts3215" name="gripper" joint="gripper")',
                             rf'\1 forcerange="{-cfg.gripper_force_limit_nm:.4g} {cfg.gripper_force_limit_nm:.4g}"', robot_actuators)
    oc = cfg.overhead_camera
    tx, ty, tz = cfg.table_center
    hx, hy, hz = cfg.table_half_extents
    bx, by, bz = cfg.block_position
    half = cfg.block_size / 2
    cup_outer = cfg.cup_outer_rgba
    cup_inner = cfg.cup_inner_rgba
    if cfg.cup_transparent_alpha is not None:
        cup_outer = (0.75, 0.85, 0.95, cfg.cup_transparent_alpha)
        cup_inner = (0.75, 0.85, 0.95, cfg.cup_transparent_alpha)
    cup_walls = "\n        ".join(cup_collision_geoms(cfg))
    cup_outer_path, cup_inner_path = cup_mesh_files(cfg)

    distractor = ""
    if cfg.distractor_block:
        dx, dy, dz = cfg.distractor_position
        distractor = f'''
    <body name="distractor" pos="{fmt(dx, dy, dz)}">
      <freejoint name="distractor_free"/>
      <geom name="distractor" type="box" size="{fmt(half, half, half)}" rgba="{fmt(*cfg.block_rgba)}" mass="{cfg.block_mass}" class="object"/>
    </body>'''

    cable = ""
    if cfg.show_cable:
        cable = f'''
    <geom name="cable" type="capsule" fromto="{fmt(0.04, 0.0, 0.005)} {fmt(0.75, -0.20, 0.004)}" size="0.003" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0" group="2"/>'''

    return f'''<mujoco model="so101_block_into_cup">
  <compiler angle="radian" meshdir="{ROBOT_MESHDIR}" texturedir="{GENERATED}" autolimits="true"/>
  <option integrator="implicitfast" timestep="{cfg.physics_timestep}" cone="elliptic" iterations="20" ls_iterations="30" impratio="10" gravity="0 0 -9.81"/>

  <visual>
    <headlight diffuse="0.18 0.18 0.18" ambient="0.22 0.22 0.22" specular="0.05 0.05 0.05"/>
    <rgba haze="0.05 0.05 0.06 1"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.01" zfar="10"/>
    <global offwidth="1280" offheight="960"/>
  </visual>

  {robot_defaults_assets}

  <default>
    <default class="object">
      <geom condim="4" friction="0.9 0.005 0.0001" solref="{cfg.block_solref} 1" solimp="0.95 0.99 0.001" priority="2"/>
    </default>
    <default class="cup_collision">
      <geom condim="3" friction="0.8 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001" group="3" rgba="1 0 0 0.15"/>
    </default>
    <default class="cup_visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" mass="0"/>
    </default>
  </default>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.08 0.07 0.07" rgb2="0.02 0.02 0.02" width="256" height="1536"/>
    <texture name="wood" type="2d" file="wood.png"/>
    <material name="wood" texture="wood" texrepeat="1 1" texuniform="false" specular="0.15" shininess="0.2" reflectance="0.03"/>
    <material name="floor_mat" rgba="0.18 0.14 0.12 1" specular="0" shininess="0"/>
    <mesh name="cup_outer" file="{cup_outer_path}"/>
    <mesh name="cup_inner" file="{cup_inner_path}"/>
  </asset>

  <worldbody>
    <light name="key" pos="0.45 0.20 1.1" dir="-0.25 -0.15 -1" directional="false" diffuse="0.55 0.50 0.42" specular="0.25 0.25 0.25" castshadow="true" cutoff="80" exponent="1"/>
    <light name="fill" pos="-0.3 -0.6 1.0" dir="0.4 0.5 -1" directional="true" diffuse="0.20 0.20 0.24" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 {cfg.floor_z}" size="3 3 0.1" material="floor_mat"/>
    <geom name="table_top" type="box" pos="{fmt(tx, ty, tz)}" size="{fmt(hx, hy, hz)}" material="wood" friction="0.9 0.005 0.0001" condim="3"/>
    <geom name="table_leg_1" type="box" pos="{fmt(tx - hx + 0.03, ty - hy + 0.03, (tz - hz + cfg.floor_z) / 2)}" size="{fmt(0.025, 0.025, (tz - hz - cfg.floor_z) / 2)}" rgba="0.35 0.28 0.2 1" contype="0" conaffinity="0"/>
    <geom name="table_leg_2" type="box" pos="{fmt(tx + hx - 0.03, ty - hy + 0.03, (tz - hz + cfg.floor_z) / 2)}" size="{fmt(0.025, 0.025, (tz - hz - cfg.floor_z) / 2)}" rgba="0.35 0.28 0.2 1" contype="0" conaffinity="0"/>
    <geom name="table_leg_3" type="box" pos="{fmt(tx - hx + 0.03, ty + hy - 0.03, (tz - hz + cfg.floor_z) / 2)}" size="{fmt(0.025, 0.025, (tz - hz - cfg.floor_z) / 2)}" rgba="0.35 0.28 0.2 1" contype="0" conaffinity="0"/>
    <geom name="table_leg_4" type="box" pos="{fmt(tx + hx - 0.03, ty + hy - 0.03, (tz - hz + cfg.floor_z) / 2)}" size="{fmt(0.025, 0.025, (tz - hz - cfg.floor_z) / 2)}" rgba="0.35 0.28 0.2 1" contype="0" conaffinity="0"/>
    <geom name="back_wall" type="box" pos="{fmt(-0.30, 0.0, 0.3)}" size="{fmt(0.02, 1.5, 1.2)}" rgba="0.12 0.10 0.10 1" contype="0" conaffinity="0"/>{cable}

    <camera name="overhead" mode="fixed" pos="{fmt(*oc.position)}" xyaxes="{camera_xyaxes(oc.position, oc.lookat)}" fovy="{oc.fovy_deg}" resolution="{oc.width} {oc.height}"/>

    <body name="block" pos="{fmt(bx, by, bz)}">
      <freejoint name="block_free"/>
      <geom name="block" type="box" size="{fmt(half, half, half)}" rgba="{fmt(*cfg.block_rgba)}" mass="{cfg.block_mass}" class="object"/>
    </body>{distractor}

    <body name="cup" pos="{fmt(*cfg.cup_position)}">
      <freejoint name="cup_free"/>
      <inertial pos="{fmt(0, 0, cfg.cup_height * 0.42)}" mass="{cfg.cup_mass}" diaginertia="{fmt(cfg.cup_mass * 0.0012, cfg.cup_mass * 0.0012, cfg.cup_mass * 0.0009)}"/>
      <geom name="cup_outer" mesh="cup_outer" rgba="{fmt(*cup_outer)}" class="cup_visual"/>
      <geom name="cup_inner" mesh="cup_inner" rgba="{fmt(*cup_inner)}" class="cup_visual"/>
        {cup_walls}
    </body>

    {robot_world}
  </worldbody>

  {robot_actuators}
</mujoco>
'''
