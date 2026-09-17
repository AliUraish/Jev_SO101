"""SO-101 block-into-cup MuJoCo simulation with a small, thread-safe interface.

Threads:
  * physics thread (optional, `realtime=True`): advances MuJoCo in wall-clock time so latency in the
    decision pipeline is not hidden. `step(dt)` is for paused/debug mode.
  * render thread: renders both cameras at `render_hz` and keeps only the newest frame per camera
    (like a USB webcam). Frames carry both simulation time and a monotonic wall-clock capture time.
Live MuJoCo state access goes through `self.lock`. Rendering uses a private data snapshot
so GPU work cannot block the physics or controller threads.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import SceneConfig, build_scene_xml
from .units import ARM_JOINTS, JOINTS, dict_from_controller, dict_to_controller, from_controller


@dataclass(frozen=True)
class Frame:
    camera: str
    rgb: np.ndarray            # (H, W, 3) uint8, RGB order (convert to BGR for OpenCV)
    sim_time: float            # seconds of simulated time at capture
    captured_at: float         # time.monotonic() at capture (wall clock)
    intrinsics: dict = field(default_factory=dict)


class SO101Sim:
    ROBOT_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")
    JAW_BODIES = ("gripper", "camera_mount")      # fixed jaw side
    MOVING_JAW_BODY = "moving_jaw_so101_v1"

    def __init__(self, scene_config: SceneConfig | None = None, realtime: bool = True, render_hz: float = 15.0,
                 realtime_factor: float = 1.0, max_catchup_s: float = 0.25):
        self.config = scene_config or SceneConfig()
        self.lock = threading.RLock()
        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(self.config))
        self.data = mujoco.MjData(self.model)
        self.realtime_factor = realtime_factor
        self.max_catchup_s = max_catchup_s
        self.render_hz = render_hz
        self._ids()
        self._frames: dict[str, Frame] = {}
        self._stop = threading.Event()
        self._physics_enabled = threading.Event()
        self._render_request = threading.Event()
        self._render_done = threading.Event()
        self._render_error: str | None = None
        self.stats = {"steps": 0, "physics_wall_s": 0.0, "render_count": 0, "render_ms_last": 0.0, "behind_s_max": 0.0}
        self._wall_anchor = None
        self._sim_anchor = None
        self._last_targets: dict[str, float] | None = None
        self._commands = 0
        self.reset()
        self._render_thread = threading.Thread(target=self._render_loop, name="so101-render", daemon=True)
        self._render_thread.start()
        self._physics_thread = threading.Thread(target=self._physics_loop, name="so101-physics", daemon=True)
        self._physics_thread.start()
        if realtime:
            self.start()
        self.wait_for_frames()

    # ------------------------------------------------------------------ setup helpers
    def _ids(self):
        m = self.model
        self.jnt_qpos = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS}
        self.jnt_dof = {j: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS}
        self.act_id = {j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS}
        self.ctrl_range = {j: tuple(m.actuator_ctrlrange[self.act_id[j]]) for j in JOINTS}
        self.body_id = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in
                        self.ROBOT_BODIES + ("block", "cup") + (("distractor",) if self.config.distractor_block else ())}
        self.block_qpos = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_free")]
        self.cup_qpos = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")]
        self.block_dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_free")]
        self.cup_dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")]
        self.table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.block_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "block")
        self.tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.robot_body_ids = {self.body_id[n] for n in self.ROBOT_BODIES}
        self.jaw_body_ids = {self.body_id[n] for n in self.JAW_BODIES}
        self.moving_jaw_id = self.body_id[self.MOVING_JAW_BODY]
        self.cameras = {
            "overhead": (self.config.overhead_camera.width, self.config.overhead_camera.height, self.config.overhead_camera.intrinsics),
            "wrist": (self.config.wrist_camera.width, self.config.wrist_camera.height, self.config.wrist_camera.intrinsics),
        }

    # ------------------------------------------------------------------ interface
    def reset(self, seed: int | None = None, scene_config: SceneConfig | None = None) -> dict:
        """Restore a reproducible starting scene. A new scene_config rebuilds the model (geometry changes)."""
        with self.lock:
            if scene_config is not None and scene_config != self.config:
                rebuild = scene_config.geometry_key() != self.config.geometry_key()
                self.config = scene_config
                if rebuild:
                    self.model = mujoco.MjModel.from_xml_string(build_scene_xml(self.config))
                    self.data = mujoco.MjData(self.model)
                    self._ids()
                    self._frames = {}
            m, d = self.model, self.data
            mujoco.mj_resetData(m, d)
            rng = np.random.default_rng(seed)
            bx, by, bz = self.config.block_position
            cx, cy, cz = self.config.cup_position
            if self.config.block_jitter > 0:
                bx, by = bx + rng.uniform(-1, 1) * self.config.block_jitter, by + rng.uniform(-1, 1) * self.config.block_jitter
            if self.config.cup_jitter > 0:
                cx, cy = cx + rng.uniform(-1, 1) * self.config.cup_jitter, cy + rng.uniform(-1, 1) * self.config.cup_jitter
            d.qpos[self.block_qpos:self.block_qpos + 7] = [bx, by, bz, 1, 0, 0, 0]
            d.qpos[self.cup_qpos:self.cup_qpos + 7] = [cx, cy, cz, 1, 0, 0, 0]
            if self.config.distractor_block:
                adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "distractor_free")]
                dx, dy, dz = self.config.distractor_position
                d.qpos[adr:adr + 7] = [dx, dy, dz, 1, 0, 0, 0]
            start = dict_from_controller(self.config.start_pose)
            for j in JOINTS:
                d.qpos[self.jnt_qpos[j]] = start[j]
                d.ctrl[self.act_id[j]] = np.clip(start[j], *self.ctrl_range[j])
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            # Let objects settle onto the table under the position hold (0.3 s of physics).
            for _ in range(int(0.3 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            self._last_targets = dict(self.config.start_pose)
            self._commands = 0
            self._wall_anchor = time.monotonic()
            self._sim_anchor = d.time
            self.stats["behind_s_max"] = 0.0
            return self.read_joint_positions()

    def read_joint_positions(self) -> dict[str, float]:
        """All six joints in controller units (degrees; gripper 0-100)."""
        with self.lock:
            rad = {j: float(self.data.qpos[self.jnt_qpos[j]]) for j in JOINTS}
        return dict_to_controller(rad)

    def read_joint_velocities(self) -> dict[str, float]:
        """deg/s for arm joints; %/s for the gripper."""
        with self.lock:
            vel = {j: float(self.data.qvel[self.jnt_dof[j]]) for j in JOINTS}
        out = {j: float(np.degrees(vel[j])) for j in ARM_JOINTS}
        out["gripper"] = 100.0 * vel["gripper"] / (from_controller("gripper", 100) - from_controller("gripper", 0))
        return out

    def send_joint_targets(self, targets: dict[str, float]) -> dict[str, float]:
        """Command position targets in controller units. Missing joints keep their last target.
        Returns the applied targets (clipped to the actuator control range) in controller units."""
        applied = dict(self._last_targets or self.config.start_pose)
        applied.update({j: float(v) for j, v in targets.items() if j in JOINTS})
        with self.lock:
            for j in JOINTS:
                rad = np.clip(from_controller(j, applied[j]), *self.ctrl_range[j])
                self.data.ctrl[self.act_id[j]] = rad
            self._last_targets = applied
            self._commands += 1
        return applied

    def last_targets(self) -> dict[str, float]:
        return dict(self._last_targets or {})

    def step(self, dt: float) -> float:
        """Advance physics by dt seconds synchronously (debug / paused mode). Returns simulation time."""
        with self.lock:
            n = max(1, int(round(dt / self.model.opt.timestep)))
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)
            self.stats["steps"] += n
            # keep the realtime anchor consistent if someone resumes later
            self._wall_anchor = time.monotonic()
            self._sim_anchor = self.data.time
            return float(self.data.time)

    def start(self):
        """Run physics continuously in wall-clock time."""
        with self.lock:
            self._wall_anchor = time.monotonic()
            self._sim_anchor = self.data.time
        self._physics_enabled.set()

    def pause(self):
        self._physics_enabled.clear()

    @property
    def running(self) -> bool:
        return self._physics_enabled.is_set()

    @property
    def sim_time(self) -> float:
        with self.lock:
            return float(self.data.time)

    def get_camera_frames(self) -> dict[str, Frame]:
        """Newest frame per camera: {'overhead': Frame, 'wrist': Frame}. RGB uint8 arrays."""
        if self._render_error:
            raise RuntimeError(f"Render thread failed: {self._render_error}")
        with self.lock:
            return dict(self._frames)

    def render_now(self, timeout_s: float = 5.0) -> dict[str, Frame]:
        """Render both cameras for the current state immediately (useful in stepped mode) and return the frames."""
        self._render_done.clear()
        self._render_request.set()
        if not self._render_done.wait(timeout_s):
            raise RuntimeError("Render thread did not respond")
        return self.get_camera_frames()

    def wait_for_frames(self, timeout_s: float = 10.0) -> dict[str, Frame]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frames = self.get_camera_frames()
            if set(frames) == set(self.cameras):
                return frames
            time.sleep(0.01)
        raise RuntimeError("Cameras did not produce frames in time")

    def get_ground_truth(self) -> dict:
        """Evaluator-only state. Do not feed this to Astra or Jev."""
        with self.lock:
            m, d = self.model, self.data
            cfg = self.config
            block_pos = d.qpos[self.block_qpos:self.block_qpos + 3].copy()
            block_quat = d.qpos[self.block_qpos + 3:self.block_qpos + 7].copy()
            block_vel = d.qvel[self.block_dof:self.block_dof + 6].copy()
            cup_pos = d.qpos[self.cup_qpos:self.cup_qpos + 3].copy()
            cup_quat = d.qpos[self.cup_qpos + 3:self.cup_qpos + 7].copy()
            cup_vel = d.qvel[self.cup_dof:self.cup_dof + 6].copy()
            cup_R = d.xmat[self.body_id["cup"]].reshape(3, 3).copy()
            tcp_pos = d.site_xpos[self.tcp_site].copy()
            tcp_R = d.site_xmat[self.tcp_site].reshape(3, 3).copy()
            contacts = self._contacts()
            sim_time = float(d.time)
            joints = {j: float(d.qpos[self.jnt_qpos[j]]) for j in JOINTS}

        up = cup_R[:, 2]
        cup_tilt_deg = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        # Block position expressed in the cup's frame (relative to the cup base center).
        rel = cup_R.T @ (block_pos - cup_pos)
        z_in_cup = float(rel[2])
        radial = float(np.hypot(rel[0], rel[1]))
        half = cfg.block_size / 2
        interior_r = cfg.cup_interior_radius_at(z_in_cup)
        block_inside_cup = bool(cup_tilt_deg < 20 and cfg.cup_floor - 0.002 <= z_in_cup - half + 0.004 and z_in_cup + half <= cfg.cup_height + 0.01 and radial + half * 0.8 <= interior_r + 0.002)
        block_touching_table = any(c["pair"] == {"block", "table"} for c in contacts)
        block_touching_cup = any("block" in c["pair"] and "cup" in c["pair"] for c in contacts)
        block_touching_fixed_jaw = any(c["pair"] == {"block", "fixed_jaw"} for c in contacts)
        block_touching_moving_jaw = any(c["pair"] == {"block", "moving_jaw"} for c in contacts)
        block_held = block_touching_fixed_jaw and block_touching_moving_jaw
        block_lifted = bool(block_pos[2] > half + 0.01 and not block_touching_table)
        block_speed = float(np.linalg.norm(block_vel[:3]))
        block_settled = block_speed < 0.01 and float(np.linalg.norm(block_vel[3:])) < 0.2
        cup_tipped = cup_tilt_deg > 30 or cup_pos[2] < -0.05
        block_on_floor = block_pos[2] < -0.05
        arm_table = any("table" in c["pair"] and c["pair"] & {"arm", "fixed_jaw", "moving_jaw"} for c in contacts)
        arm_cup = any("cup" in c["pair"] and c["pair"] & {"arm", "fixed_jaw", "moving_jaw"} for c in contacts)
        success = block_inside_cup and block_settled and not block_held and not cup_tipped
        return {
            "sim_time": sim_time,
            "joints_rad": joints,
            "block": {"position": block_pos.tolist(), "quat_wxyz": block_quat.tolist(), "linear_velocity": block_vel[:3].tolist(),
                       "angular_velocity": block_vel[3:].tolist(), "speed": block_speed},
            "cup": {"position": cup_pos.tolist(), "quat_wxyz": cup_quat.tolist(), "tilt_deg": cup_tilt_deg,
                     "linear_velocity": cup_vel[:3].tolist()},
            "tcp": {"position": tcp_pos.tolist(), "approach_dir": (-tcp_R[:, 2]).tolist(), "closing_dir": tcp_R[:, 0].tolist()},
            "block_in_cup_frame": {"z": z_in_cup, "radial": radial, "interior_radius_here": interior_r},
            "flags": {
                "block_lifted": block_lifted,
                "block_held": block_held,
                "block_touching_table": block_touching_table,
                "block_touching_cup": block_touching_cup,
                "block_inside_cup": block_inside_cup,
                "block_settled": block_settled,
                "cup_tipped": bool(cup_tipped),
                "block_on_floor": bool(block_on_floor),
                "arm_touching_table": arm_table,
                "arm_touching_cup": arm_cup,
                "success": bool(success),
            },
            "contacts": contacts,
            "stats": dict(self.stats),
        }

    def close(self):
        self._stop.set()
        self._physics_enabled.set()   # unblock the physics loop so it can observe stop
        for t in (self._physics_thread, self._render_thread):
            t.join(timeout=2)

    # ------------------------------------------------------------------ internals
    def _label(self, geom: int) -> str:
        body = self.model.geom_bodyid[geom]
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if geom == self.table_geom:
            return "table"
        if name == "block":
            return "block"
        if name == "distractor":
            return "distractor"
        if name.startswith("cup_"):
            return "cup"
        if body == self.moving_jaw_id:
            return "moving_jaw"
        if body in self.jaw_body_ids:
            return "fixed_jaw"
        if body in self.robot_body_ids:
            return "arm"
        return name or f"body{body}"

    def _contacts(self) -> list[dict]:
        d = self.data
        out = []
        force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(self.model, d, i, force)
            out.append({"pair": {self._label(c.geom1), self._label(c.geom2)}, "geoms": [c.geom1, c.geom2],
                        "normal_force": float(force[0]), "dist": float(c.dist)})
        for c in out:
            c["pair"] = set(c["pair"])
        return out

    def _physics_loop(self):
        m = self.model
        while not self._stop.is_set():
            if not self._physics_enabled.wait(timeout=0.05):
                continue
            if self._stop.is_set():
                break
            with self.lock:
                m, d = self.model, self.data
                target = self._sim_anchor + (time.monotonic() - self._wall_anchor) * self.realtime_factor
                behind = target - d.time
                if behind > self.max_catchup_s:
                    # Too far behind (e.g. a long render or GC pause): drop time rather than spiral.
                    self.stats["behind_s_max"] = max(self.stats["behind_s_max"], behind)
                    self._sim_anchor = d.time
                    self._wall_anchor = time.monotonic()
                    behind = 0.0
                n = int(behind / m.opt.timestep)
                if n > 0:
                    t0 = time.perf_counter()
                    for _ in range(n):
                        mujoco.mj_step(m, d)
                    self.stats["steps"] += n
                    self.stats["physics_wall_s"] += time.perf_counter() - t0
            time.sleep(0.001)

    def _render_loop(self):
        renderer = None
        try:
            model_ref = None
            render_data = None
            period = 1.0 / self.render_hz
            while not self._stop.is_set():
                t0 = time.perf_counter()
                requested = self._render_request.is_set()
                if requested:
                    self._render_request.clear()
                with self.lock:
                    snapshot_model = self.model
                    cameras = dict(self.cameras)
                    if model_ref is not snapshot_model:
                        render_data = mujoco.MjData(snapshot_model)
                    captured_at = time.monotonic()
                    mujoco.mj_copyData(render_data, snapshot_model, self.data)
                if renderer is None or model_ref is not snapshot_model:
                    if renderer is not None:
                        renderer.close()
                    max_h = max(h for (_, h, _) in cameras.values())
                    max_w = max(w for (w, _, _) in cameras.values())
                    renderer = mujoco.Renderer(snapshot_model, max_h, max_w)
                    model_ref = snapshot_model
                frames = {}
                for name, (w, h, intr) in cameras.items():
                    renderer.update_scene(render_data, camera=name)
                    rgb = renderer.render().copy()
                    if rgb.shape[0] != h or rgb.shape[1] != w:
                        rgb = rgb[:h, :w]
                    frames[name] = Frame(name, rgb, float(render_data.time), captured_at, intr)
                with self.lock:
                    if self.model is snapshot_model:
                        self._frames = frames
                self.stats["render_count"] += 1
                self.stats["render_ms_last"] = (time.perf_counter() - t0) * 1000
                if requested:
                    self._render_done.set()
                # Sleep until the next scheduled frame, waking early for an on-demand render.
                self._render_request.wait(max(0.0, period - (time.perf_counter() - t0)))
        except Exception as exc:  # surfaced on the next get_camera_frames()
            self._render_error = f"{type(exc).__name__}: {exc}"
        finally:
            if renderer is not None:
                renderer.close()
