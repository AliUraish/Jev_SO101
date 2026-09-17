"""Damped least-squares inverse kinematics for the SO-101 tool center point (site `tcp`).

Solves for the five arm joints given a target TCP position and a desired approach direction
(the direction from wrist to fingertips, body -z of the gripper) and optionally the jaw closing
direction (body +x). Runs on a private MjData so it never disturbs the live simulation.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .units import ARM_JOINTS, dict_to_controller, from_controller


class IKSolver:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.qpos_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS])
        self.dof_adr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS])
        self.ranges = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS])
        self.gripper_qpos = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]

    # Canonical seeds spanning the elbow-up / elbow-down branches; tried in addition to q_init.
    SEEDS = (
        np.array([0.0, -0.5, 1.0, 0.35, 1.57]),
        np.array([0.0, -0.3, 0.1, 1.55, 1.57]),
        np.array([0.0, 0.1, 0.3, 1.2, 1.57]),
        np.array([0.0, -0.6, 0.6, 1.4, 1.57]),
        np.zeros(5),
    )

    def solve(self, target_pos, approach_dir=(0, 0, -1), closing_dir=None, q_init=None, gripper_percent=50.0,
              iterations=500, tol=1e-4, damping=1e-3, orientation_weight=0.08) -> tuple[np.ndarray, float]:
        """Return (q_rad for the five arm joints, final position error in meters). Tries q_init first and, if
        that converges poorly, several canonical seeds; the closest solution to q_init among the good ones wins."""
        best = None
        seeds = ([np.array(q_init, dtype=float)] if q_init is not None else []) + list(self.SEEDS)
        for k, seed in enumerate(seeds):
            q, err = self._solve_from(target_pos, approach_dir, closing_dir, seed, gripper_percent, iterations, tol, damping, orientation_weight)
            dist = np.linalg.norm(q - seeds[0]) if q_init is not None else 0.0
            if best is None or err < best[1] - 1e-3 or (abs(err - best[1]) <= 1e-3 and dist < best[2]):
                best = (q, err, dist)
            if k == 0 and err < tol * 5:
                break
        return best[0], best[1]

    def _solve_from(self, target_pos, approach_dir, closing_dir, q_init, gripper_percent, iterations, tol, damping, orientation_weight):
        m, d = self.model, self.data
        target = np.asarray(target_pos, dtype=float)
        approach = np.asarray(approach_dir, dtype=float)
        approach /= np.linalg.norm(approach)
        closing = None if closing_dir is None else np.asarray(closing_dir, dtype=float) / np.linalg.norm(closing_dir)
        q = np.array(q_init, dtype=float)
        mujoco.mj_resetData(m, d)
        d.qpos[self.gripper_qpos] = from_controller("gripper", gripper_percent)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        err = np.inf
        for _ in range(iterations):
            d.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site_xpos[self.site]
            R = d.site_xmat[self.site].reshape(3, 3)
            cur_approach = -R[:, 2]
            e_pos = target - pos
            # orientation error as rotation vector aligning cur_approach to approach
            e_rot = np.cross(cur_approach, approach)
            if closing is not None:
                cur_close = R[:, 0]
                # only the component of closing error orthogonal to the approach axis is meaningful
                e_close = np.cross(cur_close, closing)
                e_rot = e_rot + 0.5 * e_close
            err = np.linalg.norm(e_pos)
            if err < tol and np.linalg.norm(e_rot) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            Jp = jacp[:, self.dof_adr]
            Jr = jacr[:, self.dof_adr]
            J = np.vstack([Jp, orientation_weight * Jr])
            e = np.concatenate([e_pos, orientation_weight * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            step = np.clip(dq, -0.2, 0.2)
            q = np.clip(q + step, self.ranges[:, 0] + 0.02, self.ranges[:, 1] - 0.02)
        return q, float(err)

    def to_controller(self, q_rad: np.ndarray, gripper_percent: float) -> dict[str, float]:
        rad = {j: float(v) for j, v in zip(ARM_JOINTS, q_rad)}
        rad["gripper"] = from_controller("gripper", gripper_percent)
        return dict_to_controller(rad)

    def tcp_pose(self, q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        d.qpos[self.qpos_adr] = q_rad
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
