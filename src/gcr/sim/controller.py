"""Task-space gripper controller for the two-UR10e cable scene.

Wraps the damped-least-squares IK solver with a per-step interface: the
caller pushes a desired gripper-tip world pose for each UR10e, and the
controller updates each joint's setpoint (`m.qpos_spring` for the passive
spring + `data.ctrl` for the position-controlled actuator) so the robot
tracks the target.

Design notes:
- IK runs each step against a *throwaway* MjData copy so we don't disturb
  the live integration.
- The mocap-anchor wrapper (`update_anchor_mocaps` from build_scene) must
  also be called every step so the cable's connect equality follows the
  gripper. This controller does NOT call that wrapper itself; the caller is
  responsible (the state machine in this package handles it).
- Orientation is tracked via direct axis-direction errors (e = 0.5 *
  (current_axis - target_axis)) with exact Jacobians J = -skew(axis) @
  jac_rot.  This avoids the 180° singularity of the rotation-matrix
  skew-symmetric formula, which returns zero error when the axes are
  anti-parallel and causes the end-effector to flip.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from gcr.sim.build_scene import (
    _ur10_joint_indices,
    _UR10_ACTUATOR_NAMES,
    _UR10_JOINT_NAMES,
    N_ROBOT_JOINTS,
)


def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix such that skew(v) @ u == v × u."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


@dataclass
class IKSolveOptions:
    max_iter: int = 50
    pos_tol: float = 1e-3
    # Orientation tolerance for direct axis errors (||0.5*(current-target)||).
    # Value 0.20 ≈ accepts ≤ 23° axis tilt, suitable for floor-level reaching.
    ori_tol: float = 0.20
    step_size: float = 0.1
    damping: float = 1e-3


class DualUR10Controller:
    """Drive two UR10e robots to track per-side gripper-tip world targets.

    Usage::

        ctrl = DualUR10Controller(model)
        for t in range(N):
            ctrl.set_targets(model, data, left_xyz, right_xyz)
            update_anchor_mocaps(model, data)
            mujoco.mj_step(model, data)
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self._left_qadr, self._left_dadr = _ur10_joint_indices(model, "left")
        self._right_qadr, self._right_dadr = _ur10_joint_indices(model, "right")
        self._left_tip_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "ur10_left/gripper_tip"
        )
        self._right_tip_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "ur10_right/gripper_tip"
        )
        if self._left_tip_id < 0 or self._right_tip_id < 0:
            raise RuntimeError("gripper_tip bodies missing — was build_and_compile run?")
        self._left_act = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ur10_left/{aname}")
            for aname in _UR10_ACTUATOR_NAMES
        ])
        self._right_act = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ur10_right/{aname}")
            for aname in _UR10_ACTUATOR_NAMES
        ])
        self._left_lo, self._left_hi = self._joint_ranges(model, "left")
        self._right_lo, self._right_hi = self._joint_ranges(model, "right")
        self._ik_data = mujoco.MjData(model)
        self._smooth_alpha = 0.3

    @staticmethod
    def _joint_ranges(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(N_ROBOT_JOINTS)
        hi = np.empty(N_ROBOT_JOINTS)
        for j, jname in enumerate(_UR10_JOINT_NAMES):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"ur10_{side}/{jname}"
            )
            lo[j], hi[j] = model.jnt_range[jid]
        return lo, hi

    def _solve_step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: str,
        target_xyz: np.ndarray,
        opts: IKSolveOptions,
        target_yaw: float | None = None,
    ) -> np.ndarray:
        """DLS IK: position + orientation.

        z_tool always tracks [0,0,-1] (gripper straight down).
        When ``target_yaw`` is given, x_tool is additionally driven to
        [cos θ, sin θ, 0] (cable-tangent alignment).

        Orientation uses direct axis-direction errors with exact Jacobians:
            e_axis = 0.5 * (R[:,i] - target_axis)
            J_axis = -skew(R[:,i]) @ jac_rot   (from d(R[:,i])/dt = ω × R[:,i])

        This never produces a zero gradient at 180° error, so the
        end-effector cannot get stuck in an anti-parallel (flipped) state.
        """
        if side == "left":
            qadr, dadr, tip_id = self._left_qadr, self._left_dadr, self._left_tip_id
            lo, hi = self._left_lo, self._left_hi
        else:
            qadr, dadr, tip_id = self._right_qadr, self._right_dadr, self._right_tip_id
            lo, hi = self._right_lo, self._right_hi

        _z_down = np.array([0.0, 0.0, -1.0])
        x_target = (
            np.array([float(np.cos(target_yaw)), float(np.sin(target_yaw)), 0.0])
            if target_yaw is not None else None
        )

        self._ik_data.qpos[qadr] = data.qpos[qadr]

        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))

        for _ in range(opts.max_iter):
            mujoco.mj_forward(model, self._ik_data)

            e_pos = target_xyz - self._ik_data.xpos[tip_id]
            R = self._ik_data.xmat[tip_id].reshape(3, 3)

            # z-axis: drive z_tool → [0, 0, -1]  (target − current, same sign as e_pos)
            e_z = 0.5 * (_z_down - R[:, 2])

            # x-axis: drive x_tool → [cos θ, sin θ, 0] when yaw requested
            e_x = 0.5 * (x_target - R[:, 0]) if x_target is not None else None

            pos_ok = float(np.linalg.norm(e_pos)) < opts.pos_tol
            # ||e_z|| = 0.5*||target-current||; at 23° tilt this is ~0.20
            ori_ok = float(np.linalg.norm(e_z)) < opts.ori_tol
            if e_x is not None:
                ori_ok = ori_ok and float(np.linalg.norm(e_x)) < opts.ori_tol
            if pos_ok and ori_ok:
                break

            mujoco.mj_jacBody(model, self._ik_data, jac_pos, jac_rot, tip_id)
            J_p = jac_pos[:, dadr]   # 3×6
            J_r = jac_rot[:, dadr]   # 3×6

            # Exact axis-direction Jacobians: d(R[:,i])/dq = -skew(R[:,i]) @ J_r
            J_z = -_skew(R[:, 2]) @ J_r   # 3×6
            w_z = 0.8

            if e_x is not None:
                J_x = -_skew(R[:, 0]) @ J_r   # 3×6
                w_x = 0.6
                J = np.vstack([J_p, w_z * J_z, w_x * J_x])            # 9×6
                e = np.concatenate([e_pos, w_z * e_z, w_x * e_x])     # 9
                A = J @ J.T + opts.damping * np.eye(9)
            else:
                J = np.vstack([J_p, w_z * J_z])                        # 6×6
                e = np.concatenate([e_pos, w_z * e_z])                 # 6
                A = J @ J.T + opts.damping * np.eye(6)

            delta = J.T @ np.linalg.solve(A, e)
            self._ik_data.qpos[qadr] = np.clip(
                self._ik_data.qpos[qadr] + opts.step_size * delta, lo, hi
            )

        return self._ik_data.qpos[qadr].copy()

    def set_targets(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        left_target: np.ndarray,
        right_target: np.ndarray,
        opts: IKSolveOptions | None = None,
        left_yaw: float | None = None,
        right_yaw: float | None = None,
    ) -> None:
        """Solve IK for both UR10es and write new joint setpoints.

        ``left_yaw`` / ``right_yaw``: cable-tangent yaw angles (rad) that
        additionally constrain the gripper x-axis to align with the cable
        direction.  Pass ``None`` for z-axis-only orientation tracking.
        """
        if opts is None:
            opts = IKSolveOptions()
        for side, target, qadr, act_ids, yaw in (
            ("left",  left_target,  self._left_qadr,  self._left_act,  left_yaw),
            ("right", right_target, self._right_qadr, self._right_act, right_yaw),
        ):
            q_ik = self._solve_step(
                model, data, side,
                np.asarray(target, dtype=np.float64),
                opts, target_yaw=yaw,
            )
            current = model.qpos_spring[qadr]
            blended = (1 - self._smooth_alpha) * current + self._smooth_alpha * q_ik
            for j in range(N_ROBOT_JOINTS):
                model.qpos_spring[qadr[j]] = blended[j]
                if act_ids[j] >= 0:
                    data.ctrl[act_ids[j]] = blended[j]

    def gripper_position(self, data: mujoco.MjData, side: str) -> np.ndarray:
        tip_id = self._left_tip_id if side == "left" else self._right_tip_id
        return data.xpos[tip_id].copy()
