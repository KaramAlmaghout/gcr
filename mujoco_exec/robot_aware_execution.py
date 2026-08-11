"""Robot-aware MuJoCo execution helpers.

This module is intentionally separate from ``execute_plan.py`` and
``run_csv.py``.  It adds experimental robot-aware checks and regrasp support
without changing the baseline simulation scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from dlo2026.mjlab.controller import DualUR10Controller, IKSolveOptions
import dlo2026.mjlab.scene.build_scene as scene_build
from dlo2026.mjlab.scene.build_scene import (
    CABLE_LENGTH,
    CABLE_NUM_SEGMENTS,
    CABLE_Z,
    UR10_LEFT_BASE_POS,
    UR10_RIGHT_BASE_POS,
    apply_cable_endpoint_tangent_assist,
    build_and_compile,
    set_cable_config,
    update_anchor_mocaps,
    _set_weld_relpose_to_current_state,
)
from mujoco_exec.execute_plan import (
    DEFAULT_MAX_GRIPPER_SPEED,
    GHOST_TARGET_SHAPE_CHOICES,
    ROTATE_THEN_MOVE_YAW_FRACTION,
    WS_3D_X,
    WS_3D_Y,
    YAW_MODE_CHOICES,
    YAW_MODE_ROTATE_THEN_MOVE,
    YAW_MODE_SIMULTANEOUS,
    _cosine_ease_step,
    _current_cable_points,
    _polyline_length,
    _resample_polyline_nodes,
    _steps_for_waypoint,
    _validate_yaw_mode,
    _invert_yaw_delta,
    _wrap_pi,
    compute_cable_xy_error,
    print_cable_xy_error_report,
)


Target = tuple[np.ndarray, np.ndarray, float, float]
YAW_INTERPOLATION_MODES = ("shortest", "csv", "positive", "negative")
TCP_SCHEDULE_CHOICES = ("simultaneous", "auto-stagger", "left-first", "right-first")


@dataclass
class RobotAwareOptions:
    min_gripper_distance: float = 0.11
    min_arm_segment_distance: float = 0.13
    max_yaw_jump_deg: float = 120.0
    max_reach: float = 1.18
    min_reach_margin: float = 0.035
    workspace_margin: float = 0.015
    obstacle_clearance: float = 0.025
    tracking_error_limit: float = 0.045
    auto_regrasp: bool = False
    allow_regrasp: bool = False
    swap_after_regrasp: bool = False
    regrasp_steps: int = 220
    max_auto_regrasps: int = 2
    left_yaw_interpolation: str = "csv"
    right_yaw_interpolation: str = "csv"
    initial_pose_search: bool = True
    use_profile_yaw: bool = False
    invert_red_anchor_yaw: bool = False
    cable_attachment_mode: str | None = None
    cable_weld_torquescale: float | None = None
    cable_endpoint_tangent_assist: bool = False
    cable_endpoint_tangent_alpha: float = 0.04
    cable_endpoint_clamp_segments: int = 2
    cable_endpoint_tangent_velocity_damping: float = 0.0
    ghost_target_shape: str = "final"
    ghost_target_rgba: str = "1.0 0.86 0.18 0.35"
    tcp_schedule: str = "simultaneous"
    tcp_stagger_fraction: float = 0.35
    cable_length_margin: float = 0.003


@dataclass
class RobotAwareScore:
    n_waypoints: int = 0
    min_gripper_distance: float = float("inf")
    min_arm_segment_distance: float = float("inf")
    min_left_reach_margin: float = float("inf")
    min_right_reach_margin: float = float("inf")
    max_yaw_jump_deg: float = 0.0
    workspace_violations: int = 0
    obstacle_clearance_violations: int = 0
    reach_violations: int = 0
    gripper_distance_violations: int = 0
    arm_distance_violations: int = 0
    yaw_jump_violations: int = 0
    endpoint_assignment_penalty: int = 0
    recommended_regrasp_indices: list[int] | None = None

    def total_penalty(self) -> float:
        return (
            1000.0 * (self.workspace_violations + self.reach_violations)
            + 600.0 * (self.gripper_distance_violations + self.arm_distance_violations)
            + 300.0 * self.obstacle_clearance_violations
            + 120.0 * self.yaw_jump_violations
            + 80.0 * self.endpoint_assignment_penalty
            - 5.0 * min(self.min_gripper_distance, 1.0)
            - 2.0 * min(self.min_arm_segment_distance, 1.0)
        )


@dataclass
class RuntimeStats:
    robot_robot_contacts: int = 0
    robot_obstacle_contacts: int = 0
    max_tracking_error: float = 0.0
    tracking_error_violations: int = 0
    regrasp_events: int = 0


def _xy(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=float)[:2]


def yaw_delta(target: float, previous: float, mode: str) -> float:
    """Return the commanded yaw increment from previous to target.

    ``shortest`` is the baseline behavior.  ``csv`` preserves the sign and
    magnitude implied by the CSV value itself, which is useful when a cable
    endpoint must rotate through the non-shortest equivalent direction to avoid
    twisting the welded cable.  ``positive`` and ``negative`` force a winding
    direction using an equivalent +/- 2*pi target when needed.
    """
    raw = float(target - previous)
    if mode == "csv":
        return raw
    delta = float(_wrap_pi(raw))
    if mode == "shortest":
        return delta
    if mode == "positive":
        return delta if delta >= -1e-9 else delta + 2.0 * np.pi
    if mode == "negative":
        return delta if delta <= 1e-9 else delta - 2.0 * np.pi
    raise ValueError(f"Unknown yaw interpolation mode {mode!r}; expected one of {YAW_INTERPOLATION_MODES}")


def _profile_endpoint_yaws(profile_3d: np.ndarray) -> tuple[float, float]:
    """Return left/right gripper yaws from the endpoint tangents of a cable profile."""
    profile_3d = np.asarray(profile_3d, dtype=float)
    if profile_3d.ndim != 2 or profile_3d.shape[1] < 2 or len(profile_3d) < 2:
        raise ValueError("cable profile must be an (N, >=2) array with N >= 2")
    left_vec = profile_3d[1, :2] - profile_3d[0, :2]
    right_vec = profile_3d[-2, :2] - profile_3d[-1, :2]
    return float(np.arctan2(left_vec[1], left_vec[0])), float(np.arctan2(right_vec[1], right_vec[0]))


def _unwrap_near(raw_angle: float, previous: float) -> float:
    """Choose the equivalent of raw_angle that is continuous from previous."""
    return float(previous + _wrap_pi(float(raw_angle) - float(previous)))


def _equivalent_near(raw_angle: float, reference: float) -> float:
    """Choose the equivalent of raw_angle closest to a desired yaw branch."""
    return float(reference + _wrap_pi(float(raw_angle) - float(reference)))


def _segment_distance_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    def point_seg(p: np.ndarray, s0: np.ndarray, s1: np.ndarray) -> float:
        v = s1 - s0
        denom = float(np.dot(v, v))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - s0))
        t = float(np.clip(np.dot(p - s0, v) / denom, 0.0, 1.0))
        return float(np.linalg.norm(p - (s0 + t * v)))

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def intersects() -> bool:
        o1, o2 = orient(a, b, c), orient(a, b, d)
        o3, o4 = orient(c, d, a), orient(c, d, b)
        return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

    if intersects():
        return 0.0
    return min(point_seg(a, c, d), point_seg(b, c, d), point_seg(c, a, b), point_seg(d, a, b))


def _raw_tcp_schedule_progress(t: float, schedule: str, stagger_fraction: float) -> tuple[float, float]:
    """Return left/right path progress for a TCP timing schedule."""
    t = float(np.clip(t, 0.0, 1.0))
    stagger = float(np.clip(stagger_fraction, 0.0, 0.85))
    if schedule == "simultaneous":
        return t, t
    denom = max(1.0 - stagger, 1e-6)
    lead = min(t / denom, 1.0)
    follow = min(max((t - stagger) / denom, 0.0), 1.0)
    if schedule == "left-first":
        return lead, follow
    if schedule == "right-first":
        return follow, lead
    raise ValueError(f"Unknown TCP schedule {schedule!r}; expected one of {TCP_SCHEDULE_CHOICES}")


def _tcp_pose(start: np.ndarray, target: np.ndarray, progress: float) -> np.ndarray:
    return np.asarray(start, dtype=float) + float(progress) * (np.asarray(target, dtype=float) - np.asarray(start, dtype=float))


def _tcp_distance_for_progress(
    left_start: np.ndarray,
    left_target: np.ndarray,
    right_start: np.ndarray,
    right_target: np.ndarray,
    left_progress: float,
    right_progress: float,
) -> float:
    left = _tcp_pose(left_start, left_target, left_progress)
    right = _tcp_pose(right_start, right_target, right_progress)
    return float(np.linalg.norm(left - right))


def _enforce_tcp_cable_length(
    left_start: np.ndarray,
    left_target: np.ndarray,
    right_start: np.ndarray,
    right_target: np.ndarray,
    left_progress: float,
    right_progress: float,
    schedule: str,
    max_tcp_distance: float,
) -> tuple[float, float]:
    """Keep TCP separation inside the cable length by advancing the follower."""
    if max_tcp_distance <= 0.0:
        return left_progress, right_progress
    if _tcp_distance_for_progress(left_start, left_target, right_start, right_target, left_progress, right_progress) <= max_tcp_distance:
        return left_progress, right_progress

    if schedule == "left-first":
        follower = "right"
    elif schedule == "right-first":
        follower = "left"
    else:
        # Simultaneous progress has no obvious follower. Keep the requested
        # progress; the planner endpoints should already be within cable length.
        return left_progress, right_progress

    if follower == "right":
        lo, hi = right_progress, 1.0
        if _tcp_distance_for_progress(left_start, left_target, right_start, right_target, left_progress, hi) <= max_tcp_distance:
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                if _tcp_distance_for_progress(left_start, left_target, right_start, right_target, left_progress, mid) <= max_tcp_distance:
                    hi = mid
                else:
                    lo = mid
            return left_progress, hi
        lo, hi = 0.0, left_progress
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if _tcp_distance_for_progress(left_start, left_target, right_start, right_target, mid, right_progress) <= max_tcp_distance:
                lo = mid
            else:
                hi = mid
        return lo, right_progress

    lo, hi = left_progress, 1.0
    if _tcp_distance_for_progress(left_start, left_target, right_start, right_target, hi, right_progress) <= max_tcp_distance:
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if _tcp_distance_for_progress(left_start, left_target, right_start, right_target, mid, right_progress) <= max_tcp_distance:
                hi = mid
            else:
                lo = mid
        return hi, right_progress
    lo, hi = 0.0, right_progress
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if _tcp_distance_for_progress(left_start, left_target, right_start, right_target, left_progress, mid) <= max_tcp_distance:
            lo = mid
        else:
            hi = mid
    return left_progress, lo


def _tcp_schedule_progress(
    t: float,
    schedule: str,
    left_start: np.ndarray,
    left_target: np.ndarray,
    right_start: np.ndarray,
    right_target: np.ndarray,
    cable_length: float,
    options: RobotAwareOptions,
) -> tuple[float, float]:
    left_progress, right_progress = _raw_tcp_schedule_progress(t, schedule, options.tcp_stagger_fraction)
    max_tcp_distance = max(0.0, float(cable_length) - float(options.cable_length_margin))
    return _enforce_tcp_cable_length(
        left_start,
        left_target,
        right_start,
        right_target,
        left_progress,
        right_progress,
        schedule,
        max_tcp_distance,
    )


def _tcp_schedule_score(
    schedule: str,
    left_start: np.ndarray,
    left_target: np.ndarray,
    right_start: np.ndarray,
    right_target: np.ndarray,
    cable_length: float,
    options: RobotAwareOptions,
) -> tuple[float, float, float]:
    lb = np.asarray(UR10_LEFT_BASE_POS[:2], dtype=float)
    rb = np.asarray(UR10_RIGHT_BASE_POS[:2], dtype=float)
    min_tcp = float("inf")
    min_arm = float("inf")
    max_excess = 0.0
    for t in np.linspace(0.0, 1.0, 61):
        lp, rp = _tcp_schedule_progress(
            float(t),
            schedule,
            left_start,
            left_target,
            right_start,
            right_target,
            cable_length,
            options,
        )
        left = _tcp_pose(left_start, left_target, lp)
        right = _tcp_pose(right_start, right_target, rp)
        tcp_dist = float(np.linalg.norm(left - right))
        min_tcp = min(min_tcp, tcp_dist)
        min_arm = min(min_arm, _segment_distance_2d(lb, left[:2], rb, right[:2]))
        max_excess = max(max_excess, tcp_dist - max(0.0, float(cable_length) - float(options.cable_length_margin)))
    penalty = (
        10000.0 * max(0.0, max_excess)
        + 1000.0 * max(0.0, options.min_gripper_distance - min_tcp)
        + 700.0 * max(0.0, options.min_arm_segment_distance - min_arm)
        - 6.0 * min(min_tcp, 0.50)
        - 3.0 * min(min_arm, 0.50)
    )
    return penalty, min_tcp, min_arm


def _choose_tcp_schedule(
    left_start: np.ndarray,
    left_target: np.ndarray,
    right_start: np.ndarray,
    right_target: np.ndarray,
    cable_length: float,
    options: RobotAwareOptions,
) -> tuple[str, dict[str, tuple[float, float, float]]]:
    scores = {
        schedule: _tcp_schedule_score(schedule, left_start, left_target, right_start, right_target, cable_length, options)
        for schedule in ("simultaneous", "left-first", "right-first")
    }
    best = min(scores, key=lambda name: (scores[name][0], 0 if name == "simultaneous" else 1))
    return best, scores


def _point_obstacle_clearance(point_xy: np.ndarray, obstacles_3d: list[dict]) -> float:
    best = float("inf")
    p = np.asarray(point_xy, dtype=float)
    for obs in obstacles_3d:
        if "radius" in obs:
            c = np.asarray(obs["pos"][:2], dtype=float)
            best = min(best, float(np.linalg.norm(p - c) - float(obs["radius"])))
        elif "half_size" in obs:
            c = np.asarray(obs["pos"][:2], dtype=float)
            hx, hy = float(obs["half_size"][0]), float(obs["half_size"][1])
            q = np.abs(p - c) - np.array([hx, hy])
            outside = np.linalg.norm(np.maximum(q, 0.0))
            inside = min(max(q[0], q[1]), 0.0)
            best = min(best, float(outside + inside))
        elif "vertices" in obs:
            verts = np.asarray(obs["vertices"], dtype=float)[:, :2]
            xmin, ymin = np.min(verts, axis=0)
            xmax, ymax = np.max(verts, axis=0)
            q = np.array([max(xmin - p[0], 0.0, p[0] - xmax), max(ymin - p[1], 0.0, p[1] - ymax)])
            best = min(best, float(np.linalg.norm(q)))
    return best


def score_waypoints(targets: list[Target], obstacles_3d: list[dict] | None, options: RobotAwareOptions) -> RobotAwareScore:
    obstacles_3d = list(obstacles_3d or [])
    score = RobotAwareScore(n_waypoints=len(targets), recommended_regrasp_indices=[])
    lb = np.asarray(UR10_LEFT_BASE_POS[:2], dtype=float)
    rb = np.asarray(UR10_RIGHT_BASE_POS[:2], dtype=float)
    yaw_limit = float(options.max_yaw_jump_deg)

    prev_ly = prev_ry = None
    violation_indices: list[int] = []
    for idx, (left, right, ly, ry) in enumerate(targets):
        lxy, rxy = _xy(left), _xy(right)
        grip_dist = float(np.linalg.norm(lxy - rxy))
        arm_dist = _segment_distance_2d(lb, lxy, rb, rxy)
        score.min_gripper_distance = min(score.min_gripper_distance, grip_dist)
        score.min_arm_segment_distance = min(score.min_arm_segment_distance, arm_dist)

        left_margin = options.max_reach - float(np.linalg.norm(lxy - lb))
        right_margin = options.max_reach - float(np.linalg.norm(rxy - rb))
        score.min_left_reach_margin = min(score.min_left_reach_margin, left_margin)
        score.min_right_reach_margin = min(score.min_right_reach_margin, right_margin)

        bad = False
        if grip_dist < options.min_gripper_distance:
            score.gripper_distance_violations += 1
            bad = True
        if arm_dist < options.min_arm_segment_distance:
            score.arm_distance_violations += 1
            bad = True
        if left_margin < options.min_reach_margin or right_margin < options.min_reach_margin:
            score.reach_violations += 1
            bad = True
        if not (WS_3D_X[0] - options.workspace_margin <= left[0] <= WS_3D_X[1] + options.workspace_margin):
            score.workspace_violations += 1
            bad = True
        if not (WS_3D_X[0] - options.workspace_margin <= right[0] <= WS_3D_X[1] + options.workspace_margin):
            score.workspace_violations += 1
            bad = True
        if not (WS_3D_Y[0] - options.workspace_margin <= left[1] <= WS_3D_Y[1] + options.workspace_margin):
            score.workspace_violations += 1
            bad = True
        if not (WS_3D_Y[0] - options.workspace_margin <= right[1] <= WS_3D_Y[1] + options.workspace_margin):
            score.workspace_violations += 1
            bad = True
        if min(_point_obstacle_clearance(lxy, obstacles_3d), _point_obstacle_clearance(rxy, obstacles_3d)) < options.obstacle_clearance:
            score.obstacle_clearance_violations += 1
            bad = True

        natural = float(np.linalg.norm(lxy - lb) + np.linalg.norm(rxy - rb))
        swapped = float(np.linalg.norm(rxy - lb) + np.linalg.norm(lxy - rb))
        if swapped + 0.04 < natural:
            score.endpoint_assignment_penalty += 1
            bad = True

        if prev_ly is not None:
            yaw_jump = max(
                abs(np.degrees(yaw_delta(ly, prev_ly, options.left_yaw_interpolation))),
                abs(np.degrees(yaw_delta(ry, prev_ry, options.right_yaw_interpolation))),
            )
            score.max_yaw_jump_deg = max(score.max_yaw_jump_deg, float(yaw_jump))
            if yaw_jump > yaw_limit:
                score.yaw_jump_violations += 1
                bad = True
        prev_ly, prev_ry = ly, ry

        if bad and idx > 0:
            violation_indices.append(idx)

    regrasp = []
    for idx in violation_indices:
        if not regrasp or idx - regrasp[-1] >= 2:
            regrasp.append(idx)
        if len(regrasp) >= options.max_auto_regrasps:
            break
    score.recommended_regrasp_indices = regrasp
    return score


def globally_swapped_targets(targets: list[Target]) -> list[Target]:
    return [(r.copy(), l.copy(), ry, ly) for l, r, ly, ry in targets]


def choose_safe_endpoint_assignment(
    targets: list[Target],
    obstacles_3d: list[dict] | None,
    options: RobotAwareOptions,
) -> tuple[list[Target], str, RobotAwareScore]:
    fixed_score = score_waypoints(targets, obstacles_3d, options)
    swapped = globally_swapped_targets(targets)
    swapped_score = score_waypoints(swapped, obstacles_3d, options)
    if swapped_score.total_penalty() + 1e-9 < fixed_score.total_penalty():
        return swapped, "global_swap", swapped_score
    return targets, "fixed_csv", fixed_score


def _equality_ids(model: mujoco.MjModel) -> list[int]:
    ids = []
    for name in ("anchor_left_pin", "anchor_right_pin"):
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eid >= 0:
            ids.append(eid)
    return ids


def _set_welds_active(model: mujoco.MjModel, data: mujoco.MjData, active: bool) -> None:
    for eid in _equality_ids(model):
        data.eq_active[eid] = 1 if active else 0
    mujoco.mj_forward(model, data)


def _standby_targets() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([-0.56, -0.035, CABLE_Z], dtype=np.float64),
        np.array([0.56, -0.035, CABLE_Z], dtype=np.float64),
    )


_INITIAL_IK_SEEDS = [
    np.deg2rad([155.810869, -60.525940, 155.774429, -185.248488, -90.000000, -114.189131]),
    np.deg2rad([-155.810869, -60.525940, 155.774429, -185.248488, -90.000000, -114.189131]),
    np.deg2rad([-90.0, -90.0, 90.0, -90.0, -90.0, 0.0]),
    np.deg2rad([90.0, -90.0, 90.0, -90.0, -90.0, 0.0]),
    np.deg2rad([150.0, -60.0, 120.0, -150.0, -90.0, -114.0]),
    np.deg2rad([-150.0, -60.0, 120.0, -150.0, -90.0, -114.0]),
    np.deg2rad([120.0, -75.0, 115.0, -130.0, -90.0, -90.0]),
    np.deg2rad([-120.0, -75.0, 115.0, -130.0, -90.0, -90.0]),
]


def _joint_delta_norm(a: np.ndarray, b: np.ndarray) -> float:
    delta = (np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.dot(delta, delta))


def _tool_pose_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    target: np.ndarray,
    target_yaw: float,
) -> tuple[float, float]:
    pos_err = float(np.linalg.norm(np.asarray(target) - data.xpos[body_id]))
    R = data.xmat[body_id].reshape(3, 3)
    z_err = float(np.linalg.norm(0.5 * (np.array([0.0, 0.0, -1.0]) - R[:, 2])))
    x_target = np.array([float(np.cos(target_yaw)), float(np.sin(target_yaw)), 0.0])
    x_err = float(np.linalg.norm(0.5 * (x_target - R[:, 0])))
    return pos_err, max(z_err, x_err)


def _geom_body_has_prefix(model: mujoco.MjModel, geom_id: int, prefix: str) -> bool:
    body_id = int(model.geom_bodyid[int(geom_id)])
    while body_id > 0:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(prefix):
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _geom_robot_side(model: mujoco.MjModel, geom_id: int) -> str | None:
    if _geom_body_has_prefix(model, geom_id, "ur10_left/"):
        return "left"
    if _geom_body_has_prefix(model, geom_id, "ur10_right/"):
        return "right"
    return None


def _geom_is_obstacle(model: mujoco.MjModel, geom_id: int) -> bool:
    geom_name = _geom_name(model, geom_id)
    return (
        geom_name.startswith("gcr_obs_")
        or geom_name.startswith("obs_")
        or _geom_body_has_prefix(model, geom_id, "gcr_obs_")
        or _geom_body_has_prefix(model, geom_id, "obs_")
    )


def _robot_contact_counts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, int]:
    robot_robot = 0
    robot_obstacle = 0
    total = int(data.ncon)
    for i in range(total):
        contact = data.contact[i]
        side1 = _geom_robot_side(model, contact.geom1)
        side2 = _geom_robot_side(model, contact.geom2)
        if side1 is not None and side2 is not None and side1 != side2:
            robot_robot += 1
        if (
            (side1 is not None and _geom_is_obstacle(model, contact.geom2))
            or (side2 is not None and _geom_is_obstacle(model, contact.geom1))
        ):
            robot_obstacle += 1
    return robot_robot, robot_obstacle, total


def _solve_side_from_seed(
    ctrl: DualUR10Controller,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    target: np.ndarray,
    target_yaw: float,
    seed: np.ndarray,
    opts: IKSolveOptions,
) -> np.ndarray:
    qadr = ctrl._left_qadr if side == "left" else ctrl._right_qadr
    saved_q = data.qpos[qadr].copy()
    data.qpos[qadr] = np.asarray(seed, dtype=np.float64)
    try:
        return ctrl._solve_step(model, data, side, np.asarray(target), opts, target_yaw)
    finally:
        data.qpos[qadr] = saved_q


def _candidate_seeds(current: np.ndarray) -> list[np.ndarray]:
    seeds = [np.asarray(current, dtype=np.float64).copy()]
    seeds.extend(seed.copy() for seed in _INITIAL_IK_SEEDS)
    unique: list[np.ndarray] = []
    for seed in seeds:
        if not any(_joint_delta_norm(seed, old) < 1e-8 for old in unique):
            unique.append(seed)
    return unique


def _select_initial_joint_pair(
    ctrl: DualUR10Controller,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_l: np.ndarray,
    target_r: np.ndarray,
    left_yaw: float,
    right_yaw: float,
    opts: IKSolveOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    current_l = data.qpos[ctrl._left_qadr].copy()
    current_r = data.qpos[ctrl._right_qadr].copy()

    left_solutions: list[tuple[np.ndarray, float, float]] = []
    right_solutions: list[tuple[np.ndarray, float, float]] = []
    for side, target, yaw, current, out in (
        ("left", target_l, left_yaw, current_l, left_solutions),
        ("right", target_r, right_yaw, current_r, right_solutions),
    ):
        qadr = ctrl._left_qadr if side == "left" else ctrl._right_qadr
        body_id = ctrl._left_tip_id if side == "left" else ctrl._right_tip_id
        for seed in _candidate_seeds(current):
            q = _solve_side_from_seed(ctrl, model, data, side, target, yaw, seed, opts)
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            pos_err, ori_err = _tool_pose_error(model, data, body_id, target, yaw)
            if pos_err < 0.02 and ori_err < 0.08:
                out.append((q, pos_err, ori_err))
        if not out:
            # Fall back to the branch returned from the current pose so the
            # runner remains usable even for difficult IK targets.
            q = _solve_side_from_seed(ctrl, model, data, side, target, yaw, current, opts)
            out.append((q, float("inf"), float("inf")))

    best_score = float("inf")
    best_l = left_solutions[0][0]
    best_r = right_solutions[0][0]
    best_counts = (0, 0, 0)
    for q_l, l_pos_err, l_ori_err in left_solutions:
        for q_r, r_pos_err, r_ori_err in right_solutions:
            data.qpos[:] = saved_qpos
            data.qvel[:] = 0.0
            data.qpos[ctrl._left_qadr] = q_l
            data.qpos[ctrl._right_qadr] = q_r
            mujoco.mj_forward(model, data)
            robot_robot, robot_obstacle, total_contacts = _robot_contact_counts(model, data)
            score = (
                10000.0 * robot_robot
                + 2000.0 * robot_obstacle
                + 50.0 * total_contacts
                + 5000.0 * (l_pos_err + r_pos_err)
                + 50.0 * (l_ori_err + r_ori_err)
                + 0.01 * (_joint_delta_norm(q_l, current_l) + _joint_delta_norm(q_r, current_r))
            )
            if score < best_score:
                best_score = score
                best_l = q_l.copy()
                best_r = q_r.copy()
                best_counts = (robot_robot, robot_obstacle, total_contacts)

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)
    return best_l, best_r, {
        "left_candidates": len(left_solutions),
        "right_candidates": len(right_solutions),
        "score": best_score,
        "robot_robot_contacts": best_counts[0],
        "robot_obstacle_contacts": best_counts[1],
        "total_contacts": best_counts[2],
    }


def _slack_cable_between(left: np.ndarray, right: np.ndarray, cable_length: float, n_nodes: int) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    chord_vec = right - left
    chord = float(np.linalg.norm(chord_vec[:2]))
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if chord < 1e-9:
        xs = np.linspace(-0.5 * cable_length, 0.5 * cable_length, n_nodes)
        nodes = np.column_stack((left[0] + xs, np.full(n_nodes, left[1]), np.full(n_nodes, CABLE_Z)))
        return nodes

    direction = chord_vec[:2] / chord
    normal = np.array([-direction[1], direction[0]])
    s = np.linspace(0.0, 1.0, n_nodes)
    base_xy = left[:2] + np.outer(s, chord_vec[:2])

    if cable_length <= chord * 1.002:
        amp = 0.0
    else:
        lo, hi = 0.0, max(0.02, 0.5 * cable_length)
        for _ in range(40):
            amp = 0.5 * (lo + hi)
            xy = base_xy + np.outer(np.sin(np.pi * s) * amp, normal)
            length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            if length < cable_length:
                lo = amp
            else:
                hi = amp
        amp = hi
    xy = base_xy + np.outer(np.sin(np.pi * s) * amp, normal)
    return np.column_stack((xy[:, 0], xy[:, 1], np.full(n_nodes, CABLE_Z)))


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name or ""


def _update_runtime_contacts(model: mujoco.MjModel, data: mujoco.MjData, stats: RuntimeStats) -> None:
    for i in range(data.ncon):
        contact = data.contact[i]
        side1 = _geom_robot_side(model, contact.geom1)
        side2 = _geom_robot_side(model, contact.geom2)
        if side1 is not None and side2 is not None and side1 != side2:
            stats.robot_robot_contacts += 1
        if (
            (side1 is not None and _geom_is_obstacle(model, contact.geom2))
            or (side2 is not None and _geom_is_obstacle(model, contact.geom1))
        ):
            stats.robot_obstacle_contacts += 1


def execute_robot_aware_waypoints(
    targets_3d: list[Target],
    *,
    obstacles_3d: list[dict] | None = None,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    steps_per_wp: int = 300,
    use_viewer: bool = True,
    save_video: bool = False,
    video_path: Path | None = None,
    max_gripper_speed: float = DEFAULT_MAX_GRIPPER_SPEED,
    cable_length: float = CABLE_LENGTH,
    cable_num_segments: int = CABLE_NUM_SEGMENTS,
    cable_cstart_nodes: np.ndarray | None = None,
    cable_profile_nodes: list[np.ndarray] | None = None,
    yaw_mode: str = YAW_MODE_SIMULTANEOUS,
    options: RobotAwareOptions | None = None,
    manual_regrasp_indices: list[int] | None = None,
    trajectory_path: Path | None = None,
    trajectory_stride: int = 1,
    trajectory_metadata: dict[str, object] | None = None,
    final_error_wait: float = 0.0,
    final_error_json_path: Path | None = None,
) -> tuple[RobotAwareScore, RuntimeStats]:
    if not targets_3d:
        raise ValueError("empty targets_3d list")
    yaw_mode = _validate_yaw_mode(yaw_mode)
    options = options or RobotAwareOptions()
    if final_error_wait < 0.0:
        raise ValueError("final_error_wait must be non-negative")
    obstacles_3d = list(obstacles_3d or [])
    if workspace_bounds is None:
        workspace_bounds = (WS_3D_X, WS_3D_Y)
    options.ghost_target_shape = options.ghost_target_shape.strip().lower()
    if options.ghost_target_shape not in GHOST_TARGET_SHAPE_CHOICES:
        raise ValueError(
            "ghost_target_shape must be one of: "
            f"{', '.join(GHOST_TARGET_SHAPE_CHOICES)}"
        )
    if options.cable_attachment_mode is not None:
        attachment_mode = options.cable_attachment_mode.strip().lower()
        if attachment_mode not in {"site_weld", "two_point_clamp", "body_weld"}:
            raise ValueError(
                "cable_attachment_mode must be one of: "
                "site_weld, two_point_clamp, body_weld"
            )
        scene_build.CABLE_ENDPOINT_ATTACHMENT_MODE = attachment_mode
    if options.cable_weld_torquescale is not None:
        if options.cable_weld_torquescale < 0.0:
            raise ValueError("cable_weld_torquescale must be non-negative")
        scene_build.CABLE_ENDPOINT_WELD_TORQUESCALE = float(options.cable_weld_torquescale)

    pre_score = score_waypoints(targets_3d, obstacles_3d, options)
    regrasp_indices = set(manual_regrasp_indices or [])
    if options.auto_regrasp and pre_score.recommended_regrasp_indices:
        regrasp_indices.update(pre_score.recommended_regrasp_indices)
    regrasp_indices = {idx for idx in regrasp_indices if 0 < idx < len(targets_3d)}
    if regrasp_indices and not options.allow_regrasp:
        print(f"  Robot-aware: regrasp suggested at waypoint indices {sorted(regrasp_indices)}, but regrasp is disabled.")
        regrasp_indices = set()

    tl_start, tr_start, ly_start, ry_start = targets_3d[0]
    if cable_cstart_nodes is not None:
        cstart_nodes_3d = np.asarray(cable_cstart_nodes, dtype=np.float64)
        cstart_nodes_3d = _resample_polyline_nodes(cstart_nodes_3d, cable_num_segments)
        cable_length = _polyline_length(cstart_nodes_3d)
    else:
        cstart_nodes_3d = np.linspace(np.asarray(tl_start), np.asarray(tr_start), num=cable_num_segments + 1)
        cable_length = _polyline_length(cstart_nodes_3d)

    profile_nodes_3d: list[np.ndarray] | None = None
    if cable_profile_nodes is not None:
        if len(cable_profile_nodes) == len(targets_3d):
            profile_nodes_3d = [
                _resample_polyline_nodes(np.asarray(profile, dtype=np.float64), cable_num_segments)
                for profile in cable_profile_nodes
            ]
        else:
            print(
                "  WARNING: ignoring cable profile metadata because "
                f"{len(cable_profile_nodes)} profile(s) do not match {len(targets_3d)} waypoint(s)."
            )

    if options.ghost_target_shape != "none" and profile_nodes_3d is None:
        print("  Ghost target shape: unavailable (missing cable profile metadata).")
        ghost_profiles = None
    elif options.ghost_target_shape == "final" and profile_nodes_3d is not None:
        ghost_profiles = [profile_nodes_3d[-1]]
    elif options.ghost_target_shape == "all" and profile_nodes_3d is not None:
        ghost_profiles = profile_nodes_3d
    else:
        ghost_profiles = None

    print(f"  Robot-aware pre-score: {asdict(pre_score)}")
    print(f"  Regrasp indices: {sorted(regrasp_indices) if regrasp_indices else 'none'}")
    print(f"  Cable: length={cable_length:.6f} m, segments={cable_num_segments}, nodes={cable_num_segments + 1}")
    if profile_nodes_3d is not None and options.use_profile_yaw and yaw_mode == YAW_MODE_SIMULTANEOUS:
        print("  Yaw source: interpolated cable-profile endpoint tangents")
    elif profile_nodes_3d is not None:
        print("  Yaw source: unwrapped CSV cable endpoint tangents")
    print(
        "  Cable endpoint attachment: "
        f"{scene_build.CABLE_ENDPOINT_ATTACHMENT_MODE} "
        f"(weld_torquescale={scene_build.CABLE_ENDPOINT_WELD_TORQUESCALE:g})"
    )
    if options.ghost_target_shape != "none" and ghost_profiles is not None:
        print(
            "  Ghost target shape: "
            f"{options.ghost_target_shape} ({len(ghost_profiles)} profile(s), "
            f"rgba={options.ghost_target_rgba})"
        )
    print(f"  Building robot-aware MuJoCo scene with {len(obstacles_3d)} obstacle(s)...")

    model, data, _ = build_and_compile(
        obstacles=obstacles_3d,
        lock_robot_joints=False,
        ik_targets={"left": tl_start, "right": tr_start},
        cable_cstart=cstart_nodes_3d,
        cable_length=cable_length,
        cable_num_segments=cable_num_segments,
        workspace_bounds=workspace_bounds,
        ghost_cable_profiles=ghost_profiles,
        ghost_cable_rgba=options.ghost_target_rgba,
    )
    ctrl = DualUR10Controller(model)
    stats = RuntimeStats()
    frames: list = []
    trajectory_frames: list[dict[str, np.ndarray]] = []
    trajectory_stride = max(1, int(trajectory_stride))
    trajectory_step = 0

    def record_trajectory_frame() -> None:
        nonlocal trajectory_step
        if trajectory_path is None:
            return
        if trajectory_step % trajectory_stride == 0:
            trajectory_frames.append(
                {
                    "qpos": data.qpos.copy(),
                    "qvel": data.qvel.copy(),
                    "mocap_pos": data.mocap_pos.copy(),
                    "mocap_quat": data.mocap_quat.copy(),
                }
            )
        trajectory_step += 1

    def step_capture(renderer=None, target_l=None, target_r=None):
        update_anchor_mocaps(model, data)
        if options.cable_endpoint_tangent_assist:
            apply_cable_endpoint_tangent_assist(
                model,
                data,
                alpha=options.cable_endpoint_tangent_alpha,
                clamp_segments=options.cable_endpoint_clamp_segments,
                velocity_damping=options.cable_endpoint_tangent_velocity_damping,
            )
        mujoco.mj_step(model, data)
        record_trajectory_frame()
        _update_runtime_contacts(model, data, stats)
        if target_l is not None and target_r is not None:
            err = max(
                float(np.linalg.norm(ctrl.gripper_position(data, "left") - target_l)),
                float(np.linalg.norm(ctrl.gripper_position(data, "right") - target_r)),
            )
            stats.max_tracking_error = max(stats.max_tracking_error, err)
            if err > options.tracking_error_limit:
                stats.tracking_error_violations += 1
        if renderer is not None and save_video:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

    def settle_and_measure_final_error(viewer=None) -> None:
        if final_error_wait <= 0.0 and final_error_json_path is None:
            return
        if profile_nodes_3d is None:
            print("  Final cable error: skipped (missing cable profile metadata).")
            return
        wait_steps = int(round(final_error_wait / float(model.opt.timestep)))
        if wait_steps > 0:
            print(f"  Final cable error: settling for {final_error_wait:g} s before measuring...")
        for step_idx in range(max(0, wait_steps)):
            if viewer is not None and not viewer.is_running():
                break
            step_capture()
            if viewer is not None and step_idx % 20 == 0:
                viewer.sync()

        actual_nodes = _current_cable_points(model, data)
        target_nodes = _resample_polyline_nodes(profile_nodes_3d[-1], max(len(actual_nodes) - 1, 1))
        report = compute_cable_xy_error(actual_nodes, target_nodes, cable_length=cable_length)
        payload: dict[str, object] = {
            "wait_seconds": float(final_error_wait),
            "criteria": (
                "Primary metric is RMS 2D centerline error after resampling "
                "actual and target ordered cable nodes; z is ignored."
            ),
            "metrics": report,
            "actual_nodes_xyz": actual_nodes.tolist(),
            "target_nodes_xyz": target_nodes.tolist(),
        }
        print_cable_xy_error_report(report)
        if final_error_json_path is not None:
            final_error_json_path.parent.mkdir(parents=True, exist_ok=True)
            final_error_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  Final cable error saved -> {final_error_json_path}")

    def move_to(target_l, target_r, left_yaw, right_yaw, n_steps, viewer=None, renderer=None):
        start_l = ctrl.gripper_position(data, "left").copy()
        start_r = ctrl.gripper_position(data, "right").copy()
        for k in range(max(1, n_steps)):
            if viewer is not None and not viewer.is_running():
                return False
            t = _cosine_ease_step(k, max(1, n_steps))
            cur_l = start_l + t * (target_l - start_l)
            cur_r = start_r + t * (target_r - start_r)
            ctrl.set_targets(model, data, cur_l, cur_r, left_yaw=left_yaw, right_yaw=right_yaw)
            step_capture(renderer, cur_l, cur_r)
            if viewer is not None and k % 5 == 0:
                viewer.sync()
        return True

    settle_steps = 200

    def run(viewer=None, renderer=None):
        opts = IKSolveOptions(max_iter=300, ori_tol=0.01)
        if options.initial_pose_search:
            q_l, q_r, initial_report = _select_initial_joint_pair(
                ctrl,
                model,
                data,
                np.asarray(tl_start),
                np.asarray(tr_start),
                ly_start,
                ry_start,
                opts,
            )
            print(f"  Initial IK auto-clearance: {initial_report}")
            if int(initial_report.get("robot_robot_contacts", 0)) > 0:
                print("  WARNING: no collision-free initial IK branch found among the tested seeds.")
        else:
            q_l = ctrl._solve_step(model, data, "left", np.asarray(tl_start), opts, ly_start)
            q_r = ctrl._solve_step(model, data, "right", np.asarray(tr_start), opts, ry_start)
        data.qpos[ctrl._left_qadr] = q_l
        data.qpos[ctrl._right_qadr] = q_r
        data.qvel[ctrl._left_dadr] = 0.0
        data.qvel[ctrl._right_dadr] = 0.0
        mujoco.mj_forward(model, data)
        update_anchor_mocaps(model, data)
        mujoco.mj_forward(model, data)
        _set_weld_relpose_to_current_state(model, data)
        for j in range(len(q_l)):
            model.qpos_spring[ctrl._left_qadr[j]] = q_l[j]
            model.qpos_spring[ctrl._right_qadr[j]] = q_r[j]
            if ctrl._left_act[j] >= 0:
                data.ctrl[ctrl._left_act[j]] = q_l[j]
            if ctrl._right_act[j] >= 0:
                data.ctrl[ctrl._right_act[j]] = q_r[j]

        for k in range(settle_steps):
            if viewer is not None and not viewer.is_running():
                return
            data.qpos[ctrl._left_qadr] = q_l
            data.qpos[ctrl._right_qadr] = q_r
            data.qvel[ctrl._left_dadr] = 0.0
            data.qvel[ctrl._right_dadr] = 0.0
            mujoco.mj_forward(model, data)
            step_capture(renderer)
            if viewer is not None and k % 20 == 0:
                viewer.sync()

        prev_ly, prev_ry = ly_start, ry_start
        mutable_targets = list(targets_3d)
        mutable_profiles = [profile.copy() for profile in profile_nodes_3d] if profile_nodes_3d is not None else None
        for wp_idx in range(len(mutable_targets)):
            if viewer is not None and not viewer.is_running():
                return
            if wp_idx in regrasp_indices:
                print(f"  Regrasp before waypoint {wp_idx + 1}: detach -> standby -> reattach")
                stats.regrasp_events += 1
                _set_welds_active(model, data, False)
                standby_l, standby_r = _standby_targets()
                if not move_to(standby_l, standby_r, prev_ly, prev_ry, options.regrasp_steps, viewer, renderer):
                    return
                if options.swap_after_regrasp:
                    mutable_targets[wp_idx:] = globally_swapped_targets(mutable_targets[wp_idx:])
                    if mutable_profiles is not None:
                        mutable_profiles[wp_idx:] = [profile[::-1].copy() for profile in mutable_profiles[wp_idx:]]
                tl, tr, ly, ry = mutable_targets[wp_idx]
                if not move_to(np.asarray(tl), np.asarray(tr), ly, ry, options.regrasp_steps, viewer, renderer):
                    return
                cable_nodes = _slack_cable_between(np.asarray(tl), np.asarray(tr), cable_length, cable_num_segments + 1)
                set_cable_config(model, data, cable_nodes)
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                update_anchor_mocaps(model, data)
                _set_weld_relpose_to_current_state(model, data)
                _set_welds_active(model, data, True)
                prev_ly, prev_ry = ly, ry

            tl, tr, ly, ry = mutable_targets[wp_idx]
            target_l = np.asarray(tl, dtype=np.float64)
            target_r = np.asarray(tr, dtype=np.float64)
            prev_l = ctrl.gripper_position(data, "left").copy()
            prev_r = ctrl.gripper_position(data, "right").copy()
            n_steps = _steps_for_waypoint(
                prev_l,
                prev_r,
                target_l,
                target_r,
                dt=model.opt.timestep,
                base_steps=steps_per_wp,
                max_speed=max_gripper_speed,
            )
            print(
                f"  Waypoint {wp_idx + 1}/{len(mutable_targets)} "
                f"left=({tl[0]:.3f},{tl[1]:.3f}) right=({tr[0]:.3f},{tr[1]:.3f}) "
                f"steps={n_steps}"
            )
            ramp_steps = max(int(0.7 * n_steps), 1)
            dly = yaw_delta(ly, prev_ly, options.left_yaw_interpolation)
            dry = yaw_delta(ry, prev_ry, options.right_yaw_interpolation)
            if options.invert_red_anchor_yaw:
                dly = _invert_yaw_delta(dly)
            if options.tcp_schedule == "auto-stagger":
                waypoint_schedule, schedule_scores = _choose_tcp_schedule(
                    prev_l,
                    target_l,
                    prev_r,
                    target_r,
                    cable_length,
                    options,
                )
                if waypoint_schedule != "simultaneous":
                    p, min_tcp, min_arm = schedule_scores[waypoint_schedule]
                    print(
                        f"    TCP schedule: {waypoint_schedule} "
                        f"(min_tcp={min_tcp:.3f} m, min_arm={min_arm:.3f} m)"
                    )
            else:
                waypoint_schedule = options.tcp_schedule
            profile_yaw = (
                options.use_profile_yaw
                and mutable_profiles is not None
                and yaw_mode == YAW_MODE_SIMULTANEOUS
            )
            prev_profile = None
            target_profile = None
            if profile_yaw:
                prev_profile = mutable_profiles[wp_idx - 1] if wp_idx > 0 else mutable_profiles[0]
                target_profile = mutable_profiles[wp_idx]

            if yaw_mode == YAW_MODE_ROTATE_THEN_MOVE:
                yaw_steps = max(int(ROTATE_THEN_MOVE_YAW_FRACTION * n_steps), 1) if max(abs(dly), abs(dry)) > 1e-9 else 0
                for k in range(yaw_steps):
                    if viewer is not None and not viewer.is_running():
                        return
                    t = _cosine_ease_step(k, yaw_steps)
                    ctrl.set_targets(model, data, prev_l, prev_r, left_yaw=prev_ly + t * dly, right_yaw=prev_ry + t * dry)
                    step_capture(renderer, prev_l, prev_r)
                    if viewer is not None and k % 5 == 0:
                        viewer.sync()
                for k in range(n_steps):
                    if viewer is not None and not viewer.is_running():
                        return
                    t = _cosine_ease_step(k, ramp_steps) if k < ramp_steps else 1.0
                    tl_prog, tr_prog = _tcp_schedule_progress(
                        t,
                        waypoint_schedule,
                        prev_l,
                        target_l,
                        prev_r,
                        target_r,
                        cable_length,
                        options,
                    )
                    cur_l = prev_l + tl_prog * (target_l - prev_l)
                    cur_r = prev_r + tr_prog * (target_r - prev_r)
                    ctrl.set_targets(model, data, cur_l, cur_r, left_yaw=ly, right_yaw=ry)
                    step_capture(renderer, cur_l, cur_r)
                    if viewer is not None and k % 5 == 0:
                        viewer.sync()
            else:
                for k in range(n_steps):
                    if viewer is not None and not viewer.is_running():
                        return
                    t = _cosine_ease_step(k, ramp_steps) if k < ramp_steps else 1.0
                    tl_prog, tr_prog = _tcp_schedule_progress(
                        t,
                        waypoint_schedule,
                        prev_l,
                        target_l,
                        prev_r,
                        target_r,
                        cable_length,
                        options,
                    )
                    cur_l = prev_l + tl_prog * (target_l - prev_l)
                    cur_r = prev_r + tr_prog * (target_r - prev_r)
                    if profile_yaw and prev_profile is not None and target_profile is not None:
                        profile_progress = 0.5 * (tl_prog + tr_prog)
                        cur_profile = prev_profile + profile_progress * (target_profile - prev_profile)
                        raw_ly, raw_ry = _profile_endpoint_yaws(cur_profile)
                        cmd_ly = _equivalent_near(raw_ly, prev_ly + tl_prog * dly)
                        cmd_ry = _equivalent_near(raw_ry, prev_ry + tr_prog * dry)
                    else:
                        cmd_ly, cmd_ry = prev_ly + tl_prog * dly, prev_ry + tr_prog * dry
                    ctrl.set_targets(
                        model,
                        data,
                        cur_l,
                        cur_r,
                        left_yaw=cmd_ly,
                        right_yaw=cmd_ry,
                    )
                    step_capture(renderer, cur_l, cur_r)
                    if viewer is not None and k % 5 == 0:
                        viewer.sync()
            prev_ly, prev_ry = ly, ry

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            run(viewer=viewer)
            settle_and_measure_final_error(viewer=viewer)
            if viewer.is_running():
                print("  Execution complete - close the window to exit.")
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    update_anchor_mocaps(model, data)
                    viewer.sync()
    else:
        renderer = mujoco.Renderer(model, height=480, width=640) if save_video else None
        run(renderer=renderer)
        settle_and_measure_final_error()
        if save_video and frames and video_path is not None:
            try:
                import imageio

                imageio.mimwrite(str(video_path), frames, fps=30)
                print(f"  Video saved -> {video_path}")
            except (ImportError, ValueError) as e:
                print(f"  Video save skipped ({e})")

    if trajectory_path is not None:
        trajectory_path = Path(trajectory_path)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        if trajectory_frames:
            meta = dict(trajectory_metadata or {})
            np.savez_compressed(
                trajectory_path,
                qpos=np.asarray([frame["qpos"] for frame in trajectory_frames], dtype=np.float64),
                qvel=np.asarray([frame["qvel"] for frame in trajectory_frames], dtype=np.float64),
                mocap_pos=np.asarray([frame["mocap_pos"] for frame in trajectory_frames], dtype=np.float64),
                mocap_quat=np.asarray([frame["mocap_quat"] for frame in trajectory_frames], dtype=np.float64),
                timestep=np.asarray([float(model.opt.timestep) * trajectory_stride], dtype=np.float64),
                cable_length=np.asarray([float(cable_length)], dtype=np.float64),
                cable_num_segments=np.asarray([int(cable_num_segments)], dtype=np.int64),
                workspace_x=np.asarray(workspace_bounds[0], dtype=np.float64),
                workspace_y=np.asarray(workspace_bounds[1], dtype=np.float64),
                metadata_json=np.asarray([json.dumps(meta)], dtype=np.str_),
            )
            print(f"  Trajectory saved -> {trajectory_path} ({len(trajectory_frames)} frame(s))")
        else:
            print("  Trajectory save skipped: no frames were recorded.")

    print(f"  Robot-aware runtime stats: {asdict(stats)}")
    return pre_score, stats


__all__ = [
    "RobotAwareOptions",
    "RobotAwareScore",
    "RuntimeStats",
    "TCP_SCHEDULE_CHOICES",
    "YAW_MODE_CHOICES",
    "YAW_MODE_SIMULTANEOUS",
    "YAW_MODE_ROTATE_THEN_MOVE",
    "choose_safe_endpoint_assignment",
    "execute_robot_aware_waypoints",
    "score_waypoints",
]
