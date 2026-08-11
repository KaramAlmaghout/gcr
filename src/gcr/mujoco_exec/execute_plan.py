"""Execute a GCR-planned cable path with two Universal Robots UR10e in MuJoCo.

Pipeline
--------
1. GCR planner finds a 2-D path (list of cable configurations) for the
   chosen scenario.
2. The 2-D GCR workspace [0,1]×[0,0.6] m is linearly mapped to the 3-D
   MuJoCo scene:
       2D x → 3D x   via  x3 = -0.5 + x2 * 1.0   (range  -0.5 … +0.5 m)
       2D y → 3D y   via  y3 =  0.0 + y2 * 1.0   (range   0.0 … +0.6 m)
       3D z = CABLE_Z  (cable height)
3. Scenario obstacles (Circle / Rectangle in 2-D) are converted to visual
   MuJoCo cylinders / boxes in the 3-D scene.
4. The cable and robots are initialized directly at the GCR Cstart
   configuration — no warm-up drive needed.
5. The robots then follow each waypoint in the planned path.

Usage
-----
    cd <repository-root>
    python -m gcr execute-plan --scenario 1

Options
-------
    --scenario N               1-based planner scenario to run (default: 1)
    --nodes N                  cable nodes used to build the scenario (default: 10)
    --max-iterations N         GCR planner iteration cap (default: 1600)
    --scenario-time-limit S    GCR planner time limit in seconds (default: 120)
    --steps-per-waypoint N     simulation steps per waypoint (default: 300)
    --no-viewer                headless mode, saves video to results/figures/
    --seed N                   RNG seed for the planner (default: 42)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

from gcr.planner import gcrbup as gcr
import gcr.sim.build_scene as scene_build
from gcr.sim.build_scene import (
    build_and_compile, update_anchor_mocaps, CABLE_Z,
    UR10_LEFT_BASE_POS, UR10_RIGHT_BASE_POS,
    CABLE_LENGTH, CABLE_NUM_SEGMENTS,
    apply_cable_endpoint_tangent_assist,
    _set_weld_relpose_to_current_state,
)
from gcr.sim.controller import DualUR10Controller, IKSolveOptions

# ── Workspace mapping ─────────────────────────────────────────────────────────
# GCR 2-D workspace [0,1]×[0,0.6] mapped to 3-D MuJoCo workspace.
# Compact span keeps all cable endpoints within ≈0.73 m of each floor-mounted
# robot base (±0.45 m, y=0.20 m), cable at z=0.01.  All corners verified < 0.75 m.
WS_2D_X = (0.0, 1.0)
WS_2D_Y = (0.0, 0.60)
# Isotropic 1:1 mapping (SX = SY = 1.0): the 3-D cable is the same physical
# length as the planner's 2-D cable in every direction, so set_cable_config
# places the cable exactly on the GCR Cstart nodes with no length residual.
# Robot bases are at ±0.75 m to keep all four scenarios reachable for UR10e.
WS_3D_X = (-0.50, 0.50)   # 1.00 m span in x  (SX = 1.00)
WS_3D_Y = ( 0.00, 0.60)   # 0.60 m span in y  (SY = 1.00)

# Pre-compute scale factors once (used in map_2d_to_3d and _cable_yaw_3d).
_SX = (WS_3D_X[1] - WS_3D_X[0]) / (WS_2D_X[1] - WS_2D_X[0])
_SY = (WS_3D_Y[1] - WS_3D_Y[0]) / (WS_2D_Y[1] - WS_2D_Y[0])


def map_2d_to_3d(p2d: np.ndarray) -> np.ndarray:
    """Map a single 2-D GCR point (x, y) to a 3-D MuJoCo position."""
    x3 = WS_3D_X[0] + (float(p2d[0]) - WS_2D_X[0]) * _SX
    y3 = WS_3D_Y[0] + (float(p2d[1]) - WS_2D_Y[0]) * _SY
    return np.array([x3, y3, CABLE_Z])


def _cable_yaw_3d(C: np.ndarray, at_start: bool) -> float:
    """Yaw (rad) of the cable tangent at an endpoint, in 3-D world frame.

    The tangent is computed from the two outermost 2-D cable points and then
    scaled by the workspace mapping (sx, sy may differ → tangent angle shifts).
    ``at_start=True`` gives the left-endpoint yaw, ``False`` the right-endpoint.
    """
    if at_start:
        dx2 = float(C[1, 0] - C[0, 0])
        dy2 = float(C[1, 1] - C[0, 1])
    else:
        # Tangent pointing from right endpoint back into cable interior
        dx2 = float(C[-2, 0] - C[-1, 0])
        dy2 = float(C[-2, 1] - C[-1, 1])
    return float(np.arctan2(_SY * dy2, _SX * dx2))


def _cable_length_2d(C: np.ndarray) -> float:
    """Return the 2-D cable polyline length."""
    C = np.asarray(C, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1)))


def _wrap_pi_angle(angle: float) -> float:
    """Wrap an angle in radians to ``(-pi, pi]``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _unwrap_yaw_with_chord_guide(raw_yaw: float, previous_yaw: float, chord_delta: float) -> float:
    """Unwrap endpoint yaw, resolving near-180 flips with cable-chord motion."""
    delta = _wrap_pi_angle(raw_yaw - previous_yaw)
    near_half_turn = abs(abs(delta) - np.pi) <= np.deg2rad(20.0)
    if near_half_turn and abs(chord_delta) > 1e-6:
        if chord_delta < 0.0 and delta > 0.0:
            delta -= 2.0 * np.pi
        elif chord_delta > 0.0 and delta < 0.0:
            delta += 2.0 * np.pi
    return float(previous_yaw + delta)


# ── Obstacle conversion ───────────────────────────────────────────────────────

def _gcr_obstacles_to_mujoco(obstacles) -> list[dict]:
    """Convert GCR 2-D obstacles to MuJoCo obstacle dicts.

    Circles become cylinders. Polygons and rectangles become exact extruded
    polygon prisms with the same xy footprint after workspace mapping.
    """
    # Obstacles extend from the floor to CABLE_Z, so their top reaches the
    # cable centerline and can physically interact with the cable.
    _HH = 0.5 * CABLE_Z
    _C_CIRCLE = "0.80 0.20 0.20 0.85"
    _C_POLY   = "0.20 0.20 0.20 0.90"
    result = []

    if hasattr(obstacles, "circles") or hasattr(obstacles, "polygons"):
        idx = 0
        for center, radius in getattr(obstacles, "circles", []):
            cx3 = WS_3D_X[0] + (float(center[0]) - WS_2D_X[0]) * _SX
            cy3 = WS_3D_Y[0] + (float(center[1]) - WS_2D_Y[0]) * _SY
            result.append(dict(
                name=f"gcr_obs_{idx}",
                pos=(cx3, cy3, 0.0),
                radius=float(radius) * max(_SX, _SY),
                half_height=_HH,
                rgba=_C_CIRCLE,
            ))
            idx += 1
        for polygon in getattr(obstacles, "polygons", []):
            verts = np.asarray(polygon, dtype=float)
            verts3 = np.column_stack((
                WS_3D_X[0] + (verts[:, 0] - WS_2D_X[0]) * _SX,
                WS_3D_Y[0] + (verts[:, 1] - WS_2D_Y[0]) * _SY,
            ))
            result.append(dict(
                name=f"gcr_obs_{idx}",
                vertices=verts3,
                height=CABLE_Z,
                rgba=_C_POLY,
            ))
            idx += 1
    return result


# ── Endpoint-to-robot assignment ──────────────────────────────────────────────

def _assign_endpoints(
    path: list[np.ndarray],
    *,
    endpoint_mode: str = "fixed_start",
    left_yaw_sign: float = 1.0,
    right_yaw_sign: float = 1.0,
    swap_waypoint_indices: set[int] | None = None,
    smooth_symmetric_yaw: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Assign each cable endpoint to its nearest robot base and compute yaws.

    Returns a list of (left_pos, right_pos, left_yaw, right_yaw) for every
    waypoint.  The left/right assignment is fixed from Cstart so robots never
    swap sides mid-path.

    Yaw is the cable-tangent direction at each endpoint in the 3-D world frame
    (accounts for workspace-mapping anisotropy).  The IK will orient the
    gripper x-axis along this direction while keeping the z-axis pointing down.
    """
    e0 = map_2d_to_3d(path[0][0])
    e1 = map_2d_to_3d(path[0][-1])
    lb = np.array(UR10_LEFT_BASE_POS[:2])
    rb = np.array(UR10_RIGHT_BASE_POS[:2])
    natural = (np.linalg.norm(e0[:2] - lb) + np.linalg.norm(e1[:2] - rb))
    swapped = (np.linalg.norm(e1[:2] - lb) + np.linalg.norm(e0[:2] - rb))
    use_natural = natural <= swapped

    result = []
    swap_waypoint_indices = set(swap_waypoint_indices or ())
    for wp_idx, C in enumerate(path):
        # Attach the grippers to the cable's actual outer endpoints.  The
        # rightmost MuJoCo cable body origin is the inner node of the last
        # capsule, so the right weld stores a nonzero relpos from that body
        # origin to this outer endpoint.
        p0, p1 = map_2d_to_3d(C[0]), map_2d_to_3d(C[-1])
        y0, y1 = _cable_yaw_3d(C, True), _cable_yaw_3d(C, False)
        if endpoint_mode == "x_order":
            if p0[0] <= p1[0]:
                lp, rp, ly, ry = p0, p1, y0, y1
            else:
                lp, rp, ly, ry = p1, p0, y1, y0
        elif endpoint_mode == "fixed_start":
            if use_natural:
                lp, rp, ly, ry = p0, p1, y0, y1
            else:
                lp, rp, ly, ry = p1, p0, y1, y0
        else:
            raise ValueError(f"Unknown endpoint_mode {endpoint_mode!r}")
        if wp_idx in swap_waypoint_indices:
            lp, rp, ly, ry = rp, lp, ry, ly
        result.append((lp, rp, left_yaw_sign * ly, right_yaw_sign * ry))

    if smooth_symmetric_yaw and result:
        smoothed = []
        prev_lp = result[0][0]
        prev_rp = result[0][1]
        prev_l = result[0][2]
        prev_r = result[0][3]
        smoothed.append(result[0])
        for lp, rp, ly, ry in result[1:]:
            # Preserve the directed cable-segment yaw from the planned cable
            # spline. Only unwrap by 2*pi for continuity; never use
            # theta +/- pi as an equivalent, because that can hide a real
            # endpoint segment reversal and twist the welded cable in MuJoCo.
            prev_chord = float(np.arctan2(prev_rp[1] - prev_lp[1], prev_rp[0] - prev_lp[0]))
            curr_chord = float(np.arctan2(rp[1] - lp[1], rp[0] - lp[0]))
            chord_delta = _wrap_pi_angle(curr_chord - prev_chord)
            ly_best = _unwrap_yaw_with_chord_guide(ly, prev_l, chord_delta)
            ry_best = _unwrap_yaw_with_chord_guide(ry, prev_r, chord_delta)
            smoothed.append((lp, rp, ly_best, ry_best))
            prev_lp, prev_rp = lp, rp
            prev_l, prev_r = ly_best, ry_best
        result = smoothed
    return result


def _execution_options_for_scenario(scenario_name: str) -> dict:
    """Scenario-specific MuJoCo execution corrections.

    These do not change the GCR path. They choose how the two robots hold
    the exported cable endpoints and which yaw equivalent is used for the
    gripper frame when the physical gripper would otherwise rotate backward.
    """
    options: dict = {}
    if scenario_name == "scenario_8_two_shelves":
        options.update(left_yaw_sign=-1.0)
    return options


# ── GCR planning ──────────────────────────────────────────────────────────────

def plan_path(
    scenario: gcr.Scenario,
    seed: int = 42,
    max_iterations: int = 1600,
    scenario_time_limit_s: float | None = 120.0,
    shortcut_refinement: bool = True,
) -> list[np.ndarray] | None:
    """Run GCR planner, return list of 2-D cable configs or None on failure."""
    scenario_for_plan = replace(scenario, seed=seed)
    params = gcr.PlannerParams(
        max_iterations=max_iterations,
        scenario_time_limit_s=scenario_time_limit_s,
        shortcut_refinement=shortcut_refinement,
    )
    planner = gcr.GCRPlanner(scenario_for_plan, params)
    result = planner.plan()
    path = result.path if result.path else result.raw_path
    status = "SUCCESS" if result.success else "TIMEOUT" if result.timed_out else "FAIL"
    print(
        f"  GCR {status}  iter={result.iterations}  "
        f"t={result.elapsed_s:.2f}s  waypoints={len(path) if result.success else 0}"
    )
    return path if result.success else None


# ── MuJoCo execution ──────────────────────────────────────────────────────────

# Peak gripper velocity cap.  At 0.10 m/s, the cable's mass-inertia keeps
# up with peak cable-vs-gripper lag ≈ 7-10 mm — visually smooth.  The
# velocity cap stretches each waypoint's duration based on its move
# distance, so short moves stay quick and only long moves slow down.
DEFAULT_MAX_GRIPPER_SPEED = 0.20  # m/s
YAW_MODE_SIMULTANEOUS = "simultaneous"
YAW_MODE_ROTATE_THEN_MOVE = "rotate-then-move"
YAW_MODE_CHOICES = (YAW_MODE_SIMULTANEOUS, YAW_MODE_ROTATE_THEN_MOVE)
ROTATE_THEN_MOVE_YAW_FRACTION = 0.35
TCP_SCHEDULE_CHOICES = ("simultaneous", "auto-stagger", "left-first", "right-first")
SIDE_CORRECTION_MOVING_ROBOT_CHOICES = ("farther-from-target", "left", "right")
GHOST_TARGET_SHAPE_CHOICES = ("none", "final", "all")
DEFAULT_TCP_STAGGER_FRACTION = 0.35
DEFAULT_TCP_MIN_DISTANCE = 0.11
DEFAULT_ARM_SEGMENT_MIN_DISTANCE = 0.13
DEFAULT_CABLE_LENGTH_MARGIN = 0.003
DEFAULT_CABLE_SEGMENT_MULTIPLIER = 4


def _wrap_pi(angle: float) -> float:
    """Wrap an angle in radians to ``(-pi, pi]`` — used to compute the
    shortest-path delta between two yaw values so the interpolation
    doesn't take the long way around (e.g. 170° → -170° interpolates
    through 180°, not through 0°)."""
    return _wrap_pi_angle(angle)


def _invert_yaw_delta(delta: float) -> float:
    """Use the opposite equivalent yaw winding while keeping the same final pose."""
    delta = float(delta)
    if abs(delta) <= 1e-9:
        return 0.0
    return delta - np.sign(delta) * 2.0 * np.pi


def _validate_yaw_mode(yaw_mode: str) -> str:
    if yaw_mode not in YAW_MODE_CHOICES:
        raise ValueError(f"yaw_mode must be one of {YAW_MODE_CHOICES}, got {yaw_mode!r}")
    return yaw_mode


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

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
        return 0.0
    return min(point_seg(a, c, d), point_seg(b, c, d), point_seg(c, a, b), point_seg(d, a, b))


def _raw_tcp_schedule_progress(
    t: float,
    schedule: str,
    stagger_fraction: float,
) -> tuple[float, float]:
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
    if max_tcp_distance <= 0.0:
        return left_progress, right_progress
    if _tcp_distance_for_progress(left_start, left_target, right_start, right_target, left_progress, right_progress) <= max_tcp_distance:
        return left_progress, right_progress

    follower = "right" if schedule == "left-first" else "left" if schedule == "right-first" else ""
    if not follower:
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
    *,
    stagger_fraction: float = DEFAULT_TCP_STAGGER_FRACTION,
    cable_length_margin: float = DEFAULT_CABLE_LENGTH_MARGIN,
) -> tuple[float, float]:
    left_progress, right_progress = _raw_tcp_schedule_progress(t, schedule, stagger_fraction)
    max_tcp_distance = max(0.0, float(cable_length) - float(cable_length_margin))
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
    *,
    stagger_fraction: float,
    cable_length_margin: float,
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
            stagger_fraction=stagger_fraction,
            cable_length_margin=cable_length_margin,
        )
        left = _tcp_pose(left_start, left_target, lp)
        right = _tcp_pose(right_start, right_target, rp)
        tcp_dist = float(np.linalg.norm(left - right))
        min_tcp = min(min_tcp, tcp_dist)
        min_arm = min(min_arm, _segment_distance_2d(lb, left[:2], rb, right[:2]))
        max_excess = max(max_excess, tcp_dist - max(0.0, float(cable_length) - float(cable_length_margin)))
    penalty = (
        10000.0 * max(0.0, max_excess)
        + 1000.0 * max(0.0, DEFAULT_TCP_MIN_DISTANCE - min_tcp)
        + 700.0 * max(0.0, DEFAULT_ARM_SEGMENT_MIN_DISTANCE - min_arm)
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
    *,
    stagger_fraction: float,
    cable_length_margin: float,
) -> tuple[str, dict[str, tuple[float, float, float]]]:
    scores = {
        schedule: _tcp_schedule_score(
            schedule,
            left_start,
            left_target,
            right_start,
            right_target,
            cable_length,
            stagger_fraction=stagger_fraction,
            cable_length_margin=cable_length_margin,
        )
        for schedule in ("simultaneous", "left-first", "right-first")
    }
    best = min(scores, key=lambda name: (scores[name][0], 0 if name == "simultaneous" else 1))
    return best, scores


def _cosine_ease_step(k: int, ramp_steps: int) -> float:
    return float(0.5 - 0.5 * np.cos(np.pi * (k + 1) / ramp_steps))


def _polyline_length(nodes_3d: np.ndarray) -> float:
    nodes_3d = np.asarray(nodes_3d, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(nodes_3d[:, :3], axis=0), axis=1)))


def _resample_polyline_nodes(nodes_3d: np.ndarray, target_segments: int) -> np.ndarray:
    """Resample a 3-D cable polyline to a target segment count.

    This preserves the same endpoints and arclength-parametrized shape while
    changing only the MuJoCo discretization density.
    """
    nodes_3d = np.asarray(nodes_3d, dtype=np.float64)
    if nodes_3d.ndim != 2 or nodes_3d.shape[1] != 3 or len(nodes_3d) < 2:
        raise ValueError("nodes_3d must be an (N, 3) array with N >= 2")
    if target_segments < 1:
        raise ValueError("target_segments must be at least 1")
    if target_segments == len(nodes_3d) - 1:
        return nodes_3d.copy()

    seg_lengths = np.linalg.norm(np.diff(nodes_3d, axis=0), axis=1)
    keep = np.concatenate(([True], seg_lengths > 1e-12))
    nodes_3d = nodes_3d[keep]
    if len(nodes_3d) < 2:
        raise ValueError("cannot resample a zero-length cable")

    cumulative = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(nodes_3d, axis=0), axis=1)),
    ))
    samples = np.linspace(0.0, float(cumulative[-1]), target_segments + 1)
    return np.column_stack([
        np.interp(samples, cumulative, nodes_3d[:, axis])
        for axis in range(3)
    ])


def _steps_for_waypoint(
    prev_l: np.ndarray,
    prev_r: np.ndarray,
    target_l: np.ndarray,
    target_r: np.ndarray,
    *,
    dt: float,
    base_steps: int,
    max_speed: float,
) -> int:
    """Return number of MuJoCo steps for one waypoint such that the peak
    cosine-ramp velocity stays at or below ``max_speed`` (m/s).

    Cosine ease-in/out over the first 70 % of the waypoint reaches a peak
    velocity of  π · distance / (1.4 · n_total · dt)  at its midpoint.
    Solving for n_total = π · distance / (1.4 · v_max · dt) and taking the
    larger of (base_steps, required) so short moves still respect the
    user-requested minimum duration.
    """
    if max_speed <= 0:
        return base_steps
    move = float(max(
        np.linalg.norm(np.asarray(target_l) - np.asarray(prev_l)),
        np.linalg.norm(np.asarray(target_r) - np.asarray(prev_r)),
    ))
    required = int(np.ceil(np.pi * move / (1.4 * max_speed * dt)))
    return max(base_steps, required)


def _lerp_targets(
    ctrl: DualUR10Controller,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_left: np.ndarray,
    target_right: np.ndarray,
    n_steps: int,
    left_yaw: float | None = None,
    right_yaw: float | None = None,
) -> None:
    """Linearly interpolate both grippers to new targets over n_steps steps."""
    start_l = ctrl.gripper_position(data, "left").copy()
    start_r = ctrl.gripper_position(data, "right").copy()
    for k in range(n_steps):
        t = (k + 1) / n_steps
        tl = start_l + t * (target_left  - start_l)
        tr = start_r + t * (target_right - start_r)
        ctrl.set_targets(model, data, tl, tr,
                         left_yaw=left_yaw, right_yaw=right_yaw)
        update_anchor_mocaps(model, data)
        mujoco.mj_step(model, data)


def compute_signed_side(
    line_start: np.ndarray,
    line_end: np.ndarray,
    point: np.ndarray,
    epsilon: float,
) -> int:
    """Return the robust 2-D side of ``point`` relative to a directed line.

    The sign is based on cross(line_end - line_start, point - line_start).
    Points within ``epsilon`` of the line return 0 so small numerical motion
    around the endpoint line does not start/stop correction repeatedly.
    """
    p0 = np.asarray(line_start, dtype=np.float64)[:2]
    pn = np.asarray(line_end, dtype=np.float64)[:2]
    p = np.asarray(point, dtype=np.float64)[:2]
    direction = pn - p0
    rel = p - p0
    signed_area = float(direction[0] * rel[1] - direction[1] * rel[0])
    if abs(signed_area) <= epsilon:
        return 0
    return 1 if signed_area > 0.0 else -1


def are_cable_points_on_correct_side(
    current_config: np.ndarray,
    target_config: np.ndarray,
    epsilon: float,
) -> tuple[bool, list[int]]:
    """Check whether each current cable node is on its target side.

    The reference side is computed from the target cable profile relative to
    its target endpoint line.  The current side is computed relative to the
    current endpoint line.  Target points lying on the line are ignored, and
    current points on the line are accepted to avoid oscillation near the
    switching boundary.
    """
    current = np.asarray(current_config, dtype=np.float64)
    target = np.asarray(target_config, dtype=np.float64)
    n = min(len(current), len(target))
    if n < 3:
        return True, []

    current_p0, current_pn = current[0], current[n - 1]
    target_p0, target_pn = target[0], target[n - 1]
    wrong_indices: list[int] = []
    for i in range(1, n - 1):
        target_side = compute_signed_side(target_p0, target_pn, target[i], epsilon)
        if target_side == 0:
            continue
        current_side = compute_signed_side(current_p0, current_pn, current[i], epsilon)
        if current_side != 0 and current_side != target_side:
            wrong_indices.append(i)
    return len(wrong_indices) == 0, wrong_indices


def distance_between_robots(robot1_pos: np.ndarray, robot2_pos: np.ndarray) -> float:
    """Return Euclidean distance between the two gripper/TCP positions."""
    return float(
        np.linalg.norm(
            np.asarray(robot1_pos, dtype=np.float64)
            - np.asarray(robot2_pos, dtype=np.float64)
        )
    )


def should_switch_moving_robot(
    robot_distance: float,
    cable_length: float,
    margin: float = 0.003,
) -> bool:
    """Return True when the moving TCP should stop to avoid over-stretching."""
    return robot_distance >= max(0.0, float(cable_length) - float(margin))


def _current_cable_points(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Read ordered cable body positions from the compiled MuJoCo model."""
    points: list[np.ndarray] = []
    first_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cableS_first")
    if first_site >= 0:
        points.append(data.site_xpos[first_site].copy())
    else:
        first_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_first")
        if first_bid >= 0:
            points.append(data.xpos[first_bid].copy())

    idx = 1
    while True:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cableB_{idx}")
        if bid < 0:
            break
        points.append(data.xpos[bid].copy())
        idx += 1

    last_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_last")
    if last_bid >= 0:
        points.append(data.xpos[last_bid].copy())
    last_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cableS_last")
    if last_site >= 0:
        points.append(data.site_xpos[last_site].copy())
    return np.asarray(points, dtype=np.float64)


def compute_cable_xy_error(
    actual_config: np.ndarray,
    target_config: np.ndarray,
    *,
    cable_length: float | None = None,
) -> dict[str, float | int]:
    """Return ordered centerline error between actual and target cable profiles.

    The comparison is done in x-y only because the planner is 2-D and small z
    deviations mostly measure MuJoCo settling/contact, not planning quality.
    Both centerlines are resampled to the same node count before computing the
    pointwise distances, preserving endpoint correspondence and cable order.
    """
    actual = np.asarray(actual_config, dtype=np.float64)
    target = np.asarray(target_config, dtype=np.float64)
    if actual.ndim != 2 or target.ndim != 2 or actual.shape[1] < 2 or target.shape[1] < 2:
        raise ValueError("actual_config and target_config must be (N, >=2) arrays")
    if len(actual) < 2 or len(target) < 2:
        raise ValueError("actual_config and target_config must contain at least two nodes")

    n_nodes = max(len(actual), len(target))
    if len(actual) != n_nodes:
        actual = _resample_polyline_nodes(actual[:, :3], n_nodes - 1)
    if len(target) != n_nodes:
        target = _resample_polyline_nodes(target[:, :3], n_nodes - 1)

    distances = np.linalg.norm(actual[:, :2] - target[:, :2], axis=1)
    actual_len_xy = float(np.sum(np.linalg.norm(np.diff(actual[:, :2], axis=0), axis=1)))
    target_len_xy = float(np.sum(np.linalg.norm(np.diff(target[:, :2], axis=0), axis=1)))
    length_scale = float(cable_length) if cable_length is not None and cable_length > 0.0 else max(target_len_xy, 1e-12)
    return {
        "nodes": int(n_nodes),
        "mean_xy_error_m": float(np.mean(distances)),
        "rms_xy_error_m": float(np.sqrt(np.mean(distances * distances))),
        "max_xy_error_m": float(np.max(distances)),
        "p95_xy_error_m": float(np.percentile(distances, 95.0)),
        "left_endpoint_xy_error_m": float(distances[0]),
        "right_endpoint_xy_error_m": float(distances[-1]),
        "actual_length_xy_m": actual_len_xy,
        "target_length_xy_m": target_len_xy,
        "length_xy_error_m": float(actual_len_xy - target_len_xy),
        "normalized_rms_xy_error": float(np.sqrt(np.mean(distances * distances)) / length_scale),
    }


def print_cable_xy_error_report(report: dict[str, float | int], *, prefix: str = "  ") -> None:
    """Print the final cable shape error in a compact, readable form."""
    print(prefix + "Final cable-vs-goal error (x-y only):")
    print(
        prefix
        + "  RMS={rms_xy_error_m:.4f} m, mean={mean_xy_error_m:.4f} m, "
        "p95={p95_xy_error_m:.4f} m, max={max_xy_error_m:.4f} m".format(**report)
    )
    print(
        prefix
        + "  endpoints: left={left_endpoint_xy_error_m:.4f} m, "
        "right={right_endpoint_xy_error_m:.4f} m, normalized_RMS={normalized_rms_xy_error:.4f}".format(**report)
    )


def execute_in_mujoco(
    scenario,
    path: list[np.ndarray],
    steps_per_wp: int = 300,
    use_viewer: bool = True,
    save_video: bool = False,
    video_path: Path | None = None,
    max_gripper_speed: float = DEFAULT_MAX_GRIPPER_SPEED,
    yaw_mode: str = YAW_MODE_SIMULTANEOUS,
    cable_segment_multiplier: int = DEFAULT_CABLE_SEGMENT_MULTIPLIER,
    cable_endpoint_tangent_assist: bool = False,
    cable_endpoint_tangent_alpha: float = 0.10,
    cable_endpoint_clamp_segments: int = 3,
    cable_endpoint_tangent_velocity_damping: float = 0.0,
    cable_attachment_mode: str | None = None,
    cable_weld_torquescale: float | None = None,
    tcp_schedule: str = "simultaneous",
    tcp_stagger_fraction: float = DEFAULT_TCP_STAGGER_FRACTION,
) -> None:
    """Build the MuJoCo scene and execute the planned cable path.

    The scene is built with the cable and robots pre-initialized at the GCR
    Cstart configuration, so no warm-up drive is needed.  A short settle phase
    lets the cable physics stabilize, then each waypoint in the planned path
    is tracked.
    """
    yaw_mode = _validate_yaw_mode(yaw_mode)
    if cable_segment_multiplier < 1:
        raise ValueError("cable_segment_multiplier must be at least 1")
    mj_obs = _gcr_obstacles_to_mujoco(scenario.obstacles)

    # Compute 3-D targets for every waypoint first, so we can pass the Cstart
    # endpoints to build_and_compile as IK targets.  This initialises the robots
    # directly at Cstart — no warm-up drive needed.
    targets_3d = _assign_endpoints(path, **_execution_options_for_scenario(scenario.name))
    n_wp = len(targets_3d)
    tl_start, tr_start, ly_start, ry_start = targets_3d[0]

    print(f"  Building MuJoCo scene with {len(mj_obs)} obstacles…")
    # Map every Cstart node to 3-D so the cable starts in the planner's
    # initial shape — both grippers AND the cable line up at t=0, so the
    # cable-to-gripper welds are zero-residual on both ends.
    planner_cstart_nodes_3d = np.asarray(
        [map_2d_to_3d(p) for p in path[0]], dtype=np.float64
    )
    planner_segments = len(path[0]) - 1
    cable_num_segments = planner_segments * cable_segment_multiplier
    cstart_nodes_3d = _resample_polyline_nodes(planner_cstart_nodes_3d, cable_num_segments)
    cable_length = _polyline_length(cstart_nodes_3d)
    print(
        f"  Cable: length={cable_length:.6f} m, "
        f"planner_segments={planner_segments}, sim_segments={cable_num_segments}"
    )
    model, data, _ = build_and_compile(
        obstacles=mj_obs, lock_robot_joints=False,
        ik_targets={"left": tl_start, "right": tr_start},
        cable_cstart=cstart_nodes_3d,
        cable_length=cable_length,
        cable_num_segments=cable_num_segments,
        workspace_bounds=(WS_3D_X, WS_3D_Y),
    )
    ctrl = DualUR10Controller(model)
    frames = []

    def _step_and_maybe_capture(renderer=None):
        update_anchor_mocaps(model, data)
        if cable_endpoint_tangent_assist:
            apply_cable_endpoint_tangent_assist(
                model,
                data,
                alpha=cable_endpoint_tangent_alpha,
                clamp_segments=cable_endpoint_clamp_segments,
                velocity_damping=cable_endpoint_tangent_velocity_damping,
            )
        mujoco.mj_step(model, data)
        if renderer is not None and save_video:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

    # Print the (x, y, yaw) table before running.
    print(f"\n  {'WP':>3}  {'Left x':>8} {'Left y':>8} {'Left yaw°':>10}  "
          f"{'Right x':>8} {'Right y':>8} {'Right yaw°':>10}")
    print("  " + "-" * 67)
    for wp_idx, (tl, tr, ly, ry) in enumerate(targets_3d):
        print(f"  {wp_idx+1:>3}  {tl[0]:>8.3f} {tl[1]:>8.3f} {np.degrees(ly):>10.1f}°"
              f"  {tr[0]:>8.3f} {tr[1]:>8.3f} {np.degrees(ry):>10.1f}°")
    print()
    print(f"  Yaw motion mode: {yaw_mode}\n")

    _settle_steps = 200  # hold still; let cable reach natural catenary before motion

    def _run(viewer=None, renderer=None):
        # ── Kinematic settle: pin robot joints at Cstart, let cable deform ──
        # The UR10e has no gravity compensation, so the actuator alone would
        # let the gripper sag ~0.5mm/step.  Kinematic pinning keeps the
        # gripper exactly at the IK target while cable physics run normally.
        opts = IKSolveOptions(max_iter=300, ori_tol=0.01)
        q_l = ctrl._solve_step(model, data, 'left',
                               np.asarray(tl_start, dtype=np.float64), opts, ly_start)
        q_r = ctrl._solve_step(model, data, 'right',
                               np.asarray(tr_start, dtype=np.float64), opts, ry_start)
        data.qpos[ctrl._left_qadr] = q_l
        data.qpos[ctrl._right_qadr] = q_r
        data.qvel[ctrl._left_dadr] = 0.0
        data.qvel[ctrl._right_dadr] = 0.0
        mujoco.mj_forward(model, data)
        update_anchor_mocaps(model, data)
        mujoco.mj_forward(model, data)
        _set_weld_relpose_to_current_state(model, data)
        # Prime ctrl + qpos_spring so the actuator starts from IK when settle ends.
        for j in range(len(q_l)):
            model.qpos_spring[ctrl._left_qadr[j]]  = q_l[j]
            model.qpos_spring[ctrl._right_qadr[j]] = q_r[j]
            if ctrl._left_act[j]  >= 0: data.ctrl[ctrl._left_act[j]]  = q_l[j]
            if ctrl._right_act[j] >= 0: data.ctrl[ctrl._right_act[j]] = q_r[j]

        for k in range(_settle_steps):
            if viewer is not None and not viewer.is_running():
                return
            data.qpos[ctrl._left_qadr]  = q_l
            data.qpos[ctrl._right_qadr] = q_r
            data.qvel[ctrl._left_dadr]  = 0.0
            data.qvel[ctrl._right_dadr] = 0.0
            mujoco.mj_forward(model, data)
            _step_and_maybe_capture(renderer)
            if viewer is not None and k % 20 == 0:
                viewer.sync()

        # ── Execute path waypoints ────────────────────────────────────────────
        prev_ly, prev_ry = ly_start, ry_start
        for wp_idx, (tl, tr, ly, ry) in enumerate(targets_3d):
            if viewer is not None and not viewer.is_running():
                return
            target_l = np.asarray(tl, dtype=np.float64)
            target_r = np.asarray(tr, dtype=np.float64)
            print(
                f"  Waypoint {wp_idx+1}/{n_wp}  "
                f"left=({tl[0]:.3f},{tl[1]:.3f}) yaw={np.degrees(ly):.1f}°  "
                f"right=({tr[0]:.3f},{tr[1]:.3f}) yaw={np.degrees(ry):.1f}°"
            )
            prev_l = ctrl.gripper_position(data, "left").copy()
            prev_r = ctrl.gripper_position(data, "right").copy()
            # Velocity-cap the gripper so the cable's mass-inertia keeps
            # up: stretch n_steps for long moves, leave short moves at
            # the user-specified base.  Then cosine ease-in/out over the
            # first 70 % of those steps with a 30 % hold so the cable
            # has time to settle before the next waypoint starts.
            n_steps = _steps_for_waypoint(
                prev_l, prev_r, target_l, target_r,
                dt=model.opt.timestep,
                base_steps=steps_per_wp,
                max_speed=max_gripper_speed,
            )
            ramp_steps = max(int(0.7 * n_steps), 1)
            # Shortest-path angular deltas so yaw interpolation doesn't
            # take the long way around (e.g. 170°→-170° through 180°).
            dly = _wrap_pi(ly - prev_ly)
            dry = _wrap_pi(ry - prev_ry)
            if yaw_mode == YAW_MODE_ROTATE_THEN_MOVE:
                yaw_steps = 0
                if max(abs(dly), abs(dry)) > 1e-9:
                    yaw_steps = max(int(ROTATE_THEN_MOVE_YAW_FRACTION * n_steps), 1)
                for k in range(yaw_steps):
                    if viewer is not None and not viewer.is_running():
                        return
                    t = _cosine_ease_step(k, yaw_steps)
                    cur_ly = prev_ly + t * dly
                    cur_ry = prev_ry + t * dry
                    ctrl.set_targets(model, data, prev_l, prev_r,
                                     left_yaw=cur_ly, right_yaw=cur_ry)
                    _step_and_maybe_capture(renderer)
                    if viewer is not None and k % 5 == 0:
                        viewer.sync()

                for k in range(n_steps):
                    if viewer is not None and not viewer.is_running():
                        return
                    if k < ramp_steps:
                        t = _cosine_ease_step(k, ramp_steps)
                    else:
                        t = 1.0
                    cur_l = prev_l + t * (target_l - prev_l)
                    cur_r = prev_r + t * (target_r - prev_r)
                    ctrl.set_targets(model, data, cur_l, cur_r,
                                     left_yaw=ly, right_yaw=ry)
                    _step_and_maybe_capture(renderer)
                    if viewer is not None and k % 5 == 0:
                        viewer.sync()
                prev_ly, prev_ry = ly, ry
                continue

            for k in range(n_steps):
                if viewer is not None and not viewer.is_running():
                    return
                if k < ramp_steps:
                    t = _cosine_ease_step(k, ramp_steps)
                else:
                    t = 1.0
                cur_l  = prev_l  + t * (target_l - prev_l)
                cur_r  = prev_r  + t * (target_r - prev_r)
                cur_ly = prev_ly + t * dly
                cur_ry = prev_ry + t * dry
                ctrl.set_targets(model, data, cur_l, cur_r,
                                 left_yaw=cur_ly, right_yaw=cur_ry)
                _step_and_maybe_capture(renderer)
                if viewer is not None and k % 5 == 0:
                    viewer.sync()
            prev_ly, prev_ry = ly, ry

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth  = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            _run(viewer=viewer)
            if viewer.is_running():
                print("  Execution complete - simulation running. Close window to exit.")
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    update_anchor_mocaps(model, data)
                    viewer.sync()
    else:
        renderer = mujoco.Renderer(model, height=480, width=640) if save_video else None
        _run(renderer=renderer)
        if save_video and frames and video_path is not None:
            try:
                import imageio
                imageio.mimwrite(str(video_path), frames, fps=30)
                print(f"  Video saved -> {video_path}")
            except (ImportError, ValueError) as e:
                print(f"  Video save skipped ({e})")


# ── CSV-driven execution (no scenario, no GCR cable path) ────────────────────

def execute_waypoints_in_mujoco(
    targets_3d: list[tuple[np.ndarray, np.ndarray, float, float]],
    *,
    obstacles_3d: list[dict] | None = None,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    steps_per_wp: int = 300,
    use_viewer: bool = True,
    save_video: bool = False,
    video_path: Path | None = None,
    video_four_corners: bool = False,
    video_max_seconds: float = 60.0,
    max_gripper_speed: float = DEFAULT_MAX_GRIPPER_SPEED,
    cable_length: float = CABLE_LENGTH,
    cable_num_segments: int = CABLE_NUM_SEGMENTS,
    cable_cstart_nodes: np.ndarray | None = None,
    cable_profile_nodes: list[np.ndarray] | None = None,
    ghost_target_shape: str = "final",
    ghost_target_rgba: str = "1.0 0.86 0.18 0.35",
    final_error_wait: float = 0.0,
    final_error_json_path: Path | None = None,
    yaw_mode: str = YAW_MODE_SIMULTANEOUS,
    last_transition_side_correction: bool = False,
    side_correction_epsilon: float = 1e-5,
    side_correction_cable_margin: float = 0.003,
    side_correction_moving_robot: str = "farther-from-target",
    invert_red_anchor_yaw: bool = False,
    cable_endpoint_tangent_assist: bool = False,
    cable_endpoint_tangent_alpha: float = 0.10,
    cable_endpoint_clamp_segments: int = 3,
    cable_endpoint_tangent_velocity_damping: float = 0.0,
    cable_attachment_mode: str | None = None,
    cable_weld_torquescale: float | None = None,
    tcp_schedule: str = "simultaneous",
    tcp_stagger_fraction: float = DEFAULT_TCP_STAGGER_FRACTION,
) -> None:
    """Execute a manually-specified gripper waypoint sequence in MuJoCo.

    Same simulation loop as :func:`execute_in_mujoco` but takes pre-computed
    ``(left_pos, right_pos, left_yaw, right_yaw)`` tuples directly — no
    Scenario object, no GCR cable path, no 2-D-to-3-D workspace mapping.

    Each tuple element:
      - ``left_pos``, ``right_pos``: shape-(3,) world-frame positions [m].
      - ``left_yaw``, ``right_yaw``: gripper yaw angles [rad].

    The cable is initialised as a straight chain of N=cable_num_segments+1 nodes between the
    first waypoint's left and right gripper positions.  The cable physics
    relaxes it to a natural shape during the kinematic settle phase.

    Parameters
    ----------
    obstacles_3d
        Pre-built MuJoCo obstacle dicts (same format as
        ``_gcr_obstacles_to_mujoco`` output).  Pass ``None`` or ``[]`` for
        an empty workspace.
    workspace_bounds
        ``((xmin, xmax), (ymin, ymax))`` to draw a workspace rectangle.
        ``None`` uses the fixed 1.0 m x 0.6 m GCR planner workspace.
    steps_per_wp, use_viewer, save_video, video_path
        Same meaning as in ``execute_in_mujoco``.
    """
    if not targets_3d:
        raise ValueError("execute_waypoints_in_mujoco: empty targets_3d list")
    yaw_mode = _validate_yaw_mode(yaw_mode)
    if cable_length <= 0.0:
        raise ValueError("cable_length must be positive")
    if cable_num_segments < 1:
        raise ValueError("cable_num_segments must be at least 1")
    if final_error_wait < 0.0:
        raise ValueError("final_error_wait must be non-negative")
    if video_max_seconds <= 0.0:
        raise ValueError("video_max_seconds must be positive")
    ghost_target_shape = ghost_target_shape.strip().lower()
    if ghost_target_shape not in GHOST_TARGET_SHAPE_CHOICES:
        raise ValueError(
            "ghost_target_shape must be one of: "
            f"{', '.join(GHOST_TARGET_SHAPE_CHOICES)}"
        )
    if side_correction_epsilon < 0.0:
        raise ValueError("side_correction_epsilon must be non-negative")
    if side_correction_cable_margin < 0.0:
        raise ValueError("side_correction_cable_margin must be non-negative")
    side_correction_moving_robot = side_correction_moving_robot.strip().lower()
    if side_correction_moving_robot not in SIDE_CORRECTION_MOVING_ROBOT_CHOICES:
        raise ValueError(
            "side_correction_moving_robot must be one of: "
            f"{', '.join(SIDE_CORRECTION_MOVING_ROBOT_CHOICES)}"
        )
    if cable_endpoint_tangent_alpha < 0.0 or cable_endpoint_tangent_alpha > 1.0:
        raise ValueError("cable_endpoint_tangent_alpha must be in [0, 1]")
    if cable_endpoint_clamp_segments < 1:
        raise ValueError("cable_endpoint_clamp_segments must be at least 1")
    if cable_endpoint_tangent_velocity_damping < 0.0 or cable_endpoint_tangent_velocity_damping > 1.0:
        raise ValueError("cable_endpoint_tangent_velocity_damping must be in [0, 1]")
    tcp_schedule = tcp_schedule.strip().lower()
    if tcp_schedule not in TCP_SCHEDULE_CHOICES:
        raise ValueError(f"tcp_schedule must be one of: {', '.join(TCP_SCHEDULE_CHOICES)}")
    if tcp_stagger_fraction < 0.0 or tcp_stagger_fraction > 0.85:
        raise ValueError("tcp_stagger_fraction must be in [0, 0.85]")
    if cable_attachment_mode is not None:
        cable_attachment_mode = cable_attachment_mode.strip().lower()
        if cable_attachment_mode not in {"site_weld", "two_point_clamp", "body_weld"}:
            raise ValueError(
                "cable_attachment_mode must be one of: "
                "site_weld, two_point_clamp, body_weld"
            )
        scene_build.CABLE_ENDPOINT_ATTACHMENT_MODE = cable_attachment_mode
    if cable_weld_torquescale is not None:
        if cable_weld_torquescale < 0.0:
            raise ValueError("cable_weld_torquescale must be non-negative")
        scene_build.CABLE_ENDPOINT_WELD_TORQUESCALE = float(cable_weld_torquescale)

    obstacles_3d = list(obstacles_3d or [])
    n_wp = len(targets_3d)
    tl_start, tr_start, ly_start, ry_start = targets_3d[0]

    # Each gripper attaches to the cable's actual outer endpoint.
    # Exported scenarios provide the initial cable nodes explicitly; manual
    # CSV files without metadata initialize a straight cable directly between
    # the first row's left/right gripper positions.
    if cable_cstart_nodes is not None:
        cstart_nodes_3d = np.asarray(cable_cstart_nodes, dtype=np.float64)
        if cstart_nodes_3d.ndim != 2 or cstart_nodes_3d.shape[1] != 3 or len(cstart_nodes_3d) < 2:
            raise ValueError("cable_cstart_nodes must be an (N, 3) array with N >= 2")
        cstart_nodes_3d = _resample_polyline_nodes(cstart_nodes_3d, cable_num_segments)
        cable_length = _polyline_length(cstart_nodes_3d)
    else:
        _tl = np.asarray(tl_start, dtype=np.float64)
        _tr = np.asarray(tr_start, dtype=np.float64)
        cstart_nodes_3d = np.linspace(_tl, _tr, num=cable_num_segments + 1, endpoint=True)
        cable_length = _polyline_length(cstart_nodes_3d)

    profile_nodes_3d: list[np.ndarray] | None = None
    if cable_profile_nodes is not None:
        if len(cable_profile_nodes) != n_wp:
            print(
                "  Last-transition side correction: disabled "
                f"(got {len(cable_profile_nodes)} cable profiles for {n_wp} waypoints)."
            )
        else:
            profile_nodes_3d = []
            for profile_idx, profile in enumerate(cable_profile_nodes, start=1):
                nodes = np.asarray(profile, dtype=np.float64)
                if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2:
                    raise ValueError(
                        "cable_profile_nodes entries must be (N, 3) arrays "
                        f"with N >= 2; bad profile at waypoint {profile_idx}"
                    )
                profile_nodes_3d.append(nodes.copy())
    side_correction_enabled = (
        bool(last_transition_side_correction)
        and profile_nodes_3d is not None
        and n_wp >= 2
    )
    if ghost_target_shape != "none" and profile_nodes_3d is None:
        print("  Ghost target shape: unavailable (missing cable profile metadata).")
        ghost_profiles = None
    elif ghost_target_shape == "final" and profile_nodes_3d is not None:
        ghost_profiles = [profile_nodes_3d[-1]]
    elif ghost_target_shape == "all" and profile_nodes_3d is not None:
        ghost_profiles = profile_nodes_3d
    else:
        ghost_profiles = None

    # Use the fixed 1.0 m x 0.6 m planner workspace by default. This keeps the
    # MuJoCo border consistent across all exported GCR scenarios.
    if workspace_bounds is None:
        workspace_bounds = (WS_3D_X, WS_3D_Y)

    print(
        f"  Cable: length={cable_length:.6f} m, "
        f"segments={cable_num_segments}, nodes={cable_num_segments + 1}"
    )
    print(f"  Building MuJoCo scene with {len(obstacles_3d)} obstacle(s)…")
    model, data, _ = build_and_compile(
        obstacles=obstacles_3d, lock_robot_joints=False,
        ik_targets={"left": tl_start, "right": tr_start},
        cable_cstart=cstart_nodes_3d,
        cable_length=cable_length,
        cable_num_segments=cable_num_segments,
        workspace_bounds=workspace_bounds,
        ghost_cable_profiles=ghost_profiles,
        ghost_cable_rgba=ghost_target_rgba,
    )
    ctrl = DualUR10Controller(model)
    video_writer = None
    video_frames_written = 0
    video_sim_steps_seen = 0
    video_capture_stride = 1
    video_output_fps = 30

    def _make_corner_cameras() -> list[mujoco.MjvCamera]:
        """Four fixed low oblique views around the workspace, packed 2x2."""
        cameras: list[mujoco.MjvCamera] = []
        for azimuth in (45.0, 135.0, -135.0, -45.0):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = np.array([0.0, 0.30, 0.10], dtype=np.float64)
            cam.distance = 1.55
            cam.elevation = -25.0
            cam.azimuth = azimuth
            cameras.append(cam)
        return cameras

    def _render_four_corner_frame(
        renderers: list[mujoco.Renderer],
        cameras: list[mujoco.MjvCamera],
    ) -> np.ndarray:
        views = []
        for renderer, camera in zip(renderers, cameras):
            renderer.update_scene(data, camera=camera)
            views.append(renderer.render()[2:-2, 2:-2].copy())
        top = np.concatenate((views[0], views[1]), axis=1)
        bottom = np.concatenate((views[2], views[3]), axis=1)
        frame = np.concatenate((top, bottom), axis=0)

        # Add deliberate separators between the four camera views.
        line_px = 6
        y_mid = frame.shape[0] // 2
        x_mid = frame.shape[1] // 2
        frame[max(0, y_mid - line_px // 2):y_mid + (line_px + 1) // 2, :, :] = 0
        frame[:, max(0, x_mid - line_px // 2):x_mid + (line_px + 1) // 2, :] = 0
        return frame

    def _ensure_offscreen_buffer(width: int, height: int) -> None:
        """MuJoCo's renderer cannot exceed model.vis.global offscreen size."""
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))

    def _close_renderer(renderer) -> None:
        if renderer is None:
            return
        renderers = renderer[0] if isinstance(renderer, tuple) else [renderer]
        for item in renderers:
            close = getattr(item, "close", None)
            if close is not None:
                close()

    def _close_video_writer() -> None:
        nonlocal video_writer
        if video_writer is not None:
            close = getattr(video_writer, "close", None)
            if close is not None:
                close()
            video_writer = None

    def _step_without_capture() -> None:
        update_anchor_mocaps(model, data)
        if cable_endpoint_tangent_assist:
            apply_cable_endpoint_tangent_assist(
                model,
                data,
                alpha=cable_endpoint_tangent_alpha,
                clamp_segments=cable_endpoint_clamp_segments,
                velocity_damping=cable_endpoint_tangent_velocity_damping,
            )
        mujoco.mj_step(model, data)

    def _settle_and_measure_final_error(viewer=None, renderer=None) -> None:
        if final_error_wait <= 0.0 and final_error_json_path is None:
            return
        wait_steps = int(round(final_error_wait / float(model.opt.timestep)))
        if wait_steps > 0:
            print(f"  Final cable error: settling for {final_error_wait:g} s before measuring...")
        for step_idx in range(max(0, wait_steps)):
            if viewer is not None and not viewer.is_running():
                break
            if renderer is not None:
                _step_and_maybe_capture(renderer)
            else:
                _step_without_capture()
            if viewer is not None and step_idx % 20 == 0:
                viewer.sync()

        if profile_nodes_3d is None:
            print("  Final cable error: skipped (missing cable profile metadata).")
            return
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

    def _step_and_maybe_capture(renderer=None):
        nonlocal video_frames_written, video_sim_steps_seen
        update_anchor_mocaps(model, data)
        if cable_endpoint_tangent_assist:
            apply_cable_endpoint_tangent_assist(
                model,
                data,
                alpha=cable_endpoint_tangent_alpha,
                clamp_segments=cable_endpoint_clamp_segments,
                velocity_damping=cable_endpoint_tangent_velocity_damping,
            )
        mujoco.mj_step(model, data)
        if renderer is not None and video_writer is not None:
            if video_sim_steps_seen % video_capture_stride == 0:
                if isinstance(renderer, tuple):
                    renderers, cameras = renderer
                    frame = _render_four_corner_frame(renderers, cameras)
                else:
                    renderer.update_scene(data)
                    frame = renderer.render().copy()
                video_writer.append_data(frame)
                video_frames_written += 1
            video_sim_steps_seen += 1

    print(f"\n  {'WP':>3}  {'Left x':>8} {'Left y':>8} {'Left yaw°':>10}  "
          f"{'Right x':>8} {'Right y':>8} {'Right yaw°':>10}")
    print("  " + "-" * 67)
    for wp_idx, (tl, tr, ly, ry) in enumerate(targets_3d):
        print(f"  {wp_idx+1:>3}  {tl[0]:>8.3f} {tl[1]:>8.3f} "
              f"{np.degrees(ly):>10.1f}°"
              f"  {tr[0]:>8.3f} {tr[1]:>8.3f} {np.degrees(ry):>10.1f}°")
    print()
    print(f"  Yaw motion mode: {yaw_mode}\n")
    print(
        "  TCP schedule: "
        f"{tcp_schedule}"
        + (f" (stagger_fraction={tcp_stagger_fraction:g})" if tcp_schedule != "simultaneous" else "")
        + "\n"
    )
    if invert_red_anchor_yaw:
        print("  Red-anchor yaw rotation: inverted winding for the left robot.\n")
    print(
        "  Cable endpoint attachment: "
        f"{scene_build.CABLE_ENDPOINT_ATTACHMENT_MODE} "
        f"(weld_torquescale={scene_build.CABLE_ENDPOINT_WELD_TORQUESCALE:g}).\n"
    )
    if cable_endpoint_tangent_assist:
        print(
            "  Cable endpoint tangent assist: enabled "
            f"(alpha={cable_endpoint_tangent_alpha:g}, "
            f"clamp_segments={cable_endpoint_clamp_segments}, "
            f"velocity_damping={cable_endpoint_tangent_velocity_damping:g}).\n"
        )
    if ghost_target_shape != "none" and ghost_profiles is not None:
        print(
            "  Ghost target shape: "
            f"{ghost_target_shape} ({len(ghost_profiles)} profile(s), "
            f"rgba={ghost_target_rgba}).\n"
        )
    if last_transition_side_correction:
        if side_correction_enabled:
            print(
                "  Last-transition cable side correction: enabled "
                f"(epsilon={side_correction_epsilon:g}, "
                f"stretch_margin={side_correction_cable_margin:g} m, "
                f"moving_robot={side_correction_moving_robot})\n"
            )
        else:
            print(
                "  Last-transition cable side correction: unavailable "
                "(missing cable profile metadata or only one waypoint).\n"
            )

    _settle_steps = 200

    def _estimate_recorded_sim_steps() -> int:
        """Estimate captured simulation steps to choose a time-lapse stride."""
        total = int(_settle_steps)
        prev_l_est = np.asarray(tl_start, dtype=np.float64)
        prev_r_est = np.asarray(tr_start, dtype=np.float64)
        prev_ly_est = float(ly_start)
        prev_ry_est = float(ry_start)
        for wp_idx, (tl, tr, ly, ry) in enumerate(targets_3d):
            target_l_est = np.asarray(tl, dtype=np.float64)
            target_r_est = np.asarray(tr, dtype=np.float64)
            n_steps_est = _steps_for_waypoint(
                prev_l_est,
                prev_r_est,
                target_l_est,
                target_r_est,
                dt=model.opt.timestep,
                base_steps=steps_per_wp,
                max_speed=max_gripper_speed,
            )
            if yaw_mode == YAW_MODE_ROTATE_THEN_MOVE:
                dly_est = _wrap_pi(ly - prev_ly_est)
                dry_est = _wrap_pi(ry - prev_ry_est)
                if invert_red_anchor_yaw:
                    dly_est = _invert_yaw_delta(dly_est)
                if max(abs(dly_est), abs(dry_est)) > 1e-9:
                    total += max(int(ROTATE_THEN_MOVE_YAW_FRACTION * n_steps_est), 1)
            total += n_steps_est
            if side_correction_enabled and wp_idx == n_wp - 1 and wp_idx > 0:
                total += max(n_steps_est, steps_per_wp)
            prev_l_est = target_l_est
            prev_r_est = target_r_est
            prev_ly_est = float(ly)
            prev_ry_est = float(ry)
        total += int(round(final_error_wait / float(model.opt.timestep)))
        return max(total, 1)

    def _run(viewer=None, renderer=None):
        # Kinematic settle (same as execute_in_mujoco).
        opts = IKSolveOptions(max_iter=300, ori_tol=0.01)
        q_l = ctrl._solve_step(model, data, "left",
                               np.asarray(tl_start, dtype=np.float64),
                               opts, ly_start)
        q_r = ctrl._solve_step(model, data, "right",
                               np.asarray(tr_start, dtype=np.float64),
                               opts, ry_start)
        data.qpos[ctrl._left_qadr] = q_l
        data.qpos[ctrl._right_qadr] = q_r
        data.qvel[ctrl._left_dadr] = 0.0
        data.qvel[ctrl._right_dadr] = 0.0
        mujoco.mj_forward(model, data)
        update_anchor_mocaps(model, data)
        mujoco.mj_forward(model, data)
        _set_weld_relpose_to_current_state(model, data)
        for j in range(len(q_l)):
            model.qpos_spring[ctrl._left_qadr[j]]  = q_l[j]
            model.qpos_spring[ctrl._right_qadr[j]] = q_r[j]
            if ctrl._left_act[j]  >= 0: data.ctrl[ctrl._left_act[j]]  = q_l[j]
            if ctrl._right_act[j] >= 0: data.ctrl[ctrl._right_act[j]] = q_r[j]

        for k in range(_settle_steps):
            if viewer is not None and not viewer.is_running():
                return
            data.qpos[ctrl._left_qadr]  = q_l
            data.qpos[ctrl._right_qadr] = q_r
            data.qvel[ctrl._left_dadr]  = 0.0
            data.qvel[ctrl._right_dadr] = 0.0
            mujoco.mj_forward(model, data)
            _step_and_maybe_capture(renderer)
            if viewer is not None and k % 20 == 0:
                viewer.sync()

        # Waypoint tracking with cosine ease-in/out + 30 % hold and
        # per-waypoint velocity cap.  Yaw is interpolated alongside the
        # position with the same ``t`` (and shortest-path wrap-around)
        # so the gripper doesn't snap-rotate before translating.
        prev_ly, prev_ry = ly_start, ry_start
        for wp_idx, (tl, tr, ly, ry) in enumerate(targets_3d):
            if viewer is not None and not viewer.is_running():
                return
            target_l = np.asarray(tl, dtype=np.float64)
            target_r = np.asarray(tr, dtype=np.float64)
            prev_l = ctrl.gripper_position(data, "left").copy()
            prev_r = ctrl.gripper_position(data, "right").copy()
            n_steps = _steps_for_waypoint(
                prev_l, prev_r, target_l, target_r,
                dt=model.opt.timestep,
                base_steps=steps_per_wp,
                max_speed=max_gripper_speed,
            )
            print(f"  Waypoint {wp_idx+1}/{n_wp}  "
                  f"left=({tl[0]:.3f},{tl[1]:.3f}) yaw={np.degrees(ly):.1f}°  "
                  f"right=({tr[0]:.3f},{tr[1]:.3f}) yaw={np.degrees(ry):.1f}°  "
                  f"({n_steps * model.opt.timestep:.2f}s)")
            ramp_steps = max(int(0.7 * n_steps), 1)
            dly = _wrap_pi(ly - prev_ly)
            dry = _wrap_pi(ry - prev_ry)
            if invert_red_anchor_yaw:
                dly = _invert_yaw_delta(dly)
            if tcp_schedule == "auto-stagger":
                waypoint_schedule, schedule_scores = _choose_tcp_schedule(
                    prev_l,
                    target_l,
                    prev_r,
                    target_r,
                    cable_length,
                    stagger_fraction=tcp_stagger_fraction,
                    cable_length_margin=side_correction_cable_margin,
                )
                if waypoint_schedule != "simultaneous":
                    _, min_tcp, min_arm = schedule_scores[waypoint_schedule]
                    print(
                        f"    TCP schedule: {waypoint_schedule} "
                        f"(min_tcp={min_tcp:.3f} m, min_arm={min_arm:.3f} m)"
                    )
            else:
                waypoint_schedule = tcp_schedule
            correction_this_transition = (
                side_correction_enabled
                and wp_idx == n_wp - 1
                and wp_idx > 0
                and profile_nodes_3d is not None
            )
            correction_target_profile = (
                profile_nodes_3d[wp_idx] if correction_this_transition else None
            )
            correction_state = {
                "left_progress": 0.0,
                "right_progress": 0.0,
                "last_nominal_progress": 0.0,
                "moving_robot": "",
                "active": False,
                "done": False,
                "match_speed_resume": False,
            }
            correction_extension_step = 1.0 / float(max(n_steps, steps_per_wp, 1))
            left_transition_length = float(np.linalg.norm(target_l - prev_l))
            right_transition_length = float(np.linalg.norm(target_r - prev_r))

            def _command_from_progress(
                left_progress: float,
                right_progress: float,
                *,
                final_yaw: bool,
            ) -> tuple[np.ndarray, np.ndarray, float, float]:
                """Build gripper targets from independent left/right progress."""
                left_progress = float(np.clip(left_progress, 0.0, 1.0))
                right_progress = float(np.clip(right_progress, 0.0, 1.0))
                cur_l = prev_l + left_progress * (target_l - prev_l)
                cur_r = prev_r + right_progress * (target_r - prev_r)
                if final_yaw:
                    return cur_l, cur_r, ly, ry
                cur_ly = prev_ly + left_progress * dly
                cur_ry = prev_ry + right_progress * dry
                return cur_l, cur_r, cur_ly, cur_ry

            def _choose_side_correction_moving_robot() -> tuple[str, float, float]:
                """Choose which TCP moves while the other holds during correction."""
                cur_l, cur_r, _, _ = _command_from_progress(
                    float(correction_state["left_progress"]),
                    float(correction_state["right_progress"]),
                    final_yaw=True,
                )
                left_remaining = float(np.linalg.norm(target_l - cur_l))
                right_remaining = float(np.linalg.norm(target_r - cur_r))
                if side_correction_moving_robot in {"left", "right"}:
                    return side_correction_moving_robot, left_remaining, right_remaining
                moving_robot = "left" if left_remaining >= right_remaining else "right"
                return moving_robot, left_remaining, right_remaining

            def _target_profile_for_current_points(current_points: np.ndarray) -> np.ndarray:
                assert correction_target_profile is not None
                if len(current_points) == len(correction_target_profile):
                    return correction_target_profile
                return _resample_polyline_nodes(
                    correction_target_profile,
                    max(len(current_points) - 1, 1),
                )

            def _advance_progress_toward(current: float, target: float, max_step: float) -> float:
                """Advance progress smoothly without jumping after a held phase."""
                current = float(np.clip(current, 0.0, 1.0))
                target = float(np.clip(target, 0.0, 1.0))
                if current >= target:
                    return current
                return min(target, current + max(0.0, float(max_step)))

            def _advance_both_progress_with_matched_tcp_speed(
                target_progress: float,
                progress_increment: float,
            ) -> None:
                """Advance both TCPs with the same Cartesian step length.

                This is used only after side correction reaches the cable-length
                limit. At that point one TCP may be much farther from its target
                than the other, so equal progress increments can make the longer
                endpoint motion visibly faster. Equalizing the Cartesian step
                keeps the resumed motion calm and predictable.
                """
                target_progress = float(np.clip(target_progress, 0.0, 1.0))
                progress_increment = max(0.0, float(progress_increment))
                left_progress = float(correction_state["left_progress"])
                right_progress = float(correction_state["right_progress"])
                left_active = left_progress < target_progress - 1e-9 and left_transition_length > 1e-12
                right_active = right_progress < target_progress - 1e-9 and right_transition_length > 1e-12
                active_lengths = []
                if left_active:
                    active_lengths.append(left_transition_length)
                if right_active:
                    active_lengths.append(right_transition_length)
                if not active_lengths:
                    correction_state["left_progress"] = target_progress
                    correction_state["right_progress"] = target_progress
                    return

                step_distance = progress_increment * min(active_lengths)
                if left_active:
                    correction_state["left_progress"] = min(
                        target_progress,
                        left_progress + step_distance / left_transition_length,
                    )
                else:
                    correction_state["left_progress"] = target_progress
                if right_active:
                    correction_state["right_progress"] = min(
                        target_progress,
                        right_progress + step_distance / right_transition_length,
                    )
                else:
                    correction_state["right_progress"] = target_progress

            def _progress_increment(t_nominal: float, override: float | None) -> tuple[float, float]:
                """Return the amount a moving TCP may advance this simulation step."""
                t_nominal = float(np.clip(t_nominal, 0.0, 1.0))
                if override is not None:
                    return max(0.0, float(override)), t_nominal
                last = float(correction_state["last_nominal_progress"])
                increment = max(0.0, t_nominal - last)
                correction_state["last_nominal_progress"] = max(last, t_nominal)
                return increment, t_nominal

            def _apply_side_correction_progress(
                t_nominal: float,
                progress_increment_override: float | None = None,
            ) -> tuple[float, float]:
                """Hold one TCP during the final transition until cable side is valid."""
                if not correction_this_transition:
                    return t_nominal, t_nominal
                progress_increment, t_nominal = _progress_increment(
                    t_nominal,
                    progress_increment_override,
                )

                if correction_state["done"]:
                    if correction_state["match_speed_resume"]:
                        _advance_both_progress_with_matched_tcp_speed(
                            t_nominal,
                            progress_increment,
                        )
                    else:
                        correction_state["left_progress"] = _advance_progress_toward(
                            float(correction_state["left_progress"]), t_nominal, progress_increment
                        )
                        correction_state["right_progress"] = _advance_progress_toward(
                            float(correction_state["right_progress"]), t_nominal, progress_increment
                        )
                    return (
                        float(correction_state["left_progress"]),
                        float(correction_state["right_progress"]),
                    )

                current_points = _current_cable_points(model, data)
                if len(current_points) < 3:
                    correction_state["done"] = True
                    return t_nominal, t_nominal

                target_side_profile = _target_profile_for_current_points(current_points)
                side_ok, wrong_indices = are_cable_points_on_correct_side(
                    current_points,
                    target_side_profile,
                    side_correction_epsilon,
                )
                if side_ok:
                    if correction_state["active"]:
                        print(
                            "    Last-transition side correction: "
                            "cable side corrected; resuming smooth simultaneous motion."
                        )
                        correction_state["done"] = True
                        correction_state["left_progress"] = _advance_progress_toward(
                            float(correction_state["left_progress"]), t_nominal, progress_increment
                        )
                        correction_state["right_progress"] = _advance_progress_toward(
                            float(correction_state["right_progress"]), t_nominal, progress_increment
                        )
                    else:
                        correction_state["left_progress"] = t_nominal
                        correction_state["right_progress"] = t_nominal
                    return (
                        float(correction_state["left_progress"]),
                        float(correction_state["right_progress"]),
                    )

                if not correction_state["active"]:
                    moving, left_remaining, right_remaining = (
                        _choose_side_correction_moving_robot()
                    )
                    correction_state["moving_robot"] = moving
                    fixed = "right" if moving == "left" else "left"
                    correction_state["active"] = True
                    print(
                        "    Last-transition side correction: start "
                        f"(wrong_nodes={wrong_indices}, moving={moving}, fixed={fixed}, "
                        f"left_remaining={left_remaining:.4f} m, "
                        f"right_remaining={right_remaining:.4f} m)."
                    )
                else:
                    moving = str(correction_state["moving_robot"])
                    fixed = "right" if moving == "left" else "left"

                # While the cable is on the wrong side, advance only one TCP.
                # The other TCP stays at its last progress value so the cable
                # can unwind without both endpoints sweeping through each other.
                progress_key = f"{moving}_progress"
                old_progress = float(correction_state[progress_key])
                proposed_progress = _advance_progress_toward(old_progress, 1.0, progress_increment)
                correction_state[progress_key] = proposed_progress
                left_progress = float(correction_state["left_progress"])
                right_progress = float(correction_state["right_progress"])
                cur_l, cur_r, _, _ = _command_from_progress(
                    left_progress,
                    right_progress,
                    final_yaw=True,
                )
                robot_distance = distance_between_robots(cur_l, cur_r)
                moving_progress = float(correction_state[progress_key])
                moving_reached_target = moving_progress >= 1.0 - 1e-9
                distance_limit_reached = should_switch_moving_robot(
                    robot_distance,
                    cable_length,
                    side_correction_cable_margin,
                )
                if distance_limit_reached and proposed_progress > old_progress:
                    # Clamp the moving TCP to the last safe progress before
                    # switching roles, so the commanded TCP spacing does not
                    # step past the cable-length limit.
                    threshold = max(0.0, cable_length - side_correction_cable_margin)
                    lo = old_progress
                    hi = proposed_progress
                    for _ in range(16):
                        mid = 0.5 * (lo + hi)
                        correction_state[progress_key] = mid
                        test_l = float(correction_state["left_progress"])
                        test_r = float(correction_state["right_progress"])
                        test_cur_l, test_cur_r, _, _ = _command_from_progress(
                            test_l,
                            test_r,
                            final_yaw=True,
                        )
                        if distance_between_robots(test_cur_l, test_cur_r) <= threshold:
                            lo = mid
                        else:
                            hi = mid
                    correction_state[progress_key] = lo
                    left_progress = float(correction_state["left_progress"])
                    right_progress = float(correction_state["right_progress"])
                    cur_l, cur_r, _, _ = _command_from_progress(
                        left_progress,
                        right_progress,
                        final_yaw=True,
                    )
                    robot_distance = distance_between_robots(cur_l, cur_r)
                both_at_target = left_progress >= 1.0 - 1e-9 and right_progress >= 1.0 - 1e-9
                if distance_limit_reached and not both_at_target:
                    correction_state["done"] = True
                    correction_state["match_speed_resume"] = True
                    _advance_both_progress_with_matched_tcp_speed(
                        t_nominal,
                        progress_increment,
                    )
                    print(
                        "    Last-transition side correction: cable length limit reached; "
                        "resuming smooth simultaneous motion with matched TCP speeds "
                        f"(tcp_distance={robot_distance:.4f} m)."
                    )
                    return (
                        float(correction_state["left_progress"]),
                        float(correction_state["right_progress"]),
                    )

                if moving_reached_target and not both_at_target:
                    old_moving = moving
                    new_moving = fixed
                    correction_state["moving_robot"] = new_moving
                    print(
                        "    Last-transition side correction: switch "
                        f"(reason=moving robot reached waypoint, "
                        f"moving={new_moving}, fixed={old_moving}, "
                        f"tcp_distance={robot_distance:.4f} m)."
                    )
                return left_progress, right_progress

            def _side_correction_progress_complete() -> bool:
                return (
                    float(correction_state["left_progress"]) >= 1.0 - 1e-9
                    and float(correction_state["right_progress"]) >= 1.0 - 1e-9
                )

            def _finish_side_correction_if_needed(final_yaw: bool) -> bool:
                if (
                    not correction_this_transition
                    or not correction_state["active"]
                ):
                    return True
                if correction_state["done"] and _side_correction_progress_complete():
                    return True
                print(
                    "    Last-transition side correction: extending final transition "
                    "until the cable side is corrected and both TCPs reach the waypoint."
                )
                for extra_k in range(max(n_steps, steps_per_wp)):
                    if viewer is not None and not viewer.is_running():
                        return False
                    lp, rp = _apply_side_correction_progress(
                        1.0,
                        progress_increment_override=correction_extension_step,
                    )
                    cur_l, cur_r, cur_ly, cur_ry = _command_from_progress(
                        lp,
                        rp,
                        final_yaw=final_yaw,
                    )
                    ctrl.set_targets(model, data, cur_l, cur_r,
                                     left_yaw=cur_ly, right_yaw=cur_ry)
                    _step_and_maybe_capture(renderer)
                    if viewer is not None and extra_k % 5 == 0:
                        viewer.sync()
                    if correction_state["done"] and _side_correction_progress_complete():
                        return True
                print(
                    "    Last-transition side correction: warning, "
                    "correction did not finish before the extension limit."
                )
                return True

            if yaw_mode == YAW_MODE_ROTATE_THEN_MOVE:
                yaw_steps = 0
                if max(abs(dly), abs(dry)) > 1e-9:
                    yaw_steps = max(int(ROTATE_THEN_MOVE_YAW_FRACTION * n_steps), 1)
                for k in range(yaw_steps):
                    if viewer is not None and not viewer.is_running():
                        return
                    t = _cosine_ease_step(k, yaw_steps)
                    cur_ly = prev_ly + t * dly
                    cur_ry = prev_ry + t * dry
                    ctrl.set_targets(model, data, prev_l, prev_r,
                                     left_yaw=cur_ly, right_yaw=cur_ry)
                    _step_and_maybe_capture(renderer)
                    if viewer is not None and k % 5 == 0:
                        viewer.sync()

                for k in range(n_steps):
                    if viewer is not None and not viewer.is_running():
                        return
                    if k < ramp_steps:
                        t = _cosine_ease_step(k, ramp_steps)
                    else:
                        t = 1.0
                    if correction_this_transition:
                        lp, rp = _apply_side_correction_progress(t)
                    else:
                        lp, rp = _tcp_schedule_progress(
                            t,
                            waypoint_schedule,
                            prev_l,
                            target_l,
                            prev_r,
                            target_r,
                            cable_length,
                            stagger_fraction=tcp_stagger_fraction,
                            cable_length_margin=side_correction_cable_margin,
                        )
                    cur_l, cur_r, cur_ly, cur_ry = _command_from_progress(
                        lp,
                        rp,
                        final_yaw=True,
                    )
                    ctrl.set_targets(model, data, cur_l, cur_r,
                                     left_yaw=cur_ly, right_yaw=cur_ry)
                    _step_and_maybe_capture(renderer)
                    if viewer is not None and k % 5 == 0:
                        viewer.sync()
                if not _finish_side_correction_if_needed(final_yaw=True):
                    return
                prev_ly, prev_ry = ly, ry
                continue

            for k in range(n_steps):
                if viewer is not None and not viewer.is_running():
                    return
                if k < ramp_steps:
                    t = _cosine_ease_step(k, ramp_steps)
                else:
                    t = 1.0
                if correction_this_transition:
                    lp, rp = _apply_side_correction_progress(t)
                else:
                    lp, rp = _tcp_schedule_progress(
                        t,
                        waypoint_schedule,
                        prev_l,
                        target_l,
                        prev_r,
                        target_r,
                        cable_length,
                        stagger_fraction=tcp_stagger_fraction,
                        cable_length_margin=side_correction_cable_margin,
                    )
                cur_l, cur_r, cur_ly, cur_ry = _command_from_progress(
                    lp,
                    rp,
                    final_yaw=False,
                )
                ctrl.set_targets(model, data, cur_l, cur_r,
                                 left_yaw=cur_ly, right_yaw=cur_ry)
                _step_and_maybe_capture(renderer)
                if viewer is not None and k % 5 == 0:
                    viewer.sync()
            if not _finish_side_correction_if_needed(final_yaw=False):
                return
            prev_ly, prev_ry = ly, ry

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth   = 135
            viewer.cam.elevation = -20
            viewer.cam.distance  = 2.5
            _run(viewer=viewer)
            _settle_and_measure_final_error(viewer=viewer)
            if viewer.is_running():
                print("  Execution complete - close the window to exit.")
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    update_anchor_mocaps(model, data)
                    viewer.sync()
    else:
        renderer = None
        if save_video:
            if video_path is None:
                print("  Video save skipped (missing video_path)")
                video_writer = None
            else:
                try:
                    import imageio
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    estimated_steps = _estimate_recorded_sim_steps()
                    max_frames = max(1, int(video_output_fps * video_max_seconds))
                    video_capture_stride = max(1, int(np.ceil(estimated_steps / max_frames)))
                    video_writer = imageio.get_writer(
                        str(video_path),
                        fps=video_output_fps,
                        macro_block_size=1,
                    )
                    estimated_seconds = estimated_steps / (
                        video_capture_stride * video_output_fps
                    )
                    print(
                        "  Video time compression: "
                        f"estimated {estimated_steps} sim steps, "
                        f"capture every {video_capture_stride} step(s), "
                        f"target duration <= {min(video_max_seconds, estimated_seconds):.1f}s."
                    )
                except (ImportError, ValueError, OSError) as e:
                    print(f"  Video save skipped ({e})")
                    video_writer = None
            if video_writer is not None:
                if video_four_corners:
                    _ensure_offscreen_buffer(964, 544)
                    renderer = (
                        [mujoco.Renderer(model, height=544, width=964) for _ in range(4)],
                        _make_corner_cameras(),
                    )
                    print(
                        "  Video capture: four-corner 2x2 layout enabled "
                        "(1920x1080, camera elevation -25 deg)."
                    )
                else:
                    _ensure_offscreen_buffer(640, 480)
                    renderer = mujoco.Renderer(model, height=480, width=640)
        try:
            _run(renderer=renderer)
            _settle_and_measure_final_error(renderer=renderer)
        finally:
            _close_renderer(renderer)
            _close_video_writer()
            if save_video and video_path is not None and video_frames_written > 0:
                print(
                    f"  Video saved -> {video_path} "
                    f"({video_frames_written} frames streamed)"
                )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GCR + MuJoCo two-UR10e execution demo")
    parser.add_argument("--scenario",           type=int, default=1)
    parser.add_argument("--nodes",              type=int, default=10)
    parser.add_argument("--max-iterations",     type=int, default=1600)
    parser.add_argument("--scenario-time-limit", type=float, default=120.0)
    parser.add_argument("--no-shortcut-refinement", action="store_true",
                        help="Keep the raw connected GCR path instead of removing intermediate cable profiles.")
    parser.add_argument("--steps-per-waypoint", type=int, default=300)
    parser.add_argument("--max-speed",          type=float,
                        default=DEFAULT_MAX_GRIPPER_SPEED,
                        help=(f"Peak gripper velocity cap [m/s] "
                              f"(default {DEFAULT_MAX_GRIPPER_SPEED:g}). "
                              "Lower = smoother cable, slower run. "
                              "Set to 0 to disable the cap and use steps-per-waypoint only."))
    parser.add_argument("--yaw-mode", choices=YAW_MODE_CHOICES,
                        default=YAW_MODE_SIMULTANEOUS,
                        help=("Waypoint yaw motion: interpolate with translation, "
                              "or rotate first and then translate."))
    parser.add_argument("--cable-segment-multiplier", type=int,
                        default=DEFAULT_CABLE_SEGMENT_MULTIPLIER,
                        help=("Multiply planner cable segments for MuJoCo simulation "
                              f"(default {DEFAULT_CABLE_SEGMENT_MULTIPLIER})."))
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--no-viewer",          action="store_true",
                        help="Headless run; saves MP4 to results/figures/")
    parser.add_argument("--video-path", default=None,
                        help="Optional MP4 output path for --no-viewer runs.")
    args = parser.parse_args(argv)

    scenarios = gcr.build_scenarios(args.nodes)
    if args.scenario < 1 or args.scenario > len(scenarios):
        raise SystemExit(f"--scenario must be in 1..{len(scenarios)}")
    scenario = scenarios[args.scenario - 1]
    print(f"\n{'='*60}")
    print(f"GCR + MuJoCo Execution Demo - {scenario.name}")
    print(f"{'='*60}\n")

    print("Step 1: GCR planning...")
    time_limit = None if args.scenario_time_limit <= 0.0 else args.scenario_time_limit
    path = plan_path(
        scenario,
        seed=args.seed,
        max_iterations=args.max_iterations,
        scenario_time_limit_s=time_limit,
        shortcut_refinement=not args.no_shortcut_refinement,
    )
    if path is None:
        print("GCR failed to find a path - aborting.")
        return 1

    print(f"\nStep 2: MuJoCo execution ({len(path)} waypoints)...")
    use_viewer = not args.no_viewer
    if args.video_path:
        vid_path = Path(args.video_path)
        vid_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        vid_dir = Path(__file__).parents[1] / "results" / "figures"
        vid_dir.mkdir(parents=True, exist_ok=True)
        vid_path = vid_dir / f"scenario{args.scenario}_execution.mp4"

    execute_in_mujoco(
        scenario,
        path,
        steps_per_wp=args.steps_per_waypoint,
        max_gripper_speed=args.max_speed,
        yaw_mode=args.yaw_mode,
        cable_segment_multiplier=args.cable_segment_multiplier,
        use_viewer=use_viewer,
        save_video=not use_viewer,
        video_path=vid_path,
    )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
