"""Robot-aware MuJoCo replay for exported waypoint CSVs.

This is an experimental copy path.  It does not modify ``run_csv.py``.

Adds:
  - geometric robot/cable/obstacle feasibility scoring,
  - endpoint-assignment filtering,
  - yaw/gripper-distance constraints,
  - optional automatic/manual regrasp,
  - safe standby poses for detach/reattach switching,
  - runtime robot contact and tracking-error statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlo2026.mjlab.scene.build_scene import CABLE_LENGTH, CABLE_NUM_SEGMENTS
from mujoco_exec.execute_plan import DEFAULT_CABLE_SEGMENT_MULTIPLIER, GHOST_TARGET_SHAPE_CHOICES
from mujoco_exec.robot_aware_execution import (
    RobotAwareOptions,
    TCP_SCHEDULE_CHOICES,
    YAW_INTERPOLATION_MODES,
    YAW_MODE_CHOICES,
    YAW_MODE_SIMULTANEOUS,
    choose_safe_endpoint_assignment,
    execute_robot_aware_waypoints,
    globally_swapped_targets,
    score_waypoints,
)
from mujoco_exec.run_csv import parse_obstacles_csv, parse_waypoint_csv, parse_waypoint_metadata


def parse_regrasp_indices(values: list[int] | None, n_targets: int) -> list[int]:
    if not values:
        return []
    result = []
    for display_index in values:
        if display_index <= 1 or display_index > n_targets:
            raise SystemExit(f"--regrasp-before uses 1-based waypoint numbers in 2..{n_targets}; got {display_index}")
        result.append(display_index - 1)
    return sorted(set(result))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--csv", required=True, help="Waypoint CSV/table.")
    ap.add_argument("--obstacles", default=None, help="Optional obstacle CSV.")
    ap.add_argument("--steps-per-waypoint", type=int, default=300)
    ap.add_argument("--max-speed", type=float, default=0.10)
    ap.add_argument("--yaw-mode", choices=YAW_MODE_CHOICES, default=YAW_MODE_SIMULTANEOUS)
    ap.add_argument(
        "--invert-red-anchor-yaw",
        action="store_true",
        help="Invert the yaw winding direction for the left/red-anchor robot.",
    )
    ap.add_argument(
        "--cable-attachment-mode",
        choices=("site_weld", "two_point_clamp", "body_weld"),
        default=None,
        help="Cable-to-anchor attachment model used when building the MuJoCo scene.",
    )
    ap.add_argument(
        "--cable-weld-torquescale",
        type=float,
        default=None,
        help="Override the endpoint weld torquescale for site_weld/body_weld attachment modes.",
    )
    tangent_group = ap.add_mutually_exclusive_group()
    tangent_group.add_argument(
        "--cable-endpoint-tangent-assist",
        dest="cable_endpoint_tangent_assist",
        action="store_true",
        help="Enable the experimental kinematic endpoint sleeve assist.",
    )
    tangent_group.add_argument(
        "--no-cable-endpoint-tangent-assist",
        dest="cable_endpoint_tangent_assist",
        action="store_false",
        help="Keep the experimental kinematic endpoint sleeve assist disabled.",
    )
    ap.set_defaults(cable_endpoint_tangent_assist=False)
    ap.add_argument("--cable-endpoint-tangent-alpha", type=float, default=0.04)
    ap.add_argument("--cable-endpoint-clamp-segments", type=int, default=2)
    ap.add_argument("--cable-endpoint-tangent-velocity-damping", type=float, default=0.0)
    ap.add_argument("--cable-length", type=float, default=None)
    ap.add_argument("--cable-segments", type=int, default=None)
    ap.add_argument("--cable-segment-multiplier", type=int, default=DEFAULT_CABLE_SEGMENT_MULTIPLIER)
    ap.add_argument(
        "--ghost-target-shape",
        choices=GHOST_TARGET_SHAPE_CHOICES,
        default="final",
        help="Draw a translucent non-colliding target cable profile. Default is final.",
    )
    ap.add_argument(
        "--no-ghost-target-shape",
        action="store_const",
        const="none",
        dest="ghost_target_shape",
        help="Disable the translucent target-shape ghost cable.",
    )
    ap.add_argument(
        "--ghost-target-rgba",
        default="1.0 0.86 0.18 0.35",
        help="RGBA color for --ghost-target-shape, e.g. '1 0.86 0.18 0.35'.",
    )
    ap.add_argument(
        "--final-error-wait",
        type=float,
        default=0.0,
        help=(
            "After motion finishes, keep simulating for this many seconds, "
            "then compute final cable-vs-goal error in x-y. Use 10 for the settled metric."
        ),
    )
    ap.add_argument(
        "--final-error-json",
        default=None,
        help=(
            "Optional JSON path for final cable error metrics. "
            "If omitted and --final-error-wait > 0, writes next to the CSV."
        ),
    )
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--video-path", default=None)
    ap.add_argument("--save-trajectory", default=None, help="Optional .npz MuJoCo state trajectory for replay.")
    ap.add_argument("--trajectory-stride", type=int, default=1, help="Record every Nth MuJoCo step.")
    ap.add_argument("--score-only", action="store_true", help="Only print robot-aware score; do not run MuJoCo.")
    ap.add_argument("--write-score-json", default=None, help="Optional JSON output for score/runtime stats.")

    ap.add_argument(
        "--endpoint-filter",
        choices=("fixed", "choose-global-swap", "force-global-swap"),
        default="choose-global-swap",
        help="Endpoint assignment: keep CSV order, choose safer global swap, or force global swap.",
    )
    ap.add_argument("--min-gripper-distance", type=float, default=0.11)
    ap.add_argument("--min-arm-segment-distance", type=float, default=0.13)
    ap.add_argument("--max-yaw-jump-deg", type=float, default=120.0)
    ap.add_argument("--max-reach", type=float, default=1.18)
    ap.add_argument("--min-reach-margin", type=float, default=0.035)
    ap.add_argument("--workspace-margin", type=float, default=0.015)
    ap.add_argument("--obstacle-clearance", type=float, default=0.025)
    ap.add_argument("--tracking-error-limit", type=float, default=0.045)
    ap.add_argument(
        "--tcp-schedule",
        choices=TCP_SCHEDULE_CHOICES,
        default="simultaneous",
        help="TCP timing schedule. auto-stagger chooses left/right lead timing per waypoint.",
    )
    ap.add_argument(
        "--tcp-stagger-fraction",
        type=float,
        default=0.35,
        help="Delay fraction for the follower when using left-first/right-first/auto-stagger.",
    )
    ap.add_argument(
        "--cable-length-margin",
        type=float,
        default=0.003,
        help="Keep TCP separation below cable_length - this margin during staggered timing.",
    )

    ap.add_argument("--allow-regrasp", action="store_true")
    ap.add_argument("--auto-regrasp", action="store_true")
    ap.add_argument("--regrasp-before", type=int, action="append", help="1-based waypoint number before which to regrasp. Repeatable.")
    ap.add_argument("--swap-after-regrasp", action="store_true")
    ap.add_argument("--regrasp-steps", type=int, default=220)
    ap.add_argument("--max-auto-regrasps", type=int, default=2)
    ap.add_argument(
        "--initial-pose-mode",
        choices=("auto-clearance", "current"),
        default="auto-clearance",
        help="Choose initial IK branch by robot clearance, or keep the original single-branch startup.",
    )
    ap.add_argument(
        "--use-profile-yaw",
        action="store_true",
        help=(
            "Experimental: recompute yaw from interpolated cable profile nodes at each step. "
            "Default uses the unwrapped cable-tangent yaws exported in the CSV."
        ),
    )
    ap.add_argument(
        "--yaw-interpolation",
        choices=YAW_INTERPOLATION_MODES,
        default="csv",
        help="Default yaw winding for both robots. 'csv' follows the exported cable-tangent yaw sequence.",
    )
    ap.add_argument("--left-yaw-interpolation", choices=YAW_INTERPOLATION_MODES, default=None)
    ap.add_argument("--right-yaw-interpolation", choices=YAW_INTERPOLATION_MODES, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: {csv_path} not found", file=sys.stderr)
        return 1

    metadata = parse_waypoint_metadata(csv_path)
    targets = parse_waypoint_csv(csv_path)
    obstacles_3d = []
    if args.obstacles:
        obs_path = Path(args.obstacles)
        if not obs_path.exists():
            print(f"error: {obs_path} not found", file=sys.stderr)
            return 1
        obstacles_3d = parse_obstacles_csv(obs_path)

    options = RobotAwareOptions(
        min_gripper_distance=args.min_gripper_distance,
        min_arm_segment_distance=args.min_arm_segment_distance,
        max_yaw_jump_deg=args.max_yaw_jump_deg,
        max_reach=args.max_reach,
        min_reach_margin=args.min_reach_margin,
        workspace_margin=args.workspace_margin,
        obstacle_clearance=args.obstacle_clearance,
        tracking_error_limit=args.tracking_error_limit,
        tcp_schedule=args.tcp_schedule,
        tcp_stagger_fraction=args.tcp_stagger_fraction,
        cable_length_margin=args.cable_length_margin,
        auto_regrasp=args.auto_regrasp,
        allow_regrasp=args.allow_regrasp,
        swap_after_regrasp=args.swap_after_regrasp,
        regrasp_steps=args.regrasp_steps,
        max_auto_regrasps=args.max_auto_regrasps,
        initial_pose_search=args.initial_pose_mode == "auto-clearance",
        use_profile_yaw=args.use_profile_yaw,
        invert_red_anchor_yaw=args.invert_red_anchor_yaw,
        cable_attachment_mode=args.cable_attachment_mode,
        cable_weld_torquescale=args.cable_weld_torquescale,
        cable_endpoint_tangent_assist=args.cable_endpoint_tangent_assist,
        cable_endpoint_tangent_alpha=args.cable_endpoint_tangent_alpha,
        cable_endpoint_clamp_segments=args.cable_endpoint_clamp_segments,
        cable_endpoint_tangent_velocity_damping=args.cable_endpoint_tangent_velocity_damping,
        ghost_target_shape=args.ghost_target_shape,
        ghost_target_rgba=args.ghost_target_rgba,
        left_yaw_interpolation=args.left_yaw_interpolation or args.yaw_interpolation,
        right_yaw_interpolation=args.right_yaw_interpolation or args.yaw_interpolation,
    )

    assignment_mode = "fixed_csv"
    score = score_waypoints(targets, obstacles_3d, options)
    if args.endpoint_filter == "choose-global-swap":
        targets, assignment_mode, score = choose_safe_endpoint_assignment(targets, obstacles_3d, options)
    elif args.endpoint_filter == "force-global-swap":
        targets = globally_swapped_targets(targets)
        assignment_mode = "forced_global_swap"
        score = score_waypoints(targets, obstacles_3d, options)
    reverse_initial_cable = assignment_mode in {"global_swap", "forced_global_swap"}

    print(f"Loaded {len(targets)} waypoint(s) from {csv_path}")
    print(f"Loaded {len(obstacles_3d)} obstacle(s)")
    print(f"Endpoint assignment: {assignment_mode}")
    print("Robot-aware pre-score:")
    for key, value in asdict(score).items():
        print(f"  {key}: {value}")

    manual_regrasp = parse_regrasp_indices(args.regrasp_before, len(targets))
    if manual_regrasp:
        print(f"Manual regrasp before waypoint(s): {[idx + 1 for idx in manual_regrasp]}")
    elif args.auto_regrasp and score.recommended_regrasp_indices:
        print(f"Auto regrasp before waypoint(s): {[idx + 1 for idx in score.recommended_regrasp_indices]}")

    cable_length = args.cable_length
    if cable_length is None:
        cable_length = float(metadata.get("cable_length_m", CABLE_LENGTH))
    if args.cable_segment_multiplier < 1:
        print("error: --cable-segment-multiplier must be at least 1", file=sys.stderr)
        return 1
    metadata_segments = metadata.get("cable_segments")
    if args.cable_segments is not None:
        cable_segments = args.cable_segments
    elif metadata_segments is not None:
        cable_segments = int(metadata_segments) * args.cable_segment_multiplier
    else:
        cable_segments = CABLE_NUM_SEGMENTS
    cable_cstart_nodes = metadata.get("cable_cstart_xyz")
    if not isinstance(cable_cstart_nodes, np.ndarray):
        cable_cstart_nodes = None
    elif reverse_initial_cable:
        cable_cstart_nodes = cable_cstart_nodes[::-1].copy()
        print("Initial cable node order: reversed to match swapped endpoints")
    cable_profile_nodes = metadata.get("cable_profiles_xyz")
    if not isinstance(cable_profile_nodes, list):
        cable_profile_nodes = None
    elif reverse_initial_cable:
        cable_profile_nodes = [np.asarray(profile, dtype=np.float64)[::-1].copy() for profile in cable_profile_nodes]
        print("Cable profile node order: reversed to match swapped endpoints")

    score_payload: dict[str, object] = {
        "csv": str(csv_path),
        "obstacles": str(args.obstacles) if args.obstacles else "",
        "endpoint_assignment": assignment_mode,
        "initial_cable_reversed": bool(reverse_initial_cable and cable_cstart_nodes is not None),
        "options": asdict(options),
        "pre_score": asdict(score),
        "manual_regrasp_indices_zero_based": manual_regrasp,
        "manual_regrasp_waypoints_one_based": [idx + 1 for idx in manual_regrasp],
    }

    if args.score_only:
        if args.write_score_json:
            Path(args.write_score_json).write_text(json.dumps(score_payload, indent=2), encoding="utf-8")
        return 0

    use_viewer = not args.no_viewer
    if args.video_path:
        vid_path = Path(args.video_path)
        vid_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        vid_path = csv_path.parent / f"{csv_path.stem}_robot_aware.mp4"
    if args.final_error_json:
        final_error_json_path = Path(args.final_error_json)
        final_error_json_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.final_error_wait > 0.0:
        final_error_json_path = csv_path.parent / f"{csv_path.stem}_final_error.json"
    else:
        final_error_json_path = None

    _, runtime_stats = execute_robot_aware_waypoints(
        targets,
        obstacles_3d=obstacles_3d,
        steps_per_wp=args.steps_per_waypoint,
        max_gripper_speed=args.max_speed,
        cable_length=cable_length,
        cable_num_segments=cable_segments,
        cable_cstart_nodes=cable_cstart_nodes,
        cable_profile_nodes=cable_profile_nodes,
        yaw_mode=args.yaw_mode,
        use_viewer=use_viewer,
        save_video=not use_viewer,
        video_path=vid_path,
        options=options,
        manual_regrasp_indices=manual_regrasp,
        trajectory_path=Path(args.save_trajectory) if args.save_trajectory else None,
        trajectory_stride=args.trajectory_stride,
        final_error_wait=args.final_error_wait,
        final_error_json_path=final_error_json_path,
        trajectory_metadata={
            "csv": str(csv_path),
            "obstacles": str(args.obstacles) if args.obstacles else "",
            "endpoint_assignment": assignment_mode,
            "yaw_mode": args.yaw_mode,
        },
    )
    score_payload["runtime_stats"] = asdict(runtime_stats)
    if args.write_score_json:
        Path(args.write_score_json).write_text(json.dumps(score_payload, indent=2), encoding="utf-8")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
