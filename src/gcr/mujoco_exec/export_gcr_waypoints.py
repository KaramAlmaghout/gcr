"""Solve planner scenarios and export MuJoCo waypoint/obstacle CSV files.

The generated CSV files are accepted directly by ``python -m gcr replay``.
Each row is one waypoint with both grippers' world-frame x/y positions and yaw
angles in degrees.  The z coordinate is supplied by the replay command from
``gcr.sim.build_scene.CABLE_Z``.  Obstacle CSVs are optional, but let
``python -m gcr replay --obstacles`` show the same geometry during replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from gcr.paths import results_dir
from gcr.planner import gcrbup as gcr
from gcr.planner.final_scenarios import build_final_scenarios
from gcr.mujoco_exec.execute_plan import (
    _assign_endpoints,
    _cable_length_2d,
    _execution_options_for_scenario,
    _gcr_obstacles_to_mujoco,
    map_2d_to_3d,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export solved GCR scenarios as MuJoCo waypoint and obstacle CSV files.")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output folder for waypoint CSV files. Default: results/gcr_waypoints")
    parser.add_argument("--scenario", type=int, action="append", help="1-based scenario number to export. Repeat for several. Default: all.")
    parser.add_argument("--nodes", type=int, default=10, help="Number of cable nodes for synthetic scenarios.")
    parser.add_argument("--max-iterations", type=int, default=1600, help="Maximum planner iterations per scenario.")
    parser.add_argument("--scenario-time-limit", type=float, default=120.0, help="Maximum solve time per scenario in seconds; use 0 to disable.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset added to scenario random seeds.")
    parser.add_argument("--no-shortcut-refinement", action="store_true", help="Export the raw connected path instead of removing intermediate cable profiles.")
    parser.add_argument("--endpoint-reorder", action="store_true", help="Reverse cable profile order when endpoint continuity improves and the reversed transition is valid.")
    parser.add_argument("--allow-terminal-reversal", action="store_true", help="Allow endpoint reordering to export reverse(goal) when it is the cleaner terminal endpoint assignment.")
    parser.add_argument(
        "--endpoint-reorder-cost-ratio",
        type=float,
        default=0.75,
        help="Reverse endpoint order when reversed endpoint travel is below this fraction of ordered endpoint travel.",
    )
    parser.add_argument(
        "--endpoint-reorder-tangent-margin",
        type=float,
        default=0.05,
        help="Reverse endpoint order when it improves cable tangent alignment by more than this margin; negative disables.",
    )
    parser.add_argument("--robot-execution-feasibility", action="store_true", help="Enable robot-execution transition checks during planning/export.")
    parser.add_argument("--robot-tcp-clearance", type=float, default=0.0, help="TCP radius for same-time left/right endpoint separation.")
    parser.add_argument("--robot-obstacle-clearance", type=float, default=None, help="TCP-vs-obstacle sweep clearance; use a small value to avoid over-rejecting.")
    parser.add_argument("--robot-swept-tcp-clearance", type=float, default=0.0, help="Optional TCP radius for swept left-path vs right-path clearance.")
    return parser.parse_args(argv)


def selected_indices(requested: Iterable[int] | None, n_scenarios: int) -> list[int]:
    if not requested:
        return list(range(n_scenarios))
    indices = []
    for number in requested:
        if number < 1 or number > n_scenarios:
            raise SystemExit(f"--scenario must be in 1..{n_scenarios}; got {number}")
        indices.append(number - 1)
    return indices


def write_waypoint_csv(
    path: Path,
    scenario_number: int,
    scenario: gcr.Scenario,
    cable_length: float,
    cable_nodes: int,
    cstart_nodes_3d: np.ndarray,
    waypoints: list[tuple[np.ndarray, np.ndarray, float, float]],
    cable_profiles_3d: list[np.ndarray] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("# GCR-to-MuJoCo waypoint table for `python -m gcr replay`.\n")
        f.write(f"# Scenario {scenario_number}: {scenario.name}\n")
        f.write(f"# cable_length_m: {cable_length:.9f}\n")
        f.write(f"# cable_nodes: {cable_nodes}\n")
        f.write(f"# cable_segments: {cable_nodes - 1}\n")
        cstart_text = "; ".join(
            f"{p[0]:.9f},{p[1]:.9f},{p[2]:.9f}" for p in cstart_nodes_3d
        )
        f.write(f"# cable_cstart_xyz: {cstart_text}\n")
        if cable_profiles_3d is not None:
            for profile_idx, profile in enumerate(cable_profiles_3d, start=1):
                profile_text = "; ".join(
                    f"{p[0]:.9f},{p[1]:.9f},{p[2]:.9f}" for p in np.asarray(profile, dtype=float)
                )
                f.write(f"# cable_profile_{profile_idx:03d}_xyz: {profile_text}\n")
        f.write("# yaw_source: unwrapped cable endpoint tangent from planned profiles\n")
        f.write("# Units: x/y in metres, yaw in degrees. z is supplied at replay time.\n")
        f.write("WP, Left x, Left y, Left yaw, Right x, Right y, Right yaw\n")
        for idx, (left, right, left_yaw, right_yaw) in enumerate(waypoints, start=1):
            f.write(
                f"{idx:3d}, "
                f"{left[0]: .6f}, {left[1]: .6f}, {np.degrees(left_yaw): .3f}, "
                f"{right[0]: .6f}, {right[1]: .6f}, {np.degrees(right_yaw): .3f}\n"
            )


def write_obstacle_csv(path: Path, scenario: gcr.Scenario) -> None:
    obstacles = _gcr_obstacles_to_mujoco(scenario.obstacles)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("# Optional obstacle table for `python -m gcr replay --obstacles`.\n")
        f.write("# Cylinder rows: type, name, x, y, radius, half_height\n")
        f.write("# Box rows: type, name, x, y, hx, hy, hz\n")
        f.write("# Polygon rows: polygon, name, height, x1, y1, x2, y2, ...\n")
        f.write("type, name, x_or_height, y_or_x1, radius_or_y1, hy_or_x2, half_height_or_y2, ...\n")
        for obs in obstacles:
            name = obs["name"]
            if "vertices" in obs:
                coords = ", ".join(f"{value:.6f}" for xy in obs["vertices"] for value in xy[:2])
                f.write(f"polygon, {name}, {obs['height']:.6f}, {coords}\n")
            elif "half_size" in obs:
                x, y = obs["pos"][:2]
                hx, hy, hz = obs["half_size"]
                f.write(f"box, {name}, {x:.6f}, {y:.6f}, {hx:.6f}, {hy:.6f}, {hz:.6f}\n")
            else:
                x, y = obs["pos"][:2]
                f.write(
                    f"cylinder, {name}, {x:.6f}, {y:.6f}, "
                    f"{obs['radius']:.6f}, {obs['half_height']:.6f}\n"
                )


def solve_and_export(
    scenario_number: int,
    scenario: gcr.Scenario,
    params: gcr.PlannerParams,
    output_dir: Path,
) -> dict:
    planner = gcr.GCRPlanner(scenario, params)
    result = planner.plan()
    path = result.path if result.path else result.raw_path

    row = dict(result.metrics)
    row["scenario_number"] = scenario_number
    row["waypoint_csv"] = ""
    row["obstacle_csv"] = ""

    if not result.success or not path:
        status = "timeout" if result.timed_out else "failure"
        print(f"{scenario_number}: {scenario.name}: {status}, no CSV written")
        return row

    cable_length = _cable_length_2d(path[0])
    cable_nodes = len(path[0])
    cstart_nodes_3d = np.asarray([map_2d_to_3d(p) for p in path[0]], dtype=float)
    cable_profiles_3d = [np.asarray([map_2d_to_3d(p) for p in C], dtype=float) for C in path]
    waypoints = _assign_endpoints(path, **_execution_options_for_scenario(scenario.name))
    csv_path = output_dir / f"scenario_{scenario_number:02d}_{scenario.name}_waypoints.csv"
    obstacle_path = output_dir / f"scenario_{scenario_number:02d}_{scenario.name}_obstacles.csv"
    write_waypoint_csv(
        csv_path,
        scenario_number,
        scenario,
        cable_length,
        cable_nodes,
        cstart_nodes_3d,
        waypoints,
        cable_profiles_3d,
    )
    write_obstacle_csv(obstacle_path, scenario)
    row["waypoint_csv"] = str(csv_path)
    row["obstacle_csv"] = str(obstacle_path)
    row["mujoco_cable_length_m"] = cable_length
    row["mujoco_cable_nodes"] = cable_nodes
    row["mujoco_cable_segments"] = cable_nodes - 1
    print(
        f"{scenario_number}: {scenario.name}: success, "
        f"{len(path)} cable configs -> {len(waypoints)} MuJoCo waypoints, "
        f"wrote {csv_path} and {obstacle_path}"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.endpoint_reorder_cost_ratio <= 0.0:
        raise ValueError("--endpoint-reorder-cost-ratio must be positive.")
    if args.robot_tcp_clearance < 0.0:
        raise ValueError("--robot-tcp-clearance must be non-negative.")
    if args.robot_obstacle_clearance is not None and args.robot_obstacle_clearance < 0.0:
        raise ValueError("--robot-obstacle-clearance must be non-negative.")
    if args.robot_swept_tcp_clearance < 0.0:
        raise ValueError("--robot-swept-tcp-clearance must be non-negative.")
    output_dir = Path(args.output_dir) if args.output_dir else results_dir() / "gcr_waypoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_limit = None if args.scenario_time_limit <= 0.0 else args.scenario_time_limit
    params = gcr.PlannerParams(
        max_iterations=args.max_iterations,
        scenario_time_limit_s=time_limit,
        shortcut_refinement=not args.no_shortcut_refinement,
        endpoint_reorder=args.endpoint_reorder,
        robot_execution_feasibility=args.robot_execution_feasibility,
        robot_tcp_clearance=args.robot_tcp_clearance,
        robot_obstacle_clearance=args.robot_obstacle_clearance,
        robot_swept_tcp_clearance=args.robot_swept_tcp_clearance,
        preserve_terminal_order=not args.allow_terminal_reversal,
        endpoint_reorder_cost_ratio=args.endpoint_reorder_cost_ratio,
        endpoint_reorder_tangent_margin=args.endpoint_reorder_tangent_margin,
    )
    scenarios = build_final_scenarios(args.nodes)
    if args.seed_offset:
        for scenario in scenarios:
            scenario.seed += args.seed_offset
    gcr.validate_scenarios(scenarios, params)

    metrics = []
    for idx in selected_indices(args.scenario, len(scenarios)):
        metrics.append(solve_and_export(idx + 1, scenarios[idx], params, output_dir))

    with (output_dir / "export_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Exported final catalogue metrics to {output_dir / 'export_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
