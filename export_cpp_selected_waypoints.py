"""Export selected C++ benchmark paths to MuJoCo waypoint CSV files.

Input is a C++ benchmark folder that has already been processed by
``plot_cpp_solution_extremes.py``.  The script reads:

    <benchmark_dir>/selected_solution_paths/selected_solution_summary.csv
    <benchmark_dir>/<scenario_dir>/run_paths/run_####_success_path.csv

and writes MuJoCo-ready waypoint/obstacle CSVs for the selected paths:

* fewest_configs
* most_configs
* fastest
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import gcrbup as gcr
from final_scenarios import build_final_scenarios, scenario_source_labels
from mujoco_exec.execute_plan import (
    _assign_endpoints,
    _cable_length_2d,
    _execution_options_for_scenario,
    map_2d_to_3d,
)
from mujoco_exec.export_gcr_waypoints import write_obstacle_csv, write_waypoint_csv
from plot_cpp_solution_paths import read_path_csv


SELECTIONS = ("fewest_configs", "most_configs", "fastest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected C++ solution paths for MuJoCo replay.")
    parser.add_argument("benchmark_dir", help="C++ benchmark folder with selected_solution_paths/summary CSV.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output folder. Default: <benchmark_dir>/mujoco_selected_waypoints",
    )
    parser.add_argument("--scenario", type=int, action="append", help="1-based scenario number. Repeat for several.")
    parser.add_argument("--selection", choices=SELECTIONS, action="append", help="Selection to export. Default: all three.")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--endpoint-reorder", action="store_true", help="Apply endpoint-continuity reordering to the selected path before writing MuJoCo CSVs.")
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
    parser.add_argument(
        "--robot-base-aware-endpoint-reorder",
        action="store_true",
        help="When --endpoint-reorder is enabled, also prefer endpoint assignments whose robot base-to-TCP directions are execution friendly.",
    )
    parser.add_argument(
        "--robot-base-vector-endpoint-reorder",
        action="store_true",
        help="When --endpoint-reorder is enabled, also reverse profiles that cause a large left-to-right TCP vector turn relative to the robot bases.",
    )
    parser.add_argument(
        "--robot-tcp-vector-sector-reorder",
        action="store_true",
        help="When --endpoint-reorder is enabled, prefer endpoint orders whose right-TCP-to-left-TCP vector stays in the base-adjusted left-facing sector.",
    )
    parser.add_argument(
        "--robot-tcp-vector-sector-feasibility",
        action="store_true",
        help="Reject export-time paths whose right-TCP-to-left-TCP vector leaves the base-adjusted sector.",
    )
    parser.add_argument(
        "--robot-base-vector-max-turn-deg",
        type=float,
        default=100.0,
        help="Maximum preferred turn of the left-to-right TCP vector between adjacent profiles before reversed endpoint order is preferred.",
    )
    parser.add_argument(
        "--robot-base-vector-margin-deg",
        type=float,
        default=5.0,
        help="Minimum angular improvement required before vector-based endpoint reordering prefers a reversal.",
    )
    parser.add_argument(
        "--robot-tcp-vector-sector-margin-deg",
        type=float,
        default=5.0,
        help="Tolerance/improvement margin for --robot-tcp-vector-sector-reorder.",
    )
    parser.add_argument(
        "--max-cable-axis-turn-deg",
        type=float,
        default=180.0,
        help="Reject adjacent profile transitions whose cable end-to-end axis turns more than this angle when export-time reordering is validated.",
    )
    parser.add_argument(
        "--robot-base-arm-clearance",
        type=float,
        default=gcr.ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
        help="Minimum sampled clearance between the left and right base-to-TCP rays used by --robot-base-aware-endpoint-reorder.",
    )
    parser.add_argument(
        "--robot-base-transition-samples",
        type=int,
        default=gcr.ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT,
        help="Number of interpolation samples for the base-to-TCP ray clearance check.",
    )
    parser.add_argument("--robot-execution-feasibility", action="store_true", help="Use robot-execution transition checks while applying export-time endpoint reordering.")
    parser.add_argument("--robot-tcp-clearance", type=float, default=0.0, help="TCP radius for same-time left/right endpoint separation.")
    parser.add_argument("--robot-obstacle-clearance", type=float, default=None, help="TCP-vs-obstacle sweep clearance.")
    parser.add_argument("--robot-swept-tcp-clearance", type=float, default=0.0, help="Optional TCP radius for swept left-path vs right-path clearance.")
    return parser.parse_args()


def selected_numbers(requested: Iterable[int] | None, n_scenarios: int) -> set[int] | None:
    if not requested:
        return None
    numbers = set()
    for number in requested:
        if number < 1 or number > n_scenarios:
            raise SystemExit(f"--scenario must be in 1..{n_scenarios}; got {number}")
        numbers.add(number)
    return numbers


def scenario_dir(benchmark_dir: Path, number: int) -> Path:
    prefix = f"{number:02d}_"
    matches = sorted(path for path in benchmark_dir.iterdir() if path.is_dir() and path.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"No scenario folder found for scenario {number} in {benchmark_dir}.")
    return matches[0]


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def clean_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def export_one(
    benchmark_dir: Path,
    output_dir: Path,
    scenario_number: int,
    scenario: gcr.Scenario,
    row: dict[str, str],
    source: str,
    params: gcr.PlannerParams,
) -> dict:
    selection = row["selection"]
    run_index = int(row["run_index"])
    path_csv = scenario_dir(benchmark_dir, scenario_number) / "run_paths" / f"run_{run_index:04d}_success_path.csv"
    if not path_csv.exists():
        raise FileNotFoundError(
            f"Missing {path_csv}. Re-run the C++ benchmark with --save-all-success-paths, "
            "then rerun plot_cpp_solution_extremes.py."
        )

    path = read_path_csv(path_csv)
    if not path:
        raise ValueError(f"{path_csv} has no cable profiles.")
    endpoint_reversals = 0
    if params.endpoint_reorder:
        original_path = [config.copy() for config in path]
        validation_samples = max(params.transition_samples, params.shortcut_validation_samples)
        path, endpoint_reversals = gcr.reorder_endpoint_correspondence(
            path,
            scenario.obstacles,
            validation_samples,
            params.robot_execution_feasibility,
            params.robot_tcp_clearance,
            params.robot_obstacle_clearance,
            params.robot_swept_tcp_clearance,
            params.preserve_terminal_order,
            params.endpoint_reorder_cost_ratio,
            params.endpoint_reorder_tangent_margin,
            params.robot_base_aware_endpoint_reorder,
            params.robot_base_vector_endpoint_reorder,
            params.robot_tcp_vector_sector_reorder,
            params.robot_tcp_vector_sector_feasibility,
            params.robot_base_vector_max_turn_rad,
            params.robot_base_vector_margin_rad,
            params.robot_tcp_vector_sector_margin_rad,
            params.max_cable_axis_turn_rad,
        )
        robot_base_ok = (
            (not params.robot_base_aware_endpoint_reorder)
            or gcr.robot_base_path_sequence_clearance_ok_either_assignment(
                path,
                params.robot_base_arm_clearance,
                params.robot_base_transition_samples,
            )
        )
        tcp_sector_ok = (
            (not params.robot_tcp_vector_sector_feasibility)
            or gcr.robot_tcp_vector_sector_path_sequence_ok(
                path,
                gcr.endpoint0_is_left_robot(path[0]),
                params.robot_tcp_vector_sector_margin_rad,
                validation_samples,
            )
        )
        if (not robot_base_ok) or (not tcp_sector_ok) or not gcr.path_sequence_free(
            path,
            scenario.obstacles,
            validation_samples,
            params.robot_execution_feasibility,
            params.robot_tcp_clearance,
            params.robot_obstacle_clearance,
            params.robot_swept_tcp_clearance,
            params.max_cable_axis_turn_rad,
        ):
            path = original_path
            endpoint_reversals = 0

    cable_length = _cable_length_2d(path[0])
    cable_nodes = len(path[0])
    cstart_nodes_3d = np.asarray([map_2d_to_3d(p) for p in path[0]], dtype=float)
    cable_profiles_3d = [np.asarray([map_2d_to_3d(p) for p in C], dtype=float) for C in path]
    waypoints = _assign_endpoints(path, **_execution_options_for_scenario(scenario.name))

    stem = f"scenario_{scenario_number:02d}_{scenario.name}_{selection}"
    waypoint_csv = output_dir / f"{stem}_waypoints.csv"
    obstacle_csv = output_dir / f"{stem}_obstacles.csv"
    write_waypoint_csv(
        waypoint_csv,
        scenario_number,
        scenario,
        cable_length,
        cable_nodes,
        cstart_nodes_3d,
        waypoints,
        cable_profiles_3d,
    )
    write_obstacle_csv(obstacle_csv, scenario)

    return {
        "scenario_number": scenario_number,
        "scenario": scenario.name,
        "source": source,
        "selection": selection,
        "run_index": run_index,
        "display_run": int(row["display_run"]),
        "seed": int(row["seed"]),
        "refined_path_configs": int(row["refined_path_configs"]),
        "raw_path_configs": int(row["raw_path_configs"]),
        "time_s": float(row["time_s"]),
        "iterations": int(row["iterations"]),
        "path_cost_J": float(row["path_cost_J"]),
        "cpp_path_csv": str(path_csv),
        "waypoint_csv": str(waypoint_csv),
        "obstacle_csv": str(obstacle_csv),
        "mujoco_cable_length_m": cable_length,
        "mujoco_cable_nodes": cable_nodes,
        "mujoco_cable_segments": cable_nodes - 1,
        "mujoco_waypoints": len(waypoints),
        "endpoint_order_reversals_applied": endpoint_reversals,
    }


def write_command_files(output_dir: Path, rows: list[dict]) -> None:
    ps1 = output_dir / "run_selected_mujoco_examples.ps1"
    txt = output_dir / "run_selected_mujoco_commands.txt"
    lines = [
        "# Run one line at a time from the repository root.",
        "# Remove --no-viewer and --video-four-corners when you want the interactive viewer.",
        "# Final error JSON files are written under final_errors/.",
        "# Four-corner videos are written under videos/ at 5x simulated-time playback speed.",
        "",
    ]
    for row in rows:
        waypoint = Path(row["waypoint_csv"])
        obstacle = Path(row["obstacle_csv"])
        final_error_json = output_dir / "final_errors" / f"{waypoint.stem}_final_error.json"
        video_path = output_dir / "videos" / f"{waypoint.stem}_four_corners.mp4"
        lines.append(
            "python .\\mujoco_exec\\run_csv.py "
            f"--csv \"{waypoint}\" "
            f"--obstacles \"{obstacle}\" "
            "--endpoint-filter choose-global-swap "
            "--yaw-mode simultaneous "
            "--steps-per-waypoint 300 "
            "--cable-attachment-mode site_weld "
            "--cable-weld-torquescale 6.0 "
            "--final-error-wait 10 "
            f"--final-error-json \"{final_error_json}\" "
            "--tcp-schedule auto-stagger "
            "--side-correction-moving-robot right "
            "--no-viewer "
            "--video-four-corners "
            "--video-speedup 5 "
            f"--video-path \"{video_path}\""
        )
    text = "\n".join(lines) + "\n"
    txt.write_text(text, encoding="utf-8")
    ps1.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.endpoint_reorder_cost_ratio <= 0.0:
        raise ValueError("--endpoint-reorder-cost-ratio must be positive.")
    if args.robot_tcp_clearance < 0.0:
        raise ValueError("--robot-tcp-clearance must be non-negative.")
    if args.robot_obstacle_clearance is not None and args.robot_obstacle_clearance < 0.0:
        raise ValueError("--robot-obstacle-clearance must be non-negative.")
    if args.robot_swept_tcp_clearance < 0.0:
        raise ValueError("--robot-swept-tcp-clearance must be non-negative.")
    if args.robot_base_arm_clearance < 0.0:
        raise ValueError("--robot-base-arm-clearance must be non-negative.")
    if args.robot_base_transition_samples < 1:
        raise ValueError("--robot-base-transition-samples must be at least 1.")
    if args.robot_base_vector_max_turn_deg < 0.0 or args.robot_base_vector_max_turn_deg > 180.0:
        raise ValueError("--robot-base-vector-max-turn-deg must be in [0, 180].")
    if args.robot_base_vector_margin_deg < 0.0:
        raise ValueError("--robot-base-vector-margin-deg must be non-negative.")
    if args.robot_tcp_vector_sector_margin_deg < 0.0:
        raise ValueError("--robot-tcp-vector-sector-margin-deg must be non-negative.")
    if args.max_cable_axis_turn_deg < 0.0 or args.max_cable_axis_turn_deg > 180.0:
        raise ValueError("--max-cable-axis-turn-deg must be in [0, 180].")
    benchmark_dir = Path(args.benchmark_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else benchmark_dir / "mujoco_selected_waypoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = benchmark_dir / "selected_solution_paths" / "selected_solution_summary.csv"
    if not summary_path.exists():
        raise SystemExit(
            f"Missing {summary_path}. Run:\n"
            f"  python .\\plot_cpp_solution_extremes.py {benchmark_dir}"
        )

    scenarios = build_final_scenarios(args.nodes)
    params = gcr.PlannerParams(
        endpoint_reorder=args.endpoint_reorder,
        robot_execution_feasibility=args.robot_execution_feasibility,
        robot_tcp_clearance=args.robot_tcp_clearance,
        robot_obstacle_clearance=args.robot_obstacle_clearance,
        robot_swept_tcp_clearance=args.robot_swept_tcp_clearance,
        preserve_terminal_order=not args.allow_terminal_reversal,
        endpoint_reorder_cost_ratio=args.endpoint_reorder_cost_ratio,
        endpoint_reorder_tangent_margin=args.endpoint_reorder_tangent_margin,
        robot_base_aware_endpoint_reorder=args.robot_base_aware_endpoint_reorder,
        robot_base_vector_endpoint_reorder=args.robot_base_vector_endpoint_reorder,
        robot_tcp_vector_sector_reorder=args.robot_tcp_vector_sector_reorder,
        robot_tcp_vector_sector_feasibility=args.robot_tcp_vector_sector_feasibility,
        robot_base_vector_max_turn_rad=np.deg2rad(args.robot_base_vector_max_turn_deg),
        robot_base_vector_margin_rad=np.deg2rad(args.robot_base_vector_margin_deg),
        robot_tcp_vector_sector_margin_rad=np.deg2rad(args.robot_tcp_vector_sector_margin_deg),
        max_cable_axis_turn_rad=np.deg2rad(args.max_cable_axis_turn_deg),
        robot_base_arm_clearance=args.robot_base_arm_clearance,
        robot_base_transition_samples=args.robot_base_transition_samples,
    )
    gcr.validate_scenarios(scenarios, params)
    source_by_name = scenario_source_labels()
    wanted_numbers = selected_numbers(args.scenario, len(scenarios))
    wanted_selections = set(args.selection or SELECTIONS)

    exported = []
    for row in read_summary(summary_path):
        scenario_number = int(row["scenario_number"])
        selection = row["selection"]
        if wanted_numbers is not None and scenario_number not in wanted_numbers:
            continue
        if selection not in wanted_selections:
            continue
        scenario = scenarios[scenario_number - 1]
        exported.append(
            export_one(
                benchmark_dir,
                output_dir,
                scenario_number,
                scenario,
                row,
                source_by_name.get(scenario.name, "unknown"),
                params,
            )
        )

    if not exported:
        raise SystemExit("No selected solutions matched the requested filters.")

    index_path = output_dir / "selected_mujoco_export_index.json"
    index_path.write_text(json.dumps(exported, indent=2), encoding="utf-8")
    write_command_files(output_dir, exported)

    print(f"Exported {len(exported)} selected MuJoCo waypoint set(s) to: {output_dir}")
    print(f"Index: {index_path}")
    print(f"Commands: {output_dir / 'run_selected_mujoco_commands.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
