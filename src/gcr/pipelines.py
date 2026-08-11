"""Multi-step pipelines that reproduce the paper artifacts.

``paper-report`` is the cross-platform replacement for
``scripts/plot_export_from_benchmark.ps1`` and ``paper-replay`` the replacement
for ``scripts/run_mujoco_paper_10_fewest.ps1``. The flag sets used for the
published results are module constants so they are stated once.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gcr.paths import default_benchmark_dir, results_dir

__all__ = ["EXPORT_REORDER_FLAGS", "REPLAY_FLAGS", "paper_report_main", "paper_replay_main"]

#: Endpoint-reordering flags applied when exporting selected paths for MuJoCo.
EXPORT_REORDER_FLAGS = (
    "--endpoint-reorder",
    "--allow-terminal-reversal",
    "--endpoint-reorder-cost-ratio", "0.75",
    "--endpoint-reorder-tangent-margin", "0.05",
)

#: MuJoCo replay flags used for the recorded paper videos and settled errors.
REPLAY_FLAGS = (
    "--endpoint-filter", "choose-global-swap",
    "--yaw-mode", "simultaneous",
    "--cable-attachment-mode", "site_weld",
    "--cable-weld-torquescale", "6.0",
    "--tcp-schedule", "auto-stagger",
    "--side-correction-moving-robot", "right",
)


def paper_report_main(argv: list[str] | None = None) -> int:
    """Select extreme solutions, build the paper report, export MuJoCo waypoints."""
    parser = argparse.ArgumentParser(
        prog="gcr paper-report",
        description="Run the full analysis chain over a benchmark folder: "
                    "solution selection, paper plots/tables, and MuJoCo waypoint export.",
    )
    parser.add_argument(
        "benchmark_dir",
        nargs="?",
        default=None,
        help="Benchmark folder to analyse. Default: results/benchmark_rerun",
    )
    parser.add_argument(
        "--paper-output-dir",
        default=None,
        help="Folder for paper plots and tables. Default: results/paper_report/generated",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Folder for exported MuJoCo CSVs. Default: <benchmark_dir>/mujoco_selected_waypoints",
    )
    args = parser.parse_args(argv)

    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else results_dir() / "benchmark_rerun"
    paper_dir = Path(args.paper_output_dir) if args.paper_output_dir else results_dir() / "paper_report" / "generated"
    export_dir = Path(args.export_dir) if args.export_dir else benchmark_dir / "mujoco_selected_waypoints"

    if not benchmark_dir.is_dir():
        raise SystemExit(f"Benchmark folder not found: {benchmark_dir}")

    from gcr.analysis import export_cpp_selected_waypoints, generate_paper_results, plot_cpp_solution_extremes

    print("[1/3] Selecting fastest / fewest-config / most-config solutions...")
    code = plot_cpp_solution_extremes.main([str(benchmark_dir)])
    if code:
        return code

    print("[2/3] Generating paper plots and tables...")
    code = generate_paper_results.main([str(benchmark_dir), "--output-dir", str(paper_dir)])
    if code:
        return code

    print("[3/3] Exporting selected solutions as MuJoCo waypoint CSVs...")
    code = export_cpp_selected_waypoints.main([str(benchmark_dir), str(export_dir), *EXPORT_REORDER_FLAGS])
    if code:
        return code

    print(f"Paper plots/tables: {paper_dir}")
    print(f"MuJoCo waypoint export: {export_dir}")
    return 0


def paper_replay_main(argv: list[str] | None = None) -> int:
    """Replay every fewest-configuration solution in MuJoCo."""
    parser = argparse.ArgumentParser(
        prog="gcr paper-replay",
        description="Replay the ten fewest-configuration solutions in MuJoCo, "
                    "recording four-corner videos and settled cable-shape errors.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="Benchmark folder holding mujoco_selected_waypoints/. Default: results/paper_benchmark_1000",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for videos and final-error JSON. Default: results/replay_outputs",
    )
    parser.add_argument("--steps-per-waypoint", type=int, default=300)
    parser.add_argument("--final-error-wait", type=float, default=10.0, help="Settling time before measuring shape error.")
    parser.add_argument("--video-max-seconds", type=float, default=60.0)
    parser.add_argument("--interactive", action="store_true", help="Open the viewer instead of recording videos.")
    parser.add_argument("--selection", default="fewest_configs", help="Selection to replay (default: fewest_configs).")
    args = parser.parse_args(argv)

    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else default_benchmark_dir()
    output_dir = Path(args.output_dir) if args.output_dir else results_dir() / "replay_outputs"
    waypoint_dir = benchmark_dir / "mujoco_selected_waypoints"

    waypoints = sorted(waypoint_dir.glob(f"*_{args.selection}_waypoints.csv"))
    if not waypoints:
        raise SystemExit(f"No *_{args.selection}_waypoints.csv files found in {waypoint_dir}")

    error_dir = output_dir / "final_errors"
    video_dir = output_dir / "videos"
    error_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    for waypoint in waypoints:
        obstacle = waypoint.with_name(waypoint.name.replace("_waypoints.csv", "_obstacles.csv"))
        if not obstacle.is_file():
            raise SystemExit(f"Missing obstacle CSV for {waypoint.name}: {obstacle}")

        command = [
            sys.executable, "-m", "gcr", "replay",
            "--csv", str(waypoint),
            "--obstacles", str(obstacle),
            "--steps-per-waypoint", str(args.steps_per_waypoint),
            "--final-error-wait", str(args.final_error_wait),
            "--final-error-json", str(error_dir / f"{waypoint.stem}_final_error.json"),
            *REPLAY_FLAGS,
        ]
        if not args.interactive:
            command += [
                "--no-viewer",
                "--video-four-corners",
                "--video-max-seconds", str(args.video_max_seconds),
                "--video-path", str(video_dir / f"{waypoint.stem}_four_corners.mp4"),
            ]

        print(f"Running MuJoCo replay: {waypoint.name}", flush=True)
        # One process per replay, so a single failed scene cannot take the batch down.
        code = subprocess.run(command).returncode
        if code:
            return code

    print(f"Final error JSON files: {error_dir}")
    if not args.interactive:
        print(f"Four-corner MP4 videos: {video_dir}")
    return 0
