"""The ``gcr`` command line entry point.

Every subcommand is a thin wrapper: it imports one module lazily and hands the
remaining arguments to that module's ``main(argv)``, so ``gcr <command> --help``
prints the wrapped tool's own help and no option has to be duplicated here.
Lazy importing also keeps ``gcr --help`` free of matplotlib and MuJoCo.
"""

from __future__ import annotations

import difflib
import importlib
import sys
from dataclasses import dataclass
from typing import Callable

from gcr import __version__

Runner = Callable[[list[str]], int]


@dataclass(frozen=True)
class Command:
    """One ``gcr`` subcommand."""

    name: str
    group: str
    summary: str
    module: str
    function: str = "main"

    def run(self, argv: list[str]) -> int:
        entry = getattr(importlib.import_module(self.module), self.function)
        # argparse derives its program name from sys.argv[0], so the wrapped
        # tools print "usage: gcr <command>" instead of "usage: __main__.py".
        original = sys.argv[0]
        sys.argv[0] = f"gcr {self.name}"
        try:
            return int(entry(argv) or 0)
        finally:
            sys.argv[0] = original


COMMANDS: tuple[Command, ...] = (
    # Planning and benchmarking
    Command("plan", "Planning and benchmarking",
            "Run the Python reference planner over the scenario catalogue",
            "gcr.planner.gcrbup"),
    Command("build-cpp", "Planning and benchmarking",
            "Build the C++17 benchmark binary",
            "gcr.cpp_benchmark", "build_main"),
    Command("benchmark", "Planning and benchmarking",
            "Run the C++ benchmark with the paper planner configuration",
            "gcr.cpp_benchmark"),

    # Analysis and figures
    Command("paper-report", "Analysis and figures",
            "Full chain: select solutions, build paper plots/tables, export waypoints",
            "gcr.pipelines", "paper_report_main"),
    Command("report", "Analysis and figures",
            "Generate the paper plots and benchmark table from a benchmark folder",
            "gcr.analysis.generate_paper_results"),
    Command("plot-benchmark", "Analysis and figures",
            "Plot success rate, runtime, and configuration-count summaries",
            "gcr.analysis.plot_cpp_benchmark_report"),
    Command("plot-paths", "Analysis and figures",
            "Plot first-success solution paths per scenario",
            "gcr.analysis.plot_cpp_solution_paths"),
    Command("plot-extremes", "Analysis and figures",
            "Select and plot fastest / fewest-config / most-config solutions",
            "gcr.analysis.plot_cpp_solution_extremes"),
    Command("plot-outliers", "Analysis and figures",
            "Plot runtime distributions with and without Tukey upper outliers",
            "gcr.analysis.plot_runtime_distributions"),
    Command("workspace-figure", "Analysis and figures",
            "Generate the DLO workspace problem-formulation figures",
            "gcr.planner.dlo_workspace_figure"),

    # MuJoCo replay
    Command("export-waypoints", "MuJoCo replay",
            "Export selected benchmark solutions as MuJoCo waypoint/obstacle CSVs",
            "gcr.analysis.export_cpp_selected_waypoints"),
    Command("export-scenario-waypoints", "MuJoCo replay",
            "Plan scenarios in Python and export their waypoint/obstacle CSVs",
            "gcr.mujoco_exec.export_gcr_waypoints"),
    Command("paper-replay", "MuJoCo replay",
            "Replay every fewest-configuration solution and record videos/errors",
            "gcr.pipelines", "paper_replay_main"),
    Command("replay", "MuJoCo replay",
            "Replay one waypoint CSV on the two-UR10e cable scene",
            "gcr.mujoco_exec.run_csv"),
    Command("replay-robot-aware", "MuJoCo replay",
            "Replay one waypoint CSV with robot-aware checks and regrasps",
            "gcr.mujoco_exec.run_csv_robot_aware"),
    Command("replay-trajectory", "MuJoCo replay",
            "Replay a recorded robot-aware trajectory JSON",
            "gcr.mujoco_exec.replay_robot_aware_trajectory"),
    Command("execute-plan", "MuJoCo replay",
            "Plan a scenario and execute it in MuJoCo in one process",
            "gcr.mujoco_exec.execute_plan"),
    Command("view-scenario", "MuJoCo replay",
            "Open an exported scenario in the interactive MuJoCo viewer",
            "gcr.mujoco_exec.view_gcr_scenario"),

    # Repository
    Command("verify", "Repository",
            "Verify the checksums recorded in SHA256SUMS",
            "gcr.verify"),
)

_BY_NAME = {command.name: command for command in COMMANDS}

_GROUP_ORDER = ("Planning and benchmarking", "Analysis and figures", "MuJoCo replay", "Repository")


def format_help() -> str:
    width = max(len(command.name) for command in COMMANDS)
    lines = [
        "usage: gcr <command> [options]",
        "",
        "Guided Cable Reconfiguration planner, benchmark, and MuJoCo replay.",
        "",
        "Run 'gcr <command> --help' for the options of a single command.",
    ]
    for group in _GROUP_ORDER:
        lines += ["", f"{group}:"]
        for command in COMMANDS:
            if command.group == group:
                lines.append(f"  {command.name.ljust(width)}  {command.summary}")
    lines += [
        "",
        "Options:",
        f"  {'-h, --help'.ljust(width)}  Show this message.",
        f"  {'-V, --version'.ljust(width)}  Show the package version.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)

    if not arguments or arguments[0] in ("-h", "--help", "help"):
        print(format_help())
        return 0
    if arguments[0] in ("-V", "--version"):
        print(f"gcr {__version__}")
        return 0

    name, rest = arguments[0], arguments[1:]
    command = _BY_NAME.get(name)
    if command is None:
        print(f"gcr: unknown command '{name}'", file=sys.stderr)
        close = difflib.get_close_matches(name, list(_BY_NAME), n=3)
        if close:
            print(f"Did you mean: {', '.join(close)}?", file=sys.stderr)
        print("Run 'gcr --help' for the list of commands.", file=sys.stderr)
        return 2

    return command.run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
