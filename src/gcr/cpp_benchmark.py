"""Build and run the C++17 GCR benchmark.

This is the cross-platform replacement for ``scripts/run_benchmark_paper.ps1``:
the paper planner flags live in :data:`PAPER_PLANNER_FLAGS` so the reported
benchmark configuration is defined in exactly one place.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from gcr.paths import cpp_binary, cpp_dir, repo_root, results_dir

__all__ = ["PAPER_SCENARIOS", "PAPER_PLANNER_FLAGS", "build", "run", "build_main", "main"]

#: Source-scenario numbers of the ten paper scenarios.
PAPER_SCENARIOS = (2, 3, 5, 7, 8, 9, 10, 11, 12, 14)

#: Planner flags used for the archived 1000-run paper benchmark.
PAPER_PLANNER_FLAGS = (
    "--endpoint-reorder",
    "--allow-terminal-reversal",
    "--endpoint-reorder-cost-ratio", "0.75",
    "--endpoint-reorder-tangent-margin", "0.05",
    "--robot-execution-feasibility",
    "--robot-tcp-clearance", "0.035",
    "--robot-obstacle-clearance", "0.003",
    "--robot-swept-tcp-clearance", "0.015",
)


def build(force: bool = False) -> Path:
    """Build the benchmark binary if needed and return its path."""
    existing = cpp_binary()
    if existing is not None and not force:
        return existing

    source_dir = cpp_dir()
    build_dir = source_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        subprocess.run([str(source_dir / "build_msvc.bat")], cwd=source_dir, check=True)
    elif shutil.which("cmake"):
        subprocess.run(
            ["cmake", "-S", str(source_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release"], check=True)
    else:
        compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            raise SystemExit(
                "No C++ toolchain found. Install CMake and a C++17 compiler, or on "
                "Windows the Visual Studio 2022 C++ Build Tools."
            )
        subprocess.run(
            [compiler, "-O3", "-std=c++17", "-Wall", "-Wextra",
             str(source_dir / "src" / "gcr_benchmark.cpp"),
             "-o", str(build_dir / "gcr_benchmark")],
            check=True,
        )

    built = cpp_binary()
    if built is None:
        raise SystemExit(f"Build finished but no benchmark binary was found under {build_dir}")
    return built


def run(binary: Path, output_dir: Path, arguments: list[str]) -> int:
    """Run the benchmark binary and stream its output."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [str(binary), str(output_dir), *arguments]
    print(f"Running: {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=repo_root()).returncode


def build_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gcr build-cpp", description="Build the C++ GCR benchmark.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a binary already exists.")
    args = parser.parse_args(argv)
    binary = build(force=args.force)
    print(f"C++ benchmark binary: {binary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gcr benchmark",
        description="Run the C++ GCR benchmark with the paper planner configuration.",
        epilog="Unrecognized options are forwarded to the benchmark binary unchanged.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Benchmark output folder. Default: results/benchmark_rerun",
    )
    parser.add_argument("--runs", type=int, default=1000, help="Runs per scenario (default: 1000).")
    parser.add_argument("--max-iterations", type=int, default=100000, help="Iteration cap per run (default: 100000).")
    parser.add_argument("--scenario-time-limit", type=float, default=120.0, help="Seconds per run; 0 disables (default: 120).")
    parser.add_argument(
        "--scenario",
        type=int,
        action="append",
        help=f"Source scenario number; repeatable. Default: the paper set {', '.join(map(str, PAPER_SCENARIOS))}.",
    )
    parser.add_argument(
        "--no-save-all-success-paths",
        action="store_true",
        help="Skip writing one CSV per successful run (the snapshot in results/ was pruned this way).",
    )
    parser.add_argument("--no-build", action="store_true", help="Fail instead of building when no binary exists.")
    args, passthrough = parser.parse_known_args(argv)

    binary = cpp_binary()
    if binary is None:
        if args.no_build:
            raise SystemExit("No benchmark binary found. Run 'gcr build-cpp' first.")
        binary = build()

    output_dir = Path(args.output_dir) if args.output_dir else results_dir() / "benchmark_rerun"
    scenarios = args.scenario or list(PAPER_SCENARIOS)

    arguments: list[str] = []
    for number in scenarios:
        arguments += ["--scenario", str(number)]
    arguments += [
        "--runs", str(args.runs),
        "--max-iterations", str(args.max_iterations),
        "--scenario-time-limit", str(args.scenario_time_limit),
        *PAPER_PLANNER_FLAGS,
    ]
    if not args.no_save_all_success_paths:
        arguments.append("--save-all-success-paths")
    arguments += passthrough

    code = run(binary, output_dir, arguments)
    if code == 0:
        print(f"Benchmark output: {output_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
