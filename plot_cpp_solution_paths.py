"""Plot one C++-generated solution path for each final GCR scenario.

The C++ runner writes one path file per scenario folder:

    <benchmark_dir>/01_scenario_name/first_success_path.csv

This script overlays those C++ path configurations on the matching final
scenario obstacles.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

import gcrbup as gcr
from final_scenarios import build_final_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot C++ first-success solution paths.")
    parser.add_argument("input_dir", help="C++ benchmark output folder containing scenario subfolders.")
    parser.add_argument("--output-dir", default=None, help="Default: <input_dir>/solution_plots")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--swap-start-goal", action="store_true", help="Display scenario start/goal swapped to match a benchmark run made with --swap-start-goal.")
    return parser.parse_args()


def swap_start_goal_for_display(scenarios: list[gcr.Scenario]) -> list[gcr.Scenario]:
    for scenario in scenarios:
        scenario.start, scenario.goal = scenario.goal.copy(), scenario.start.copy()
        scenario.title = f"{scenario.title} (start/goal swapped)"
    return scenarios


def read_path_csv(path: Path) -> list[np.ndarray]:
    grouped: dict[int, list[tuple[int, float, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(row for row in fh if not row.lstrip().startswith("#"))
        for row in reader:
            config_index = int(row["config_index"])
            node_index = int(row["node_index"])
            grouped.setdefault(config_index, []).append((node_index, float(row["x"]), float(row["y"])))

    configs: list[np.ndarray] = []
    for config_index in sorted(grouped):
        nodes = sorted(grouped[config_index], key=lambda item: item[0])
        configs.append(np.asarray([[x, y] for _, x, y in nodes], dtype=float))
    return configs


def scenario_dir_number(path: Path) -> int | None:
    prefix = path.name.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def draw_scene(ax: plt.Axes, scenario: gcr.Scenario) -> None:
    xmin, xmax, ymin, ymax = scenario.workspace
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="#f8fafc",
            edgecolor="#dc2626",
            linewidth=1.5,
            zorder=0,
        )
    )
    for polygon in scenario.obstacles.polygons:
        ax.add_patch(Polygon(polygon, closed=True, facecolor="#3f3f46", edgecolor="#18181b", linewidth=1.0, zorder=2))
    for center, radius in scenario.obstacles.circles:
        ax.add_patch(Circle(center, radius, facecolor="#3f3f46", edgecolor="#18181b", linewidth=1.0, zorder=2))
    ax.set_xlim(xmin - 0.035, xmax + 0.035)
    ax.set_ylim(ymin - 0.035, ymax + 0.035)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.75)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_cable(ax: plt.Axes, cable: np.ndarray, color: str, label: str | None, linewidth: float, alpha: float, zorder: int) -> None:
    ax.plot(cable[:, 0], cable[:, 1], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round", label=label, zorder=zorder)
    ax.scatter(cable[:, 0], cable[:, 1], s=18, facecolor=color, edgecolor="white", linewidth=0.45, alpha=alpha, zorder=zorder + 1)


def draw_swept_corridors(ax: plt.Axes, configs: list[np.ndarray], *, label: bool) -> None:
    if len(configs) < 2:
        return
    for idx in range(len(configs) - 1):
        corridor = gcr.connection_polygon(configs[idx], configs[idx + 1])
        ax.add_patch(
            Polygon(
                corridor,
                closed=True,
                facecolor="#facc15",
                edgecolor="#facc15",
                linewidth=0.8,
                alpha=0.16,
                label="swept path corridor" if label and idx == 0 else None,
                zorder=1,
            )
        )


def plot_solution(scenario: gcr.Scenario, scenario_number: int, configs: list[np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    draw_scene(ax, scenario)

    if configs:
        draw_swept_corridors(ax, configs, label=True)
        cmap = plt.get_cmap("viridis")
        for idx, cable in enumerate(configs):
            frac = 0.0 if len(configs) == 1 else idx / (len(configs) - 1)
            is_endpoint = idx == 0 or idx == len(configs) - 1
            plot_cable(
                ax,
                cable,
                color=cmap(frac),
                label="C++ path profiles" if idx == 0 else None,
                linewidth=2.5 if is_endpoint else 1.6,
                alpha=1.0 if is_endpoint else 0.64,
                zorder=5 + idx,
            )

    plot_cable(ax, scenario.start, "#2563eb", "start", 2.8, 1.0, 30)
    plot_cable(ax, scenario.goal, "#7c3aed", "goal", 2.8, 1.0, 31)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.95)
    ax.set_title(f"{scenario_number:02d}: {scenario.title} | C++ solution path, {len(configs)} configs", fontsize=11.5)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_overview(items: list[tuple[int, gcr.Scenario, list[np.ndarray]]], output_path: Path) -> None:
    n = len(items)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.3 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)

    for ax in axes_arr:
        ax.axis("off")

    for ax, (number, scenario, configs) in zip(axes_arr, items):
        ax.axis("on")
        draw_scene(ax, scenario)
        draw_swept_corridors(ax, configs, label=False)
        for idx, cable in enumerate(configs):
            frac = 0.0 if len(configs) == 1 else idx / (len(configs) - 1)
            ax.plot(cable[:, 0], cable[:, 1], color=plt.get_cmap("viridis")(frac), linewidth=1.25, alpha=0.68, zorder=4)
        ax.plot(scenario.start[:, 0], scenario.start[:, 1], color="#2563eb", linewidth=2.0, zorder=20)
        ax.plot(scenario.goal[:, 0], scenario.goal[:, 1], color="#7c3aed", linewidth=2.0, zorder=21)
        ax.set_title(f"{number:02d}: {scenario.name}\n{len(configs)} configs", fontsize=9.0)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("C++ First Successful Solution Path Per Scenario", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "solution_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_final_scenarios(args.nodes)
    if args.swap_start_goal:
        scenarios = swap_start_goal_for_display(scenarios)
    scenario_by_number = {idx + 1: scenario for idx, scenario in enumerate(scenarios)}

    items: list[tuple[int, gcr.Scenario, list[np.ndarray]]] = []
    for scenario_dir in sorted([path for path in input_dir.iterdir() if path.is_dir()]):
        number = scenario_dir_number(scenario_dir)
        if number is None or number not in scenario_by_number:
            continue
        path_csv = scenario_dir / "first_success_path.csv"
        if not path_csv.exists():
            print(f"Skipping {scenario_dir.name}: no first_success_path.csv")
            continue
        scenario = scenario_by_number[number]
        configs = read_path_csv(path_csv)
        output_path = output_dir / f"{number:02d}_{scenario.name}_cpp_solution.png"
        plot_solution(scenario, number, configs, output_path)
        items.append((number, scenario, configs))

    if not items:
        raise SystemExit(
            "No C++ path CSV files found. Rebuild the C++ runner and rerun it so each scenario folder gets first_success_path.csv."
        )

    plot_overview(items, output_dir / "cpp_solution_paths_overview.png")
    print(f"Wrote {len(items)} scenario solution plot(s) to: {output_dir}")
    print(f"Overview: {output_dir / 'cpp_solution_paths_overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
