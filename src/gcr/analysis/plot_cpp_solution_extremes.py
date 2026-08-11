"""Plot selected C++ solution paths per scenario.

This script expects a C++ benchmark folder produced with:

    gcr_benchmark.exe <out> --runs N --save-all-success-paths

For each scenario, it plots:

* the successful run with the fewest refined configurations;
* the successful run with the most refined configurations;
* the fastest successful run, regardless of configuration count.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gcr.planner import gcrbup as gcr
from gcr.planner.final_scenarios import build_final_scenarios
from gcr.analysis.plot_cpp_solution_paths import draw_scene, draw_swept_corridors, plot_cable, read_path_csv


@dataclass(frozen=True)
class MetricRow:
    scenario_number: int
    scenario: str
    run_index: int
    seed: int
    time_s: float
    iterations: int
    refined_path_configs: int
    raw_path_configs: int
    path_cost_j: float


@dataclass(frozen=True)
class Selection:
    label: str
    stem: str
    row: MetricRow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot min-config, max-config, and fastest C++ solution paths.")
    parser.add_argument("input_dir", help="C++ benchmark output folder.")
    parser.add_argument("--output-dir", default=None, help="Default: <input_dir>/selected_solution_paths")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--swap-start-goal", action="store_true", help="Display scenario start/goal swapped to match a benchmark run made with --swap-start-goal.")
    return parser.parse_args(argv)


def swap_start_goal_for_display(scenarios: list[gcr.Scenario]) -> list[gcr.Scenario]:
    for scenario in scenarios:
        scenario.start, scenario.goal = scenario.goal.copy(), scenario.start.copy()
        scenario.title = f"{scenario.title} (start/goal swapped)"
    return scenarios


def csv_bool(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def csv_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "nan")
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def csv_int(row: dict[str, str], key: str) -> int:
    if row.get(key) is None:
        raise ValueError(f"Missing required CSV field {key!r}")
    return int(float(row[key]))


def read_success_metrics(path: Path) -> list[MetricRow]:
    rows: list[MetricRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not csv_bool(row.get("success", "")):
                continue
            required = ("final_scenario_number", "run_index", "seed", "iterations", "refined_path_configs", "raw_path_configs")
            if any(row.get(key) in {None, ""} for key in required):
                continue
            if "static_collision_free" in row and not csv_bool(row.get("static_collision_free", "")):
                continue
            if "transition_path_free" in row and not csv_bool(row.get("transition_path_free", "")):
                continue
            rows.append(
                MetricRow(
                    scenario_number=csv_int(row, "final_scenario_number"),
                    scenario=row["scenario"],
                    run_index=csv_int(row, "run_index"),
                    seed=csv_int(row, "seed"),
                    time_s=csv_float(row, "time_s"),
                    iterations=csv_int(row, "iterations"),
                    refined_path_configs=csv_int(row, "refined_path_configs"),
                    raw_path_configs=csv_int(row, "raw_path_configs"),
                    path_cost_j=csv_float(row, "path_cost_J"),
                )
            )
    return rows


def scenario_dir(input_dir: Path, number: int) -> Path:
    prefix = f"{number:02d}_"
    matches = sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"Could not find scenario directory for scenario {number}.")
    return matches[0]


def run_path(input_dir: Path, row: MetricRow) -> Path:
    return scenario_dir(input_dir, row.scenario_number) / "run_paths" / f"run_{row.run_index:04d}_success_path.csv"


def choose(rows: list[MetricRow], key: Callable[[MetricRow], tuple]) -> MetricRow:
    return sorted(rows, key=key)[0]


def make_selections(rows: list[MetricRow]) -> list[Selection]:
    return [
        Selection(
            "Fewest configurations",
            "fewest_configs",
            choose(rows, lambda r: (r.refined_path_configs, r.time_s, r.iterations, r.run_index)),
        ),
        Selection(
            "Most configurations",
            "most_configs",
            choose(rows, lambda r: (-r.refined_path_configs, r.time_s, r.iterations, r.run_index)),
        ),
        Selection(
            "Fastest success",
            "fastest",
            choose(rows, lambda r: (r.time_s, r.refined_path_configs, r.iterations, r.run_index)),
        ),
    ]


def plot_one(ax: plt.Axes, scenario: gcr.Scenario, configs: list, selection: Selection) -> None:
    draw_scene(ax, scenario)
    draw_swept_corridors(ax, configs, label=False)
    if configs:
        cmap = plt.get_cmap("viridis")
        for idx, cable in enumerate(configs):
            frac = 0.0 if len(configs) == 1 else idx / (len(configs) - 1)
            is_endpoint = idx == 0 or idx == len(configs) - 1
            plot_cable(
                ax,
                cable,
                color=cmap(frac),
                label="path profiles" if idx == 0 else None,
                linewidth=2.35 if is_endpoint else 1.45,
                alpha=1.0 if is_endpoint else 0.62,
                zorder=5 + idx,
            )
    plot_cable(ax, scenario.start, "#2563eb", "start", 2.4, 1.0, 30)
    plot_cable(ax, scenario.goal, "#7c3aed", "goal", 2.4, 1.0, 31)
    row = selection.row
    ax.set_title(
        f"{selection.label}\n"
        f"run {row.run_index + 1}, configs={row.refined_path_configs}, "
        f"t={row.time_s:.4f}s, iters={row.iterations}",
        fontsize=9.6,
    )


def write_summary(path: Path, selections_by_scenario: list[tuple[int, str, list[Selection]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "scenario_number",
                "scenario",
                "selection",
                "run_index",
                "display_run",
                "seed",
                "refined_path_configs",
                "raw_path_configs",
                "time_s",
                "iterations",
                "path_cost_J",
            ]
        )
        for number, scenario_name, selections in selections_by_scenario:
            for selection in selections:
                row = selection.row
                writer.writerow(
                    [
                        number,
                        scenario_name,
                        selection.stem,
                        row.run_index,
                        row.run_index + 1,
                        row.seed,
                        row.refined_path_configs,
                        row.raw_path_configs,
                        f"{row.time_s:.10g}",
                        row.iterations,
                        f"{row.path_cost_j:.10g}",
                    ]
                )


def write_markdown(path: Path, selections_by_scenario: list[tuple[int, str, list[Selection]]]) -> None:
    lines = [
        "# C++ Selected Solution Paths",
        "",
        "Each scenario image compares the fewest-configuration success, the most-configuration success, and the fastest success.",
        "",
        "## Full-Scenario Overviews",
        "",
        "### Fewest Configurations",
        "",
        "![Fewest configurations](overview_fewest_configs.png)",
        "",
        "### Most Configurations",
        "",
        "![Most configurations](overview_most_configs.png)",
        "",
        "### Fastest Success",
        "",
        "![Fastest success](overview_fastest.png)",
        "",
        "## Per-Scenario Comparisons",
        "",
    ]
    for number, scenario_name, _ in selections_by_scenario:
        image_name = f"{number:02d}_{scenario_name}_selected_paths.png"
        lines.extend([f"## {number:02d}: `{scenario_name}`", "", f"![{scenario_name}]({image_name})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_selection_overview(
    items: list[tuple[int, gcr.Scenario, Selection, list]],
    output_path: Path,
    title: str,
) -> None:
    if not items:
        return
    cols = 3
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15.2, 4.35 * rows), constrained_layout=True)
    axes_arr = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax in axes_arr:
        ax.axis("off")

    for ax, (number, scenario, selection, configs) in zip(axes_arr, items):
        ax.axis("on")
        draw_scene(ax, scenario)
        draw_swept_corridors(ax, configs, label=False)
        if configs:
            cmap = plt.get_cmap("viridis")
            for idx, cable in enumerate(configs):
                frac = 0.0 if len(configs) == 1 else idx / (len(configs) - 1)
                ax.plot(cable[:, 0], cable[:, 1], color=cmap(frac), linewidth=1.2, alpha=0.68, zorder=4)
        ax.plot(scenario.start[:, 0], scenario.start[:, 1], color="#2563eb", linewidth=1.9, zorder=20)
        ax.plot(scenario.goal[:, 0], scenario.goal[:, 1], color="#7c3aed", linewidth=1.9, zorder=21)
        row = selection.row
        ax.set_title(
            f"{number:02d}: {scenario.name}\n"
            f"run {row.run_index + 1}, configs={row.refined_path_configs}, t={row.time_s:.4f}s",
            fontsize=8.9,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(title, fontsize=15.0, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "selected_solution_paths"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = input_dir / "final_metrics_all.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")

    rows = read_success_metrics(metrics_path)
    if not rows:
        raise SystemExit("No successful runs found in final_metrics_all.csv.")

    scenarios = build_final_scenarios(args.nodes)
    if args.swap_start_goal:
        scenarios = swap_start_goal_for_display(scenarios)
    scenario_by_number = {idx + 1: scenario for idx, scenario in enumerate(scenarios)}

    rows_by_scenario: dict[int, list[MetricRow]] = {}
    for row in rows:
        rows_by_scenario.setdefault(row.scenario_number, []).append(row)

    summaries: list[tuple[int, str, list[Selection]]] = []
    overview_items: dict[str, list[tuple[int, gcr.Scenario, Selection, list]]] = {
        "fewest_configs": [],
        "most_configs": [],
        "fastest": [],
    }
    missing_paths: list[Path] = []
    for number in sorted(rows_by_scenario):
        scenario = scenario_by_number.get(number)
        if scenario is None:
            continue
        selections = make_selections(rows_by_scenario[number])
        selected_configs: list[tuple[Selection, list]] = []
        for selection in selections:
            path = run_path(input_dir, selection.row)
            if not path.exists():
                missing_paths.append(path)
                continue
            configs = read_path_csv(path)
            selected_configs.append((selection, configs))
            overview_items[selection.stem].append((number, scenario, selection, configs))
        if len(selected_configs) != len(selections):
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)
        for ax, (selection, configs) in zip(axes, selected_configs):
            plot_one(ax, scenario, configs, selection)
        fig.suptitle(f"{number:02d}: {scenario.title}", fontsize=13.0, fontweight="bold")
        output_path = output_dir / f"{number:02d}_{scenario.name}_selected_paths.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        summaries.append((number, scenario.name, selections))

    if missing_paths:
        sample = "\n".join(str(path) for path in missing_paths[:5])
        raise SystemExit(
            "Some selected run path CSV files are missing. Re-run the C++ benchmark with --save-all-success-paths.\n"
            f"Examples:\n{sample}"
        )

    if not summaries:
        raise SystemExit("No selected paths were plotted.")

    plot_selection_overview(
        overview_items["fewest_configs"],
        output_dir / "overview_fewest_configs.png",
        "Fewest-Configuration Successful Solution Per Scenario",
    )
    plot_selection_overview(
        overview_items["most_configs"],
        output_dir / "overview_most_configs.png",
        "Most-Configuration Successful Solution Per Scenario",
    )
    plot_selection_overview(
        overview_items["fastest"],
        output_dir / "overview_fastest.png",
        "Fastest Successful Solution Per Scenario",
    )
    write_summary(output_dir / "selected_solution_summary.csv", summaries)
    write_markdown(output_dir / "selected_solution_report.md", summaries)
    print(f"Wrote selected solution plots to: {output_dir}")
    print(f"Report: {output_dir / 'selected_solution_report.md'}")
    print(f"Summary CSV: {output_dir / 'selected_solution_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
