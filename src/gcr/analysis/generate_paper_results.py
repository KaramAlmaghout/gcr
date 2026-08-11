"""Generate paper-ready plots and tables from a C++ GCR benchmark folder."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle

from gcr.planner import gcrbup as gcr
from gcr.planner.final_scenarios import build_final_scenarios
from gcr.analysis.plot_cpp_solution_paths import read_path_csv


PAPER_SCENARIOS = [2, 3, 5, 7, 8, 9, 10, 11, 12, 14]
START_COLOR = "#007EA7"
GOAL_COLOR = "#009E8F"
SWEPT_CORRIDOR_COLOR = "#facc15"
RAINBOW_ANCHORS = [
    START_COLOR,
    "#2962FF",
    "#9500FF",
    "#FF0059",
    "#B4E600",
    GOAL_COLOR,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready GCR benchmark report files.")
    parser.add_argument(
        "benchmark_dir",
        nargs="?",
        default="results/paper_benchmark_1000",
        help="Benchmark folder containing final_metrics_all.csv and selected_solution_paths/selected_solution_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper_report/generated",
        help="Output folder for regenerated paper files.",
    )
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--scenario", type=int, action="append", help="1-based scenario number. Default: paper subset.")
    return parser.parse_args(argv)


def csv_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def csv_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def csv_int(row: dict[str, str] | None, key: str) -> int | None:
    if row is None:
        return None
    value = csv_float(row, key)
    if not math.isfinite(value):
        return None
    return int(round(value))


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.pstdev(clean)


def scenario_dir(benchmark_dir: Path, scenario_number: int) -> Path:
    prefix = f"{scenario_number:02d}_"
    matches = sorted(path for path in benchmark_dir.iterdir() if path.is_dir() and path.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"No scenario folder found for scenario {scenario_number} in {benchmark_dir}")
    return matches[0]


def read_metrics(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_selected_summary(summary_path: Path) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("selection") != "fewest_configs":
                continue
            selected[int(row["scenario_number"])] = row
    return selected


def transition_polygon(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.vstack([a, b[::-1]])


def save_figure(fig: plt.Figure, path: Path, **kwargs) -> Path:
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError:
        for index in range(1, 100):
            fallback = path.with_name(f"{path.stem}_{index}{path.suffix}")
            try:
                fig.savefig(fallback, **kwargs)
                print(f"Warning: {path} is locked; wrote {fallback} instead.")
                return fallback
            except PermissionError:
                continue
        raise


def ordinal(index: int) -> str:
    value = index + 1
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def interpolate_color(color_a: str, color_b: str, t: float) -> tuple[float, float, float, float]:
    rgba_a = np.asarray(plt.matplotlib.colors.to_rgba(color_a), dtype=float)
    rgba_b = np.asarray(plt.matplotlib.colors.to_rgba(color_b), dtype=float)
    return tuple((1.0 - t) * rgba_a + t * rgba_b)


def config_gradient(index: int, count: int) -> tuple[float, float, float, float]:
    if count <= 0:
        return plt.matplotlib.colors.to_rgba(RAINBOW_ANCHORS[3])
    position = (index + 1) * (len(RAINBOW_ANCHORS) - 1) / (count + 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(RAINBOW_ANCHORS) - 1)
    t = position - lower
    return interpolate_color(RAINBOW_ANCHORS[lower], RAINBOW_ANCHORS[upper], t)


def intermediate_legend_handles(count: int, *, linewidth: float = 1.8, markersize: float = 5) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=config_gradient(index, count),
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            label=f"{ordinal(index)} intermediate",
        )
        for index in range(count)
    ]


def draw_scene(
    ax: plt.Axes,
    scenario: gcr.Scenario,
    configs: list[np.ndarray],
    selected_row: dict[str, str],
    *,
    title_fontsize: float = 8.5,
    axis_label: bool = False,
) -> None:
    xmin, xmax, ymin, ymax = scenario.workspace
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="#f8fafc",
            edgecolor="#ef4444",
            linewidth=1.2,
            zorder=0,
        )
    )
    for polygon in scenario.obstacles.polygons:
        ax.add_patch(Polygon(polygon, closed=True, facecolor="#3f3f46", edgecolor="#18181b", linewidth=0.8, zorder=5))
    for center, radius in scenario.obstacles.circles:
        ax.add_patch(Circle(center, radius, facecolor="#3f3f46", edgecolor="#18181b", linewidth=0.8, zorder=5))

    for a, b in zip(configs[:-1], configs[1:]):
        ax.add_patch(
            Polygon(
                transition_polygon(a, b),
                closed=True,
                facecolor=SWEPT_CORRIDOR_COLOR,
                edgecolor="none",
                alpha=0.16,
                zorder=2,
            )
        )

    intermediate_count = max(0, len(configs) - 2)
    for idx, config in enumerate(configs[1:-1]):
        ax.plot(
            config[:, 0],
            config[:, 1],
            "-o",
            color=config_gradient(idx, intermediate_count),
            linewidth=1.35,
            markersize=3.0,
            alpha=0.88,
            zorder=12 + idx,
        )

    start = configs[0]
    goal = configs[-1]
    ax.plot(start[:, 0], start[:, 1], "-o", color=START_COLOR, linewidth=1.9, markersize=3.2, zorder=20)
    ax.plot(goal[:, 0], goal[:, 1], "-o", color=GOAL_COLOR, linewidth=1.9, markersize=3.2, zorder=21)

    time_ms = 1000.0 * csv_float(selected_row, "time_s")
    configs_count = int(float(selected_row.get("refined_path_configs", len(configs))))
    title = f"configs={configs_count}, t={time_ms:.1f} ms"
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlim(xmin - 0.035, xmax + 0.035)
    ax.set_ylim(ymin - 0.035, ymax + 0.035)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.45, alpha=0.7)
    ax.tick_params(labelsize=7)
    if axis_label:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")


def make_overview_plot(
    output_dir: Path,
    benchmark_dir: Path,
    scenario_numbers: list[int],
    scenarios: list[gcr.Scenario],
    selected: dict[int, dict[str, str]],
) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(11.0, 15.2), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, scenario_number in zip(axes_flat, scenario_numbers):
        row = selected[scenario_number]
        run_index = int(row["run_index"])
        path_csv = scenario_dir(benchmark_dir, scenario_number) / "run_paths" / f"run_{run_index:04d}_success_path.csv"
        configs = read_path_csv(path_csv)
        draw_scene(ax, scenarios[scenario_number - 1], configs, row)

    for ax in axes_flat[len(scenario_numbers) :]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=START_COLOR, marker="o", linewidth=2.0, markersize=4, label="start"),
        Line2D([0], [0], color=GOAL_COLOR, marker="o", linewidth=2.0, markersize=4, label="goal"),
        Line2D([0], [0], color=config_gradient(0, 3), marker="o", linewidth=1.5, markersize=4, label="1st intermediate"),
        Line2D([0], [0], color=config_gradient(1, 3), marker="o", linewidth=1.5, markersize=4, label="2nd intermediate"),
        Line2D([0], [0], color=config_gradient(2, 3), marker="o", linewidth=1.5, markersize=4, label="later intermediates"),
        Patch(facecolor=SWEPT_CORRIDOR_COLOR, alpha=0.18, label="swept corridor"),
        Patch(facecolor="#3f3f46", label="obstacle"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=True)
    fig.suptitle("Fewest-Configuration Solutions", fontsize=15, y=1.003)
    save_figure(fig, output_dir / "paper_fewest_config_solutions.png", dpi=300, bbox_inches="tight")
    save_figure(fig, output_dir / "paper_fewest_config_solutions.pdf", bbox_inches="tight")
    plt.close(fig)


def make_per_scenario_plots(
    output_dir: Path,
    benchmark_dir: Path,
    scenario_numbers: list[int],
    scenarios: list[gcr.Scenario],
    selected: dict[int, dict[str, str]],
) -> None:
    per_dir = output_dir / "per_scenario"
    per_dir.mkdir(parents=True, exist_ok=True)
    for paper_scenario_number, scenario_number in enumerate(scenario_numbers, start=1):
        row = selected[scenario_number]
        run_index = int(row["run_index"])
        path_csv = scenario_dir(benchmark_dir, scenario_number) / "run_paths" / f"run_{run_index:04d}_success_path.csv"
        configs = read_path_csv(path_csv)
        scenario = scenarios[scenario_number - 1]
        fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
        draw_scene(ax, scenario, configs, row, title_fontsize=11, axis_label=True)
        intermediate_count = max(0, len(configs) - 2)
        handles = [
            Line2D([0], [0], color=START_COLOR, marker="o", linewidth=2.2, markersize=5, label="start"),
            Line2D([0], [0], color=GOAL_COLOR, marker="o", linewidth=2.2, markersize=5, label="goal"),
            *intermediate_legend_handles(intermediate_count),
            Patch(facecolor=SWEPT_CORRIDOR_COLOR, alpha=0.18, label="swept corridor"),
            Patch(facecolor="#3f3f46", label="obstacle"),
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
        stem = f"scenario_{paper_scenario_number:02d}_{scenario_display_slug(scenario.name)}_fewest_config"
        save_figure(fig, per_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        save_figure(fig, per_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def summarize(metrics_rows: list[dict[str, str]], scenario_numbers: list[int]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for paper_scenario_number, scenario_number in enumerate(scenario_numbers, start=1):
        rows = [row for row in metrics_rows if int(row["final_scenario_number"]) == scenario_number]
        success_rows = [row for row in rows if csv_bool(row.get("success"))]
        scenario_name = rows[0]["scenario"] if rows else f"scenario_{scenario_number:02d}"
        fastest_row = min(
            success_rows,
            key=lambda row: csv_float(row, "time_s") if math.isfinite(csv_float(row, "time_s")) else float("inf"),
            default=None,
        )
        fewest_configs_row = min(
            success_rows,
            key=lambda row: (
                csv_float(row, "refined_path_configs")
                if math.isfinite(csv_float(row, "refined_path_configs"))
                else float("inf"),
                csv_float(row, "time_s") if math.isfinite(csv_float(row, "time_s")) else float("inf"),
            ),
            default=None,
        )
        success_times_ms = [1000.0 * csv_float(row, "time_s") for row in success_rows]
        time_mean, time_std = mean_std(success_times_ms)
        fastest_time_ms = 1000.0 * csv_float(fastest_row, "time_s") if fastest_row else float("nan")
        fastest_run_configs = csv_int(fastest_row, "refined_path_configs")
        fewest_configs = csv_int(fewest_configs_row, "refined_path_configs")
        fewest_configs_time_ms = 1000.0 * csv_float(fewest_configs_row, "time_s") if fewest_configs_row else float("nan")
        iter_mean, iter_std = mean_std([csv_float(row, "iterations") for row in success_rows])
        raw_mean, raw_std = mean_std([csv_float(row, "raw_path_configs") for row in success_rows])
        refined_mean, refined_std = mean_std([csv_float(row, "refined_path_configs") for row in success_rows])
        out.append(
            {
                "scenario_number": paper_scenario_number,
                "source_scenario_number": scenario_number,
                "scenario": scenario_name,
                "scenario_display_name": scenario_display_name(scenario_name),
                "runs": len(rows),
                "successes": len(success_rows),
                "success_rate_percent": 100.0 * len(success_rows) / len(rows) if rows else float("nan"),
                "time_mean_ms": time_mean,
                "time_std_ms": time_std,
                "fastest_time_ms": fastest_time_ms,
                "fastest_run_configs": fastest_run_configs,
                "fewest_configs_time_ms": fewest_configs_time_ms,
                "fewest_configs": fewest_configs,
                "iterations_mean": iter_mean,
                "iterations_std": iter_std,
                "configs_before_rewiring_mean": raw_mean,
                "configs_before_rewiring_std": raw_std,
                "configs_after_rewiring_mean": refined_mean,
                "configs_after_rewiring_std": refined_std,
            }
        )
    return out


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "nan"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def fmt_count(value: object) -> str:
    if value is None:
        return "nan"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "nan"
        return str(int(round(float(value))))
    return str(value)


def scenario_display_name(source_name: str) -> str:
    parts = source_name.split("_")
    if len(parts) > 2 and parts[0] == "scenario" and parts[1].isdigit():
        parts = parts[2:]
    return " ".join(parts).replace("dlo", "DLO").title().replace("Dlo", "DLO")


def scenario_display_slug(source_name: str) -> str:
    return scenario_display_name(source_name).lower().replace(" ", "_")


def tex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_tables(output_dir: Path, rows: list[dict[str, object]]) -> None:
    csv_path = output_dir / "paper_benchmark_table.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Scenario | Success rate [%] | Time [ms] | Minimum time [ms] | Iterations | Configs before rewiring | Configs after rewiring |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            f"Scenario {row['scenario_number']}: {row['scenario_display_name']} | "
            f"{fmt(row['success_rate_percent'])} | "
            f"{fmt(row['time_mean_ms'])} +/- {fmt(row['time_std_ms'])} | "
            f"{fmt(row['fastest_time_ms'])} | "
            f"{fmt_count(row['iterations_mean'])} +/- {fmt_count(row['iterations_std'])} | "
            f"{fmt_count(row['configs_before_rewiring_mean'])} +/- {fmt_count(row['configs_before_rewiring_std'])} | "
            f"{fmt_count(row['configs_after_rewiring_mean'])} +/- {fmt_count(row['configs_after_rewiring_std'])} |"
        )
    (output_dir / "paper_benchmark_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"No. & Scenario & Success (\%) & Time (ms) & Min. time (ms) & Iter. & Configs pre & Configs post \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['scenario_number']} & {tex_escape(row['scenario_display_name'])} & "
            f"{fmt(row['success_rate_percent'])} & "
            f"{fmt(row['time_mean_ms'])} $\\pm$ {fmt(row['time_std_ms'])} & "
            f"{fmt(row['fastest_time_ms'])} & "
            f"{fmt_count(row['iterations_mean'])} $\\pm$ {fmt_count(row['iterations_std'])} & "
            f"{fmt_count(row['configs_before_rewiring_mean'])} $\\pm$ {fmt_count(row['configs_before_rewiring_std'])} & "
            f"{fmt_count(row['configs_after_rewiring_mean'])} $\\pm$ {fmt_count(row['configs_after_rewiring_std'])} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (output_dir / "paper_benchmark_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_readme(output_dir: Path, benchmark_dir: Path, scenario_numbers: list[int]) -> None:
    mapping = "\n".join(
        f"- Scenario {paper_number}: source scenario {source_number}"
        for paper_number, source_number in enumerate(scenario_numbers, start=1)
    )
    text = f"""# Paper Results

Source benchmark folder: `{benchmark_dir}`

Paper scenario mapping:
{mapping}

Files:
- `paper_fewest_config_solutions.png`
- `paper_fewest_config_solutions.pdf`
- `per_scenario/scenario_*_fewest_config.png`
- `per_scenario/scenario_*_fewest_config.pdf`
- `paper_benchmark_table.csv`
- `paper_benchmark_table.md`
- `paper_benchmark_table.tex`

The table averages are computed over successful runs only. The success rate is computed over all runs in `final_metrics_all.csv`.
Time is reported in milliseconds. The fastest-run time is the minimum successful `time_s` for each scenario. Fastest configs uses the `refined_path_configs` value from that same fastest run. Fewest configs is the minimum successful `refined_path_configs` value, with time from the fastest run among ties. Configs before rewiring use `raw_path_configs`; configs after rewiring use `refined_path_configs`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark_dir = Path(args.benchmark_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_numbers = args.scenario or PAPER_SCENARIOS

    metrics_path = benchmark_dir / "final_metrics_all.csv"
    summary_path = benchmark_dir / "selected_solution_paths" / "selected_solution_summary.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing selected summary: {summary_path}. Run plot_cpp_solution_extremes.py first.")

    metrics_rows = read_metrics(metrics_path)
    selected = read_selected_summary(summary_path)
    missing = [number for number in scenario_numbers if number not in selected]
    if missing:
        raise SystemExit(f"Missing fewest-config selected paths for scenarios: {missing}")

    scenarios = build_final_scenarios(args.nodes)
    make_overview_plot(output_dir, benchmark_dir, scenario_numbers, scenarios, selected)
    make_per_scenario_plots(output_dir, benchmark_dir, scenario_numbers, scenarios, selected)
    table_rows = summarize(metrics_rows, scenario_numbers)
    write_tables(output_dir, table_rows)
    write_readme(output_dir, benchmark_dir, scenario_numbers)

    print(f"Wrote paper report files to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
