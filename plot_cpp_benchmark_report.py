"""Create easy-to-read plots for the C++ GCR benchmark CSV output.

Usage:
    python plot_cpp_benchmark_report.py results/paper_benchmark_1000

The input folder must contain:
    final_metrics_all.csv
    final_benchmark_summary.csv

Outputs are written to:
    <input>/visual_report/
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_FILE = "final_metrics_all.csv"
SUMMARY_FILE = "final_benchmark_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot C++ GCR benchmark CSV results.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="results/paper_benchmark_1000",
        help="Folder containing final_metrics_all.csv and final_benchmark_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for plots. Default: <input_dir>/visual_report.",
    )
    return parser.parse_args()


def as_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except ValueError:
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value)))
    except ValueError:
        return default


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def pct(values: Iterable[float], q: float, default: float = float("nan")) -> float:
    arr = finite(values)
    if arr.size == 0:
        return default
    return float(np.percentile(arr, q))


def mean(values: Iterable[float], default: float = float("nan")) -> float:
    arr = finite(values)
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def scenario_number(row: dict[str, str]) -> int:
    return as_int(row.get("final_scenario_number"), 999)


def short_label(scenario: str, number: int | None = None) -> str:
    label = scenario
    if label.startswith("scenario_"):
        label = label[len("scenario_") :]
    if number is not None and number < 999:
        numbered_prefix = f"{number:02d}_"
        if label.startswith(numbered_prefix):
            label = label[len(numbered_prefix) :]
    label = label.replace("_", " ")
    if number is not None and number < 999:
        return f"{number:02d} {label}"
    return label


def group_rows(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)
    order = sorted(grouped, key=lambda name: scenario_number(grouped[name][0]))
    return order, grouped


def metric_values(rows: list[dict[str, str]], key: str, success_only: bool = False) -> list[float]:
    out: list[float] = []
    for row in rows:
        if success_only and not as_bool(row.get("success")):
            continue
        out.append(as_float(row.get(key)))
    return out


def scenario_stats(order: list[str], grouped: dict[str, list[dict[str, str]]], summary_by_name: dict[str, dict[str, str]]) -> list[dict]:
    stats: list[dict] = []
    for scenario in order:
        rows = grouped[scenario]
        number = scenario_number(rows[0])
        successes = [row for row in rows if as_bool(row.get("success"))]
        failures = [row for row in rows if not as_bool(row.get("success"))]
        time_all = metric_values(rows, "time_s")
        time_success = metric_values(rows, "time_s", success_only=True)
        iterations_all = metric_values(rows, "iterations")
        path_cost = metric_values(rows, "path_cost_J", success_only=True)
        clearance = metric_values(rows, "min_clearance_m", success_only=True)
        shortcut = metric_values(rows, "shortcut_reduction_pct", success_only=True)
        reject_static = metric_values(rows, "reject_static")
        reject_no_parent = metric_values(rows, "reject_no_parent")
        reject_no_connect = metric_values(rows, "reject_no_connect")

        stats.append(
            {
                "scenario": scenario,
                "label": short_label(scenario, number),
                "number": number,
                "title": summary_by_name.get(scenario, {}).get("title", scenario),
                "source": rows[0].get("source", ""),
                "n_runs": len(rows),
                "n_success": len(successes),
                "n_failure": len(failures),
                "n_timeout": sum(1 for row in rows if as_bool(row.get("timed_out"))),
                "success_rate": len(successes) / len(rows) if rows else 0.0,
                "time_mean_s": mean(time_all),
                "time_median_s": pct(time_all, 50),
                "time_p90_s": pct(time_all, 90),
                "time_p95_s": pct(time_all, 95),
                "success_time_median_s": pct(time_success, 50),
                "success_time_p90_s": pct(time_success, 90),
                "iterations_mean": mean(iterations_all),
                "iterations_median": pct(iterations_all, 50),
                "iterations_p90": pct(iterations_all, 90),
                "iterations_p95": pct(iterations_all, 95),
                "path_cost_median": pct(path_cost, 50),
                "path_cost_p90": pct(path_cost, 90),
                "min_clearance_median_m": pct(clearance, 50),
                "min_clearance_min_m": pct(clearance, 0),
                "shortcut_reduction_median_pct": pct(shortcut, 50),
                "reject_static_mean": mean(reject_static),
                "reject_no_parent_mean": mean(reject_no_parent),
                "reject_no_connect_mean": mean(reject_no_connect),
            }
        )
    return stats


def save_enriched_csv(path: Path, stats: list[dict]) -> None:
    fields = [
        "number",
        "scenario",
        "title",
        "source",
        "n_runs",
        "n_success",
        "n_failure",
        "n_timeout",
        "success_rate",
        "time_mean_s",
        "time_median_s",
        "time_p90_s",
        "time_p95_s",
        "iterations_mean",
        "iterations_median",
        "iterations_p90",
        "iterations_p95",
        "path_cost_median",
        "path_cost_p90",
        "min_clearance_median_m",
        "min_clearance_min_m",
        "shortcut_reduction_median_pct",
        "reject_static_mean",
        "reject_no_parent_mean",
        "reject_no_connect_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in stats:
            writer.writerow({field: row.get(field, "") for field in fields})


def colors_for_success(stats: list[dict]) -> list[str]:
    colors = []
    for row in stats:
        rate = row["success_rate"]
        if rate >= 0.95:
            colors.append("#16a34a")
        elif rate >= 0.80:
            colors.append("#f59e0b")
        else:
            colors.append("#dc2626")
    return colors


def apply_common(ax, title: str, xlabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def plot_dashboard(out_dir: Path, stats: list[dict]) -> Path:
    labels = [row["label"] for row in stats]
    y = np.arange(len(stats))
    fig, axes = plt.subplots(3, 2, figsize=(15, 16), constrained_layout=True)
    axes = axes.ravel()

    success = np.array([row["success_rate"] for row in stats])
    axes[0].barh(y, success * 100.0, color=colors_for_success(stats), edgecolor="#111827", linewidth=0.4)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 105)
    apply_common(axes[0], "Success Rate", "success [%]")
    for yi, value in zip(y, success):
        axes[0].text(value * 100.0 + 1.0, yi, f"{value*100:.0f}%", va="center", fontsize=8)

    med_ms = np.array([row["time_median_s"] * 1000.0 for row in stats])
    p90_ms = np.array([row["time_p90_s"] * 1000.0 for row in stats])
    axes[1].barh(y, med_ms, color="#2563eb", alpha=0.82)
    axes[1].scatter(p90_ms, y, marker="|", s=120, color="#111827", label="p90")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    apply_common(axes[1], "Solve Time", "median ms, p90 tick")

    iter_med = np.array([row["iterations_median"] for row in stats])
    iter_p90 = np.array([row["iterations_p90"] for row in stats])
    axes[2].barh(y, iter_med, color="#7c3aed", alpha=0.82)
    axes[2].scatter(iter_p90, y, marker="|", s=120, color="#111827", label="p90")
    axes[2].set_yticks(y, labels)
    axes[2].invert_yaxis()
    axes[2].legend(loc="lower right", fontsize=8)
    apply_common(axes[2], "Iterations", "median iterations, p90 tick")

    cost_med = np.array([row["path_cost_median"] for row in stats])
    cost_p90 = np.array([row["path_cost_p90"] for row in stats])
    axes[3].barh(y, cost_med, color="#0891b2", alpha=0.82)
    axes[3].scatter(cost_p90, y, marker="|", s=120, color="#111827", label="p90")
    axes[3].set_yticks(y, labels)
    axes[3].invert_yaxis()
    axes[3].legend(loc="lower right", fontsize=8)
    apply_common(axes[3], "Path Cost J", "median J, p90 tick")

    clear_med_mm = np.array([row["min_clearance_median_m"] * 1000.0 for row in stats])
    clear_min_mm = np.array([row["min_clearance_min_m"] * 1000.0 for row in stats])
    axes[4].barh(y, clear_med_mm, color="#10b981", alpha=0.82)
    axes[4].scatter(clear_min_mm, y, marker="x", s=35, color="#dc2626", label="minimum")
    axes[4].axvline(0.0, color="#991b1b", linewidth=1.0)
    axes[4].set_yticks(y, labels)
    axes[4].invert_yaxis()
    axes[4].legend(loc="lower right", fontsize=8)
    apply_common(axes[4], "Clearance", "median mm, red x = minimum")

    static = np.array([row["reject_static_mean"] for row in stats])
    no_parent = np.array([row["reject_no_parent_mean"] for row in stats])
    no_connect = np.array([row["reject_no_connect_mean"] for row in stats])
    axes[5].barh(y, static, color="#ef4444", label="static")
    axes[5].barh(y, no_parent, left=static, color="#f59e0b", label="no parent")
    axes[5].barh(y, no_connect, left=static + no_parent, color="#64748b", label="no connect")
    axes[5].set_yticks(y, labels)
    axes[5].invert_yaxis()
    axes[5].legend(loc="lower right", fontsize=8)
    apply_common(axes[5], "Mean Rejections Per Run", "count")

    fig.suptitle("C++ GCR Benchmark Report", fontsize=16, fontweight="bold")
    path = out_dir / "benchmark_dashboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_boxplot(
    out_dir: Path,
    filename: str,
    title: str,
    xlabel: str,
    order: list[str],
    grouped: dict[str, list[dict[str, str]]],
    key: str,
    scale: float = 1.0,
    success_only: bool = False,
) -> Path:
    data = []
    labels = []
    for scenario in order:
        rows = grouped[scenario]
        vals = finite(scale * np.array(metric_values(rows, key, success_only=success_only)))
        if vals.size == 0:
            vals = np.array([np.nan])
        data.append(vals)
        labels.append(short_label(scenario, scenario_number(rows[0])))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    try:
        ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False, patch_artist=True)
    except TypeError:
        ax.boxplot(data, vert=False, labels=labels, showfliers=False, patch_artist=True)
    ax.invert_yaxis()
    apply_common(ax, title, xlabel)
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_success_counts(out_dir: Path, stats: list[dict]) -> Path:
    labels = [row["label"] for row in stats]
    y = np.arange(len(stats))
    success = np.array([row["n_success"] for row in stats])
    timeout = np.array([row["n_timeout"] for row in stats])
    fail = np.array([row["n_failure"] - row["n_timeout"] for row in stats])

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.barh(y, success, color="#16a34a", label="success")
    ax.barh(y, timeout, left=success, color="#f59e0b", label="timeout")
    ax.barh(y, fail, left=success + timeout, color="#dc2626", label="other failure")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    apply_common(ax, "Run Outcomes", "count")
    path = out_dir / "run_outcomes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_quality_scatter(out_dir: Path, order: list[str], grouped: dict[str, list[dict[str, str]]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    for idx, scenario in enumerate(order):
        xs = []
        ys = []
        for row in grouped[scenario]:
            if not as_bool(row.get("success")):
                continue
            cost = as_float(row.get("path_cost_J"))
            clearance = as_float(row.get("min_clearance_m")) * 1000.0
            if math.isfinite(cost) and math.isfinite(clearance):
                xs.append(cost)
                ys.append(clearance)
        if xs:
            ax.scatter(xs, ys, s=18, alpha=0.55, color=cmap(idx % 20), label=short_label(scenario, scenario_number(grouped[scenario][0])))
    ax.axhline(0.0, color="#991b1b", linewidth=1.0)
    ax.set_xlabel("path cost J")
    ax.set_ylabel("minimum clearance [mm]")
    ax.set_title("Path Quality: Cost vs Clearance", fontsize=11, fontweight="bold")
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True)
    path = out_dir / "path_cost_vs_clearance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_markdown_report(out_dir: Path, input_dir: Path, stats: list[dict], plot_paths: list[Path]) -> Path:
    total_runs = sum(row["n_runs"] for row in stats)
    total_success = sum(row["n_success"] for row in stats)
    overall_rate = total_success / total_runs if total_runs else 0.0

    by_success = sorted(stats, key=lambda row: (row["success_rate"], -row["iterations_median"]))
    by_time = sorted(stats, key=lambda row: row["time_p90_s"], reverse=True)
    by_iters = sorted(stats, key=lambda row: row["iterations_p90"], reverse=True)
    by_clearance = sorted(stats, key=lambda row: row["min_clearance_min_m"])

    def table(rows: list[dict], columns: list[tuple[str, str]], limit: int = 5) -> str:
        lines = ["| " + " | ".join(label for label, _ in columns) + " |"]
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for row in rows[:limit]:
            values = []
            for _, key in columns:
                value = row[key]
                if isinstance(value, float):
                    if "rate" in key:
                        values.append(f"{100.0 * value:.1f}%")
                    elif key.endswith("_s"):
                        values.append(f"{1000.0 * value:.2f} ms")
                    elif "clearance" in key:
                        values.append(f"{1000.0 * value:.2f} mm")
                    elif "iter" in key:
                        values.append(f"{value:.0f}")
                    else:
                        values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    rel_plots = [path.name for path in plot_paths]
    md = [
        "# C++ GCR Benchmark Visual Report",
        "",
        f"Input folder: `{input_dir}`",
        "",
        f"- Total runs: `{total_runs}`",
        f"- Successful runs: `{total_success}`",
        f"- Overall success rate: `{100.0 * overall_rate:.1f}%`",
        f"- Scenarios: `{len(stats)}`",
        "",
        "## Main Dashboard",
        "",
        f"![Benchmark dashboard]({rel_plots[0]})",
        "",
        "## Scenarios To Watch",
        "",
        "Lowest success rate:",
        "",
        table(by_success, [("Scenario", "label"), ("Success", "success_rate"), ("p90 iter", "iterations_p90")]),
        "",
        "Highest p90 solve time:",
        "",
        table(by_time, [("Scenario", "label"), ("p90 time", "time_p90_s"), ("Success", "success_rate")]),
        "",
        "Highest p90 iterations:",
        "",
        table(by_iters, [("Scenario", "label"), ("p90 iter", "iterations_p90"), ("Success", "success_rate")]),
        "",
        "Lowest observed clearance:",
        "",
        table(by_clearance, [("Scenario", "label"), ("min clearance", "min_clearance_min_m"), ("Success", "success_rate")]),
        "",
        "## Additional Plots",
        "",
    ]
    for plot in rel_plots[1:]:
        title = plot.replace("_", " ").replace(".png", "").title()
        md.extend([f"### {title}", "", f"![{title}]({plot})", ""])

    selected_dir = input_dir / "selected_solution_paths"
    selected_overviews = [
        ("Fewest-Configuration Paths", selected_dir / "overview_fewest_configs.png"),
        ("Most-Configuration Paths", selected_dir / "overview_most_configs.png"),
        ("Fastest Successful Paths", selected_dir / "overview_fastest.png"),
    ]
    existing_overviews = [(title, path) for title, path in selected_overviews if path.exists()]
    if existing_overviews:
        md.extend(
            [
                "## Selected Solution Path Overviews",
                "",
                "These images are generated by `plot_cpp_solution_extremes.py` from runs saved with `--save-all-success-paths`.",
                "",
            ]
        )
        for title, path in existing_overviews:
            rel_path = path.relative_to(out_dir).as_posix() if path.is_relative_to(out_dir) else f"../selected_solution_paths/{path.name}"
            md.extend([f"### {title}", "", f"![{title}]({rel_path})", ""])

    md.extend(
        [
            "## How To Read This",
            "",
            "- Use `success_rate` first. A fast scenario that fails often is not robust.",
            "- Use p90 time and p90 iterations to understand worst-case behavior, not just averages.",
            "- Low or negative clearance means the path is close to obstacle contact and should be inspected.",
            "- Large `reject_static` means the sampler/relaxation often creates invalid cable shapes.",
            "- Large `reject_no_parent` means candidates are valid but hard to connect to the active tree.",
            "- Large `reject_no_connect` means the trees grow but struggle to meet.",
        ]
    )

    path = out_dir / "benchmark_visual_report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "visual_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = input_dir / METRIC_FILE
    summary_path = input_dir / SUMMARY_FILE
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")

    rows = load_csv(metrics_path)
    summary_rows = load_csv(summary_path)
    summary_by_name = {row["scenario"]: row for row in summary_rows}
    order, grouped = group_rows(rows)
    stats = scenario_stats(order, grouped, summary_by_name)

    save_enriched_csv(output_dir / "benchmark_summary_enriched.csv", stats)
    plot_paths = [
        plot_dashboard(output_dir, stats),
        plot_success_counts(output_dir, stats),
        plot_boxplot(output_dir, "solve_time_boxplot.png", "Solve Time Distribution", "time [ms]", order, grouped, "time_s", scale=1000.0),
        plot_boxplot(output_dir, "iterations_boxplot.png", "Iteration Distribution", "iterations", order, grouped, "iterations"),
        plot_boxplot(
            output_dir,
            "path_cost_boxplot.png",
            "Successful Path Cost Distribution",
            "path cost J",
            order,
            grouped,
            "path_cost_J",
            success_only=True,
        ),
        plot_boxplot(
            output_dir,
            "clearance_boxplot.png",
            "Successful Minimum Clearance Distribution",
            "minimum clearance [mm]",
            order,
            grouped,
            "min_clearance_m",
            scale=1000.0,
            success_only=True,
        ),
        plot_boxplot(
            output_dir,
            "shortcut_reduction_boxplot.png",
            "Shortcut Reduction Distribution",
            "shortcut reduction [%]",
            order,
            grouped,
            "shortcut_reduction_pct",
            success_only=True,
        ),
        plot_quality_scatter(output_dir, order, grouped),
    ]
    report_path = write_markdown_report(output_dir, input_dir, stats, plot_paths)

    print(f"Read {len(rows)} runs across {len(order)} scenarios.")
    print(f"Wrote plots and report to: {output_dir}")
    print(f"Open: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
