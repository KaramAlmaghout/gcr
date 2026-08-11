from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
METRICS_CSV = ROOT / "results" / "paper_benchmark_1000" / "final_metrics_all.csv"
OUT_DIR = Path(__file__).resolve().parent / "runtime_distribution_plots"

# This is the paper order used by GCR_2026_ghadeer_cpp/paper_benchmark_table_for_paper.tex.
PAPER_SCENARIOS = [
    (1, 2, "Block Reorient"),
    (2, 3, "L Narrow Corridor"),
    (3, 5, "Arc To Straight"),
    (4, 7, "L Corridor Hard"),
    (5, 8, "DLO Workspace"),
    (6, 9, "S Corridor"),
    (7, 10, "Cluttered Bent"),
    (8, 11, "Narrow Gate"),
    (9, 12, "U Shaped Trap"),
    (10, 14, "Staggered Blocks"),
]


def tukey_upper_fence(values: pd.Series) -> float:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return float(q3 + 1.5 * (q3 - q1))


def add_stats_box(ax, values: pd.Series, label: str) -> None:
    text = (
        f"{label}\n"
        f"n = {len(values)}\n"
        f"mean = {values.mean():.2f} ms\n"
        f"std = {values.std(ddof=0):.2f} ms\n"
        f"median = {values.median():.2f} ms"
    )
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#b8b8b8", "alpha": 0.9},
    )


def plot_hist(ax, values: pd.Series, title: str, color: str, bins: int = 42) -> None:
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.55, alpha=0.88)
    ax.axvline(values.mean(), color="#c1121f", linestyle="--", linewidth=1.6, label="mean")
    ax.axvline(values.median(), color="#111111", linestyle="-", linewidth=1.4, label="median")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("solve time [ms]")
    ax.set_ylabel("runs")
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_CSV)
    pdf_path = OUT_DIR / "paper_runtime_distributions_with_without_outliers.pdf"

    with PdfPages(pdf_path) as pdf:
        for paper_no, source_no, display_name in PAPER_SCENARIOS:
            rows = df[
                df["final_scenario_number"].eq(source_no)
                & df["success"].astype(str).str.lower().eq("true")
            ].copy()
            if rows.empty:
                raise RuntimeError(f"No successful rows found for source scenario {source_no}")

            rows["time_ms"] = rows["time_s"] * 1000.0
            all_times = rows["time_ms"]
            fence = tukey_upper_fence(all_times)
            clean_times = all_times[all_times <= fence]
            outlier_times = all_times[all_times > fence]

            fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
            fig.suptitle(f"Scenario {paper_no}", fontsize=15, fontweight="bold")

            plot_hist(
                axes[0],
                all_times,
                f"All successful runs ({len(all_times)} runs)",
                color="#2f6bff",
            )
            axes[0].axvline(fence, color="#ff8c00", linestyle=":", linewidth=1.8, label="Tukey fence")
            axes[0].legend(loc="upper left", fontsize=8, frameon=True)
            add_stats_box(axes[0], all_times, "All runs")

            plot_hist(
                axes[1],
                clean_times,
                f"Without outliers ({len(outlier_times)} removed)",
                color="#0fc5d7",
            )
            add_stats_box(axes[1], clean_times, "Without outliers")
            axes[1].text(
                0.02,
                0.96,
                f"outlier rule:\ntime > {fence:.2f} ms",
                transform=axes[1].transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#b8b8b8", "alpha": 0.9},
            )

            stem = f"scenario_{paper_no:02d}_{display_name.lower().replace(' ', '_')}_runtime_distribution"
            fig.savefig(OUT_DIR / f"{stem}.png", dpi=220)
            fig.savefig(OUT_DIR / f"{stem}.pdf")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote per-scenario plots to: {OUT_DIR}")
    print(f"Wrote combined PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
