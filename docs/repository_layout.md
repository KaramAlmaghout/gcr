# Repository Layout

The release has four layers: the planner (Python and C++), the analysis tools,
the MuJoCo replay stack, and the immutable paper artifacts.

```text
gcr/
├── pyproject.toml                  packaging; declares the `gcr` command
├── requirements.txt                installs the project and its dependencies
├── SHA256SUMS                      checksums of the released artifacts
├── src/gcr/                        the Python package
│   ├── cli.py                      `gcr` command dispatcher
│   ├── paths.py                    repository/asset path resolution
│   ├── cpp_benchmark.py            build and run the C++ benchmark
│   ├── pipelines.py                paper-report and paper-replay pipelines
│   ├── verify.py                   SHA256SUMS verification
│   ├── planner/                    Python reference planner
│   │   ├── gcrbup.py               GCR planner (paper Sections 4-5)
│   │   ├── final_scenarios.py      benchmark scenario catalogue
│   │   └── dlo_workspace_figure.py workspace geometry and paper figures
│   ├── analysis/                   benchmark plots, tables, waypoint export
│   ├── mujoco_exec/                MuJoCo replay of planned paths
│   └── sim/                        MuJoCo scene construction and control
├── cpp/                            C++17 benchmark used for the paper runs
├── scripts/                        PowerShell wrappers around `gcr` commands
├── examples/                       sample replay input
├── docs/                           parameter reference and math comparison
├── results/                        checked-in benchmark snapshot and analyses
├── videos/                         retained four-camera MuJoCo replays
└── third_party/                    vendored MuJoCo Menagerie UR10e model
```

## Where Generated Files Go

| Output | Location | Tracked |
|---|---|---|
| Paper benchmark snapshot | `results/paper_benchmark_1000/` | yes |
| Outlier sensitivity analysis | `results/paper_report/outlier_analysis/` | yes |
| Settled MuJoCo cable errors | `results/mujoco_final_errors/` | yes |
| Retained replay videos | `videos/` | yes |
| New benchmark runs | `results/benchmark_rerun/` | no |
| Regenerated paper report | `results/paper_report/generated/` | no |
| Replay videos and errors | `results/replay_outputs/` | no |
| Python-planned waypoint exports | `results/gcr_waypoints/` | no |
| C++ build products | `cpp/build/` | no |

## Pre-Release Path Mapping

The code moved into an installable package in the reorganization; the module
contents are unchanged apart from imports, entry points, and path resolution.

| Before | Now | Command |
|---|---|---|
| `gcrbup.py` | `src/gcr/planner/gcrbup.py` | `gcr plan` |
| `final_scenarios.py` | `src/gcr/planner/final_scenarios.py` | — |
| `generate_dlo_workspace_figure.py` | `src/gcr/planner/dlo_workspace_figure.py` | `gcr workspace-figure` |
| `generate_paper_results.py` | `src/gcr/analysis/generate_paper_results.py` | `gcr report` |
| `plot_cpp_benchmark_report.py` | `src/gcr/analysis/plot_cpp_benchmark_report.py` | `gcr plot-benchmark` |
| `plot_cpp_solution_paths.py` | `src/gcr/analysis/plot_cpp_solution_paths.py` | `gcr plot-paths` |
| `plot_cpp_solution_extremes.py` | `src/gcr/analysis/plot_cpp_solution_extremes.py` | `gcr plot-extremes` |
| `export_cpp_selected_waypoints.py` | `src/gcr/analysis/export_cpp_selected_waypoints.py` | `gcr export-waypoints` |
| `results/paper_report/outlier_analysis/plot_runtime_distributions.py` | `src/gcr/analysis/plot_runtime_distributions.py` | `gcr plot-outliers` |
| `mujoco_exec/*.py` | `src/gcr/mujoco_exec/*.py` | `gcr replay`, `gcr execute-plan`, … |
| `mujoco_exec/example_waypoints.csv` | `examples/example_waypoints.csv` | — |
| `dlo2026/mjlab/scene/build_scene.py` | `src/gcr/sim/build_scene.py` | — |
| `dlo2026/mjlab/controller.py` | `src/gcr/sim/controller.py` | — |
| `run_benchmark_paper.ps1` | `scripts/run_benchmark_paper.ps1` | `gcr benchmark` |
| `plot_export_from_benchmark.ps1` | `scripts/plot_export_from_benchmark.ps1` | `gcr paper-report` |
| `run_mujoco_paper_10_fewest.ps1` | `scripts/run_mujoco_paper_10_fewest.ps1` | `gcr paper-replay` |
| `README_PLANNING_AND_MUJOCO_PARAMETERS.md` | `docs/planning_and_mujoco_parameters.md` | — |
| `README_MATH_COMPARISON.md` | `docs/math_comparison.md` | — |
| `cpp/README_CPP.md` | `cpp/README.md` | — |

Import paths changed accordingly: `import gcrbup` is now
`from gcr.planner import gcrbup`, `mujoco_exec.*` is `gcr.mujoco_exec.*`, and
`dlo2026.mjlab.scene.build_scene` is `gcr.sim.build_scene`.

## Conventions

* Modules never compute the repository root by counting `..` levels; they call
  `gcr.paths` helpers, so assets resolve the same way from any working
  directory. `GCR_REPO_ROOT` overrides the search.
* Every command-line module exposes `main(argv)` and is registered in
  `gcr/cli.py`; the CLI forwards arguments unchanged, so `gcr <command> --help`
  is the tool's own help.
* Flag sets that define published results live in one constant each:
  `PAPER_PLANNER_FLAGS` (`cpp_benchmark.py`), `EXPORT_REORDER_FLAGS` and
  `REPLAY_FLAGS` (`pipelines.py`).
