# GCR Cable Planning and MuJoCo Replay

This repository is the isolated release of the Guided Cable Reconfiguration
(GCR) planner used for the paper benchmark, together with the matching MuJoCo
two-UR10e cable simulation, the final benchmark snapshot, and recorded replay
videos.

The core planner and simulation files were copied from the archived paper code
snapshot. They are kept separate from later experimental working folders so
the checked-in results can be traced to one implementation. Unused legacy RRT,
state-machine, and standalone-scene experiments are intentionally excluded.

## Repository Contents

```text
src/gcr/        the gcr package: planner, analysis, MuJoCo replay, scene build
cpp/            C++17 benchmark implementation used for the final runs
scripts/        PowerShell wrappers around the gcr commands
docs/           parameter reference, paper math comparison, layout map
results/        checked-in benchmark snapshot and derived analyses
videos/         six retained four-camera MuJoCo replay videos
third_party/    vendored MuJoCo Menagerie UR10e model and its license
```

- `src/gcr/planner/`: the Python reference planner (`gcrbup.py`) and the
  scenario catalogue (`final_scenarios.py`).
- `src/gcr/analysis/`: benchmark plots, the paper table, the runtime-outlier
  analysis, and waypoint export for MuJoCo.
- `src/gcr/mujoco_exec/` and `src/gcr/sim/`: MuJoCo replay, UR10e control, and
  cable scene construction.
- `third_party/mujoco_menagerie/universal_robots_ur10e/`: required MuJoCo
  Menagerie UR10e XML, referenced meshes, metadata, and original license. The
  optional upstream preview and demonstration scene are omitted.
- `results/paper_benchmark_1000/`: 10 scenarios, 1000 runs per scenario,
  complete run metrics, selected paths, plots, and MuJoCo waypoint exports.
- `results/paper_report/outlier_analysis/`: Tukey runtime-outlier sensitivity
  analysis and distribution plots.
- `results/mujoco_final_errors/`: retained settled cable-error JSON files.

The compact benchmark snapshot includes all 10,000 metric rows and every path
selected as fastest, fewest-configuration, or most-configuration. The 10,000
redundant per-run path CSV files are intentionally omitted; they can be
regenerated with `--save-all-success-paths` by rerunning the benchmark.

[docs/repository_layout.md](docs/repository_layout.md) has the full file map,
where generated outputs go, and the pre-release path mapping.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

That installs the project in editable mode with its dependencies and provides
the `gcr` command. `python -m gcr` is equivalent if the console script is not
on `PATH`.

MuJoCo automatically finds the vendored UR10e assets. To use an external
MuJoCo Menagerie checkout instead, set `MUJOCO_MENAGERIE` to its root path.
Commands resolve repository assets from their own location or the working
directory; `GCR_REPO_ROOT` overrides that.

The release was verified with Python 3.11 and MuJoCo 3.8.1 on Windows, and the
reorganized package with Python 3.10 and MuJoCo 3.1.6 on Linux.

## Commands

```bash
gcr --help              # all commands
gcr <command> --help    # the options of one command
```

| Command | Purpose |
|---|---|
| `gcr plan` | Run the Python reference planner over the scenario catalogue |
| `gcr build-cpp` | Build the C++17 benchmark binary |
| `gcr benchmark` | Run the C++ benchmark with the paper planner configuration |
| `gcr paper-report` | Select solutions, build paper plots/tables, export waypoints |
| `gcr report` | Paper plots and benchmark table from a benchmark folder |
| `gcr plot-benchmark` | Success-rate, runtime, and configuration-count plots |
| `gcr plot-paths` / `gcr plot-extremes` | Solution-path plots |
| `gcr plot-outliers` | Runtime distributions with and without Tukey outliers |
| `gcr workspace-figure` | DLO workspace problem-formulation figures |
| `gcr export-waypoints` | Export selected benchmark solutions for MuJoCo |
| `gcr paper-replay` | Replay every fewest-configuration solution in MuJoCo |
| `gcr replay` | Replay one waypoint CSV on the two-UR10e cable scene |
| `gcr view-scenario` | Open an exported scenario in the MuJoCo viewer |
| `gcr verify` | Verify the checksums in `SHA256SUMS` |

The scripts in [scripts/](scripts/) are PowerShell wrappers around these same
commands, kept for the original Windows workflow.

## Build and Run the Benchmark

Build the C++ runner (Visual Studio 2022 C++ Build Tools on Windows, CMake or
any C++17 compiler elsewhere):

```bash
gcr build-cpp
```

Run a short smoke benchmark:

```bash
gcr benchmark results/benchmark_rerun/smoke --runs 1 --max-iterations 5000 --scenario-time-limit 15
```

Run the full paper benchmark with the original planner flags:

```bash
gcr benchmark
```

The full command uses 1000 runs per scenario, endpoint reordering, terminal
reversal, travel and tangent continuity tests, robot execution clearance
checks, and saved success paths. The exact values are listed in
[docs/planning_and_mujoco_parameters.md](docs/planning_and_mujoco_parameters.md)
and defined in `PAPER_PLANNER_FLAGS` in
[cpp_benchmark.py](src/gcr/cpp_benchmark.py).

## Recreate Reports

Generate the paper plots and table from the checked-in benchmark snapshot:

```bash
gcr report results/paper_benchmark_1000 --output-dir results/paper_report/generated
```

Run the whole analysis chain over a rerun (solution selection, paper report,
MuJoCo waypoint export):

```bash
gcr paper-report results/benchmark_rerun
```

Regenerate the runtime distributions with and without Tukey outliers:

```bash
gcr plot-outliers
```

## MuJoCo Replay

Replay all ten fewest-configuration solutions interactively:

```bash
gcr paper-replay --interactive
```

Record all ten solutions and recompute settled final-shape errors:

```bash
gcr paper-replay
```

Replay a single waypoint CSV:

```bash
gcr replay --csv examples/example_waypoints.csv
```

Generated videos and error files are written to `results/replay_outputs/`,
which is ignored by Git. Existing retained recordings are in `videos/`.

## Paper Scenario Mapping

| Paper scenario | Source scenario | Name |
|---:|---:|---|
| 1 | 2 | `scenario_02_block_reorient` |
| 2 | 3 | `scenario_03_L_narrow_corridor` |
| 3 | 5 | `scenario_05_arc_to_straight` |
| 4 | 7 | `scenario_07_L_corridor_hard` |
| 5 | 8 | `scenario_08_dlo_workspace` |
| 6 | 9 | `scenario_09_S_corridor` |
| 7 | 10 | `scenario_10_cluttered_bent` |
| 8 | 11 | `scenario_2_narrow_gate` |
| 9 | 12 | `scenario_6_u_shaped_trap` |
| 10 | 14 | `scenario_4_staggered_blocks` |

## Benchmark Notes

- Runtime statistics are computed over successful runs; all selected scenarios
  achieved 100% success in the archived 1000-run benchmark.
- `results/paper_report/paper_benchmark_table_original.tex` contains the table
  used before runtime-outlier filtering.
- `results/paper_report/outlier_analysis/` contains the separate sensitivity
  analysis using the upper Tukey fence `Q3 + 1.5 IQR` per scenario.

## Integrity

`SHA256SUMS` records the core source, aggregate benchmark CSVs, and retained
videos. Verify it with:

```bash
gcr verify
```

