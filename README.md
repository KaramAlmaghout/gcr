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

- `cpp/`: C++17 benchmark implementation used for the final runs.
- `gcrbup.py` and `final_scenarios.py`: Python planner and scenario catalogue.
- `mujoco_exec/` and `dlo2026/`: MuJoCo replay, UR10e control, and cable scene.
- `third_party/mujoco_menagerie/universal_robots_ur10e/`: required MuJoCo
  Menagerie UR10e XML, referenced meshes, metadata, and original license. The
  optional upstream preview and demonstration scene are omitted.
- `results/paper_benchmark_1000/`: 10 scenarios, 1000 runs per scenario,
  complete run metrics, selected paths, plots, and MuJoCo waypoint exports.
- `results/paper_report/outlier_analysis/`: Tukey runtime-outlier sensitivity
  analysis and distribution plots.
- `results/mujoco_final_errors/`: retained settled cable-error JSON files.
- `videos/`: six retained four-camera MuJoCo replay videos.

The compact benchmark snapshot includes all 10,000 metric rows and every path
selected as fastest, fewest-configuration, or most-configuration. The 10,000
redundant per-run path CSV files are intentionally omitted; they can be
regenerated with `--save-all-success-paths` by rerunning the benchmark.

The release has four clear layers:

```text
cpp/                           C++17 planner and build files
gcrbup.py, final_scenarios.py  Python reference planner and scenarios
mujoco_exec/, dlo2026/         replay CLI, controller, and scene construction
results/, videos/              immutable paper artifacts and retained replays
```

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

## Setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MuJoCo automatically finds the vendored UR10e assets. To use an external
MuJoCo Menagerie checkout instead, set `MUJOCO_MENAGERIE` to its root path.

The release was verified with Python 3.11 and MuJoCo 3.8.1 on Windows.

## Build and Run the Planner

With Visual Studio 2022 C++ Build Tools:

```powershell
cd .\cpp
.\build_msvc.bat
cd ..
```

Run a short smoke benchmark:

```powershell
.\run_benchmark_paper.ps1 `
  -OutputDir ".\results\benchmark_rerun\smoke" `
  -Runs 1 `
  -MaxIterations 5000 `
  -ScenarioTimeLimit 15
```

Run the full paper benchmark with the original planner flags:

```powershell
.\run_benchmark_paper.ps1
```

The full command uses 1000 runs per scenario, endpoint reordering, terminal
reversal, travel and tangent continuity tests, robot execution clearance
checks, and saved success paths. The exact values are documented in
`README_PLANNING_AND_MUJOCO_PARAMETERS.md`.

## Recreate Reports

Generate the paper plots and table from the checked-in benchmark snapshot:

```powershell
python .\generate_paper_results.py `
  ".\results\paper_benchmark_1000" `
  --output-dir ".\results\paper_report\generated"
```

Regenerate the runtime distributions with and without Tukey outliers:

```powershell
python .\results\paper_report\outlier_analysis\plot_runtime_distributions.py
```

## MuJoCo Replay

Replay all ten fewest-configuration solutions interactively:

```powershell
.\run_mujoco_paper_10_fewest.ps1 -Interactive
```

Record all ten solutions and recompute settled final-shape errors:

```powershell
.\run_mujoco_paper_10_fewest.ps1
```

Generated videos and error files are written to `results/replay_outputs/`,
which is ignored by Git. Existing retained recordings are in `videos/`.

## Benchmark Notes

- Runtime statistics are computed over successful runs; all selected scenarios
  achieved 100% success in the archived 1000-run benchmark.
- `results/paper_report/paper_benchmark_table_original.tex` contains the table
  used before runtime-outlier filtering.
- `results/paper_report/outlier_analysis/` contains the separate sensitivity
  analysis using the upper Tukey fence `Q3 + 1.5 IQR` per scenario.
- Build products, caches, logs, new benchmark reruns, and replay outputs are
  excluded through `.gitignore`.

## Integrity

`SHA256SUMS` records the core source, aggregate benchmark CSVs, and retained
videos. Verify it from PowerShell with:

```powershell
Get-Content .\SHA256SUMS | ForEach-Object {
    $hash, $path = $_ -split '\s+', 2
    if ((Get-FileHash -Algorithm SHA256 $path).Hash.ToLower() -ne $hash) {
        throw "Checksum mismatch: $path"
    }
}
```

## Licensing

The project source does not currently declare an open-source license. Choose a
project license before making the repository public. The vendored UR10e model
retains the MuJoCo Menagerie license in
`third_party/mujoco_menagerie/universal_robots_ur10e/LICENSE`.
