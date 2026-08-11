# Scripts

PowerShell wrappers for the workflows the paper release was run with on
Windows. Each one is a thin call into the `gcr` command, so there is a single
implementation of every pipeline:

| Script | Equivalent command |
|---|---|
| `run_benchmark_paper.ps1` | `gcr benchmark` |
| `plot_export_from_benchmark.ps1` | `gcr paper-report` |
| `run_mujoco_paper_10_fewest.ps1` | `gcr paper-replay` |

The `gcr` commands work on Windows, Linux, and macOS and take the same
options; use the scripts only if you prefer the original PowerShell entry
points. Run them from anywhere — each one switches to the repository root
first.
