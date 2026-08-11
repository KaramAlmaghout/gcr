# Result Snapshot

`paper_benchmark_1000/` is the compact final benchmark used by the paper:

- `final_metrics_all.csv`: all 10,000 run-level metric rows;
- `final_benchmark_summary.csv`: per-scenario aggregate output from C++;
- `<scenario>/metrics.csv`: the 1000 rows for that source scenario;
- `<scenario>/run_paths/`: only the paths selected as fastest, fewest
  configurations, or most configurations;
- `selected_solution_paths/`: selection summary and overview plots;
- `mujoco_selected_waypoints/`: exported waypoint and obstacle CSV files.

The complete set of 10,000 successful path CSVs was omitted to keep Git history
manageable. Rerunning `gcr benchmark` recreates them.

Generated output folders (`benchmark_rerun/`, `replay_outputs/`,
`paper_report/generated/`, `gcr_waypoints/`, `figures/`) are ignored by Git;
everything else here is a tracked paper artifact.

`paper_report/outlier_analysis/` contains a separate runtime sensitivity
analysis. It removes only upper runtime outliers identified independently for
each scenario by `time > Q3 + 1.5 IQR`; it does not redefine planner success.

`mujoco_final_errors/` contains the retained 10-second settled cable-shape
measurements from the recorded MuJoCo runs.
