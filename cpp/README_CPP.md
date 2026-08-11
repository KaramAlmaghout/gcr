# C++ GCR Benchmark Runner

This folder contains a standalone C++17 implementation of the final GCR
benchmark runner. It ports the planner/scenario/metrics core from the Python
final package:

- final 15-scenario catalogue;
- APF-biased sampling;
- projected relaxation;
- bidirectional tree expansion;
- strict transition validation;
- optional shortcut refinement;
- optional endpoint-order reordering for execution experiments;
- optional robot-execution feasibility checks for endpoint sweeps;
- per-run and summary CSV metrics.

It intentionally does not port plotting or MuJoCo replay. Use the existing
Python tools for figures, waypoint CSV export, and MuJoCo visualization.

## Build

On this Windows setup, CMake may not be installed. If `cmake` is not recognized,
use the included Visual Studio build script from this `cpp/` folder:

```powershell
.\build_msvc.bat
```

That creates:

```text
build\gcr_benchmark.exe
```

If CMake is installed, this also works:

```powershell
cd .\cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Or compile the single source file with any C++17 compiler, for example:

```powershell
g++ -std=c++17 -O3 -o gcr_benchmark.exe .\src\gcr_benchmark.cpp
```

## Run

Run all final scenarios once:

```powershell
.\build\Release\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_benchmark --runs 1
```

If you built with `build_msvc.bat`, use:

```powershell
.\build\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_benchmark --runs 1
```

Run one scenario:

```powershell
.\build\Release\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_s11 --scenario 11 --runs 1
```

Keep the raw connected path without shortcut refinement:

```powershell
.\build\Release\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_raw --no-shortcut-refinement
```

Try endpoint-order reordering:

```powershell
.\build\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_endpoint_reorder --endpoint-reorder
```

This is off by default. When enabled, the runner checks endpoint correspondence
both on the connected path and during shortcut refinement. If a shortcut from
profile `i` to profile `j` would make `p0(i)->p0(j)` intersect
`pN(i)->pN(j)`, the shortcut is tested against `reverse(C_j)` instead. When a
reversed shortcut is accepted, the suffix from `j` onward is reversed too, so
later cable profiles stay internally consistent. By default, this heuristic now
preserves the terminal goal endpoint order, so it will not finish at
`reverse(goal)`. Use `--allow-terminal-reversal` only if you want the old
experimental behavior. The output metrics record `endpoint_reorder_enabled`,
`preserve_terminal_order_enabled`, and `endpoint_order_reversals`.

For robot-base-aware endpoint assignment, add:

```powershell
.\build\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_robot_base_endpoint_reorder --endpoint-reorder --allow-terminal-reversal --robot-base-aware-endpoint-reorder
```

This keeps the same geometric cable validity checks, but adds a soft preference
for endpoint orders whose TCP travel, base-to-TCP directions, and swept endpoint
paths are friendlier for the two UR10 bases. The same flag also prevents
shortcut refinement from accepting a shortcut when it worsens this robot-base
execution score relative to the intermediate subpath it would remove. By
default this is a soft preference. To make it a hard sampled arm-ray clearance
filter, add for example `--robot-base-arm-clearance 0.025
--robot-base-transition-samples 16`; larger values are stricter and can increase
solve time. The robot-base-aware score evaluates both global assignments: left
holding `P0` and left holding `PN`.

For the explicit robot-base/TCP vector-sector endpoint ordering rule, add:

```powershell
.\build\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_robot_base_vector_endpoint_reorder --endpoint-reorder --allow-terminal-reversal --robot-base-vector-endpoint-reorder --robot-base-vector-max-turn-deg 100 --robot-base-vector-margin-deg 5
```

This checks the left-to-right TCP vector between adjacent profiles. If the
ordered cable profile would cause a large TCP-vector turn, but reversing
`[P0..PN]` to `[PN..P0]` keeps that vector direction more continuous relative
to the robot bases, the reversed endpoint order is preferred as long as the
cable transition remains valid. The preferred sector is centered from each
robot base to its TCP. As the right robot's base-to-TCP vector approaches the
forward `90 deg` direction, that sector is rotated clockwise; the left robot
uses the mirrored counter-clockwise rotation. It can be combined with
`--robot-base-aware-endpoint-reorder` and `--robot-execution-feasibility`.

To reject hidden cable-axis spins between adjacent profiles, add for example:

```powershell
--max-cable-axis-turn-deg 100
```

This guard is off by default at `180 deg`. It is useful for scenario 9 cases
where the fewest-configuration path is geometrically collision-free but rotates
the cable axis by more than 100 degrees in one transition, causing robot
execution to spin the cable and collide.

Save every successful run path for later selection plots:

```powershell
.\build\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_many_runs --runs 1000 --endpoint-reorder --save-all-success-paths
```

Try the stricter robot-execution feasibility layer:

```powershell
.\build\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_robot_exec --runs 1000 --endpoint-reorder --robot-execution-feasibility --save-all-success-paths
```

This is off by default. When enabled, every accepted transition also requires
the two endpoint motion segments, `p0(i)->p0(j)` and `pN(i)->pN(j)`, to avoid
obstacles with a TCP clearance radius. The default clearance is `0.0` m, so pass
a value if you want a nonzero TCP margin:

```powershell
.\build\gcr_benchmark.exe ..\results\benchmark_rerun\cpp_robot_exec_clearance --runs 1000 --endpoint-reorder --robot-execution-feasibility --robot-tcp-clearance 0.035 --save-all-success-paths
```

The same check also rejects transitions where the two endpoint/TCP centers get
closer than twice this clearance. The output metrics record
`robot_execution_feasibility_enabled` and `robot_tcp_clearance_m`.

Then `plot_cpp_solution_extremes.py` can plot the fewest-configuration,
most-configuration, and fastest successful path per scenario from those saved
run paths.

## Outputs

The runner writes:

- `final_metrics_all.csv`;
- `final_benchmark_summary.csv`;
- one `metrics.csv` per scenario folder.
- optional `run_paths/run_####_success_path.csv` files when
  `--save-all-success-paths` is used.

## Math Parity With Python

The C++ runner ports the final Python planner math from `gcrbup.py` and
`final_scenarios.py`:

- same final 15-scenario catalogue;
- same cable representation `C = (p_1, ..., p_N)` in 2D;
- same distance `d(C_i,C_j) = max_n ||p_n^i - p_n^j||`;
- same deformation energy terms;
- same APF force equations and step limiting;
- same projected relaxation objective and projections;
- same strict `PathFree` predicate;
- same bidirectional tree expansion and shortcut acceptance;
- same default planner parameters.

The `--endpoint-reorder` and `--robot-execution-feasibility` options are extra
execution heuristics and are not part of the default paper-math benchmark path.
Leave them off when you want the final Python/C++ math comparison to stay
unchanged.

It is not bit-identical to Python because the C++ RNG is `std::mt19937_64`
while Python uses NumPy `default_rng`, and floating-point operation order can
differ slightly. Therefore the exact sampled paths, iterations, and timings can
differ, but the planner math and benchmark definitions are intended to match the
final Python implementation.

For hard cases, you can raise `--max-iterations`, for example
`--max-iterations 5000`; the output CSV records the iteration limit used by
each run.
