# Planning And MuJoCo Parameters

This file summarizes the current parameters used by the GCR benchmark planner and by the MuJoCo replay scripts.

## Paper Benchmark Command

Run the canonical command from the repository root:

```bash
gcr benchmark
```

(or `.\scripts\run_benchmark_paper.ps1` on Windows PowerShell, which calls the
same command). The flag set lives in `PAPER_PLANNER_FLAGS` in
[cpp_benchmark.py](../src/gcr/cpp_benchmark.py).

The command selects source scenarios `2, 3, 5, 7, 8, 9, 10, 11, 12, 14`
from the original 15-scenario run and records 1000 runs per scenario. Its
defaults and flags reproduce the configuration represented by the checked-in
benchmark CSVs. In particular, later TCP-sector and cable-axis-turn experiments
are not enabled by this paper command.

## Planner Parameters

`--runs 1000`
Runs each scenario 1000 times with different random seeds.

`--max-iterations 100000`
Maximum bidirectional tree iterations per scenario/run.

`--scenario-time-limit 120`
Maximum solve time per scenario/run in seconds. A value of `0` disables the time limit.

`--endpoint-reorder`
Allows reversing ordered cable profiles, changing `[P0..PN]` to `[PN..P0]`, when that gives better endpoint continuity and the transition remains valid.

`--allow-terminal-reversal`
Allows the final goal to be exported as `reverse(goal)`. The geometric cable shape is the same, but the robot-to-cable-end assignment is swapped.

`--endpoint-reorder-cost-ratio 0.75`
Reverses a profile when reversed endpoint travel is less than `75%` of ordered endpoint travel.

`--endpoint-reorder-tangent-margin 0.05`
Reverses a profile when doing so improves the end-to-end cable tangent continuity by more than `0.05`. This prevents fake 180 degree ordered-cable flips.

`--robot-base-aware-endpoint-reorder`
Adds robot-execution awareness to endpoint reordering and shortcut refinement. For each possible profile reversal, the planner compares TCP travel, crossing TCP paths, base-to-TCP arm ray clearance, endpoint-vector continuity, and whether the left-to-right endpoint direction remains compatible with the robot base layout. It only applies the reversal if the reversed cable order remains transition-valid and has a clearly better execution score. During shortcut refinement, it also rejects shortcuts that make this robot-base execution score worse than the intermediate subpath they remove. By default this is a soft ordering preference, so it does not add a large search-time penalty.

The robot-base-aware check evaluates both global endpoint assignments: left robot holding `P0` and left robot holding `PN`. This matters for cases such as the S-corridor, where the same cable shape can be executable only if the robots hold opposite cable ends.

`--robot-base-vector-endpoint-reorder`
Adds the explicit vector-sector rule discussed for the robot start poses. For each adjacent profile pair, it looks at the left-to-right TCP vector and compares its angular turn against the previous profile. The acceptable sector is centered from each robot base to its TCP, with an extra base-to-TCP modifier: when the right robot reach vector approaches the forward `90 deg` direction, its preferred sector rotates clockwise; the left robot uses the mirrored counter-clockwise adjustment. If the ordered cable profile would rotate the TCP vector too far, but reversing the ordered cable points keeps the vector turn inside the allowed sector, the profile is reversed, provided the cable transition remains valid. This is separate from the soft score above and is intended to catch endpoint-order flips that are visually obvious from robot-base/TCP vector directions.

`--robot-base-vector-max-turn-deg 100`
Preferred maximum angular turn of the left-to-right TCP vector between adjacent profiles. Smaller values make the vector-sector rule more aggressive.

`--robot-base-vector-margin-deg 5`
Minimum angular improvement required before the vector-sector rule chooses the reversed endpoint order when both orders are otherwise acceptable.

The `--robot-base-aware-endpoint-reorder` and
`--robot-base-vector-endpoint-reorder` options above remain available for
experiments, but they were not enabled in the archived paper benchmark.

`--robot-tcp-vector-sector-reorder`
Experimental endpoint-order preference added after the archived benchmark. It
compares the left-to-right TCP vector for ordered and reversed candidates and
prefers the order that stays in the robot-compatible angular sector.

`--robot-tcp-vector-sector-feasibility`
Turns the experimental TCP-vector sector from a preference into an acceptance
condition for candidate transitions.

`--robot-tcp-vector-sector-margin-deg 15`
Example 15 degree angular margin for the experimental TCP-vector sector
decision. The executable default is 5 degrees.

`--max-cable-axis-turn-deg 115`
Example setting for the post-benchmark cable-axis guard. It rejects adjacent profile transitions whose cable end-to-end axis rotates more than this amount. This is different from endpoint reordering: it prevents shortcut refinement or connection acceptance from hiding a large physical cable spin inside one transition. Smaller values such as `90` to `100` are more aggressive. The archived benchmark used the executable default of `180`, which disables this guard.

`--robot-base-arm-clearance 0.0`
Optional hard sampled clearance between the two robot base-to-TCP arm rays. The default is `0.0`, meaning soft scoring only. Use values like `0.025` or `0.035` only when you intentionally want stricter robot-overlap filtering; larger values can reject many otherwise valid cable paths and increase solve time.

`--robot-base-transition-samples 8`
Number of interpolation samples used by `--robot-base-arm-clearance`. This matters only when the hard clearance is greater than zero.

`--robot-execution-feasibility`
Enables robot-aware transition checks during planning.

`--robot-tcp-clearance 0.035`
Same-time TCP separation radius. Two TCP centers must stay about `2 * 0.035 = 0.070 m` apart.

`--robot-obstacle-clearance 0.003`
TCP swept-path clearance from obstacles. This is intentionally small so narrow passages are not over-rejected.

`--robot-swept-tcp-clearance 0.015`
Clearance radius between the swept TCP path of one robot and the swept TCP path of the other. The centerline clearance is about `0.030 m`.

`--no-shortcut-refinement`
Disables shortcut refinement and keeps the raw connected path.

`--save-all-success-paths`
Writes every successful run path so the fewest/most/fastest solutions can be selected and plotted later.

## Planner Internal Validity Checks

Static cable collision:
Each accepted cable profile must be collision-free with obstacles.

Segment length bounds:
Cable segments must stay within the configured elastic length limits.

Workspace bounds:
Cable points must remain inside the `1.0 m x 0.6 m` workspace.

Connection polygon collision:
The polygon connecting two cable profiles must not intersect obstacles.

Swept segment quads:
Each cable segment's swept quadrilateral between two profiles must not intersect obstacles.

Interpolated transition samples:
Intermediate linear samples between two profiles must be cable-collision-free.

Endpoint/order continuity:
When endpoint reordering is enabled, the planner checks endpoint travel cost and cable tangent continuity before reversing profile order.

Robot execution feasibility:
When enabled, the planner checks TCP-vs-obstacle sweep clearance, same-time TCP separation, and swept TCP-vs-TCP clearance.

## Plot And Export Commands

Run from the repository root:

```bash
gcr paper-report results/paper_benchmark_1000 \
  --paper-output-dir results/paper_report/generated
```

## Recommended MuJoCo Replay Command Pattern

Run from the repository root:

```bash
gcr replay \
  --csv "path_to_waypoints.csv" \
  --obstacles "path_to_obstacles.csv" \
  --yaw-mode simultaneous \
  --steps-per-waypoint 300 \
  --cable-attachment-mode site_weld \
  --cable-weld-torquescale 6.0 \
  --final-error-wait 10 \
  --endpoint-filter choose-global-swap \
  --tcp-schedule auto-stagger \
  --side-correction-moving-robot right
```

`gcr paper-replay` applies exactly this flag set to every selected solution.

## MuJoCo Replay CLI Parameters

`--csv`
Waypoint CSV containing left/right TCP x, y, yaw and exported cable profile metadata.

`--obstacles`
Obstacle CSV exported with the scenario. Obstacles are drawn with the same x-y footprint and height up to cable height.

`--yaw-mode simultaneous`
Interpolates TCP translation and yaw together.

`--yaw-mode rotate-then-move`
Rotates first, then translates. Useful for comparison, but can twist the cable in some scenarios.

`--steps-per-waypoint 300`
Number of MuJoCo control steps per waypoint transition.

`--max-speed 0.10`
Peak gripper velocity cap in m/s. Set `--max-speed 0` to use only `--steps-per-waypoint`.

`--invert-red-anchor-yaw`
Inverts the yaw winding direction for the red-anchor robot while preserving the final yaw.

`--cable-attachment-mode site_weld`
Uses site weld constraints between each gripper anchor and the cable endpoint.

`--cable-attachment-mode two_point_clamp`
Adds endpoint and nearby-segment constraints to model a short clamp. Useful experimentally, but can be stiffer.

`--cable-attachment-mode body_weld`
Legacy rigid body weld attachment.

`--cable-weld-torquescale 6.0`
Overrides endpoint weld rotational coupling. Larger values transmit yaw more strongly, but too large can destabilize the cable.

`--tcp-schedule simultaneous`
Moves both TCPs with the same progress parameter.

`--tcp-schedule auto-stagger`
Automatically picks a left/right lead schedule when staggered timing looks safer.

`--tcp-schedule left-first` / `--tcp-schedule right-first`
Forces one TCP to lead and the other to follow.

`--tcp-stagger-fraction 0.35`
Delay fraction used by staggered TCP schedules.

`--endpoint-filter fixed`
Preserves the endpoint assignment exactly as written in the CSV.

`--endpoint-filter choose-global-swap`
Chooses between fixed CSV order and globally swapped endpoint assignment before replay.

`--endpoint-filter force-global-swap`
Forces the globally swapped endpoint assignment from the initial pose, without simulating a swap motion.

`--cable-endpoint-tangent-assist`
Enables experimental kinematic endpoint sleeve assist.

`--no-cable-endpoint-tangent-assist`
Keeps endpoint tangent assist disabled. This is the current default.

`--cable-endpoint-tangent-alpha 0.04`
Blend factor for tangent assist when enabled.

`--cable-endpoint-clamp-segments 2`
Number of endpoint cable segments treated as the gripper sleeve for tangent assist.

`--cable-endpoint-tangent-velocity-damping 0.0`
Optional qvel damping after tangent assist.

`--cable-length`
Overrides cable length. Default comes from CSV metadata, then `gcr.sim.build_scene`.

`--cable-segments`
Overrides MuJoCo cable segment count. Default comes from CSV metadata, then `gcr.sim.build_scene`.

`--cable-segment-multiplier 4`
When CSV metadata gives planner segment count, MuJoCo uses `planner_segments * 4` unless `--cable-segments` is given.

`--no-last-transition-side-correction`
Disables final-transition cable side correction.

`--side-correction-epsilon 1e-5`
Signed cross-product tolerance used to decide whether intermediate cable points are on the target side.

`--side-correction-cable-margin 0.003`
Distance margin below cable length before switching which robot moves during side correction.

`--side-correction-moving-robot farther-from-target`
During final-transition side correction, moves the robot farther from its target first.

`--side-correction-moving-robot left` / `right`
Forces which robot moves first during side correction.

`--ghost-target-shape final`
Shows a translucent non-colliding ghost of the final target cable. This is the current default.

`--ghost-target-shape all`
Shows translucent ghosts for all exported cable profiles.

`--no-ghost-target-shape`
Hides target ghost visualization.

`--ghost-target-rgba "1.0 0.86 0.18 0.35"`
RGBA color for the ghost target cable.

`--final-error-wait 10`
After the movement finishes, simulates for 10 more seconds before measuring final cable error.

`--final-error-json path.json`
Writes final error metrics to a JSON file. If omitted and `--final-error-wait > 0`, the JSON is written beside the CSV.

`--no-viewer`
Runs headless and saves video output.

`--video-path`
Optional MP4 output path for headless runs.

## MuJoCo Internal Simulation Parameters

These defaults live in [build_scene.py](../src/gcr/sim/build_scene.py).

`CABLE_LENGTH = 0.40 m`
Nominal MuJoCo cable length.

`CABLE_DIAMETER = 0.01 m`
Cable capsule diameter.

`CABLE_GROUND_CLEARANCE = 0.0025 m`
Bottom clearance from the ground plane.

`CABLE_Z = 0.0075 m`
Cable centerline height: radius `0.005 m` plus `0.0025 m` clearance.

`CABLE_NUM_SEGMENTS = 40`
Default standalone MuJoCo cable segment count. Exported CSV metadata can override this.

`CABLE_TWIST_STIFFNESS = 1e4`
Twist stiffness for MuJoCo's cable elasticity plugin.

`CABLE_BEND_STIFFNESS = 3e5`
Bend stiffness for MuJoCo's cable elasticity plugin.

`CABLE_JOINT_DAMPING = 0.04`
Damping on cable joints.

`CABLE_STRESS_VMAX = 0.05`
Velocity limit used by the elasticity stress settings.

`CABLE_GRIPPER_CLAMP_LENGTH = 0.035 m`
Physical length scale used for endpoint yaw coupling.

`CABLE_ENDPOINT_ATTACHMENT_MODE = "site_weld"`
Default cable endpoint attachment mode.

`CABLE_ENDPOINT_WELD_SOLREF = (0.02, 1.0)`
Legacy body weld solver reference.

`CABLE_ENDPOINT_SITE_WELD_SOLREF = (0.01, 1.0)`
Site weld solver reference.

`CABLE_ENDPOINT_SITE_WELD_SOLIMP = (0.90, 0.95, 0.001, 0.5, 2.0)`
Site weld solver impedance.

`CABLE_ENDPOINT_WELD_TORQUESCALE`
Defaults to `CABLE_GRIPPER_CLAMP_LENGTH = 0.035`, but can be overridden by the environment variable `DLO_CABLE_WELD_TORQUESCALE` or by `--cable-weld-torquescale`.

`CABLE_ENDPOINT_CONNECT_SOLREF = (0.008, 1.0)`
Solver reference for the two-point clamp endpoint connection.

`CABLE_ENDPOINT_CLAMP_SOLREF = (0.018, 1.0)`
Solver reference for the two-point clamp auxiliary clamp connection.

`CABLE_ENDPOINT_EXACT_SEGMENT_CLAMP_ENABLED = False`
Debug-only hard kinematic segment clamp. It can enforce endpoint segment orientation but can inject impulses and break the simulation.

`CABLE_ENDPOINT_TANGENT_PIN_ENABLED = False`
Disabled by default.

`CABLE_ENDPOINT_TANGENT_SOLREF = (0.003, 1.0)`
Solver reference for the optional tangent pin.

`CABLE_ENDPOINT_TANGENT_MAX_BEND_DEG = 65.0`
Maximum endpoint tangent bend angle used by tangent handling.

`GRIPPER_TIP_GEOM_SIZE = 0.0045 m`
Visual/collision size of the gripper tip marker.

`END_EFFECTOR_VISUAL_ENABLED = True`
Shows the visual adapter between robot flange and cable anchor.

`END_EFFECTOR_VISUAL_LENGTH = 0.115 m`
Length of the visual shaft from flange toward anchor.

`END_EFFECTOR_VISUAL_SHAFT_RADIUS = 0.009 m`
Radius of the visual shaft.

`END_EFFECTOR_VISUAL_COLLAR_RADIUS = 0.022 m`
Radius of the visual collar.

`END_EFFECTOR_VISUAL_COLLAR_HALFHEIGHT = 0.005 m`
Half-height of the visual collar.

`END_EFFECTOR_VISUAL_SLEEVE_LENGTH = 0.005 m`
Length of the colored red/green sleeve at the cable anchor.

`END_EFFECTOR_VISUAL_SLEEVE_RADIUS = 0.0075 m`
Radius of the colored red/green sleeve.

## MuJoCo Runtime Constants

These defaults live in [execute_plan.py](../src/gcr/mujoco_exec/execute_plan.py).

`DEFAULT_MAX_GRIPPER_SPEED = 0.20 m/s`
Default internal max gripper speed. `gcr replay` exposes `--max-speed` with default `0.10 m/s`.

`ROTATE_THEN_MOVE_YAW_FRACTION = 0.35`
Fraction of each waypoint used for yaw rotation in rotate-then-move mode.

`DEFAULT_TCP_STAGGER_FRACTION = 0.35`
Delay fraction for staggered TCP schedules.

`DEFAULT_TCP_MIN_DISTANCE = 0.11 m`
Nominal TCP separation used in robot-aware scoring.

`DEFAULT_ARM_SEGMENT_MIN_DISTANCE = 0.13 m`
Nominal arm segment clearance used in robot-aware scoring.

`DEFAULT_CABLE_LENGTH_MARGIN = 0.003 m`
Cable stretch margin used by side correction.

`DEFAULT_CABLE_SEGMENT_MULTIPLIER = 4`
Upsamples planner cable segments for MuJoCo simulation when CSV metadata is available.

## Environment Overrides

`DLO_CABLE_WELD_TORQUESCALE`
Overrides default cable endpoint weld torquescale.

`DLO_CABLE_ATTACHMENT_MODE`
Overrides default cable endpoint attachment mode.

`DLO_EXACT_SEGMENT_CLAMP`
Set to `1`, `true`, `yes`, or `on` to enable exact segment clamp debugging.

`DLO_ENDEFFECTOR_VISUAL`
Set to `0`, `false`, `no`, or `off` to hide the visual adapter.

`DLO_ENDEFFECTOR_VISUAL_LENGTH`
Overrides visual adapter shaft length.

`DLO_ENDEFFECTOR_VISUAL_SHAFT_RADIUS`
Overrides visual adapter shaft radius.

`DLO_ENDEFFECTOR_VISUAL_COLLAR_RADIUS`
Overrides visual collar radius.

`DLO_ENDEFFECTOR_VISUAL_COLLAR_HALFHEIGHT`
Overrides visual collar half-height.

`DLO_ENDEFFECTOR_VISUAL_SLEEVE_LENGTH`
Overrides colored sleeve length.

`DLO_ENDEFFECTOR_VISUAL_SLEEVE_RADIUS`
Overrides colored sleeve radius.
