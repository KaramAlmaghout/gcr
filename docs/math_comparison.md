# Math Comparison: Final Script vs. GCR Paper

This note compares the planner math implemented in the final package with the
formulation in the GCR paper. The final scenario catalogue in
`final_scenarios.py` changes the benchmark geometries only; the planner math is
in `gcrbup.py`. MuJoCo modules under `gcr/mujoco_exec/` and `gcr/sim/` replay
exported waypoints and do not change the GCR planner objective.

File names and line numbers below refer to the planner package:
`gcrbup.py` is [src/gcr/planner/gcrbup.py](../src/gcr/planner/gcrbup.py) and
`final_scenarios.py` is
[src/gcr/planner/final_scenarios.py](../src/gcr/planner/final_scenarios.py).

## Short Summary

The final script keeps the same core math as the paper:

- cable configurations are 2D node chains `C = (p_1, ..., p_N)`;
- the configuration distance is `d(C_i, C_j) = max_n ||p_n^i - p_n^j||`;
- APF biasing uses attractive and repulsive potentials;
- local relaxation preserves segment vectors, bending, endpoint positions, and
  obstacle clearance;
- two trees grow bidirectionally and connect only through feasible transitions;
- final path cost is `J(P) = sum_k d(C_k, C_{k+1})`;
- shortcut refinement is optional and validated with the same path-free logic.

The main implementation differences are numerical and validation choices:

- sampling is guided/tree-biased for repeatable benchmark solving instead of
  being purely uniform at every draw;
- `path_free` is stricter than the paper's connection-polygon test because it
  also checks swept segment quadrilaterals and sampled interpolated cable
  states;
- relaxation is solved by projected-gradient/projection sweeps rather than by
  calling a generic convex optimizer;
- APF descent includes step limiting and segment-length projection to avoid
  unstable intermediate cables.

## Equation-to-Code Map

| Paper math | Meaning | Final code | Match |
|---|---|---|---|
| Eq. (7) | Cable configuration `C = (p_1, ..., p_N)`, `p_n in R^2` | `line_nodes`, `polyline_nodes`, all cable arrays use shape `(N, 2)` in `gcrbup.py` | Direct |
| Eq. (8) | Segment `s_k(C) = [p_k, p_{k+1}]` | `segment_lengths`, segment collision utilities | Direct |
| Eqs. (9)-(10) | Nominal segment length and admissible length bounds | `nominal_segment_length`, `configuration_admissible`; bounds from `ell_min_ratio`, `ell_max_ratio` in `PlannerParams` | Direct, parameterized |
| Eqs. (13)-(15) | Stretching/bending energy ideas | `deformation_energy`, `relaxation_gradient` | Same segment-vector and second-difference bending terms used in planner cost/relaxation |
| Eqs. (16)-(17) | Feasible set `M` and collision-free set `C_free` | `configuration_admissible`, `cable_collision_free` | Direct static feasibility check |
| Eq. (19) | `d(C_i, C_j) = max_n ||p_n^i - p_n^j||` | `configuration_distance` at `gcrbup.py:122` | Direct |
| Eqs. (25)-(29) | Linear interpolation and transition/path-free condition | `connection_polygon` at `gcrbup.py:516`, `path_free` at `gcrbup.py:546` | Stricter than paper |
| Eqs. (30)-(33) | Feasible path and path objective `J(P)` | `compute_metrics` at `gcrbup.py:1206`, `refine_path` at `gcrbup.py:1171` | Direct cost metric |
| Eqs. (34)-(39) | Bidirectional start/goal trees and alternating expansion | `GCRPlanner.plan` and `expand_tree` | Same algorithmic structure |
| Eq. (40) | Raw configuration sample | `sample_raw_configuration` at `gcrbup.py:1083`, `sample_broad_configuration` at `gcrbup.py:1136` | Guided implementation of the sampler |
| Eqs. (41)-(45) | Configuration potential `U = U_att + U_rep` and force field | `potential_values` at `gcrbup.py:612`, `configuration_forces` at `gcrbup.py:623` | Direct structure |
| Eqs. (46)-(47) | Attractive potential and force | `u_att = 0.5*k_att*sum(||C-target||^2)`, `f_att = k_att*(target-C)` | Direct |
| Eqs. (48)-(50) | Segment samples and obstacle repulsion | `segment_sample_points`, `signed_distance_to_obstacles`, `configuration_forces` | Direct outside obstacles; robust signed push-out inside obstacles |
| Eqs. (51)-(53) | APF descent update | `apf_descent` at `gcrbup.py:644` | Same update with numerical step cap and length projection |
| Eqs. (54)-(59) | Biased sample and local relaxation | `relax_configuration` at `gcrbup.py:755` | Same objective/constraints, numerical projected implementation |
| Eqs. (60)-(62) | Segment-deviation bound and static validation | `project_segment_vector_deviation`, `configuration_admissible` | Direct constraint/check |
| Eqs. (63)-(66) | Parent-to-candidate transition validation | `select_parent`, `path_free` | Direct plus stricter transition tests |
| Eqs. (67)-(73) | Feasible parent selection, insertion, and cost-to-come | `select_parent`, `expand_tree`, `deformation_energy` | Direct |
| Eqs. (74)-(81) | Opposite-tree connection and path extraction | `GCRPlanner.plan`, `SearchTree.trace_root_to`, `SearchTree.trace_to_root` | Direct |
| Eqs. (82)-(85) | Shortcut refinement | `refine_path`; controlled by `shortcut_refinement` / `--no-shortcut-refinement` | Direct and optional |

## Detailed Comparison

### 1. Cable Configuration

The paper models the cable as a finite 2D chain:

```text
C = (p_1, ..., p_N),    p_n in R^2
```

The final script uses exactly this representation: every cable profile is a
NumPy array with shape `(N, 2)`. `line_nodes` creates straight configurations,
and `polyline_nodes` creates piecewise-linear configurations sampled into the
same number of nodes.

The final package uses the final 13-scenario set from `final_scenarios.py`, but
those scenarios are just inputs: workspace, obstacles, start cable, goal cable,
seed, and optional guide points/configurations.

### 2. Segment Length and Static Feasibility

The paper constrains cable segment lengths around a nominal segment length. In
the final code:

```text
ell0 = mean(segment_lengths(C_start))
ell_min = ell_min_ratio * ell0
ell_max = ell_max_ratio * ell0
```

`configuration_admissible` then checks:

- all nodes stay inside the workspace;
- every segment length is within `[ell_min, ell_max]`;
- the cable is collision-free against polygon and circular obstacles.

This is a direct implementation of the paper's admissible-configuration check,
with the tolerance ratios exposed in `PlannerParams`.

### 3. Configuration Distance and Path Cost

The paper's distance metric is the maximum pointwise node displacement:

```text
d(C_i, C_j) = max_n ||p_n^i - p_n^j||
```

The final code implements the same formula in `configuration_distance`:

```python
diff = config_a - config_b
return sqrt(max(sum(diff * diff, axis=1)))
```

The path objective is also the same:

```text
J(P) = sum_k d(C_k, C_{k+1})
```

`compute_metrics` reports this as `path_cost_J`. It also reports the raw path
cost, shortcut reduction percentage, deformation energy, clearance, static
validity, and transition validity.

### 4. APF Biasing

The paper uses an artificial potential field:

```text
U(C) = U_att(C, C_target) + U_rep(C, O)
```

The final script keeps the same structure:

```text
U_att = 0.5 * k_att * sum_n ||p_n - p_n,target||^2
F_att = k_att * (C_target - C)
```

For obstacle repulsion, the implementation samples points along each cable
segment, computes signed/unsigned obstacle distances, and distributes the
repulsive force back to the segment endpoints using the segment interpolation
weights. Outside obstacles, this follows the paper's exponential distance-based
repulsion. If a sample is already inside an obstacle, the code uses a signed
push-out direction because the ideal distance expression is not useful there.
That is a robustness addition, not a change to the intended APF role.

The APF descent update is:

```text
C <- C + eta * F(C)
```

with two stabilizers:

- per-node step limiting through `max_apf_node_step`;
- segment-length projection after each APF step.

These stabilizers prevent the APF sample from becoming numerically invalid
before relaxation.

### 5. Local Relaxation

The paper relaxes the APF-biased cable by minimizing tracking/deformation terms
while enforcing endpoint, segment, and obstacle-clearance constraints.

The final script implements the same mathematical terms:

- tracking to the APF-biased configuration;
- segment-vector preservation relative to a reference configuration;
- bending preservation using second differences;
- fixed endpoints;
- obstacle clearance at sampled segment points;
- segment-vector deviation bounds.

The practical difference is the solver. Instead of sending Eq. (59) to a
generic convex-program solver, `relax_configuration` performs projected-gradient
steps:

1. take a gradient step on the tracking, segment, and bending objective;
2. restore fixed endpoints;
3. project against linearized obstacle planes;
4. periodically apply exact clearance push-out;
5. project segment-vector deviations;
6. clip to the workspace.

So the objective and constraints match the paper, but the numerical optimizer is
custom and lightweight.

### 6. Transition Validation

The paper defines a path-free check for moving from one cable configuration to
another. The final implementation is intentionally stricter.

`path_free(C_a, C_b)` requires all of the following:

- the paper-style connection polygon is obstacle-free;
- every individual segment swept quadrilateral is obstacle-free;
- sampled interpolated cable configurations between `C_a` and `C_b` are
  collision-free.

This makes some transitions that might pass a coarse paper-level check fail in
the implementation, but it avoids exporting unsafe paths to MuJoCo.

### 7. Bidirectional GCR Search

The final planner follows the paper's bidirectional tree structure:

- initialize one tree at `C_start` and one at `C_goal`;
- alternate which tree is expanded;
- sample a raw candidate;
- bias it with APF toward the opposite side;
- relax it;
- reject it if static feasibility fails;
- choose a feasible parent inside `epsilon`;
- insert the node with cost-to-come;
- attempt connection to the opposite tree.

The code stores edges through parent pointers in `SearchTree`, which is an
implementation detail equivalent to an explicit edge set.

### 8. Shortcut Refinement

The paper includes shortcut refinement after a path is found. The final script
implements this in `refine_path`.

A shortcut is accepted only if:

```text
d(C_i, C_j) < sum_k d(C_k, C_{k+1})
```

for the skipped section, and the shortcut transition is `path_free`. The final
code validates shortcuts with dense transition sampling through
`shortcut_validation_samples`.

This step is controllable:

```bash
gcr plan results/benchmark_rerun/python_raw --no-shortcut-refinement
gcr benchmark results/benchmark_rerun/cpp_raw --no-shortcut-refinement
gcr export-scenario-waypoints results/gcr_waypoints --no-shortcut-refinement
```

## Final Benchmark and MuJoCo Notes

The final C++ and Python catalogues contain 15 source scenarios. The archived
paper benchmark in this repository evaluates source scenarios
`2, 3, 5, 7, 8, 9, 10, 11, 12, 14`, mapped to paper scenarios 1 through 10 as
listed in the root [README.md](../README.md).

This changes only the benchmark inputs, not the planner math.

MuJoCo replay uses exported cable waypoints and obstacle CSV files. It maps the
2D planner workspace to a 1:1 3D workspace, preserves cable length from the
exported path, and may use a denser simulated cable through segment resampling.
That denser MuJoCo cable is for physics replay compliance; the GCR path itself
is still generated from the planner's node chain.

## Main Differences From the Paper

This section replaces the prose-only summary with the actual math used by the
final code where it is not a literal line-by-line implementation of the paper.
The table gives the short form first; the subsections below expand each item.

| Difference | Paper form | Final code math |
|---|---|---|
| Guided sampling | `C_raw ~ Uniform(X)` | `C_raw ~ p_goal*q_goal + p_tree*q_tree + p_broad*q_broad`, with `p_goal=0.06`, `p_tree=0.84`, `p_broad=0.10` |
| Tree-biased sample | not specified beyond raw sampling | `C_base = C_p + alpha*(C_l - C_p)`, `alpha = clip(0.70*epsilon/d(C_p,C_l), 0.16, 0.64)` |
| APF stabilizer | `C^{t+1}=C^t+eta*F(C^t)` | `C^{t+1}=Pi_X Pi_len Pi_X(C^t + clip_smax(eta*F(C^t)))` |
| Inside-obstacle repulsion | obstacle repulsion from distance field | `f_rep_inside = (beta/sigma)*n_out` |
| Transition check | connection polygon is free | `Free(Gamma) and all_k Free(Q_k) and all_tau CollisionFree((1-tau)C_a+tau C_b)` |
| Relaxation solve | constrained optimization problem | `C^{m+1}=Pi_X Pi_seg Pi_clear Pi_planes Pi_end(C^m - h*grad Phi(C^m))` |
| Shortcut acceptance | cost reduction plus path-free | `new_cost < old_cost and new_cost <= max_shortcut_distance and PathFree_code(C_i,C_j)` |
| MuJoCo replay resampling | not planner math | `q_j = C_start(j*L/M)`, arc-length resampling for denser simulated cable |

### 1. Guided raw sampling

The paper describes the raw sample as a sample from the ambient configuration
space. The final code instead uses a mixture sampler:

```text
p_goal = 0.06
p_tree = 0.84
p_broad = 0.10

C_raw =
  target-biased sample,      with probability p_goal
  tree-biased sample,        with probability p_tree
  broad random-walk sample,  with probability p_broad
```

For the tree-biased branch, the code chooses a parent cable `C_p` from the
active tree, often from nodes already closest to the opposite target. It then
steps only part of the way toward a local target `C_l`:

```text
alpha = clip(0.70 * epsilon / d(C_p, C_l), 0.16, 0.64)
C_base = C_p + alpha * (C_l - C_p)
```

The local target is either the next guide configuration, the next guide-point
center, or the final target. After that, the code adds noise and a smooth
lateral perturbation:

```text
C_raw = Pi_X Pi_len(
    C_base + Xi + a * sin(pi*i/(N-1)) * n_i(C_base)
)
```

where `Pi_X` clips to the workspace, `Pi_len` projects segment lengths back
toward the nominal length, `Xi` is node-wise Gaussian noise, `a` is a scalar
Gaussian lateral amplitude, and `n_i(C_base)` is the local cable normal at node
`i`.

For the broad branch, the code creates a random walk with fixed nominal segment
length:

```text
p_0 ~ Uniform(workspace interior)
theta_0 ~ Uniform(0, 2*pi)
theta_{k+1} = theta_k + eta_k,    eta_k ~ Normal(0, 0.36^2)
p_{k+1} = p_k + ell0 * [cos(theta_k), sin(theta_k)]
```

Then the whole cable is shifted/clipped into the workspace.

### 2. Repulsive force inside obstacles

Outside obstacles, the code uses the same exponential APF idea as the paper.
For each sampled point `x` on a cable segment:

```text
U_rep(x) = beta * exp(-d(x,O)^2 / (2*sigma^2))
f_rep(x) = beta * d(x,O) / sigma^2
           * exp(-d(x,O)^2 / (2*sigma^2)) * n(x)
```

The implementation adds a special inside-obstacle rule. If a segment sample is
already inside an obstacle, the distance-gradient expression is not enough to
recover cleanly, so the code uses:

```text
f_rep_inside(x) = (beta / sigma) * n_out(x)
```

The sample force is distributed back to the segment endpoints using the segment
interpolation coordinate `rho`:

```text
F_k     += (1 - rho) * f_rep(x)
F_{k+1} += rho       * f_rep(x)
```

### 3. APF step limiting and length projection

The paper APF update is:

```text
C^{t+1} = C^t + eta * F(C^t)
```

The final code applies the same update, but caps the maximum node step:

```text
Delta_i = eta * F_i
s = max_i ||Delta_i||

if s > s_max:
    Delta_i <- Delta_i * s_max / s
```

Then it projects the result:

```text
C^{t+1} = Pi_X Pi_len Pi_X(C^t + Delta)
```

where `s_max = max_apf_node_step`. This keeps APF-biased samples from
stretching too far before relaxation.

### 4. Stricter path-free predicate

The paper path-free check is based on the connection polygon between two
configurations. The final code uses this stronger predicate:

```text
PathFree_code(C_a, C_b) =
    Free(Gamma(C_a, C_b))
    and all_k Free(Q_k(C_a, C_b))
    and all_tau CollisionFree((1 - tau) C_a + tau C_b)
```

where:

```text
Gamma(C_a, C_b) = polygon[p_1^a, ..., p_N^a, p_N^b, ..., p_1^b]
Q_k(C_a, C_b)  = quad[p_k^a, p_{k+1}^a, p_{k+1}^b, p_k^b]
tau in linspace(0, 1, transition_samples + 2)[1:-1]
```

So a transition must pass the connection-polygon check, every individual
segment sweep check, and sampled interpolation checks.

### 5. Projected relaxation instead of a generic convex solver

The paper presents local relaxation as a constrained optimization problem. The
code uses a projected-gradient implementation of the same objective terms:

```text
Phi(C) =
    0.5*w_track * ||C - C_b||^2
  + 0.5*lambda_s * sum_k ||s_k(C) - s_k(C_ref)||^2
  + 0.5*lambda_b * sum_k ||b_k(C) - b_k(C_ref)||^2
```

where:

```text
s_k(C) = p_{k+1} - p_k
b_k(C) = p_{k+2} - 2*p_{k+1} + p_k
```

Each relaxation iteration performs:

```text
C_half = C^m - h * grad Phi(C^m)
C^{m+1} = Pi_X Pi_seg Pi_clear Pi_planes Pi_end(C_half)
```

The projections enforce:

```text
p_1 = p_1^biased,  p_N = p_N^biased
signed_distance(sample, obstacle) >= dsafe
||s_k(C) - s_k(C_ref)|| <= delta_l
C inside workspace
```

For exact clearance push-out, when a sampled point
`x = (1-rho)*p_k + rho*p_{k+1}` violates clearance by:

```text
v = dsafe - signed_distance(x, O)
```

the code shifts the free endpoints along the outward normal `n`:

```text
denom = (1-rho)^2 + rho^2
p_k     += ((1-rho) / denom) * v * n
p_{k+1} += (rho       / denom) * v * n
```

with weights removed for fixed endpoints.

### 6. Shortcut acceptance uses the stronger predicate

The shortcut rule is still cost reduction plus feasibility, but in the code it
uses the stronger `PathFree_code` predicate and denser shortcut sampling:

```text
old_cost = sum_{k=i}^{j-1} d(C_k, C_{k+1})
new_cost = d(C_i, C_j)

accept shortcut i -> j iff:
    new_cost < old_cost
    and new_cost <= max_shortcut_distance
    and PathFree_code(C_i, C_j)
```

The transition sample count for shortcuts is:

```text
max(transition_samples, shortcut_validation_samples)
```

The default `shortcut_validation_samples` is 100.

### 7. MuJoCo replay-only resampling

This is not part of the GCR planner math, but it is used by the final package
when running the solved paths in MuJoCo. The exported planner cable has `N`
nodes and length:

```text
L = sum_k ||p_{k+1} - p_k||
```

If MuJoCo uses a denser cable with `M` segments, the initial cable is resampled
by arc length:

```text
s_j = j * L / M,    j = 0, ..., M
q_j = C_start(s_j)
```

This preserves the planner cable length and shape while giving the physics
simulation more cable segments.

## Bottom Line

The final script is mathematically aligned with the paper at the level of cable
representation, distance, APF biasing, local relaxation terms, bidirectional GCR
search, connection validation, and path-cost metrics. The code-level formulas
above are the intentional differences: guided sampling, stronger transition
validation, inside-obstacle push-out, APF/relaxation projections, optional
shortcut refinement, and MuJoCo-only arc-length resampling.
