"""Final experiment scenario catalogue.

This module combines:

* the 10 planner benchmark scenarios from ``gcr_last/gcr_claude_benchmark.py``;
* the current project scenarios 2, 6, 8, 4, and 9.

It uses the current ``gcrbup`` planner/scenario classes so the same catalogue
can be plotted, benchmarked, exported to MuJoCo CSV, and replayed with the
MuJoCo tools packaged in this folder.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

import gcrbup as gcr
from generate_dlo_workspace_figure import get_problem_setup as get_dlo_problem_setup


DEFAULT_N_NODES = 10
DEFAULT_SEGMENT_LENGTH = 0.030


def straight_at(
    center: Sequence[float],
    angle_rad: float,
    *,
    n_nodes: int = DEFAULT_N_NODES,
    segment_length: float = DEFAULT_SEGMENT_LENGTH,
) -> np.ndarray:
    length = (n_nodes - 1) * segment_length
    c = np.asarray(center, dtype=float)
    d = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
    return gcr.line_nodes(c - 0.5 * length * d, c + 0.5 * length * d, n_nodes)


def bent_at(
    start: Sequence[float],
    dir_a: Sequence[float],
    n_a: int,
    dir_b: Sequence[float],
    *,
    n_nodes: int = DEFAULT_N_NODES,
    segment_length: float = DEFAULT_SEGMENT_LENGTH,
) -> np.ndarray:
    p1 = np.asarray(start, dtype=float)
    da = np.asarray(dir_a, dtype=float)
    db = np.asarray(dir_b, dtype=float)
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    n_total_segments = n_nodes - 1
    n_b = n_total_segments - n_a
    corner = p1 + n_a * segment_length * da
    p2 = corner + n_b * segment_length * db
    leg1 = gcr.line_nodes(p1, corner, n_a + 1)
    leg2 = gcr.line_nodes(corner, p2, n_b + 1)
    return np.vstack([leg1, leg2[1:]])


def arc_at(
    center: Sequence[float],
    theta_start: float,
    theta_end: float,
    *,
    n_nodes: int = DEFAULT_N_NODES,
    chord: float = DEFAULT_SEGMENT_LENGTH,
) -> np.ndarray:
    dtheta = (theta_end - theta_start) / (n_nodes - 1)
    radius = chord / (2.0 * np.sin(abs(dtheta) / 2.0))
    thetas = np.linspace(theta_start, theta_end, n_nodes)
    cx, cy = center
    return np.column_stack([cx + radius * np.cos(thetas), cy + radius * np.sin(thetas)])


def rounded_l_down_right(
    top: Sequence[float],
    *,
    n_nodes: int = DEFAULT_N_NODES,
    chord: float = DEFAULT_SEGMENT_LENGTH,
    vertical_segments: int = 3,
    arc_segments: int = 3,
) -> np.ndarray:
    n_segments = n_nodes - 1
    arc_segments = min(max(2, arc_segments), n_segments - 2)
    vertical_segments = min(max(1, vertical_segments), n_segments - arc_segments - 1)
    horizontal_segments = n_segments - vertical_segments - arc_segments

    top_arr = np.asarray(top, dtype=float)
    points = [top_arr.copy()]
    for i in range(1, vertical_segments + 1):
        points.append(top_arr + np.array([0.0, -chord * i], dtype=float))

    delta = 0.5 * np.pi / arc_segments
    radius = chord / (2.0 * np.sin(0.5 * delta))
    arc_start = points[-1]
    arc_center = arc_start + np.array([radius, 0.0], dtype=float)
    for i in range(1, arc_segments + 1):
        theta = np.pi + i * delta
        points.append(arc_center + radius * np.array([np.cos(theta), np.sin(theta)], dtype=float))

    bend_end = points[-1]
    for i in range(1, horizontal_segments + 1):
        points.append(bend_end + np.array([chord * i, 0.0], dtype=float))

    return np.asarray(points, dtype=float)


def _guide(points: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(points, dtype=float)


def build_gcr_last_benchmark_scenarios(n_nodes: int = DEFAULT_N_NODES) -> List[gcr.Scenario]:
    """Return the 10 scenarios from ``gcr_last/gcr_claude_benchmark.py``."""
    workspace = (0.0, 1.0, 0.0, 0.6)
    scenarios: List[gcr.Scenario] = []

    scenarios.append(gcr.Scenario(
        name="scenario_01_empty_rotation",
        title="01: Rotation with top-left + top-right obstacles",
        workspace=workspace,
        obstacles=gcr.ObstacleSet(polygons=[
            gcr.rectangle(0.27, 0.38, 0.45, 0.55),
            gcr.rectangle(0.55, 0.38, 0.73, 0.55),
        ]),
        start=straight_at((0.50, 0.30), 0.0, n_nodes=n_nodes),
        goal=straight_at((0.50, 0.30), np.pi / 2.0, n_nodes=n_nodes),
        epsilon=0.22,
        seed=101,
        guide_points=_guide([[0.50, 0.30], [0.50, 0.20], [0.55, 0.15], [0.50, 0.20], [0.50, 0.30]]),
    ))

    scenario_02_goal = rounded_l_down_right(
        (0.18, 0.34),
        n_nodes=n_nodes,
    )
    scenarios.append(gcr.Scenario(
        name="scenario_02_block_reorient",
        title="02: Reorient around an L-extended block with rounded target",
        workspace=workspace,
        obstacles=gcr.ObstacleSet(polygons=[
            gcr.rectangle(0.30, 0.20, 0.45, 0.45),
            gcr.rectangle(0.10, 0.40, 0.30, 0.45),
        ]),
        start=straight_at((0.15, 0.52), 0.0, n_nodes=n_nodes),
        goal=scenario_02_goal,
        epsilon=0.22,
        seed=202,
        guide_points=_guide([[0.15, 0.52], [0.05, 0.48], [0.05, 0.32], [0.10, 0.20],
                             np.mean(scenario_02_goal, axis=0)]),
    ))

    scenarios.append(gcr.Scenario(
        name="scenario_03_L_narrow_corridor",
        title="03: Long-horizontal corridor ('-----|')",
        workspace=workspace,
        obstacles=gcr.ObstacleSet(polygons=[
            gcr.rectangle(0.00, 0.4025, 1.00, 0.60),
            gcr.rectangle(0.00, 0.00, 0.65, 0.25),
            gcr.rectangle(0.80, 0.00, 1.00, 0.2475),
            gcr.rectangle(0.80, 0.2525, 1.00, 0.3975),
        ]),
        start=straight_at((0.18, 0.325), 0.0, n_nodes=n_nodes),
        goal=straight_at((0.725, 0.165), np.pi / 2.0, n_nodes=n_nodes),
        epsilon=0.22,
        seed=303,
        guide_points=_guide([[0.18, 0.325], [0.32, 0.325], [0.48, 0.325],
                             [0.64, 0.325], [0.725, 0.30], [0.725, 0.22], [0.725, 0.165]]),
    ))

    scenarios.append(gcr.Scenario(
        name="scenario_04_narrow_gate_mirror_L",
        title="04: 0.15 narrow gate with opposite-concavity arcs",
        workspace=workspace,
        obstacles=gcr.ObstacleSet(polygons=[
            gcr.rectangle(0.35, 0.00, 0.65, 0.225),
            gcr.rectangle(0.35, 0.375, 0.65, 0.60),
        ]),
        start=arc_at((0.10, 0.30), -np.pi / 2.0, np.pi / 2.0, n_nodes=n_nodes),
        goal=arc_at((0.90, 0.30), np.pi / 2.0, 3.0 * np.pi / 2.0, n_nodes=n_nodes),
        epsilon=0.22,
        seed=404,
        guide_points=_guide([[0.15, 0.30], [0.25, 0.30], [0.40, 0.30], [0.50, 0.30],
                             [0.60, 0.30], [0.75, 0.30], [0.85, 0.30]]),
    ))

    scenarios.append(gcr.Scenario(
        name="scenario_05_arc_to_straight",
        title="05: Arc -> distant straight through zigzag",
        workspace=workspace,
        obstacles=gcr.ObstacleSet(circles=[
            ((0.28, 0.42), 0.035), ((0.36, 0.18), 0.035),
            ((0.44, 0.42), 0.035), ((0.52, 0.18), 0.035),
            ((0.60, 0.42), 0.035), ((0.68, 0.18), 0.035),
        ]),
        start=arc_at((0.18, 0.30), -np.pi / 2.0, np.pi / 2.0, n_nodes=n_nodes),
        goal=straight_at((0.85, 0.30), 0.0, n_nodes=n_nodes),
        epsilon=0.22,
        seed=505,
        guide_points=_guide([[0.20, 0.30], [0.32, 0.28], [0.40, 0.30], [0.48, 0.28],
                             [0.56, 0.30], [0.64, 0.28], [0.72, 0.30], [0.80, 0.30]]),
    ))

    scenarios.append(gcr.Scenario(
        name="scenario_06_cross_corridor",
        title="06: Cross corridor with TR-corner goal arc",
        workspace=workspace,
        obstacles=gcr.ObstacleSet(polygons=[
            gcr.rectangle(0.00, 0.00, 0.38, 0.23),
            gcr.rectangle(0.52, 0.00, 1.00, 0.23),
            gcr.rectangle(0.00, 0.37, 0.38, 0.60),
            gcr.rectangle(0.52, 0.37, 1.00, 0.60),
        ]),
        start=straight_at((0.18, 0.30), 0.0, n_nodes=n_nodes),
        goal=arc_at((0.46, 0.30), 0.0, np.pi, n_nodes=n_nodes),
        epsilon=0.22,
        seed=606,
        guide_points=_guide([[0.18, 0.30], [0.30, 0.30], [0.40, 0.30], [0.45, 0.35], [0.46, 0.32]]),
    ))

    scenarios.append(gcr.Scenario(
        name="scenario_07_L_corridor_hard",
        title="07: L-corridor with two bend obstacles",
        workspace=workspace,
        obstacles=gcr.ObstacleSet(
            polygons=[
                gcr.rectangle(0.00, 0.40, 1.00, 0.60),
                gcr.rectangle(0.00, 0.00, 0.40, 0.25),
                gcr.rectangle(0.55, 0.00, 1.00, 0.25),
                gcr.rectangle(0.55, 0.25, 1.00, 0.40),
            ],
            circles=[((0.43, 0.36), 0.022), ((0.43, 0.28), 0.022)],
        ),
        start=straight_at((0.18, 0.325), 0.0, n_nodes=n_nodes),
        goal=straight_at((0.475, 0.195), np.pi / 2.0, n_nodes=n_nodes),
        epsilon=0.22,
        seed=707,
        guide_points=_guide([[0.18, 0.325], [0.30, 0.325], [0.40, 0.32],
                             [0.475, 0.30], [0.475, 0.24], [0.475, 0.195]]),
    ))

    dlo_workspace, dlo_polygons, dlo_circles, dlo_start, dlo_goal = get_dlo_problem_setup()
    dlo_n = len(dlo_start)
    dlo_length = float(np.sum(gcr.segment_lengths(dlo_start)))

    def dlo_line(center: Sequence[float], angle: float) -> np.ndarray:
        c = np.asarray(center, dtype=float)
        d = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        return gcr.line_nodes(c - 0.5 * dlo_length * d, c + 0.5 * dlo_length * d, dlo_n)

    dlo_guides = [
        dlo_start,
        dlo_line((0.30, 0.50), 0.0),
        dlo_line((0.20, 0.50), 0.0),
        dlo_line((0.135, 0.45), -0.5 * np.pi),
        dlo_line((0.135, 0.36), -0.5 * np.pi),
        dlo_line((0.135, 0.20), -0.5 * np.pi),
        dlo_line((0.30, 0.07), 0.0),
        dlo_line((0.52, 0.07), 0.0),
        dlo_goal,
    ]

    scenarios.append(gcr.Scenario(
        name="scenario_08_dlo_workspace",
        title="08: DLO workspace (generate_dlo_workspace_figure)",
        workspace=dlo_workspace,
        obstacles=gcr.ObstacleSet(polygons=[np.asarray(p, dtype=float) for p in dlo_polygons], circles=dlo_circles),
        start=dlo_start,
        goal=dlo_goal,
        epsilon=0.30,
        seed=808,
        guide_points=_guide([np.mean(dlo_start, axis=0), [0.15, 0.47], [0.12, 0.12],
                             [0.47, 0.07], np.mean(dlo_goal, axis=0)]),
        guide_configs=dlo_guides,
    ))

    scenarios.append(gcr.Scenario(
        name="scenario_09_S_corridor",
        title="09: S-corridor, arcs with different curvatures",
        workspace=workspace,
        obstacles=gcr.ObstacleSet(polygons=[
            gcr.rectangle(0.00, 0.00, 0.30, 0.22),
            gcr.rectangle(0.30, 0.30, 0.65, 0.60),
            gcr.rectangle(0.65, 0.00, 1.00, 0.22),
        ]),
        start=arc_at((0.15, 0.45), np.pi, 0.0, n_nodes=n_nodes),
        goal=arc_at((0.80, 0.45), -np.pi / 4.0, np.pi / 4.0, n_nodes=n_nodes),
        epsilon=0.22,
        seed=909,
        guide_points=_guide([[0.15, 0.52], [0.30, 0.26], [0.50, 0.18], [0.70, 0.26], [0.85, 0.40]]),
    ))

    scenario_10_workspace = (0.0, 1.0, -0.05, 0.6)
    scenario_10_goal = arc_at((0.94, 0.30), 5.0 * np.pi / 6.0, 7.0 * np.pi / 6.0, n_nodes=n_nodes)
    scenario_10_goal = scenario_10_goal + np.array([0.10, -0.20], dtype=float)
    scenarios.append(gcr.Scenario(
        name="scenario_10_cluttered_bent",
        title="10: Cluttered field, arcs with different curvatures",
        workspace=scenario_10_workspace,
        obstacles=gcr.ObstacleSet(circles=[
            ((0.25, 0.20), 0.035), ((0.30, 0.45), 0.035),
            ((0.45, 0.30), 0.035), ((0.50, 0.10), 0.035),
            ((0.55, 0.50), 0.035), ((0.65, 0.22), 0.035),
            ((0.75, 0.45), 0.035), ((0.85, 0.18), 0.035),
        ]),
        start=arc_at((0.10, 0.30), -np.pi / 3.0, np.pi / 3.0, n_nodes=n_nodes),
        goal=scenario_10_goal,
        epsilon=0.22,
        seed=1010,
        guide_points=_guide([[0.15, 0.30], [0.30, 0.30], [0.40, 0.40], [0.55, 0.30],
                             [0.70, 0.25], [0.82, 0.16], np.mean(scenario_10_goal, axis=0)]),
    ))

    return scenarios


def build_current_selected_scenarios(n_nodes: int = DEFAULT_N_NODES) -> List[gcr.Scenario]:
    """Return the selected current-project scenarios."""
    current = {scenario.name: scenario for scenario in gcr.build_scenarios(n_nodes)}
    return [
        current["scenario_2_narrow_gate"],
        current["scenario_6_u_shaped_trap"],
        current["scenario_8_two_shelves"],
        current["scenario_4_staggered_blocks"],
        current["scenario_9_dlo_workspace"],
    ]


def build_final_scenarios(n_nodes: int = DEFAULT_N_NODES) -> List[gcr.Scenario]:
    return build_gcr_last_benchmark_scenarios(n_nodes) + build_current_selected_scenarios(n_nodes)


def scenario_source_labels() -> dict[str, str]:
    labels = {scenario.name: "gcr_last_benchmark" for scenario in build_gcr_last_benchmark_scenarios()}
    labels.update({
        "scenario_2_narrow_gate": "current_code_selected",
        "scenario_6_u_shaped_trap": "current_code_selected",
        "scenario_8_two_shelves": "current_code_selected",
        "scenario_4_staggered_blocks": "current_code_selected",
        "scenario_9_dlo_workspace": "current_code_selected",
    })
    return labels
