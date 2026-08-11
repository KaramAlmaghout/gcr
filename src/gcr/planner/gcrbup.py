"""
Full Global Cable Routing (GCR) implementation and scenario runner.

The implementation follows the formulation in Sections 4 and 5 of the paper:

* C = (p_1, ..., p_N) is a 2D discretized cable configuration.
* d(C_i, C_j) is the maximum point-wise node displacement.
* U(C) = U_att(C, C_target) + U_rep(C, O) is used only as a sampling bias.
* The APF-biased sample is locally relaxed with segment-vector, bending, and
  obstacle-clearance terms.
* A node is inserted only after static collision checking and swept-corridor
  transition validation.
* Two trees are grown bidirectionally and shortcut refinement is applied after
  connection.

Outputs are written under gcr_full_outputs/ by default:

* one intermediate-stage plot per scenario;
* one final-path/corridor plot per scenario;
* a metrics dashboard plus metrics.json and metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

plt = None
Line2D = None
Circle = None
Polygon = None
Rectangle = None

from gcr.planner.dlo_workspace_figure import get_problem_setup as get_dlo_problem_setup


EPS = 1e-9
Array = np.ndarray


def ensure_matplotlib() -> None:
    global plt, Line2D, Circle, Polygon, Rectangle
    if plt is not None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.lines import Line2D as MatplotlibLine2D
    from matplotlib.patches import Circle as MatplotlibCircle
    from matplotlib.patches import Polygon as MatplotlibPolygon
    from matplotlib.patches import Rectangle as MatplotlibRectangle

    plt = pyplot
    Line2D = MatplotlibLine2D
    Circle = MatplotlibCircle
    Polygon = MatplotlibPolygon
    Rectangle = MatplotlibRectangle


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------


def rectangle(xmin: float, ymin: float, xmax: float, ymax: float) -> Array:
    return np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        dtype=float,
    )


def line_nodes(start: Sequence[float], end: Sequence[float], n_nodes: int) -> Array:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    t = np.linspace(0.0, 1.0, n_nodes)[:, None]
    return (1.0 - t) * start + t * end


def polyline_nodes(points: Sequence[Sequence[float]], n_nodes: int) -> Array:
    pts = np.asarray(points, dtype=float)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumulative[-1])
    if total < EPS:
        return np.repeat(pts[:1], n_nodes, axis=0)

    targets = np.linspace(0.0, total, n_nodes)
    out: List[Array] = []
    j = 0
    for s in targets:
        while j < len(seg_len) - 1 and cumulative[j + 1] < s:
            j += 1
        span = cumulative[j + 1] - cumulative[j]
        alpha = 0.0 if span < EPS else (s - cumulative[j]) / span
        out.append((1.0 - alpha) * pts[j] + alpha * pts[j + 1])
    return np.asarray(out, dtype=float)


def arc_nodes(
    center: Sequence[float],
    theta_start: float,
    theta_end: float,
    n_nodes: int,
    chord: float = 0.030,
) -> Array:
    dtheta = (theta_end - theta_start) / (n_nodes - 1)
    radius = chord / (2.0 * np.sin(abs(dtheta) / 2.0))
    thetas = np.linspace(theta_start, theta_end, n_nodes)
    center_arr = np.asarray(center, dtype=float)
    return np.column_stack(
        [
            center_arr[0] + radius * np.cos(thetas),
            center_arr[1] + radius * np.sin(thetas),
        ]
    )


def segment_lengths(cable: Array) -> Array:
    diff = np.diff(cable, axis=0)
    return np.sqrt(np.sum(diff * diff, axis=1))


def nominal_segment_length(cable: Array) -> float:
    return float(np.mean(segment_lengths(cable)))


def configuration_distance(config_a: Array, config_b: Array) -> float:
    diff = config_a - config_b
    return float(np.sqrt(np.max(np.sum(diff * diff, axis=1))))


def deformation_energy(new_config: Array, parent_config: Array) -> float:
    seg_new = np.diff(new_config, axis=0)
    seg_parent = np.diff(parent_config, axis=0)
    e_seg = float(np.sum((seg_new - seg_parent) ** 2))

    bend_new = new_config[2:] - 2.0 * new_config[1:-1] + new_config[:-2]
    bend_parent = parent_config[2:] - 2.0 * parent_config[1:-1] + parent_config[:-2]
    e_bend = float(np.sum((bend_new - bend_parent) ** 2))
    return e_seg + e_bend


def point_to_segment_distance(point: Array, start: Array, end: Array) -> float:
    sx = float(end[0] - start[0])
    sy = float(end[1] - start[1])
    length_sq = sx * sx + sy * sy
    if length_sq < EPS:
        return math.hypot(float(point[0] - start[0]), float(point[1] - start[1]))
    t = ((float(point[0] - start[0]) * sx) + (float(point[1] - start[1]) * sy)) / length_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    px = float(start[0]) + t * sx
    py = float(start[1]) + t * sy
    return math.hypot(float(point[0]) - px, float(point[1]) - py)


def closest_point_on_segment(point: Array, start: Array, end: Array) -> Array:
    segment = end - start
    length_sq = float(segment[0] * segment[0] + segment[1] * segment[1])
    if length_sq < EPS:
        return start.copy()
    t = float(((point[0] - start[0]) * segment[0] + (point[1] - start[1]) * segment[1]) / length_sq)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return start + t * segment


def orientation(a: Array, b: Array, c: Array) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) < EPS:
        return 0
    return 1 if value > 0.0 else -1


def on_segment(a: Array, b: Array, c: Array) -> bool:
    return (
        min(a[0], c[0]) - EPS <= b[0] <= max(a[0], c[0]) + EPS
        and min(a[1], c[1]) - EPS <= b[1] <= max(a[1], c[1]) + EPS
        and orientation(a, b, c) == 0
    )


def segments_intersect(a: Array, b: Array, c: Array, d: Array) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def segment_to_segment_distance(a: Array, b: Array, c: Array, d: Array) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_to_segment_distance(a, c, d),
        point_to_segment_distance(b, c, d),
        point_to_segment_distance(c, a, b),
        point_to_segment_distance(d, a, b),
    )


def point_in_polygon(point: Array, polygon: Array) -> bool:
    x, y = point
    if (
        x < float(np.min(polygon[:, 0])) - EPS
        or x > float(np.max(polygon[:, 0])) + EPS
        or y < float(np.min(polygon[:, 1])) - EPS
        or y > float(np.max(polygon[:, 1])) + EPS
    ):
        return False
    return _point_in_polygon_core(point, polygon)


def _point_in_polygon_core(point: Array, polygon: Array) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        pi = polygon[i]
        pj = polygon[j]
        if on_segment(pj, point, pi):
            return True
        yi_gt = pi[1] > y
        yj_gt = pj[1] > y
        if yi_gt != yj_gt:
            x_cross = (pj[0] - pi[0]) * (y - pi[1]) / (pj[1] - pi[1] + EPS) + pi[0]
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def closest_point_on_polygon(point: Array, polygon: Array) -> Tuple[Array, float]:
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    segments = ends - starts
    length_sq = np.einsum("ij,ij->i", segments, segments)
    return _closest_point_on_polygon_edges(point, starts, ends, length_sq)


def _closest_point_on_polygon_edges(point: Array, starts: Array, ends: Array, length_sq: Array) -> Tuple[Array, float]:
    segments = ends - starts
    relative = point[None, :] - starts
    t = np.zeros(len(starts), dtype=float)
    valid = length_sq >= EPS
    if np.any(valid):
        t[valid] = np.clip(np.einsum("ij,ij->i", relative[valid], segments[valid]) / length_sq[valid], 0.0, 1.0)
    projections = starts + t[:, None] * segments
    delta = point[None, :] - projections
    distances_sq = np.einsum("ij,ij->i", delta, delta)
    index = int(np.argmin(distances_sq))
    return projections[index].copy(), math.sqrt(float(distances_sq[index]))


def segment_intersects_polygon(start: Array, end: Array, polygon: Array) -> bool:
    if point_in_polygon(start, polygon) or point_in_polygon(end, polygon):
        return True
    for i in range(len(polygon)):
        if segments_intersect(start, end, polygon[i], polygon[(i + 1) % len(polygon)]):
            return True
    return False


def polygon_intersects_polygon(poly_a: Array, poly_b: Array) -> bool:
    for i in range(len(poly_a)):
        a0 = poly_a[i]
        a1 = poly_a[(i + 1) % len(poly_a)]
        for j in range(len(poly_b)):
            b0 = poly_b[j]
            b1 = poly_b[(j + 1) % len(poly_b)]
            if segments_intersect(a0, a1, b0, b1):
                return True
    return any(point_in_polygon(p, poly_b) for p in poly_a) or any(point_in_polygon(p, poly_a) for p in poly_b)


def polygon_intersects_circle(poly: Array, center: Array, radius: float) -> bool:
    if point_in_polygon(center, poly):
        return True
    if any(np.linalg.norm(p - center) <= radius + EPS for p in poly):
        return True
    for i in range(len(poly)):
        if point_to_segment_distance(center, poly[i], poly[(i + 1) % len(poly)]) <= radius + EPS:
            return True
    return False


def segment_intersects_circle(start: Array, end: Array, center: Array, radius: float) -> bool:
    return point_to_segment_distance(center, start, end) <= radius + EPS


def segment_clearance_to_polygon(start: Array, end: Array, polygon: Array) -> float:
    if segment_intersects_polygon(start, end, polygon):
        return 0.0
    return min(
        segment_to_segment_distance(start, end, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


def segment_clearance_to_obstacles(start: Array, end: Array, obstacles: ObstacleSet) -> float:
    best = np.inf
    for polygon in obstacles.polygons:
        best = min(best, segment_clearance_to_polygon(start, end, polygon))
    for center, radius in zip(obstacles._circle_centers, obstacles._circle_radii):
        best = min(best, point_to_segment_distance(center, start, end) - float(radius))
    return float(best)


def segment_clearance_free(start: Array, end: Array, obstacles: ObstacleSet, clearance: float) -> bool:
    if clearance <= EPS:
        return segment_collision_free(start, end, obstacles)
    return segment_clearance_to_obstacles(start, end, obstacles) >= clearance - EPS


def segment_sample_points(cable: Array, samples_per_segment: int) -> Iterable[Tuple[int, float, Array]]:
    for k in range(len(cable) - 1):
        for q in range(samples_per_segment + 1):
            rho = q / max(samples_per_segment, 1)
            yield k, rho, (1.0 - rho) * cable[k] + rho * cable[k + 1]


def clip_to_workspace(cable: Array, workspace: Tuple[float, float, float, float], margin: float = 0.010) -> Array:
    xmin, xmax, ymin, ymax = workspace
    out = cable.copy()
    out[:, 0] = np.clip(out[:, 0], xmin + margin, xmax - margin)
    out[:, 1] = np.clip(out[:, 1], ymin + margin, ymax - margin)
    return out


def project_uniform_segment_lengths(cable: Array, desired_length: float, n_sweeps: int = 10) -> Array:
    result = np.asarray(cable, dtype=float).copy()
    for _ in range(n_sweeps):
        for direction in (range(len(result) - 1), range(len(result) - 2, -1, -1)):
            for k in direction:
                dx = float(result[k + 1, 0] - result[k, 0])
                dy = float(result[k + 1, 1] - result[k, 1])
                length = math.hypot(dx, dy)
                if length < EPS:
                    continue
                correction_scale = 0.5 * (length - desired_length) / length
                correction_x = correction_scale * dx
                correction_y = correction_scale * dy
                result[k, 0] += correction_x
                result[k, 1] += correction_y
                result[k + 1, 0] -= correction_x
                result[k + 1, 1] -= correction_y
    return result


def project_segment_vector_deviation(
    cable: Array,
    reference: Array,
    delta_l: float,
    fixed_endpoints: bool,
    n_sweeps: int = 5,
) -> Array:
    result = cable.copy()
    fixed = np.zeros(len(result), dtype=bool)
    if fixed_endpoints:
        fixed[0] = True
        fixed[-1] = True

    ref_seg = np.diff(reference, axis=0)
    for _ in range(n_sweeps):
        for direction in (range(len(result) - 1), range(len(result) - 2, -1, -1)):
            for k in direction:
                seg_x = float(result[k + 1, 0] - result[k, 0])
                seg_y = float(result[k + 1, 1] - result[k, 1])
                q_x = seg_x - float(ref_seg[k, 0])
                q_y = seg_y - float(ref_seg[k, 1])
                q_norm = math.hypot(q_x, q_y)
                if q_norm <= delta_l:
                    continue
                scale = delta_l / q_norm
                desired_x = float(ref_seg[k, 0]) + q_x * scale
                desired_y = float(ref_seg[k, 1]) + q_y * scale
                error_x = seg_x - desired_x
                error_y = seg_y - desired_y
                if fixed[k] and fixed[k + 1]:
                    continue
                if fixed[k]:
                    result[k + 1, 0] -= error_x
                    result[k + 1, 1] -= error_y
                elif fixed[k + 1]:
                    result[k, 0] += error_x
                    result[k, 1] += error_y
                else:
                    result[k, 0] += 0.5 * error_x
                    result[k, 1] += 0.5 * error_y
                    result[k + 1, 0] -= 0.5 * error_x
                    result[k + 1, 1] -= 0.5 * error_y
    return result


# -----------------------------------------------------------------------------
# Obstacles, distances, and collision predicates
# -----------------------------------------------------------------------------


@dataclass
class ObstacleSet:
    polygons: List[Array] = field(default_factory=list)
    circles: List[Tuple[Tuple[float, float], float]] = field(default_factory=list)
    _polygon_starts: List[Array] = field(init=False, repr=False, default_factory=list)
    _polygon_ends: List[Array] = field(init=False, repr=False, default_factory=list)
    _polygon_length_sq: List[Array] = field(init=False, repr=False, default_factory=list)
    _polygon_bounds: List[Tuple[float, float, float, float]] = field(init=False, repr=False, default_factory=list)
    _circle_centers: Array = field(init=False, repr=False)
    _circle_radii: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.polygons = [np.asarray(polygon, dtype=float) for polygon in self.polygons]
        self.circles = [
            ((float(center[0]), float(center[1])), float(radius))
            for center, radius in self.circles
        ]
        self._polygon_starts = []
        self._polygon_ends = []
        self._polygon_length_sq = []
        self._polygon_bounds = []
        for polygon in self.polygons:
            ends = np.roll(polygon, -1, axis=0)
            segments = ends - polygon
            self._polygon_starts.append(polygon)
            self._polygon_ends.append(ends)
            self._polygon_length_sq.append(np.einsum("ij,ij->i", segments, segments))
            self._polygon_bounds.append(
                (
                    float(np.min(polygon[:, 0])),
                    float(np.max(polygon[:, 0])),
                    float(np.min(polygon[:, 1])),
                    float(np.max(polygon[:, 1])),
                )
            )
        if self.circles:
            self._circle_centers = np.asarray([center for center, _ in self.circles], dtype=float)
            self._circle_radii = np.asarray([radius for _, radius in self.circles], dtype=float)
        else:
            self._circle_centers = np.empty((0, 2), dtype=float)
            self._circle_radii = np.empty(0, dtype=float)


def closest_obstacle_boundary_point(point: Array, obstacles: ObstacleSet) -> Tuple[Array, float, bool]:
    closest: Optional[Array] = None
    best_distance = np.inf
    best_inside = False

    for idx, polygon in enumerate(obstacles.polygons):
        candidate, distance = _closest_point_on_polygon_edges(
            point,
            obstacles._polygon_starts[idx],
            obstacles._polygon_ends[idx],
            obstacles._polygon_length_sq[idx],
        )
        min_x, max_x, min_y, max_y = obstacles._polygon_bounds[idx]
        inside = (
            min_x - EPS <= point[0] <= max_x + EPS
            and min_y - EPS <= point[1] <= max_y + EPS
            and _point_in_polygon_core(point, polygon)
        )
        if distance < best_distance:
            closest = candidate
            best_distance = distance
            best_inside = inside

    if len(obstacles._circle_radii) > 0:
        deltas = point[None, :] - obstacles._circle_centers
        norms = np.sqrt(np.sum(deltas * deltas, axis=1))
        distances = np.abs(norms - obstacles._circle_radii)
        circle_idx = int(np.argmin(distances))
        distance = float(distances[circle_idx])
        if distance < best_distance:
            center = obstacles._circle_centers[circle_idx]
            radius = float(obstacles._circle_radii[circle_idx])
            norm = float(norms[circle_idx])
            delta = deltas[circle_idx]
            unit = np.array([1.0, 0.0]) if norm < EPS else delta / norm
            closest = center + radius * unit
            best_distance = distance
            best_inside = norm <= radius

    if closest is None:
        raise ValueError("At least one obstacle is required for obstacle queries.")
    return np.asarray(closest, dtype=float), float(best_distance), bool(best_inside)

def outward_direction_from_obstacle(point: Array, obstacles: ObstacleSet) -> Tuple[Array, float, bool, Array]:
    closest, distance, inside = closest_obstacle_boundary_point(point, obstacles)
    direction = closest - point if inside else point - closest
    norm = math.hypot(float(direction[0]), float(direction[1]))
    if norm < EPS:
        direction = np.array([1.0, 0.0])
        norm = 1.0
    return direction / norm, distance, inside, closest


def signed_distance_to_obstacles(point: Array, obstacles: ObstacleSet) -> float:
    _, distance, inside = closest_obstacle_boundary_point(point, obstacles)
    return -distance if inside else distance


def min_configuration_clearance(cable: Array, obstacles: ObstacleSet, samples_per_segment: int) -> float:
    distances = [signed_distance_to_obstacles(sample, obstacles) for _, _, sample in segment_sample_points(cable, samples_per_segment)]
    return float(np.min(distances))


def point_in_any_obstacle(point: Array, obstacles: ObstacleSet) -> bool:
    return signed_distance_to_obstacles(point, obstacles) <= 0.0


def segment_collision_free(start: Array, end: Array, obstacles: ObstacleSet) -> bool:
    for polygon in obstacles.polygons:
        if segment_intersects_polygon(start, end, polygon):
            return False
    for center, radius in zip(obstacles._circle_centers, obstacles._circle_radii):
        if segment_intersects_circle(start, end, center, float(radius)):
            return False
    return True


def cable_collision_free(cable: Array, obstacles: ObstacleSet) -> bool:
    for k in range(len(cable) - 1):
        if not segment_collision_free(cable[k], cable[k + 1], obstacles):
            return False
    return True


def configuration_admissible(
    cable: Array,
    obstacles: ObstacleSet,
    ell_min: float,
    ell_max: float,
    workspace: Tuple[float, float, float, float],
) -> bool:
    xmin, xmax, ymin, ymax = workspace
    if np.any(cable[:, 0] < xmin - EPS) or np.any(cable[:, 0] > xmax + EPS):
        return False
    if np.any(cable[:, 1] < ymin - EPS) or np.any(cable[:, 1] > ymax + EPS):
        return False
    lengths = segment_lengths(cable)
    if np.any(lengths < ell_min - EPS) or np.any(lengths > ell_max + EPS):
        return False
    return cable_collision_free(cable, obstacles)


def connection_polygon(config_a: Array, config_b: Array) -> Array:
    return np.vstack((config_a, config_b[::-1]))


def swept_segment_quads(config_a: Array, config_b: Array) -> Iterable[Array]:
    for k in range(len(config_a) - 1):
        yield np.array([config_a[k], config_a[k + 1], config_b[k + 1], config_b[k]], dtype=float)


def polygon_collision_free(poly: Array, obstacles: ObstacleSet) -> bool:
    for obstacle in obstacles.polygons:
        if polygon_intersects_polygon(poly, obstacle):
            return False
    for center, radius in zip(obstacles._circle_centers, obstacles._circle_radii):
        if polygon_intersects_circle(poly, center, float(radius)):
            return False
    return True


def cable_axis_turn_rad(config_a: Array, config_b: Array) -> float:
    return angle_distance(angle_of(cable_axis(config_a)), angle_of(cable_axis(config_b)))


def cable_axis_turn_ok(config_a: Array, config_b: Array, max_turn_rad: float) -> bool:
    max_turn = float(np.clip(max_turn_rad, 0.0, np.pi))
    if max_turn >= np.pi - EPS:
        return True
    return bool(cable_axis_turn_rad(config_a, config_b) <= max_turn + EPS)


_TRANSITION_TAUS: Dict[int, Array] = {}


def transition_taus(transition_samples: int) -> Array:
    taus = _TRANSITION_TAUS.get(transition_samples)
    if taus is None:
        taus = np.linspace(0.0, 1.0, transition_samples + 2)[1:-1]
        _TRANSITION_TAUS[transition_samples] = taus
    return taus


def path_free(
    config_a: Array,
    config_b: Array,
    obstacles: ObstacleSet,
    transition_samples: int,
    robot_execution_feasibility: bool = False,
    robot_tcp_clearance: float = 0.0,
    robot_obstacle_clearance: Optional[float] = None,
    robot_swept_tcp_clearance: float = 0.0,
    max_cable_axis_turn_rad: float = np.pi,
) -> bool:
    if not cable_axis_turn_ok(config_a, config_b, max_cable_axis_turn_rad):
        return False
    if robot_execution_feasibility:
        tcp_clearance = max(0.0, float(robot_tcp_clearance))
        obstacle_clearance = tcp_clearance if robot_obstacle_clearance is None else max(0.0, float(robot_obstacle_clearance))
        if not segment_clearance_free(config_a[0], config_b[0], obstacles, obstacle_clearance):
            return False
        if not segment_clearance_free(config_a[-1], config_b[-1], obstacles, obstacle_clearance):
            return False
        min_endpoint_separation = 2.0 * tcp_clearance
        if min_endpoint_separation > EPS:
            for tau in np.linspace(0.0, 1.0, transition_samples + 2):
                p0 = (1.0 - tau) * config_a[0] + tau * config_b[0]
                p1 = (1.0 - tau) * config_a[-1] + tau * config_b[-1]
                if np.linalg.norm(p0 - p1) < min_endpoint_separation - EPS:
                    return False
        swept_min_separation = 2.0 * max(0.0, float(robot_swept_tcp_clearance))
        if swept_min_separation > EPS:
            if segment_to_segment_distance(config_a[0], config_b[0], config_a[-1], config_b[-1]) < swept_min_separation - EPS:
                return False
    if not polygon_collision_free(connection_polygon(config_a, config_b), obstacles):
        return False
    for quad in swept_segment_quads(config_a, config_b):
        if not polygon_collision_free(quad, obstacles):
            return False

    for tau in transition_taus(transition_samples):
        interp = (1.0 - tau) * config_a + tau * config_b
        if not cable_collision_free(interp, obstacles):
            return False
    return True


def path_sequence_free(
    path: Sequence[Array],
    obstacles: ObstacleSet,
    transition_samples: int,
    robot_execution_feasibility: bool = False,
    robot_tcp_clearance: float = 0.0,
    robot_obstacle_clearance: Optional[float] = None,
    robot_swept_tcp_clearance: float = 0.0,
    max_cable_axis_turn_rad: float = np.pi,
) -> bool:
    if len(path) <= 1:
        return False
    return all(
        path_free(
            path[k],
            path[k + 1],
            obstacles,
            transition_samples,
            robot_execution_feasibility,
            robot_tcp_clearance,
            robot_obstacle_clearance,
            robot_swept_tcp_clearance,
            max_cable_axis_turn_rad,
        )
        for k in range(len(path) - 1)
    )


def endpoint_correspondence_overlaps(config_a: Array, config_b: Array) -> bool:
    """Return True when the two endpoint correspondence segments intersect."""
    if len(config_a) < 2 or len(config_b) < 2:
        return False
    return segments_intersect(config_a[0], config_b[0], config_a[-1], config_b[-1])


def endpoint_motion_cost(config_a: Array, config_b: Array) -> float:
    return float(np.linalg.norm(config_a[0] - config_b[0]) + np.linalg.norm(config_a[-1] - config_b[-1]))


# MuJoCo workspace mapping places the UR10 bases at x=-0.75 and x=0.75 in
# 3-D while planner x is shifted by +0.5.  These planner-frame base points let
# endpoint reordering prefer robot-friendly endpoint assignments without
# changing the geometric cable planner.
ROBOT_LEFT_BASE_2D = np.array([-0.25, 0.0], dtype=float)
ROBOT_RIGHT_BASE_2D = np.array([1.25, 0.0], dtype=float)
ROBOT_BASE_ENDPOINT_REORDER_MARGIN = 0.02
ROBOT_BASE_TCP_PATH_CLEARANCE = 0.055
ROBOT_BASE_ARM_CLEARANCE = 0.070
ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT = 0.0
ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT = 8
ROBOT_TCP_VECTOR_SECTOR_HALF_WIDTH = math.pi / 2.0
ROBOT_TCP_VECTOR_SECTOR_MAX_ROTATION = math.radians(10.0)


def endpoint0_is_left_robot(config: Array) -> bool:
    natural = float(np.linalg.norm(config[0] - ROBOT_LEFT_BASE_2D) + np.linalg.norm(config[-1] - ROBOT_RIGHT_BASE_2D))
    swapped = float(np.linalg.norm(config[-1] - ROBOT_LEFT_BASE_2D) + np.linalg.norm(config[0] - ROBOT_RIGHT_BASE_2D))
    return natural <= swapped


def robot_assigned_endpoints(config: Array, endpoint0_left: bool) -> Tuple[Array, Array]:
    if endpoint0_left:
        return config[0], config[-1]
    return config[-1], config[0]


def unit_vector(vec: Array, fallback: Array) -> Array:
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return np.asarray(fallback, dtype=float)
    return np.asarray(vec, dtype=float) / norm


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def angle_of(vec: Array) -> float:
    return float(np.arctan2(vec[1], vec[0]))


def angle_distance(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def robot_base_endpoint_assignment_score(previous: Array, candidate: Array, endpoint0_left: bool) -> float:
    """Soft execution score for assigning ordered cable endpoints to robots.

    Lower is better.  This is intentionally a preference, not a validity test:
    obstacle and cable-sweep validity stay in ``path_free``.  The score favors
    endpoint assignments whose TCP motions do not cross, whose base-to-TCP arm
    rays keep clearance, and whose left-to-right endpoint direction stays
    compatible with the robot-base layout.
    """
    left_prev, right_prev = robot_assigned_endpoints(previous, endpoint0_left)
    left_next, right_next = robot_assigned_endpoints(candidate, endpoint0_left)

    travel = float(np.linalg.norm(left_next - left_prev) + np.linalg.norm(right_next - right_prev))
    reach = float(np.linalg.norm(left_next - ROBOT_LEFT_BASE_2D) + np.linalg.norm(right_next - ROBOT_RIGHT_BASE_2D))
    tcp_path_clearance = segment_to_segment_distance(left_prev, left_next, right_prev, right_next)
    arm_clearance = segment_to_segment_distance(ROBOT_LEFT_BASE_2D, left_next, ROBOT_RIGHT_BASE_2D, right_next)

    base_axis = unit_vector(ROBOT_RIGHT_BASE_2D - ROBOT_LEFT_BASE_2D, np.array([1.0, 0.0]))
    prev_axis = unit_vector(right_prev - left_prev, base_axis)
    next_axis = unit_vector(right_next - left_next, prev_axis)
    left_reach_axis = unit_vector(left_next - ROBOT_LEFT_BASE_2D, base_axis)
    right_reach_axis = unit_vector(right_next - ROBOT_RIGHT_BASE_2D, -base_axis)

    path_cross_penalty = 1.0 if segments_intersect(left_prev, left_next, right_prev, right_next) else 0.0
    arm_cross_penalty = 1.0 if segments_intersect(ROBOT_LEFT_BASE_2D, left_next, ROBOT_RIGHT_BASE_2D, right_next) else 0.0
    tcp_clearance_penalty = max(0.0, ROBOT_BASE_TCP_PATH_CLEARANCE - tcp_path_clearance)
    arm_clearance_penalty = max(0.0, ROBOT_BASE_ARM_CLEARANCE - arm_clearance)
    direction_flip_penalty = max(0.0, 1.0 - float(np.dot(prev_axis, next_axis)))
    base_direction_penalty = max(0.0, -float(np.dot(next_axis, base_axis)))
    reach_direction_penalty = (
        max(0.0, -float(np.dot(left_reach_axis, base_axis)))
        + max(0.0, -float(np.dot(right_reach_axis, -base_axis)))
    )

    return (
        travel
        + 0.08 * reach
        + 1.20 * path_cross_penalty
        + 0.90 * arm_cross_penalty
        + 8.0 * tcp_clearance_penalty
        + 6.0 * arm_clearance_penalty
        + 0.20 * direction_flip_penalty
        + 0.25 * base_direction_penalty
        + 0.15 * reach_direction_penalty
    )


def robot_base_reversal_is_preferred(
    previous: Array,
    candidate: Array,
    endpoint0_left: bool,
    margin: float = ROBOT_BASE_ENDPOINT_REORDER_MARGIN,
) -> bool:
    ordered_score = robot_base_endpoint_assignment_score(previous, candidate, endpoint0_left)
    reversed_score = robot_base_endpoint_assignment_score(previous, candidate[::-1], endpoint0_left)
    return bool(reversed_score + max(0.0, float(margin)) < ordered_score)


def robot_base_best_endpoint_assignment_score(previous: Array, candidate: Array) -> float:
    return min(
        robot_base_endpoint_assignment_score(previous, candidate, True),
        robot_base_endpoint_assignment_score(previous, candidate, False),
    )


def robot_base_reversal_is_preferred_any_assignment(
    previous: Array,
    candidate: Array,
    margin: float = ROBOT_BASE_ENDPOINT_REORDER_MARGIN,
) -> bool:
    ordered_score = robot_base_best_endpoint_assignment_score(previous, candidate)
    reversed_score = robot_base_best_endpoint_assignment_score(previous, candidate[::-1])
    return bool(reversed_score + max(0.0, float(margin)) < ordered_score)


def robot_base_vector_frontness(reach_angle: float) -> float:
    front_distance = angle_distance(reach_angle, np.pi / 2.0)
    return float(np.clip((np.pi / 2.0 - front_distance) / (np.pi / 2.0), 0.0, 1.0))


def robot_base_vector_sector_center(reach_angle: float, is_right_robot: bool) -> float:
    max_sector_rotation = math.radians(25.0)
    rotation = max_sector_rotation * robot_base_vector_frontness(reach_angle)
    # When the right robot reaches nearly straight forward, rotate its preferred
    # endpoint-vector sector clockwise.  The left robot uses the mirrored rule.
    return wrap_angle(reach_angle - rotation if is_right_robot else reach_angle + rotation)


def robot_base_vector_sector_excess(tcp_direction: float, reach_angle: float, is_right_robot: bool) -> float:
    sector_center = robot_base_vector_sector_center(reach_angle, is_right_robot)
    return max(0.0, angle_distance(tcp_direction, sector_center) - np.pi / 2.0)


def robot_base_vector_sector_score(previous: Array, candidate: Array, endpoint0_left: bool) -> float:
    left_prev, right_prev = robot_assigned_endpoints(previous, endpoint0_left)
    left_next, right_next = robot_assigned_endpoints(candidate, endpoint0_left)
    previous_tcp_angle = angle_of(right_prev - left_prev)
    next_tcp_angle = angle_of(right_next - left_next)
    tcp_turn = angle_distance(next_tcp_angle, previous_tcp_angle)

    left_reach_angle = angle_of(left_next - ROBOT_LEFT_BASE_2D)
    right_reach_angle = angle_of(right_next - ROBOT_RIGHT_BASE_2D)
    left_sector_excess = robot_base_vector_sector_excess(next_tcp_angle, left_reach_angle, False)
    right_sector_excess = robot_base_vector_sector_excess(wrap_angle(next_tcp_angle + np.pi), right_reach_angle, True)
    return float(tcp_turn + left_sector_excess + right_sector_excess)


def robot_base_vector_reversal_is_preferred_any_assignment(
    previous: Array,
    candidate: Array,
    max_turn_rad: float,
    margin_rad: float,
) -> bool:
    reversed_candidate = candidate[::-1]
    max_turn = max(0.0, float(max_turn_rad))
    margin = max(0.0, float(margin_rad))
    for endpoint0_left in (True, False):
        ordered_score = robot_base_vector_sector_score(previous, candidate, endpoint0_left)
        reversed_score = robot_base_vector_sector_score(previous, reversed_candidate, endpoint0_left)
        ordered_ok = ordered_score <= max_turn + EPS
        reversed_ok = reversed_score <= max_turn + EPS
        if ((not ordered_ok) and reversed_ok) or (reversed_score + margin < ordered_score):
            return True
    return False


def robot_tcp_vector_sector_center(left_tcp: Array, right_tcp: Array) -> float:
    left_reach_angle = angle_of(left_tcp - ROBOT_LEFT_BASE_2D)
    right_reach_angle = angle_of(right_tcp - ROBOT_RIGHT_BASE_2D)
    left_rotation = ROBOT_TCP_VECTOR_SECTOR_MAX_ROTATION * robot_base_vector_frontness(left_reach_angle)
    right_rotation = ROBOT_TCP_VECTOR_SECTOR_MAX_ROTATION * robot_base_vector_frontness(right_reach_angle)
    return wrap_angle(math.pi - right_rotation + left_rotation)


def robot_tcp_vector_sector_endpoint_score(left_tcp: Array, right_tcp: Array) -> float:
    tcp_direction_right_to_left = angle_of(left_tcp - right_tcp)
    sector_center = robot_tcp_vector_sector_center(left_tcp, right_tcp)
    return max(0.0, angle_distance(tcp_direction_right_to_left, sector_center) - ROBOT_TCP_VECTOR_SECTOR_HALF_WIDTH)


def robot_tcp_vector_sector_score(candidate: Array, endpoint0_left: bool) -> float:
    left_tcp, right_tcp = robot_assigned_endpoints(candidate, endpoint0_left)
    return robot_tcp_vector_sector_endpoint_score(left_tcp, right_tcp)


def robot_tcp_vector_sector_config_ok(candidate: Array, endpoint0_left: bool, margin_rad: float) -> bool:
    return bool(robot_tcp_vector_sector_score(candidate, endpoint0_left) <= max(0.0, float(margin_rad)) + EPS)


def robot_tcp_vector_sector_transition_ok(
    previous: Array,
    candidate: Array,
    endpoint0_left: bool,
    margin_rad: float,
    transition_samples: int,
) -> bool:
    left_prev, right_prev = robot_assigned_endpoints(previous, endpoint0_left)
    left_next, right_next = robot_assigned_endpoints(candidate, endpoint0_left)
    samples = max(1, int(transition_samples))
    margin = max(0.0, float(margin_rad))
    for sample_index in range(samples + 1):
        tau = sample_index / samples
        left_tcp = (1.0 - tau) * left_prev + tau * left_next
        right_tcp = (1.0 - tau) * right_prev + tau * right_next
        if robot_tcp_vector_sector_endpoint_score(left_tcp, right_tcp) > margin + EPS:
            return False
    return True


def robot_tcp_vector_sector_path_sequence_ok(
    path: Sequence[Array],
    endpoint0_left: bool,
    margin_rad: float,
    transition_samples: int,
) -> bool:
    if len(path) < 2:
        return True
    if not robot_tcp_vector_sector_config_ok(path[0], endpoint0_left, margin_rad):
        return False
    return all(
        robot_tcp_vector_sector_transition_ok(path[k], path[k + 1], endpoint0_left, margin_rad, transition_samples)
        for k in range(len(path) - 1)
    )


def robot_tcp_vector_sector_path_score(
    path: Sequence[Array],
    start_index: int,
    end_index: int,
    endpoint0_left: bool,
) -> float:
    return sum(robot_tcp_vector_sector_score(path[k], endpoint0_left) for k in range(start_index + 1, end_index + 1))


def robot_tcp_vector_sector_reversal_is_preferred(
    previous: Array,
    candidate: Array,
    margin_rad: float,
    endpoint0_left: bool,
) -> bool:
    del previous
    ordered_score = robot_tcp_vector_sector_score(candidate, endpoint0_left)
    reversed_score = robot_tcp_vector_sector_score(candidate[::-1], endpoint0_left)
    margin = max(0.0, float(margin_rad))
    ordered_ok = ordered_score <= margin + EPS
    reversed_ok = reversed_score <= margin + EPS
    return bool((not ordered_ok and reversed_ok) or (reversed_score + margin < ordered_score))


def robot_base_vector_path_score(path: Sequence[Array], start_index: int, end_index: int, endpoint0_left: bool) -> float:
    score = 0.0
    for k in range(start_index, end_index):
        score += robot_base_vector_sector_score(path[k], path[k + 1], endpoint0_left)
    return float(score)


def robot_base_path_assignment_score(
    path: Sequence[Array],
    start_index: int,
    end_index: int,
    endpoint0_left: bool,
) -> float:
    score = 0.0
    for k in range(start_index, end_index):
        score += robot_base_endpoint_assignment_score(path[k], path[k + 1], endpoint0_left)
    return float(score)


def robot_base_transition_min_arm_clearance(
    previous: Array,
    candidate: Array,
    endpoint0_left: bool,
    transition_samples: int = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT,
) -> float:
    left_prev, right_prev = robot_assigned_endpoints(previous, endpoint0_left)
    left_next, right_next = robot_assigned_endpoints(candidate, endpoint0_left)
    min_clearance = float("inf")
    samples = max(1, int(transition_samples))
    for tau in np.linspace(0.0, 1.0, samples + 1):
        left_tcp = (1.0 - tau) * left_prev + tau * left_next
        right_tcp = (1.0 - tau) * right_prev + tau * right_next
        clearance = segment_to_segment_distance(
            ROBOT_LEFT_BASE_2D,
            left_tcp,
            ROBOT_RIGHT_BASE_2D,
            right_tcp,
        )
        min_clearance = min(min_clearance, clearance)
    return float(min_clearance)


def robot_base_transition_clearance_ok(
    previous: Array,
    candidate: Array,
    endpoint0_left: bool,
    arm_clearance: float = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
    transition_samples: int = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT,
) -> bool:
    return bool(
        robot_base_transition_min_arm_clearance(previous, candidate, endpoint0_left, transition_samples)
        >= max(0.0, float(arm_clearance)) - EPS
    )


def robot_base_transition_clearance_ok_either_assignment(
    previous: Array,
    candidate: Array,
    arm_clearance: float = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
    transition_samples: int = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT,
) -> bool:
    return bool(
        robot_base_transition_clearance_ok(previous, candidate, True, arm_clearance, transition_samples)
        or robot_base_transition_clearance_ok(previous, candidate, False, arm_clearance, transition_samples)
    )


def robot_base_path_sequence_clearance_ok(
    path: Sequence[Array],
    endpoint0_left: bool,
    arm_clearance: float = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
    transition_samples: int = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT,
) -> bool:
    if len(path) < 2:
        return True
    return all(
        robot_base_transition_clearance_ok(
            path[k],
            path[k + 1],
            endpoint0_left,
            arm_clearance,
            transition_samples,
        )
        for k in range(len(path) - 1)
    )


def robot_base_path_sequence_clearance_ok_either_assignment(
    path: Sequence[Array],
    arm_clearance: float = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
    transition_samples: int = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT,
) -> bool:
    if len(path) < 2:
        return True
    return bool(
        robot_base_path_sequence_clearance_ok(path, True, arm_clearance, transition_samples)
        or robot_base_path_sequence_clearance_ok(path, False, arm_clearance, transition_samples)
    )


def cable_axis(config: Array) -> Array:
    axis = config[-1] - config[0]
    norm = float(np.linalg.norm(axis))
    if norm < EPS:
        return np.array([1.0, 0.0])
    return axis / norm


def endpoint_tangent_reversal_is_preferred(
    previous: Array,
    candidate: Array,
    tangent_margin: float,
) -> bool:
    """Prefer reversal when it preserves the cable end-to-end tangent direction.

    This catches cases where endpoint-distance alone would rotate the ordered
    cable by roughly 180 degrees even though reversing the profile gives the
    same physical cable shape with a smoother robot execution.
    """
    if tangent_margin < 0.0:
        return False
    order_dot = float(np.dot(cable_axis(previous), cable_axis(candidate)))
    reverse_dot = -order_dot
    return bool(reverse_dot > order_dot + tangent_margin)


def endpoint_reversal_is_cheaply_preferred(
    previous: Array,
    candidate: Array,
    endpoint_reorder_cost_ratio: float,
    endpoint_reorder_tangent_margin: float = 0.05,
    robot_base_aware_endpoint_reorder: bool = False,
    robot_base_vector_endpoint_reorder: bool = False,
    robot_tcp_vector_sector_reorder: bool = False,
    robot_base_vector_max_turn_rad: float = math.radians(100.0),
    robot_base_vector_margin_rad: float = math.radians(5.0),
    robot_tcp_vector_sector_margin_rad: float = math.radians(5.0),
    endpoint0_left: Optional[bool] = None,
    max_cable_axis_turn_rad: float = math.pi,
) -> bool:
    """Return True when cheap checks prefer the reversed endpoint order.

    The cable geometry is unchanged by reversing the node order. This test only
    scores which physical robot/end effector is assigned to which cable end.
    Expensive swept-collision validation is handled by the caller.
    """
    if len(previous) < 2 or len(candidate) < 2:
        return False
    reversed_candidate = candidate[::-1]
    ordered_axis_ok = cable_axis_turn_ok(previous, candidate, max_cable_axis_turn_rad)
    reversed_axis_ok = cable_axis_turn_ok(previous, reversed_candidate, max_cable_axis_turn_rad)
    if (not ordered_axis_ok) and reversed_axis_ok:
        return True
    if ordered_axis_ok and (not reversed_axis_ok):
        return False

    ordered_overlaps = endpoint_correspondence_overlaps(previous, candidate)
    reversed_overlaps = endpoint_correspondence_overlaps(previous, reversed_candidate)
    if ordered_overlaps and not reversed_overlaps:
        return True
    if (not ordered_overlaps) and reversed_overlaps:
        return False

    if endpoint_tangent_reversal_is_preferred(previous, candidate, endpoint_reorder_tangent_margin):
        return True
    if robot_base_aware_endpoint_reorder:
        endpoint0_left = endpoint0_is_left_robot(previous) if endpoint0_left is None else endpoint0_left
        if robot_base_reversal_is_preferred(previous, candidate, endpoint0_left) or robot_base_reversal_is_preferred_any_assignment(
            previous,
            candidate,
        ):
            return True
    if robot_base_vector_endpoint_reorder and robot_base_vector_reversal_is_preferred_any_assignment(
        previous,
        candidate,
        robot_base_vector_max_turn_rad,
        robot_base_vector_margin_rad,
    ):
        return True
    if robot_tcp_vector_sector_reorder:
        endpoint0_left = endpoint0_is_left_robot(previous) if endpoint0_left is None else endpoint0_left
        if robot_tcp_vector_sector_reversal_is_preferred(
            previous,
            candidate,
            robot_tcp_vector_sector_margin_rad,
            endpoint0_left,
        ):
            return True
    old_cost = endpoint_motion_cost(previous, candidate)
    reverse_cost = endpoint_motion_cost(previous, reversed_candidate)
    return bool(reverse_cost + EPS < endpoint_reorder_cost_ratio * old_cost)


def endpoint_reversal_is_preferred(
    previous: Array,
    candidate: Array,
    candidate_valid: bool,
    endpoint_reorder_cost_ratio: float,
    endpoint_reorder_tangent_margin: float = 0.05,
    robot_base_aware_endpoint_reorder: bool = False,
    robot_base_vector_endpoint_reorder: bool = False,
    robot_tcp_vector_sector_reorder: bool = False,
    robot_base_vector_max_turn_rad: float = math.radians(100.0),
    robot_base_vector_margin_rad: float = math.radians(5.0),
    robot_tcp_vector_sector_margin_rad: float = math.radians(5.0),
    endpoint0_left: Optional[bool] = None,
    max_cable_axis_turn_rad: float = math.pi,
) -> bool:
    if not candidate_valid:
        return True
    return endpoint_reversal_is_cheaply_preferred(
        previous,
        candidate,
        endpoint_reorder_cost_ratio,
        endpoint_reorder_tangent_margin,
        robot_base_aware_endpoint_reorder,
        robot_base_vector_endpoint_reorder,
        robot_tcp_vector_sector_reorder,
        robot_base_vector_max_turn_rad,
        robot_base_vector_margin_rad,
        robot_tcp_vector_sector_margin_rad,
        endpoint0_left,
        max_cable_axis_turn_rad,
    )


def reorder_endpoint_correspondence(
    path: Sequence[Array],
    obstacles: ObstacleSet,
    transition_samples: int,
    robot_execution_feasibility: bool = False,
    robot_tcp_clearance: float = 0.0,
    robot_obstacle_clearance: Optional[float] = None,
    robot_swept_tcp_clearance: float = 0.0,
    preserve_terminal_order: bool = True,
    endpoint_reorder_cost_ratio: float = 0.75,
    endpoint_reorder_tangent_margin: float = 0.05,
    robot_base_aware_endpoint_reorder: bool = False,
    robot_base_vector_endpoint_reorder: bool = False,
    robot_tcp_vector_sector_reorder: bool = False,
    robot_tcp_vector_sector_feasibility: bool = False,
    robot_base_vector_max_turn_rad: float = math.radians(100.0),
    robot_base_vector_margin_rad: float = math.radians(5.0),
    robot_tcp_vector_sector_margin_rad: float = math.radians(5.0),
    max_cable_axis_turn_rad: float = math.pi,
) -> Tuple[List[Array], int]:
    """Greedily reverse cable profile order when endpoint continuity improves.

    This is an optional endpoint-labeling heuristic for execution. It keeps the
    geometry of a cable profile unchanged, but swaps node order when the ordered
    transition is invalid, when endpoint correspondence segments cross, or when
    the reversed order is a valid transition with clearly shorter endpoint
    motion.
    """
    if not path:
        return [], 0

    ordered = [path[0].copy()]
    reversals = 0
    for config_index, config in enumerate(path[1:], start=1):
        candidate = config.copy()
        is_terminal = config_index == len(path) - 1
        can_reverse = not (preserve_terminal_order and is_terminal)
        def transition_valid(target: Array) -> bool:
            valid = path_free(
                ordered[-1],
                target,
                obstacles,
                transition_samples,
                robot_execution_feasibility,
                robot_tcp_clearance,
                robot_obstacle_clearance,
                robot_swept_tcp_clearance,
                max_cable_axis_turn_rad,
            )
            if robot_base_aware_endpoint_reorder:
                valid = valid and robot_base_transition_clearance_ok_either_assignment(
                    ordered[-1],
                    target,
                )
            if robot_tcp_vector_sector_feasibility:
                valid = valid and robot_tcp_vector_sector_transition_ok(
                    ordered[-1],
                    target,
                    endpoint0_is_left_robot(ordered[0]),
                    robot_tcp_vector_sector_margin_rad,
                    transition_samples,
                )
            return bool(valid)

        if can_reverse:
            reversed_candidate = candidate[::-1].copy()
            prefer_reversed = endpoint_reversal_is_cheaply_preferred(
                ordered[-1],
                candidate,
                endpoint_reorder_cost_ratio,
                endpoint_reorder_tangent_margin,
                robot_base_aware_endpoint_reorder,
                robot_base_vector_endpoint_reorder,
                robot_tcp_vector_sector_reorder,
                robot_base_vector_max_turn_rad,
                robot_base_vector_margin_rad,
                robot_tcp_vector_sector_margin_rad,
                endpoint0_is_left_robot(ordered[0]),
                max_cable_axis_turn_rad,
            )
            if prefer_reversed:
                if transition_valid(reversed_candidate):
                    candidate = reversed_candidate
                    reversals += 1
            elif (not transition_valid(candidate)) and transition_valid(reversed_candidate):
                candidate = reversed_candidate
                reversals += 1
        ordered.append(candidate)
    return ordered, reversals


# -----------------------------------------------------------------------------
# APF biasing and elastic relaxation
# -----------------------------------------------------------------------------


@dataclass
class PlannerParams:
    max_iterations: int = 1600
    scenario_time_limit_s: Optional[float] = 120.0
    apf_steps: int = 8
    eta: float = 0.035
    k_att: float = 0.22
    beta: float = 0.0055
    sigma: float = 0.060
    segment_samples: int = 3
    transition_samples: int = 6
    epsilon: float = 0.115
    ell_min_ratio: float = 0.65
    ell_max_ratio: float = 1.45
    dsafe: float = 0.014
    delta_l: float = 0.075
    relax_iterations: int = 24
    relax_step: float = 0.060
    lambda_s: float = 0.34
    lambda_b: float = 0.075
    tracking_weight: float = 1.0
    alpha_def: float = 0.10
    max_apf_node_step: float = 0.006
    max_parent_checks: int = 28
    shortcut_refinement: bool = True
    endpoint_reorder: bool = False
    robot_execution_feasibility: bool = False
    robot_tcp_clearance: float = 0.0
    robot_obstacle_clearance: Optional[float] = None
    robot_swept_tcp_clearance: float = 0.0
    preserve_terminal_order: bool = True
    endpoint_reorder_cost_ratio: float = 0.75
    endpoint_reorder_tangent_margin: float = 0.05
    robot_base_aware_endpoint_reorder: bool = False
    robot_base_vector_endpoint_reorder: bool = False
    robot_tcp_vector_sector_reorder: bool = False
    robot_tcp_vector_sector_feasibility: bool = False
    robot_base_vector_max_turn_rad: float = math.radians(100.0)
    robot_base_vector_margin_rad: float = math.radians(5.0)
    robot_tcp_vector_sector_margin_rad: float = math.radians(5.0)
    max_cable_axis_turn_rad: float = math.pi
    robot_base_arm_clearance: float = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT
    robot_base_transition_samples: int = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT
    shortcut_passes: int = 3
    shortcut_max_distance_factor: Optional[float] = None
    shortcut_validation_samples: int = 100
    goal_bias: float = 0.06
    tree_bias: float = 0.84
    broad_bias: float = 0.10
    exact_clearance_period: int = 4


@dataclass
class RelaxationPlane:
    k: int
    rho: float
    normal: Array
    boundary_point: Array


def potential_values(cable: Array, target: Array, obstacles: ObstacleSet, params: PlannerParams) -> Dict[str, float]:
    diff = cable - target
    u_att = 0.5 * params.k_att * float(np.sum(diff * diff))
    u_rep = 0.0
    for _, _, sample in segment_sample_points(cable, params.segment_samples):
        signed_d = signed_distance_to_obstacles(sample, obstacles)
        r = max(signed_d, 0.0)
        u_rep += params.beta * np.exp(-(r * r) / (2.0 * params.sigma * params.sigma))
    return {"att": u_att, "rep": u_rep, "total": u_att + u_rep}


def configuration_forces(cable: Array, target: Array, obstacles: ObstacleSet, params: PlannerParams) -> Dict[str, Array]:
    f_att = params.k_att * (target - cable)
    f_rep = np.zeros_like(cable)
    sigma = max(params.sigma, EPS)
    sigma_sq = max(params.sigma * params.sigma, EPS)
    rep_inside_strength = params.beta / sigma

    for k, rho, sample in segment_sample_points(cable, params.segment_samples):
        direction, boundary_distance, inside, _ = outward_direction_from_obstacle(sample, obstacles)
        if inside:
            strength = rep_inside_strength
        else:
            exponent = np.exp(-(boundary_distance * boundary_distance) / (2.0 * params.sigma * params.sigma))
            strength = params.beta * boundary_distance / sigma_sq * exponent
        sample_force = strength * direction
        f_rep[k] += (1.0 - rho) * sample_force
        f_rep[k + 1] += rho * sample_force

    return {"att": f_att, "rep": f_rep, "total": f_att + f_rep}


def apf_descent(
    raw: Array,
    target: Array,
    obstacles: ObstacleSet,
    workspace: Tuple[float, float, float, float],
    ell0: float,
    params: PlannerParams,
    capture_sequence: bool = True,
) -> List[Array]:
    sequence = [raw.copy()] if capture_sequence else []
    current = raw.copy()
    for _ in range(params.apf_steps):
        forces = configuration_forces(current, target, obstacles, params)["total"]
        displacement = params.eta * forces
        max_step = float(np.max(np.linalg.norm(displacement, axis=1)))
        if max_step > params.max_apf_node_step:
            displacement *= params.max_apf_node_step / max_step
        current = clip_to_workspace(current + displacement, workspace)
        current = project_uniform_segment_lengths(current, ell0, n_sweeps=2)
        current = clip_to_workspace(current, workspace)
        if capture_sequence:
            sequence.append(current.copy())
    return sequence if capture_sequence else [current]


def make_relaxation_planes(biased: Array, obstacles: ObstacleSet, params: PlannerParams) -> List[RelaxationPlane]:
    planes: List[RelaxationPlane] = []
    for k, rho, sample in segment_sample_points(biased, params.segment_samples):
        normal, _, _, boundary = outward_direction_from_obstacle(sample, obstacles)
        planes.append(RelaxationPlane(k=k, rho=rho, normal=normal, boundary_point=boundary))
    return planes


def relaxation_gradient(cable: Array, biased: Array, reference: Array, params: PlannerParams) -> Array:
    grad = params.tracking_weight * (cable - biased)

    ref_seg = np.diff(reference, axis=0)
    seg_q = np.diff(cable, axis=0) - ref_seg
    grad[:-1] -= 2.0 * params.lambda_s * seg_q
    grad[1:] += 2.0 * params.lambda_s * seg_q

    ref_bend = reference[2:] - 2.0 * reference[1:-1] + reference[:-2]
    bend_q = (cable[2:] - 2.0 * cable[1:-1] + cable[:-2]) - ref_bend
    grad[:-2] += 2.0 * params.lambda_b * bend_q
    grad[1:-1] -= 4.0 * params.lambda_b * bend_q
    grad[2:] += 2.0 * params.lambda_b * bend_q

    return grad


def project_linearized_obstacle_constraints(
    cable: Array,
    planes: Sequence[RelaxationPlane],
    dsafe: float,
    fixed_endpoints: bool,
) -> Array:
    result = cable.copy()
    fixed = np.zeros(len(result), dtype=bool)
    if fixed_endpoints:
        fixed[0] = True
        fixed[-1] = True

    for plane in planes:
        k = plane.k
        rho = plane.rho
        w0 = 1.0 - rho
        w1 = rho
        sample = w0 * result[k] + w1 * result[k + 1]
        gap = float(np.dot(plane.normal, sample - plane.boundary_point))
        violation = dsafe - gap
        if violation <= 0.0:
            continue

        fw0 = 0.0 if fixed[k] else w0
        fw1 = 0.0 if fixed[k + 1] else w1
        denom = fw0 * fw0 + fw1 * fw1
        if denom < EPS:
            continue
        if fw0 > 0.0:
            result[k] += (fw0 / denom) * violation * plane.normal
        if fw1 > 0.0:
            result[k + 1] += (fw1 / denom) * violation * plane.normal
    return result


def push_sample_clearance(cable: Array, obstacles: ObstacleSet, params: PlannerParams, fixed_endpoints: bool) -> Array:
    result = cable.copy()
    fixed = np.zeros(len(result), dtype=bool)
    if fixed_endpoints:
        fixed[0] = True
        fixed[-1] = True
    for k, rho, sample in segment_sample_points(result, params.segment_samples):
        direction, distance, inside, _ = outward_direction_from_obstacle(sample, obstacles)
        signed_distance = -distance if inside else distance
        violation = params.dsafe - signed_distance
        if violation <= 0.0:
            continue
        w0 = 1.0 - rho
        w1 = rho
        fw0 = 0.0 if fixed[k] else w0
        fw1 = 0.0 if fixed[k + 1] else w1
        denom = fw0 * fw0 + fw1 * fw1
        if denom < EPS:
            continue
        if fw0 > 0.0:
            result[k] += (fw0 / denom) * violation * direction
        if fw1 > 0.0:
            result[k + 1] += (fw1 / denom) * violation * direction
    return result


def relax_configuration(
    biased: Array,
    reference: Array,
    obstacles: ObstacleSet,
    workspace: Tuple[float, float, float, float],
    ell0: float,
    params: PlannerParams,
    capture_sequence: bool = True,
) -> List[Array]:
    planes = make_relaxation_planes(biased, obstacles, params)
    result = biased.copy()
    sequence = [result.copy()] if capture_sequence else []
    capture = (
        {0, params.relax_iterations // 3, 2 * params.relax_iterations // 3, params.relax_iterations - 1}
        if capture_sequence
        else set()
    )

    for iteration in range(params.relax_iterations):
        grad = relaxation_gradient(result, biased, reference, params)
        grad[0] = 0.0
        grad[-1] = 0.0
        max_grad = float(np.max(np.linalg.norm(grad, axis=1)))
        if max_grad > 2.0:
            grad *= 2.0 / max_grad

        result = result - params.relax_step * grad
        result[0] = biased[0]
        result[-1] = biased[-1]
        result = project_linearized_obstacle_constraints(result, planes, params.dsafe, fixed_endpoints=True)
        result[0] = biased[0]
        result[-1] = biased[-1]
        if iteration % params.exact_clearance_period == 0 or iteration == params.relax_iterations - 1:
            result = push_sample_clearance(result, obstacles, params, fixed_endpoints=True)
            result[0] = biased[0]
            result[-1] = biased[-1]
        result = project_segment_vector_deviation(result, reference, params.delta_l, fixed_endpoints=True, n_sweeps=3)
        result[0] = biased[0]
        result[-1] = biased[-1]
        result = clip_to_workspace(result, workspace)
        result[0] = clip_to_workspace(result[:1], workspace)[0]
        result[-1] = clip_to_workspace(result[-1:], workspace)[0]

        if iteration in capture:
            sequence.append(result.copy())

    if capture_sequence and not np.allclose(sequence[-1], result):
        sequence.append(result.copy())
    return sequence if capture_sequence else [result]


# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    title: str
    workspace: Tuple[float, float, float, float]
    obstacles: ObstacleSet
    start: Array
    goal: Array
    epsilon: float
    seed: int
    guide_points: Optional[Array] = None
    guide_configs: Optional[List[Array]] = None


@dataclass
class TreeNode:
    config: Array
    parent: Optional[int]
    cost: float


@dataclass
class SearchTree:
    name: str
    nodes: List[TreeNode]
    _configs_cache: Optional[Array] = field(default=None, init=False, repr=False)

    def configs(self) -> Array:
        if self._configs_cache is None or len(self._configs_cache) != len(self.nodes):
            self._configs_cache = np.stack([node.config for node in self.nodes], axis=0)
        return self._configs_cache

    def add(self, config: Array, parent: int, cost: float) -> int:
        self.nodes.append(TreeNode(config=config.copy(), parent=parent, cost=cost))
        self._configs_cache = None
        return len(self.nodes) - 1

    def trace_root_to(self, index: int) -> List[Array]:
        path = []
        current: Optional[int] = index
        while current is not None:
            node = self.nodes[current]
            path.append(node.config)
            current = node.parent
        return list(reversed(path))

    def trace_to_root(self, index: int) -> List[Array]:
        path = []
        current: Optional[int] = index
        while current is not None:
            node = self.nodes[current]
            path.append(node.config)
            current = node.parent
        return path


@dataclass
class ExpansionRecord:
    iteration: int
    tree_name: str
    raw: Array
    apf_sequence: List[Array]
    relaxation_sequence: List[Array]
    parent: Optional[Array]
    relaxed: Optional[Array]
    forces_raw: Dict[str, Array]
    accepted: bool
    reject_reason: str


@dataclass
class PlanResult:
    scenario: Scenario
    success: bool
    raw_path: List[Array]
    path: List[Array]
    iterations: int
    elapsed_s: float
    timed_out: bool
    tree_start: SearchTree
    tree_goal: SearchTree
    debug_record: Optional[ExpansionRecord]
    reject_counts: Dict[str, int]
    endpoint_reversals: int
    metrics: Dict[str, float | int | str | bool]


class GCRPlanner:
    def __init__(self, scenario: Scenario, params: PlannerParams):
        self.scenario = scenario
        self.params = replace(params, epsilon=scenario.epsilon)
        self.rng = np.random.default_rng(scenario.seed)
        self.ell0 = nominal_segment_length(scenario.start)
        self.ell_min = self.params.ell_min_ratio * self.ell0
        self.ell_max = self.params.ell_max_ratio * self.ell0
        self.reject_counts: Dict[str, int] = {
            "static": 0,
            "no_parent": 0,
            "no_connect": 0,
            "relax": 0,
        }
        self.endpoint_reversals = 0
        self.debug_record: Optional[ExpansionRecord] = None

    def plan(self) -> PlanResult:
        start_time = time.perf_counter()
        ts = SearchTree("start", [TreeNode(self.scenario.start.copy(), None, 0.0)])
        tg = SearchTree("goal", [TreeNode(self.scenario.goal.copy(), None, 0.0)])

        success = False
        timed_out = False
        raw_path: List[Array] = []
        refined_path: List[Array] = []
        self.endpoint_reversals = 0
        final_iteration = self.params.max_iterations
        time_limit = self.params.scenario_time_limit_s

        for iteration in range(1, self.params.max_iterations + 1):
            if time_limit is not None and time.perf_counter() - start_time >= time_limit:
                timed_out = True
                final_iteration = iteration - 1
                break

            if iteration % 2 == 1:
                active, opposite = ts, tg
                target = self.scenario.goal
                reference = self.scenario.start
            else:
                active, opposite = tg, ts
                target = self.scenario.start
                reference = self.scenario.goal

            expanded, new_index, record = self.expand_tree(iteration, active, target, reference)
            if self.debug_record is None and record is not None:
                self.debug_record = record
            if not expanded or new_index is None:
                continue

            new_config = active.nodes[new_index].config
            conn_index, conn_distance = self.nearest_index(opposite, new_config)
            if conn_distance <= self.params.epsilon and path_free(
                new_config,
                opposite.nodes[conn_index].config,
                self.scenario.obstacles,
                self.params.transition_samples,
                self.params.robot_execution_feasibility,
                self.params.robot_tcp_clearance,
                self.params.robot_obstacle_clearance,
                self.params.robot_swept_tcp_clearance,
                self.params.max_cable_axis_turn_rad,
            ):
                if active.name == "start":
                    meet_s_index = new_index
                    meet_g_index = conn_index
                else:
                    meet_s_index = conn_index
                    meet_g_index = new_index

                start_path = ts.trace_root_to(meet_s_index)
                goal_path = tg.trace_to_root(meet_g_index)
                raw_path = start_path + goal_path
                original_raw_path = [config.copy() for config in raw_path]
                if self.params.endpoint_reorder:
                    raw_path, self.endpoint_reversals = reorder_endpoint_correspondence(
                        raw_path,
                        self.scenario.obstacles,
                        self.params.transition_samples,
                        self.params.robot_execution_feasibility,
                        self.params.robot_tcp_clearance,
                        self.params.robot_obstacle_clearance,
                        self.params.robot_swept_tcp_clearance,
                        self.params.preserve_terminal_order,
                        self.params.endpoint_reorder_cost_ratio,
                        self.params.endpoint_reorder_tangent_margin,
                        self.params.robot_base_aware_endpoint_reorder,
                        self.params.robot_base_vector_endpoint_reorder,
                        self.params.robot_tcp_vector_sector_reorder,
                        self.params.robot_tcp_vector_sector_feasibility,
                        self.params.robot_base_vector_max_turn_rad,
                        self.params.robot_base_vector_margin_rad,
                        self.params.robot_tcp_vector_sector_margin_rad,
                        self.params.max_cable_axis_turn_rad,
                    )
                    raw_robot_base_ok = (
                        (not self.params.robot_base_aware_endpoint_reorder)
                        or robot_base_path_sequence_clearance_ok_either_assignment(
                            raw_path,
                            self.params.robot_base_arm_clearance,
                            self.params.robot_base_transition_samples,
                        )
                    )
                    raw_tcp_sector_ok = (
                        (not self.params.robot_tcp_vector_sector_feasibility)
                        or robot_tcp_vector_sector_path_sequence_ok(
                            raw_path,
                            endpoint0_is_left_robot(raw_path[0]),
                            self.params.robot_tcp_vector_sector_margin_rad,
                            self.params.transition_samples,
                        )
                    )
                    if (not raw_robot_base_ok) or (not raw_tcp_sector_ok) or not path_sequence_free(
                        raw_path,
                        self.scenario.obstacles,
                        self.params.transition_samples,
                        self.params.robot_execution_feasibility,
                        self.params.robot_tcp_clearance,
                        self.params.robot_obstacle_clearance,
                        self.params.robot_swept_tcp_clearance,
                        self.params.max_cable_axis_turn_rad,
                    ):
                        raw_path = original_raw_path
                        self.endpoint_reversals = 0
                if self.params.robot_base_aware_endpoint_reorder and not robot_base_path_sequence_clearance_ok_either_assignment(
                    raw_path,
                    self.params.robot_base_arm_clearance,
                    self.params.robot_base_transition_samples,
                ):
                    self.reject_counts["no_connect"] += 1
                    continue
                refined_path = self.refine_path(raw_path)
                validation_samples = max(self.params.transition_samples, self.params.shortcut_validation_samples)
                refined_robot_base_ok = (
                    (not self.params.robot_base_aware_endpoint_reorder)
                    or robot_base_path_sequence_clearance_ok_either_assignment(
                        refined_path,
                        self.params.robot_base_arm_clearance,
                        self.params.robot_base_transition_samples,
                    )
                )
                refined_tcp_sector_ok = (
                    (not self.params.robot_tcp_vector_sector_feasibility)
                    or robot_tcp_vector_sector_path_sequence_ok(
                        refined_path,
                        endpoint0_is_left_robot(refined_path[0]),
                        self.params.robot_tcp_vector_sector_margin_rad,
                        validation_samples,
                    )
                )
                if (not refined_robot_base_ok) or (not refined_tcp_sector_ok) or not path_sequence_free(
                    refined_path,
                    self.scenario.obstacles,
                    validation_samples,
                    self.params.robot_execution_feasibility,
                    self.params.robot_tcp_clearance,
                    self.params.robot_obstacle_clearance,
                    self.params.robot_swept_tcp_clearance,
                    self.params.max_cable_axis_turn_rad,
                ):
                    refined_path = list(raw_path)
                final_robot_base_ok = (
                    (not self.params.robot_base_aware_endpoint_reorder)
                    or robot_base_path_sequence_clearance_ok_either_assignment(
                        refined_path,
                        self.params.robot_base_arm_clearance,
                        self.params.robot_base_transition_samples,
                    )
                )
                final_tcp_sector_ok = (
                    (not self.params.robot_tcp_vector_sector_feasibility)
                    or robot_tcp_vector_sector_path_sequence_ok(
                        refined_path,
                        endpoint0_is_left_robot(refined_path[0]),
                        self.params.robot_tcp_vector_sector_margin_rad,
                        validation_samples,
                    )
                )
                if final_robot_base_ok and final_tcp_sector_ok and path_sequence_free(
                    refined_path,
                    self.scenario.obstacles,
                    validation_samples,
                    self.params.robot_execution_feasibility,
                    self.params.robot_tcp_clearance,
                    self.params.robot_obstacle_clearance,
                    self.params.robot_swept_tcp_clearance,
                    self.params.max_cable_axis_turn_rad,
                ):
                    success = True
                    final_iteration = iteration
                    break
            self.reject_counts["no_connect"] += 1

        elapsed = time.perf_counter() - start_time
        if not success and not timed_out and time_limit is not None and elapsed >= time_limit:
            timed_out = True
        metrics = self.compute_metrics(success, timed_out, raw_path, refined_path, final_iteration, elapsed, ts, tg)
        return PlanResult(
            scenario=self.scenario,
            success=success,
            raw_path=raw_path,
            path=refined_path,
            iterations=final_iteration,
            elapsed_s=elapsed,
            timed_out=timed_out,
            tree_start=ts,
            tree_goal=tg,
            debug_record=self.debug_record,
            reject_counts=dict(self.reject_counts),
            endpoint_reversals=self.endpoint_reversals,
            metrics=metrics,
        )

    def expand_tree(
        self,
        iteration: int,
        active: SearchTree,
        target: Array,
        reference: Array,
    ) -> Tuple[bool, Optional[int], Optional[ExpansionRecord]]:
        capture_record = self.debug_record is None
        raw = self.sample_raw_configuration(active, target)
        apf_sequence = apf_descent(
            raw,
            target,
            self.scenario.obstacles,
            self.scenario.workspace,
            self.ell0,
            self.params,
            capture_sequence=capture_record,
        )
        biased = apf_sequence[-1]
        relaxation_sequence = relax_configuration(
            biased,
            reference,
            self.scenario.obstacles,
            self.scenario.workspace,
            self.ell0,
            self.params,
            capture_sequence=capture_record,
        )
        relaxed = relaxation_sequence[-1]
        forces_raw = configuration_forces(raw, target, self.scenario.obstacles, self.params) if capture_record else {}

        if not configuration_admissible(relaxed, self.scenario.obstacles, self.ell_min, self.ell_max, self.scenario.workspace):
            self.reject_counts["static"] += 1
            return (
                False,
                None,
                ExpansionRecord(iteration, active.name, raw, apf_sequence, relaxation_sequence, None, relaxed, forces_raw, False, "static")
                if capture_record
                else None,
            )

        parent_index = self.select_parent(active, relaxed)
        if parent_index is None:
            self.reject_counts["no_parent"] += 1
            return (
                False,
                None,
                ExpansionRecord(iteration, active.name, raw, apf_sequence, relaxation_sequence, None, relaxed, forces_raw, False, "no_parent")
                if capture_record
                else None,
            )

        parent = active.nodes[parent_index]
        d_parent = configuration_distance(parent.config, relaxed)
        cost = parent.cost + d_parent + self.params.alpha_def * deformation_energy(relaxed, parent.config)
        new_index = active.add(relaxed, parent_index, cost)
        record = (
            ExpansionRecord(
                iteration,
                active.name,
                raw,
                apf_sequence,
                relaxation_sequence,
                parent.config.copy(),
                relaxed.copy(),
                forces_raw,
                True,
                "",
            )
            if capture_record
            else None
        )
        return True, new_index, record

    def select_parent(self, tree: SearchTree, relaxed: Array) -> Optional[int]:
        configs = tree.configs()
        diff = configs - relaxed[None, :, :]
        distances = np.sqrt(np.max(np.sum(diff * diff, axis=2), axis=1))
        candidates = np.where(distances <= self.params.epsilon)[0]
        if len(candidates) == 0:
            return None
        ordered = candidates[np.argsort(distances[candidates])]
        for index in ordered[: self.params.max_parent_checks]:
            if path_free(
                tree.nodes[int(index)].config,
                relaxed,
                self.scenario.obstacles,
                self.params.transition_samples,
                self.params.robot_execution_feasibility,
                self.params.robot_tcp_clearance,
                self.params.robot_obstacle_clearance,
                self.params.robot_swept_tcp_clearance,
                self.params.max_cable_axis_turn_rad,
            ):
                return int(index)
        return None

    def nearest_index(self, tree: SearchTree, config: Array) -> Tuple[int, float]:
        configs = tree.configs()
        diff = configs - config[None, :, :]
        distances = np.sqrt(np.max(np.sum(diff * diff, axis=2), axis=1))
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def sample_raw_configuration(self, active: SearchTree, target: Array) -> Array:
        p = float(self.rng.random())
        if p < self.params.goal_bias:
            base = target.copy()
            noise_scale = 0.018
        elif p < self.params.goal_bias + self.params.tree_bias:
            configs = active.configs()
            diff = configs - target[None, :, :]
            target_dist = np.sqrt(np.max(np.sum(diff * diff, axis=2), axis=1))
            if len(configs) > 4 and self.rng.random() < 0.65:
                pool = np.argsort(target_dist)[: max(3, min(12, len(configs)))]
                parent = active.nodes[int(self.rng.choice(pool))].config
            else:
                parent = active.nodes[int(self.rng.integers(len(active.nodes)))].config
            local_target = self.local_sampling_target(parent, target)
            distance = max(configuration_distance(parent, local_target), EPS)
            alpha = min(0.64, max(0.16, 0.70 * self.params.epsilon / distance))
            base = parent + alpha * (local_target - parent)
            noise_scale = 0.012
        else:
            return self.sample_broad_configuration()

        noise = self.rng.normal(scale=noise_scale, size=base.shape)
        sinusoid = np.sin(np.linspace(0.0, np.pi, len(base)))[:, None]
        normal = cable_normals(base)
        lateral = self.rng.normal(scale=0.024) * sinusoid * normal
        raw = base + noise + lateral
        raw = clip_to_workspace(raw, self.scenario.workspace)
        raw = project_uniform_segment_lengths(raw, self.ell0, n_sweeps=5)
        return clip_to_workspace(raw, self.scenario.workspace)

    def local_sampling_target(self, parent: Array, final_target: Array) -> Array:
        if self.scenario.guide_configs:
            forward = np.allclose(final_target, self.scenario.goal)
            configs = self.scenario.guide_configs if forward else list(reversed(self.scenario.guide_configs))
            distances = np.array([configuration_distance(parent, config) for config in configs], dtype=float)
            index = int(np.argmin(distances))
            next_index = min(index + 1, len(configs) - 1)
            return configs[next_index]

        guide = self.scenario.guide_points
        if guide is None or len(guide) < 2:
            return final_target

        forward = np.allclose(final_target, self.scenario.goal)
        points = guide if forward else guide[::-1]
        center = np.mean(parent, axis=0)
        distances = np.linalg.norm(points - center[None, :], axis=1)
        index = int(np.argmin(distances))
        next_index = min(index + 1, len(points) - 1)
        next_center = points[next_index]
        return parent + (next_center - center)

    def sample_broad_configuration(self) -> Array:
        workspace = self.scenario.workspace
        xmin, xmax, ymin, ymax = workspace
        n_nodes = len(self.scenario.start)
        for _ in range(80):
            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            p0 = np.array(
                [
                    self.rng.uniform(xmin + 0.06, xmax - 0.06),
                    self.rng.uniform(ymin + 0.06, ymax - 0.06),
                ],
                dtype=float,
            )
            pts = [p0]
            angle = theta
            for _k in range(n_nodes - 1):
                angle += float(self.rng.normal(scale=0.36))
                pts.append(pts[-1] + self.ell0 * np.array([np.cos(angle), np.sin(angle)]))
            cable = np.asarray(pts, dtype=float)
            min_xy = cable.min(axis=0)
            max_xy = cable.max(axis=0)
            shift = np.zeros(2)
            if min_xy[0] < xmin + 0.015:
                shift[0] = xmin + 0.015 - min_xy[0]
            if max_xy[0] > xmax - 0.015:
                shift[0] = xmax - 0.015 - max_xy[0]
            if min_xy[1] < ymin + 0.015:
                shift[1] = ymin + 0.015 - min_xy[1]
            if max_xy[1] > ymax - 0.015:
                shift[1] = ymax - 0.015 - max_xy[1]
            cable = clip_to_workspace(cable + shift, workspace)
            if min_configuration_clearance(cable, self.scenario.obstacles, self.params.segment_samples) > -0.03:
                return cable
        return clip_to_workspace(cable, workspace)

    def refine_path(self, path: List[Array]) -> List[Array]:
        if len(path) <= 2:
            return list(path)
        if not self.params.shortcut_refinement:
            return list(path)
        refined = list(path)
        for _ in range(self.params.shortcut_passes):
            improved = False
            i = 0
            while i < len(refined) - 2:
                accepted = False
                for j in range(len(refined) - 1, i + 1, -1):
                    old_cost = sum(configuration_distance(refined[k], refined[k + 1]) for k in range(i, j))
                    if (
                        self.params.endpoint_reorder
                        and self.params.preserve_terminal_order
                        and endpoint_correspondence_overlaps(refined[i], refined[j])
                    ):
                        continue
                    shortcut_candidates: List[Tuple[Array, bool]] = []
                    can_reverse_shortcut = self.params.endpoint_reorder and not self.params.preserve_terminal_order
                    prefer_reversed_shortcut = (
                        can_reverse_shortcut
                        and endpoint_reversal_is_cheaply_preferred(
                            refined[i],
                            refined[j],
                            self.params.endpoint_reorder_cost_ratio,
                            self.params.endpoint_reorder_tangent_margin,
                            self.params.robot_base_aware_endpoint_reorder,
                            self.params.robot_base_vector_endpoint_reorder,
                            self.params.robot_tcp_vector_sector_reorder,
                            self.params.robot_base_vector_max_turn_rad,
                            self.params.robot_base_vector_margin_rad,
                            self.params.robot_tcp_vector_sector_margin_rad,
                            endpoint0_is_left_robot(refined[0]),
                            self.params.max_cable_axis_turn_rad,
                        )
                    )
                    if prefer_reversed_shortcut:
                        shortcut_candidates.append((refined[j][::-1].copy(), True))
                        shortcut_candidates.append((refined[j], False))
                    else:
                        shortcut_candidates.append((refined[j], False))
                        if can_reverse_shortcut:
                            shortcut_candidates.append((refined[j][::-1].copy(), True))
                    for shortcut_target, reverse_suffix in shortcut_candidates:
                        new_cost = configuration_distance(refined[i], shortcut_target)
                        max_shortcut_distance = (
                            np.inf
                            if self.params.shortcut_max_distance_factor is None
                            else self.params.shortcut_max_distance_factor * self.params.epsilon
                        )
                        robot_base_shortcut_ok = True
                        if self.params.robot_base_aware_endpoint_reorder:
                            old_robot_score = min(
                                robot_base_path_assignment_score(refined, i, j, True),
                                robot_base_path_assignment_score(refined, i, j, False),
                            )
                            new_robot_score = robot_base_best_endpoint_assignment_score(
                                refined[i],
                                shortcut_target,
                            )
                            robot_base_shortcut_ok = (
                                new_robot_score <= old_robot_score + ROBOT_BASE_ENDPOINT_REORDER_MARGIN
                                and robot_base_transition_clearance_ok_either_assignment(
                                    refined[i],
                                    shortcut_target,
                                    self.params.robot_base_arm_clearance,
                                    self.params.robot_base_transition_samples,
                                )
                            )
                        if self.params.robot_base_vector_endpoint_reorder:
                            old_vector_score = min(
                                robot_base_vector_path_score(refined, i, j, True),
                                robot_base_vector_path_score(refined, i, j, False),
                            )
                            new_vector_score = min(
                                robot_base_vector_sector_score(refined[i], shortcut_target, True),
                                robot_base_vector_sector_score(refined[i], shortcut_target, False),
                            )
                            robot_base_shortcut_ok = (
                                robot_base_shortcut_ok
                                and new_vector_score
                                <= old_vector_score + self.params.robot_base_vector_margin_rad
                            )
                        if self.params.robot_tcp_vector_sector_reorder:
                            old_sector_score = min(
                                robot_tcp_vector_sector_path_score(refined, i, j, True),
                                robot_tcp_vector_sector_path_score(refined, i, j, False),
                            )
                            new_sector_score = min(
                                robot_tcp_vector_sector_score(shortcut_target, True),
                                robot_tcp_vector_sector_score(shortcut_target, False),
                            )
                            robot_base_shortcut_ok = (
                                robot_base_shortcut_ok
                                and new_sector_score
                                <= old_sector_score + self.params.robot_tcp_vector_sector_margin_rad
                            )
                        if self.params.robot_tcp_vector_sector_feasibility:
                            robot_base_shortcut_ok = (
                                robot_base_shortcut_ok
                                and robot_tcp_vector_sector_transition_ok(
                                    refined[i],
                                    shortcut_target,
                                    endpoint0_is_left_robot(refined[0]),
                                    self.params.robot_tcp_vector_sector_margin_rad,
                                    max(self.params.transition_samples, self.params.shortcut_validation_samples),
                                )
                            )
                        if not (
                            robot_base_shortcut_ok
                            and new_cost <= max_shortcut_distance
                            and new_cost < old_cost
                        ):
                            continue
                        suffix = refined[j:]
                        if reverse_suffix:
                            suffix = [config[::-1].copy() for config in suffix]
                        next_refined = refined[: i + 1] + suffix
                        if (
                            self.params.robot_base_aware_endpoint_reorder
                            and not robot_base_path_sequence_clearance_ok_either_assignment(
                                next_refined,
                                self.params.robot_base_arm_clearance,
                                self.params.robot_base_transition_samples,
                            )
                        ):
                            continue
                        if (
                            self.params.robot_tcp_vector_sector_feasibility
                            and not robot_tcp_vector_sector_path_sequence_ok(
                                next_refined,
                                endpoint0_is_left_robot(next_refined[0]),
                                self.params.robot_tcp_vector_sector_margin_rad,
                                max(self.params.transition_samples, self.params.shortcut_validation_samples),
                            )
                        ):
                            continue
                        if not path_free(
                            refined[i],
                            shortcut_target,
                            self.scenario.obstacles,
                            max(self.params.transition_samples, self.params.shortcut_validation_samples),
                            self.params.robot_execution_feasibility,
                            self.params.robot_tcp_clearance,
                            self.params.robot_obstacle_clearance,
                            self.params.robot_swept_tcp_clearance,
                            self.params.max_cable_axis_turn_rad,
                        ):
                            continue
                        if reverse_suffix:
                            self.endpoint_reversals += 1
                        refined = next_refined
                        improved = True
                        accepted = True
                        break
                    if accepted:
                        break
                if not accepted:
                    i += 1
            if not improved:
                break
        return refined

    def compute_metrics(
        self,
        success: bool,
        timed_out: bool,
        raw_path: List[Array],
        path: List[Array],
        iterations: int,
        elapsed: float,
        ts: SearchTree,
        tg: SearchTree,
    ) -> Dict[str, float | int | str | bool]:
        path_used = path if path else raw_path
        path_cost = sum(configuration_distance(path_used[k], path_used[k + 1]) for k in range(max(0, len(path_used) - 1)))
        raw_cost = sum(configuration_distance(raw_path[k], raw_path[k + 1]) for k in range(max(0, len(raw_path) - 1)))
        def_cost = sum(deformation_energy(path_used[k + 1], path_used[k]) for k in range(max(0, len(path_used) - 1)))
        edge_distances = [configuration_distance(path_used[k], path_used[k + 1]) for k in range(max(0, len(path_used) - 1))]
        max_edge_distance = max(edge_distances) if edge_distances else float("nan")
        min_clearance = (
            min(min_configuration_clearance(c, self.scenario.obstacles, self.params.segment_samples) for c in path_used)
            if path_used
            else float("nan")
        )
        transition_ok = (
            all(
                path_free(
                    path_used[k],
                    path_used[k + 1],
                    self.scenario.obstacles,
                    max(self.params.transition_samples, self.params.shortcut_validation_samples),
                    self.params.robot_execution_feasibility,
                    self.params.robot_tcp_clearance,
                    self.params.robot_obstacle_clearance,
                    self.params.robot_swept_tcp_clearance,
                    self.params.max_cable_axis_turn_rad,
                )
                for k in range(len(path_used) - 1)
            )
            if len(path_used) > 1
            else False
        )
        static_ok = (
            all(configuration_admissible(c, self.scenario.obstacles, self.ell_min, self.ell_max, self.scenario.workspace) for c in path_used)
            if path_used
            else False
        )
        seg_dev = (
            max(float(np.max(np.abs(segment_lengths(c) - self.ell0))) for c in path_used)
            if path_used
            else float("nan")
        )

        return {
            "scenario": self.scenario.name,
            "success": bool(success),
            "timed_out": bool(timed_out),
            "iterations": int(iterations),
            "time_s": float(elapsed),
            "time_limit_s": (
                float(self.params.scenario_time_limit_s)
                if self.params.scenario_time_limit_s is not None
                else float("nan")
            ),
            "start_vertices": len(ts.nodes),
            "goal_vertices": len(tg.nodes),
            "raw_path_configs": len(raw_path),
            "refined_path_configs": len(path),
            "shortcut_refinement_enabled": bool(self.params.shortcut_refinement),
            "endpoint_reorder_enabled": bool(self.params.endpoint_reorder),
            "robot_execution_feasibility_enabled": bool(self.params.robot_execution_feasibility),
            "robot_tcp_clearance_m": float(self.params.robot_tcp_clearance if self.params.robot_execution_feasibility else 0.0),
            "robot_obstacle_clearance_m": float(
                (
                    self.params.robot_tcp_clearance
                    if self.params.robot_obstacle_clearance is None
                    else self.params.robot_obstacle_clearance
                )
                if self.params.robot_execution_feasibility
                else 0.0
            ),
            "robot_swept_tcp_clearance_m": float(
                self.params.robot_swept_tcp_clearance if self.params.robot_execution_feasibility else 0.0
            ),
            "preserve_terminal_order_enabled": bool(self.params.preserve_terminal_order),
            "endpoint_order_reversals": int(self.endpoint_reversals),
            "epsilon_m": float(self.params.epsilon),
            "max_edge_distance_m": float(max_edge_distance),
            "shortcut_validation_samples": int(max(self.params.transition_samples, self.params.shortcut_validation_samples)),
            "path_cost_J": float(path_cost),
            "raw_path_cost_J": float(raw_cost),
            "shortcut_reduction_pct": float(0.0 if raw_cost <= EPS else 100.0 * (raw_cost - path_cost) / raw_cost),
            "deformation_energy": float(def_cost),
            "min_clearance_m": float(min_clearance),
            "max_segment_length_error_m": float(seg_dev),
            "static_collision_free": bool(static_ok),
            "transition_path_free": bool(transition_ok),
            "reject_static": self.reject_counts["static"],
            "reject_no_parent": self.reject_counts["no_parent"],
            "reject_no_connect": self.reject_counts["no_connect"],
        }


def cable_normals(cable: Array) -> Array:
    tangents = np.zeros_like(cable)
    tangents[0] = cable[1] - cable[0]
    tangents[-1] = cable[-1] - cable[-2]
    tangents[1:-1] = cable[2:] - cable[:-2]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > EPS
    normals[valid] /= norms[valid, None]
    normals[~valid] = np.array([0.0, 1.0])
    return normals


# -----------------------------------------------------------------------------
# Scenarios
# -----------------------------------------------------------------------------


def build_scenarios(n_nodes: int) -> List[Scenario]:
    workspace = (0.0, 1.0, 0.0, 0.6)
    start_1 = line_nodes((0.07, 0.12), (0.36, 0.12), n_nodes)
    goal_1 = line_nodes((0.64, 0.48), (0.93, 0.48), n_nodes)

    start_2 = arc_nodes((0.19, 0.12), np.pi, 0.0, n_nodes)
    goal_2 = arc_nodes((0.81, 0.48), 0.0, -np.pi, n_nodes)

    start_3 = line_nodes((0.06, 0.08), (0.28, 0.08), n_nodes)
    goal_3 = line_nodes((0.72, 0.52), (0.94, 0.52), n_nodes)

    scenario_4_segment = 0.24 / (n_nodes - 1)
    start_4 = arc_nodes((0.17, 0.10), np.pi, 2.0 * np.pi, n_nodes, chord=scenario_4_segment)
    goal_4 = line_nodes((0.95, 0.10), (0.95, 0.34), n_nodes)

    start_5 = line_nodes((0.06, 0.30), (0.31, 0.30), n_nodes)
    goal_5 = line_nodes((0.69, 0.30), (0.94, 0.30), n_nodes)

    start_6 = line_nodes((0.07, 0.08), (0.31, 0.08), n_nodes)
    goal_6 = line_nodes((0.53, 0.43), (0.77, 0.43), n_nodes)

    start_7 = line_nodes((0.06, 0.15), (0.28, 0.15), n_nodes)
    goal_7 = line_nodes((0.72, 0.45), (0.94, 0.45), n_nodes)

    start_8 = line_nodes((0.05, 0.53), (0.32, 0.53), n_nodes)
    goal_8 = line_nodes((0.68, 0.07), (0.95, 0.07), n_nodes)

    dlo_workspace, dlo_polygons, dlo_circles, dlo_start, dlo_goal = get_dlo_problem_setup()
    dlo_polygons = [np.asarray(poly, dtype=float).copy() for poly in dlo_polygons]
    dlo_circles = [
        ((float(center[0]), float(center[1])), float(radius))
        for center, radius in dlo_circles
    ]
    # Scenario 9 adjustments while keeping the DLO start/goal unchanged.
    # Move the left circle and left pentagon 10 cm left, and move the middle
    # circle 10 cm right.
    dlo_polygons[0][:, 0] -= 0.10
    (left_circle_x, left_circle_y), left_circle_radius = dlo_circles[0]
    dlo_circles[0] = ((left_circle_x - 0.10, left_circle_y), left_circle_radius)
    (middle_circle_x, middle_circle_y), middle_circle_radius = dlo_circles[1]
    dlo_circles[1] = ((middle_circle_x + 0.10, middle_circle_y), middle_circle_radius)
    dlo_n = len(dlo_start)
    dlo_length = float(np.sum(segment_lengths(dlo_start)))

    def dlo_line_config(center: Sequence[float], angle: float) -> Array:
        center_arr = np.asarray(center, dtype=float)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        return line_nodes(center_arr - 0.5 * dlo_length * direction, center_arr + 0.5 * dlo_length * direction, dlo_n)

    dlo_guides = [
        dlo_start,
        dlo_line_config((0.30, 0.50), 0.0),
        dlo_line_config((0.20, 0.50), 0.0),
        dlo_line_config((0.135, 0.45), -0.5 * np.pi),
        dlo_line_config((0.135, 0.36), -0.5 * np.pi),
        dlo_line_config((0.135, 0.20), -0.5 * np.pi),
        dlo_line_config((0.30, 0.07), 0.0),
        dlo_line_config((0.52, 0.07), 0.0),
        dlo_goal,
    ]

    def guide_points(points: Sequence[Sequence[float]]) -> Array:
        return np.asarray(points, dtype=float)

    scenarios = [
        Scenario(
            name="scenario_1_single_obstacle",
            title="Scenario 1: single rectangular obstacle",
            workspace=workspace,
            obstacles=ObstacleSet(polygons=[rectangle(0.43, 0.20, 0.59, 0.40)]),
            start=start_1,
            goal=goal_1,
            epsilon=0.240,
            seed=11,
            guide_points=guide_points(
                [
                    np.mean(start_1, axis=0),
                    [0.56, 0.08],
                    [0.76, 0.10],
                    [0.76, 0.48],
                    np.mean(goal_1, axis=0),
                ],
            ),
        ),
        Scenario(
            name="scenario_2_narrow_gate",
            title="Scenario 2: narrow gate between two blocks",
            workspace=workspace,
            obstacles=ObstacleSet(
                polygons=[
                    rectangle(0.42, 0.00, 0.58, 0.24),
                    rectangle(0.42, 0.36, 0.58, 0.60),
                ]
            ),
            start=start_2,
            goal=goal_2,
            epsilon=0.205,
            seed=23,
            guide_points=guide_points(
                [
                    np.mean(start_2, axis=0),
                    [0.34, 0.18],
                    [0.50, 0.30],
                    [0.66, 0.42],
                    np.mean(goal_2, axis=0),
                ],
            ),
        ),
        Scenario(
            name="scenario_3_multi_obstacle_maze",
            title="Scenario 3: multi-obstacle maze",
            workspace=workspace,
            obstacles=ObstacleSet(
                polygons=[
                    rectangle(0.22, 0.18, 0.31, 0.44),
                    rectangle(0.56, 0.00, 0.66, 0.28),
                    rectangle(0.72, 0.30, 0.83, 0.47),
                ],
                circles=[((0.53, 0.45), 0.045)],
            ),
            start=start_3,
            goal=goal_3,
            epsilon=0.170,
            seed=37,
            guide_points=guide_points(
                [
                    np.mean(start_3, axis=0),
                    [0.42, 0.08],
                    [0.42, 0.53],
                    [0.64, 0.53],
                    np.mean(goal_3, axis=0),
                ],
            ),
        ),
        Scenario(
            name="scenario_4_staggered_blocks",
            title="Scenario 4: staggered rectangular blocks",
            workspace=workspace,
            obstacles=ObstacleSet(
                polygons=[
                    rectangle(0.24, 0.18, 0.36, 0.40),
                    rectangle(0.48, 0.00, 0.60, 0.24),
                    rectangle(0.74, 0.20, 0.86, 0.58),
                ]
            ),
            start=start_4,
            goal=goal_4,
            epsilon=0.190,
            seed=41,
            guide_points=guide_points(
                [
                    np.mean(start_4, axis=0),
                    [0.28, 0.12],
                    [0.42, 0.28],
                    [0.64, 0.28],
                    [0.90, 0.22],
                    np.mean(goal_4, axis=0),
                ],
            ),
        ),
        Scenario(
            name="scenario_5_circle_slalom",
            title="Scenario 5: circular obstacle slalom",
            workspace=workspace,
            obstacles=ObstacleSet(
                circles=[
                    ((0.42, 0.23), 0.045),
                    ((0.50, 0.37), 0.050),
                    ((0.60, 0.24), 0.045),
                ]
            ),
            start=start_5,
            goal=goal_5,
            epsilon=0.185,
            seed=53,
            guide_points=guide_points(
                [
                    np.mean(start_5, axis=0),
                    [0.39, 0.43],
                    [0.52, 0.17],
                    [0.64, 0.42],
                    np.mean(goal_5, axis=0),
                ],
            ),
        ),
        Scenario(
            name="scenario_6_u_shaped_trap",
            title="Scenario 6: U-shaped obstacle trap",
            workspace=workspace,
            obstacles=ObstacleSet(
                polygons=[
                    rectangle(0.39, 0.28, 0.45, 0.56),
                    rectangle(0.45, 0.50, 0.73, 0.56),
                    rectangle(0.45, 0.28, 0.73, 0.34),
                ]
            ),
            start=goal_6,
            goal=start_6,
            epsilon=0.205,
            seed=61,
            guide_points=guide_points(
                [
                    np.mean(goal_6, axis=0),
                    [0.78, 0.43],
                    [0.78, 0.18],
                    [0.34, 0.08],
                    np.mean(start_6, axis=0),
                ],
            ),
        ),
        Scenario(
            name="scenario_7_diagonal_slalom",
            title="Scenario 7: diagonal slalom through offset blocks",
            workspace=workspace,
            obstacles=ObstacleSet(
                polygons=[
                    rectangle(0.28, 0.30, 0.38, 0.37),
                    rectangle(0.58, 0.20, 0.68, 0.27),
                ]
            ),
            start=start_7,
            goal=goal_7,
            epsilon=0.245,
            seed=71,
            guide_points=guide_points(
                [
                    np.mean(start_7, axis=0),
                    [0.35, 0.17],
                    [0.48, 0.42],
                    [0.66, 0.31],
                    [0.78, 0.45],
                    np.mean(goal_7, axis=0),
                ],
            ),
        ),
        Scenario(
            name="scenario_8_two_shelves",
            title="Scenario 8: two long shelf obstacles",
            workspace=workspace,
            obstacles=ObstacleSet(
                polygons=[
                    rectangle(0.48, 0.39, 0.62, 0.43),
                    rectangle(0.38, 0.17, 0.52, 0.21),
                ]
            ),
            start=start_8,
            goal=goal_8,
            epsilon=0.360,
            seed=83,
            guide_points=guide_points(
                [
                    np.mean(start_8, axis=0),
                    [0.36, 0.50],
                    [0.46, 0.30],
                    [0.66, 0.22],
                    [0.82, 0.07],
                    np.mean(goal_8, axis=0),
                ],
            ),
        ),
        Scenario(
            name="scenario_9_dlo_workspace",
            title="Scenario 9: DLO workspace from generate_dlo_workspace_figure.py",
            workspace=dlo_workspace,
            obstacles=ObstacleSet(polygons=[np.asarray(poly, dtype=float) for poly in dlo_polygons], circles=dlo_circles),
            start=dlo_start,
            goal=dlo_goal,
            epsilon=0.300,
            seed=107,
            guide_points=guide_points(
                [
                    np.mean(dlo_start, axis=0),
                    [0.15, 0.47],
                    [0.12, 0.12],
                    [0.47, 0.07],
                    np.mean(dlo_goal, axis=0),
                ],
            ),
            guide_configs=dlo_guides,
        ),
    ]

    return scenarios


def validate_scenarios(scenarios: Sequence[Scenario], params: PlannerParams) -> None:
    for scenario in scenarios:
        ell0 = nominal_segment_length(scenario.start)
        if len(scenario.start) != len(scenario.goal):
            raise ValueError(f"{scenario.name}: start and goal must have the same node count.")
        if not np.allclose(segment_lengths(scenario.start), ell0, atol=2e-4):
            raise ValueError(f"{scenario.name}: start segment lengths are not uniform.")
        if not np.allclose(segment_lengths(scenario.goal), ell0, atol=2e-4):
            raise ValueError(f"{scenario.name}: goal segment lengths do not match start.")
        ell_min = params.ell_min_ratio * ell0
        ell_max = params.ell_max_ratio * ell0
        if not configuration_admissible(scenario.start, scenario.obstacles, ell_min, ell_max, scenario.workspace):
            raise ValueError(f"{scenario.name}: start configuration is not collision-free.")
        if not configuration_admissible(scenario.goal, scenario.obstacles, ell_min, ell_max, scenario.workspace):
            raise ValueError(f"{scenario.name}: goal configuration is not collision-free.")


# -----------------------------------------------------------------------------
# Plotting and metrics output
# -----------------------------------------------------------------------------


OBSTACLE_LEGEND = None


def obstacle_legend_handle() -> object:
    global OBSTACLE_LEGEND
    ensure_matplotlib()
    if OBSTACLE_LEGEND is None:
        OBSTACLE_LEGEND = Line2D(
            [0],
            [0],
            marker="s",
            color="#18181b",
            markerfacecolor="#3f3f46",
            markersize=9,
            linestyle="None",
            label="obstacle",
        )
    return OBSTACLE_LEGEND


def draw_workspace_scene(ax: plt.Axes, scenario: Scenario) -> None:
    ensure_matplotlib()
    xmin, xmax, ymin, ymax = scenario.workspace
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="#f7fbf9",
            edgecolor="#dc2626",
            linewidth=1.4,
            zorder=0,
        )
    )
    for polygon in scenario.obstacles.polygons:
        ax.add_patch(Polygon(polygon, closed=True, facecolor="#3f3f46", edgecolor="#18181b", linewidth=1.2, zorder=3))
    for center, radius in scenario.obstacles.circles:
        ax.add_patch(Circle(center, radius, facecolor="#3f3f46", edgecolor="#18181b", linewidth=1.2, zorder=3))


def configure_axes(ax: plt.Axes, scenario: Scenario) -> None:
    xmin, xmax, ymin, ymax = scenario.workspace
    ax.set_xlim(xmin - 0.035, xmax + 0.035)
    ax.set_ylim(ymin - 0.035, ymax + 0.035)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.75)


def plot_configuration(
    ax: plt.Axes,
    cable: Array,
    color: str,
    node_color: str,
    label: Optional[str],
    linewidth: float = 2.7,
    node_size: float = 28.0,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 8,
) -> None:
    ax.plot(
        cable[:, 0],
        cable[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    ax.scatter(
        cable[:, 0],
        cable[:, 1],
        s=node_size,
        facecolor=node_color,
        edgecolor=color,
        linewidth=0.75,
        alpha=alpha,
        zorder=zorder + 1,
    )


def draw_start_goal(ax: plt.Axes, scenario: Scenario) -> None:
    plot_configuration(ax, scenario.start, "#2563eb", "#bfdbfe", "start", linewidth=2.6, node_size=25, zorder=9)
    plot_configuration(ax, scenario.goal, "#7c3aed", "#ddd6fe", "goal", linewidth=2.6, node_size=25, zorder=9)


def draw_force_vectors(ax: plt.Axes, cable: Array, forces: Array, color: str, label: str) -> None:
    norms = np.linalg.norm(forces, axis=1)
    if len(norms) == 0 or float(np.max(norms)) < EPS:
        return
    displayed = forces / float(np.max(norms)) * 0.060
    ax.quiver(
        cable[:, 0],
        cable[:, 1],
        displayed[:, 0],
        displayed[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        width=0.0055,
        label=label,
        zorder=14,
    )


def draw_corridor(ax: plt.Axes, config_a: Array, config_b: Array, color: str, alpha: float, label: Optional[str], zorder: int) -> None:
    poly = connection_polygon(config_a, config_b)
    ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=color, linewidth=0.8, alpha=alpha, label=label, zorder=zorder))


def add_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_"):
            unique[label] = handle
    unique.setdefault("obstacle", obstacle_legend_handle())
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="lower left",
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d1d5db",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    ensure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_intermediate_stages(result: PlanResult, output_dir: Path) -> None:
    ensure_matplotlib()
    record = result.debug_record
    if record is None:
        return
    scenario = result.scenario
    biased = record.apf_sequence[-1]
    relaxed = record.relaxed if record.relaxed is not None else record.relaxation_sequence[-1]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    axes = axes.ravel()

    for ax in axes:
        draw_workspace_scene(ax, scenario)
        draw_start_goal(ax, scenario)
        configure_axes(ax, scenario)

    plot_configuration(axes[0], record.raw, "#ef4444", "#fecaca", "raw sample", linewidth=2.5, zorder=10)
    axes[0].set_title("(a) raw configuration sample", fontsize=11.5)

    plot_configuration(axes[1], record.raw, "#ef4444", "#fecaca", "raw sample", linewidth=2.2, zorder=10)
    draw_force_vectors(axes[1], record.raw, record.forces_raw["total"], "#111827", "APF force")
    axes[1].set_title("(b) configuration-level APF force", fontsize=11.5)

    for cable in record.apf_sequence[1:-1:3]:
        axes[2].plot(cable[:, 0], cable[:, 1], color="#64748b", linewidth=1.0, alpha=0.38, zorder=8)
    plot_configuration(axes[2], record.raw, "#ef4444", "#fecaca", "raw sample", linewidth=2.0, alpha=0.55, zorder=9)
    plot_configuration(axes[2], biased, "#0f766e", "#99f6e4", "APF-biased", linewidth=2.8, zorder=12)
    axes[2].set_title("(c) APF-biased candidate", fontsize=11.5)

    if record.parent is not None:
        draw_corridor(axes[3], record.parent, relaxed, "#facc15", 0.22, "swept corridor", zorder=4)
        plot_configuration(axes[3], record.parent, "#475569", "#cbd5e1", "selected parent", linewidth=2.1, node_size=21, zorder=10)
    for cable in record.relaxation_sequence[:-1]:
        axes[3].plot(cable[:, 0], cable[:, 1], color="#94a3b8", linewidth=1.0, alpha=0.35, zorder=8)
    plot_configuration(axes[3], biased, "#0f766e", "#99f6e4", "APF-biased", linewidth=2.1, alpha=0.55, zorder=10)
    plot_configuration(axes[3], relaxed, "#f97316", "#fed7aa", "relaxed/inserted", linewidth=2.9, zorder=13)
    axes[3].set_title("(d) elastic relaxation and accepted corridor", fontsize=11.5)

    for ax in axes:
        add_legend(ax)
    fig.suptitle(scenario.title, fontsize=13.0)
    save_figure(fig, output_dir, f"{scenario.name}_intermediate_stages")


def plot_final_path(result: PlanResult, output_dir: Path) -> None:
    ensure_matplotlib()
    scenario = result.scenario
    path = result.path if result.path else result.raw_path
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    draw_workspace_scene(ax, scenario)

    if len(path) > 1:
        for k in range(len(path) - 1):
            label = "swept path corridor" if k == 0 else None
            draw_corridor(ax, path[k], path[k + 1], "#facc15", 0.14, label, zorder=2)

    if path:
        cmap = plt.get_cmap("viridis")
        for k, cable in enumerate(path):
            frac = 0.0 if len(path) == 1 else k / (len(path) - 1)
            color = cmap(frac)
            label = "planned path configurations" if k == 0 else None
            plot_configuration(
                ax,
                cable,
                color=color,
                node_color=color,
                label=label,
                linewidth=1.9 if 0 < k < len(path) - 1 else 2.8,
                node_size=17 if 0 < k < len(path) - 1 else 28,
                alpha=0.62 if 0 < k < len(path) - 1 else 1.0,
                zorder=7 + k,
            )

    draw_start_goal(ax, scenario)
    configure_axes(ax, scenario)
    add_legend(ax)
    status = "success" if result.success else "failure"
    ax.set_title(
        f"{scenario.title} | {status}, {len(path)} configs, J={result.metrics['path_cost_J']:.3f}, "
        f"max d={result.metrics['max_edge_distance_m']:.3f}",
        fontsize=12.0,
    )
    save_figure(fig, output_dir, f"{scenario.name}_final_path_corridor")


def save_metrics(results: Sequence[PlanResult], output_dir: Path, write_dashboard: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = [result.metrics for result in results]
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2)

    if metric_rows:
        fieldnames = list(metric_rows[0].keys())
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

    if not write_dashboard:
        return

    ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(12.5, 3.6), constrained_layout=True)
    ax.axis("off")
    columns = [
        "scenario",
        "success",
        "timed_out",
        "iterations",
        "time_s",
        "time_limit_s",
        "refined_path_configs",
        "shortcut_refinement_enabled",
        "endpoint_reorder_enabled",
        "robot_execution_feasibility_enabled",
        "robot_tcp_clearance_m",
        "robot_obstacle_clearance_m",
        "robot_swept_tcp_clearance_m",
        "preserve_terminal_order_enabled",
        "endpoint_order_reversals",
        "epsilon_m",
        "max_edge_distance_m",
        "shortcut_validation_samples",
        "path_cost_J",
        "min_clearance_m",
        "transition_path_free",
    ]
    cell_text = []
    for row in metric_rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        cell_text.append(values)
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1.0, 1.35)
    ax.set_title("GCR planner metrics", fontsize=12.0, pad=10)
    save_figure(fig, output_dir, "gcr_metrics_dashboard")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full GCR planner and generate scenario plots/metrics.")
    parser.add_argument("output_dir", nargs="?", default="gcr_full_outputs", help="Output folder for plots and metrics.")
    parser.add_argument("--max-iterations", type=int, default=1600, help="Maximum bidirectional tree iterations per scenario.")
    parser.add_argument("--nodes", type=int, default=10, help="Number of cable nodes.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset added to scenario random seeds.")
    parser.add_argument("--scenario-time-limit", type=float, default=120.0, help="Maximum solve time per scenario in seconds; use 0 to disable.")
    parser.add_argument("--no-shortcut-refinement", action="store_true", help="Keep the raw connected path instead of removing intermediate cable profiles with shortcut refinement.")
    parser.add_argument("--endpoint-reorder", action="store_true", help="Optionally reverse cable profile order when endpoint continuity improves and the reversed transition is valid.")
    parser.add_argument(
        "--endpoint-reorder-cost-ratio",
        type=float,
        default=0.75,
        help="Reverse endpoint order when reversed endpoint travel is below this fraction of ordered endpoint travel.",
    )
    parser.add_argument(
        "--endpoint-reorder-tangent-margin",
        type=float,
        default=0.05,
        help="Reverse endpoint order when it improves end-to-end cable tangent alignment by more than this margin; use a negative value to disable.",
    )
    parser.add_argument(
        "--robot-base-aware-endpoint-reorder",
        action="store_true",
        help="When --endpoint-reorder is enabled, also prefer endpoint assignments whose base-to-TCP directions and TCP sweeps are robot-execution friendly.",
    )
    parser.add_argument(
        "--robot-base-vector-endpoint-reorder",
        action="store_true",
        help="When --endpoint-reorder is enabled, also reverse profiles that cause a large left-to-right TCP vector turn relative to the robot bases.",
    )
    parser.add_argument(
        "--robot-tcp-vector-sector-reorder",
        action="store_true",
        help="When --endpoint-reorder is enabled, prefer endpoint orders whose right-TCP-to-left-TCP vector stays in the base-adjusted left-facing sector.",
    )
    parser.add_argument(
        "--robot-tcp-vector-sector-feasibility",
        action="store_true",
        help="Reject final transitions/shortcuts whose right-TCP-to-left-TCP vector leaves the base-adjusted sector.",
    )
    parser.add_argument(
        "--robot-base-vector-max-turn-deg",
        type=float,
        default=100.0,
        help="Maximum preferred turn of the left-to-right TCP vector between adjacent profiles before reversed endpoint order is preferred.",
    )
    parser.add_argument(
        "--robot-base-vector-margin-deg",
        type=float,
        default=5.0,
        help="Minimum angular improvement required before vector-based endpoint reordering prefers a reversal.",
    )
    parser.add_argument(
        "--robot-tcp-vector-sector-margin-deg",
        type=float,
        default=5.0,
        help="Tolerance/improvement margin for --robot-tcp-vector-sector-reorder.",
    )
    parser.add_argument(
        "--max-cable-axis-turn-deg",
        type=float,
        default=180.0,
        help="Reject adjacent profile transitions whose cable end-to-end axis turns more than this angle. Default 180 disables this guard.",
    )
    parser.add_argument(
        "--robot-base-arm-clearance",
        type=float,
        default=ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
        help="Minimum sampled clearance between the left and right base-to-TCP rays used by --robot-base-aware-endpoint-reorder.",
    )
    parser.add_argument(
        "--robot-base-transition-samples",
        type=int,
        default=ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT,
        help="Number of interpolation samples for the base-to-TCP ray clearance check.",
    )
    parser.add_argument("--robot-execution-feasibility", action="store_true", help="Enable stricter endpoint-sweep transition checks for robot-executable paths.")
    parser.add_argument("--robot-tcp-clearance", type=float, default=0.0, help="TCP radius used for same-time left/right endpoint separation when --robot-execution-feasibility is enabled.")
    parser.add_argument(
        "--robot-obstacle-clearance",
        type=float,
        default=None,
        help="TCP-vs-obstacle sweep clearance. Default uses --robot-tcp-clearance for backward compatibility; use a small value to avoid over-rejecting near obstacles.",
    )
    parser.add_argument(
        "--robot-swept-tcp-clearance",
        type=float,
        default=0.0,
        help="Optional TCP radius for swept left-path vs right-path clearance. This catches crossing robot endpoint paths.",
    )
    parser.add_argument("--allow-terminal-reversal", action="store_true", help="Allow endpoint reordering to finish at reverse(goal), matching the old experimental endpoint-reorder behavior.")
    parser.add_argument("--skip-plots", action="store_true", help="Run planning and write metrics without generating PNG/PDF plots.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.robot_tcp_clearance < 0.0:
        raise ValueError("--robot-tcp-clearance must be non-negative.")
    if args.robot_obstacle_clearance is not None and args.robot_obstacle_clearance < 0.0:
        raise ValueError("--robot-obstacle-clearance must be non-negative.")
    if args.robot_swept_tcp_clearance < 0.0:
        raise ValueError("--robot-swept-tcp-clearance must be non-negative.")
    if args.endpoint_reorder_cost_ratio <= 0.0:
        raise ValueError("--endpoint-reorder-cost-ratio must be positive.")
    if args.robot_base_arm_clearance < 0.0:
        raise ValueError("--robot-base-arm-clearance must be non-negative.")
    if args.robot_base_transition_samples < 1:
        raise ValueError("--robot-base-transition-samples must be at least 1.")
    if args.robot_base_vector_max_turn_deg < 0.0 or args.robot_base_vector_max_turn_deg > 180.0:
        raise ValueError("--robot-base-vector-max-turn-deg must be in [0, 180].")
    if args.robot_base_vector_margin_deg < 0.0:
        raise ValueError("--robot-base-vector-margin-deg must be non-negative.")
    if args.robot_tcp_vector_sector_margin_deg < 0.0:
        raise ValueError("--robot-tcp-vector-sector-margin-deg must be non-negative.")
    if args.max_cable_axis_turn_deg < 0.0 or args.max_cable_axis_turn_deg > 180.0:
        raise ValueError("--max-cable-axis-turn-deg must be in [0, 180].")
    scenario_time_limit = None if args.scenario_time_limit <= 0.0 else args.scenario_time_limit
    params = PlannerParams(
        max_iterations=args.max_iterations,
        scenario_time_limit_s=scenario_time_limit,
        shortcut_refinement=not args.no_shortcut_refinement,
        endpoint_reorder=args.endpoint_reorder,
        robot_execution_feasibility=args.robot_execution_feasibility,
        robot_tcp_clearance=args.robot_tcp_clearance,
        robot_obstacle_clearance=args.robot_obstacle_clearance,
        robot_swept_tcp_clearance=args.robot_swept_tcp_clearance,
        preserve_terminal_order=not args.allow_terminal_reversal,
        endpoint_reorder_cost_ratio=args.endpoint_reorder_cost_ratio,
        endpoint_reorder_tangent_margin=args.endpoint_reorder_tangent_margin,
        robot_base_aware_endpoint_reorder=args.robot_base_aware_endpoint_reorder,
        robot_base_vector_endpoint_reorder=args.robot_base_vector_endpoint_reorder,
        robot_tcp_vector_sector_reorder=args.robot_tcp_vector_sector_reorder,
        robot_tcp_vector_sector_feasibility=args.robot_tcp_vector_sector_feasibility,
        robot_base_vector_max_turn_rad=math.radians(args.robot_base_vector_max_turn_deg),
        robot_base_vector_margin_rad=math.radians(args.robot_base_vector_margin_deg),
        robot_tcp_vector_sector_margin_rad=math.radians(args.robot_tcp_vector_sector_margin_deg),
        max_cable_axis_turn_rad=math.radians(args.max_cable_axis_turn_deg),
        robot_base_arm_clearance=args.robot_base_arm_clearance,
        robot_base_transition_samples=args.robot_base_transition_samples,
    )
    from gcr.planner.final_scenarios import build_final_scenarios

    scenarios = build_final_scenarios(args.nodes)
    if args.seed_offset:
        for scenario in scenarios:
            scenario.seed += args.seed_offset
    validate_scenarios(scenarios, params)

    output_dir = Path(args.output_dir)
    results: List[PlanResult] = []
    limit_label = "unlimited" if scenario_time_limit is None else f"{scenario_time_limit:.0f}s/scenario"
    shortcut_label = "on" if params.shortcut_refinement else "off"
    robot_exec_label = "on" if params.robot_execution_feasibility else "off"
    print(
        f"Solving GCR on {len(scenarios)} final scenarios "
        f"({limit_label}, shortcut_refinement={shortcut_label}, "
        f"endpoint_reorder={params.endpoint_reorder}, "
        f"robot_base_aware_endpoint_reorder={params.robot_base_aware_endpoint_reorder}, "
        f"robot_base_vector_endpoint_reorder={params.robot_base_vector_endpoint_reorder}, "
        f"robot_tcp_vector_sector_reorder={params.robot_tcp_vector_sector_reorder}, "
        f"robot_tcp_vector_sector_feasibility={params.robot_tcp_vector_sector_feasibility}, "
        f"max_cable_axis_turn_deg={args.max_cable_axis_turn_deg:.1f}, "
        f"robot_execution_feasibility={robot_exec_label}, robot_tcp_clearance={params.robot_tcp_clearance:.3f}m)..."
    )
    solve_start = time.perf_counter()
    for scenario in scenarios:
        planner = GCRPlanner(scenario, params)
        result = planner.plan()
        results.append(result)
        status = "success" if result.success else "timeout" if result.timed_out else "failure"
        print(
            f"{scenario.name}: status={status}, "
            f"iterations={result.iterations}, configs={result.metrics['refined_path_configs']}, "
            f"J={result.metrics['path_cost_J']:.4f}, solve_time={result.elapsed_s:.2f}s"
        )
    solve_elapsed = time.perf_counter() - solve_start
    print(f"Solve phase total: {solve_elapsed:.2f}s")

    plot_elapsed = 0.0
    if not args.skip_plots:
        print("Generating plots...")
        plot_start = time.perf_counter()
        for result in results:
            plot_intermediate_stages(result, output_dir)
            plot_final_path(result, output_dir)
        save_metrics(results, output_dir, write_dashboard=True)
        plot_elapsed = time.perf_counter() - plot_start
        print(f"Plot phase total: {plot_elapsed:.2f}s")
    else:
        save_metrics(results, output_dir, write_dashboard=False)

    artifact_label = "metrics" if args.skip_plots else "plots and metrics"
    print(f"Generated {artifact_label} in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
