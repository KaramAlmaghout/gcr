import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.lines import Line2D


EPS = 1e-9


def point_to_segment_distance(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < EPS:
        return float(np.linalg.norm(point - start))
    t = np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0)
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def orientation(a, b, c):
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) < EPS:
        return 0
    return 1 if value > 0.0 else -1


def on_segment(a, b, c):
    return (
        min(a[0], c[0]) - EPS <= b[0] <= max(a[0], c[0]) + EPS
        and min(a[1], c[1]) - EPS <= b[1] <= max(a[1], c[1]) + EPS
        and orientation(a, b, c) == 0
    )


def segments_intersect(a, b, c, d):
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


def point_in_polygon(point, polygon):
    inside = False
    x, y = point
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if on_segment(polygon[j], point, polygon[i]):
            return True
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def segment_intersects_polygon(start, end, polygon):
    if point_in_polygon(start, polygon) or point_in_polygon(end, polygon):
        return True
    for i in range(len(polygon)):
        edge_start = polygon[i]
        edge_end = polygon[(i + 1) % len(polygon)]
        if segments_intersect(start, end, edge_start, edge_end):
            return True
    return False


def assert_collision_free(cable, polygon_obstacles, circular_obstacles, name):
    for i in range(len(cable) - 1):
        start = cable[i]
        end = cable[i + 1]
        for obstacle_index, polygon in enumerate(polygon_obstacles, start=1):
            if segment_intersects_polygon(start, end, polygon):
                raise ValueError(f"{name} segment {i + 1} intersects polygon obstacle O_{obstacle_index}")
        for circle_index, (center, radius) in enumerate(circular_obstacles, start=1 + len(polygon_obstacles)):
            center = np.array(center, dtype=float)
            if point_to_segment_distance(center, start, end) <= radius:
                raise ValueError(f"{name} segment {i + 1} intersects circular obstacle O_{circle_index}")


def circular_arc_nodes(center, radius, start_angle, end_angle, n_nodes):
    angles = np.linspace(start_angle, end_angle, n_nodes)
    center = np.asarray(center, dtype=float)
    return np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        )
    )


def segment_lengths(cable):
    return np.linalg.norm(np.diff(cable, axis=0), axis=1)


def cable_length(cable):
    return float(np.sum(segment_lengths(cable)))


def assert_same_uniform_segments(cable_start, cable_goal):
    start_lengths = segment_lengths(cable_start)
    goal_lengths = segment_lengths(cable_goal)
    if len(cable_start) != len(cable_goal):
        raise ValueError("Start and goal configurations must have the same number of nodes.")
    if not np.allclose(start_lengths, start_lengths[0]):
        raise ValueError("Start configuration does not have uniform segment lengths.")
    if not np.allclose(goal_lengths, goal_lengths[0]):
        raise ValueError("Goal configuration does not have uniform segment lengths.")
    if not np.allclose(start_lengths, goal_lengths):
        raise ValueError("Start and goal configurations do not have the same segment length.")


def get_problem_setup():
    workspace = (0.0, 1.0, 0.0, 0.6)
    polygon_obstacle_1 = np.array(
        [
            [0.22, 0.15],
            [0.36, 0.17],
            [0.38, 0.30],
            [0.27, 0.35],
            [0.19, 0.27],
        ]
    )
    polygon_obstacle_2 = np.array(
        [
            [0.60, 0.36],
            [0.76, 0.33],
            [0.82, 0.45],
            [0.71, 0.53],
            [0.58, 0.47],
        ]
    )
    circular_obstacles = [
        ((0.51, 0.20), 0.075),
        ((0.79, 0.18), 0.055),
    ]

    n_nodes = 8
    arc_radius = 0.20
    arc_span = np.deg2rad(80.0)
    cable_start = circular_arc_nodes(
        center=[0.310, 0.320],
        radius=arc_radius,
        start_angle=np.deg2rad(130.0),
        end_angle=np.deg2rad(130.0) - arc_span,
        n_nodes=n_nodes,
    )
    cable_goal = circular_arc_nodes(
        center=[0.660, 0.230],
        radius=arc_radius,
        start_angle=np.deg2rad(230.0),
        end_angle=np.deg2rad(230.0) + arc_span,
        n_nodes=n_nodes,
    )
    return workspace, [polygon_obstacle_1, polygon_obstacle_2], circular_obstacles, cable_start, cable_goal


def closest_point_on_segment(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < EPS:
        return start
    t = np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0)
    return start + t * segment


def closest_point_on_polygon(point, polygon):
    closest = None
    best_distance = np.inf
    for i in range(len(polygon)):
        candidate = closest_point_on_segment(point, polygon[i], polygon[(i + 1) % len(polygon)])
        distance = np.linalg.norm(point - candidate)
        if distance < best_distance:
            best_distance = distance
            closest = candidate
    return closest, best_distance


def closest_obstacle_point(point, polygon_obstacles, circular_obstacles):
    point = np.asarray(point, dtype=float)
    closest = None
    best_distance = np.inf

    for polygon in polygon_obstacles:
        candidate, distance = closest_point_on_polygon(point, polygon)
        if distance < best_distance:
            closest = candidate
            best_distance = distance

    for center, radius in circular_obstacles:
        center = np.asarray(center, dtype=float)
        direction = point - center
        norm = np.linalg.norm(direction)
        if norm < EPS:
            candidate = center + np.array([radius, 0.0])
        else:
            candidate = center + radius * direction / norm
        distance = np.linalg.norm(point - candidate)
        if distance < best_distance:
            closest = candidate
            best_distance = distance

    return closest, best_distance


def point_in_any_obstacle(point, polygon_obstacles, circular_obstacles):
    for polygon in polygon_obstacles:
        if point_in_polygon(point, polygon):
            return True
    for center, radius in circular_obstacles:
        if np.linalg.norm(np.asarray(point) - np.asarray(center)) <= radius:
            return True
    return False


def potential_gradient_components(points, target_points, polygon_obstacles, circular_obstacles, k_att=1.0, beta=0.018, sigma=0.075):
    attractive = 2.0 * k_att * (target_points - points)
    repulsive = np.zeros_like(points)
    for i, point in enumerate(points):
        closest, distance = closest_obstacle_point(point, polygon_obstacles, circular_obstacles)
        away = point - closest
        norm = np.linalg.norm(away)
        if norm < EPS:
            continue
        repulsive_strength = beta * (distance / sigma**2) * np.exp(-(distance**2) / (2.0 * sigma**2))
        repulsive[i] = repulsive_strength * away / norm
    return attractive, repulsive


def negative_potential_gradient(points, target_points, polygon_obstacles, circular_obstacles, k_att=1.0, beta=0.018, sigma=0.075):
    attractive, repulsive = potential_gradient_components(
        points,
        target_points,
        polygon_obstacles,
        circular_obstacles,
        k_att=k_att,
        beta=beta,
        sigma=sigma,
    )
    return attractive + repulsive


def scale_vectors_for_plot(vectors, max_length):
    norms = np.linalg.norm(vectors, axis=1)
    if len(norms) == 0 or np.max(norms) < EPS:
        return vectors
    return vectors / np.max(norms) * max_length


def draw_workspace_scene(ax, workspace, polygon_obstacles, circular_obstacles):
    ax.add_patch(
        Rectangle(
            (workspace[0], workspace[2]),
            workspace[1] - workspace[0],
            workspace[3] - workspace[2],
            facecolor="#eef7f2",
            edgecolor="#1f2937",
            linewidth=1.8,
            zorder=0,
        )
    )
    obstacle_style = {
        "facecolor": "#3f3f46",
        "edgecolor": "#18181b",
        "linewidth": 1.4,
        "alpha": 0.95,
        "zorder": 2,
    }
    for polygon in polygon_obstacles:
        ax.add_patch(Polygon(polygon, closed=True, **obstacle_style))
    for center, radius in circular_obstacles:
        ax.add_patch(Circle(center, radius, **obstacle_style))


def plot_configuration(ax, cable, color, node_color, label, linestyle="-", zorder=5):
    ax.plot(
        cable[:, 0],
        cable[:, 1],
        color=color,
        linewidth=3.0,
        linestyle=linestyle,
        solid_capstyle="round",
        label=label,
        zorder=zorder,
    )
    ax.scatter(
        cable[:, 0],
        cable[:, 1],
        s=45,
        color=node_color,
        edgecolor=color,
        linewidth=1.0,
        zorder=zorder + 1,
    )


def generate_biased_sampling_potential_field_figure():
    workspace, polygon_obstacles, circular_obstacles, cable_start, cable_goal = get_problem_setup()
    assert_same_uniform_segments(cable_start, cable_goal)
    assert_collision_free(cable_start, polygon_obstacles, circular_obstacles, "start configuration")
    assert_collision_free(cable_goal, polygon_obstacles, circular_obstacles, "goal configuration")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    annotation_fontsize = 14
    rng = np.random.default_rng(4)

    raw_start_side = cable_start + 0.35 * (cable_goal - cable_start) + rng.normal(scale=0.010, size=cable_start.shape)
    raw_goal_side = cable_goal + 0.35 * (cable_start - cable_goal) + rng.normal(scale=0.010, size=cable_goal.shape)

    scenarios = [
        {
            "ax": axes[0],
            "title": r"start tree bias: $C_{\mathrm{target}}=C_{\mathrm{goal}}$",
            "raw": raw_start_side,
            "target": cable_goal,
            "source": cable_start,
            "gradient_color": "#111827",
            "biased_label": r"$\Phi(C^{\mathrm{raw}})$ toward $C_{\mathrm{goal}}$",
        },
        {
            "ax": axes[1],
            "title": r"goal tree bias: $C_{\mathrm{target}}=C_{\mathrm{start}}$",
            "raw": raw_goal_side,
            "target": cable_start,
            "source": cable_goal,
            "gradient_color": "#111827",
            "biased_label": r"$\Phi(C^{\mathrm{raw}})$ toward $C_{\mathrm{start}}$",
        },
    ]

    def draw_biased_sampling_panel(fig_to_colorbar, ax, item):
        draw_workspace_scene(ax, workspace, polygon_obstacles, circular_obstacles)
        plot_configuration(ax, cable_start, "#2563eb", "#60a5fa", r"$C_{\mathrm{start}}$", zorder=6)
        plot_configuration(ax, cable_goal, "#7c3aed", "#c4b5fd", r"$C_{\mathrm{goal}}$", zorder=6)

        raw = item["raw"]
        target = item["target"]
        heatmap_x = np.linspace(workspace[0], workspace[1], 130)
        heatmap_y = np.linspace(workspace[2], workspace[3], 80)
        xx, yy = np.meshgrid(heatmap_x, heatmap_y)
        heatmap_points = np.column_stack((xx.ravel(), yy.ravel()))
        target_centroid = np.repeat(np.mean(target, axis=0)[None, :], len(heatmap_points), axis=0)
        attractive_intensity = np.linalg.norm(heatmap_points - target_centroid, axis=1)
        attractive_intensity /= max(np.max(attractive_intensity), EPS)

        repulsive_intensity = np.zeros(len(heatmap_points))
        repulsive_sigma = 0.025
        for point_index, point in enumerate(heatmap_points):
            _, distance = closest_obstacle_point(point, polygon_obstacles, circular_obstacles)
            repulsive_intensity[point_index] = np.exp(-(distance**2) / (2.0 * repulsive_sigma**2))

        gradient_magnitude = 0.55 * attractive_intensity + 1.25 * repulsive_intensity
        gradient_magnitude -= np.min(gradient_magnitude)
        gradient_magnitude /= max(np.max(gradient_magnitude), EPS)
        gradient_magnitude = gradient_magnitude.reshape(xx.shape)
        cmap = plt.colormaps["turbo"].copy()
        heatmap = ax.imshow(
            gradient_magnitude,
            extent=(workspace[0], workspace[1], workspace[2], workspace[3]),
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            alpha=0.62,
            interpolation="bicubic",
            zorder=1,
        )

        attractive, repulsive = potential_gradient_components(
            raw,
            target,
            polygon_obstacles,
            circular_obstacles,
            k_att=1.0,
            beta=0.030,
            sigma=0.065,
        )
        combined = attractive + repulsive
        combined_display = scale_vectors_for_plot(combined, 0.070)
        biased = raw + combined_display

        # ax.text(
        #     0.50,
        #     0.575,
        #     r"$\Phi(C)=C-\eta\nabla U(C),\quad U=U_{\mathrm{att}}+U_{\mathrm{rep}}$",
        #     fontsize=annotation_fontsize,
        #     color="#111827",
        #     fontweight="bold",
        #     ha="center",
        #     va="center",
        # )
        # ax.set_title(item["title"], fontsize=14, pad=10)
        ax.set_xlim(workspace[0] - 0.025, workspace[1] + 0.025)
        ax.set_ylim(workspace[2] - 0.025, workspace[3] + 0.025)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.65)
        cbar = fig_to_colorbar.colorbar(heatmap, ax=ax, fraction=0.034, pad=0.012)
        cbar.set_label(r"Normalized potential-field intensity", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        return ax

    for item in scenarios:
        draw_biased_sampling_panel(fig, item["ax"], item)

    separate_outputs = [
        (scenarios[0], "biased_sampling_start_to_goal"),
        (scenarios[1], "biased_sampling_goal_to_start"),
    ]
    for item, output_stem in separate_outputs:
        separate_fig, separate_ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
        draw_biased_sampling_panel(separate_fig, separate_ax, item)
        handles, labels = separate_ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        separate_fig.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            ncol=2,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#d1d5db",
            fontsize=10,
        )
        separate_fig.savefig(f"{output_stem}.png", dpi=300)
        separate_fig.savefig(f"{output_stem}.pdf")
        plt.close(separate_fig)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d1d5db",
        fontsize=10,
    )
    fig.savefig("biased_sampling_potential_field.png", dpi=300)
    fig.savefig("biased_sampling_potential_field.pdf")
    plt.close(fig)


def main():
    # Workspace W = [0, 1.0] x [0, 0.6] meters.
    workspace = (0.0, 1.0, 0.0, 0.6)

    # Obstacles O_m: a mix of closed polygonal and circular occupied regions.
    polygon_obstacle_1 = np.array(
        [
            [0.22, 0.15],
            [0.36, 0.17],
            [0.38, 0.30],
            [0.27, 0.35],
            [0.19, 0.27],
        ]
    )
    polygon_obstacle_2 = np.array(
        [
            [0.60, 0.36],
            [0.76, 0.33],
            [0.82, 0.45],
            [0.71, 0.53],
            [0.58, 0.47],
        ]
    )
    circular_obstacles = [
        ((0.51, 0.20), 0.075),
        ((0.79, 0.18), 0.055),
    ]

    # DLO configurations C_start and C_goal. Both have the same N and the same
    # segment length ell, but their endpoint locations differ.
    n_nodes = 8
    arc_radius = 0.20
    arc_span = np.deg2rad(80.0)
    cable_start = circular_arc_nodes(
        center=[0.310, 0.320],
        radius=arc_radius,
        start_angle=np.deg2rad(130.0),
        end_angle=np.deg2rad(130.0) - arc_span,
        n_nodes=n_nodes,
    )
    cable_goal = circular_arc_nodes(
        center=[0.660, 0.230],
        radius=arc_radius,
        start_angle=np.deg2rad(230.0),
        end_angle=np.deg2rad(230.0) + arc_span,
        n_nodes=n_nodes,
    )
    a_start, b_start = cable_start[0], cable_start[-1]
    a_goal, b_goal = cable_goal[0], cable_goal[-1]
    if len(cable_start) != n_nodes or len(cable_goal) != n_nodes:
        raise ValueError("Both cable configurations must contain exactly N nodes.")
    assert_same_uniform_segments(cable_start, cable_goal)
    total_cable_length = cable_length(cable_start)
    ell = total_cable_length / (n_nodes - 1)

    polygon_obstacles = [polygon_obstacle_1, polygon_obstacle_2]
    assert_collision_free(cable_start, polygon_obstacles, circular_obstacles, "start configuration")
    assert_collision_free(cable_goal, polygon_obstacles, circular_obstacles, "goal configuration")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    annotation_fontsize = 17
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(workspace[0] - 0.04, workspace[1] + 0.04)
    ax.set_ylim(workspace[2] - 0.04, workspace[3] + 0.04)

    # Workspace and collision-free region.
    workspace_patch = Rectangle(
        (workspace[0], workspace[2]),
        workspace[1] - workspace[0],
        workspace[3] - workspace[2],
        facecolor="#eef7f2",
        edgecolor="#1f2937",
        linewidth=2.0,
        zorder=0,
    )
    ax.add_patch(workspace_patch)
    # ax.text(
    #     0.25,
    #     5.68,
    #     r"$\mathcal{W}$",
    #     fontsize=12,
    #     color="#1f2937",
    #     ha="left",
    #     va="center",
    # )
    # ax.text(
    #     4.25,
    #     5.62,
    #     r"$\mathcal{F}$",
    #     fontsize=12,
    #     color="#166534",
    #     ha="left",
    #     va="center",
    # )

    # Static obstacles.
    obstacle_style = {
        "facecolor": "#3f3f46",
        "edgecolor": "#18181b",
        "linewidth": 1.8,
        "alpha": 0.95,
        "zorder": 2,
    }
    ax.add_patch(Polygon(polygon_obstacle_1, closed=True, **obstacle_style))
    ax.add_patch(Polygon(polygon_obstacle_2, closed=True, **obstacle_style))
    for center, radius in circular_obstacles:
        ax.add_patch(Circle(center, radius, **obstacle_style))

    ax.text(0.275, 0.25, r"$O_1$", fontsize=13, color="white", ha="center", va="center")
    ax.text(0.51, 0.20, r"$O_2$", fontsize=13, color="white", ha="center", va="center")
    ax.text(0.71, 0.435, r"$O_3$", fontsize=13, color="white", ha="center", va="center")
    ax.text(0.79, 0.18, r"$O_4$", fontsize=13, color="white", ha="center", va="center")

    # Cable centerlines and node positions.
    ax.plot(
        cable_start[:, 0],
        cable_start[:, 1],
        color="#2563eb",
        linewidth=3.2,
        solid_capstyle="round",
        label=r"$C_{\mathrm{start}}$",
        zorder=4,
    )
    ax.plot(
        cable_goal[:, 0],
        cable_goal[:, 1],
        color="#7c3aed",
        linewidth=3.2,
        linestyle="-",
        dash_capstyle="round",
        label=r"$C_{\mathrm{goal}}$",
        zorder=4,
    )
    ax.scatter(
        cable_start[1:-1, 0],
        cable_start[1:-1, 1],
        s=62,
        color="#60a5fa",
        edgecolor="#1e3a8a",
        linewidth=1.2,
        zorder=5,
    )
    ax.scatter(
        cable_goal[1:-1, 0],
        cable_goal[1:-1, 1],
        s=62,
        color="#c4b5fd",
        edgecolor="#4c1d95",
        linewidth=1.2,
        zorder=5,
    )
    ax.scatter(
        [a_start[0], b_start[0], a_goal[0], b_goal[0]],
        [a_start[1], b_start[1], a_goal[1], b_goal[1]],
        s=125,
        color="#f59e0b",
        edgecolor="#92400e",
        linewidth=1.6,
        zorder=6,
        label=r"fixed endpoints $p_1=a,\ p_N=b$",
    )

    # ax.text(3.9, 5.18, r"$C_{\mathrm{start}}$", fontsize=13, color="#1e3a8a", ha="center")
    # ax.text(4.8, 0.86, r"$C_{\mathrm{goal}}$", fontsize=13, color="#4c1d95", ha="center")


    # Labels for representative nodes and a nominal segment length ell.
    ax.annotate(
        r"$p_1$",
        xy=a_start,
        xytext=(a_start[0] - 0.01, a_start[1] + 0.02),
        # arrowprops={"arrowstyle": "->", "color": "#92400e", "lw": 1.2},
        fontsize=annotation_fontsize,
        ha="right",
    )
    ax.annotate(
        r"$p_N$",
        xy=b_start,
        xytext=(b_start[0] + 0.01, b_start[1] + 0.02),
        # arrowprops={"arrowstyle": "->", "color": "#92400e", "lw": 1.2},
        fontsize=annotation_fontsize,
        ha="left",
    )
    ax.annotate(
        r"$p_1$",
        xy=a_goal,
        xytext=(a_goal[0] - 0.01, a_goal[1] - 0.021),
        # arrowprops={"arrowstyle": "->", "color": "#92400e", "lw": 1.2},
        fontsize=annotation_fontsize,
        ha="right",
    )
    ax.annotate(
        r"$p_N$",
        xy=b_goal,
        xytext=(b_goal[0] + 0.01, b_goal[1] - 0.021),
        # arrowprops={"arrowstyle": "->", "color": "#92400e", "lw": 1.2},
        fontsize=annotation_fontsize,
        ha="left",
    )
    mid = 3
    start_segment_midpoint = 0.5 * (cable_start[mid] + cable_start[mid + 1])
    ax.annotate(
        rf"$\ell_0$",
        xy=start_segment_midpoint,
        xytext=(start_segment_midpoint[0] - 0.012, start_segment_midpoint[1] + 0.044),
        arrowprops={"arrowstyle": "->", "color": "#1e3a8a", "lw": 1.2},
        fontsize=annotation_fontsize,
        color="#1e3a8a",
        ha="center",
    )
    goal_segment_midpoint = 0.5 * (cable_goal[mid] + cable_goal[mid + 1])
    ax.annotate(
        rf"$\ell_0$",
        xy=goal_segment_midpoint,
        xytext=(goal_segment_midpoint[0] - 0.012, goal_segment_midpoint[1] + 0.044),
        arrowprops={"arrowstyle": "->", "color": "#4c1d95", "lw": 1.2},
        fontsize=annotation_fontsize,
        color="#4c1d95",
        ha="center",
    )
    node_index = 5
    start_node = cable_start[node_index]
    ax.annotate(
        r"$p_n$",
        xy=start_node,
        xytext=(start_node[0] + 0.010, start_node[1] + 0.048),
        arrowprops={"arrowstyle": "->", "color": "#1e3a8a", "lw": 1.2},
        fontsize=annotation_fontsize,
        color="#1e3a8a",
        ha="left",
    )
    goal_node = cable_goal[node_index]
    ax.annotate(
        r"$p_n$",
        xy=goal_node,
        xytext=(goal_node[0] + 0.010, goal_node[1] + 0.048),
        arrowprops={"arrowstyle": "->", "color": "#4c1d95", "lw": 1.2},
        fontsize=annotation_fontsize,
        color="#4c1d95",
        ha="left",
    )
    # ax.text(
    #     4.7,
    #     0.35,
    #     r"shorter configurations $C_{\mathrm{start}},C_{\mathrm{goal}}\subset\mathcal{F}$",
    #     fontsize=13,
    #     color="#1f2937",
    #     ha="center",
    # )

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#eef7f2", edgecolor="#1f2937", linewidth=1.5),
        Rectangle((0, 0), 1, 1, facecolor="#3f3f46", edgecolor="#18181b", linewidth=1.5),
        Line2D([0], [0], color="#2563eb", lw=3.2, marker="o", markersize=7),
        Line2D([0], [0], color="#7c3aed", lw=3.2, ls="-", marker="o", markersize=7),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#f59e0b",
            markeredgecolor="#92400e",
            markersize=9,
        ),
    ]
    legend_labels = [
        r"free space $\mathcal{F}$",
        r"static obstacles $\mathcal{O}=\{O_m\}_{m=1}^M$",
        r"start configuration $C_{\mathrm{start}}$",
        r"goal configuration $C_{\mathrm{goal}}$",
        r"robot end-effectors at start/goal",
    ]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d1d5db",
        fontsize=10,
    )

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    # ax.set_title("Planar DLO Workspace With Static Obstacles", fontsize=15, pad=12)
    ax.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.75)

    fig.savefig("dlo_workspace_problem_formulation.png", dpi=300)
    fig.savefig("dlo_workspace_problem_formulation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
    generate_biased_sampling_potential_field_figure()
