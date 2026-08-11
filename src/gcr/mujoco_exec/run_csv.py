"""Run the MuJoCo simulation with a user-supplied waypoint CSV/table.

Usage
-----
    cd <repository-root>
    python -m gcr replay \
        --csv examples/example_waypoints.csv

    # headless run - saves an MP4 alongside the CSV file:
    python -m gcr replay \
        --csv my_waypoints.csv --no-viewer

Accepted file format
--------------------
The parser accepts either comma-separated values or whitespace-aligned
tables (such as the table that ``execute_plan.py`` already prints), so
both of these are valid:

    # comma-separated:
    WP, Left x, Left y, Left yaw, Right x, Right y, Right yaw
    1, -0.450, 0.500,    0.0, -0.071, 0.500,  180.0
    2,  0.131, 0.261,  -50.0,  0.464, 0.243, -123.1
    3,  0.050, 0.100,    0.0,  0.429, 0.100,  180.0

    # whitespace-aligned (paste the table directly):
       WP    Left x   Left y  Left yaw°   Right x  Right y Right yaw°
      -------------------------------------------------------------------
        1    -0.450    0.500        0.0°    -0.071    0.500      180.0°
        2     0.131    0.261      -50.0°     0.464    0.243     -123.1°
        3     0.050    0.100        0.0°     0.429    0.100      180.0°

Rules:
  - lines starting with ``#`` are ignored
  - separator lines made entirely of dashes are ignored
  - degree symbols ``°`` and ``deg`` are stripped automatically
  - units: x, y in metres; yaw in **degrees**; cable height z is taken from
    ``CABLE_Z`` in ``build_scene.py``

Each row defines ONE waypoint with both grippers' world-frame positions
and yaws.  The cable is initialised straight between the first row's
left/right positions; physics relaxes it during the kinematic settle.

CLI
---
    --csv PATH                      Required: waypoint table file.
    --steps-per-waypoint N          MuJoCo steps per row (default 300).
    --yaw-mode MODE                 simultaneous or rotate-then-move.
    --invert-red-anchor-yaw         Make the left/red-anchor robot use the
                                    opposite equivalent yaw winding.
    --no-cable-endpoint-tangent-assist
                                    Disable the gentle endpoint tangent
                                    correction that makes cable yaw follow
                                    gripper yaw.
    --cable-endpoint-clamp-segments N
                                    Number of cable segments held by the
                                    gripper sleeve model.
    --cable-segment-multiplier N    Multiplies exported planner segments.
    --no-last-transition-side-correction
                                    Disable the final-transition cable side
                                    correction when cable profile metadata is
                                    available.
    --no-viewer                     Headless mode; saves an MP4.
    --video-path PATH               Optional headless MP4 output path.
    --video-four-corners            Headless MP4 uses a 1920x1080 2x2 layout
                                    with one camera view from each workspace
                                    corner.
    --video-max-seconds S           Time-compress headless video to at most S
                                    seconds (default 60).
    --obstacles PATH                Optional obstacle CSV. Legacy rows are
                                    ``cx, cy, radius`` cylinders; exported
                                    rows may be cylinders, boxes, or polygon
                                    prisms.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gcr.mujoco_exec.execute_plan import (
    DEFAULT_CABLE_SEGMENT_MULTIPLIER,
    GHOST_TARGET_SHAPE_CHOICES,
    SIDE_CORRECTION_MOVING_ROBOT_CHOICES,
    YAW_MODE_CHOICES,
    YAW_MODE_SIMULTANEOUS,
    execute_waypoints_in_mujoco,
)
from gcr.mujoco_exec.robot_aware_execution import (
    RobotAwareOptions,
    choose_safe_endpoint_assignment,
    globally_swapped_targets,
    score_waypoints,
)
from gcr.sim.build_scene import CABLE_LENGTH, CABLE_NUM_SEGMENTS, CABLE_Z


def _clean_token(tok: str) -> str:
    """Strip degree symbols and whitespace."""
    return tok.replace("°", "").replace("Â°", "").replace("deg", "").strip()


def _split_row(line: str) -> list[str]:
    """Split a row on commas OR whitespace; drop empty fields."""
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
    else:
        parts = line.split()
    return [_clean_token(p) for p in parts if p.strip()]


def _is_header_row(fields: list[str]) -> bool:
    """A row is a header row if its first field can't be parsed as a number."""
    if not fields:
        return True
    try:
        float(fields[0])
        return False
    except ValueError:
        return True


def _is_separator_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and all(c in "-=_+|" for c in s)


def _rectangle_polygon_as_box(
    name: str,
    vertices: np.ndarray,
    height: float,
    rgba: str,
) -> dict | None:
    """Return a box obstacle when a polygon is exactly an axis-aligned rectangle.

    Exported GCR rectangles arrive as polygon prisms.  In MuJoCo, box geoms
    give the same shape but avoid much slower capsule-vs-mesh collision checks.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (4, 2):
        return None

    rounded = np.round(vertices, decimals=12)
    xs = np.unique(rounded[:, 0])
    ys = np.unique(rounded[:, 1])
    if len(xs) != 2 or len(ys) != 2:
        return None

    expected = {(float(x), float(y)) for x in xs for y in ys}
    observed = {(float(x), float(y)) for x, y in rounded}
    if observed != expected:
        return None

    xmin, xmax = float(np.min(vertices[:, 0])), float(np.max(vertices[:, 0]))
    ymin, ymax = float(np.min(vertices[:, 1])), float(np.max(vertices[:, 1]))
    return dict(
        name=name,
        pos=((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 0.0),
        half_size=((xmax - xmin) / 2.0, (ymax - ymin) / 2.0, float(height) / 2.0),
        rgba=rgba,
    )


def parse_waypoint_metadata(path: Path) -> dict[str, object]:
    """Read optional ``# cable_*`` metadata comments from a waypoint CSV."""
    metadata: dict[str, object] = {}
    cable_profiles: dict[int, np.ndarray] = {}

    def parse_xyz_nodes(value: str) -> np.ndarray:
        points = []
        for point_text in value.split(";"):
            coords = [float(v.strip()) for v in point_text.split(",") if v.strip()]
            if len(coords) != 3:
                raise ValueError
            points.append(coords)
        nodes = np.asarray(points, dtype=np.float64)
        # Older exports stored the scene height in this metadata. Keep the
        # exported xy shape, but use the current MuJoCo cable height so
        # grippers and initial cable start aligned.
        nodes[:, 2] = CABLE_Z
        return nodes

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                continue
            body = stripped[1:].strip()
            if ":" not in body:
                continue
            key, value = [part.strip() for part in body.split(":", 1)]
            key = key.lower()
            try:
                if key == "cable_length_m":
                    metadata[key] = float(value)
                elif key in {"cable_nodes", "cable_segments"}:
                    metadata[key] = int(float(value))
                elif key == "cable_cstart_xyz":
                    metadata[key] = parse_xyz_nodes(value)
                elif key.startswith("cable_profile_") and key.endswith("_xyz"):
                    idx_text = key[len("cable_profile_"):-len("_xyz")]
                    cable_profiles[int(idx_text)] = parse_xyz_nodes(value)
            except ValueError:
                continue
    if cable_profiles:
        metadata["cable_profiles_xyz"] = [cable_profiles[idx] for idx in sorted(cable_profiles)]
    return metadata


def parse_waypoint_csv(
    path: Path,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Read a CSV/table and return list of ``(tl, tr, left_yaw, right_yaw)``.

    Positions are 3-D (z is auto-set from ``CABLE_Z``); yaws are in radians.
    """
    targets: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line_idx, raw in enumerate(fh, start=1):
            line = raw.split("#", 1)[0].rstrip("\n")
            if not line.strip():
                continue
            if _is_separator_line(line):
                continue
            fields = _split_row(line)
            if _is_header_row(fields):
                continue
            if len(fields) < 7:
                raise ValueError(
                    f"{path}:{raw_line_idx}: expected 7 numeric columns "
                    f"(WP, Lx, Ly, Lyaw, Rx, Ry, Ryaw), got {len(fields)}: "
                    f"{line.strip()!r}"
                )
            try:
                # fields[0] is the WP index; we don't need its value but we
                # require it to parse as an integer to catch malformed rows.
                _wp = int(float(fields[0]))
                lx, ly = float(fields[1]), float(fields[2])
                l_yaw_deg = float(fields[3])
                rx, ry = float(fields[4]), float(fields[5])
                r_yaw_deg = float(fields[6])
            except ValueError as e:
                raise ValueError(
                    f"{path}:{raw_line_idx}: failed to parse '{line.strip()}': {e}"
                )
            tl = np.array([lx, ly, CABLE_Z], dtype=np.float64)
            tr = np.array([rx, ry, CABLE_Z], dtype=np.float64)
            targets.append((tl, tr,
                            float(np.deg2rad(l_yaw_deg)),
                            float(np.deg2rad(r_yaw_deg))))
    if not targets:
        raise ValueError(f"No data rows found in {path}")
    return targets


def parse_obstacles_csv(path: Path) -> list[dict]:
    """Read optional obstacle CSV rows.

    Accepted rows:
      - legacy cylinder: ``cx, cy, radius``
      - typed cylinder: ``cylinder, name, x, y, radius, half_height``
      - typed box: ``box, name, x, y, hx, hy, hz``
    """
    obs: list[dict] = []
    _HH = 0.005
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line_idx, raw in enumerate(fh, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            fields = _split_row(line)
            if not fields:
                continue
            kind = fields[0].lower()

            try:
                if kind in {"cylinder", "circle"}:
                    if len(fields) < 6:
                        raise ValueError("typed cylinder needs: type,name,x,y,radius,half_height")
                    name = fields[1] or f"obs_{len(obs)}"
                    cx, cy = float(fields[2]), float(fields[3])
                    r, half_height = float(fields[4]), float(fields[5])
                    obs.append(dict(
                        name=name,
                        pos=(cx, cy, 0.0),
                        radius=r,
                        half_height=half_height,
                        rgba="0.80 0.20 0.20 0.85",
                    ))
                    continue

                if kind == "box":
                    if len(fields) < 7:
                        raise ValueError("typed box needs: type,name,x,y,hx,hy,hz")
                    name = fields[1] or f"obs_{len(obs)}"
                    cx, cy = float(fields[2]), float(fields[3])
                    hx, hy, hz = float(fields[4]), float(fields[5]), float(fields[6])
                    obs.append(dict(
                        name=name,
                        pos=(cx, cy, 0.0),
                        half_size=(hx, hy, hz),
                        rgba="0.20 0.20 0.20 0.90",
                    ))
                    continue

                if kind == "polygon":
                    if len(fields) < 9 or len(fields[3:]) % 2:
                        raise ValueError("typed polygon needs: type,name,height,x1,y1,x2,y2,x3,y3,...")
                    name = fields[1] or f"obs_{len(obs)}"
                    height = float(fields[2])
                    values = [float(v) for v in fields[3:]]
                    vertices = np.asarray(list(zip(values[0::2], values[1::2])), dtype=float)
                    rgba = "0.20 0.20 0.20 0.90"
                    box = _rectangle_polygon_as_box(name, vertices, height, rgba)
                    if box is not None:
                        obs.append(box)
                    else:
                        obs.append(dict(
                            name=name,
                            vertices=vertices,
                            height=height,
                            rgba=rgba,
                        ))
                    continue

                if _is_header_row(fields):
                    continue
                if len(fields) < 3:
                    raise ValueError("legacy cylinder needs: cx,cy,radius")
                cx, cy, r = float(fields[0]), float(fields[1]), float(fields[2])
                obs.append(dict(
                    name=f"obs_{len(obs)}",
                    pos=(cx, cy, 0.0),
                    radius=r,
                    half_height=_HH,
                    rgba="0.80 0.20 0.20 0.85",
                ))
            except ValueError as e:
                raise ValueError(f"{path}:{raw_line_idx}: failed to parse obstacle row {line!r}: {e}")
    return obs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--csv", required=True,
                    help="Path to the waypoint CSV / whitespace table.")
    ap.add_argument("--obstacles", default=None,
                    help=("Optional obstacle CSV: legacy cx,cy,r cylinders, "
                          "or exported cylinder/box/polygon rows."))
    ap.add_argument("--steps-per-waypoint", type=int, default=300)
    ap.add_argument("--max-speed", type=float, default=0.10,
                    help=("Peak gripper velocity cap [m/s] (default 0.10). "
                          "Lower = smoother cable, slower run. "
                          "Set to 0 to disable and use only --steps-per-waypoint."))
    ap.add_argument("--yaw-mode", choices=YAW_MODE_CHOICES,
                    default=YAW_MODE_SIMULTANEOUS,
                    help=("Waypoint yaw motion: interpolate with translation, "
                          "or rotate first and then translate."))
    ap.add_argument("--invert-red-anchor-yaw", action="store_true",
                    help=("Invert the yaw winding direction for the left robot "
                          "(the red-anchor robot) while keeping the same final yaw."))
    ap.add_argument("--cable-attachment-mode",
                    choices=("site_weld", "two_point_clamp", "body_weld"),
                    default=None,
                    help=("Override the cable endpoint attachment mode for this run. "
                          "Default comes from build_scene.py."))
    ap.add_argument("--cable-weld-torquescale", type=float, default=None,
                    help=("Override endpoint weld torquescale for this run. "
                          "Larger values transmit more yaw but can destabilize the cable."))
    ap.add_argument("--tcp-schedule",
                    choices=("simultaneous", "auto-stagger", "left-first", "right-first"),
                    default="simultaneous",
                    help=("TCP timing schedule. auto-stagger chooses left/right "
                          "lead timing per waypoint."))
    ap.add_argument("--tcp-stagger-fraction", type=float, default=0.35,
                    help=("Delay fraction for the follower when using "
                          "left-first/right-first/auto-stagger (default 0.35)."))
    ap.add_argument("--endpoint-filter",
                    choices=("fixed", "choose-global-swap", "force-global-swap"),
                    default="fixed",
                    help=("Endpoint assignment before replay. 'fixed' preserves CSV order; "
                          "'choose-global-swap' uses the same robot-aware global swap "
                          "scoring as run_csv_robot_aware.py."))
    tangent_group = ap.add_mutually_exclusive_group()
    tangent_group.add_argument("--cable-endpoint-tangent-assist",
                    dest="cable_endpoint_tangent_assist", action="store_true",
                    help=("Enable the experimental kinematic endpoint sleeve assist. "
                          "Default is off so the MuJoCo cable physics controls the shape."))
    tangent_group.add_argument("--no-cable-endpoint-tangent-assist",
                    dest="cable_endpoint_tangent_assist", action="store_false",
                    help="Keep the experimental kinematic endpoint sleeve assist disabled.")
    ap.set_defaults(cable_endpoint_tangent_assist=False)
    ap.add_argument("--cable-endpoint-tangent-alpha", type=float, default=0.04,
                    help="Endpoint tangent assist blend factor in [0, 1] (default 0.04 when enabled).")
    ap.add_argument("--cable-endpoint-clamp-segments", type=int, default=2,
                    help="Number of endpoint cable segments modeled as a short gripper sleeve (default 2 when enabled).")
    ap.add_argument("--cable-endpoint-tangent-velocity-damping", type=float, default=0.0,
                    help="Optional cable qvel damping after tangent assist in [0, 1] (default 0).")
    ap.add_argument("--cable-length", type=float, default=None,
                    help="Cable length [m]. Defaults to CSV metadata, then build_scene.py default.")
    ap.add_argument("--cable-segments", type=int, default=None,
                    help="Number of MuJoCo cable segments. Defaults to CSV metadata, then build_scene.py default.")
    ap.add_argument("--cable-segment-multiplier", type=int,
                    default=DEFAULT_CABLE_SEGMENT_MULTIPLIER,
                    help=("When CSV metadata provides planner segments, multiply them "
                          f"for MuJoCo simulation (default {DEFAULT_CABLE_SEGMENT_MULTIPLIER}). "
                          "--cable-segments overrides this."))
    ap.add_argument("--no-last-transition-side-correction", action="store_true",
                    help=("Disable the final-transition cable side correction. "
                          "By default it runs when the waypoint CSV contains "
                          "exported cable_profile_* metadata."))
    ap.add_argument("--side-correction-epsilon", type=float, default=1e-5,
                    help=("Signed cross-product tolerance for final-transition "
                          "cable side checking (default 1e-5)."))
    ap.add_argument("--side-correction-cable-margin", type=float, default=0.003,
                    help=("TCP distance margin below cable length before switching "
                          "the moving robot during side correction (default 0.003 m)."))
    ap.add_argument("--side-correction-moving-robot",
                    choices=SIDE_CORRECTION_MOVING_ROBOT_CHOICES,
                    default="farther-from-target",
                    help=("Which robot moves first when final-transition side correction "
                          "holds one TCP. Default moves the robot farther from its target "
                          "and holds the closer one."))
    ap.add_argument("--ghost-target-shape",
                    choices=GHOST_TARGET_SHAPE_CHOICES,
                    default="final",
                    help=("Draw a translucent non-colliding ghost cable from exported "
                          "cable_profile_* metadata. Default is 'final'; use 'all' "
                          "for every waypoint profile."))
    ap.add_argument("--no-ghost-target-shape",
                    action="store_const",
                    const="none",
                    dest="ghost_target_shape",
                    help="Disable the translucent target-shape ghost cable.")
    ap.add_argument("--ghost-target-rgba",
                    default="1.0 0.86 0.18 0.35",
                    help="RGBA color for --ghost-target-shape, e.g. '1 0.86 0.18 0.35'.")
    ap.add_argument("--final-error-wait", type=float, default=0.0,
                    help=("After the motion finishes, keep simulating for this many "
                          "seconds, then compute final cable-vs-goal error in x-y. "
                          "Use 10 for the requested settled metric."))
    ap.add_argument("--final-error-json", default=None,
                    help=("Optional JSON path for final cable error metrics. "
                          "If omitted and --final-error-wait > 0, writes next to the CSV."))
    ap.add_argument("--no-viewer", action="store_true",
                    help="Headless mode; saves an MP4 next to the CSV file.")
    ap.add_argument("--video-path", default=None,
                    help="Optional MP4 output path for --no-viewer runs.")
    ap.add_argument("--video-four-corners", action="store_true",
                    help=("For --no-viewer runs, save one 1920x1080 MP4 as a "
                          "2x2 grid with four fixed camera views around the "
                          "workspace."))
    ap.add_argument("--video-max-seconds", type=float, default=60.0,
                    help=("Time-compress --no-viewer videos by skipping rendered "
                          "frames so the final MP4 is at most this many seconds "
                          "(default 60)."))
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: {csv_path} not found", file=sys.stderr)
        return 1

    metadata = parse_waypoint_metadata(csv_path)
    targets = parse_waypoint_csv(csv_path)
    print(f"Loaded {len(targets)} waypoint(s) from {csv_path}.")
    cable_length = args.cable_length
    if cable_length is None:
        cable_length = float(metadata.get("cable_length_m", CABLE_LENGTH))
    if args.cable_segment_multiplier < 1:
        print("error: --cable-segment-multiplier must be at least 1", file=sys.stderr)
        return 1
    metadata_segments = metadata.get("cable_segments")
    if args.cable_segments is not None:
        cable_segments = args.cable_segments
    elif metadata_segments is not None:
        cable_segments = int(metadata_segments) * args.cable_segment_multiplier
    else:
        cable_segments = CABLE_NUM_SEGMENTS
    cable_cstart_nodes = metadata.get("cable_cstart_xyz")
    if not isinstance(cable_cstart_nodes, np.ndarray):
        cable_cstart_nodes = None
    cable_profile_nodes = metadata.get("cable_profiles_xyz")
    if not isinstance(cable_profile_nodes, list):
        cable_profile_nodes = None
    print(f"Cable: length={cable_length:.6f} m, sim segments={cable_segments}, sim nodes={cable_segments + 1}.")

    obstacles_3d: list[dict] = []
    if args.obstacles:
        obs_path = Path(args.obstacles)
        if not obs_path.exists():
            print(f"error: {obs_path} not found", file=sys.stderr)
            return 1
        obstacles_3d = parse_obstacles_csv(obs_path)
        print(f"Loaded {len(obstacles_3d)} obstacle(s) from {obs_path}.")

    endpoint_options = RobotAwareOptions(
        tcp_schedule=args.tcp_schedule,
        tcp_stagger_fraction=args.tcp_stagger_fraction,
        cable_attachment_mode=args.cable_attachment_mode,
        cable_weld_torquescale=args.cable_weld_torquescale,
        ghost_target_shape=args.ghost_target_shape,
        ghost_target_rgba=args.ghost_target_rgba,
    )
    assignment_mode = "fixed_csv"
    if args.endpoint_filter == "choose-global-swap":
        targets, assignment_mode, _score = choose_safe_endpoint_assignment(targets, obstacles_3d, endpoint_options)
    elif args.endpoint_filter == "force-global-swap":
        targets = globally_swapped_targets(targets)
        assignment_mode = "forced_global_swap"
        _score = score_waypoints(targets, obstacles_3d, endpoint_options)
    reverse_initial_cable = assignment_mode in {"global_swap", "forced_global_swap"}
    if reverse_initial_cable and cable_cstart_nodes is not None:
        cable_cstart_nodes = cable_cstart_nodes[::-1].copy()
        print("Initial cable node order: reversed to match swapped endpoints")
    if reverse_initial_cable and cable_profile_nodes is not None:
        cable_profile_nodes = [np.asarray(profile, dtype=np.float64)[::-1].copy() for profile in cable_profile_nodes]
        print("Cable profile node order: reversed to match swapped endpoints")
    print(f"Endpoint assignment: {assignment_mode}")

    use_viewer = not args.no_viewer
    if args.video_path:
        vid_path = Path(args.video_path)
        vid_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = csv_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        vid_path = out_dir / f"{csv_path.stem}.mp4"
    if args.final_error_json:
        final_error_json_path = Path(args.final_error_json)
        final_error_json_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.final_error_wait > 0.0:
        final_error_json_path = csv_path.parent / f"{csv_path.stem}_final_error.json"
    else:
        final_error_json_path = None

    execute_waypoints_in_mujoco(
        targets,
        obstacles_3d=obstacles_3d,
        steps_per_wp=args.steps_per_waypoint,
        max_gripper_speed=args.max_speed,
        cable_length=cable_length,
        cable_num_segments=cable_segments,
        cable_cstart_nodes=cable_cstart_nodes,
        cable_profile_nodes=cable_profile_nodes,
        ghost_target_shape=args.ghost_target_shape,
        ghost_target_rgba=args.ghost_target_rgba,
        final_error_wait=args.final_error_wait,
        final_error_json_path=final_error_json_path,
        yaw_mode=args.yaw_mode,
        last_transition_side_correction=not args.no_last_transition_side_correction,
        side_correction_epsilon=args.side_correction_epsilon,
        side_correction_cable_margin=args.side_correction_cable_margin,
        side_correction_moving_robot=args.side_correction_moving_robot,
        invert_red_anchor_yaw=args.invert_red_anchor_yaw,
        cable_endpoint_tangent_assist=args.cable_endpoint_tangent_assist,
        cable_endpoint_tangent_alpha=args.cable_endpoint_tangent_alpha,
        cable_endpoint_clamp_segments=args.cable_endpoint_clamp_segments,
        cable_endpoint_tangent_velocity_damping=args.cable_endpoint_tangent_velocity_damping,
        cable_attachment_mode=args.cable_attachment_mode,
        cable_weld_torquescale=args.cable_weld_torquescale,
        tcp_schedule=args.tcp_schedule,
        tcp_stagger_fraction=args.tcp_stagger_fraction,
        use_viewer=use_viewer,
        save_video=not use_viewer,
        video_path=vid_path,
        video_four_corners=args.video_four_corners,
        video_max_seconds=args.video_max_seconds,
    )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
