"""Replay a saved robot-aware MuJoCo trajectory in the viewer.

First record a trajectory with ``run_csv_robot_aware.py --save-trajectory``.
Then replay it with this script.  The CSV/obstacle files are used only to
rebuild the same MuJoCo scene; the motion comes from the saved state frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from gcr.sim.build_scene import CABLE_Z, build_and_compile
from gcr.mujoco_exec.execute_plan import _resample_polyline_nodes
from gcr.mujoco_exec.run_csv import parse_obstacles_csv, parse_waypoint_csv, parse_waypoint_metadata
from gcr.mujoco_exec.robot_aware_execution import globally_swapped_targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trajectory", required=True, help="Saved .npz trajectory from run_csv_robot_aware.py.")
    parser.add_argument("--csv", default=None, help="Waypoint CSV. Defaults to metadata stored in the trajectory.")
    parser.add_argument("--obstacles", default=None, help="Obstacle CSV. Defaults to metadata stored in the trajectory.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop until the viewer is closed.")
    return parser.parse_args(argv)


def load_metadata(data: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in data:
        return {}
    try:
        return json.loads(str(data["metadata_json"][0]))
    except (json.JSONDecodeError, IndexError, TypeError):
        return {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    traj_path = Path(args.trajectory)
    if not traj_path.exists():
        print(f"error: {traj_path} not found", file=sys.stderr)
        return 1

    traj = np.load(traj_path)
    meta = load_metadata(traj)
    csv_path = Path(args.csv or meta.get("csv", ""))
    obs_text = args.obstacles if args.obstacles is not None else meta.get("obstacles", "")
    obs_path = Path(obs_text) if obs_text else None

    if not csv_path.exists():
        print("error: provide --csv because it was not found in trajectory metadata", file=sys.stderr)
        return 1
    if obs_path is not None and not obs_path.exists():
        print(f"error: obstacle file {obs_path} not found", file=sys.stderr)
        return 1

    targets = parse_waypoint_csv(csv_path)
    assignment = str(meta.get("endpoint_assignment", "fixed_csv"))
    if assignment in {"global_swap", "forced_global_swap"}:
        targets = globally_swapped_targets(targets)
    tl_start, tr_start = targets[0][0], targets[0][1]

    metadata = parse_waypoint_metadata(csv_path)
    cable_num_segments = int(traj["cable_num_segments"][0])
    cable_cstart = metadata.get("cable_cstart_xyz")
    if not isinstance(cable_cstart, np.ndarray):
        cable_cstart = np.linspace(np.asarray(tl_start), np.asarray(tr_start), cable_num_segments + 1)
        cable_cstart[:, 2] = CABLE_Z
    elif assignment in {"global_swap", "forced_global_swap"}:
        cable_cstart = cable_cstart[::-1].copy()
    cable_cstart = _resample_polyline_nodes(cable_cstart, cable_num_segments)

    obstacles = parse_obstacles_csv(obs_path) if obs_path is not None else []
    workspace_bounds = (tuple(float(v) for v in traj["workspace_x"]), tuple(float(v) for v in traj["workspace_y"]))
    model, data, _ = build_and_compile(
        obstacles=obstacles,
        lock_robot_joints=False,
        ik_targets={"left": tl_start, "right": tr_start},
        cable_cstart=cable_cstart,
        cable_length=float(traj["cable_length"][0]),
        cable_num_segments=cable_num_segments,
        workspace_bounds=workspace_bounds,
    )

    qpos = traj["qpos"]
    qvel = traj["qvel"]
    mocap_pos = traj["mocap_pos"]
    mocap_quat = traj["mocap_quat"]
    dt = float(traj["timestep"][0]) / max(float(args.speed), 1e-6)

    print(f"Loaded {len(qpos)} replay frame(s) from {traj_path}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.5
        while viewer.is_running():
            for idx in range(len(qpos)):
                if not viewer.is_running():
                    break
                data.qpos[:] = qpos[idx]
                data.qvel[:] = qvel[idx]
                data.mocap_pos[:] = mocap_pos[idx]
                data.mocap_quat[:] = mocap_quat[idx]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)
            if not args.loop:
                break
        while viewer.is_running() and not args.loop:
            viewer.sync()
            time.sleep(0.03)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
