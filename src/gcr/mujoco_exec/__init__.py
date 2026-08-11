"""MuJoCo execution harness: replays GCR-planned paths with two UR10e arms.

* :mod:`gcr.mujoco_exec.run_csv` — replay an exported waypoint CSV (``gcr replay``);
* :mod:`gcr.mujoco_exec.execute_plan` — plan and replay in one process;
* :mod:`gcr.mujoco_exec.robot_aware_execution` — robot-aware checks and regrasps;
* :mod:`gcr.mujoco_exec.export_gcr_waypoints` — export scenario solutions to CSV;
* :mod:`gcr.mujoco_exec.view_gcr_scenario` — open a scenario in the MuJoCo viewer.
"""
