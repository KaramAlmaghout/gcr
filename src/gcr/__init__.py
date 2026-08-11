"""Guided Cable Reconfiguration (GCR) planner, benchmark tooling, and MuJoCo replay.

Subpackages:

* :mod:`gcr.planner` — the Python reference planner and the scenario catalogue;
* :mod:`gcr.analysis` — benchmark plots, paper tables, and waypoint export;
* :mod:`gcr.mujoco_exec` — MuJoCo replay of planned paths on two UR10e arms;
* :mod:`gcr.sim` — MuJoCo scene construction and the dual-arm controller.

Nothing heavy is imported here: the plotting and MuJoCo dependencies are pulled
in only by the module that needs them, so ``import gcr`` stays cheap.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
