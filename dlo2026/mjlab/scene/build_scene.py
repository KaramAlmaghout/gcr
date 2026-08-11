"""Programmatically compose the full two-UR10e + cable + obstacles scene.

Why programmatic instead of hand-written MJCF: we need two copies of the same
Universal Robots UR10e model side by side, each with its own prefixed
actuators/joints, plus the cable composite, plus weld constraints between
the gripper bodies and the cable endpoints. MuJoCo's ``MjSpec`` lets us
attach two prefixed instances of the menagerie UR10e cleanly; hand-rolling
that in XML is error-prone.

Usage::

    from dlo2026.mjlab.scene.build_scene import build_two_ur10_cable_spec
    spec = build_two_ur10_cable_spec()
    model = spec.compile()

The build resolves the Menagerie path from (in order):
1. ``MUJOCO_MENAGERIE`` environment variable
2. ``<repo>/third_party/mujoco_menagerie``

The required UR10e model is vendored in this release. See the repository
``README.md`` for setup and licensing details.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize as _sp_minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Cable + workspace constants — ported from
# archive/isaac_sim/cable_simulation_integrated.py:19-29 so behavior is comparable.
# ---------------------------------------------------------------------------
# 1:1 mapping with the GCR planner: the 3-D workspace is laid out in
# execute_plan.py with SX = SY = 1.0, so the physical cable matches the
# planner's cable exactly.  set_cable_config places the cable bodies on the
# GCR Cstart node positions with no length residual at either weld.
CABLE_LENGTH = 0.40  # m  — matches GCR (cable_L = 0.40 m, l0 = 0.021053 m)
CABLE_DIAMETER = 0.01  # m
CABLE_GROUND_CLEARANCE = 0.0025  # m, capsule bottom clearance from the floor
# Default standalone MuJoCo resolution. Exported GCR scenarios can override
# this with their own metadata; the replay scripts upsample planner nodes to
# use a denser MuJoCo cable without changing cable length.
CABLE_NUM_SEGMENTS = 40

# MuJoCo's cable elasticity plugin needs nonzero material parameters. The
# original sample-like bend value made the DLO behave too much like a chain at
# our small 0.4 m scale, producing sharp endpoint kinks. This preset keeps the
# native MuJoCo cable but makes it behave more like a continuous rubber cable.
CABLE_TWIST_STIFFNESS = 1e4
CABLE_BEND_STIFFNESS = 3e5
CABLE_JOINT_DAMPING = 0.04
CABLE_STRESS_VMAX = 0.05

# Treat endpoint yaw as a short physical clamp, not a 1 m orientation lever.
# In MuJoCo welds, torquescale converts rotational residual to an equivalent
# positional scale; using the sleeve length avoids over-driving the first
# cable joint and creating an artificial >90 degree bend.
CABLE_GRIPPER_CLAMP_LENGTH = 0.035
CABLE_ENDPOINT_WELD_SOLREF = (0.02, 1.0)
CABLE_ENDPOINT_WELD_TORQUESCALE = float(
    os.environ.get("DLO_CABLE_WELD_TORQUESCALE", str(CABLE_GRIPPER_CLAMP_LENGTH))
)
CABLE_ENDPOINT_SITE_WELD_SOLREF = (0.01, 1.0)
CABLE_ENDPOINT_SITE_WELD_SOLIMP = (0.90, 0.95, 0.001, 0.5, 2.0)
CABLE_ENDPOINT_ATTACHMENT_MODE = os.environ.get(
    "DLO_CABLE_ATTACHMENT_MODE", "site_weld"
).strip().lower()
CABLE_ENDPOINT_CONNECT_SOLREF = (0.008, 1.0)
CABLE_ENDPOINT_CLAMP_SOLREF = (0.018, 1.0)
# Exact segment clamping is useful as a diagnostic boundary-condition check
# (45 deg anchor yaw -> 45 deg first-segment yaw), but it is intentionally off
# by default because directly rewriting cable qpos each step injects impulses
# and can break the MuJoCo simulation. Enable only for a short debug run with:
#   $env:DLO_EXACT_SEGMENT_CLAMP='1'
CABLE_ENDPOINT_EXACT_SEGMENT_CLAMP_ENABLED = (
    os.environ.get("DLO_EXACT_SEGMENT_CLAMP", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
CABLE_ENDPOINT_TANGENT_PIN_ENABLED = False
CABLE_ENDPOINT_TANGENT_SOLREF = (0.003, 1.0)
CABLE_ENDPOINT_TANGENT_MAX_BEND_DEG = 65.0

# Cable centerline height. A 10 mm cable at z = radius + 2.5 mm has 2.5 mm
# bottom clearance, keeping the setup close to the planner's 2-D plane.
CABLE_Z = CABLE_DIAMETER / 2.0 + CABLE_GROUND_CLEARANCE

CABLE_LEFT_ANCHOR_POS  = (-CABLE_LENGTH / 2, 0.0, CABLE_Z)
CABLE_RIGHT_ANCHOR_POS = ( CABLE_LENGTH / 2, 0.0, CABLE_Z)

# Floor-mounted UR10e robots on the x-axis, flanking the cable.
# UR10e reach ≈ 1300 mm; with cable ends at ±200 mm and bases at ±600 mm,
# the closest cable-end is 400 mm away — well inside the workspace.
# UR10e's `base` body has a built-in `quat="0 0 0 -1"` (180° around z).  The
# user-specified elbow-up seed (shoulder_pan ≈ 155.81°) is calibrated for
# the UR10e's NATIVE orientation, so we keep the base quat unmodified for
# the LEFT arm (frame_quat=identity → world quat = built-in 180°-around-z,
# and the user's joints reach toward +x in world).  RIGHT arm is mirrored
# by an additional 180°-around-z (frame_quat=(0,0,0,1)) so its reach points
# toward −x.
UR10_LEFT_BASE_POS   = (-0.75, 0.0, 0.0)
UR10_LEFT_BASE_QUAT  = (1.0, 0.0, 0.0, 0.0)
UR10_RIGHT_BASE_POS  = ( 0.75, 0.0, 0.0)
UR10_RIGHT_BASE_QUAT = (0.0, 0.0, 0.0, 1.0)

# UR10e joints in actuator order. The `solve_initial_pose` IK and the
# `DualUR10Controller` both rely on this ordering.
_UR10_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
_UR10_ACTUATOR_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
)
N_ROBOT_JOINTS = 6

# Default cylinder obstacles (overridden by execute_plan.py with scenario-specific ones).
# Placed near the cable centerline (y≈0) matching the new x-axis layout.
OBSTACLES = [
    dict(name="obs_0", pos=( 0.0,   0.06, 0.0), radius=0.04, half_height=0.005),
    dict(name="obs_1", pos=( 0.10,  0.0,  0.0), radius=0.03, half_height=0.005),
    dict(name="obs_2", pos=(-0.10,  0.0,  0.0), radius=0.03, half_height=0.005),
]


# ---------------------------------------------------------------------------
# Asset path resolution
# ---------------------------------------------------------------------------
def _find_menagerie() -> Path:
    env = os.environ.get("MUJOCO_MENAGERIE")
    if env:
        path = Path(env)
        if path.is_dir():
            return path
        raise FileNotFoundError(f"MUJOCO_MENAGERIE='{env}' but path does not exist")

    candidate = REPO_ROOT / "third_party" / "mujoco_menagerie"
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        "Could not locate mujoco_menagerie. Either set MUJOCO_MENAGERIE or run\n"
        "    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "
        "third_party/mujoco_menagerie\n"
        "from the repository root. See README.md for details."
    )


def _ur10_xml_path() -> Path:
    return _find_menagerie() / "universal_robots_ur10e" / "ur10e.xml"


# ---------------------------------------------------------------------------
# Base scene fragment — everything that does not depend on menagerie.
# ---------------------------------------------------------------------------
def _base_scene_xml(
    obstacles: list | None = None,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    cable_length: float = CABLE_LENGTH,
    cable_num_segments: int = CABLE_NUM_SEGMENTS,
    ghost_cable_profiles: list[np.ndarray] | None = None,
    ghost_cable_rgba: str = "1.0 0.86 0.18 0.35",
) -> str:
    if cable_length <= 0.0:
        raise ValueError("cable_length must be positive")
    if cable_num_segments < 1:
        raise ValueError("cable_num_segments must be at least 1")

    obs_list = OBSTACLES if obstacles is None else obstacles
    pedestal_xml = ""  # no pedestals — robots sit directly on the floor
    cable_left_anchor_pos = (-cable_length / 2.0, 0.0, CABLE_Z)
    cable_right_anchor_pos = (cable_length / 2.0, 0.0, CABLE_Z)
    mesh_assets: list[str] = []

    def _polygon_prism_mesh(vertices_xy, height: float) -> tuple[str, str]:
        verts_xy = np.asarray(vertices_xy, dtype=float)
        if verts_xy.ndim != 2 or verts_xy.shape[1] != 2 or len(verts_xy) < 3:
            raise ValueError("polygon obstacle vertices must be an Nx2 array")

        vertices = []
        for x, y in verts_xy:
            vertices.append((float(x), float(y), 0.0))
        for x, y in verts_xy:
            vertices.append((float(x), float(y), float(height)))

        n = len(verts_xy)
        faces = []
        for i in range(1, n - 1):
            faces.append((0, i + 1, i))
            faces.append((n, n + i, n + i + 1))
        for i in range(n):
            j = (i + 1) % n
            faces.append((i, j, n + j))
            faces.append((i, n + j, n + i))

        vertex_attr = " ".join(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
        face_attr = " ".join(f"{a} {b} {c}" for a, b, c in faces)
        return vertex_attr, face_attr

    def _obs_geom(o: dict) -> str:
        rgba = o.get('rgba', '0.8 0.2 0.2 0.7')
        if 'vertices' in o:
            mesh_name = f"{o['name']}_mesh"
            height = float(o.get('height', CABLE_Z))
            vertex_attr, face_attr = _polygon_prism_mesh(o['vertices'], height)
            mesh_assets.append(
                f"""    <mesh name="{mesh_name}" vertex="{vertex_attr}" face="{face_attr}"/>"""
            )
            return (
                f"""    <body name="{o['name']}" pos="0 0 0">
      <geom type="mesh" mesh="{mesh_name}" rgba="{rgba}" condim="3"/>
    </body>"""
            )
        if 'half_size' in o:
            hx, hy, hz = o['half_size']
            return (
                f"""    <body name="{o['name']}" pos="{o['pos'][0]} {o['pos'][1]} {o['pos'][2]}">
      <geom type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" rgba="{rgba}" condim="3"/>
    </body>"""
            )
        # default: cylinder
        hh = o['half_height']
        return (
            f"""    <body name="{o['name']}" pos="{o['pos'][0]} {o['pos'][1]} {o['pos'][2]}">
      <geom type="cylinder" size="{o['radius']} {hh}" pos="0 0 {hh}" rgba="{rgba}" condim="3"/>
    </body>"""
        )

    obstacle_xml = "\n".join(_obs_geom(o) for o in obs_list)
    mesh_asset_xml = "\n".join(mesh_assets)

    def _ghost_cable_xml() -> str:
        if not ghost_cable_profiles:
            return ""
        geoms: list[str] = []
        radius = max(CABLE_DIAMETER * 0.42, 0.0015)
        node_radius = max(CABLE_DIAMETER * 0.60, 0.0020)
        z_offset = 0.003
        for profile_idx, profile in enumerate(ghost_cable_profiles, start=1):
            nodes = np.asarray(profile, dtype=float)
            if nodes.ndim != 2 or nodes.shape[1] < 3 or len(nodes) < 2:
                raise ValueError(
                    "ghost_cable_profiles entries must be (N, >=3) arrays "
                    f"with N >= 2; bad profile {profile_idx}"
                )
            for seg_idx, (p0, p1) in enumerate(zip(nodes[:-1], nodes[1:]), start=1):
                a = np.asarray(p0[:3], dtype=float).copy()
                b = np.asarray(p1[:3], dtype=float).copy()
                a[2] += z_offset
                b[2] += z_offset
                if float(np.linalg.norm(b - a)) < 1e-9:
                    continue
                geoms.append(
                    f"""      <geom name="ghost_cable_{profile_idx:02d}_{seg_idx:03d}"
            type="capsule" fromto="{a[0]:.9g} {a[1]:.9g} {a[2]:.9g} {b[0]:.9g} {b[1]:.9g} {b[2]:.9g}"
            size="{radius:.9g}" rgba="{ghost_cable_rgba}" contype="0" conaffinity="0"/>"""
                )
            for node_idx, p in enumerate(nodes, start=1):
                q = np.asarray(p[:3], dtype=float).copy()
                q[2] += z_offset
                geoms.append(
                    f"""      <geom name="ghost_cable_node_{profile_idx:02d}_{node_idx:03d}"
            type="sphere" pos="{q[0]:.9g} {q[1]:.9g} {q[2]:.9g}"
            size="{node_radius:.9g}" rgba="{ghost_cable_rgba}" contype="0" conaffinity="0"/>"""
                )
        return "\n".join(
            [
                '    <body name="ghost_target_cable" pos="0 0 0">',
                *geoms,
                "    </body>",
            ]
        )

    ghost_target_xml = _ghost_cable_xml()

    # Workspace outline: four thin edge bars marking the planner's xy bounds,
    # so the user can see where the GCR 2-D workspace lives in the 3-D scene.
    if workspace_bounds is not None:
        (xmin, xmax), (ymin, ymax) = workspace_bounds
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        wx = (xmax - xmin) / 2   # half-extent along x
        wy = (ymax - ymin) / 2   # half-extent along y
        T = 0.005                # bar half-thickness in xy
        H = 0.002                # bar half-height in z (very thin)
        rgba = "1.0 0.0 0.0 0.95"
        workspace_xml = f"""
    <body name="workspace_box" pos="0 0 0">
      <geom type="box" size="{wx + T} {T} {H}" pos="{cx} {ymin} {H}" rgba="{rgba}" contype="0" conaffinity="0"/>
      <geom type="box" size="{wx + T} {T} {H}" pos="{cx} {ymax} {H}" rgba="{rgba}" contype="0" conaffinity="0"/>
      <geom type="box" size="{T} {wy} {H}" pos="{xmin} {cy} {H}" rgba="{rgba}" contype="0" conaffinity="0"/>
      <geom type="box" size="{T} {wy} {H}" pos="{xmax} {cy} {H}" rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>"""
    else:
        workspace_xml = ""

    if CABLE_ENDPOINT_ATTACHMENT_MODE == "body_weld":
        endpoint_attachment_xml = f"""
  <!-- Cable-to-anchor welds: legacy rigid 6-DOF body attachment. -->
  <equality>
    <weld name="anchor_left_pin"  body1="cableB_first" body2="anchor_left"
          relpose="0 0 0 1 0 0 0"
          solref="{CABLE_ENDPOINT_WELD_SOLREF[0]} {CABLE_ENDPOINT_WELD_SOLREF[1]}"/>
    <weld name="anchor_right_pin" body1="cableB_last"  body2="anchor_right"
          relpose="0 0 0 1 0 0 0"
          solref="{CABLE_ENDPOINT_WELD_SOLREF[0]} {CABLE_ENDPOINT_WELD_SOLREF[1]}"/>
  </equality>"""
    elif CABLE_ENDPOINT_ATTACHMENT_MODE in {"two_point_clamp", "site_weld"}:
        endpoint_attachment_xml = """
  <!-- Cable endpoint attachment is added programmatically. Supported modes:
       site_weld: site-based weld from cable endpoint sites to mocap gripper
                  anchor sites, with realistic torquescale.
       two_point_clamp: endpoint + one-interior-point soft connect constraints. -->"""
    else:
        raise ValueError(f"unknown CABLE_ENDPOINT_ATTACHMENT_MODE={CABLE_ENDPOINT_ATTACHMENT_MODE!r}")

    return f"""<?xml version="1.0" ?>
<mujoco model="dlo2026_base">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" iterations="200" tolerance="1e-10"
          ls_iterations="50"/>
  <compiler angle="radian" autolimits="true"/>

  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.7 0.7 0.7" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="0"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <material name="groundplane" rgba="0.75 0.75 0.75 1"
              specular="0" shininess="0" reflectance="0" emission="0"/>
{mesh_asset_xml}
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05" material="groundplane"/>
{workspace_xml}
{ghost_target_xml}

    <!-- Cable anchors as MOCAP bodies. Mocap bodies have positions set via
         data.mocap_pos[i] each step (no DOFs from the solver's perspective)
         so the connect-equality from each cable end to its anchor doesn't
         apply impulses to a moving dynamic body. The simulation step wrapper
         in dlo2026.mjlab.controller writes data.mocap_pos = gripper_tip xpos
         each step, so the cable end follows the gripper. -->
    <body name="anchor_left"  mocap="true" pos="{cable_left_anchor_pos[0]} {cable_left_anchor_pos[1]} {cable_left_anchor_pos[2]}">
      <geom type="box" size="0.003 0.003 0.003" rgba="1 0.6 0.6 0.4" contype="0" conaffinity="0"/>
    </body>
    <body name="anchor_right" mocap="true" pos="{cable_right_anchor_pos[0]} {cable_right_anchor_pos[1]} {cable_right_anchor_pos[2]}">
      <geom type="box" size="0.003 0.003 0.003" rgba="0.6 1 0.6 0.4" contype="0" conaffinity="0"/>
    </body>

    <!-- MuJoCo native cable composite with the built-in elasticity plugin.
         The material values below are the first-party DeepMind cable sample
         parameters; without nonzero twist/bend the plugin contributes no
         visible elastic rod force and the composite behaves like a chain. -->
    <composite type="cable" prefix="cable" curve="s"
               count="{cable_num_segments + 1} 1 1" size="{cable_length}"
               offset="{cable_left_anchor_pos[0]} {cable_left_anchor_pos[1]} {cable_left_anchor_pos[2]}"
               initial="free">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="{CABLE_TWIST_STIFFNESS}"/>
        <config key="bend" value="{CABLE_BEND_STIFFNESS}"/>
        <config key="vmax" value="{CABLE_STRESS_VMAX}"/>
      </plugin>
      <joint kind="main" damping="{CABLE_JOINT_DAMPING}"/>
      <geom type="capsule" size="{CABLE_DIAMETER / 2.0}"
            rgba="0.10 0.35 0.90 1" condim="1"/>
    </composite>

{pedestal_xml}
{obstacle_xml}
  </worldbody>
{endpoint_attachment_xml}
</mujoco>
"""


# ---------------------------------------------------------------------------
# Gripper-tip placement on each UR10e wrist_3_link.
#
# UR10e's wrist_3_link extends along its local +y axis (the joint axis).
# The attachment_site sits at (0, 0.1, 0); the eef collision cylinder ends
# around y=0.117. We place gripper_tip at y=0.185 so it is clearly past the
# wrist housing — same 85 mm overhang past the eef as the iiwa14 setup.
#
# We rotate the gripper_tip body by −90° around its parent's x-axis so the
# tip's *local* z-axis aligns with the wrist_3 tool direction (+y in
# wrist_3's frame). That keeps the IK code unchanged: it tracks gripper_tip
# R[:,2] = (0,0,−1) for "pointing down", as it did for iiwa14's link7.
# ---------------------------------------------------------------------------
GRIPPER_TIP_OFFSET_LOCAL = (0.0, 0.185, 0.0)
# −90° around x: (cos −π/4, sin −π/4, 0, 0)
GRIPPER_TIP_QUAT_LOCAL = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)
GRIPPER_TIP_GEOM_SIZE = 0.0045  # 1/4 of the previous 0.018 m sphere
END_EFFECTOR_VISUAL_ENABLED = (
    os.environ.get("DLO_ENDEFFECTOR_VISUAL", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)
END_EFFECTOR_VISUAL_LENGTH = float(os.environ.get("DLO_ENDEFFECTOR_VISUAL_LENGTH", "0.115"))
END_EFFECTOR_VISUAL_SHAFT_RADIUS = float(os.environ.get("DLO_ENDEFFECTOR_VISUAL_SHAFT_RADIUS", "0.009"))
END_EFFECTOR_VISUAL_COLLAR_RADIUS = float(os.environ.get("DLO_ENDEFFECTOR_VISUAL_COLLAR_RADIUS", "0.022"))
END_EFFECTOR_VISUAL_COLLAR_HALFHEIGHT = float(os.environ.get("DLO_ENDEFFECTOR_VISUAL_COLLAR_HALFHEIGHT", "0.005"))
END_EFFECTOR_VISUAL_SLEEVE_LENGTH = float(
    os.environ.get("DLO_ENDEFFECTOR_VISUAL_SLEEVE_LENGTH", str(0.005))
)
END_EFFECTOR_VISUAL_SLEEVE_RADIUS = float(os.environ.get("DLO_ENDEFFECTOR_VISUAL_SLEEVE_RADIUS", "0.0075"))


# Names of every UR10e link used by ``_enable_robot_gravcomp``.
_UR10_LINK_NAMES = (
    "base", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
)


def _enable_robot_gravcomp(spec: mujoco.MjSpec) -> None:
    """Set per-body gravcomp=1 on every UR10e link (and the gripper tip).

    Why
    ---
    UR10e position actuators (gain 5000, damping 500) have a small but
    non-zero steady-state error under load:
    Δqpos ≈ τ_external / gain.  The cable's vertical pull (~1 N at full
    mass) plus link gravity cause a few-millimetre droop at the
    end-effector — visible in scenario 1 because the gripper target z
    sits just above the floor.  Setting body
    ``gravcomp`` to 1 makes MuJoCo apply a per-body upward force equal to
    body_mass × g, which exactly cancels gravity for the robot links and
    eliminates the droop.  The cable still has gravity (we don't touch
    its bodies), so it still hangs naturally.
    """
    for side in ("left", "right"):
        for name in _UR10_LINK_NAMES:
            try:
                spec.body(f"ur10_{side}/{name}").gravcomp = 1.0
            except (KeyError, ValueError, AttributeError):
                # Body missing (defensive) — older MjSpec API may need a
                # different lookup; in that case fall back silently.
                pass


def _add_gripper_tips(spec: mujoco.MjSpec) -> None:
    """Add a ``gripper_tip`` body as a child of each UR10e wrist_3_link."""
    for side in ("left", "right"):
        wrist3 = spec.body(f"ur10_{side}/wrist_3_link")
        tip = wrist3.add_body(
            name=f"ur10_{side}/gripper_tip",
            pos=np.array(GRIPPER_TIP_OFFSET_LOCAL),
            quat=np.array(GRIPPER_TIP_QUAT_LOCAL),
        )
        # Inherit gravcomp from the rest of the robot — although the tip
        # has zero rest mass, this keeps it weightless if the user later
        # adds a gripper geom with non-zero mass.
        try:
            tip.gravcomp = 1.0
        except (AttributeError,):
            pass
        # Visual marker: a small tinted sphere so the user can see where the
        # cable is welded. contype=0/conaffinity=0 means it doesn't collide.
        rgba = (1.0, 0.4, 0.4, 1.0) if side == "left" else (0.4, 1.0, 0.4, 1.0)
        if END_EFFECTOR_VISUAL_ENABLED:
            # Visual-only adapter: local +z points from the flange toward the
            # cable anchor when the IK solution is pointing down, so the shaft
            # extends back toward the flange along local -z. The sleeve lies
            # along local x, the same axis used for endpoint yaw/tangent.
            shaft_len = max(END_EFFECTOR_VISUAL_LENGTH, 1e-4)
            shaft_end = -shaft_len
            tip.add_geom(
                name=f"ur10_{side}/end_effector_shaft",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=np.array([0.0, 0.0, shaft_end, 0.0, 0.0, 0.0]),
                size=np.array([END_EFFECTOR_VISUAL_SHAFT_RADIUS, 0.0, 0.0]),
                rgba=np.array([0.12, 0.12, 0.12, 1.0]),
                contype=0,
                conaffinity=0,
                density=0.0,
                group=1,
            )
            tip.add_geom(
                name=f"ur10_{side}/end_effector_flange_collar",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=np.array([0.0, 0.0, shaft_end]),
                size=np.array([
                    END_EFFECTOR_VISUAL_COLLAR_RADIUS,
                    END_EFFECTOR_VISUAL_COLLAR_HALFHEIGHT,
                    0.0,
                ]),
                rgba=np.array([0.08, 0.08, 0.08, 1.0]),
                contype=0,
                conaffinity=0,
                density=0.0,
                group=1,
            )
            sleeve_half = 0.5 * max(END_EFFECTOR_VISUAL_SLEEVE_LENGTH, 1e-4)
            tip.add_geom(
                name=f"ur10_{side}/end_effector_cable_sleeve",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=np.array([-sleeve_half, 0.0, 0.0, sleeve_half, 0.0, 0.0]),
                size=np.array([END_EFFECTOR_VISUAL_SLEEVE_RADIUS, 0.0, 0.0]),
                rgba=np.array(rgba),
                contype=0,
                conaffinity=0,
                density=0.0,
                group=1,
            )
        tip.add_geom(
            name=f"ur10_{side}/gripper_tip_marker",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([GRIPPER_TIP_GEOM_SIZE, 0.0, 0.0]),
            rgba=np.array(rgba),
            contype=0,
            conaffinity=0,
            density=0.0,
            group=1,
        )


def _add_cable_endpoint_two_point_clamps(spec: mujoco.MjSpec) -> None:
    """Attach each cable end with symmetric endpoint + tangent-point connects.

    A body weld to ``cableB_first``/``cableB_last`` is not geometrically
    symmetric for MuJoCo cable composites: the left endpoint is the first body
    origin, while the right endpoint is the ``cableS_last`` site one segment
    beyond ``cableB_last``. These connect constraints use sites at the actual
    cable endpoints plus one interior point on both ends, so red and green
    anchors transmit position and yaw through the same mechanism.
    """
    if CABLE_ENDPOINT_ATTACHMENT_MODE != "two_point_clamp":
        return

    specs = [
        ("left", "cableS_first", "cableB_1"),
        ("right", "cableS_last", "cableB_last"),
    ]
    for side, endpoint_site_name, clamp_body_name in specs:
        try:
            anchor_body = spec.body(f"anchor_{side}")
            clamp_body = spec.body(clamp_body_name)
        except (KeyError, ValueError, AttributeError):
            continue

        anchor_site_name = f"anchor_{side}_endpoint_site"
        anchor_body.add_site(
            name=anchor_site_name,
            pos=np.zeros(3),
            size=np.array([0.001]),
            rgba=np.array([1.0, 0.8, 0.2, 0.0]),
        )

        clamp_anchor_body_name = f"anchor_{side}_clamp"
        clamp_anchor_site_name = f"{clamp_anchor_body_name}_site"
        clamp_anchor_body = spec.worldbody.add_body(
            name=clamp_anchor_body_name,
            mocap=True,
            pos=np.array([0.0, 0.0, CABLE_Z]),
        )
        clamp_anchor_body.add_site(
            name=clamp_anchor_site_name,
            pos=np.zeros(3),
            size=np.array([0.001]),
            rgba=np.array([1.0, 0.8, 0.2, 0.0]),
        )

        cable_clamp_site_name = f"cable_{side}_clamp_site"
        clamp_body.add_site(
            name=cable_clamp_site_name,
            pos=np.zeros(3),
            size=np.array([0.001]),
            rgba=np.array([0.2, 0.7, 1.0, 0.0]),
        )

        endpoint_eq = spec.add_equality()
        endpoint_eq.name = f"anchor_{side}_endpoint_pin"
        endpoint_eq.type = mujoco.mjtEq.mjEQ_CONNECT
        endpoint_eq.objtype = mujoco.mjtObj.mjOBJ_SITE
        endpoint_eq.name1 = endpoint_site_name
        endpoint_eq.name2 = anchor_site_name
        endpoint_eq.solref = np.array(CABLE_ENDPOINT_CONNECT_SOLREF)

        clamp_eq = spec.add_equality()
        clamp_eq.name = f"anchor_{side}_clamp_pin"
        clamp_eq.type = mujoco.mjtEq.mjEQ_CONNECT
        clamp_eq.objtype = mujoco.mjtObj.mjOBJ_SITE
        clamp_eq.name1 = cable_clamp_site_name
        clamp_eq.name2 = clamp_anchor_site_name
        clamp_eq.solref = np.array(CABLE_ENDPOINT_CLAMP_SOLREF)


def _add_cable_endpoint_site_welds(spec: mujoco.MjSpec) -> None:
    """Attach cable endpoint sites to gripper-tracking mocap sites with welds.

    This is the recommended MuJoCo-style rigid grasp test: use site semantics
    on the generated cable endpoint sites (``cableS_first``/``cableS_last``)
    and a site on each mocap anchor body that tracks the robot gripper pose.
    The weld transfers rotational moment without direct qpos rewriting.
    """
    if CABLE_ENDPOINT_ATTACHMENT_MODE != "site_weld":
        return

    specs = [
        ("left", "cableS_first"),
        ("right", "cableS_last"),
    ]
    for side, cable_site_name in specs:
        try:
            anchor_body = spec.body(f"anchor_{side}")
        except (KeyError, ValueError, AttributeError):
            continue

        anchor_site_name = f"anchor_{side}_weld_site"
        anchor_body.add_site(
            name=anchor_site_name,
            pos=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            size=np.array([0.001]),
            rgba=np.array([1.0, 0.8, 0.2, 0.0]),
        )

        eq = spec.add_equality()
        eq.name = f"anchor_{side}_site_weld"
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_SITE
        eq.name1 = cable_site_name
        eq.name2 = anchor_site_name
        eq.solref = np.array(CABLE_ENDPOINT_SITE_WELD_SOLREF)
        eq.solimp = np.array(CABLE_ENDPOINT_SITE_WELD_SOLIMP)
        # For welds, data[10] is torquescale. It acts like the diameter of a
        # glue patch, so use the physical gripper clamp length rather than 1 m.
        eq.data = np.array(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             CABLE_ENDPOINT_WELD_TORQUESCALE],
            dtype=np.float64,
        )


def _add_cable_endpoint_tangent_pins(spec: mujoco.MjSpec) -> None:
    """Add soft one-segment-inward pins so endpoint yaw reaches the cable.

    The endpoint weld attaches the gripper to the cable endpoint position.
    With MuJoCo's generated cable composite, that rotational weld can be
    absorbed by the soft equality/cable elasticity without visibly rotating
    the first cable segment. A second connect equality, one cable segment
    inward, makes the gripper act like a small tangent handle.
    """
    if not CABLE_ENDPOINT_TANGENT_PIN_ENABLED:
        return

    pin_specs = [
        ("left", "cableB_1"),
        ("right", "cableB_last"),
    ]
    for side, cable_body_name in pin_specs:
        try:
            cable_body = spec.body(cable_body_name)
        except (KeyError, ValueError, AttributeError):
            continue

        cable_site_name = f"cable_{side}_tangent_site"
        anchor_body_name = f"anchor_{side}_tangent"
        anchor_site_name = f"{anchor_body_name}_site"

        cable_body.add_site(
            name=cable_site_name,
            pos=np.zeros(3),
            size=np.array([0.001]),
            rgba=np.array([0.2, 0.7, 1.0, 0.0]),
        )
        anchor_body = spec.worldbody.add_body(
            name=anchor_body_name,
            mocap=True,
            pos=np.array([0.0, 0.0, CABLE_Z]),
        )
        anchor_body.add_site(
            name=anchor_site_name,
            pos=np.zeros(3),
            size=np.array([0.001]),
            rgba=np.array([1.0, 0.8, 0.2, 0.0]),
        )

        eq = spec.add_equality()
        eq.name = f"anchor_{side}_tangent_pin"
        eq.type = mujoco.mjtEq.mjEQ_CONNECT
        eq.objtype = mujoco.mjtObj.mjOBJ_SITE
        eq.name1 = cable_site_name
        eq.name2 = anchor_site_name
        eq.solref = np.array(CABLE_ENDPOINT_TANGENT_SOLREF)


def update_anchor_mocaps(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Drive each anchor mocap body to its UR10e gripper tip's full 6-DOF pose.

    Call this once per simulation step (before ``mj_step``). Both the
    position and orientation are copied so the weld equality between cable
    end and mocap anchor reproduces a rigid gripper hold (no ball-joint
    swinging).

    Mocap bodies are kinematic from the constraint solver's perspective, so
    the weld only acts on the cable side — no impulse is applied to the
    robot DOFs."""
    for side in ("left", "right"):
        anchor_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"anchor_{side}"
        )
        tip_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"ur10_{side}/gripper_tip"
        )
        if anchor_bid < 0 or tip_bid < 0:
            raise RuntimeError(f"missing body anchor_{side} or gripper_tip")
        mocap_id = model.body_mocapid[anchor_bid]
        if mocap_id < 0:
            raise RuntimeError(f"anchor_{side} is not a mocap body")
        data.mocap_pos[mocap_id] = data.xpos[tip_bid]
        data.mocap_quat[mocap_id] = data.xquat[tip_bid]

        clamp_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"anchor_{side}_clamp"
        )
        if clamp_bid >= 0:
            clamp_mocap_id = model.body_mocapid[clamp_bid]
            if clamp_mocap_id < 0:
                raise RuntimeError(f"anchor_{side}_clamp is not a mocap body")

            nodes = _ordered_cable_nodes(model, data)
            segment_length = (
                _cable_segment_length(model, data, nodes)
                if len(nodes) >= 2
                else CABLE_LENGTH / max(CABLE_NUM_SEGMENTS, 1)
            )
            x_axis = data.xmat[tip_bid].reshape(3, 3)[:, 0].copy()
            x_axis[2] = 0.0
            norm = float(np.linalg.norm(x_axis))
            if norm > 1e-9:
                x_axis /= norm
                clamp_pos = data.xpos[tip_bid] + segment_length * x_axis
                clamp_pos[2] = data.xpos[tip_bid, 2]
                data.mocap_pos[clamp_mocap_id] = clamp_pos
                data.mocap_quat[clamp_mocap_id] = data.xquat[tip_bid]

        tangent_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"anchor_{side}_tangent"
        )
        if tangent_bid < 0:
            continue
        tangent_mocap_id = model.body_mocapid[tangent_bid]
        if tangent_mocap_id < 0:
            raise RuntimeError(f"anchor_{side}_tangent is not a mocap body")

        # The endpoint weld pins the cable endpoint position. This tangent
        # mocap pins the next cable point one segment inward along the gripper
        # x-axis, so gripper yaw is transmitted to the cable shape instead of
        # being absorbed by a soft rotational weld residual.
        if side == "left":
            tangent_cable_bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "cableB_1"
            )
            end_cable_bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "cableB_first"
            )
        else:
            tangent_cable_bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "cableB_last"
            )
            end_cable_bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "cableB_last"
            )
        if tangent_cable_bid < 0 or end_cable_bid < 0:
            continue

        body_geom_adr = int(model.body_geomadr[tangent_cable_bid])
        if body_geom_adr >= 0:
            segment_length = 2.0 * float(model.geom_size[body_geom_adr, 1])
        else:
            segment_length = float(
                np.linalg.norm(data.xpos[tangent_cable_bid] - data.xpos[end_cable_bid])
            )
        x_axis = data.xmat[tip_bid].reshape(3, 3)[:, 0].copy()
        x_axis[2] = 0.0
        norm = float(np.linalg.norm(x_axis))
        if norm <= 1e-9:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis /= norm
        tangent_pos = data.xpos[tip_bid] + segment_length * x_axis
        tangent_pos[2] = data.xpos[tip_bid, 2]
        data.mocap_pos[tangent_mocap_id] = tangent_pos
        data.mocap_quat[tangent_mocap_id] = data.xquat[tip_bid]

    if CABLE_ENDPOINT_EXACT_SEGMENT_CLAMP_ENABLED:
        apply_cable_endpoint_exact_segment_clamp(model, data)


def _ordered_cable_nodes(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return MuJoCo cable nodes as body positions plus the final endpoint site."""
    points: list[np.ndarray] = []
    first_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_first")
    if first_bid < 0:
        return np.empty((0, 3), dtype=np.float64)
    points.append(data.xpos[first_bid].copy())

    idx = 1
    while True:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cableB_{idx}")
        if bid < 0:
            break
        points.append(data.xpos[bid].copy())
        idx += 1

    last_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_last")
    if last_bid >= 0:
        points.append(data.xpos[last_bid].copy())
    last_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cableS_last")
    if last_site >= 0:
        points.append(data.site_xpos[last_site].copy())
    return np.asarray(points, dtype=np.float64)


def _cable_segment_length(model: mujoco.MjModel, data: mujoco.MjData, nodes: np.ndarray) -> float:
    first_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_first")
    if first_bid >= 0:
        geom_adr = int(model.body_geomadr[first_bid])
        if geom_adr >= 0:
            return 2.0 * float(model.geom_size[geom_adr, 1])
    if len(nodes) >= 2:
        return float(np.mean(np.linalg.norm(np.diff(nodes[:, :3], axis=0), axis=1)))
    return CABLE_LENGTH / max(CABLE_NUM_SEGMENTS, 1)


def _anchor_clamp_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    anchor_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"anchor_{side}"
    )
    if anchor_bid < 0:
        return None
    mocap_id = model.body_mocapid[anchor_bid]
    if mocap_id < 0:
        return None

    endpoint = data.mocap_pos[mocap_id].copy()
    x_axis = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(
        x_axis,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        data.mocap_quat[mocap_id],
    )
    x_axis[2] = 0.0
    norm = float(np.linalg.norm(x_axis))
    if norm <= 1e-9:
        return None
    return endpoint, x_axis / norm


def apply_cable_endpoint_exact_segment_clamp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Make the first/last cable segment exactly follow anchor yaw.

    The gripper is modeled as a finite clamp on the cable, not a point
    contact. Therefore the cable segment directly under the clamp is a
    boundary condition: rotating the anchor by 45 degrees rotates that segment
    by 45 degrees. MuJoCo's cable elasticity then handles how the bend
    propagates to the remaining free segments.
    """
    nodes = _ordered_cable_nodes(model, data)
    if len(nodes) < 4:
        return
    segment_length = _cable_segment_length(model, data, nodes)

    left_pose = _anchor_clamp_pose(model, data, "left")
    if left_pose is not None:
        endpoint, x_axis = left_pose
        endpoint[2] = nodes[0, 2]
        nodes[0] = endpoint
        nodes[1] = endpoint + segment_length * x_axis
        nodes[1, 2] = nodes[0, 2]

    right_pose = _anchor_clamp_pose(model, data, "right")
    if right_pose is not None:
        endpoint, x_axis = right_pose
        endpoint[2] = nodes[-1, 2]
        nodes[-1] = endpoint
        nodes[-2] = endpoint + segment_length * x_axis
        nodes[-2, 2] = nodes[-1, 2]

    set_cable_config(model, data, nodes)


def apply_cable_endpoint_tangent_assist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    alpha: float = 0.10,
    clamp_segments: int = 3,
    velocity_damping: float = 0.0,
) -> None:
    """Kinematically model each gripper as a short sleeve on the cable.

    This is intentionally not a MuJoCo equality constraint. A hard tangent
    constraint on the first interior cable body can inject large impulses and
    make the cable shake violently. Instead, this helper softly edits a short
    run of cable nodes near each endpoint before the physics step. The endpoint
    receives the strongest correction and the correction tapers toward the
    free cable, approximating a small gripper sleeve without hard constraints.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return
    nodes = _ordered_cable_nodes(model, data)
    if len(nodes) < 4:
        return

    segment_length = _cable_segment_length(model, data, nodes)
    max_clamp_segments = max(1, (len(nodes) - 2) // 2)
    clamp_segments = int(np.clip(int(clamp_segments), 1, max_clamp_segments))

    def _sleeve_weight(i: int) -> float:
        # i=0 is the endpoint; the last controlled point gets a small nudge
        # so the clamp blends into the free cable instead of making a kink.
        return alpha * float(clamp_segments + 1 - i) / float(clamp_segments + 1)

    max_bend_rad = float(np.deg2rad(CABLE_ENDPOINT_TANGENT_MAX_BEND_DEG))

    def _unit_xy(vec: np.ndarray) -> np.ndarray | None:
        v = np.asarray(vec, dtype=np.float64).copy()
        v[2] = 0.0
        norm = float(np.linalg.norm(v))
        if norm <= 1e-9:
            return None
        return v / norm

    def _rotate_xy(unit_vec: np.ndarray, angle: float) -> np.ndarray:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.array(
            [
                c * unit_vec[0] - s * unit_vec[1],
                s * unit_vec[0] + c * unit_vec[1],
                0.0,
            ],
            dtype=np.float64,
        )

    def _bounded_inward_tangent(raw_axis: np.ndarray, inward_dir: np.ndarray) -> np.ndarray | None:
        """Choose the gripper tangent sign and cap the sleeve bend angle.

        The gripper x-axis has a sign ambiguity for our sleeve model. If we
        use the outward sign, the first cable nodes can fold back more than
        90 degrees. We choose the sign that points into the current cable and
        limit the tangent correction relative to that current cable direction.
        """
        tangent = _unit_xy(raw_axis)
        inward = _unit_xy(inward_dir)
        if tangent is None:
            return None
        if inward is None:
            return tangent

        if float(np.dot(tangent[:2], inward[:2])) < 0.0:
            tangent = -tangent

        cross_z = float(inward[0] * tangent[1] - inward[1] * tangent[0])
        dot = float(np.clip(np.dot(inward[:2], tangent[:2]), -1.0, 1.0))
        delta = float(np.arctan2(cross_z, dot))
        if abs(delta) <= max_bend_rad:
            return tangent
        return _rotate_xy(inward, np.sign(delta) * max_bend_rad)

    for side in ("left", "right"):
        tip_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"ur10_{side}/gripper_tip"
        )
        if tip_bid < 0:
            continue
        raw_x_axis = data.xmat[tip_bid].reshape(3, 3)[:, 0].copy()

        endpoint = data.xpos[tip_bid].copy()
        if side == "left":
            reference_idx = min(clamp_segments + 1, len(nodes) - 1)
            x_axis = _bounded_inward_tangent(raw_x_axis, nodes[reference_idx] - nodes[0])
            if x_axis is None:
                continue
            for i in range(clamp_segments + 1):
                desired = endpoint + (i * segment_length) * x_axis
                desired[2] = nodes[i, 2]
                w = _sleeve_weight(i)
                nodes[i] = (1.0 - w) * nodes[i] + w * desired
        else:
            reference_idx = max(0, len(nodes) - clamp_segments - 2)
            x_axis = _bounded_inward_tangent(raw_x_axis, nodes[reference_idx] - nodes[-1])
            if x_axis is None:
                continue
            for i in range(clamp_segments + 1):
                idx = -1 - i
                desired = endpoint + (i * segment_length) * x_axis
                desired[2] = nodes[idx, 2]
                w = _sleeve_weight(i)
                nodes[idx] = (1.0 - w) * nodes[idx] + w * desired

    set_cable_config(model, data, nodes)
    damping = float(np.clip(velocity_damping, 0.0, 1.0))
    if damping > 0.0:
        idx = 0
        while True:
            if idx == 0:
                bname = "cableB_first"
            else:
                bname = f"cableB_{idx}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_last")
                if bid < 0:
                    break
                last = True
            else:
                last = False
            jid = int(model.body_jntadr[bid])
            dof_adr = int(model.jnt_dofadr[jid])
            dof_num = 6 if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 3
            data.qvel[dof_adr:dof_adr + dof_num] *= (1.0 - damping)
            if last:
                break
            idx += 1


def set_cable_config(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nodes_3d: np.ndarray,
) -> None:
    """Set cable qpos so cable body i is centered at nodes_3d[i].

    nodes_3d is an (N, 3) array of cable node positions. MuJoCo's cable
    composite creates N-1 dynamic bodies for N nodes, named
    ``cableB_first``, optional middle bodies, and ``cableB_last``.

    The free joint (body 0) gets an absolute position + z-rotation quaternion.
    Each ball joint (bodies 1..) gets a relative z-rotation from the previous
    segment direction.  Z-bending is ignored (cable is nearly horizontal).
    """
    nodes_3d = np.asarray(nodes_3d, dtype=np.float64)
    if nodes_3d.ndim != 2 or nodes_3d.shape[1] != 3 or len(nodes_3d) < 2:
        raise ValueError("nodes_3d must be an (N, 3) array with N >= 2")
    N_BODIES = len(nodes_3d) - 1

    qadr = np.empty(N_BODIES, dtype=np.int64)
    for i in range(N_BODIES):
        if i == 0:
            bname = "cableB_first"
        elif i == N_BODIES - 1:
            bname = "cableB_last"
        else:
            bname = f"cableB_{i}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            raise RuntimeError(f"missing cable body {bname}; scene/node count mismatch")
        jid = model.body_jntadr[bid]
        qadr[i] = model.jnt_qposadr[jid]

    dirs = np.empty((N_BODIES, 2))
    for i in range(N_BODIES):
        nxt = i + 1 if i < len(nodes_3d) - 1 else i - 1
        d = nodes_3d[nxt, :2] - nodes_3d[i, :2] if i < len(nodes_3d) - 1 \
            else nodes_3d[i, :2] - nodes_3d[i - 1, :2]
        norm = float(np.linalg.norm(d))
        dirs[i] = d / norm if norm > 1e-9 else np.array([1.0, 0.0])

    theta0 = float(np.arctan2(dirs[0, 1], dirs[0, 0]))
    qa = int(qadr[0])
    data.qpos[qa:qa + 3] = nodes_3d[0]
    data.qpos[qa + 3] = float(np.cos(theta0 / 2))
    data.qpos[qa + 4] = 0.0
    data.qpos[qa + 5] = 0.0
    data.qpos[qa + 6] = float(np.sin(theta0 / 2))

    prev_theta = theta0
    for i in range(1, N_BODIES):
        theta_i = float(np.arctan2(dirs[i, 1], dirs[i, 0]))
        dtheta = float(((theta_i - prev_theta) + np.pi) % (2 * np.pi) - np.pi)
        qa = int(qadr[i])
        data.qpos[qa] = float(np.cos(dtheta / 2))
        data.qpos[qa + 1] = 0.0
        data.qpos[qa + 2] = 0.0
        data.qpos[qa + 3] = float(np.sin(dtheta / 2))
        prev_theta = theta_i

    mujoco.mj_forward(model, data)


def _set_weld_relpose_to_current_state(
    model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    """After IK is solved and the cable is in its initial straight pose,
    rewrite each weld equality's relpose so the constraint is satisfied
    *as-is* — i.e. the weld preserves the current relative orientation
    between cable_first/last and the mocap anchor. Without this, activating
    the weld at t=0 would jerk the cable to align identically with the
    gripper, which can cause the cable composite to snap unrealistically.

    The weld then enforces a *fixed relative pose* between cable end and
    mocap.  This means the cable end rotates **1-to-1 in world frame**
    with the gripper (Δgripper_world_quat == Δcable_world_quat), but with
    a constant rotational offset baked in at t=0.  That offset is the
    natural difference between the cable's local frame (z-up, x along
    segment direction) and the gripper's local frame (z-down, x along
    tool axis) — the cable still tracks gripper yaw, just in its own
    coordinate system.

    eq_data layout for mjEQ_WELD (length 11):
      [0:3]   anchor on body2 (we use body origin, all zeros)
      [3:6]   relpos (translation of body2 expressed in body1 frame)
      [6:10]  relquat (quaternion of body2 in body1 frame, [w x y z])
      [10]    torquescale
    """
    mujoco.mj_forward(model, data)
    _align_endpoint_site_welds_to_current_state(model, data)
    mujoco.mj_forward(model, data)
    pairs = [
        ("anchor_left_pin", "cableB_first", "anchor_left"),
        ("anchor_right_pin", "cableB_last", "anchor_right"),
    ]
    for eq_name, b1_name, b2_name in pairs:
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        if eid < 0:
            continue
        if model.eq_type[eid] != mujoco.mjtEq.mjEQ_WELD:
            continue
        b1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b1_name)
        b2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b2_name)
        if b1 < 0 or b2 < 0:
            raise RuntimeError(f"missing equality/body for {eq_name}")
        # World-frame poses of the two bodies.
        p1 = data.xpos[b1].copy()
        p2 = data.xpos[b2].copy()
        q1 = data.xquat[b1].copy()
        q2 = data.xquat[b2].copy()
        # rel = pose of body2 in body1's frame.
        # quat: q1^-1 * q2  (current relative orientation, baked in at t=0)
        q1_inv = np.empty(4)
        mujoco.mju_negQuat(q1_inv, q1)
        relquat = np.empty(4)
        mujoco.mju_mulQuat(relquat, q1_inv, q2)
        # pos: rotate (p2 - p1) by q1^-1
        relpos = np.empty(3)
        diff = p2 - p1
        mujoco.mju_rotVecQuat(relpos, diff, q1_inv)

        eq_data = np.zeros(11)
        eq_data[0:3] = 0.0       # anchor on body2 (origin)
        eq_data[3:6] = relpos
        eq_data[6:10] = relquat
        eq_data[10] = CABLE_ENDPOINT_WELD_TORQUESCALE
        model.eq_data[eid] = eq_data


def _align_endpoint_site_welds_to_current_state(
    model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    """Remove startup residual from site-weld cable endpoint constraints.

    The selected GCR waypoint CSV can initialize the cable in a curved shape.
    Site welds are correct for the gripper/cable boundary, but their anchor
    sites must be initialized to match the already-curved cable endpoint site
    frames. Otherwise a stiff weld closes the initial frame mismatch in the
    first few solver steps, which looks like the cable jumping from the
    planned start shape.
    """
    site_pairs = [
        ("cableS_first", "anchor_left", "anchor_left_weld_site"),
        ("cableS_last", "anchor_right", "anchor_right_weld_site"),
    ]
    for cable_site_name, anchor_body_name, anchor_site_name in site_pairs:
        cable_sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, cable_site_name
        )
        anchor_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name
        )
        anchor_sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, anchor_site_name
        )
        if cable_sid < 0 or anchor_bid < 0 or anchor_sid < 0:
            continue

        p_anchor = data.xpos[anchor_bid].copy()
        q_anchor = data.xquat[anchor_bid].copy()
        q_anchor_inv = np.empty(4)
        mujoco.mju_negQuat(q_anchor_inv, q_anchor)

        relpos = np.empty(3)
        mujoco.mju_rotVecQuat(relpos, data.site_xpos[cable_sid] - p_anchor, q_anchor_inv)

        cable_quat = np.empty(4)
        mujoco.mju_mat2Quat(cable_quat, data.site_xmat[cable_sid].reshape(9))
        relquat = np.empty(4)
        mujoco.mju_mulQuat(relquat, q_anchor_inv, cable_quat)

        model.site_pos[anchor_sid] = relpos
        model.site_quat[anchor_sid] = relquat


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_two_ur10_cable_spec(
    obstacles: list | None = None,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    cable_length: float = CABLE_LENGTH,
    cable_num_segments: int = CABLE_NUM_SEGMENTS,
    ghost_cable_profiles: list[np.ndarray] | None = None,
    ghost_cable_rgba: str = "1.0 0.86 0.18 0.35",
) -> mujoco.MjSpec:
    """Compose the full two-UR10e + cable + obstacles scene as an ``MjSpec``.

    Two prefixed copies of the menagerie UR10e are attached to the base
    scene; a ``gripper_tip`` child body is added to each robot's wrist_3_link;
    the cable endpoints are pinned to those tips by weld equalities. The
    caller is expected to run ``solve_initial_pose`` (or ``build_and_compile``,
    which does it automatically) so the tips coincide with the cable
    endpoints at t=0.

    obstacles: optional list of dicts (same format as OBSTACLES) to override
               the hardcoded obstacle set.  Pass ``[]`` for an empty scene.
    workspace_bounds: optional ((xmin, xmax), (ymin, ymax)) — when provided,
               draws a thin red rectangle outlining the planner's xy bounds.
    """
    spec = mujoco.MjSpec.from_string(
        _base_scene_xml(
            obstacles,
            workspace_bounds,
            cable_length=cable_length,
            cable_num_segments=cable_num_segments,
            ghost_cable_profiles=ghost_cable_profiles,
            ghost_cable_rgba=ghost_cable_rgba,
        )
    )

    ur10_path = _ur10_xml_path()
    ur10_left = mujoco.MjSpec.from_file(str(ur10_path))
    ur10_right = mujoco.MjSpec.from_file(str(ur10_path))

    left_frame = spec.worldbody.add_frame(
        pos=np.array(UR10_LEFT_BASE_POS),
        quat=np.array(UR10_LEFT_BASE_QUAT),
    )
    left_frame.attach_body(ur10_left.worldbody.first_body(), "ur10_left/", "")

    right_frame = spec.worldbody.add_frame(
        pos=np.array(UR10_RIGHT_BASE_POS),
        quat=np.array(UR10_RIGHT_BASE_QUAT),
    )
    right_frame.attach_body(ur10_right.worldbody.first_body(), "ur10_right/", "")

    _add_gripper_tips(spec)
    _add_cable_endpoint_site_welds(spec)
    _add_cable_endpoint_two_point_clamps(spec)
    _add_cable_endpoint_tangent_pins(spec)
    _enable_robot_gravcomp(spec)

    return spec


# ---------------------------------------------------------------------------
# Damped-least-squares IK
# ---------------------------------------------------------------------------
def _ur10_joint_indices(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (qposadr, dofadr) arrays for the 6 hinge joints of one UR10e."""
    qadr = np.empty(N_ROBOT_JOINTS, dtype=np.int64)
    dadr = np.empty(N_ROBOT_JOINTS, dtype=np.int64)
    for j, jname in enumerate(_UR10_JOINT_NAMES):
        full_name = f"ur10_{side}/{jname}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
        if jid < 0:
            raise RuntimeError(f"missing joint {full_name}")
        qadr[j] = model.jnt_qposadr[jid]
        dadr[j] = model.jnt_dofadr[jid]
    return qadr, dadr


def solve_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    targets: dict[str, np.ndarray] | None = None,
    max_iter: int = 500,
    pos_tol: float = 5e-4,
) -> dict:
    """Pose both UR10e robots so each gripper tip starts at the given targets.

    targets: optional dict with "left" and "right" 3-D positions. If None,
        falls back to reading cableB_first / cableB_last body positions.
    """
    mujoco.mj_forward(model, data)
    if targets is None:
        left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cableS_first")
        right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cableS_last")
        targets = {
            "left": (
                data.site_xpos[left_site].copy()
                if left_site >= 0
                else data.xpos[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_first")
                ].copy()
            ),
            "right": (
                data.site_xpos[right_site].copy()
                if right_site >= 0
                else data.xpos[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_last")
                ].copy()
            ),
        }

    report: dict = {"converged": {}, "iters": {}, "final_err_mm": {}, "targets": targets}
    _z_down = np.array([0.0, 0.0, -1.0])

    # Multiple physically distinct elbow-up seeds.  The user-specified
    # config works for the left arm and for cable poses near the centre,
    # but for high-y Cstart configurations on the right arm the same seed
    # can land in a backward-reaching IK branch (visible in the image as
    # the right arm folded over its shoulder).  Trying the user seed AND
    # its shoulder-pan-mirrored variant AND the UR10e canonical home pose
    # ensures the IK finds an elbow-up forward-reaching solution
    # regardless of the target's quadrant.
    _seed_user = np.deg2rad([
        155.810869, -60.525940, 155.774429, -185.248488, -90.000000, -114.189131
    ])
    _seed_pan_neg = _seed_user.copy()
    _seed_pan_neg[0] = -_seed_pan_neg[0]  # negate shoulder_pan
    _seed_home = np.deg2rad([-90.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    _seed_home_mirror = np.deg2rad([90.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    _seed_elbow_up_a = np.deg2rad([150.0, -60.0, 120.0, -150.0, -90.0, -114.0])
    _seed_elbow_up_b = np.deg2rad([-150.0, -60.0, 120.0, -150.0, -90.0, -114.0])
    _SEED_LIST = [
        _seed_user,
        _seed_pan_neg,
        _seed_home,
        _seed_home_mirror,
        _seed_elbow_up_a,
        _seed_elbow_up_b,
    ]
    _SEEDS = {"left": _SEED_LIST, "right": _SEED_LIST}
    _posture_reference = _seed_user.copy()

    def _angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi

    def _posture_cost(q: np.ndarray) -> float:
        elbow = abs(float(q[2]))
        shoulder_lift = abs(float(q[1]))
        elbow_straight = max(0.0, np.deg2rad(35.0) - elbow)
        shoulder_folded = max(0.0, shoulder_lift - np.deg2rad(150.0))
        wrist_regularization = float(np.dot(_angle_delta(q[3:], _posture_reference[3:]), _angle_delta(q[3:], _posture_reference[3:])))
        joint_regularization = float(np.dot(_angle_delta(q, _posture_reference), _angle_delta(q, _posture_reference)))
        return (
            200.0 * elbow_straight * elbow_straight
            + 40.0 * shoulder_folded * shoulder_folded
            + 0.02 * wrist_regularization
            + 0.002 * joint_regularization
        )

    for side, target in targets.items():
        qadr, _ = _ur10_joint_indices(model, side)
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"ur10_{side}/gripper_tip"
        )
        if body_id < 0:
            raise RuntimeError(f"ur10_{side}/gripper_tip body missing")

        joint_lo = np.array(
            [model.jnt_range[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ur10_{side}/{jname}")
            ][0] for jname in _UR10_JOINT_NAMES]
        )
        joint_hi = np.array(
            [model.jnt_range[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ur10_{side}/{jname}")
            ][1] for jname in _UR10_JOINT_NAMES]
        )
        bounds = [(joint_lo[j], joint_hi[j]) for j in range(N_ROBOT_JOINTS)]

        def _ik_cost(q, _qadr=qadr, _bid=body_id, _tgt=target):
            data.qpos[_qadr] = q
            mujoco.mj_forward(model, data)
            e_pos = _tgt - data.xpos[_bid]
            R = data.xmat[_bid].reshape(3, 3)
            # Direct-difference orientation error avoids the anti-parallel
            # singularity of the cross-product formula.
            e_ori = 0.5 * (R[:, 2] - _z_down)
            return 10000.0 * float(np.dot(e_pos, e_pos)) + 20.0 * float(np.dot(e_ori, e_ori))

        # L-BFGS-B is a local optimiser, so try several physically distinct
        # elbow-up seeds and choose a branch that reaches the target while
        # avoiding straight-elbow / folded-over poses that visually collide.
        seeds: list[np.ndarray] = list(_SEEDS[side])
        best_result = None
        best_score = np.inf
        for s in seeds:
            res = _sp_minimize(
                _ik_cost, s, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
            )
            score = float(res.fun) + _posture_cost(np.asarray(res.x, dtype=float))
            if best_result is None or score < best_score:
                best_result = res
                best_score = score

        data.qpos[qadr] = np.clip(best_result.x, joint_lo, joint_hi)
        mujoco.mj_forward(model, data)
        last_err = float(np.linalg.norm(target - data.xpos[body_id]))
        R = data.xmat[body_id].reshape(3, 3)
        ori_err = float(np.linalg.norm(0.5 * (R[:, 2] - _z_down)))
        converged = last_err < pos_tol and ori_err < 0.10

        report["converged"][side] = converged
        report["iters"][side] = len(seeds)
        report["final_err_mm"][side] = last_err * 1000.0

    # UR10e actuators are position-controlled (gainprm=5000, biasprm with
    # affine bias = -5000*qpos - 500*qvel). With ctrl=0 and qpos != 0, they
    # would yank each joint back to zero, destabilizing the cable. Set ctrl
    # to match the IK-found qpos so the actuators hold the IK pose.
    # Also pre-seed qpos_spring so DualUR10Controller's blending starts from
    # the IK pose (not zero), avoiding a ctrl step-change at t=0.
    for side in ("left", "right"):
        qadr, _ = _ur10_joint_indices(model, side)
        for j, aname in enumerate(_UR10_ACTUATOR_NAMES):
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ur10_{side}/{aname}"
            )
            if aid >= 0:
                data.ctrl[aid] = data.qpos[qadr[j]]
            model.qpos_spring[qadr[j]] = data.qpos[qadr[j]]

    # Final forward kinematics so callers see consistent xpos.
    mujoco.mj_forward(model, data)
    return report


def _align_cable_endpoints_to_grippers(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> None:
    """Force ``cableB_first.world_quat = ur10_left/gripper_tip.world_quat``
    and ``cableB_last.world_quat = ur10_right/gripper_tip.world_quat``.

    Without this, ``set_cable_config`` orients each cable body purely from
    the GCR Cstart segment direction (yaw-only, local +z = world +z), but
    the gripper ends up at a 3-D rotation (local +z = world −z, rotated
    around its tool axis by the IK).  The cable's first segment direction
    therefore doesn't match the gripper's tool direction, and the
    rigid-orientation weld (relquat = identity) is fighting an existing
    mismatch.  Overriding the cable endpoints' world quats here makes
    ``cable.world_quat == gripper.world_quat`` at t=0, so the weld has
    zero residual and the cable's first/last segment rigidly tracks the
    gripper yaw.

    The cable bodies between ``cableB_first`` and ``cableB_last`` keep
    their ``set_cable_config`` ball-joint qposes — so the *relative* shape
    of the cable is preserved, only the whole chain is rotated to match
    the grippers' frames.
    """
    L_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ur10_left/gripper_tip")
    R_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ur10_right/gripper_tip")
    cb_first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_first")
    cb_last  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cableB_last")
    if min(L_tip, R_tip, cb_first, cb_last) < 0:
        return  # nothing to do — bodies missing (defensive)

    mujoco.mj_forward(model, data)
    q_grip_L = data.xquat[L_tip].copy()
    q_grip_R = data.xquat[R_tip].copy()

    # cableB_first has a free joint (qpos = [x, y, z, qw, qx, qy, qz]).
    jid_first = model.body_jntadr[cb_first]
    qadr_first = int(model.jnt_qposadr[jid_first])
    data.qpos[qadr_first + 3:qadr_first + 7] = q_grip_L

    # Re-run forward kinematics so cableB_last's PARENT world quat reflects
    # the new chain orientation before we set the last ball-joint qpos.
    mujoco.mj_forward(model, data)

    # cableB_last has a ball joint (qpos = 4 quat values, expressed in the
    # parent body's frame).  We want cable_last.world_quat = q_grip_R, so
    # the local (= parent-frame) quat is parent_world_quat^-1 * q_grip_R.
    parent_id = int(model.body_parentid[cb_last])
    q_parent = data.xquat[parent_id].copy()
    q_parent_inv = np.empty(4)
    mujoco.mju_negQuat(q_parent_inv, q_parent)
    local_q_last = np.empty(4)
    mujoco.mju_mulQuat(local_q_last, q_parent_inv, q_grip_R)

    jid_last = model.body_jntadr[cb_last]
    qadr_last = int(model.jnt_qposadr[jid_last])
    data.qpos[qadr_last:qadr_last + 4] = local_q_last
    mujoco.mj_forward(model, data)


_UR10_JOINT_STIFFNESS = 5000.0  # N*m/rad — enough to resist gravity, low enough to avoid cable-weld resonance
_UR10_JOINT_DAMPING = 200.0     # N*m*s/rad


def _add_ur10_joint_locks(spec: mujoco.MjSpec, ik_qpos: dict[str, np.ndarray]) -> None:
    """Make each UR10e joint passively spring back to its IK angle.

    Implementation: set ``stiffness`` and ``springref`` on each joint plus
    extra ``damping``. This is purely additive to the menagerie UR10e's
    actuator-based PD controller, and unlike joint-equality constraints it
    does not couple into the equality solver — so it composes cleanly with
    the cable-to-gripper weld equalities."""
    for side in ("left", "right"):
        for j, jname in enumerate(_UR10_JOINT_NAMES):
            joint = spec.joint(f"ur10_{side}/{jname}")
            # MjsJoint properties accept (3,) arrays even for hinge joints
            # (only index 0 is read for hinges).
            joint.stiffness = np.array([_UR10_JOINT_STIFFNESS, 0.0, 0.0])
            joint.springref = float(ik_qpos[side][j])
            # Extra damping on top of whatever the menagerie default provides.
            current_damping = float(np.atleast_1d(joint.damping)[0])
            joint.damping = np.array([max(current_damping, _UR10_JOINT_DAMPING), 0.0, 0.0])


def build_and_compile(
    *, ik: bool = True, lock_robot_joints: bool = True,
    obstacles: list | None = None,
    ik_targets: dict[str, np.ndarray] | None = None,
    cable_cstart: np.ndarray | None = None,
    cable_length: float | None = None,
    cable_num_segments: int | None = None,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ghost_cable_profiles: list[np.ndarray] | None = None,
    ghost_cable_rgba: str = "1.0 0.86 0.18 0.35",
) -> tuple[mujoco.MjModel, mujoco.MjData, mujoco.MjSpec]:
    """Compose, compile, and (by default) run IK so the grippers start at the
    cable endpoints, with joint locks holding each UR10e rigidly in pose.

    Args:
        ik: run inverse kinematics so each gripper tip starts at the cable
            endpoint. Skip with ``ik=False`` for debugging.
        lock_robot_joints: add joint stiffness/springref so each UR10e joint
            passively springs back to its IK value, so the cable hangs stably
            without a controller running. Set ``False`` once you have a real
            controller (DualUR10Controller).
        ik_targets: optional dict with "left" and "right" 3-D positions to
            override the default IK targets (cable body positions). Used when
            the robots should initialize at the scenario's Cstart endpoints
            rather than at the default straight cable endpoints.
        cable_cstart: optional (N, 3) array of 3-D node positions for the
            cable's initial shape. When provided, the cable qpos is set to
            this configuration before IK runs, so ``cableB_first.xpos =
            nodes_3d[0]`` and ``cableB_last.xpos = nodes_3d[-2]``. Combined
            with ``ik_targets`` pointing at the same Cstart endpoints, the
            cable-to-gripper welds have zero residual at t=0."""
    if not ik and lock_robot_joints:
        raise ValueError("lock_robot_joints=True requires ik=True")

    if cable_cstart is not None:
        cable_cstart = np.asarray(cable_cstart, dtype=np.float64)
        if cable_num_segments is None:
            cable_num_segments = len(cable_cstart) - 1
        if cable_length is None:
            cable_length = float(np.sum(np.linalg.norm(np.diff(cable_cstart[:, :3], axis=0), axis=1)))
    if cable_num_segments is None:
        cable_num_segments = CABLE_NUM_SEGMENTS
    if cable_length is None:
        cable_length = CABLE_LENGTH

    # First pass: compile, run IK, capture target joint angles.
    spec = build_two_ur10_cable_spec(
        obstacles=obstacles,
        workspace_bounds=workspace_bounds,
        cable_length=cable_length,
        cable_num_segments=cable_num_segments,
        ghost_cable_profiles=ghost_cable_profiles,
        ghost_cable_rgba=ghost_cable_rgba,
    )
    if not ik:
        model = spec.compile()
        data = mujoco.MjData(model)
        if cable_cstart is not None:
            set_cable_config(model, data, cable_cstart)
        return model, data, spec

    tmp_model = spec.compile()
    tmp_data = mujoco.MjData(tmp_model)
    if cable_cstart is not None:
        set_cable_config(tmp_model, tmp_data, cable_cstart)
    report = solve_initial_pose(tmp_model, tmp_data, targets=ik_targets)
    for side in ("left", "right"):
        print(
            f"[ik] ur10_{side}: converged={report['converged'][side]} "
            f"iters={report['iters'][side]} "
            f"final_err={report['final_err_mm'][side]:.2f} mm"
        )
    ik_qpos = {
        side: tmp_data.qpos[_ur10_joint_indices(tmp_model, side)[0]].copy()
        for side in ("left", "right")
    }

    if not lock_robot_joints:
        # IK pose is already in tmp_data (qpos + ctrl + qpos_spring set by
        # solve_initial_pose).  Still need to align the mocap anchors and
        # freeze the weld relpose so the cable sees zero constraint residual
        # at t=0.
        update_anchor_mocaps(tmp_model, tmp_data)
        mujoco.mj_forward(tmp_model, tmp_data)
        _set_weld_relpose_to_current_state(tmp_model, tmp_data)
        mujoco.mj_forward(tmp_model, tmp_data)
        return tmp_model, tmp_data, spec

    # Second pass: rebuild spec with joint locks at the IK angles.
    spec = build_two_ur10_cable_spec(
        obstacles=obstacles,
        workspace_bounds=workspace_bounds,
        cable_length=cable_length,
        cable_num_segments=cable_num_segments,
        ghost_cable_profiles=ghost_cable_profiles,
        ghost_cable_rgba=ghost_cable_rgba,
    )
    _add_ur10_joint_locks(spec, ik_qpos)
    model = spec.compile()
    data = mujoco.MjData(model)
    if cable_cstart is not None:
        set_cable_config(model, data, cable_cstart)

    # Initialize qpos and ctrl at the IK pose so the joint locks have nothing
    # to fight at t=0.
    for side in ("left", "right"):
        qadr, _ = _ur10_joint_indices(model, side)
        for j, aname in enumerate(_UR10_ACTUATOR_NAMES):
            data.qpos[qadr[j]] = ik_qpos[side][j]
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ur10_{side}/{aname}"
            )
            if aid >= 0:
                data.ctrl[aid] = ik_qpos[side][j]
    mujoco.mj_forward(model, data)
    # Align the mocap anchors and freeze the weld relpose so the cable sees
    # zero constraint residual at t=0.
    update_anchor_mocaps(model, data)
    mujoco.mj_forward(model, data)
    _set_weld_relpose_to_current_state(model, data)
    mujoco.mj_forward(model, data)
    return model, data, spec


def save_scene_xml(out_path: Path) -> Path:
    spec = build_two_ur10_cable_spec()
    spec.compile()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(spec.to_xml())
    return out_path


if __name__ == "__main__":
    out = SCENE_DIR / "_compiled_two_ur10_cable.xml"
    save_scene_xml(out)
    print(f"Compiled scene written to {out}")
