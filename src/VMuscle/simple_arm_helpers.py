"""Shared helpers for SimpleArm examples.

Extracts duplicated code from the 5 SimpleArm examples into a single module:
- build_mjcf: MuJoCo XML generation
- build_attach_constraints: XPBD ATTACH constraint construction
- compute_excitation: Hermite smoothstep excitation schedule (matches OpenSim)
- write_sto: OpenSim .sto file writer
"""

import textwrap

import numpy as np

from VMuscle.constraints import ATTACH


# ---------------------------------------------------------------------------
# MuJoCo model
# ---------------------------------------------------------------------------

def build_mjcf(cfg):
    """Generate MJCF XML for the SimpleArm model.

    Args:
        cfg: Config dict with cfg["geometry"] containing humerus_length,
             radius_length, muscle_origin_on_humerus, muscle_insertion_on_radius.
    """
    geo = cfg["geometry"]
    L_h = geo["humerus_length"]
    L_r = geo["radius_length"]
    mo = geo["muscle_origin_on_humerus"]
    mi = geo["muscle_insertion_on_radius"]
    ox, oy, oz = mo[0], -(L_h - mo[1]), mo[2] if len(mo) > 2 else 0
    ix, iy, iz = mi[0], -(L_r - mi[1]), mi[2] if len(mi) > 2 else 0

    return textwrap.dedent(f"""\
    <?xml version="1.0" ?>
    <mujoco model="simple_arm">
      <option timestep="0.002" gravity="0 -9.81 0"
              integrator="implicit" solver="Newton"
              iterations="50" tolerance="1e-10"/>
      <compiler angle="radian"/>

      <worldbody>
        <body name="humerus" pos="0 0 0">
          <geom type="capsule" size="0.04" fromto="0 0 0 0 {-L_h} 0"
                rgba="0.7 0.7 0.7 0.8" mass="0"
                contype="0" conaffinity="0"/>
          <site name="muscle_origin" pos="{ox} {oy} {oz}" size="0.015"
                rgba="1 0 0 1"/>

          <body name="radius" pos="0 {-L_h} 0">
            <joint name="elbow" type="hinge" axis="0 0 1"
                   limited="true" range="0 3.14159"
                   damping="0" armature="0"/>
            <inertial pos="0 {-L_r} 0" mass="1"
                      diaginertia="0.001 0.001 0.001"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0 {-L_r} 0"
                  rgba="0.5 0.5 0.8 0.8" mass="0"
                  contype="0" conaffinity="0"/>
            <site name="muscle_insertion" pos="{ix} {iy} {iz}" size="0.015"
                  rgba="0 0 1 1"/>
          </body>
        </body>
      </worldbody>

      <tendon>
        <spatial name="biceps_tendon" stiffness="0" damping="0"
                 width="0.008" rgba="0.8 0.2 0.2 1">
          <site site="muscle_origin"/>
          <site site="muscle_insertion"/>
        </spatial>
      </tendon>

      <actuator>
        <motor name="biceps_motor" tendon="biceps_tendon" gear="-1"/>
      </actuator>
    </mujoco>""")


# ---------------------------------------------------------------------------
# XPBD ATTACH constraints
# ---------------------------------------------------------------------------

def build_attach_constraints(vertices, origin_ids, insertion_ids,
                             bone_targets_np, tets,
                             origin_stiffness, insertion_stiffness):
    """Build ATTACH constraint dicts for origin and insertion vertices.

    Returns a list of constraint dicts ready for MuscleSim.build_constraints().
    """
    pt2tet = {}
    for i, t in enumerate(tets):
        for vi in t:
            pt2tet.setdefault(int(vi), int(i))

    n_origin = len(origin_ids)
    attach_cons = []

    for j, vid in enumerate(origin_ids):
        bone_idx = j
        tgt = bone_targets_np[bone_idx]
        dist = float(np.linalg.norm(vertices[vid] - tgt))
        attach_cons.append(dict(
            type=ATTACH, pts=[int(vid), -1, int(bone_idx), -1],
            stiffness=origin_stiffness, dampingratio=0.0,
            tetid=pt2tet.get(int(vid), -1), L=[0.0, 0.0, 0.0],
            restlength=dist,
            restvector=[float(tgt[0]), float(tgt[1]), float(tgt[2]), 1.0],
            restdir=[0.0, 0.0, 0.0], compressionstiffness=-1.0))

    for j, vid in enumerate(insertion_ids):
        bone_idx = n_origin + j
        tgt = bone_targets_np[bone_idx]
        dist = float(np.linalg.norm(vertices[vid] - tgt))
        attach_cons.append(dict(
            type=ATTACH, pts=[int(vid), -1, int(bone_idx), -1],
            stiffness=insertion_stiffness, dampingratio=0.0,
            tetid=pt2tet.get(int(vid), -1), L=[0.0, 0.0, 0.0],
            restlength=dist,
            restvector=[float(tgt[0]), float(tgt[1]), float(tgt[2]), 1.0],
            restdir=[0.0, 0.0, 0.0], compressionstiffness=-1.0))

    return attach_cons


# ---------------------------------------------------------------------------
# Excitation schedule
# ---------------------------------------------------------------------------

def compute_excitation(t, act_cfg):
    """Compute excitation at time t using Hermite smoothstep.

    Matches OpenSim StepFunction(start, end, off, on):
    - t < start: excitation_off
    - start <= t <= end: smooth Hermite ramp from off to on
    - t > end: excitation_on
    """
    t_start = act_cfg["excitation_start_time"]
    t_end = act_cfg["excitation_end_time"]
    e_off = act_cfg["excitation_off"]
    e_on = act_cfg["excitation_on"]
    if t < t_start:
        return e_off
    elif t >= t_end:
        return e_on
    else:
        frac = (t - t_start) / (t_end - t_start)
        frac = frac * frac * (3.0 - 2.0 * frac)  # 3t^2 - 2t^3
        return e_off + (e_on - e_off) * frac


# ---------------------------------------------------------------------------
# STO file output
# ---------------------------------------------------------------------------

def write_sto(path, header_name, columns, times, data, in_degrees=False):
    """Write an OpenSim .sto file.

    Args:
        path: Output file path.
        header_name: Name written as the first line of the header.
        columns: List of column names (excluding 'time').
        times: (N,) sequence of time values.
        data: List of (N,) sequences, one per column, in same order as columns.
        in_degrees: Whether angle values are in degrees.
    """
    n_rows = len(times)
    with open(path, "w") as f:
        f.write(f"{header_name}\n")
        f.write(f"inDegrees={'yes' if in_degrees else 'no'}\n")
        f.write(f"nColumns={len(columns) + 1}\n")
        f.write(f"nRows={n_rows}\n")
        f.write("DataType=double\n")
        f.write("version=3\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(columns) + "\n")
        for i in range(n_rows):
            vals = "\t".join(str(data[c][i]) for c in range(len(columns)))
            f.write(f"{times[i]}\t{vals}\n")
