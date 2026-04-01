"""MJCF builder for the SimpleArm MuJoCo model.

Provides build_mjcf(cfg) which generates the MJCF XML string for a 2-body
arm (humerus fixed, radius on elbow hinge) used in SimpleArm comparison
experiments.
"""

import textwrap


def build_mjcf(cfg):
    """Generate MJCF XML for the simple arm model.

    Geometry matches the OpenSim model exactly:
    - Humerus: fixed to world, extends L_h downward from origin
    - Radius: connected to humerus bottom via elbow hinge (Z-axis)
    - Spatial tendon: muscle_origin (on humerus) -> muscle_insertion (on radius)
    - Motor actuator on tendon for external force injection

    Coordinate convention: Y-up, gravity = (0, -9.81, 0).

    OpenSim-to-MuJoCo body frame mapping:
    - OpenSim: body origin at bottom (joint_in_child = (0, L, 0))
    - MuJoCo: body origin at top (joint at body frame origin)
    - OpenSim (0, 0.8, 0) on humerus -> MuJoCo (0, -0.2, 0)
    - OpenSim (0, 0.7, 0) on radius  -> MuJoCo (0, -0.3, 0)
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
        <!-- Humerus: fixed to world (no joint). Body origin = shoulder. -->
        <body name="humerus" pos="0 0 0">
          <geom type="capsule" size="0.04" fromto="0 0 0 0 {-L_h} 0"
                rgba="0.7 0.7 0.7 0.8" mass="0"
                contype="0" conaffinity="0"/>
          <site name="muscle_origin" pos="{ox} {oy} {oz}" size="0.015"
                rgba="1 0 0 1"/>

          <!-- Radius: elbow hinge at humerus bottom. -->
          <body name="radius" pos="0 {-L_h} 0">
            <joint name="elbow" type="hinge" axis="0 0 1"
                   limited="true" range="0 3.14159"
                   damping="0" armature="0"/>
            <!-- COM at bottom of radius (matching OpenSim COM at body origin).
                 Tiny rotational inertia (OpenSim uses 0). -->
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
        <!-- Spatial tendon: pure geometric path, no spring (stiffness=0). -->
        <spatial name="biceps_tendon" stiffness="0" damping="0"
                 width="0.008" rgba="0.8 0.2 0.2 1">
          <site site="muscle_origin"/>
          <site site="muscle_insertion"/>
        </spatial>
      </tendon>

      <actuator>
        <!-- Motor: gear=-1 so positive ctrl = flexion torque.
             MuJoCo tendon actuator: qfrc = ten_J * ctrl * gear.
             ten_J < 0 for flexor (tendon shortens with flexion), so
             gear=-1 makes positive ctrl produce positive (flexion) torque. -->
        <motor name="biceps_motor" tendon="biceps_tendon" gear="-1"/>
      </actuator>
    </mujoco>""")
