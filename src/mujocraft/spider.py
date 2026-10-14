"""A four-legged box-built spider (3 DoF per leg). Boxes only, so the MuJoCo geometry
is rendered 1:1 by Minecraft block displays."""

import math

NAME = "spider"
LEGS = {"fl": math.pi / 4, "fr": -math.pi / 4, "bl": 3 * math.pi / 4, "br": -3 * math.pi / 4}
HIP = 0.25  # hip offset from torso centre along x and y
KP = 60.0
FORCE = 40.0

YAW_RANGE = (-0.6, 0.6)
PITCH_RANGE = (-0.8, 0.8)
KNEE_RANGE = (-1.2, 1.2)
_JOINTS = (("yaw", YAW_RANGE), ("pitch", PITCH_RANGE), ("knee", KNEE_RANGE))


def _sign(v: float) -> int:
    return 1 if v >= 0 else -1


def body_xml(pos_mj=(0.0, 0.0, 0.5)) -> str:
    n = NAME
    legs = []
    for leg, yaw in LEGS.items():
        sx, sy = _sign(math.cos(yaw)), _sign(math.sin(yaw))
        legs.append(f"""
      <body name="{n}/{leg}" pos="{sx * HIP} {sy * HIP} 0" euler="0 0 {yaw}">
        <joint name="{n}/{leg}_yaw" axis="0 0 1" range="{YAW_RANGE[0]} {YAW_RANGE[1]}" damping="1.5"/>
        <geom name="{n}/{leg}_hip" type="box" pos="0.03 0 0" size="0.06 0.05 0.05"/>
        <body name="{n}/{leg}_upper" pos="0.06 0 0">
          <joint name="{n}/{leg}_pitch" axis="0 1 0" range="{PITCH_RANGE[0]} {PITCH_RANGE[1]}" damping="1.5"/>
          <geom name="{n}/{leg}_upper" type="box" pos="0.12 0 0" size="0.12 0.04 0.04"/>
          <body name="{n}/{leg}_lower" pos="0.24 0 0">
            <joint name="{n}/{leg}_knee" axis="0 1 0" range="{KNEE_RANGE[0]} {KNEE_RANGE[1]}" damping="1.5"/>
            <geom name="{n}/{leg}_lower" type="box" pos="0 0 -0.2" size="0.04 0.04 0.2"/>
          </body>
        </body>
      </body>""")
    px, py, pz = pos_mj
    return f"""
    <body name="{n}/torso" pos="{px} {py} {pz}">
      <freejoint name="{n}/root"/>
      <geom name="{n}/torso" type="box" size="0.25 0.25 0.1"/>
      <geom name="{n}/head" type="box" pos="0.3 0 0.02" size="0.08 0.1 0.08"/>{''.join(legs)}
    </body>"""


def actuator_xml() -> str:
    rows = []
    for leg in LEGS:
        for joint, (lo, hi) in _JOINTS:
            rows.append(
                f'<position name="{NAME}/{leg}_{joint}" joint="{NAME}/{leg}_{joint}" kp="{KP}" '
                f'ctrlrange="{lo} {hi}" forcerange="-{FORCE} {FORCE}"/>'
            )
    return "\n    ".join(rows)


def render_block(geom_name: str) -> str:
    """Which Minecraft block a spider geom is drawn as."""
    if geom_name.endswith("head"):
        return "minecraft:red_concrete"
    if geom_name.endswith("torso"):
        return "minecraft:black_concrete"
    return "minecraft:gray_concrete"
