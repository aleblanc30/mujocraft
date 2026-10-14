"""Compose terrain boxes + spider into one MJCF document."""

from . import spider
from .coords import mc_to_mj
from .voxel import Box


def build_scene_xml(boxes: list[Box], spawn_mj=(0.0, 0.0, 0.5), timestep: float = 0.005) -> str:
    """`boxes` are absolute Minecraft block-coordinate boxes (end exclusive)."""
    geoms = []
    for i, (x0, y0, z0, x1, y1, z1) in enumerate(boxes):
        cx, cy, cz = mc_to_mj(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
        hx, hy, hz = (x1 - x0) / 2, (z1 - z0) / 2, (y1 - y0) / 2
        geoms.append(
            f'<geom name="block_{i}" type="box" pos="{cx} {cy} {cz}" '
            f'size="{hx} {hy} {hz}" rgba="0.45 0.55 0.35 1" friction="1 0.005 0.0001"/>'
        )
    return f"""<mujoco model="mujocraft">
  <compiler angle="radian"/>
  <option timestep="{timestep}"/>
  <default><geom density="100"/></default>
  <worldbody>
    {chr(10).join(geoms)}
    {spider.body_xml(spawn_mj)}
  </worldbody>
  <actuator>
    {spider.actuator_xml()}
  </actuator>
</mujoco>"""
