import mujoco
import numpy as np

from mujocraft.scene import build_scene_xml


def _model(boxes):
    return mujoco.MjModel.from_xml_string(build_scene_xml(boxes))


def test_scene_compiles_with_no_terrain():
    assert _model([]).nu == 12


def test_one_static_geom_per_box():
    m = _model([(0, 0, 0, 4, 1, 4), (2, 1, 2, 3, 2, 3)])
    names = [m.geom(i).name for i in range(m.ngeom)]
    assert sum(n.startswith("block_") for n in names) == 2


def test_block_position_follows_axis_convention():
    m = _model([(10, 5, 3, 11, 6, 4)])
    g = m.geom("block_0")
    assert np.allclose(g.pos, (10.5, -3.5, 5.5))
    assert np.allclose(g.size, (0.5, 0.5, 0.5))


def test_box_half_sizes_swap_y_and_z():
    m = _model([(0, 0, 0, 4, 2, 6)])  # dx=4, dy=2 (up), dz=6
    assert np.allclose(m.geom("block_0").size, (2.0, 3.0, 1.0))


def test_spider_has_four_legs_with_three_joints_each():
    m = _model([])
    joints = [m.joint(i).name for i in range(m.njnt)]
    assert len([j for j in joints if j.startswith("spider/") and j != "spider/root"]) == 12


def test_scene_has_no_player_body():
    m = _model([])
    assert m.nmocap == 0
    assert all("player" not in m.body(i).name for i in range(m.nbody))
