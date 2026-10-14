import numpy as np

from mujocraft.world import Region


def _region():
    return Region.flat(size=(16, 8, 16), origin=(-8, 0, -8), floor_y=0)


def test_flat_region_has_solid_floor_only():
    r = _region()
    assert r.is_solid(0, 0, 0)
    assert not r.is_solid(0, 1, 0)


def test_contains_uses_absolute_coordinates():
    r = _region()
    assert r.contains(-8, 0, -8)
    assert r.contains(7, 7, 7)
    assert not r.contains(8, 0, 0)
    assert not r.contains(0, -1, 0)


def test_set_block_reports_change():
    r = _region()
    assert r.set_block(0, 3, 0, True) is True
    assert r.is_solid(0, 3, 0)


def test_set_block_same_value_reports_no_change():
    r = _region()
    assert r.set_block(0, 0, 0, True) is False


def test_set_block_outside_window_is_ignored():
    r = _region()
    assert r.set_block(100, 0, 0, True) is False


def test_world_boxes_are_absolute():
    r = _region()
    assert r.world_boxes() == [(-8, 0, -8, 8, 1, 8)]


def test_needs_recentre_near_edge_only():
    r = _region()
    assert not r.needs_recentre((0.0, 4.0, 0.0), margin=3)
    assert r.needs_recentre((6.0, 4.0, 0.0), margin=3)
    assert r.needs_recentre((0.0, 4.0, -7.0), margin=3)


def test_origin_for_centres_window_on_point():
    assert Region.origin_for((10.4, 5.0, -3.2), (16, 8, 16)) == (2, 1, -12)


def test_save_load_roundtrip(tmp_path):
    r = _region()
    r.set_block(1, 4, 2, True)
    p = tmp_path / "r.npz"
    r.save(p)
    r2 = Region.load(p)
    assert r2.origin == r.origin
    assert np.array_equal(r2.solid, r.solid)
