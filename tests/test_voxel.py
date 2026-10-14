import numpy as np

from mujocraft.voxel import greedy_mesh


def _rasterise(boxes, shape):
    grid = np.zeros(shape, dtype=int)
    for x0, y0, z0, x1, y1, z1 in boxes:
        grid[x0:x1, y0:y1, z0:z1] += 1
    return grid


def test_empty_grid_has_no_boxes():
    assert greedy_mesh(np.zeros((3, 3, 3), dtype=bool)) == []


def test_single_block_is_one_unit_box():
    g = np.zeros((4, 4, 4), dtype=bool)
    g[1, 2, 3] = True
    assert greedy_mesh(g) == [(1, 2, 3, 2, 3, 4)]


def test_full_cube_merges_to_one_box():
    g = np.ones((3, 4, 5), dtype=bool)
    assert greedy_mesh(g) == [(0, 0, 0, 3, 4, 5)]


def test_flat_floor_merges_to_one_box():
    g = np.zeros((10, 3, 10), dtype=bool)
    g[:, 0, :] = True
    assert len(greedy_mesh(g)) == 1


def test_l_shape_is_covered_exactly_once():
    g = np.zeros((4, 4, 4), dtype=bool)
    g[0:3, 0, 0] = True
    g[0, 0:3, 0] = True
    covered = _rasterise(greedy_mesh(g), g.shape)
    assert np.array_equal(covered, g.astype(int))


def test_random_grid_is_covered_exactly_once():
    rng = np.random.default_rng(0)
    g = rng.random((8, 6, 7)) < 0.5
    covered = _rasterise(greedy_mesh(g), g.shape)
    assert np.array_equal(covered, g.astype(int))


def test_checkerboard_cannot_merge():
    x, y, z = np.indices((4, 4, 4))
    g = (x + y + z) % 2 == 0
    assert len(greedy_mesh(g)) == int(g.sum())


def test_input_is_not_mutated():
    g = np.ones((2, 2, 2), dtype=bool)
    before = g.copy()
    greedy_mesh(g)
    assert np.array_equal(g, before)
