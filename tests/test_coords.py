import numpy as np

from mujocraft.coords import mc_to_mj, mj_to_mc, quat_mj_to_mc, size_mj_to_mc


def test_minecraft_up_is_mujoco_up():
    assert np.allclose(mc_to_mj((0, 1, 0)), (0, 0, 1))


def test_minecraft_south_is_negative_mujoco_y():
    assert np.allclose(mc_to_mj((0, 0, 1)), (0, -1, 0))


def test_position_roundtrip():
    p = np.array([3.5, -2.0, 11.25])
    assert np.allclose(mj_to_mc(mc_to_mj(p)), p)


def test_identity_quaternion_stays_identity():
    assert np.allclose(quat_mj_to_mc([1, 0, 0, 0]), [0, 0, 0, 1])


def test_rotation_about_up_axis_maps_to_rotation_about_minecraft_y():
    s = np.sqrt(0.5)
    q_mj = [s, 0, 0, s]  # 90 deg about +z (wxyz)
    assert np.allclose(quat_mj_to_mc(q_mj), [0, s, 0, s])  # xyzw about +y


def test_quaternion_mapping_preserves_norm():
    q = np.array([0.4, 0.1, -0.7, 0.2])
    q /= np.linalg.norm(q)
    assert np.isclose(np.linalg.norm(quat_mj_to_mc(q)), 1.0)


def test_size_is_full_extent_with_y_and_z_swapped():
    assert np.allclose(size_mj_to_mc((0.5, 1.0, 1.5)), (1.0, 3.0, 2.0))
