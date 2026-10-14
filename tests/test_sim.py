import numpy as np

from mujocraft.sim import OBS_DIM, Simulation
from mujocraft.world import Region


def _sim():
    region = Region.flat(size=(32, 12, 32), origin=(-16, 0, -16), floor_y=0)
    return Simulation(region, spawn_mc=(0.0, 1.6, 0.0))


def test_spider_stands_on_flat_floor_with_zero_action():
    sim = _sim()
    sim.step(n=400)  # 2 s
    x, y, z = sim.torso_pos_mc()
    assert 1.2 < y < 1.7
    assert abs(x) < 1.0 and abs(z) < 1.0


def test_torso_stays_upright_when_standing():
    sim = _sim()
    sim.step(n=400)
    assert sim.up_z() > 0.9


def test_poses_have_unique_ids_and_unit_quaternions():
    sim = _sim()
    poses = sim.poses()
    ids = [p["id"] for p in poses]
    assert len(ids) == len(set(ids)) > 10
    for p in poses:
        assert np.isclose(np.linalg.norm(p["quat"]), 1.0, atol=1e-5)
        assert len(p["pos"]) == 3 and len(p["size"]) == 3
        assert p["block"].startswith("minecraft:")


def test_set_block_marks_dirty_and_preserves_agent_state():
    sim = _sim()
    sim.step(n=100)
    before = sim.torso_pos_mc().copy()
    assert sim.set_block(10, 1, 10, True) is True
    sim.flush()
    assert np.allclose(sim.torso_pos_mc(), before, atol=1e-6)


def test_set_block_outside_window_does_not_rebuild():
    sim = _sim()
    assert sim.set_block(500, 1, 500, True) is False


def _n_blocks(sim):
    m = sim.model
    return sum(m.geom(i).name.startswith("block_") for i in range(m.ngeom))


def test_placed_block_adds_a_static_geom():
    sim = _sim()
    before = _n_blocks(sim)
    sim.set_block(10, 1, 10, True)
    sim.flush()
    assert _n_blocks(sim) == before + 1


def test_broken_block_removes_geometry():
    sim = _sim()
    sim.set_block(10, 1, 10, True)
    sim.flush()
    n = _n_blocks(sim)
    sim.set_block(10, 1, 10, False)
    sim.flush()
    assert _n_blocks(sim) == n - 1


def test_observation_shape_and_finite():
    sim = _sim()
    obs = sim.observation(target_mc=(3.0, 1.0, 0.0))
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))


def test_apply_action_clips_to_unit_range():
    sim = _sim()
    sim.apply_action(np.full(12, 10.0))
    lo, hi = sim.model.actuator_ctrlrange.T
    assert np.allclose(sim.data.ctrl, hi)
