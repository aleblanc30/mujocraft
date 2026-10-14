import numpy as np

from mujocraft.env import SpiderEnv
from mujocraft.sim import OBS_DIM


def _env(**kw):
    return SpiderEnv(max_steps=20, **kw)


def test_reset_returns_observation_of_declared_shape():
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.contains(obs)


def test_step_returns_gymnasium_tuple():
    env = _env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert terminated in (True, False)
    assert truncated in (True, False)


def test_truncates_at_max_steps():
    env = _env()
    env.reset(seed=0)
    truncated = False
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(np.zeros(12, dtype=np.float32))
        if terminated:
            break
    assert truncated or terminated


def test_fixed_target_is_respected():
    env = _env(target_mc=(4.0, 1.0, 0.0))
    env.reset(seed=0)
    assert np.allclose(env.target_mc, (4.0, 1.0, 0.0))


def test_random_target_sits_on_terrain_surface():
    from mujocraft.world import Region

    r = Region.flat(size=(32, 12, 32), origin=(-16, 0, -16), floor_y=0)
    r.solid[:, 1:4, :] = True  # raise the ground: surface is now y=4
    env = SpiderEnv(region=r, max_steps=5)
    env.reset(seed=1)
    assert env.target_mc[1] == 4.0


def test_spawn_is_above_raised_terrain():
    from mujocraft.world import Region

    r = Region.flat(size=(32, 12, 32), origin=(-16, 0, -16), floor_y=0)
    r.solid[:, 1:4, :] = True
    env = SpiderEnv(region=r, max_steps=5)
    env.reset(seed=1)
    assert env.sim.torso_pos_mc()[1] > 4.0


def test_seeded_reset_is_deterministic():
    a, _ = _env().reset(seed=3)
    b, _ = _env().reset(seed=3)
    assert np.allclose(a, b)


def test_reward_prefers_approaching_target():
    # target straight ahead (+x in mc == +x in mj == torso forward)
    env = _env(target_mc=(5.0, 1.0, 0.0))
    env.reset(seed=0)
    env.sim.data.qpos[0] += 0.5  # teleport closer
    import mujoco

    mujoco.mj_forward(env.sim.model, env.sim.data)
    _, r_close, *_ = env.step(np.zeros(12, dtype=np.float32))
    env.reset(seed=0)
    env.sim.data.qpos[0] -= 0.5  # teleport farther
    mujoco.mj_forward(env.sim.model, env.sim.data)
    _, r_far, *_ = env.step(np.zeros(12, dtype=np.float32))
    assert r_close >= r_far
