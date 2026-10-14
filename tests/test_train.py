import numpy as np
import pytest

from mujocraft.train import load_region, make_env, parse_args, train
from mujocraft.world import Region


def test_load_region_defaults_to_flat_world():
    r = load_region(None)
    assert r.is_solid(0, 0, 0)
    assert not r.is_solid(0, 1, 0)


def test_load_region_reads_saved_file(tmp_path):
    src = Region.flat(size=(8, 4, 8), origin=(0, 0, 0), floor_y=0)
    src.set_block(3, 2, 3, True)
    p = tmp_path / "w.npz"
    src.save(p)
    r = load_region(p)
    assert r.is_solid(3, 2, 3)


def test_make_env_copies_the_region():
    src = Region.flat(size=(8, 4, 8), origin=(0, 0, 0), floor_y=0)
    env = make_env(src)
    env.region.set_block(1, 1, 1, True)
    assert not src.is_solid(1, 1, 1)


def test_parse_args_defaults_and_overrides():
    a = parse_args(["--steps", "123", "--out", "p.zip", "--region", "w.npz", "--envs", "2"])
    assert (a.steps, a.out, a.region, a.envs) == (123, "p.zip", "w.npz", 2)
    d = parse_args([])
    assert d.region is None and d.steps > 0 and d.envs >= 1


def test_train_writes_a_loadable_policy(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    region = Region.flat(size=(24, 8, 24), origin=(-12, 0, -12), floor_y=0)
    out = train(
        region=region,
        total_timesteps=64,
        out=tmp_path / "ppo",
        n_envs=1,
        n_steps=32,
        batch_size=32,
        max_steps=16,
        seed=0,
    )
    assert out.exists() and out.suffix == ".zip"
    model = PPO.load(out)
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (12,)
