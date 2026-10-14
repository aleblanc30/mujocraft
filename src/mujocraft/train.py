"""PPO training of the spider on a Minecraft region (or a flat world)."""

import argparse
from pathlib import Path

from .env import SpiderEnv
from .world import Region


def load_region(path) -> Region:
    return Region.load(path) if path else Region.flat()


def make_env(region: Region, max_steps: int = 500) -> SpiderEnv:
    """Each env gets its own copy so edits in one cannot leak into another."""
    return SpiderEnv(region=Region(region.origin, region.solid.copy()), max_steps=max_steps)


def train(
    region: Region,
    total_timesteps: int,
    out,
    n_envs: int = 4,
    seed: int = 0,
    n_steps: int = 2048,
    batch_size: int = 256,
    max_steps: int = 500,
    verbose: int = 1,
) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(lambda: make_env(region, max_steps), n_envs=n_envs, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps)
    out = Path(out)
    model.save(out)  # SB3 appends .zip when missing
    return out if out.suffix == ".zip" else out.with_suffix(out.suffix + ".zip")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the MuJoCo spider with PPO.")
    p.add_argument("--region", help="Region .npz from mujocraft-dump; omit for a flat world")
    p.add_argument("--steps", type=int, default=1_000_000, help="total environment steps")
    p.add_argument("--envs", type=int, default=4, help="parallel environments")
    p.add_argument("--out", default="ppo.zip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-steps", type=int, default=500)
    return p.parse_args(argv)


def main() -> None:
    a = parse_args()
    out = train(
        load_region(a.region),
        total_timesteps=a.steps,
        out=a.out,
        n_envs=a.envs,
        seed=a.seed,
        max_steps=a.episode_steps,
    )
    print(f"saved policy to {out}")


if __name__ == "__main__":
    main()
