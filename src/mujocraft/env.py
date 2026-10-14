"""Gymnasium environment: the spider must reach a waypoint over Minecraft terrain."""

import gymnasium as gym
import numpy as np

from .sim import N_JOINTS, OBS_DIM, Simulation
from .world import Region


class SpiderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        region: Region | None = None,
        spawn_mc=None,
        target_mc=None,
        frame_skip: int = 10,  # 10 x 5 ms = 20 Hz control; must match live.py / view.py substeps
        max_steps: int = 1000,
        target_radius: float = 6.0,
        success_radius: float = 0.5,
    ):
        self.region = region if region is not None else Region.flat()
        self._fixed_spawn = spawn_mc
        self._fixed_target = target_mc
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.success_radius = success_radius
        self.action_space = gym.spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.sim: Simulation | None = None
        self.target_mc = np.zeros(3)
        self._steps = 0
        self._prev_dist = 0.0

    def _column_top(self, x: float, z: float) -> float | None:
        """Top surface y of the highest solid block in the column containing (x, z), if any."""
        ox, oy, oz = self.region.origin
        nx, _, nz = self.region.size
        ix = min(max(int(np.floor(x)) - ox, 0), nx - 1)
        iz = min(max(int(np.floor(z)) - oz, 0), nz - 1)
        col = np.flatnonzero(self.region.solid[ix, :, iz])
        return float(oy + int(col.max()) + 1) if col.size else None

    def _surface_spawn(self, rng: np.random.Generator) -> np.ndarray:
        """Centre of the window, 0.5 m above the highest solid block in that column."""
        ox, oy, oz = self.region.origin
        nx, _, nz = self.region.size
        x, z = ox + nx // 2 + 0.5, oz + nz // 2 + 0.5
        top = self._column_top(x, z)
        return np.array([x, (top if top is not None else oy) + 0.5, z])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        spawn = (
            np.asarray(self._fixed_spawn, dtype=float)
            if self._fixed_spawn is not None
            else self._surface_spawn(rng)
        )
        if self.sim is None:
            self.sim = Simulation(self.region, spawn)
        else:
            self.sim.reset(spawn)
        if self._fixed_target is not None:
            self.target_mc = np.asarray(self._fixed_target, dtype=float)
        else:
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(2.0, self.target_radius)
            tx, tz = spawn[0] + r * np.cos(ang), spawn[2] + r * np.sin(ang)
            top = self._column_top(tx, tz)  # waypoint rests on the terrain surface
            self.target_mc = np.array([tx, top if top is not None else spawn[1], tz])
        self._steps = 0
        self._prev_dist = self.sim.horizontal_distance(self.target_mc)
        return self.sim.observation(self.target_mc), {}

    def step(self, action):
        assert self.sim is not None, "call reset() first"
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.sim.apply_action(a)
        self.sim.step(self.frame_skip)
        self._steps += 1

        dist = self.sim.horizontal_distance(self.target_mc)
        reward = 10.0 * (self._prev_dist - dist) + 0.02 - 0.001 * float(np.sum(a * a))
        self._prev_dist = dist

        fell = self.sim.up_z() < 0.3
        reached = dist < self.success_radius
        if reached:
            reward += 10.0
        if fell:
            reward -= 5.0
        terminated = bool(fell or reached)
        truncated = bool(self._steps >= self.max_steps)
        return (
            self.sim.observation(self.target_mc),
            float(reward),
            terminated,
            truncated,
            {"distance": dist},
        )
