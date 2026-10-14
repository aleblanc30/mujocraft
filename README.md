# mujocraft

Two-way sync between Minecraft (Java, Paper) and MuJoCo.

```
 Minecraft world ──block window + deltas──▶  MuJoCo (terrain = static box geoms)
 Minecraft world ◀──pose of every spider geom──  four-legged spider simulated at 20 Hz
```

There is no human player: the spider is the actor.

* **Minecraft → MuJoCo**: a window of solid blocks around the agent is greedy-meshed into box
  geoms. Block place/break events patch the window; the MuJoCo model is recompiled with the
  agent's `qpos/qvel` carried over. The window re-centres as the spider walks.
* **MuJoCo → Minecraft**: each spider geom is a `BlockDisplay` that is teleported/rotated/scaled
  every tick with client-side interpolation. No server-side entity AI or physics involved.

## Run

1. Build the plugin (needs JDK 25; Gradle 9.1+ recommended) and drop the jar into a Paper 26.3
   server's `plugins/`: `cd plugin && gradle build` → `mujocraft-plugin-0.1.0.jar` in
   `%LOCALAPPDATA%\mujocraft-plugin-build\libs\` (build output is kept outside the synced folder)
2. `pip install -e .`
3. Start the server (no one needs to join), then `mujocraft-live` (stands still) or
   `mujocraft-live --policy ppo.zip --goal 120 70 -40`. Spawn is set in the plugin's `config.yml`.

## Train

`SpiderEnv` is a Gymnasium env over a `Region` (a saved Minecraft window or `Region.flat()`).

```
pip install -e ".[train]"

# 1. (optional) snapshot terrain from the running server. Stop mujocraft-live first:
#    the plugin serves one client at a time.
mujocraft-dump --out world.npz                          # 64x32x64 around spawn
mujocraft-dump --center 120 70 -40 --size 96 32 96 --out hills.npz

# 2. train (omit --region for a flat world)
mujocraft-train --region world.npz --steps 1000000 --envs 4 --out ppo.zip

# 3. run it in the live world
mujocraft-live --policy ppo.zip --goal 120 70 -40
```

Or from Python:

```python
from mujocraft.env import SpiderEnv
from mujocraft.world import Region
env = SpiderEnv(region=Region.load("world.npz"))
```

Training waypoints are sampled on the terrain surface within 6 m of the spawn.

Reward is progress toward a waypoint (random per episode, or fixed via `target_mc`), with a fall
penalty. In live mode the waypoint is `--goal`.
Observations are body-frame only, so a policy trained on one terrain transfers to the live loop.

## Known limits

* Solidity is `Material#isSolid()`; slabs, stairs and fences are treated as full cubes or empty.
* Explosions, pistons and fluids don't emit block deltas yet (only player place/break).
* Region reads happen on the main thread (≈55k blocks for the default window), only on re-centre.
* The socket has no auth; the plugin binds to loopback by default.
# mujocraft
