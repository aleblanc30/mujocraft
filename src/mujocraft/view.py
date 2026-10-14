"""Desktop MuJoCo viewer for the imported Minecraft geometry and the spider.

Terrain sources (pick one):
  --region FILE.npz   a saved `Region` (see Region.save)
  --live              pull one terrain window from the running plugin, then simulate locally
  (neither)           a flat demo floor with a few steps and a wall
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from .bridge import Bridge
from .live import stand_still
from .protocol import PROTOCOL_VERSION, unpack_solid
from .sim import Simulation
from .world import Region


def demo_region() -> Region:
    region = Region.flat()
    ox, oy, oz = region.origin
    for i in range(4):  # staircase along +x
        for x in range(6 + 2 * i, 6 + 2 * i + 2):
            region.set_block(x, 1 + i, oz + 28, True)
            for z in range(oz + 28, oz + 36):
                region.set_block(x, 1 + i, z, True)
    for y in range(1, 4):  # wall
        for z in range(-8, 8):
            region.set_block(-10, y, z, True)
    return region


def fetch_region(host: str, port: int, window=(48, 24, 48)) -> tuple[Region, np.ndarray]:
    bridge = Bridge.connect(host, port)
    try:
        bridge.send({"type": "hello", "version": PROTOCOL_VERSION})
        spawn = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for msg in bridge.poll(0.2):
                if msg["type"] == "welcome":
                    spawn = np.asarray(msg["spawn"], dtype=float)
                    origin = Region.origin_for(spawn, window)
                    bridge.send({"type": "request_region", "origin": list(origin), "size": list(window)})
                elif msg["type"] == "region":
                    size = tuple(msg["size"])
                    return Region(tuple(msg["origin"]), unpack_solid(msg["solid"], size)), spawn
        raise TimeoutError("plugin did not answer in time")
    finally:
        bridge.close()


def run_viewer(sim: Simulation, controller=stand_still, goal=None, substeps: int = 10) -> None:
    dt = substeps * sim.timestep
    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sim._torso
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 5.0, -25.0, 135.0
        while v.is_running():
            t0 = time.monotonic()
            torso = sim.torso_pos_mc()
            target = np.asarray(goal, dtype=float) if goal is not None else torso
            sim.apply_action(controller(sim.observation(target)))
            sim.step(substeps)
            v.sync()
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))


def main() -> None:
    p = argparse.ArgumentParser(description="View the Minecraft terrain and MuJoCo spider in a native MuJoCo window.")
    p.add_argument("--region", help="Region .npz file")
    p.add_argument("--live", action="store_true", help="import terrain from the running Minecraft plugin")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25566)
    p.add_argument("--spawn", type=float, nargs=3, metavar=("X", "Y", "Z"), help="spawn in Minecraft coordinates")
    p.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"))
    p.add_argument("--policy", help="stable-baselines3 PPO .zip; omit to stand still")
    args = p.parse_args()

    spawn = np.asarray(args.spawn, dtype=float) if args.spawn else None
    if args.live:
        region, live_spawn = fetch_region(args.host, args.port)
        spawn = spawn if spawn is not None else live_spawn
    elif args.region:
        region = Region.load(args.region)
    else:
        region = demo_region()
    if spawn is None:
        cx = region.origin[0] + region.size[0] / 2
        cz = region.origin[2] + region.size[2] / 2
        spawn = np.array([cx, region.origin[1] + 4.0, cz])

    controller = stand_still
    if args.policy:
        from .live import _sb3_controller

        controller = _sb3_controller(args.policy)
    run_viewer(Simulation(region, spawn), controller, goal=args.goal)


if __name__ == "__main__":
    main()
