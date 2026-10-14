"""Live loop: Minecraft terrain/player in, spider pose out."""

import argparse
import time
from typing import Callable

import numpy as np

from .bridge import Bridge
from .protocol import PROTOCOL_VERSION, unpack_solid
from .sim import N_JOINTS, Simulation
from .world import Region

Controller = Callable[[np.ndarray], np.ndarray]


def stand_still(_obs: np.ndarray) -> np.ndarray:
    return np.zeros(N_JOINTS)


def run_live(
    bridge: Bridge,
    controller: Controller = stand_still,
    window=(48, 24, 48),
    substeps: int = 10,
    tick_hz: float = 20.0,
    margin: int = 8,
    stop: Callable[[], bool] = lambda: False,
    goal=None,
) -> None:
    """One tick = `substeps` MuJoCo steps (default 10 x 5 ms = 50 ms, i.e. real time at 20 Hz).

    There is no player: the spider is the only actor. `goal` is a Minecraft-coordinate
    waypoint fed to the policy; None means "no goal" (target = own position)."""
    bridge.send({"type": "hello", "version": PROTOCOL_VERSION})
    sim: Simulation | None = None
    spawn = None
    pending = False
    period = 1.0 / tick_hz

    def request(center) -> None:
        nonlocal pending
        origin = Region.origin_for(center, window)
        bridge.send({"type": "request_region", "origin": list(origin), "size": list(window)})
        pending = True

    while not stop():
        t0 = time.monotonic()
        for msg in bridge.poll(0.0):
            kind = msg["type"]
            if kind == "welcome":
                spawn = np.asarray(msg["spawn"], dtype=float)
                request(spawn)
            elif kind == "region":
                size = tuple(msg["size"])
                region = Region(tuple(msg["origin"]), unpack_solid(msg["solid"], size))
                if sim is None:
                    sim = Simulation(region, spawn, timestep=1.0 / (tick_hz * substeps))
                else:
                    sim.replace_region(region)
                pending = False
            elif kind == "blocks" and sim is not None:
                for x, y, z, s in msg["changes"]:
                    sim.set_block(x, y, z, bool(s))

        if sim is not None:
            torso = sim.torso_pos_mc()
            target = np.asarray(goal, dtype=float) if goal is not None else torso
            sim.apply_action(controller(sim.observation(target)))
            sim.step(substeps)
            bridge.send({"type": "pose", "geoms": sim.poses()})
            if not pending and sim.region.needs_recentre(sim.torso_pos_mc(), margin):
                request(sim.torso_pos_mc())

        time.sleep(max(0.0, period - (time.monotonic() - t0)))


def _sb3_controller(path: str) -> Controller:
    from stable_baselines3 import PPO

    model = PPO.load(path)
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Run the MuJoCo spider inside a live Minecraft server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25566)
    p.add_argument("--policy", help="stable-baselines3 PPO .zip; omit to stand still")
    p.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"), help="waypoint in Minecraft coordinates")
    args = p.parse_args()
    controller = _sb3_controller(args.policy) if args.policy else stand_still
    bridge = Bridge.connect(args.host, args.port)
    try:
        run_live(bridge, controller, goal=args.goal)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
