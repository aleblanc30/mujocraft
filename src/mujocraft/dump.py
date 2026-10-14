"""Dump a window of the live Minecraft world to an .npz file for offline training.

The plugin serves one client at a time: stop `mujocraft-live` before dumping.
"""

import argparse
import time

import numpy as np

from .bridge import Bridge
from .protocol import PROTOCOL_VERSION, unpack_solid
from .world import Region


def fetch_region(bridge: Bridge, size=(64, 32, 64), center=None, timeout: float = 15.0) -> Region:
    """Handshake, then request a `size` window centred on `center` (default: the spawn point)."""
    bridge.send({"type": "hello", "version": PROTOCOL_VERSION})
    requested = False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for msg in bridge.poll(0.1):
            kind = msg["type"]
            if kind == "welcome" and not requested:
                c = np.asarray(center if center is not None else msg["spawn"], dtype=float)
                origin = Region.origin_for(c, size)
                bridge.send({"type": "request_region", "origin": list(origin), "size": list(size)})
                requested = True
            elif kind == "region" and requested:
                shape = tuple(msg["size"])
                return Region(tuple(msg["origin"]), unpack_solid(msg["solid"], shape))
    raise TimeoutError("plugin did not answer in time" + ("" if requested else " (no welcome)"))


def main() -> None:
    p = argparse.ArgumentParser(description="Save a Minecraft block window for offline training.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25566)
    p.add_argument("--size", type=int, nargs=3, default=(64, 32, 64), metavar=("NX", "NY", "NZ"))
    p.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"), help="default: spawn point")
    p.add_argument("--out", required=True, help="output .npz path")
    args = p.parse_args()

    bridge = Bridge.connect(args.host, args.port)
    try:
        region = fetch_region(bridge, tuple(args.size), args.center)
    finally:
        bridge.close()
    region.save(args.out)
    n = int(region.solid.sum())
    print(f"saved {args.out}: origin={region.origin} size={region.size} solid_blocks={n}")


if __name__ == "__main__":
    main()
