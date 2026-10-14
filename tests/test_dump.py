import socket
import threading
import time

import numpy as np
import pytest

from mujocraft.bridge import Bridge
from mujocraft.dump import fetch_region
from mujocraft.protocol import LineDecoder, encode, pack_solid


class FakePlugin(threading.Thread):
    """Plays the server side: answers hello with welcome and request_region with a region."""

    def __init__(self, sock, spawn=(10.5, 70.0, -4.5), answer_region=True, noise=False):
        super().__init__(daemon=True)
        self.sock, self.spawn = sock, spawn
        self.answer_region = answer_region
        self.noise = noise
        self.requests: list[dict] = []

    def run(self):
        dec = LineDecoder()
        self.sock.settimeout(5.0)
        try:
            while True:
                data = self.sock.recv(65536)
                if not data:
                    return
                for msg in dec.feed(data):
                    self.requests.append(msg)
                    if msg["type"] == "hello":
                        self.sock.sendall(encode({"type": "welcome", "version": 1, "spawn": list(self.spawn)}))
                    elif msg["type"] == "request_region" and self.answer_region:
                        if self.noise:
                            self.sock.sendall(encode({"type": "blocks", "changes": [[0, 0, 0, 1]]}))
                        size = tuple(msg["size"])
                        solid = np.zeros(size, dtype=bool)
                        solid[:, 0, :] = True
                        self.sock.sendall(
                            encode(
                                {
                                    "type": "region",
                                    "origin": msg["origin"],
                                    "size": msg["size"],
                                    "solid": pack_solid(solid),
                                }
                            )
                        )
        except (OSError, TimeoutError):
            return


@pytest.fixture
def plugin_pair():
    a, b = socket.socketpair()
    yield a, b
    a.close()
    b.close()


def _start(plugin_pair, **kw):
    a, b = plugin_pair
    fake = FakePlugin(b, **kw)
    fake.start()
    return Bridge(a), fake


def test_fetch_region_centres_on_spawn_when_no_center_given(plugin_pair):
    bridge, fake = _start(plugin_pair)
    region = fetch_region(bridge, size=(16, 8, 16), timeout=3.0)
    req = next(m for m in fake.requests if m["type"] == "request_region")
    assert req["origin"] == [2, 66, -13]  # floor(c) - size//2 per axis: 10-8, 70-4, -5-8
    assert region.origin == (2, 66, -13)
    assert region.size == (16, 8, 16)


def test_fetch_region_uses_explicit_center(plugin_pair):
    bridge, fake = _start(plugin_pair)
    region = fetch_region(bridge, size=(8, 8, 8), center=(100.0, 64.0, 100.0), timeout=3.0)
    assert region.origin == (96, 60, 96)


def test_fetch_region_returns_decoded_solid_grid(plugin_pair):
    bridge, _ = _start(plugin_pair)
    region = fetch_region(bridge, size=(4, 4, 4), timeout=3.0)
    assert region.solid[:, 0, :].all()
    assert not region.solid[:, 1:, :].any()


def test_fetch_region_ignores_unrelated_messages(plugin_pair):
    bridge, _ = _start(plugin_pair, noise=True)
    region = fetch_region(bridge, size=(4, 4, 4), timeout=3.0)
    assert region.size == (4, 4, 4)


def test_fetch_region_times_out_when_plugin_never_answers(plugin_pair):
    bridge, _ = _start(plugin_pair, answer_region=False)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        fetch_region(bridge, size=(4, 4, 4), timeout=0.5)
    assert time.monotonic() - t0 < 3.0
