import socket

import pytest

from mujocraft.bridge import Bridge
from mujocraft.protocol import LineDecoder, encode


@pytest.fixture
def pair():
    a, b = socket.socketpair()
    yield Bridge(a), b
    a.close()
    b.close()


def test_poll_returns_nothing_when_idle(pair):
    bridge, _ = pair
    assert bridge.poll(timeout=0.0) == []


def test_poll_receives_messages_from_peer(pair):
    bridge, peer = pair
    peer.sendall(encode({"type": "welcome", "spawn": [0, 70, 0]}))
    out = bridge.poll(timeout=1.0)
    assert out == [{"type": "welcome", "spawn": [0, 70, 0]}]


def test_poll_reassembles_split_message(pair):
    bridge, peer = pair
    raw = encode({"type": "blocks", "changes": [[1, 2, 3, 1]]})
    peer.sendall(raw[:5])
    assert bridge.poll(timeout=0.2) == []
    peer.sendall(raw[5:])
    assert bridge.poll(timeout=1.0)[0]["type"] == "blocks"


def test_send_writes_a_decodable_line(pair):
    bridge, peer = pair
    bridge.send({"type": "pose", "geoms": []})
    peer.settimeout(1.0)
    assert LineDecoder().feed(peer.recv(4096)) == [{"type": "pose", "geoms": []}]


def test_poll_raises_when_peer_closes(pair):
    bridge, peer = pair
    peer.close()
    with pytest.raises(ConnectionError):
        bridge.poll(timeout=1.0)
