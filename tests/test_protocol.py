import numpy as np
import pytest

from mujocraft.protocol import (
    LineDecoder,
    ProtocolError,
    encode,
    pack_solid,
    unpack_solid,
)


def test_encode_is_single_newline_terminated_line():
    data = encode({"type": "hello", "version": 1})
    assert data.endswith(b"\n")
    assert data.count(b"\n") == 1


def test_roundtrip_single_message():
    dec = LineDecoder()
    assert dec.feed(encode({"type": "pose", "t": 1.5})) == [{"type": "pose", "t": 1.5}]


def test_multiple_messages_in_one_chunk():
    dec = LineDecoder()
    out = dec.feed(encode({"type": "a"}) + encode({"type": "b"}))
    assert [m["type"] for m in out] == ["a", "b"]


def test_message_split_across_chunks():
    dec = LineDecoder()
    raw = encode({"type": "split", "payload": "x" * 100})
    assert dec.feed(raw[:17]) == []
    assert dec.feed(raw[17:])[0]["type"] == "split"


def test_blank_lines_are_ignored():
    assert LineDecoder().feed(b"\n\n") == []


def test_invalid_json_raises():
    with pytest.raises(ProtocolError):
        LineDecoder().feed(b"{not json\n")


def test_message_without_type_raises():
    with pytest.raises(ProtocolError):
        LineDecoder().feed(b'{"x": 1}\n')


def test_pack_unpack_roundtrip_odd_shape():
    rng = np.random.default_rng(1)
    g = rng.random((5, 3, 7)) < 0.4
    assert np.array_equal(unpack_solid(pack_solid(g), g.shape), g)


def test_pack_is_msb_first_x_major_z_fastest():
    g = np.zeros((1, 1, 8), dtype=bool)
    g[0, 0, 0] = True
    # first block along z is the high bit of the first byte
    assert unpack_solid(pack_solid(g), g.shape)[0, 0, 0]
    import base64

    assert base64.b64decode(pack_solid(g)) == b"\x80"


def test_unpack_too_short_raises():
    with pytest.raises(ProtocolError):
        unpack_solid(pack_solid(np.ones((1, 1, 8), dtype=bool)), (4, 4, 4))
