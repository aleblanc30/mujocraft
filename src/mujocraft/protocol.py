"""Wire format: newline-delimited JSON over TCP.

plugin -> python
  welcome  {version, spawn:[x,y,z]}
  region   {origin:[x,y,z], size:[nx,ny,nz], solid:<base64 bits>}
  blocks   {changes:[[x,y,z,solid(0/1)], ...]}

python -> plugin
  hello          {version}
  request_region {origin:[x,y,z], size:[nx,ny,nz]}
  pose           {geoms:[{id, pos:[x,y,z], quat:[x,y,z,w], size:[sx,sy,sz], block}]}

`solid` is np.packbits over an [x][y][z] array (z fastest), most significant bit first.
"""

import base64
import json

import numpy as np

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    pass


def encode(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode() + b"\n"


class LineDecoder:
    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[dict]:
        self._buf += data
        out: list[dict] = []
        while (i := self._buf.find(b"\n")) >= 0:
            line = bytes(self._buf[:i])
            del self._buf[: i + 1]
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except ValueError as e:
                raise ProtocolError(f"invalid JSON: {e}") from e
            if not isinstance(msg, dict) or "type" not in msg:
                raise ProtocolError("message must be an object with a 'type'")
            out.append(msg)
        return out


def pack_solid(solid: np.ndarray) -> str:
    bits = np.packbits(np.asarray(solid, dtype=bool).ravel())
    return base64.b64encode(bits.tobytes()).decode()


def unpack_solid(data: str, shape: tuple[int, int, int]) -> np.ndarray:
    n = int(np.prod(shape))
    bits = np.unpackbits(np.frombuffer(base64.b64decode(data), dtype=np.uint8))
    if bits.size < n:
        raise ProtocolError(f"expected {n} bits, got {bits.size}")
    return bits[:n].astype(bool).reshape(shape)
