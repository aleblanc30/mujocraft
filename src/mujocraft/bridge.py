"""Thin non-blocking client for the plugin's TCP socket."""

import select
import socket

from .protocol import LineDecoder, encode


class Bridge:
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._decoder = LineDecoder()
        try:
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass  # socketpair in tests is not TCP

    @classmethod
    def connect(cls, host: str = "127.0.0.1", port: int = 25566, timeout: float = 10.0) -> "Bridge":
        return cls(socket.create_connection((host, port), timeout=timeout))

    def send(self, msg: dict) -> None:
        self._sock.sendall(encode(msg))

    def poll(self, timeout: float = 0.0) -> list[dict]:
        """Return all complete messages currently available; wait up to `timeout` for the first."""
        out: list[dict] = []
        wait = timeout
        while True:
            ready, _, _ = select.select([self._sock], [], [], wait)
            if not ready:
                return out
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("plugin closed the connection")
            out += self._decoder.feed(chunk)
            wait = 0.0  # drain whatever else is already queued, then return

    def close(self) -> None:
        self._sock.close()
