"""A solid/non-solid voxel window of the Minecraft world, in absolute block coordinates."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .voxel import Box, greedy_mesh


@dataclass(eq=False)
class Region:
    origin: tuple[int, int, int]
    solid: np.ndarray  # bool [x][y][z]

    @property
    def size(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.solid.shape)  # type: ignore[return-value]

    def _local(self, x: int, y: int, z: int) -> tuple[int, int, int]:
        return x - self.origin[0], y - self.origin[1], z - self.origin[2]

    def contains(self, x: int, y: int, z: int) -> bool:
        lx, ly, lz = self._local(x, y, z)
        nx, ny, nz = self.size
        return 0 <= lx < nx and 0 <= ly < ny and 0 <= lz < nz

    def is_solid(self, x: int, y: int, z: int) -> bool:
        return self.contains(x, y, z) and bool(self.solid[self._local(x, y, z)])

    def set_block(self, x: int, y: int, z: int, solid: bool) -> bool:
        """Returns True only if the window changed."""
        if not self.contains(x, y, z):
            return False
        idx = self._local(x, y, z)
        if bool(self.solid[idx]) == solid:
            return False
        self.solid[idx] = solid
        return True

    def world_boxes(self) -> list[Box]:
        ox, oy, oz = self.origin
        return [
            (x0 + ox, y0 + oy, z0 + oz, x1 + ox, y1 + oy, z1 + oz)
            for x0, y0, z0, x1, y1, z1 in greedy_mesh(self.solid)
        ]

    def needs_recentre(self, pos_mc, margin: int = 8) -> bool:
        for p, o, n in zip(pos_mc, self.origin, self.size):
            if p - o < margin or (o + n) - p < margin:
                return True
        return False

    @staticmethod
    def origin_for(center_mc, size) -> tuple[int, int, int]:
        return tuple(int(math.floor(c)) - s // 2 for c, s in zip(center_mc, size))  # type: ignore[return-value]

    @classmethod
    def flat(cls, size=(64, 16, 64), origin=(-32, 0, -32), floor_y: int = 0) -> "Region":
        solid = np.zeros(size, dtype=bool)
        solid[:, floor_y - origin[1], :] = True
        return cls(tuple(origin), solid)

    def save(self, path) -> None:
        np.savez_compressed(Path(path), origin=np.array(self.origin), solid=self.solid)

    @classmethod
    def load(cls, path) -> "Region":
        with np.load(Path(path)) as f:
            return cls(tuple(int(v) for v in f["origin"]), f["solid"].astype(bool))
