"""Greedy meshing of a boolean voxel grid into a small set of axis-aligned boxes."""

import numpy as np

Box = tuple[int, int, int, int, int, int]  # x0, y0, z0, x1, y1, z1 (end exclusive)


def greedy_mesh(solid: np.ndarray) -> list[Box]:
    """Cover every True cell exactly once with boxes, extending along x, then y, then z."""
    free = np.array(solid, dtype=bool, copy=True)
    nx, ny, nz = free.shape
    boxes: list[Box] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not free[x, y, z]:
                    continue
                x1 = x + 1
                while x1 < nx and free[x1, y, z]:
                    x1 += 1
                y1 = y + 1
                while y1 < ny and free[x:x1, y1, z].all():
                    y1 += 1
                z1 = z + 1
                while z1 < nz and free[x:x1, y:y1, z1].all():
                    z1 += 1
                free[x:x1, y:y1, z:z1] = False
                boxes.append((x, y, z, x1, y1, z1))
    return boxes
