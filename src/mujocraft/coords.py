"""Coordinate conventions.

Minecraft: x east, y up, z south (right-handed). One block = 1 m.
MuJoCo:    z up. We use mj = (x, -z, y), a proper rotation (det = +1).

Quaternions: MuJoCo is (w, x, y, z); JOML / Minecraft is (x, y, z, w).
"""

import numpy as np


def mc_to_mj(p) -> np.ndarray:
    x, y, z = p
    return np.array([x, -z, y], dtype=float)


def mj_to_mc(p) -> np.ndarray:
    x, y, z = p
    return np.array([x, z, -y], dtype=float)


def quat_mj_to_mc(q) -> np.ndarray:
    """wxyz in MuJoCo axes -> xyzw in Minecraft axes."""
    w, x, y, z = q
    return np.array([x, z, -y, w], dtype=float)


def size_mj_to_mc(half) -> np.ndarray:
    """MuJoCo box half-extents -> full extents along the (rotated) Minecraft display axes."""
    hx, hy, hz = half
    return np.array([2 * hx, 2 * hz, 2 * hy], dtype=float)
