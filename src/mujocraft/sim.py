"""MuJoCo simulation whose static geometry is a Minecraft Region."""

import mujoco
import numpy as np

from . import spider
from .coords import mc_to_mj, mj_to_mc, quat_mj_to_mc, size_mj_to_mc
from .scene import build_scene_xml
from .world import Region

N_JOINTS = 12
OBS_DIM = 3 + 3 + 3 + N_JOINTS + N_JOINTS + 2 + 1  # up, linvel, angvel, q, qd, target xy, dist


class Simulation:
    def __init__(self, region: Region, spawn_mc, timestep: float = 0.005):
        self.region = region
        self.timestep = timestep
        self._spawn_mj = mc_to_mj(spawn_mc)
        self._dirty = False
        self.model: mujoco.MjModel
        self.data: mujoco.MjData
        self._build(previous=None)

    # -- model lifecycle -------------------------------------------------
    def _build(self, previous: mujoco.MjData | None) -> None:
        xml = build_scene_xml(self.region.world_boxes(), self._spawn_mj, self.timestep)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        if previous is not None:  # agent layout is identical across rebuilds
            self.data.qpos[:] = previous.qpos
            self.data.qvel[:] = previous.qvel
            self.data.ctrl[:] = previous.ctrl
            self.data.time = previous.time
        mujoco.mj_forward(self.model, self.data)
        self._torso = self.model.body(f"{spider.NAME}/torso").id
        self._agent_geoms = [
            i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith(f"{spider.NAME}/")
        ]
        self._dirty = False

    def reset(self, spawn_mc=None) -> None:
        if spawn_mc is not None:
            self._spawn_mj = mc_to_mj(spawn_mc)
        self._build(previous=None)

    # -- world edits -----------------------------------------------------
    def set_block(self, x: int, y: int, z: int, solid: bool) -> bool:
        changed = self.region.set_block(x, y, z, solid)
        self._dirty |= changed
        return changed

    def replace_region(self, region: Region) -> None:
        self.region = region
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self._build(previous=self.data)

    # -- stepping --------------------------------------------------------
    def apply_action(self, action) -> None:
        """action in [-1, 1]^12, mapped onto each actuator's control range."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        lo, hi = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = (lo + hi) / 2 + a * (hi - lo) / 2

    def step(self, n: int = 1) -> None:
        self.flush()
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # -- readout ---------------------------------------------------------
    def torso_pos_mc(self) -> np.ndarray:
        # free-joint qpos is the exact current position; xpos lags one step behind mj_step
        return mj_to_mc(self.data.qpos[:3])

    def up_z(self) -> float:
        return float(self.data.xmat[self._torso].reshape(3, 3)[2, 2])

    def observation(self, target_mc) -> np.ndarray:
        d = self.data
        R = d.xmat[self._torso].reshape(3, 3)
        to_target = mc_to_mj(target_mc) - d.xpos[self._torso]
        to_target[2] = 0.0
        local = R.T @ to_target
        obs = np.concatenate(
            [
                R[2, :],  # world up in body frame
                R.T @ d.qvel[:3],  # linear velocity in body frame
                d.qvel[3:6],  # free-joint angular velocity is already body-frame
                d.qpos[7 : 7 + N_JOINTS],
                d.qvel[6 : 6 + N_JOINTS],
                local[:2],
                [np.linalg.norm(to_target)],
            ]
        )
        return obs.astype(np.float32)

    def horizontal_distance(self, target_mc) -> float:
        d = mc_to_mj(target_mc) - self.data.xpos[self._torso]
        return float(np.hypot(d[0], d[1]))

    def poses(self) -> list[dict]:
        """Render payload: one oriented box per spider geom, already in Minecraft axes."""
        mujoco.mj_kinematics(self.model, self.data)  # make geom poses match current qpos
        out = []
        quat = np.empty(4)
        for gid in self._agent_geoms:
            mujoco.mju_mat2Quat(quat, self.data.geom_xmat[gid])
            name = self.model.geom(gid).name
            out.append(
                {
                    "id": name,
                    "pos": mj_to_mc(self.data.geom_xpos[gid]).tolist(),
                    "quat": quat_mj_to_mc(quat).tolist(),
                    "size": size_mj_to_mc(self.model.geom_size[gid]).tolist(),
                    "block": spider.render_block(name),
                }
            )
        return out
