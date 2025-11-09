from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gym
import mujoco
import numpy as np
import mujoco.viewer



@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        is_actor: bool,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
    ):
        # self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model = mujoco.MjModel.from_xml_path("/home/zhou/cfy/gs_sim/mujoco_menagerie/agilex_piper/piper.xml")

        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

        # 开一个物理引擎的线程
        if is_actor:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3
            self.handle.cam.azimuth = 0
            self.handle.cam.elevation = -30
            self.opt = mujoco.MjvOption()

    def render(self):
        # if self._viewer is None:
        #     self._viewer = mujoco.Renderer(
        #         model=self._model,
        #         height=self._render_specs.height,
        #         width=self._render_specs.width,
        #     )
        # self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        # return self._viewer.render()
        self.sync()

    def close(self) -> None:
        # if self._viewer is not None:
        #     self._viewer.close()
        #     self._viewer = None
        if hasattr(self, 'handle'):
            self.handle.close()

        if hasattr(self, 'window'):
            glfw.destroy_window(self.window)
            glfw.terminate()

    def time_limit_exceeded(self) -> bool:
        return self._data.time >= self._time_limit

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random
    
    def sync(self):
        self.handle.sync()
