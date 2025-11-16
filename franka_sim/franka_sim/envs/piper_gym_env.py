from pathlib import Path
from typing import Any, Literal, Tuple, Dict

# import gym  #Origin
import gymnasium as gym # startear
import mujoco
import numpy as np
from gym import spaces
import time 
import cv2

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv


_HERE = Path(__file__).parent
# _XML_PATH = _HERE / "xmls" / "arena.xml"
_XML_PATH = _HERE / "piper" / "scene.xml"

_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4)) # Origin
# _PANDA_HOME = np.asarray((0, -0.785, -0.2, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]]) # Origin
# _SAMPLING_BOUNDS = np.asarray([[0.35, -0.15], [0.45, 0.15]])


import time
from lerobot.common.cameras.utils import make_cameras_from_configs
import rerun as rr
import numpy as np
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.utils.visualization_utils import _init_rerun
from piper_sdk import *


class PiperPickCubeGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        config = None,
        hz = 10,
    ):
        print(f"[Info] Initializing PiperPickCubeGymEnv !!!!!!!!!!!!!")
        self.hz = hz
        self._action_scale = action_scale
        # render_mode = "rgb_array"
        # control_dt = 0.1
        # physics_dt = 0.01
        self.is_actor = config.ACTOR

        self.is_real_env = True

        super().__init__(
            self.is_real_env,
            self.is_actor,
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs
        self.env_step = 0
        self.intervened = False


        if self.image_obs:
            # startear
            self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_pose": gym.spaces.Box(
                            -1.0, 1.0, shape=(7,),dtype=np.float32
                        ),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8) 
                                for key in config.REALSENSE_CAMERAS}
                ),
            }
            )

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0,-1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self.success    = False
        self.terminated = False


        # 真实机器人
        # 
        self.cameras_cfg = {
            "top"  : OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=30),
            # "wrist": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30),
        }

        if config.ACTOR:
            _init_rerun(session_name="sim_mobile_ai_env")
            self.cameras = make_cameras_from_configs(self.cameras_cfg)
            for cam in self.cameras.values():
                cam.connect()
                print(f"[Info] Camera connected.")

        self.piper_left = C_PiperInterface("can1")
        self.piper_left.ConnectPort()
        self.piper_left.EnableArm(7)
        self.piper_left.GripperCtrl(0,1000,0x01, 0)

        self.joint_limits = np.array([
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
            (0.0, 1.0),
        ])

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if self.is_real_env is True:
            # step 1 : 夹爪先松开
            gripper_count = 0
            while gripper_count < 50:
                gripper_count += 1
                self.piper_left.GripperCtrl(1, 1000, 0x01, 0)
                time.sleep(0.01)
            factor = 57324.840764  # 1000*180/3.14
            left_pos = [0.0, 1.03506108, -0.8056719, -0.01169284, 0.78328098, 0.0, 1]
            left_joint_0 = round(left_pos[0] * factor)                                 
            left_joint_1 = round(left_pos[1] * factor)
            left_joint_2 = round(left_pos[2] * factor)
            left_joint_3 = round(left_pos[3] * factor)
            left_joint_4 = round(left_pos[4] * factor)
            left_joint_5 = round(left_pos[5] * factor)
            left_joint_6 = round(left_pos[6] * 100 * 1000)

            # step 2 : 机械臂回初始位置
            arm_reset_count = 0
            while arm_reset_count < 800:
                arm_reset_count += 1
                # 控制左机械臂
                self.piper_left.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                self.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
                self.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)
                time.sleep(0.005)

            
        else:
            sim_count = 0
            left_pos = [0.0, 1.03506108, -0.8056719, -0.01169284, 0.78328098, 0.0, 1]
            sim_reset_action = left_pos
            sim_reset_action[1] = -1
            reset_action = [left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5, left_joint_6]
            while sim_count < 1000:
                sim_count += 1
                self._data.ctrl = sim_reset_action
                mujoco.mj_step(self._model, self._data)
                self.render()
                time.sleep(0.002)
        

        # step 3 : reset
        self.env_step = 0
        obs = self._compute_observation()
        self.success = False
        self.terminated = False
        print(f"[Info] ============ Reset done ============")
        return obs, {"succeed": False}
    
    def close(self) -> None:
        super().close()
        for cam in self.cameras.values():
            cam.disconnect()

    def step(
        self, action: np.ndarray, replaced
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        ## 网络与接管后的动作均为 (-1, 1)
        # step 1 : 获取当前机械臂关节位置
        cur_joint_state = self.get_cur_joint_pos()

        # 此时 action 是 delta 量
        max_delta_per_step = np.array([
            0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.005
        ], dtype=np.float32)

        assert action.shape == (7,), f"Action must be 7D for single arm, got {action.shape}"
        
        # Process active arm
        delta_action = action * max_delta_per_step

        # step 2 : 计算目标关节位置
        joint_targets = cur_joint_state + delta_action

        joint_targets[6] = 0
        # step 3 : 将 joint_targets 限制在 joint_limits 内
        joint_targets = np.clip(joint_targets, self.joint_limits[:, 0], self.joint_limits[:, 1])
        # if not replaced:
        #     print(f"[neural ] target_joint_action: {delta_action}")
        
        
        if self.is_real_env:
            # step 4 : 缩放到真实控制
            factor = 57324.840764  # 1000*180/3.14
            left_joint_0 = round(joint_targets[0] * factor)
            left_joint_1 = round(joint_targets[1] * factor)
            left_joint_2 = round(joint_targets[2] * factor)
            left_joint_3 = round(joint_targets[3] * factor)
            left_joint_4 = round(joint_targets[4] * factor)
            left_joint_5 = round(joint_targets[5] * factor)
            left_joint_6 = round(joint_targets[6] * 100 * 1000)
            self.piper_left.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            self.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
            self.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)
        else:
            for _ in range(self._n_substeps):
                self._data.ctrl = joint_targets
                mujoco.mj_step(self._model, self._data)
                self.render()
                time.sleep(0.002)

        obs = self._compute_observation()
        rew = self._compute_reward()
        grasp_penalty = 0.0

        # terminated = self.time_limit_exceeded()
        self.env_step += 1
        terminated = False
        if self.env_step >= 5000 or self.terminated:
            terminated = True
        if self.env_step % 100 == 0:
            print(f"[Info] ======= Step {self.env_step} ========")


        success = self._compute_success()
        if success:
            print(f'success!')
            rew = 1
        else:
            rew = 0
        done = terminated or success

        return obs, rew, done, False, {"succeed": success, "grasp_penalty": grasp_penalty}

    def _compute_success(self):
        return self.success
    
    def render(self):
        self.sync()


    def get_cur_joint_pos(self):
        left_joint_state = self.piper_left.GetArmJointMsgs()
        left_joint_1_pos = round(left_joint_state.joint_state.joint_1 * 0.001 / 57.3, 8)
        left_joint_2_pos = round(left_joint_state.joint_state.joint_2 * 0.001 / 57.3, 8)
        left_joint_3_pos = round(left_joint_state.joint_state.joint_3 * 0.001 / 57.3, 8)
        left_joint_4_pos = round(left_joint_state.joint_state.joint_4 * 0.001 / 57.3, 8)
        left_joint_5_pos = round(left_joint_state.joint_state.joint_5 * 0.001 / 57.3, 8)
        left_joint_6_pos = round(left_joint_state.joint_state.joint_6 * 0.001 / 57.3, 8)
        left_gripper_raw = self.piper_left.GetArmGripperMsgs().gripper_state.grippers_angle
        left_gripper_pos = round((left_gripper_raw * 0.001) / 70.0, 8)
        left_joint_array = np.array(
            [
                left_joint_1_pos,
                left_joint_2_pos,
                left_joint_3_pos,
                left_joint_4_pos,
                left_joint_5_pos,
                left_joint_6_pos,
                left_gripper_pos
            ],
            dtype=np.float32
        )
        return left_joint_array

    # Helper methods.


    def _compute_observation(self) -> dict:
        obs = {"state": {}, "images": {}}
        for cam_key, cam in self.cameras.items():
            cam_ori = cam.async_read()
            # cam_resized = cv2.resize(cam_ori, (128, 128), interpolation=cv2.INTER_AREA)
            obs["images"][cam_key] = cam_ori

        left_joint_array = self.get_cur_joint_pos()
        obs['state'] = {
            "joint_pose": left_joint_array,
        }

        joint_vec = obs["state"]["joint_pose"]  # shape (7,)
        for i, v in enumerate(joint_vec):
            rr.log(f"observation.state.joint_pose/{i}", rr.Scalar(float(v)))

        # 2) 图像：每个相机一帧
        for cam_key, img in obs["images"].items():
            # 确保是 uint8，HWC
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            # 支持灰度或彩色
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] in (1, 3, 4)):
                rr.log(f"observation.images/{cam_key}", rr.Image(img), static=True)
            else:
                # 形状异常就跳过，避免抛错
                print(f"[warn] skip logging {cam_key}, unexpected shape {img.shape}")

        return obs


    def _compute_reward(self) -> float:

        if self.success:
            return 1.0
        return 0.0


if __name__ == "__main__":
    env = PiperPickCubeGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()
