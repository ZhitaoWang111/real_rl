import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    # KeyBoardIntervention2
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

# from experiments.config import DefaultTrainingConfig
# from experiments.ram_insertion.wrapper import RAMEnv
from examples.experiments.config import DefaultTrainingConfig
from examples.experiments.ram_insertion.wrapper import RAMEnv

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 0.06, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    # setup_mode = "single-arm-fixed-gripper"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="human"):
        # env = RAMEnv(
        #     fake_env=fake_env,
        #     save_video=save_video,
        #     config=EnvConfig(),
        # )
        env = PandaPickCubeGymEnv(render_mode=render_mode, image_obs=True, hz=8, config=EnvConfig())
        classifier=False
        # fake_env=True
        # env = GripperCloseEnv(env)
        if not fake_env:
            # env = SpacemouseIntervention(env)
            env = KeyBoardIntervention2(env)
            pass
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)

        # 把复杂的机器人观测（包含多相机图像和多维状态字典）转化为标准格式
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        # 把 多步观测堆叠成短期记忆 多个动作并依次执行
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env



import glfw
import pynput as keyboard
import gymnasium as gym
class KeyBoardIntervention2(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False
        self.action_indices = action_indices
        self.intervened = False                                                     # 是用pika的动作，还是用智能体原来的动作
        self.human_action = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)     # pika当前的动作, 目前定义为7维（含夹爪）

        listener = keyboard.Listener(on_press=self.on_key)
        listener.daemon = True   # 随主程序退出
        listener.start()

    def on_key(self, key):
        # 分号在 pynput 里是 KeyCode，需要这样比
        try:
            if key.char == ';':
                self.intervened = not self.intervened
                print(f"======== [Intervention] -> {self.intervened} ========")
        except AttributeError:
            # 其他特殊键不处理
            pass


    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a = self.human_action.copy()

        
        # 如果定义了 action_indices，即干预的维度，则只替换这些维度
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if self.intervened:
            return expert_a, True
        else:
            return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_state = 'open'
        return obs, info