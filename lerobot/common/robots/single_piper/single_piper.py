#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_single_piper import SinglePiperConfig



# piper sdk
from piper_sdk import *

logger = logging.getLogger(__name__)


class SinglePiper(Robot):
    """
    Designed by cfy, jzh
    """

    config_class = SinglePiperConfig
    name = "single_piper"

    def __init__(self, config: SinglePiperConfig):
        super().__init__(config)
        self.config = config
        # 创建 robot 连接
        self.piper = C_PiperInterface("can1")
        # 创建 motor 映射
        self.motors = {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
            "gripper.pos": 0.0,
        }
        # 创建相机
        self.cameras = make_cameras_from_configs(config.cameras)
        # 创建 robot 是否连接的标志位
        self.is_robot_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {k: float for k in self.motors.keys()}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.is_robot_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        # 如果 robot 和 cam 都已成功连接, 则报错
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # robot connect
        self.piper.ConnectPort()
        self.is_robot_connected = True

        for cam in self.cameras.values():
            cam.connect()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        # 暂时空实现
        pass

    def configure(self):
        # 暂时空实现
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        # ==== 左臂 ====
        joint_state = self.piper.GetArmJointMsgs()
        self.motors["joint_1.pos"] = round(joint_state.joint_state.joint_1 * 0.001 / 57.3, 8)
        self.motors["joint_2.pos"] = round(joint_state.joint_state.joint_2 * 0.001 / 57.3, 8)
        self.motors["joint_3.pos"] = round(joint_state.joint_state.joint_3 * 0.001 / 57.3, 8)
        self.motors["joint_4.pos"] = round(joint_state.joint_state.joint_4 * 0.001 / 57.3, 8)
        self.motors["joint_5.pos"] = round(joint_state.joint_state.joint_5 * 0.001 / 57.3, 8)
        self.motors["joint_6.pos"] = round(joint_state.joint_state.joint_6 * 0.001 / 57.3, 8)
        gripper_raw = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        self.motors["gripper.pos"] = round((gripper_raw * 0.001) / 70.0, 8)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        obs_dict = self.motors.copy()

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.motors is None:
            raise DeviceNotConnectedError(f"self motor value is None.")
        
        action_dict = self.motors.copy()
        return action_dict


    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
