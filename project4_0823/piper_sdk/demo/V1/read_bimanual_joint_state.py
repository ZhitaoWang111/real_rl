


#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 读取机械臂消息并打印,需要先安装piper_sdk
from typing import (
    Optional,
)
from piper_sdk import *
import time
import numpy as np
import torch

# 测试代码
if __name__ == "__main__":
    piper_left = C_PiperInterface("can0")
    piper_right = C_PiperInterface("can1")
    piper_left.ConnectPort()
    piper_right.ConnectPort()
    while True:
        
        joint_state = []

        left_read_joint_state = piper_left.GetArmJointMsgs()

        left_joint_state_1 = torch.from_numpy(np.array([round(left_read_joint_state.joint_state.joint_1 * 0.001 / 57.3, 8)]))
        left_joint_state_2 = torch.from_numpy(np.array([round(left_read_joint_state.joint_state.joint_2 * 0.001 / 57.3, 8)]))
        left_joint_state_3 = torch.from_numpy(np.array([round(left_read_joint_state.joint_state.joint_3 * 0.001 / 57.3, 8)]))
        left_joint_state_4 = torch.from_numpy(np.array([round(left_read_joint_state.joint_state.joint_4 * 0.001 / 57.3, 8)]))
        left_joint_state_5 = torch.from_numpy(np.array([round(left_read_joint_state.joint_state.joint_5 * 0.001 / 57.3, 8)]))
        left_joint_state_6 = torch.from_numpy(np.array([round(left_read_joint_state.joint_state.joint_6 * 0.001 / 57.3, 8)]))


        left_read_gripper_state = piper_left.GetArmGripperMsgs().gripper_state.grippers_angle
        left_read_gripper_state_norm = round((left_read_gripper_state * 0.001) / 70.0, 8)
        left_gripper_state = torch.from_numpy(np.array([left_read_gripper_state_norm]))


        right_read_joint_state = piper_right.GetArmJointMsgs()
        right_joint_state_1 = torch.from_numpy(
            np.array([round(right_read_joint_state.joint_state.joint_1 * 0.001 / 57.3, 8)]))
        right_joint_state_2 = torch.from_numpy(
            np.array([round(right_read_joint_state.joint_state.joint_2 * 0.001 / 57.3, 8)]))
        right_joint_state_3 = torch.from_numpy(
            np.array([round(right_read_joint_state.joint_state.joint_3 * 0.001 / 57.3, 8)]))
        right_joint_state_4 = torch.from_numpy(
            np.array([round(right_read_joint_state.joint_state.joint_4 * 0.001 / 57.3, 8)]))
        right_joint_state_5 = torch.from_numpy(
            np.array([round(right_read_joint_state.joint_state.joint_5 * 0.001 / 57.3, 8)]))
        right_joint_state_6 = torch.from_numpy(
            np.array([round(right_read_joint_state.joint_state.joint_6 * 0.001 / 57.3, 8)]))


        right_read_gripper_state = piper_right.GetArmGripperMsgs().gripper_state.grippers_angle
        right_read_gripper_state_norm = round((right_read_gripper_state * 0.001) / 70.0, 8)
        right_gripper_state = torch.from_numpy(np.array([right_read_gripper_state_norm]))


        joint_state.append(left_joint_state_1)
        joint_state.append(left_joint_state_2)
        joint_state.append(left_joint_state_3)
        joint_state.append(left_joint_state_4)
        joint_state.append(left_joint_state_5)
        joint_state.append(left_joint_state_6)
        joint_state.append(left_gripper_state)

        joint_state.append(right_joint_state_1)
        joint_state.append(right_joint_state_2)
        joint_state.append(right_joint_state_3)
        joint_state.append(right_joint_state_4)
        joint_state.append(right_joint_state_5)
        joint_state.append(right_joint_state_6)
        joint_state.append(right_gripper_state)
        

        joint_state_values = [
            left_joint_state_1.item(),
            left_joint_state_2.item(),
            left_joint_state_3.item(),
            left_joint_state_4.item(),
            left_joint_state_5.item(),
            left_joint_state_6.item(),
            left_gripper_state.item(),
            right_joint_state_1.item(),
            right_joint_state_2.item(),
            right_joint_state_3.item(),
            right_joint_state_4.item(),
            right_joint_state_5.item(),
            right_joint_state_6.item(),
            right_gripper_state.item()
        ]

        print("joint_state (14-dim):", joint_state_values)
        time.sleep(0.005)
        pass