#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
from typing import (
    Optional,
)
import time
from piper_sdk import *

def enable_fun(piper: C_PiperInterface):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:", enable_flag)
        piper.EnableArm(7)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if (elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)


if __name__ == "__main__":
    # 连接上左机械臂
    piper_left = C_PiperInterface("can1")
    piper_left.ConnectPort()
    piper_left.EnableArm(7)
    enable_fun(piper=piper_left)

    # 连接上右机械臂
    piper_right = C_PiperInterface("can2")
    piper_right.ConnectPort()
    piper_right.EnableArm(7)
    enable_fun(piper=piper_right)


    factor = 57324.840764  # 1000*180/3.14
    count = 0
    while True:
        left_pos = [0, 0, 0, 0, 0, 0, 1]
        left_joint_0 = round(left_pos[0] * factor)                                 
        left_joint_1 = round(left_pos[1] * factor)
        left_joint_2 = round(left_pos[2] * factor)
        left_joint_3 = round(left_pos[3] * factor)
        left_joint_4 = round(left_pos[4] * factor)
        left_joint_5 = round(left_pos[5] * factor)
        left_joint_6 = round(left_pos[6] * 70 * 1000)

        right_pos = [0, 0, 0, 0, 0, 0, 1]
        right_joint_0 = round(right_pos[0] * factor)
        right_joint_1 = round(right_pos[1] * factor)
        right_joint_2 = round(right_pos[2] * factor)
        right_joint_3 = round(right_pos[3] * factor)
        right_joint_4 = round(right_pos[4] * factor)
        right_joint_5 = round(right_pos[5] * factor)
        right_joint_6 = round(position[6] * 70 * 1000)

        # 控制左机械臂
        piper_left.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
        piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)



        # 控制右机械臂
        piper_right.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
        piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
        time.sleep(0.005)
        pass