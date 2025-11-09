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
    piper_left = C_PiperInterface("can0")
    piper_left.ConnectPort()
    piper_left.EnableArm(7)
    enable_fun(piper=piper_left)

    # 连接上右机械臂
    piper_right = C_PiperInterface("can1")
    piper_right.ConnectPort()
    piper_right.EnableArm(7)
    enable_fun(piper=piper_right)


    factor = 57324.840764  # 1000*180/3.14

    while True:
        # 全0位置
        position = [0, 0, 0, 0, 0, 0, 1]
        left_pos = [0, 0, 0, 0, 0, 0, 1]

        


        # 第一阶段
        left_pos = [-0.3603663742542267,
                    1.6800129413604736,
                    -1.8956528902053833,
                    0.23706620931625366,
                    0.7979092001914978,
                    -0.46836790442466736,
                    1.0]
        right_pos = [0.34830811619758606,
                    1.6340149641036987,
                    -1.870426893234253,
                    -0.2440505474805832,
                    0.8208054900169373,
                    0.45904210209846497,
                    1.0]

        # 第二阶段
        left_pos = [-0.11027884483337402,
                    1.7975748777389526,
                    -1.7928476333618164,
                    0.14957267045974731,
                    0.6700260043144226,
                    -0.3282527029514313,
                    1.0]
        right_pos = [0.11376554518938065,
                    1.8936513662338257,
                    -1.8519798517227173,
                    -0.11229217797517776,
                    0.6949493288993835,
                    0.17838777601718903,
                    1.0]


        # 第三阶段 目标抓取点
        left_pos = [-0.0108157554641366, 2.155057907104492, -1.6888703107833862, 0.011970444582402706, 0.6170534491539001, -0.019101770594716072, 1]
        right_pos = [-0.11016009747982025, 2.291276216506958, -1.9080159664154053, 0.04827946424484253, 0.7143278121948242, -0.1274968385696411, 1]

        # 第四阶段 目标抓取点 抓起来
        left_pos = [-0.0108157554641366, 2.155057907104492, -1.6888703107833862, 0.011970444582402706, 0.6170534491539001, -0.019101770594716072, 0.3]
        right_pos = [-0.11016009747982025, 2.291276216506958, -1.9080159664154053, 0.04827946424484253, 0.7143278121948242, -0.1274968385696411, 0.2]

        # 第五阶段
        left_pos = [-0.007099011447280645,
                    1.9656808376312256,
                    -1.695284366607666,
                    0.017518533393740654,
                    0.49078014492988586,
                    -0.018388209864497185,
                    0.3]
        right_pos = [-0.08009626716375351,
                    2.202010154724121,
                    -1.8663830757141113,
                    0.06570275127887726,
                    0.6138795614242554,
                    -0.12093475461006165,
                    0.2]

        # 第六阶段
        left_pos = [-0.03662741929292679,
                    1.8841909170150757,
                    -1.6492260694503784,
                    0.10993267595767975,
                    0.4380464553833008,
                    -0.054775040596723557,
                    0.3]
        right_pos = [-0.086286261677742,
                    2.11618971824646,
                    -1.869980812072754,
                    0.06601328402757645,
                    0.5663893818855286,
                    -0.12062764912843704,
                    0.2]

        # 第七阶段 目标放置点
        left_pos = [-0.48458826541900635, 
                    1.8510364294052124, 
                    -1.339819312095642,
                    0.5357859134674072,
                    0.5000568628311157,
                    -0.9217303395271301,
                    0.3]
        right_pos = [0.4061899185180664,
                    1.8742057085037231,
                    -1.351300597190857,
                    -0.5426676273345947,
                    0.5037045478820801,
                    0.9289683103561401,
                    0.2]

        # 第八阶段 目标放手
        left_pos = [-0.48458826541900635, 
                    1.8510364294052124, 
                    -1.339819312095642,
                    0.5357859134674072,
                    0.5000568628311157,
                    -0.9217303395271301,
                    1.0]
        right_pos = [0.4061899185180664,
                    1.8742057085037231,
                    -1.351300597190857,
                    -0.5426676273345947,
                    0.5037045478820801,
                    0.9289683103561401,
                    1.0]

        position = [0, 0, 0, 0, 0, 0, 1]
        right_pos = [0, 0, 0, 0, 0, 0, 1]

        left_joint_0 = round(left_pos[0] * factor)                                 
        left_joint_1 = round(left_pos[1] * factor)
        left_joint_2 = round(left_pos[2] * factor)
        left_joint_3 = round(left_pos[3] * factor)
        left_joint_4 = round(left_pos[4] * factor)
        left_joint_5 = round(left_pos[5] * factor)
        left_joint_6 = round(position[6] * 70 * 1000)

        right_joint_0 = round(right_pos[0] * factor)
        right_joint_1 = round(right_pos[1] * factor)
        right_joint_2 = round(right_pos[2] * factor)
        right_joint_3 = round(right_pos[3] * factor)
        right_joint_4 = round(right_pos[4] * factor)
        right_joint_5 = round(right_pos[5] * factor)
        right_joint_6 = round(position[6] * 70 * 1000)



        piper_left.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
        piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)



        # 控制右机械臂
        piper_right.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
        piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
        time.sleep(0.005)
        pass