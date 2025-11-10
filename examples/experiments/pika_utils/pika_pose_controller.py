import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from examples.experiments.pika_utils.transformations import euler_from_quaternion, quaternion_from_euler
import threading
import math
from examples.experiments.pika_utils.forward_inverse_kinematics import Arm_FK, Arm_IK
import argparse

class PikaPoseController:
    def __init__(self, 
                 index_name="",
                 robot_interface=None):
        self.index_name = index_name
        self.baseline_pose = None # 接管时的piper ee pose
        self.baseline_pose_matrix = None   
        self.baseline_pika_pose = None # 接管时的pika sensor pose
        self.baseline_pika_pose_matrix = None
        self.current_piper_pose = None
        self.current_piper_pose_matrix = None
        self.current_pika_pose = None
        self.current_pika_pose_matrix = None
        self.intervention_action = False

        # TODO: 0.19是什么? urdf路径
        args = argparse.Namespace()
        args.index_name = index_name
        args.gripper_xyzrpy = [0.19, 0.0, 0.0, 0.0, 0.0, 0.0]
        args.lift = False
        
        self.arm_fk = Arm_FK(args)
        self.arm_ik = Arm_IK(args)
        self.robot_interface = robot_interface
        rospy.Subscriber('/pika_pose',PoseStamped, self.pika_pose_callback)
        self.lock = threading.Lock()

    def pika_pose_callback(self, msg):
        with self.lock:
            self.current_pika_pose = self._pose_msg_to_array(msg)
            self.current_pika_pose_matrix = self.create_transformation_matrix(
                *self.current_pika_pose
            )

    def _pose_msg_to_array(self, pose_msg):
        """将PoseStamped消息转换为6D数组 [x, y, z, roll, pitch, yaw]"""
        pos = pose_msg.pose.position
        ori = pose_msg.pose.orientation
        
        # 注意：这里根据teleop脚本，orientation直接存储的是RPY而不是quaternion
        if hasattr(ori, 'x') and hasattr(ori, 'y') and hasattr(ori, 'z'):
            # 如果是RPY格式
            return np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z])
        else:
            # 如果是quaternion格式，需要转换
            roll, pitch, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            return np.array([pos.x, pos.y, pos.z, roll, pitch, yaw])
    
    def set_robot_interface(self, robot_interface):
        """设置机械臂接口"""
        self.robot_interface = robot_interface

    def get_current_joint_positions(self):
        """从机械臂接口获取当前关节位置"""
        if self.robot_interface is None:
            return None
        
        # 从 SinglePiper 获取关节状态
        joint_state = self.robot_interface.GetArmJointMsgs()
        joint_positions = np.array([
            joint_state.joint_state.joint_1 * 0.001 / 57.3,
            joint_state.joint_state.joint_2 * 0.001 / 57.3,
            joint_state.joint_state.joint_3 * 0.001 / 57.3,
            joint_state.joint_state.joint_4 * 0.001 / 57.3,
            joint_state.joint_state.joint_5 * 0.001 / 57.3,
            joint_state.joint_state.joint_6 * 0.001 / 57.3,
        ])
        print("get demo joint positions:", joint_positions)
        return joint_positions
    
    def get_current_end_effector_pose(self):
        """通过FK计算当前末端执行器位姿"""
        joint_positions = self.get_current_joint_positions()
        if joint_positions is None:
            print("No joint positions available")
            return None
        try:
            # 使用 FK 计算末端位姿
            pose_xyzrpy = self.arm_fk.get_pose(joint_positions)
            return self.create_transformation_matrix(*pose_xyzrpy)
        except Exception as e:
            print(f"[PikaPoseController] Error computing FK: {e}")
            return None

    def start_intervention(self):
        with self.lock:
            self.current_piper_pose_matrix = self.get_current_end_effector_pose()
            if self.baseline_pose_matrix is not None or self.baseline_pika_pose_matrix is not None:
                return
            if self.current_piper_pose_matrix is not None and self.current_pika_pose is not None:
                
                self.baseline_pose_matrix = self.current_piper_pose_matrix.copy()
                self.baseline_pika_pose = self.current_pika_pose.copy()
                self.baseline_pika_pose_matrix = self.create_transformation_matrix(*self.baseline_pika_pose)
                self.intervention_action = True
                print("[PikaPoseController] Intervention started.")
            # else:
            #     print("[PikaPoseController] Warning: Current poses are not available yet.")

    def stop_intervention(self):
        with self.lock:
            self.intervention_action       = False
            self.baseline_pose             = None
            self.baseline_pika_pose        = None
            self.baseline_pika_pose_matrix = None
            self.baseline_pose_matrix      = None
            print("[PikaPoseController] Intervention stopped.")

    def calculate_piper_goal_pose(self):
        """
        goal_pose = baseline_pose_matrix * inv(baseline_pika_pose_matrix) * current_pika_pose_matrix
        """
        with self.lock:
            if not self.intervention_action or self.baseline_pose_matrix is None or self.baseline_pika_pose_matrix is None:
                # print(f"self.intervention_action {self.intervention_action}")
                # print(f"self.baseline_pose_matrix {self.baseline_pose_matrix}")
                # print(f"self.baseline_pika_pose_matrix {self.baseline_pika_pose_matrix}")
                # print("3333")
                return None
            
            self.current_piper_pose_matrix = self.get_current_end_effector_pose()
            # print(f"ciurrent_piper_pose_matrix in cal:{self.current_piper_pose_matrix}")

        pika_relative_transform = np.dot(np.linalg.inv(self.baseline_pika_pose_matrix), 
                                        self.current_pika_pose_matrix)
        goal_pose_matrix = np.dot(self.baseline_pose_matrix, pika_relative_transform)
        goal_pose_xyzrpy = self.matrix_to_xyzrpy(goal_pose_matrix)
        return goal_pose_xyzrpy
    
    def get_target_joint_action(self):
        goal_pose_xyzrpy = self.calculate_piper_goal_pose()
        if goal_pose_xyzrpy is None:
            print("all None")
            return np.zeros(7)  # 返回7维零动作（6个关节+夹爪）
    

        q = quaternion_from_euler(goal_pose_xyzrpy[3], goal_pose_xyzrpy[4], goal_pose_xyzrpy[5])
        import pinocchio as pin
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([goal_pose_xyzrpy[0], goal_pose_xyzrpy[1], goal_pose_xyzrpy[2]]),
        )
        # 使用IK求解目标关节角度
        target_joints, tau_ff, success = self.arm_ik.ik_fun(target.homogeneous, 0.0)
        if not success:
            print("[PikaPoseController] IK solution failed")
            return np.zeros(7)
        current_joints = self.get_current_joint_positions()
        if current_joints is None:
            return np.zeros(7)
        joint_action = target_joints[:6]
        # 添加夹爪动作（保持当前状态或根据需要设置）
        gripper_action = 0.0  # 或者根据需要设置夹爪动作
        full_action = np.concatenate([joint_action, [gripper_action]])
        print(f"full_action from pika controller:{full_action}")
        return full_action
    
        
    def create_transformation_matrix(self, x, y, z, roll, pitch, yaw):
        """创建4x4齐次变换矩阵"""
        transformation_matrix = np.eye(4)
        A = np.cos(yaw)
        B = np.sin(yaw)
        C = np.cos(pitch)
        D = np.sin(pitch)
        E = np.cos(roll)
        F = np.sin(roll)
        DE = D * E
        DF = D * F
        transformation_matrix[0, 0] = A * C
        transformation_matrix[0, 1] = A * DF - B * E
        transformation_matrix[0, 2] = B * F + A * DE
        transformation_matrix[0, 3] = x
        transformation_matrix[1, 0] = B * C
        transformation_matrix[1, 1] = A * E + B * DF
        transformation_matrix[1, 2] = B * DE - A * F
        transformation_matrix[1, 3] = y
        transformation_matrix[2, 0] = -D
        transformation_matrix[2, 1] = C * F
        transformation_matrix[2, 2] = C * E
        transformation_matrix[2, 3] = z
        transformation_matrix[3, 0] = 0
        transformation_matrix[3, 1] = 0
        transformation_matrix[3, 2] = 0
        transformation_matrix[3, 3] = 1
        return transformation_matrix
    
    def matrix_to_xyzrpy(self, matrix):
        """从4x4变换矩阵提取位置和姿态"""
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.asin(-matrix[2, 0])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        return np.array([x, y, z, roll, pitch, yaw])