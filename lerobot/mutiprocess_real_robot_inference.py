# 导入机器人模型
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
    moving_dual_piper,
)
# 导入 util
from lerobot.common.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from dataclasses import asdict, dataclass
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
# 导入 policy
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.configs import parser
# piper sdk
from piper_sdk import *

import numpy as np
import torch
import time
import threading
import multiprocessing
from queue import Empty, Full, Queue
from dataclasses import dataclass
import logging
from threading import BrokenBarrierError

from lerobot.common.utils.visualization_utils import _init_rerun
import rerun as rr

# 配置日志记录，使其在多进程中更清晰
logging.basicConfig(level=logging.INFO, format='[%(levelname)s|%(processName)s] %(message)s')

@dataclass
class InitialConfig:
    robot: RobotConfig
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    ckpt_path: str = None
    task: str = None
    activate_mobile_base: bool = False  # 是否启用移动底盘控制（仅对 moving_dual_piper 有效）

    def __post_init__(self):

        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
    

# ===================================================================
#                      辅助初始化函数
# ===================================================================

def init_policy_in_process(cfg: InitialConfig):
    """在子进程中创建并加载策略模型"""
    logging.info("Initializing policy...")
    device = "cuda"
    # policy = ACTPolicy.from_pretrained(cfg.ckpt_path) # 示例
    if cfg.policy.type == "act":
        policy = ACTPolicy.from_pretrained(cfg.ckpt_path)
    elif cfg.policy.type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(cfg.ckpt_path)
    elif cfg.policy.type == "smolvla":
        policy = SmolVLAPolicy.from_pretrained(cfg.ckpt_path)
    else:
        raise ValueError("You need to provide a valid policy between act/diffusion/smolvla.")
    policy.eval()
    policy.to(device)
    policy.reset()
    logging.info(f"Policy loaded successfully on {device}.")


    return policy

# ===================================================================
#                 使能函数 (带错误处理)
# ===================================================================
def enable_fun(piper: C_PiperInterface):
    """
    使能机械臂并检测使能状态,尝试5s,如果使能超时则抛出异常
    """
    enable_flag = False
    timeout = 5
    start_time = time.time()
    
    while not enable_flag:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise RuntimeError("机械臂使能超时!") # 抛出异常而不是退出

        # ... (与您之前代码相同的使能状态检测逻辑)
        # piper.GetArmLowSpdInfoMsgs()...
        enable_flag = True # 假设使能成功
        logging.info("正在尝试使能机械臂...")
        piper.EnableArm(7)
        piper.GripperCtrl(1 * 80 * 1000, 1000, 0x01, 0)
        time.sleep(1)
        
    logging.info("机械臂使能成功。")



# ===================================================================
#                 感知进程 (Perception Process)
# ===================================================================
def perception_process_main(perception_queue: multiprocessing.Queue, cfg: InitialConfig, stop_event: multiprocessing.Event, barrier: multiprocessing.Barrier):
    try:
        logging.info("Initializing...")
        robot = make_robot_from_config(cfg.robot)
        if not robot.is_connected:
            robot.connect()
        logging.info("Robot connection successful.")

        _init_rerun(session_name="real_robot_inference")
        
        logging.info("✅ READY. Waiting for start signal.")
        barrier.wait() # 第一阶段同步：报告已就绪
        barrier.wait() # 第二阶段同步：等待开始指令

    except Exception as e:
        logging.error("Initialization failed:", exc_info=True) 
        barrier.abort()
        return

    logging.info("🚀 STARTING perception loop.")
    TARGET_PERIOD = 1.0 / 30
    
    while not stop_event.is_set():
        loop_start = time.perf_counter()
        
        observation = robot.get_observation()
        observation_frame = {}

        # 用于保存 state 数值
        state_values = []

        for key, value in observation.items():
            # 把所有带 .pos 的 float/ndarray 合并成 state
            if key.endswith('.pos'):
                state_values.append(np.float32(value))

            # 把值为 HWC 格式的 ndarray（图像）保存为 observation.images.{key}
            elif isinstance(value, np.ndarray) and value.ndim == 3:
                observation_frame[f'observation.images.{key}'] = value

        # 添加合并后的状态
        observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)

        for name in observation_frame:
            if "image" in name:
                # 图像预处理
                obs = observation_frame[name].astype(np.float32) / 255.0
                obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
            else:
                obs = observation_frame[name]  # 例如 state 是 np.ndarray

            # 增加 batch 维度
            obs = np.expand_dims(obs, axis=0)
            observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device="cpu")
        if cfg.task == None:
            raise ValueError("You need to provide a task name.")
        observation_frame["task"] = [cfg.task]
        # 可视化当前观测数据
        for obs, val in observation.items():
            
            if isinstance(val, float):
                rr.log(f"observation.{obs}", rr.Scalar(val))
            elif isinstance(val, np.ndarray):
                rr.log(f"observation.{obs}", rr.Image(val), static=True)
        
        # --- 放入队列 (非阻塞，size=1时会自动覆盖旧数据) ---
        try:
            # 先清空，再放入，确保最新
            while not perception_queue.empty():
                perception_queue.get_nowait()
            perception_queue.put_nowait(observation_frame)
        except Full:
            pass # 队列已满（理论上清空后不会发生），忽略
            
        # --- 频率控制 ---
        elapsed = time.perf_counter() - loop_start
        sleep_time = TARGET_PERIOD - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    logging.info("Peroception Process Shutting down.")



# ===================================================================
#                    推理进程 (Inference Process)
# ===================================================================
def inference_process_main(perception_queue: multiprocessing.Queue, action_queue: multiprocessing.Queue, cfg: InitialConfig, stop_event: multiprocessing.Event, barrier: multiprocessing.Barrier):
    try:
        logging.info("Initializing Inference Porcess...")
        policy = init_policy_in_process(cfg)
        
        logging.info("✅ READY. Waiting for start signal.")
        barrier.wait()
        barrier.wait()
        
    except Exception as e:
        logging.error(f"Initialization failed", exc_info=True)
        barrier.abort()
        return

    logging.info("🚀 STARTING inference loop.")
    is_fir = True
    while not stop_event.is_set():
        try:
            # --- 获取最新感知数据 (阻塞式获取，但因为上游频率固定，可以接受) ---
            observation_frame = perception_queue.get().copy()

            # 遍历字典，把里面的 numpy / CPU tensor 搬到 GPU
            for k, v in list(observation_frame.items()):
                if k == "task":
                    continue  # task 是字符串，不需要 to()
                if isinstance(v, torch.Tensor):
                    v = v.to("cuda", non_blocking=True)  # 搬到 GPU
                observation_frame[k] = v
            
            # --- 生成推理结果 ---
            with torch.inference_mode():
                action = policy.select_action(observation_frame)

            numpy_action = action.squeeze(0).cpu().numpy()  # 去掉 batch 维，转到 cpu，再转 numpy
            position = numpy_action.tolist()                # 转成 python list，方便后续使用

            if is_fir:
                is_fir = False
                print(f"fir inference action: {position}")

            # --- 放入动作队列 (非阻塞，size=1时会自动覆盖旧数据) ---
            try:
                while not action_queue.empty():
                    action_queue.get_nowait()
                action_queue.put_nowait(position)
            except Full:
                pass

        except Empty:
            time.sleep(0.001) # 如果队列为空，稍微等待

    logging.info("Inference Process Shutting down.")


# ===================================================================
#                   执行进程 (Execution Process)
# ===================================================================
def execution_process_main(action_queue: multiprocessing.Queue, cfg: "InitialConfig", stop_event: multiprocessing.Event, barrier: multiprocessing.Barrier):
    """
    The main function for the execution process. It handles:
    1.  Independent initialization of the robot and hardware.
    2.  Synchronization with other processes using a barrier.
    3.  A 200Hz real-time loop to send commands.
    4.  Graceful shutdown on signal.
    """
    # ===================================================================
    #   1. INITIALIZATION & SYNCHRONIZATION
    # ===================================================================
    try:
        logging.info("Initializing...")
        

        # Enable arms within this process
        logging.info("Enabling robot arms...")
        if cfg.robot.type == "single_piper":
            piper = C_PiperInterface("can1")
            piper.ConnectPort()
            piper.EnableArm(7)
            enable_fun(piper=piper)
            piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        elif cfg.robot.type in ["dual_piper", "moving_dual_piper"]:
            # 连接上左机械臂
            piper_left = C_PiperInterface("can1")
            piper_left.ConnectPort()
            piper_left.EnableArm(7)

            # 连接上右机械臂
            piper_right = C_PiperInterface("can2")
            piper_right.ConnectPort()
            piper_right.EnableArm(7)
            enable_fun(piper=piper_left)
            piper_left.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            enable_fun(piper=piper_right)
            piper_right.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        
        logging.info("✅ READY. Waiting for start signal.")
        # First wait: Signal that this process is initialized and ready
        barrier.wait()
        # Second wait: Wait for the main process (after user input) to give the 'go' signal
        barrier.wait()

    except Exception as e:
        logging.error(f"Initialization failed", exc_info=True)
        # Abort the barrier to prevent other processes from waiting forever
        barrier.abort()
        return
        
    logging.info("🚀 STARTING execution loop.")

    # ===================================================================
    #   2. REAL-TIME CONTROL LOOP
    # ===================================================================
    factor = 57324.840764  # 1000 * 180 / 3.14
    sdk_command_queue = Queue(maxsize=1)
    
    def sdk_sender_thread():
        """
        This thread is dedicated to sending commands via the SDK, 
        isolating potential blocking from the main 200Hz loop.
        """
        # Initialize ROS publisher only if needed and only within this thread
        cmd_vel_pub = None
        if cfg.robot.type == "moving_dual_piper" and cfg.activate_mobile_base:
            try:
                import rospy
                from geometry_msgs.msg import Twist
                # Ensure ROS node is unique if multiple processes use it
                if not rospy.core.is_initialized():
                    rospy.init_node(f"cmd_vel_pub_{multiprocessing.current_process().pid}", anonymous=True)
                cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
            except (ImportError, ModuleNotFoundError):
                logging.warning("ROS libraries not found. Cannot control mobile base.")
            except Exception as e:
                logging.error(f"Failed to initialize ROS publisher", exc_info=True)

        if_first = True
        count = 0

        while True:
            position = sdk_command_queue.get() 
            if if_first == True and position[2] != 0:
                print(f"command: {position}")
                if_first = False
            if position is None:  # Sentinel value for graceful shutdown
                break

            try:
                # --- This is your detailed action parsing and command logic ---
                joint_limits = [(-3, 3)] * 6
                joint_limits[0] = (-2.687, 2.687); joint_limits[1] = (0.0, 3.403)
                joint_limits[2] = (-3.0541012, 0.0); joint_limits[3] = (-1.5499, 1.5499)
                joint_limits[4] = (-1.22, 1.22); joint_limits[5] = (-1.7452, 1.7452)
                def clamp(value, min_val, max_val):
                    return max(min(value, max_val), min_val)

                if cfg.robot.type == "single_piper":
                    joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
                    joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
                    joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
                    joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
                    joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
                    joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
                    print(f"position[6] : {position[6]}")
                    if position[6] < 0.8:
                        position[6] = 0.7
                    joint_6 = round(position[6] * 70 * 1000)
                    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                elif cfg.robot.type == "dual_piper":
                    # Left arm control (position 0-5)
                    left_joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
                    left_joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
                    left_joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
                    left_joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
                    left_joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
                    left_joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
                    left_joint_6 = round(position[6] * 70 * 1000)
                    piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
                    piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)

                    # Right arm control (position 7-12)
                    right_joint_0 = round(clamp(position[7], joint_limits[0][0], joint_limits[0][1]) * factor)
                    right_joint_1 = round(clamp(position[8], joint_limits[1][0], joint_limits[1][1]) * factor)
                    right_joint_2 = round(clamp(position[9], joint_limits[2][0], joint_limits[2][1]) * factor)
                    right_joint_3 = round(clamp(position[10], joint_limits[3][0], joint_limits[3][1]) * factor)
                    right_joint_4 = round(clamp(position[11], joint_limits[4][0], joint_limits[4][1]) * factor)
                    right_joint_5 = round(clamp(position[12], joint_limits[5][0], joint_limits[5][1]) * factor)
                    right_joint_6 = round(position[13] * 70 * 1000)
                    piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
                    piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
                elif cfg.robot.type == "moving_dual_piper":
                    if cfg.activate_mobile_base and cmd_vel_pub is not None:
                        # scout_mini control (position 0-1)
                        twist_msg = Twist()
                        twist_msg.linear.x = position[0]   # 线速度
                        twist_msg.angular.z = position[1]  # 角速度
                        cmd_vel_pub.publish(twist_msg)


                    # Left arm control (position 2-8)
                    left_joint_0 = round(clamp(position[2], joint_limits[0][0], joint_limits[0][1]) * factor)
                    left_joint_1 = round(clamp(position[3], joint_limits[1][0], joint_limits[1][1]) * factor)
                    left_joint_2 = round(clamp(position[4], joint_limits[2][0], joint_limits[2][1]) * factor)
                    left_joint_3 = round(clamp(position[5], joint_limits[3][0], joint_limits[3][1]) * factor)
                    left_joint_4 = round(clamp(position[6], joint_limits[4][0], joint_limits[4][1]) * factor)
                    left_joint_5 = round(clamp(position[7], joint_limits[5][0], joint_limits[5][1]) * factor)

                    if position[8] < 0.5:
                        position[8] = 0.
                    left_joint_6 = round(position[8] * 100 * 1000)
                    # piper_left.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                    # piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
                    # piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)

                    # Right arm control (position 9-15)
                    right_joint_0 = round(clamp(position[9], joint_limits[0][0], joint_limits[0][1]) * factor)
                    right_joint_1 = round(clamp(position[10], joint_limits[1][0], joint_limits[1][1]) * factor)
                    right_joint_2 = round(clamp(position[11], joint_limits[2][0], joint_limits[2][1]) * factor)
                    right_joint_3 = round(clamp(position[12], joint_limits[3][0], joint_limits[3][1]) * factor)
                    right_joint_4 = round(clamp(position[13], joint_limits[4][0], joint_limits[4][1]) * factor)
                    right_joint_5 = round(clamp(position[14], joint_limits[5][0], joint_limits[5][1]) * factor)
                    if position[15] < 0.5:
                        position[15] = 0.
                    right_joint_6 = round(position[15] * 100 * 1000)
                    piper_right.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                    piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
                    piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)

                # --- End of your action logic ---
            except Exception as e:
                logging.error(f"SDK sender thread error", exc_info=False)

            sdk_command_queue.task_done()
        logging.info("SDK sender thread has shut down.")

    # Start the dedicated thread for sending SDK commands
    sender_thread = threading.Thread(target=sdk_sender_thread, name="SDK_Sender", daemon=True)
    sender_thread.start()

    TARGET_FREQ = 300
    TARGET_PERIOD = 1.0 / TARGET_FREQ
    
    action_dim_map = {"single_piper": 7, "dual_piper": 14, "moving_dual_piper": 16}
    action_dim = action_dim_map.get(cfg.robot.type)
    last_action = [0.0] * action_dim
    if cfg.robot.type == "single_piper":
        last_action[-1] = 1.0  # Gripper open
        last_action[:6] = [-0.079173743724823, 0.5733827948570251, -0.28674960136413574, -0.0282147079706192, 0.07699346542358398, -0.03259479999542236]
    elif cfg.robot.type == "dual_piper":
        last_action[8]  = 1.0
        last_action[-1] = 1.0
    elif cfg.robot.type == "moving_dual_piper":
        last_action[2:8] = [-0.19889281690120697, 1.5113189220428467, -0.9937765598297119, -0.031814172863960266, 0.387959748506546, -0.03050227090716362]
        last_action[8]  = 1.0
        last_action[9:15] = [-0.04206351935863495, 1.9215259552001953, -1.5553820133209229, 0.2345607578754425, 0.7912604808807373, -0.11750923842191696]
        last_action[-1] = 1.0
    
    
    # This is the main 200Hz loop. It checks the stop_event for shutdown.
    while not stop_event.is_set():
        loop_start = time.perf_counter()
        
        try:
            new_action = action_queue.get_nowait()
            last_action = new_action
        except Empty:
            pass 

        try:
            while not sdk_command_queue.empty():
                sdk_command_queue.get_nowait()
            sdk_command_queue.put_nowait(last_action)
        except Full:
            pass

        elapsed = time.perf_counter() - loop_start
        sleep_time = TARGET_PERIOD - elapsed
        if sleep_time < 0:
            logging.warning(f"Execution loop missed target! Overrun by {-sleep_time*1000:.2f} ms")
        else:
            time.sleep(sleep_time)


    # 3. SHUTDOWN
    logging.info("Execution Process Shutting down...")
    # Signal the sender thread to exit and wait for it to finish
    sdk_command_queue.put(None)
    sender_thread.join(timeout=2)

    logging.info("Execution process has terminated.")


# ===================================================================
#                  主函数 (程序入口)
# ===================================================================
@parser.wrap()
def main(cfg: InitialConfig):
    # --- 1. 创建同步和通信对象 ---
    stop_event = multiprocessing.Event()
    perception_q = multiprocessing.Queue(maxsize=1)
    action_q = multiprocessing.Queue(maxsize=1)
    
    # Barrier for 3 child processes + 1 main process
    start_barrier = multiprocessing.Barrier(4)

    processes = [
        multiprocessing.Process(name="Execution",  target=execution_process_main, args=(action_q, cfg, stop_event, start_barrier)),
        multiprocessing.Process(name="Perception", target=perception_process_main, args=(perception_q, cfg, stop_event, start_barrier)),
        multiprocessing.Process(name="Inference",  target=inference_process_main, args=(perception_q, action_q, cfg, stop_event, start_barrier)),
    ]

    

    # --- 2. 启动所有进程，它们将进行初始化并等待 ---
    logging.info("🚀 Starting all processes for initialization...")
    for p in processes:
        p.start()

    # --- 3. 等待所有进程初始化完成 ---
    try:
        logging.info("⏳ Main process is waiting for all subprocesses to be ready...")
        start_barrier.wait() # 第一阶段等待：等待所有进程初始化完毕
        
        logging.info("✅ All processes are initialized and ready.")
        input("   Press ENTER to start the real-time control loop... ")
        
        start_barrier.wait() # 第二阶段等待：同步释放所有进程，开始执行
        
        while any(p.is_alive() for p in processes):
            time.sleep(0.5)

    except (KeyboardInterrupt, BrokenPipeError, multiprocessing.context.BrokenBarrierError):
        logging.warning("\n🚨 Shutdown signal received or a process crashed!")
        stop_event.set()
        
    finally:
        logging.info("⏳ Waiting for all processes to terminate...")
        for p in processes:
            # 添加超时以防进程卡死
            p.join(timeout=5)
            if p.is_alive():
                logging.warning(f"Process {p.name} did not terminate gracefully. Forcing shutdown.")
                p.terminate()

        logging.info("✅ All processes have been shut down.")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()