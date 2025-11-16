from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("dual_piper")
@dataclass
class DualPiperConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # 最大相对目标位置的限制
    max_relative_target: int | None = None

    # cameras
    # 定义机器人使用的相机配置字典，键是相机的名称（字符串），值是 CameraConfig 类型的配置对象
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    # 是否使用角度单位（degree）而不是弧度（radian）
    use_degrees: bool = False
