"""集中配置：被控对象参数与控制器参数。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CartConfig:
    """人工一维小车模型参数。

    状态/观测: [位置 x, 速度 v]；动作: 推力 u。
    连续动力学: m * x_ddot = u + d（d 为控制器未知的扰动力）。
    """

    mass: float = 1.0
    dt: float = 0.1
    u_max: float = 2.0          # 动作幅度限制 |u| <= u_max
    du_max: float = 4.0         # 兼容旧配置：对称动作变化限制
    du_up_max: float | None = None    # 动作增大时每步最大增量
    du_down_max: float | None = None  # 动作减小时每步最大减量（正数）
    x_bounds: tuple[float, float] = (-10.0, 10.0)
    v_bounds: tuple[float, float] = (-5.0, 5.0)

    def __post_init__(self) -> None:
        if self.du_up_max is None:
            object.__setattr__(self, "du_up_max", self.du_max)
        if self.du_down_max is None:
            object.__setattr__(self, "du_down_max", self.du_max)


@dataclass(frozen=True)
class MPCConfig:
    """有限时域模型预测控制器参数。"""

    horizon: int = 25           # 预测时域 N（步）
    q_pos: float = 4.0          # 位置偏差权重
    q_vel: float = 0.5          # 速度权重
    r_u: float = 0.02           # 动作能量权重
    q_term_pos: float = 12.0    # 终端位置权重
    q_term_vel: float = 2.0     # 终端速度权重
    brake_gain: float = 6.0     # 规划失败时的保守刹车增益
    max_iter: int = 80
    tol: float = 1e-7
