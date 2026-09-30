"""人工一维小车模型。"""

from __future__ import annotations

import numpy as np

from ..config import CartConfig
from .base import Plant


class CartPlant(Plant):
    """F = u + d 的一维小车，半隐式欧拉离散。

    内部状态 x = [位置, 速度]，与观测一致（全状态反馈）。
    扰动力 d 只作用于真实仿真，规划器的 predict 不含扰动。
    """

    n_obs = 2
    n_act = 1

    def __init__(self, config: CartConfig | None = None):
        self.cfg = config or CartConfig()
        self.dt = self.cfg.dt
        self._state = np.zeros(2)

    def reset(self, initial_state: np.ndarray | None = None) -> np.ndarray:
        if initial_state is None:
            self._state = np.zeros(2)
        else:
            self._state = np.asarray(initial_state, dtype=float).copy()
            if self._state.shape != (2,):
                raise ValueError("initial_state 必须是形状 (2,) 的 [位置, 速度]")
        return self.get_observation()

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def get_observation(self) -> np.ndarray:
        """当前观测 = 全状态。"""
        return self._state.copy()

    def step(self, action: np.ndarray, disturbance: np.ndarray | None = None) -> np.ndarray:
        u = float(np.asarray(action, dtype=float).reshape(-1)[0])
        d = 0.0 if disturbance is None else float(np.asarray(disturbance).reshape(-1)[0])
        pos, vel = self._state
        acc = (u + d) / self.cfg.mass
        # 半隐式欧拉：先更新速度再用新速度更新位置
        new_vel = vel + self.cfg.dt * acc
        new_vel = float(np.clip(new_vel, *self.cfg.v_bounds))
        new_pos = pos + self.cfg.dt * new_vel
        new_pos = float(np.clip(new_pos, *self.cfg.x_bounds))
        self._state = np.array([new_pos, new_vel])
        return self.get_observation()

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """规划器使用的标称模型：无扰动、无饱和（线性模型）。"""
        pos, vel = float(state[0]), float(state[1])
        u = float(np.asarray(action).reshape(-1)[0])
        acc = u / self.cfg.mass
        new_vel = vel + self.cfg.dt * acc
        new_pos = pos + self.cfg.dt * new_vel
        return np.array([new_pos, new_vel])

    def observation_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([self.cfg.x_bounds[0], self.cfg.v_bounds[0]]),
            np.array([self.cfg.x_bounds[1], self.cfg.v_bounds[1]]),
        )

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-self.cfg.u_max]),
            np.array([self.cfg.u_max]),
        )

    def action_rate_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-float(self.cfg.du_down_max)]),
            np.array([float(self.cfg.du_up_max)]),
        )
