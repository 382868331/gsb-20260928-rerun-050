"""被控对象统一接口。

后续替换被控对象（不同动力学）时，只需实现本接口，并保持观测维度、
动作维度及 observation_bounds/action_bounds 契约一致，控制器与实验
框架无需改动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Plant(ABC):
    """离散时间被控对象基类。"""

    n_obs: int
    n_act: int
    dt: float

    @abstractmethod
    def reset(self, initial_state: np.ndarray | None = None) -> np.ndarray:
        """复位内部状态，返回首次观测。"""

    @abstractmethod
    def step(self, action: np.ndarray, disturbance: np.ndarray | None = None) -> np.ndarray:
        """执行一个控制周期，返回执行后的观测。"""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """返回当前内部状态。"""

    @abstractmethod
    def observation_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """返回观测的 (下界, 上界)，形状均为 (n_obs,)。"""

    @abstractmethod
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """返回动作的 (下界, 上界)，形状均为 (n_act,)。"""

    @abstractmethod
    def action_rate_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """返回相邻两步动作差的 (下界, 上界)，形状均为 (n_act,)。"""

    @abstractmethod
    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """标称（无扰动）预测模型，供规划器使用。"""

    def neutral_action(self) -> np.ndarray:
        """零动作（自然保守动作），默认取动作范围中点。"""
        lo, hi = self.action_bounds()
        return (lo + hi) / 2.0
