"""执行器模型：动态延迟、幅度饱和与方向不对称变化限制。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .capability import Capability
from .plants.base import Plant


@dataclass(frozen=True)
class QueuedCommand:
    command_id: int
    action: np.ndarray
    submitted_step: int
    due_step: int
    actuator: str = "a"

    def to_dict(self) -> dict:
        return {
            "command_id": int(self.command_id),
            "action": np.asarray(self.action, dtype=float).tolist(),
            "submitted_step": int(self.submitted_step),
            "due_step": int(self.due_step),
            "actuator": str(self.actuator),
        }


class Actuator:
    def __init__(self, plant: Plant, delay_steps: int = 0,
                 withdraw_allowed: bool = False,
                 capability: Capability | None = None,
                 name: str = "a",
                 command_id_offset: int = 0):
        if delay_steps < 0:
            raise ValueError("delay_steps 必须非负")
        self.plant = plant
        self.name = str(name)
        if capability is None:
            self.capability = Capability.from_plant(
                plant, delay_steps, withdraw_allowed
            )
        else:
            self.capability = Capability(
                float(capability.u_min), float(capability.u_max),
                float(capability.du_up), float(capability.du_down),
                int(capability.delay_steps), bool(capability.withdraw_allowed),
            )
        neutral = np.asarray(plant.neutral_action(), dtype=float).copy()
        self.previous_applied = neutral.copy()
        self.last_applied_command_id = -1
        self.last_applied_due_step = -1
        self._next_command_id = int(command_id_offset)
        self._submit_count = 0
        self._queue: deque[QueuedCommand] = deque()
        for due in range(self.capability.delay_steps):
            self._append(
                neutral.copy(),
                submitted_step=-self.capability.delay_steps + due,
                due_step=due,
            )

    def reset(self) -> None:
        neutral = np.asarray(self.plant.neutral_action(), dtype=float).copy()
        self.previous_applied = neutral.copy()
        self.last_applied_command_id = -1
        self.last_applied_due_step = -1
        current = self.capability
        self.capability = Capability.from_plant(
            self.plant, current.delay_steps, current.withdraw_allowed
        )
        self._next_command_id = 0
        self._submit_count = 0
        self._queue = deque()
        for due in range(self.capability.delay_steps):
            self._append(
                neutral.copy(),
                submitted_step=-self.capability.delay_steps + due,
                due_step=due,
            )

    def _append(self, action: np.ndarray, submitted_step: int,
                due_step: int) -> QueuedCommand:
        item = QueuedCommand(
            command_id=self._next_command_id,
            action=np.asarray(action, dtype=float).reshape(-1).copy(),
            submitted_step=int(submitted_step),
            due_step=int(due_step),
            actuator=self.name,
        )
        self._next_command_id += 1
        self._queue.append(item)
        return item

    def pending(self) -> list[np.ndarray]:
        return [item.action.copy() for item in self._queue]

    def pending_commands(self) -> list[QueuedCommand]:
        return list(self._queue)

    def to_state(self) -> dict:
        return {
            "capability": self.capability.to_dict(),
            "name": self.name,
            "previous_applied": self.previous_applied.tolist(),
            "next_command_id": int(self._next_command_id),
            "last_applied_command_id": int(self.last_applied_command_id),
            "last_applied_due_step": int(self.last_applied_due_step),
            "queue": [item.to_dict() for item in self._queue],
        }

    def load_state(self, state: dict) -> None:
        data = self.capability.to_dict()
        data.update(state.get("capability", {}))
        self.name = str(state.get("name", self.name))
        self.capability = Capability(
            float(data["u_min"]), float(data["u_max"]),
            float(data["du_up"]), float(data["du_down"]),
            int(data["delay_steps"]), bool(data.get("withdraw_allowed", False)),
        )
        self.previous_applied = np.asarray(
            state["previous_applied"], dtype=float
        ).reshape(-1).copy()
        self._next_command_id = int(state["next_command_id"])
        self._submit_count = 0
        self.last_applied_command_id = int(state.get("last_applied_command_id", -1))
        self.last_applied_due_step = int(state.get("last_applied_due_step", -1))
        self._queue = deque(
            QueuedCommand(
                int(item["command_id"]),
                np.asarray(item["action"], dtype=float),
                int(item["submitted_step"]),
                int(item["due_step"]),
                str(item.get("actuator", self.name)),
            )
            for item in state.get("queue", [])
        )

    def save_state(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as f:
            json.dump(self.to_state(), f, ensure_ascii=False, indent=2)
        return destination

    def load_state_file(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as f:
            self.load_state(json.load(f))

    def projected_pending(self) -> list[np.ndarray]:
        projected: list[np.ndarray] = []
        previous = self.previous_applied.copy()
        cap = self.capability
        for item in self._queue:
            lower = max(cap.u_min, float(previous[0]) - cap.du_down)
            upper = min(cap.u_max, float(previous[0]) + cap.du_up)
            value = float(np.clip(item.action[0], lower, upper))
            previous = np.array([value])
            projected.append(previous.copy())
        return projected

    def update_capability(self, step: int, capability: Capability) -> dict:
        old = self.capability
        delay_delta = int(capability.delay_steps - old.delay_steps)
        retained = [item.command_id for item in self._queue]
        inserted_neutral = 0

        if delay_delta > 0:
            neutral = np.asarray(self.plant.neutral_action(), dtype=float)
            existing = list(self._queue)
            shifted: deque[QueuedCommand] = deque()
            for offset in range(delay_delta):
                item = QueuedCommand(
                    command_id=self._next_command_id,
                    action=neutral.copy(),
                    submitted_step=step,
                    due_step=step + offset,
                    actuator=self.name,
                )
                self._next_command_id += 1
                shifted.append(item)
                inserted_neutral += 1
            for item in existing:
                shifted.append(QueuedCommand(
                    command_id=item.command_id,
                    action=item.action.copy(),
                    submitted_step=item.submitted_step,
                    due_step=item.due_step + delay_delta,
                    actuator=item.actuator,
                ))
            self._queue = shifted

        self.capability = capability
        return {
            "step": int(step),
            "old": old.to_dict(),
            "new": capability.to_dict(),
            "retained_command_ids": retained,
            "withdrawn_command_ids": [],
            "withdraw_allowed": bool(capability.withdraw_allowed),
            "inserted_neutral_commands": inserted_neutral,
            "pending_after": [item.to_dict() for item in self._queue],
        }

    def withdraw_pending(self, step: int) -> dict:
        before = [item.command_id for item in self._queue]
        if not self.capability.withdraw_allowed:
            return {
                "step": int(step),
                "allowed": False,
                "withdrawn_command_ids": [],
                "retained_command_ids": before,
            }
        self._queue.clear()
        return {
            "step": int(step),
            "allowed": True,
            "withdrawn_command_ids": before,
            "retained_command_ids": [],
        }

    def submit(self, command: np.ndarray, step: int | None = None) -> np.ndarray:
        """提交当前规划命令，并返回本周期真正生效的命令。"""
        current_step = self._submit_count if step is None else int(step)
        self._submit_count += 1
        requested = np.asarray(command, dtype=float).reshape(-1).copy()
        if self._queue:
            due_step = max(
                self._queue[-1].due_step + 1,
                current_step + self.capability.delay_steps,
            )
        else:
            due_step = current_step + self.capability.delay_steps
        self._append(requested, current_step, due_step)

        if self._queue and self._queue[0].due_step <= current_step:
            due = self._queue[0]
            while self._queue and self._queue[0].due_step <= current_step:
                self._queue.popleft()
            return self._apply_safety(due)
        return self.previous_applied.copy()

    def _apply_safety(self, due: QueuedCommand) -> np.ndarray:
        cap = self.capability
        lower = max(cap.u_min, float(self.previous_applied[0]) - cap.du_down)
        upper = min(cap.u_max, float(self.previous_applied[0]) + cap.du_up)
        applied = np.array([float(np.clip(due.action[0], lower, upper))])
        self.previous_applied = applied.copy()
        self.last_applied_command_id = due.command_id
        self.last_applied_due_step = due.due_step
        return applied
