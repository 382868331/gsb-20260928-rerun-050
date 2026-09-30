# 控制算法研发系统（人工小车 + 有限时域 MPC）

以一维人工小车为首个被控对象，用 Python 搭建模型、控制器与实验记录，
打通有限时域滚动规划；被控对象通过统一接口替换，观测/动作边界契约保持不变。

本轮加入三类真实链路效应：

- **执行器延迟**：规划动作经 `actuator_delay_steps`（默认 2 步）后才施加。
  在途命令作为固定前缀进入预测，只优化其后的自由动作；`issued_u`（下发命令）
  与 `applied_u`（真正生效动作）分别记录，延迟期间绝不把未生效控制量当成即时效果。
- **方向不对称的动作变化限制**：增力方向 `du_inc_max` 与减力/卸载方向
  `du_dec_max` 可分别配置（默认增力慢、卸载快），幅度与两个方向的变化斜率
  在每一步生效动作上核对。
- **观测短时缺失**：丢观测时用标称模型 + 真实生效动作做开环估计并短时保持
  规划；连续缺失超过 `max_dropout_hold` 步后放弃预测、强制保守刹车；
  观测恢复后重新对齐真实状态并收敛。

## 结构

- `control_lab/plants/base.py`：被控对象统一接口（`step`/`predict`/边界）。
- `control_lab/plants/cart.py`：人工小车 `m*x_ddot = u + d`（d 仅真实仿真存在），
  方向不对称的动作变化边界。
- `control_lab/controller.py`：延迟感知的有限时域 MPC（在途命令前缀固定 +
  自由段二次规划，SLSQP + 精确梯度）；规划失败或观测超时输出明确的保守刹车。
- `control_lab/experiment.py`：闭环循环（延迟 FIFO 队列、丢观测开环估计/超时
  退化）与可重放的 `ScenarioSpec`/`replay`。
- `control_lab/logger.py` / `analysis.py`：逐步记录下发 vs 生效动作、真实 vs
  估计状态、在途队列、丢观测计数，并做预测-实际差异分析与动作边界核对。
- `experiments/run_scenarios.py`：持续扰动、严格限制下目标突变、连续丢观测+
  延迟恢复、规划器故障四个场景。
- `tests/`：动力学、不对称约束、延迟 FIFO、丢观测退化、边界核对、可重放等测试。

## 运行

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.run_scenarios
```

场景结果写入 `results/<场景>/`：`steps.csv`（逐步记录，含 `issued_u`/`applied_u`/
`x_true`/`x`(估计)/`obs_available`/`committed_*`）、`plans.json`（每周期含在途
前缀的完整预测）、`plan_vs_actual.csv`（预测 vs 真实）、`scenario.json`（场景
规格，可重放）、`plot.png`、`results/summary.json`（含逐步动作边界核对结果与
保守动作原因统计）。

## 重放

```python
from control_lab.experiment import replay
log = replay("results/03_observation_dropout/scenario.json", out_dir="/tmp/replay")
```

重放基于保存的确定性序列（目标、扰动、观测可用性、故障周期、全部配置），
输出与原始实验逐行一致，用于核对每一步动作边界及观测恢复过程。

## 替换被控对象

新对象实现 `control_lab.plants.base.Plant`：保持 `n_obs`/`n_act` 与
`observation_bounds`/`action_bounds`/`action_rate_bounds` 语义一致，提供规划用
标称模型 `predict`；控制器与实验框架即可复用（线性 MPC 的 `A/B` 需按新模型
调整或改用通用非线性优化）。
