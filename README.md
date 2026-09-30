# 控制算法研发系统（人工小车 + 有限时域 MPC）

以一维人工小车为首个被控对象，用 Python 搭建模型、控制器与实验记录，
打通有限时域滚动规划；被控对象通过统一接口替换，观测/动作边界契约保持不变。

## 结构

- `control_lab/plants/base.py`：被控对象统一接口（`step`/`predict`/边界）。
- `control_lab/plants/cart.py`：人工小车 `m*x_ddot = u + d`（d 仅真实仿真存在）。
- `control_lab/actuator.py`：执行器模型（固定 FIFO 延迟、动作幅度饱和、
  动作增大/减小不对称变化限制），支持运行期能力切换与检查点恢复。
- `control_lab/capability.py`：执行器能力描述（动作范围、变化限制、延迟、
  单向/双向调节），运行期可降级与恢复。
- `control_lab/controller.py`：延迟感知有限时域 MPC；预测前几拍使用执行器队列
  中尚未生效的旧命令。规划失败、观测缺失和观测恢复首拍输出明确保守动作。
- `control_lab/experiment.py`：滚动时域闭环，分离规划命令、提交命令与设备实际
  施加动作；下一周期的变化限制以真实施加动作为基准。
- `control_lab/logger.py` / `analysis.py`：逐步记录计划与执行并做差异分析。
- `control_lab/replay.py`：内置场景重放清单与重放入口。
- `experiments/run_scenarios.py`：持续扰动、目标突变、规划器故障，以及执行器延迟
  + 不对称变化限制 + 连续丢观测的复合安全场景，以及运行期执行器能力
  下降与恢复场景（`05_capability_degradation_recovery`：动作范围缩小、
  延迟增长、只能单向调节，缺观测与降级同时发生，恢复后平稳回到目标）。
- `tests/`：动力学、约束、保守动作、计划-执行差异等测试。

## 运行

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.run_scenarios
.venv/bin/python -c "from control_lab.replay import replay; replay('04_delay_asymmetric_rate_observation_loss', 'results/replay_04')"
```

场景结果写入 `results/<场景>/`：`steps.csv`（逐步记录）、`plans.json`（每周期完整
预测）、`plan_vs_actual.csv`（预测 vs 真实）、`plot.png`、`replay.json`、
`actuator_checkpoint.json`（执行器检查点）、`results/summary.json`。

`steps.csv` 中 `commanded_u`/`submitted_u`/`applied_u` 分别表示控制器命令、
提交给执行器的命令和设备真正施加的动作；`observation_available=false`、
`reason=observation_missing` 表示缺观测保守刹车，恢复后的第一拍标记为
`observation_recovery_age_<连续缺失拍数>`。汇总中的 `action_bound_violation`、
`rate_up_violation`、`rate_down_violation` 必须为 0。

## 运行期能力变化

- `Actuator.set_capability` 切换能力并返回变更报告：已排队但尚未生效的命令
  不可撤回（`pending_policy=kept_not_withdrawable`），保留在原队列位置、按原
  生效时刻（报告中 `effective_step`）施加，仅在生效时按新能力裁剪；延迟增长
  用队尾保持填充新槽位，延迟缩短不丢弃已排队命令（实际延迟取
  max(新延迟, 队列长度)）。
- 能力变化拍在 `plans.json` 中记录 `capability_change`：变化前/后两份对照计划
  （`plan_before`/`plan_after`）与 `reachability`（当前能力下哪些目标仍可达到；
  保持定点需要零动作在动作范围内）。控制器随即按新能力规划，不再把旧执行
  能力当作当前能力；缺观测与能力变化同时发生时保守动作同样遵守新边界。
- `steps.csv` 的 `cap_u_lo/cap_u_hi/cap_du_up/cap_du_down/cap_delay_steps/`
  `cap_direction` 列记录每拍实际生效能力；汇总按每拍能力边界核对约束。
- `actuator_checkpoint.json` 保存执行器队列、上一施加动作与当前能力；
  `run_experiment(..., actuator_checkpoint=...)` 从检查点重开，队列里尚未
  生效的动作不会被重复施加。

## 替换被控对象

新对象实现 `control_lab.plants.base.Plant`：保持 `n_obs`/`n_act` 与
`observation_bounds`/`action_bounds`/`action_rate_bounds` 语义一致，提供规划用
标称模型 `predict`；控制器与实验框架即可复用（线性 MPC 的 `A/B` 需按新模型
调整或改用通用非线性优化）。
