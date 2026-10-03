# 控制算法研发系统（人工小车 + 有限时域 MPC）

以一维人工小车为首个被控对象，用 Python 搭建模型、控制器与实验记录，
打通有限时域滚动规划；被控对象通过统一接口替换，观测/动作边界契约保持不变。

## 结构

- `control_lab/plants/base.py`：被控对象统一接口（`step`/`predict`/边界）。
- `control_lab/plants/cart.py`：人工小车 `m*x_ddot = u + d`（d 仅真实仿真存在）。
- `control_lab/capability.py`：运行时执行器能力（动作范围、双向变化速率、
  延迟和是否允许撤回）。
- `control_lab/actuator.py`：执行器模型（FIFO 延迟、动作幅度饱和、方向不对称
  变化限制、运行中能力变化和命令检查点）。
- `control_lab/controller.py`：延迟感知有限时域 MPC；预测前几拍使用执行器队列
  中尚未生效的旧命令。能力变化后立即使用新约束和当前可达目标区间，规划失败、
  观测缺失和观测恢复首拍输出明确保守动作。
- `control_lab/experiment.py`：滚动时域闭环，分离规划命令、提交命令与设备实际
  施加动作；下一周期的变化限制以真实施加动作为基准。
- `control_lab/logger.py` / `analysis.py`：逐步记录计划与执行并做差异分析。
- `control_lab/replay.py`：内置场景重放清单与重放入口。
- `experiments/run_scenarios.py`：持续扰动、目标突变、规划器故障，以及执行器延迟
  + 不对称变化限制 + 连续丢观测的复合安全场景，以及运行时能力收缩、延迟增长、
  单向调节、观测缺失同时发生和能力恢复场景。
- `tests/`：动力学、约束、保守动作、计划-执行差异等测试。

## 运行

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.run_scenarios
.venv/bin/python -c "from control_lab.replay import replay; replay('04_delay_asymmetric_rate_observation_loss', 'results/replay_04')"
```

场景结果写入 `results/<场景>/`：`steps.csv`（逐步记录）、`plans.json`（每周期完整
预测）、`plan_vs_actual.csv`（预测 vs 真实）、`plot.png`、`replay.json`、
`results/summary.json`。

`steps.csv` 中 `commanded_u`/`submitted_u`/`applied_u` 分别表示控制器命令、
提交给执行器的命令和设备真正施加的动作；`observation_available=false`、
`reason=observation_missing` 表示缺观测保守刹车，恢复后的第一拍标记为
`observation_recovery_age_<连续缺失拍数>`。汇总中的 `action_bound_violation`、
`rate_up_violation`、`rate_down_violation` 必须为 0。

场景 `05_runtime_capability_change` 额外记录运行时能力：

- `plans.json` 中 `capability_event` 保存变化前后的能力、保留的排队命令 ID、
  是否允许撤回，以及每条排队命令的 `due_step`（实际计划生效时刻）。
- 同一周期的 `plan_before_capability_change` 保存能力变化前旧计划，变化后的
  `planned_actions`/`planned_states` 使用当前能力重新计算，形成前后对照。
- 默认不允许撤回已排队但未生效动作；延迟增长时插入中性动作并顺延旧命令，
  延迟降低或范围恢复不会删除队列，旧命令仍按原 FIFO 顺序到点生效。
- `capability` 中给出 `direction`、`target_attainable`、
  `reachable_target_min/max`；当观测缺失与能力变化同时发生，仍只输出
  `observation_missing` 保守动作，不把旧目标可达性当作当前可达性。
- `actuator_state` 保存最后生效命令 ID 和仅含未生效命令的队列。续跑时恢复该
  状态并提供对应的真实状态；已生效命令不会因保存/重开再次施加。

## 双执行器与不确定回执

两台能力、延迟和允许方向不同的执行器共同作用于同一目标，命令计划、
设备接收与实际作用分离记录：

- \`control_lab/actuator_group.py\`：双执行器组，各自独立的 FIFO 延迟队列
  与命令 ID；一个周期内每条命令只入队一次。
- \`control_lab/dual_controller.py\`：双执行器有限时域 MPC，两台的幅度、
  变化速率、延迟和允许方向分别作为硬约束；输出每台分担比例、相反作用
  的抵消量与目标可达区间。
- \`control_lab/receipts.py\`：回执台账。每条命令按 (执行器, 命令 ID)
  归属，状态分为排队 / 已确认发生 / 明确未发生 / 尚不能确认。目标状态
  已变化但回执未到的命令记为"尚不能确认"：不当作未执行重发，也不把
  另一台的作用算成它的完成。重复回执保留首次归属，较旧回执晚于新命令
  到达仍按命令 ID 归账，部分执行记录已作用量与明确未发生的差额，重连
  后批量补报（resync）逐项对应。人工确认未发生后相应作用才允许重新
  安排；确认发生则避免重复补偿。
- \`control_lab/dual_experiment.py\`：双执行器滚动实验。对尚不能确认的
  命令，计划记录给出当前作用力的可能区间与计划合力的上下界
  （\`uncertainty.planned_total_min/max\`），不把不确定性藏进精确预测值。
  \`strategy="wait"\` 保守等待（输出明确保守刹车动作）；
  \`strategy="compensate"\` 把不确定命令可能的作用从能力中扣除后，只在
  剩余能力范围内继续补偿。能力收缩后重新核对尚未发生的动作
  （\`capability_event.recheck\`），已发生事实不回退。
  \`snapshot_step\` + \`snapshot_path\` 在含不确定回执的中途保存，
  \`resume_state\` 恢复后原命令与核实决定逐项对应，轨迹与连续运行一致。

\`\`\`bash
.venv/bin/python -m experiments.run_dual_scenarios
\`\`\`

场景写入 \`results/dual_*/\`：\`steps.csv\` 中 \`commanded_a/b\`、
\`submitted_a/b\`、\`applied_a/b\` 分列计划、提交与实际作用，
\`cancellation\` 为两台相反作用的实际抵消量，\`residual_error\` 为剩余
误差，\`unmet_reason\` 说明不能继续的原因；\`plans.json\` 中
\`ledger_view\` 给出每台执行器排队 / 已确认发生 / 明确未发生 /
尚不能确认的动作清单。汇总见 \`results/dual_summary.json\`。

## 替换被控对象


新对象实现 `control_lab.plants.base.Plant`：保持 `n_obs`/`n_act` 与
`observation_bounds`/`action_bounds`/`action_rate_bounds` 语义一致，提供规划用
标称模型 `predict`；控制器与实验框架即可复用（线性 MPC 的 `A/B` 需按新模型
调整或改用通用非线性优化）。
