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

## 双执行器与命令回执不确定性

两台能力、延迟、允许方向不同的执行器共同作用于同一目标，命令计划、设备接收
与实际作用三层严格分开：

- `control_lab/command_ledger.py`：每台执行器一本命令台账。命令状态为
  `queued`/`in_flight`/`acked`/`uncertain`/`executed`/`not_applied`，使用者可
  分别查看排队、已确认发生、明确未发生和尚不能确认的动作。回执只按原
  `command_id` 归档：重复回执标记 `duplicate`，较旧命令的迟到回执不会写到新
  命令上，与人工核实冲突的迟到回执标记 `superseded`；部分执行量保留为
  `uncertain` 区间（`confirmed_value` 为已发生部分，`remaining_value` 为待
  核实部分）。回执通过 `measure` 明确计量方式：`incremental` 只携带新增
  执行量（在已确认累计水平上累加一次），`cumulative` 携带截至目前累计
  水平（直接对齐，不重复累计）；内容完全相同的再次投递才标记
  `duplicate`，先增量后累计的合法序列会被分别接受。只携带已执行量而
  没有比例的部分回执记为 `uncertain`，余额保留，绝不推断为全部完成；
  数量与比例矛盾或累计量倒退时，记录 `disputed`/`dispute_detail`/
  `disputed_values` 保留争议双方；尚未确认的余额继续保持 `uncertain`，
  但可能作用区间只能从已确认量延伸到请求量，不能纳入更小的倒退值。命令若已
  进入 `executed`/`not_applied` 终态，普通迟到回执只会标记为 `superseded`
  并追加争议证据，不能撤销已确认事实，也不能让已发生动作重新进入待补偿
  区间。部分确认后（如 2.5 已确认 1.5）再收到设备未接收通知或更小的迟到
  累计量时，已发生事实与待核实余额同样保持不变，矛盾通知只标记
  `superseded` 并记入争议，绝不会把剩余量恢复为全部请求量。独立调查
  结论必须通过显式纠正决定登记，纠正记录同时保存纠正前后
  的状态、已确认依据、可能作用区间和剩余量；普通迟到回执不会自动触发纠正。
  设备回执自带 `base_value`/`clipped_value`，数量与比例在同一基准上核对。
- `control_lab/device_channel.py`：真实设备侧（FIFO 延迟、限幅、断连、部分
  执行）与 `ReceiptBus`（回执延迟、重复、丢失）。断连期间设备收不到命令，
  重连时只补一条 `not_received` 回执，不自动重发。
- `control_lab/dual_controller.py`：区间包络双执行器 MPC。已确认事实作为确定
  量；尚不能确认的动作用“可能作用区间”传播（`possible_min/max`、
  `envelope_states`），不压成精确预测值。每拍同时给出 `wait`（不确定通道冻结
  等待、其余通道在自身能力内继续调节）与 `compensate`（最坏情形鲁棒可行才
  补偿）两套方案，并输出目标是否仍可达、`cancellation_first_step`（两通道相反
  作用的实际抵消量）、剩余误差与不能继续的原因。能力降低后按新约束重新核对
  尚未发生的动作，已发生事实不回退。
- `control_lab/actuator_group.py`：执行器组，串联通道与台账；只有人工/设备
  确认未发生后才允许 `reschedule` 重新安排作用并记录替代命令 ID，确认发生后
  迟到回执不产生重复补偿；`save_state/load_state` 保存不确定中途状态，恢复后
  原命令与核实决定逐项对应。
- `control_lab/dual_experiment.py` / `dual_logger.py`：双执行器闭环与记录，
  `steps.csv` 含每台的计划/提交/实际作用及六种状态计数，`plans.json` 含每台
  通道区间、两套方案、回执流水与全部命令记录。
- `experiments/run_dual_scenarios.py`：场景 `06_dual_uncertain_receipt`（回执
  丢失/重复/迟到/部分执行/断连重连）与 `07_dual_capability_manual`（能力收缩、
  人工核实/显式纠正后重排、中途保存恢复）。保存恢复只反序列化一次台账和
  真实通道事实，已执行命令不会因重启再次施加或计入控制基准两次。

```bash
.venv/bin/python -m experiments.run_dual_scenarios
```

## 运行

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.run_scenarios
.venv/bin/python -c "from control_lab.replay import replay; replay('04_delay_asymmetric_rate_observation_loss', 'results/replay_04')"
```

原有单执行器五个场景与测试保持不变；`tests/test_dual.py` 覆盖双执行器回执
丢失、重复/迟到归属、部分执行区间、等待/补偿、人工核实重排、能力重核、
断连重连与中途保存恢复。

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

## 替换被控对象

新对象实现 `control_lab.plants.base.Plant`：保持 `n_obs`/`n_act` 与
`observation_bounds`/`action_bounds`/`action_rate_bounds` 语义一致，提供规划用
标称模型 `predict`；控制器与实验框架即可复用（线性 MPC 的 `A/B` 需按新模型
调整或改用通用非线性优化）。
