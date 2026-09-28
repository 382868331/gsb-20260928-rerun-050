# 控制算法研发系统：人工小车滚动时域控制

这个项目先用一个透明的人工点质量小车作为被控对象，搭好模型、控制器、实验运行和记录链路。后续替换被控对象时，只需继续提供相同的观测与动作边界。

## 模型与边界

小车采用离散时间点质量模型：

```text
x[k+1] = x[k] + dt * v[k]
v[k+1] = v[k] + dt * (u[k] + d[k] - b * v[k]) / m
```

- `x`：位置，单位 m
- `v`：速度，单位 m/s
- `u`：加速度动作，单位 m/s²
- `d`：控制器不可测的外部过程扰动
- 默认约束：`-2 <= u <= 2`，相邻采样动作变化量 `|Δu| <= 1.5`

统一边界定义在 `cart_control/interfaces.py`：

- `Observation(time, position, velocity, target_position)`
- `Action(acceleration)`
- `ActionBounds(min_acceleration, max_acceleration, max_slew)`

替换新被控对象时，应让新模型继续输出 `Observation`、接受 `Action`，并复用或显式声明同一组 `ActionBounds`。

## 控制策略

控制器位于 `cart_control/mpc.py`，使用有限时域二次规划：

- 每个采样周期根据当前 `Observation` 重新求解未来 `N=25` 步计划。
- 只执行计划中的当前一步动作，下一次由真实新观测重新规划。
- 优化中同时约束动作幅度与相邻动作变化速度。
- 上一周期预测不会被直接延用；热启动只作为优化器初值。
- 若求解失败、结果非有限值或违反边界，立即采用明确的保守刹车动作：
  `u = clip_slew(-braking_gain * v, previous_action)`，并清空热启动解。

## 实验场景

实验不是单条理想轨迹，包含四个场景：

- `nominal`：固定目标、无扰动。
- `wind_disturbance`：中段持续时变风扰。
- `target_jump`：目标从 `1.5 m` 突变为 `-2.0 m`。
- `disturbance_and_target_jumps`：持续扰动叠加两次目标突变。

每一步记录：

- 执行前真实位置、速度和当前目标。
- 规划动作与实际执行动作。
- 规划是否成功及优化器消息。
- 一步预测位置/速度与真实下一步位置/速度。
- 计划与实际的一步误差。
- 当次完整时域动作、预测位置和预测速度序列。

## 运行

```bash
./.venv/bin/python scripts/run_experiments.py --output-dir outputs
```

输出：

- `outputs/<scenario>.jsonl`：逐步详细记录和完整计划。
- `outputs/summary.csv`：每个场景的约束、误差和终态汇总。

运行测试：

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

当前验证结果：5 个单元测试全部通过；四个实验场景均能在动作幅度和变化速度约束内完成重规划。扰动场景的速度预测误差由未建模扰动产生，目标突变场景会从突变后的新观测重新生成计划。
