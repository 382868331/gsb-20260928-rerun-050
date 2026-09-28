# 控制算法研发系统（人工小车 + 有限时域 MPC）

以一维人工小车为首个被控对象，用 Python 搭建模型、控制器与实验记录，
打通有限时域滚动规划；被控对象通过统一接口替换，观测/动作边界契约保持不变。

## 结构

- `control_lab/plants/base.py`：被控对象统一接口（`step`/`predict`/边界）。
- `control_lab/plants/cart.py`：人工小车 `m*x_ddot = u + d`（d 仅真实仿真存在）。
- `control_lab/controller.py`：有限时域 MPC（二次目标，动作幅度 + 变化速度
  线性约束，SLSQP + 精确梯度）；规划失败输出明确的保守刹车动作，不延用旧预测。
- `control_lab/experiment.py`：滚动时域闭环：每周期重新规划、只执行第一步、再观测。
- `control_lab/logger.py` / `analysis.py`：逐步记录计划与执行并做差异分析。
- `experiments/run_scenarios.py`：持续扰动、目标突变、规划器故障三个场景。
- `tests/`：动力学、约束、保守动作、计划-执行差异等测试。

## 运行

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.run_scenarios
```

场景结果写入 `results/<场景>/`：`steps.csv`（逐步记录）、`plans.json`（每周期完整
预测）、`plan_vs_actual.csv`（预测 vs 真实）、`plot.png`、`results/summary.json`。

## 替换被控对象

新对象实现 `control_lab.plants.base.Plant`：保持 `n_obs`/`n_act` 与
`observation_bounds`/`action_bounds`/`action_rate_bounds` 语义一致，提供规划用
标称模型 `predict`；控制器与实验框架即可复用（线性 MPC 的 `A/B` 需按新模型
调整或改用通用非线性优化）。
