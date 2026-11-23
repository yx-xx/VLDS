# VLDS 结构化模型与训练实现说明

论文项目中的[整体算法结构与训练方案](../../VLDS_paper/算法结构与训练方案.md)是方法定义的主文档；本文只记录代码侧的实现边界和当前状态。

## 当前状态

现有 openpi 训练入口面向 `Observation + Actions`，使用动作流匹配损失。当前数据处理已经完成大规模回放场、分段和独立审计，但最终 `vlds.fields.v1` 结构化训练集尚未完成，现有离线回放结果中的 `parameter_supervision` 仍为 false。因此暂不把普通 `pi05_libero` 训练配置改成 VLDS 训练配置。

## 目标代码边界

```text
src/openpi/models/vlds.py
src/openpi/models/vlds_config.py
src/openpi/training/vlds_dataset.py
src/openpi/training/vlds_train.py
src/openpi/policies/vlds_policy.py
```

结构化 batch 至少包含：

```text
observation: images, language, state_history, execution_context
current: goal, command, primitive, hold, field_raw, masks
future: goals, commands, valid_mask
```

训练损失由目标位姿、夹爪/阶段类型、驻留时间、当前场行为、规范化参数和结构约束组成。场行为损失在源轨迹点、局部扰动点和成功外力轨迹状态上比较预测场与参考场的速度向量；参数损失不能单独替代行为损失。

## 实现顺序

1. 完成 `vlds.fields.v1` 的读取器、掩码和归一化统计；
2. 用小型假数据验证结构化 batch、参数解码和分项损失；
3. 实现 H=1 模型，先冻结视觉语言骨干；
4. 完成 H=1 微型过拟合和无扰动闭环；
5. 扩展 H=3 未来目标头和事件驱动执行；
6. 加入外力成功轨迹的增广采样；
7. 最后实现候选评价器作为可选扩展。

不得把动力系统参数拼接到普通动作块中训练，也不得在最终字段和掩码未验收前启动大规模 VLDS 训练。

## 可直接搭建的首版模型（2026-11-23）

已在代码中加入一个可独立运行的结构化 VLDS 头，文件边界如下：

```text
src/openpi/models/vlds_config.py       # H=3、K=4、输入维数和场数值配置
src/openpi/models/vlds.py              # 输入编码、计划头、场头、可微场解码和分项损失
src/openpi/training/vlds_dataset.py    # vlds.fields.v1 batch 名称和合成因果 smoke 数据
src/openpi/training/vlds_train.py      # AdamW 训练、验证、指标和 checkpoint
```

输入张量固定为：`state_history[B,4,12]`、`execution_context[B,12]`、`language[B,32]`。当前实现先把视觉语言编码结果作为 `language` 向量传入，后续接入 `pi05_libero` 时只替换输入编码器。输出固定为 H=3 个有序目标槽位，以及当前槽位的 `field_raw[B,4,19]`、合法性和速度尺度。四分量场的矩阵使用

```text
A_k = skew(omega_k) - B_k B_k^T - 0.01 I
```

解码后天然具有共同吸引点和严格负定对称部；训练损失另外比较采样位置的预测速度与参考速度，避免只回归等价的原始因子。

可运行的最小检查：

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/models/test_vlds.py -q
env -u PYTHONPATH .venv/bin/python -m openpi.training.vlds_train \
  --steps 2000 --output data/experiments/vlds_smoke_20261123
```

该命令只验证结构、掩码、参数解码和小数据过拟合，不把当前尚未验收的生产字段标记为 `training_ready=true`。生产数据接入时保持同一 batch 名称，并将 `field_supervision_mask=false` 的样本从参数和行为监督中屏蔽。

首版计划槽位的 18 维输出按固定顺序切分为：`goal_position[3]`、`goal_rotation_6d[6]`、`gripper_logit[1]`、`primitive_logits[2]`、`hold_log_seconds[1]`、`valid_logit[1]`、`handoff_role_logits[4]`。当前场头接收共享特征和第一个计划槽位，输出 `76 + 1 + 1` 个数，依次为 `field_raw[4,19]`、`field_valid_logit` 和正速度尺度 `field_scale`。分项损失对应目标、姿态、夹爪、MOVE/HOLD、驻留、交接、场行为、场因子、合法性、速度尺度和速度上限约束。

共享特征的具体网络是：状态历史逐帧经过 `12 → 128 → LayerNorm → GELU → 128 → GELU`，取掩码平均和最后一步；上下文使用 `12 → 128 → LayerNorm → GELU`；语言嵌入使用 `32 → 128 → LayerNorm → GELU`；四路拼接后使用 `512 → 256 → LayerNorm → GELU → 128 → GELU`。计划头为 `128 → 128 → GELU → H×18`，场头为 `146 → 128 → GELU → 78`。这些维度已经写入 `VLDSConfig`，因此可以直接按配置复现，不依赖隐含的张量尺寸。
