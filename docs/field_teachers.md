# 参考动力系统拟合与完整环境诊断

**2026-11-22 全量回放补充：** 五类本地完整数据集已生成 9,249 个回合、262,654 个线性稳定场，并独立重算连续积分。浏览器优先读取登记结果；原 `fit-fields` 仍用于四分量候选池研究。回放场使用独立 schema 与标定控制配置，不能直接当作原控制接口训练标签或执行器输入。参数、命令、覆盖和未解决的控制合规问题见[全量构建与播放](dataset_dynamics.md)。


更新：2026-11-22。位置场 F、姿态/HOLD、有限候选选段、完整 LIBERO 连续执行及诊断绘图已有参考实现。输出是 `vlds.libero.offline_fields.v1` 离线池；完整模仿主集 `vlds.fields.v1`、在线事实观察器、预测场模型和外力评测尚未完成。实际结果见[本地核验汇总](validation/field_teacher_2026-11-22.json)与[研究分析](../../VLDS_paper/实验记录.md)。

本文将离线拟合、用于监督或执行对照的模型称为“参考动力系统”；从成功源回放中提取、用于重新拟合的末端位姿及导数称为“回放参考轨迹”。具体位置拟合方法采用 SEDS 约束参数化，术语定义见[技术路线](../../VLDS_paper/技术路线.md#2-运动表示与-f-的接入)。现有文件名、数据字段和归档标识中的 `teacher` 保留为兼容名称。

## 命令与产物

以下命令均从本代码仓库根目录运行。拟合使用主环境；`execute-fields` 自动调用独立 LIBERO 环境。无须启动模型服务。实例索引对应已固定版本的 6,500 条示教清单，6000 是空间取放任务的一条训练示教；更换来源后先核对 `manifest.json`，不能沿用索引猜测任务。

```bash
# 默认 design-v1：0.1 m/s、0.5 s 候选网格、0.01 /s 稳定裕量。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.libero fit-fields data/processed/libero_v2 \
  --episodes 6000 --split train --output data/experiments/field_default

# 训练侧探索中实际运行的细网格配置；仍保持 1 cm / 5° 时间对齐 RMS 验收。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.libero fit-fields data/processed/libero_v2 \
  --episodes 6000 --split train --max-speed 0.3 --grid-seconds 0.25 \
  --output data/experiments/field_fine

# 固定原有场参数，只检查三档速度，不重新拟合；仅允许训练/验证侧校准。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.libero calibrate-fields data/processed/libero_v2 \
  --fields-dir data/experiments/field_default --output data/experiments/field_speed_check

# 同一候选池的简单吸引子对照，以及基于当前 kp=150、阻尼比=1 的偏移补偿。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.libero rescore-fields data/processed/libero_v2 \
  --fields-dir data/experiments/field_fine --representation linear \
  --reference-mode damping-compensated --output data/experiments/field_linear

# 先检查源动作能否在恢复的完整场景中完成任务。
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero execute-fields \
  data/processed/libero_v2 --episode 6000 --source-actions \
  --output data/experiments/source_actions_6000

# 连续执行最多三条备选序列；场之间不重置状态。
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero execute-fields \
  data/processed/libero_v2 --episode 6000 \
  --fields data/experiments/field_linear/episode_06000.json \
  --output data/experiments/field_execution_6000

# 1.5 倍预算敏感性必须对相应比较方法统一使用。
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero execute-fields \
  data/processed/libero_v2 --episode 6000 \
  --fields data/experiments/field_linear/episode_06000.json --budget-multiplier 1.5 \
  --gripper-feedback --output data/experiments/field_execution_6000_budget150

# 实际轨迹、活动目标误差、速度、命令与激活前两路图像；生成 PNG、PDF、来源 JSON。
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero inspect-fields \
  data/processed/libero_v2 --fields data/experiments/field_linear/episode_06000.json \
  --execution data/experiments/field_execution_6000 --trial 0 \
  --output data/experiments/field_execution_6000/diagnostic.png
```

`fit-fields` 未显式给出 `--episodes` 时，按 `--split train`、可选 `--suite` 选取前三条；显式索引仍须匹配所声明划分。`--stability-margin {0.01,0.1,0.5}` 是记录版本的参考动力系统探索参数，0.5 已在一条训练示教上实际拟合，不能写成最优配置。所有输出单独存放，不改写原始数据或 v2 候选掩码。

拟合目录包含 `run.json`、逐回合结果、逐候选缓存和源码快照。指纹包括来源清单、实现、配置及依赖；中断后同指纹可续跑，改变指纹须使用新目录。每回合先尝试 128 个候选，无完整路径才扩至最多 256 个；必须保留的边超额时另记超额。执行目录含请求、工作进程状态/日志、结果、逐次连续轨迹 HDF5 及执行源码快照。数值拒绝、超时、任务失败和完整几何序列完成分别记录；进程成功退出不代表任务成功。

## 场可视化与逐段拟合诊断

观察台的数据集浏览页在命令分段下方提供“动力系统场”控件，默认自动准备与切换场：先核验并复用兼容的原始示教完整方案，否则对当前命令段后台拟合，未通过则有界细分。最近 32 个命令段在会话内缓存。点击“诊断整命令段”仍可用 SEDS 或线性吸引子检查整段；限速可选 0.1/0.2/0.3 m/s，默认探索值为 0.3 m/s，不修改已有 design-v1 配置。也可点击“载入拟合结果…”打开当前回合的 `episode_XXXXX.json`；列表包含已有参数的全部候选，标出所属备选方案及原接受状态，未通过名义验收的候选仍可诊断。没有有效数值参数的候选不能显示。

三维场景显示世界坐标下的位置速度箭头、橙色参考轨迹、紫色名义积分轨迹和绿色目标。紫色游标随源时间移动，白色连线表示当前参考位置与名义位置的偏差；名义轨迹包含约定的额外收敛时间。当前片段内原来的命令配色连线让位于橙色参考线，其他片段仍提供轨迹上下文。自动模式随当前片段切换固定场；关闭自动模式时，离开手动场的来源区间会隐藏，回到该区间恢复。“播放该场区间”只播放该场对应的示教片段；“聚焦当前场”调整视角；“显示场 / 速度箭头 / 名义轨迹”分别控制显示。叠加只改变显示，不向机械臂发送控制动作。

箭头使用与 `primitive.feedback` 相同的限速和离散下降约束；箭头长度按本图统一时间倍率缩放，不能当成一个真实控制周期的位移。它们只表示三维位置场，姿态另用名义姿态 RMS 报告。积分复用 20 Hz 名义验收函数，报告物理时间对齐的位置/姿态 RMS、到达超时及原验收原因。HOLD 的保持场与持续时间沿用现有实现。名义验收通过不代表接触、抓取或完整物理执行成功。

载入保存场时核对源示教签名、分片、组、划分、清单哈希和运行指纹；成功回放派生的参考轨迹按归档哈希加载，橙线使用实际拟合的参考轨迹，界面注明来源。“导出诊断图…”同时保存 PNG/PDF、同名来源/参数 JSON 和不含 pickle 的数值 NPZ。临时场预览不会写入训练标签或修改原分段。

自动预览分别报告局部验收和“按源时刻连续切换、不重置状态”的命令段 RMS。后一项不检查到达切换或物理接触；即使局部验收全部通过，连续切换仍可能未通过。新细分只是浏览诊断，不替代离线候选 DP，也不生成已验收训练标签。原始 HDF5、任意 LeRobot/ALOHA 状态尚不具备此自动三维位置场入口，须先提供匹配的世界坐标控制点轨迹。

2026-11-22 对 episode 6000 的 CLOSE 区间 `[36,85]` 做了专门复核：单目标欧氏下降场的时间对齐位置 RMS 理论下界约 10.71 mm，超过当前 10 mm 门槛，整段实际为 89.18 mm。既有四段方案局部约 4.99 mm，按源时刻连续切换约 8.52 mm；自动二分方案局部约 5.68 mm，连续约 21.94 mm。带作者署名的 SEDS 1.95 源码镜像已在 Octave 中实际试跑，兼容求解器与约束复核详见[诊断记录](validation/field_visualization_2026-11-22.md)。不能把局部拟合通过或原版求解器输出直接当成完整任务成功。

同一功能也可从命令行导出，无须启动图形界面，所有索引从 0 开始：

```bash
# 已有方案 0 的第 1 个场；继承归档的控制配置及参考轨迹。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.libero inspect-vector-field data/processed/libero_v2 \
  --episode 6000 --fields data/experiments/fields_20261122/pilot_c/episode_06000.json \
  --sequence 0 --field 1 --output data/experiments/field_visualization/saved_field.png

# 直接检查完整命令区间 1 是否适合一个固定场；不做动态规划细分。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.libero inspect-vector-field data/processed/libero_v2 \
  --episode 6000 --interval 1 --max-speed 0.3 \
  --output data/experiments/field_visualization/command_span.png
```

`--representation linear` 可查看简单吸引子对照；`--density 3..11` 调整每轴采样数。保存场模式继承其原控制配置，不用命令行新拟合选项重写参数。默认诊断图含三维箭头/轨迹、位置误差时间曲线、示教速度和沿示教位置求得的场速度。

## 表示和接受条件

实现入口是 [`openpi.fields`](../src/openpi/fields/)、[离线构建](../src/openpi/datasets/libero/fields.py)、[校准/对照](../src/openpi/datasets/libero/calibration.py)和[执行进程](../src/openpi/simulation/workers/fields.py)。研究定义继续以[实施规格](../../VLDS_paper/设计细节/实现规格.md)为准。

- 世界坐标、米/秒、控制点 `gripper0_grip_site`；手部四元数先乘来源保存的固定旋转，得到控制点姿态。
- 四分量、每分量 19 个因子；目标单独保存。协方差使用正定因子，线性矩阵使用反对称项减正定项；门控采用 Cholesky 解方程及 log-sum-exp。归档中的物理参数须与因子重解码一致。
- 三次初始化，每次最多 300 次 L-BFGS-B；有限搜索范围约束矩阵因子，部署前仍检查协方差条件数、矩阵范数和稳定裕量。优化器收敛状态单列；预算内产生的可行解仍须通过同一数值轨迹验收，不把优化器状态作为性能结论。
- 少于八个位置样本用受约束的标量收缩率，并屏蔽分量参数监督；普通片段的同类吸引子另作基线。HOLD 按整段位置/姿态范围识别，使用解析保持场和真实等待时长，不能称作 SEDS 拟合成功。
- 数值回放以 20 Hz、物理时间对齐，位置 RMS ≤1 cm、姿态 RMS ≤5°，额外收敛最多 2 s；保存速度残差、径向冲突和限幅。只有通过这些检查的边进入 DAG，代价按源命令时长归一化，加每场 0.1。
- 完整命令边及相邻强制确认边完整保留；可选短尾段也保留为相邻边，避免网格端点留下无法覆盖的最后几个动作。确认边界的 API 要求核验与可观察证据；本轮没有自动生成这些语义确认边。
- `sequences` 是该有限图上最多三条最小代价路径，不代表物理执行最优。最终训练集还需可靠操作交接、在线因果记忆和导出验收。

## 实际控制接口中发现的差异

恢复源状态后，执行器先强制刷新控制器缓存再重置参考；读取实际控制点前调用 `sim.forward()`。每步显式设置位置及姿态参考，并检查实际 `goal_pos / goal_ori`。这是必要集成修正：单独把增量换算为动作，不能排除恢复后的旧控制器状态或零旋转保留旧目标。

默认映射仍是 `reference_seconds=0.05`，即位移/旋转偏移等于场速度乘一个控制周期。已安装 OSC 的阻尼项以零期望速度计算，近似式为 `kp*位置偏移 - kd*实测速度`，故该映射的实际运动速度不等于名义场积分速度。诊断模式 `damping-compensated` 采用统一偏移时间 `kd/kp = 2/sqrt(150) ≈0.1633 s`；这是当前固定控制器下的探索性补偿，不是已验证的机器人速度控制律。名义积分仍按 0.05 s；正标量限幅同时约束参考偏移的名义径向下降。更改 kp/阻尼后不能沿用此数值。

`--gripper-feedback` 在命令切换后要求实测手指间距速度 ≤0.01 m/s 持续 0.15 s，期间当前场继续执行。它只确认执行器停稳，不能确认持物或放置。真实 `held/placed` 观察器尚缺失，查询中的记忆保持缺失状态。

HDF5 的 `privileged_audit` 单独保存仿真物体位姿及双侧指垫接触，结果保存逐项最终任务谓词；这些量只供离线分析，没有参与动作或切换。双侧接触不直接标为 `held`，还需脱离支撑、相对位姿稳定和视觉可判定性等条件。

## 操作证据与源回放对照

`execute-fields --audit-stride 5` 每五个控制步在 `audit_observations/` 另存两路图像，记录在动作后，明确不属于策略输入。`privileged_audit` v2 还逐步保存独立任务谓词；原有无谓词轨迹的归档仍可读取。激活前查询继续独立保存在 `queries/`，不能用动作后的审计图像替换它们。

```bash
# 对同一源初态补采离线审计；使用新目录保留旧结果。
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero execute-fields \
  data/processed/libero_v2 --episode 200 --source-actions --audit-stride 5 \
  --output data/experiments/source_200_audit

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero execute-fields \
  data/processed/libero_v2 --episode 200 \
  --fields data/experiments/fields_20261122/two_linear_compensated/episode_00200.json \
  --max-sequences 1 --gripper-feedback --audit-stride 5 \
  --output data/experiments/field_200_audit

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero audit-fields \
  data/experiments/field_200_audit --compare-source data/experiments/source_200_audit \
  --output data/experiments/operation_evidence_200.json \
  --figure data/experiments/operation_evidence_200.png
```

[`operation_audit.py`](../src/openpi/datasets/libero/operation_audit.py) 检查来源签名、清单、种子及初始位置一致性；候选由过去的证据窗口计算，图像只引用不晚于候选的帧。默认探索判据为：双侧接触且闭爪，抬升至少 2 cm，相对位置/姿态变化不超过 1 cm/10°，持续至少 0.15 s；放置候选要求独立 `in/on` 谓词、开爪、无双侧接触且物体速度不超过 0.02 m/s 持续 0.15 s。物体高度基准是首个动作后样本，并非精确初始物体位姿。放置事实持续成立时，不因其他物体的开闭爪重复产生事件。

输出是 `vlds.libero.operation_evidence.v1`，所有候选仍为 `verified=false / observable=false`，不传入已确认边界 API。它们不是已验证 held/placed 标签，视觉判定和观察器标定仍须完成。2026-11-22 的 200 对照中，源回放产生两个物体各一条抬升接触及放置候选，实际场执行只产生汤罐的这两条候选，奶酪盒未入篮；来源及逐步度量见[研究记录](../../VLDS_paper/实验记录.md)。图表呈现第一条固定路径，不把备选路径作为独立重复。

## 由成功源回放构建派生参考轨迹

源动作回放的任务成功不保证其末端轨迹与旧版引擎记录完全重合。200 的本轮成功回放相对 v2 记录位置 RMS 为 4.35 mm、最大为 6.03 mm；第二次闭爪附近的 y 差约 5.5 mm。可用成功完整源回放建立单独的派生参考位姿轨迹：

```bash
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero fit-fields \
  data/processed/libero_v2 --episodes 200 --split train --max-speed 0.3 --grid-seconds 0.25 \
  --representation linear \
  --source-replay data/experiments/fields_20261122/source_replay_200_audit_v2 \
  --output data/experiments/replay_linear_200

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero rescore-fields \
  data/processed/libero_v2 --fields-dir data/experiments/replay_linear_200 \
  --representation linear --reference-mode damping-compensated \
  --output data/experiments/replay_linear_compensated_200
```

`--representation linear` 直接拟合标量收缩率，无须先运行非线性优化；`fitted` 仍为默认。`--source-replay` 限单条训练/验证回合，拒绝失败、截断、重置、动作/时间/坐标/来源不符的回放。[派生适配器](../src/openpi/datasets/libero/replay_teacher.py) 保存便携的 `teacher_trajectory.npz`，只含末端位置、手部四元数及重算导数；初始位姿来自已核对的来源 FK，后续为实际回放。原数据、动作、划分和掩码保留；物体真值及任务谓词不进入这个 NPZ。`teacher_source` 保存来源与派生文件哈希，重评分、校准及绘图沿用同一派生文件。执行请求也记录该来源。

`rescore-fields --arrival-position-tolerance 0.002` 可另存同池的 2 mm 到达门槛复查；默认沿用父池门槛。本轮 200 中，基于回放参考轨迹拟合的动力系统在 5 mm 下三条备选均完成 39 场却漏抓奶酪盒，2 mm 下三条在 520 步只完成 8/8/27 场。该参数是探索性敏感性检查，未升级 design-v1 主配置。

这是明确记名的回放参考轨迹探索，候选边界、路径和线性速率可能随实际轨迹一起变化；不能把结果单独归因于一个 5.5 mm 偏移，更不能归因于某一引擎版本。语义事件和可训练主集仍须独立验收。

## 已验证的范围与后续依赖

2026-11-22 的五条训练示教均找到完整离线路径，五条源动作在完整环境中均通过任务判据。默认增量映射的拟合场备选均未在原任务预算内完成任务。简单吸引子对照已有成功任务和一条完整连续序列实例；也出现几何序列全部完成而任务失败的双物体实例。因此，离线拟合、几何接力、操作成功和可训练主集须分别验收。

单物体实例在 1.5 倍预算下三条简单吸引子备选均完成全部 17 个场且任务判据为真；这三条路径共享同一源初态，不能作为三个独立任务或训练种子。预测场、恢复、受扰成功率和 H=3 相对 H=1 的收益均未测量。后续首先补源操作事实/确认边界及观察器，再检查参考动力系统在真实入口和接触下的可执行性，之后才导出主训练集。

可复核运行归档：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m pytest tests/fields tests/datasets tests/simulation -q \
  --junitxml=data/experiments/fields_20261122/tests.xml

env -u PYTHONPATH .venv/bin/python scripts/check_field_teacher.py \
  data/experiments/fields_20261122 --test-report data/experiments/fields_20261122/tests.xml \
  --output docs/validation/field_teacher_2026-11-22.json
```

汇总检查只支持其列出的归档一致性、覆盖、划分和终态检查；不会将主集标为可训练。后续运行使用新的目录和日期，保留本轮探索来源。
