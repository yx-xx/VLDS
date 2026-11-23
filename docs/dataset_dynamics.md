# 全量数据集动力系统构建与播放

2026-11-22 已完成本机五类完整数据集的离线构建和逐步独立重算。输出是稳定线性场的名义回放模型。LIBERO 另有固定生产限速、角速度和姿态增益的结果：原速版本保留失败，显式放慢时间的版本在全部 6,500 回合通过名义验收。两者都不提供完整非线性 SEDS 参数监督，也不证明物理操作成功。

## 实际覆盖

| 数据集 | 回合 | 保存场 | 已覆盖观测转移 | 单转移场 |
| --- | ---: | ---: | ---: | ---: |
| 处理后 LIBERO v2 | 6,500 | 183,566 | 1,007,618 | 1,523 |
| LeRobot LIBERO | 1,693 | 49,596 | 271,772 | 325 |
| ALOHA transfer cube | 50 | 1,986 | 19,950 | 0 |
| PushT | 206 | 8,420 | 25,444 | 1,405 |
| xArm lift medium | 800 | 19,086 | 19,200 | 18,978 |
| **合计** | **9,249** | **262,654** | **1,343,984** | **22,231** |

所有列出的场均通过本轮回放拟合验收。独立审计逐帧从参数重算连续轨迹，检查边界覆盖、夹爪命令边界、稳定性、保存轨迹一致性与姿态误差，没有在分段处重置状态。LIBERO 回合位置 RMS 的 P50 / P95 / 最大值为 3.286 / 4.345 / 5.559 mm；这是本体名义积分的拟合指标。

完整源 LIBERO 与处理后 LIBERO 是同一批 6,500 条轨迹，未重复计数；下载预览复用完整数据对应回合，读取时校验源文件散列。历史 `libero_v1`、pilot、stride 示例不作为新的独立数据集。本机构建结果位于 `data/processed/dynamics_20261122_final/`，统计见 `collections.json`，各类的独立复核见 `audit.json`。

处理后的 LIBERO 有 T+1 条观测，覆盖全部 T 条动作；其他 LeRobot 文件只有 N 条已观测状态，构建 N−1 条可验证转移，不伪造最后一条动作的后继状态。原始 LIBERO 图像按发布的动作后观测与处理后轨迹对应，场投影使用 +1 帧偏移；校正后的三维场请使用 `libero_v2`。

## 模型、分段和验收

位置或原生关节状态使用固定目标的线性场 `f(x) = -rate * (x - goal)`，学习量为正稳定率及正标量限速。原生单位下有 `d(||x-goal||²)/dt < 0`；限速只缩放速度，不改变方向。它是非线性 SEDS 家族之外的简化模型，不能从四个相同线性分量推断完整高斯混合参数。所有结果保留 `parameter_supervision=false`。

每个夹爪命令区间先尝试完整拟合；优化多个稳定率和限速的离散组合，目标同时包含源起点局部积分和上段传递状态的连续积分误差。不通过再二分，覆盖每个观测转移，失败的单步片段也必须保留为失败。当前构建通过逐步收紧细分余量重试；不以假造位置重置使失败消失。二分是确定性工程方案，不保证场数最小或语义边界最优。

处理后和 LeRobot LIBERO 保留二值夹爪命令边界。ALOHA、PushT、xArm 使用各自原状态轨迹，不将连续关节/夹爪标量伪造为 LIBERO 的二值命令。

| 坐标 | 误差尺度及验收 |
| --- | --- |
| 处理后 LIBERO | 世界系 `gripper0_grip_site`，位置向量范数 10 mm；SO(3) 姿态误差 5° |
| LeRobot LIBERO | 来源记录的末端 xyz，位置 10 mm；来源旋转向量转 SO(3)，姿态 5°；无已核验世界场景映射时显示原坐标投影 |
| PushT | 原图像平面坐标，向量范数 5 pixel |
| ALOHA | 14 维双臂关节/夹爪状态；关节尺度 0.05 rad，归一化夹爪尺度 0.05 |
| xArm | 记录的 xyz 各尺度 10 mm，夹爪原生标量尺度 0.05；混合单位的误差使用尺度归一化范数 |

拟合优先要求整段最大归一化误差不超过 0.9；最终验收不超过 1，姿态不超过 5°，并检查额外 2 s 后的位置收敛。整体原生坐标 RMS 在关节/夹爪混合单位下不等价于米制位置 RMS。采样周期来自数据时间戳，要求均匀。

### 段末减速与过点场

“不重置状态”只保证边界位置连续，不保证边界速度连续。旧线性场和长段高斯场均把当前段末观测设为吸引点，场积分可能在切换前提前减速，然后在下一场中重新加速。这种名义轨迹的减速与 Tk 采样阻塞、慢速三维渲染属于不同问题。三维机器人读取源仿真状态，紫色名义积分由保存的场参数产生；不能由紫色轨迹减速推断源机器人也在等待。

新增可选 `coherent_flow` 配置：在源轨迹边界仍沿原方向运动时，尝试未来 1、2、4 个源观测作为固定吸引点，以整段时间对齐误差和末步速度误差选择。片段的实际结束时刻、源观测数和夹爪命令边界保持原语义，在源边界切换场，不等待抵达前视吸引点。真正的零速度、反向边界和整条轨迹的最后观测使用原终点。候选仍有单场共同二次 Lyapunov 收缩证书、限速及额外 2 s 的目标收敛检查；不提供整个切换系统的全局稳定性或 C1 连续证书。姿态模型本轮继续使用原目标，主要修正位置运动的段末减速。

前视改变传给后续场的状态，因此也要验收整条连续积分。前视方案不通过、原终点方案在同样配置下通过时，回退整条终点方案，并记录 `flow_fallback=true` 和失败尝试的统计；不局部重置、不隐藏失败长段。该回合仍可能保留段末减速，应结合边界诊断报告检查。新配置写入独立目录；未生成或仅完成样例时不会出现为全量可选配置。

前视目标先做快速回归粗筛，闭环梯度优化由外层候选端点的有限预算统一调度，避免每个目标都执行完整优化。对最终选中的合格片段，若末步速度不足示教的 65%，至多补一次前视目标优化，只有误差仍通过且速度拟合改善才采用。普通优化预算通过 `--iterations` 设置，失败边界细搜通过 `--fine-iterations` 单独设置（默认 240）；降低后者可能影响困难片段的拟合通过率，应连同实际配置记录。

```bash
# 先在新目录进行代表回合试验；不能把样例复核当成全量完成
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.coherent data/processed/dynamics_20261122_final/libero \
  --goal-lookahead-steps 4 --episodes 6000 --workers 1 \
  --output data/experiments/passthrough_check/libero

OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.coherent_audit data/experiments/passthrough_check/libero --workers 1

# 单独比较边界速度、原轨迹停留和最大误差，导出 JSON 与静态速度图
OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.continuity \
  data/experiments/passthrough_check/libero/episode_06000.json \
  --baseline data/processed/dynamics_20261122_final/libero/profiles/coherent/episode_06000.json \
  --output data/experiments/passthrough_check/analysis
```

省略 `--episodes` 和 `--output` 会构建全量独立目录 `profiles/coherent_flow`；完整性审计覆盖全量后，观察台会自动发现“长片段/过点场”。边界诊断中的 `slowdown_candidate` 是源轨迹继续运动、而积分末步速度不足源速度一半的提示，不是新的拟合合规证书。

已审计的单回合样例也可显式载入：打开处理后 LIBERO v2，选择全局回合索引 6000，再点击“载入拟合结果…”，选择 `data/experiments/dynamics_passthrough_20261123/v5/libero/episode_06000.json`。当前会话会出现“长片段/过点场”，时间轴、彩色分段、三维场和名义积分同步更新。LeRobot LIBERO、ALOHA、PushT、xArm 同样可选择对应回合后载入各自 `v5/<dataset>/episode_*.json`，查看原生状态场截面及区间播放；失败片段会明确显示验收原因。

手动载入会核对当前数据集、回合身份、源文件及轨迹散列，并从参数逐步重新计算，旧 `audit.json` 本身不足以让被篡改结果通过。样例只加入当前会话，不登记为全量默认配置。已运行的旧观察台需要在方便时重新启动才能使用新增入口；无需中断现有窗口。

代表回合的通过/失败、整回合回退和速度对照见[分段末端收敛与过点目标复核](validation/dataset_dynamics_passthrough_2026-11-23.md)。该记录是样例验证，不能替代前述全量构建结果。

**回放控制配置与原控制配置不同。** 回放姿态增益使用 `1/dt`，角速度上限根据源姿态变化标定。处理后 LIBERO 的位置总上限为 `max(0.3 m/s, 1.05 × 最大源差分速度)`，各场在此上限内进一步拟合；其他原状态空间同样保存标定上限。每个场的 `rate_per_second`、`maximum_speed_scaled_per_second`、目标、限速和误差都保存在 JSON 中。处理后 LIBERO 的矩阵范数仍不超过 20/s；50 Hz 的 ALOHA 原关节场允许到 50/s，并不属于原 LIBERO 参数接口。

默认回放配置的 6,500 个 LIBERO 回合中有 1,029 个的保存场限速高于 0.3 m/s，最大 1.650 m/s。这个统计是保存上限，不能单独推出在 0.3 m/s 下必然拟合失败。后续原速/放慢时间的生产配置复验见下节；接触操作仍未验收，保持 `training_ready=false`、`physical_execution_validated=false`。

## 原生产控制配置复验

为区分“可稳定回放”和“能在原控制器约束下拟合”，新增独立的 `production_libero` 配置。它固定使用位置限速 `0.3 m/s`、角速度上限 `0.5 rad/s` 和姿态反馈增益 `2/s`，仍使用同一位置/姿态误差、额外收敛时间和连续状态传递规则。生产配置在 `profiles/production_libero/` 下保存自己的逐回合场和名义轨迹，不覆盖默认回放结果；无法通过的叶子段仍保存，报告会记录 `position_maximum`、`rotation_maximum` 或 `nominal_arrival_timeout`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.profiles \
  data/processed/dynamics_20261122_final/libero --profile production_libero --workers 10 \
  --output data/experiments/production_libero_recheck

OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.audit \
  data/processed/dynamics_20261122_final/libero/profiles/production_libero
```

浏览器的动力系统面板在该目录存在时提供“回放配置 / 生产控制配置”切换；切换会同步时间轴、三维场边界和原生投影。生产配置的结果用于定位约束瓶颈，仍不等价于实际机器人执行合规。

本轮全量生产复验覆盖 6,500 个回合、676,614 个场，其中 425,084 个场通过，2,787 个回合的全部场通过。未通过叶子共 251,530 个，主要是 `rotation_maximum`（251,226 个），位置误差超限 337 个；没有到达时间超限。独立审计的保存轨迹与逐帧重算最大差异为 2.22e-15，审计 `passed=true` 表示产物完整且可重算，不把 3,713 个含失败叶子的回合算作生产合规回合。

这些失败是当前固定终点目标、增益和原速时序组合下的结果，不能仅凭 `rotation_maximum` 证明其他分段/目标或模型都无解。低增益向分段末姿态收敛会产生滞后。

### 固定生产约束下的时间缩放修复

新增 `production_retimed`，在保持 `0.3 m/s`、`0.5 rad/s`、姿态增益 `2/s` 的前提下，依次尝试整回合时间倍率 1、2、4、8、16，选取首次通过者。仍在原控制周期 0.05 s 内积分，每两个原观测之间执行相应数量的控制步；不能用放大 Euler 积分步长代替真实放慢。验收同时检查原观测与中间数值插值（位置线性插值、姿态 SO(3) 插值），中间点不计作新增测量观测。

全部 6,500 回合、137,599 个场通过名义验收，单转移场 387 个。2,786 回合保持原速，3,238 回合放慢 2 倍，475 回合放慢 4 倍，1 回合放慢 8 倍。独立审计重算 1,808,414 个控制步，最大位置误差 9.727 mm、最大姿态误差约 4.988°，保存/重算位置差异最大 1.38e-14。除拟合误差外，新配置检查额外 2 s 内达到位置 5 mm、姿态 5°、速度 0.02 m/s、角速度 0.1 rad/s 并保持 0.1 s，局部起点和连续传递起点均须通过。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.profiles \
  data/processed/dynamics_20261122_final/libero --profile production_retimed --workers 10

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.audit \
  data/processed/dynamics_20261122_final/libero/profiles/production_retimed --workers 10
```

面板新增“生产约束/放慢时间”，显示原时间与放慢后的时间，按保存时长播放。误差曲线、紫色轨迹和验收状态都使用连续传递轨迹，避免用每段重起点的局部报告替代连续结果。原问题回合 6000 在该配置下为 18 场、时长 9.8 s，位置 RMS 3.933 mm、姿态最大误差 2.750°；原速生产配置 98 场的姿态最大误差为 8.304°。详见[放慢时间的生产配置复核](validation/dataset_dynamics_production_retimed_2026-11-23.md)。

xArm 的 18,978 / 19,086 个场只有一个观测转移，约 99.43%。它们可以显示稳定场和重现相邻状态，但缺乏多点参数辨识信息；本轮并未解决该数据的高质量长片段参数监督问题。其他数据也保留单步场统计供筛选，而不通过删除片段制造全覆盖结果。

## 命令

从代码仓库根目录执行。保持代码、配置和源版本不变时支持断点恢复；版本散列变化必须使用新的输出目录。

```bash
# 全部已安装的五类完整数据集；自动登记播放指针并复用逐回合有效结果
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics --data data --output data/processed/dynamics_new --workers 10

# 只构建指定数据集；不显式指定 episodes 时处理所有回合和划分
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.build data/processed/libero_v2 \
  --output data/processed/libero_dynamics_new --workers 10

# 独立逐步重算某个已完成集合
OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.audit data/processed/dynamics_20261122_final/libero

# 打开已处理 LIBERO，时间轴自动显示保存的动力系统分段和场
DISPLAY=:1 env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py \
  --dataset data/processed/libero_v2
```

每个集合保存 `run.json / index.json / summary.json / audit.json`、逐回合 `episode_XXXXX.json/.npz` 和构建时源码快照。数据集自身只增加忽略目录中的 `.vlds/dynamics.json` 相对路径指针；校验指纹、源清单/文件散列、回合身份、帧数和缓存 SHA 后才播放。构建结果和原始数据不进入 Git。

## 观看

打开“数据集浏览”，已登记的回合直接载入保存的分段；切换段只从参数采样场，不调用优化器。时间轴支持前后段、单段播放、拖动与逐帧。处理后 LIBERO 将真实三维场箭头、橙色示教、紫色**连续**名义轨迹和绿色目标叠加在录制状态回放中；投影图可切换原生坐标轴。ALOHA、PushT、xArm 和 LeRobot LIBERO 在录制画面下方显示对应状态空间投影。缺少完整场景状态的数据继续显示录制画面，不从关节或末端状态猜测整个三维场景。

未登记离线结果的处理后 LIBERO 仍保留原按需拟合诊断入口；“诊断整命令段”的 SEDS/线性选项和限速仅作用于手动诊断，不改变已保存模型的配置。全量输出与原 `fit-fields` 候选池是不同 schema，不能直接输入原 `execute-fields` 执行器。

原问题回合 6000 的 `[36,85)` 命令区间现分为 12 场，整回合 20 场；按原比较的梯形时间权重，连续位置 RMS 从 89.18 mm 降至 3.15 mm，整回合均匀样本 RMS 为 3.81 mm。此对照同时改变了分段、拟合目标、模型和姿态回放配置，不是单独证明某一因素有效的控制消融。正式证据见[全量构建复核](validation/dataset_dynamics_2026-11-22.md)。

## 长片段分段与高斯稳定场（2026-11-23）

此前“全部场拟合通过”没有验收原始片段长度。旧回放配置中，LIBERO 中位只有 5 个观测转移、约 0.25 s，167,323 / 183,566 个场不到 0.5 s；ALOHA 中位约 0.14 s，xArm 几乎全部是单转移场。`production_retimed` 的显示时间虽然延长，中位仍只有 7 个原始转移。原始观测数量不随时间放慢或插值增加，旧结果不能作为长片段参数辨识已经完成的证据。

新增实验配置 `coherent`：运动段至少包含 **10 个原始转移，且源时长至少 0.5 s**。完整的短夹爪命令事件仍保存，标为“短命令事件，仅诊断”；算法不会制造新的短运动段。先尝试长段，失败时在粗候选附近逐帧细搜边界，再做闭环误差优化；仍未通过的长段保留为失败。源观测、命令边界和连续传递状态均须完整保留。

位置场采用高斯权重混合的离散 LPV 收缩映射，保存归一化尺度、共同度量变换、门控中心/方差和局部映射。每个局部映射在共同度量下的最大奇异值严格小于 1，非负归一化门控和正标量限速保持该度量下降。拟合同时检查原起点和上段传入起点的完整闭环轨迹。它可以表达弯曲运动，是新的实验估计器，**不是原 SEDS 优化器的复现，也不兼容原 SEDS 参数监督接口**。姿态在目标相对 SO(3) 主对数图中另拟合同类映射，复核访问轨迹没有越过 π 边界及实际角速度；不声称整个 SO(3) 的全局稳定保证。

该配置继承基础回放配置的速度上限，使用原始时间、单步积分及 2 s 额外位置到达检查；不等同于 `production_libero` 或 `production_retimed` 的固定生产约束和完整到达条件。片段长度合格、稳定证书成立、轨迹拟合通过是分别检查的条件，不能互相替代，也不保证参数可唯一辨识。`parameter_supervision=false`、`training_ready=false`、`physical_execution_validated=false` 继续保留。

```bash
# 对一种完整数据集重新拟合；已有结果会按代码、配置及源数据身份检查后续跑
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.coherent \
  data/processed/dynamics_20261122_final/libero --workers 8

# 独立逐步重算长片段、场、连续轨迹、原始观测长度及失败原因
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.coherent_audit \
  data/processed/dynamics_20261122_final/libero/profiles/coherent --workers 8

# 汇总五类数据的新旧片段长度、通过覆盖和实际完成情况
OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.coherent_report data/processed/dynamics_20261122_final \
  --output data/experiments/dynamics_long_segments_20261123/analysis
```

其他集合对应目录名为 `lerobot_libero`、`aloha`、`pusht`、`xarm`。代码或拟合配置改变后须用新目录；不要在运行期间修改拟合与审计实现。本轮夜间进程同时最多使用 10 个拟合 worker，完成每类后自动独立审计，全部完成后生成 `analysis/comparison.md/.json`。运行进度位于 `data/experiments/dynamics_long_segments_20261123/full_status.json`、`full.log` 和各配置的 `progress.json`；这些是实时产物，完成数及失败数以实际文件为准。全量完成前的代表回合试验不能代替全量结论，见[本轮复核记录](validation/dataset_dynamics_long_segments_2026-11-23.md)。

某类 `coherent` 全量生成且独立完整性复核通过后，观察台默认选择“长片段/非线性场”；生成中的小样本配置不会列入可选项。仍可切回旧回放及可用的生产配置作对照。三维箭头直接采样保存的新场；其他数据在原状态空间选择两个轴，真实采样该截面的场，其余维度固定当前源状态。面板显示原始转移数、源时长、长度质量和名义误差，时间轴显示共同度量收缩界；拟合失败会如实显示。浏览与切换场不会调用优化器。
