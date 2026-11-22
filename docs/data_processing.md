# LIBERO 示教数据处理

本流程将已校验的原始 LIBERO 发布数据转换为时序对齐、单位明确的完整示教，按夹爪命令区间准备轨迹拟合方法 F 的输入。源码集中在 `src/openpi/datasets/libero/`，统一通过 `python -m openpi.datasets.libero` 运行。数据产物在 `data/processed/` 中维护，原始 HDF5 只读。

当前版本为 `vlds.libero.trajectories.v2`，默认输出到 `data/processed/libero_v2/`。它完成示教预处理和 F 候选准备；2026-11-22 已新增独立的参考动力系统拟合与连续执行诊断层，见[运行说明](field_teachers.md)；它不改写 v2 的来源语义。单场恢复、在线事实观察器和完整主集导出仍待完成。`fitted_field_dataset_ready=false`，不能将候选窗口直接用于场生成器的监督训练。

## 统一运行入口

以下命令从代码仓库根目录运行，使用现有主环境，无需模型或 GPU 训练：

```bash
# 小规模验证；临时产物不混入正式数据目录

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero process \
  --suites libero_spatial --limit-tasks 1 --workers 1 \
  --output /tmp/vlds_libero_pilot

# 全部五个套件；指纹一致的已完成任务分片直接复用

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero process \
  --workers 4 --output data/processed/libero_v2

# 独立检查全部回合，包括候选动作覆盖和两路图像的首个可用帧、终帧映射

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero audit \
  data/processed/libero_v2

# 示教图像、本体轨迹、夹爪事件；全量质量统计与动作覆盖图

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero inspect \
  data/processed/libero_v2 --episode 0

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero inspect \
  data/processed/libero_v2 --summary
```

`process --suites` 选择套件，`--tasks` 接受任务文件名去掉 `.hdf5` 或 `_demo.hdf5` 的部分；`--seed` 固定回合划分，`--max-candidates` 限制每回合候选数。各子命令的 `--help` 给出全部参数。未经识别的来源不会自动套用已知时序；核对为动作后观测的其他来源可显式指定 `--observation-timing post_action`。

输出目录锁保护并发转换，每个分片先写 `.hdf5.tmp`，关闭后原子发布。续跑指纹包含核心实现、配置、源文件大小和修改时间、声明内容哈希、来源与依赖版本。任务选择固定，不同选择或不同 schema 应使用不同输出目录。源码快照、实际命令和运行状态随结果保存。

文件错误及回合拒绝写入清单，命令返回非零。`processing_complete` 表示所选文件均已处理，`all_selected_sources_represented` 进一步要求没有拒绝回合；二者均不表示场通过物理执行。检查失败不会删除来源或补造标签。


## 在仿真中观看分段

使用同一个观察台直接打开已处理数据：

```bash
env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py \
  --dataset data/processed/libero_v2 --replay
```

默认暂停，并自动加载所选回合的三维回放；切换任务或回合时自动更新。观察台只使用一个窗口：MuJoCo 三维主视图默认占画面区约 80% 宽度，“演示回合”选择器位于三维视图上方的通栏，右侧辅助栏显示录制相机和回放选项；彩色命令分段条与播放时间轴固定在底部，本体数值按需展开；三维视图恢复该观测对应的原始完整仿真状态。可拖动两个画面之间的分隔条调整宽度，在三维视图内左键旋转、右键平移、滚轮缩放，点击“复位视角”恢复初始方向。MuJoCo 场景内直接叠加世界坐标下的末端轨迹，颜色与分段条一致；当前段加粗，白点标出当前观测，端点标记显示段号，当前位置标注当前段号和 OPEN/CLOSE。点击彩色区间或选择下拉项可跳到段首，“上一段 / 下一段”切换区间，“播放此段”从段首播放到段尾观测后暂停；普通“播放”用于整回合。区间描述给出动作 `[start,end)`、共享观测端点和候选数。

分段条下方的“三维轨迹”控制显示开关，“只看当前段”隐藏其他区间，“分段端点”控制边界球和段号。开关即时生效，切换或重新载入回合时保持设置。叠加只改变显示，不改变机器人状态、碰撞或动力学。常规回合逐点连线；极长回合超出 MuJoCo 显示容量时只对连线抽样，保留各段端点，数据集不变。

这是命令区间可视化与示教状态回放，不是 F 拟合场执行，也没有将重叠候选当作最终场序列。初始录制图像缺失时显示缺失提示，三维状态仍可打开。终端观测的真实录制图像和本体可查看，但来源没有完整仿真状态：到达该观测时清空三维视图并显示缺失提示，避免保留上一帧冒充终态。开启自动加载时，跳回此前有完整状态的观测会自动恢复三维回放。

导出与界面使用相同配色的三维末端轨迹、夹爪命令、反馈及分段时间线：

```bash
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero inspect \
  data/processed/libero_v2 --episode 0 --segments
```

输出 `segmentation.png`；`--output` 可指定其他路径。任意回合都可以使用这一入口，无需重新处理数据或复制图像。

## 时序、坐标与缺失值

原始快照为 `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`，本地 LIBERO 转换源码版本为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。转换脚本先执行动作，再保存图像和本体；`states[t]` 对应动作前记录。发布数据已跳过最初五个采集步骤，且直接将末尾 `dones/rewards` 设为 1。处理保留全部已发布动作，不补造发布前缺帧，也不将末值视为独立成功依据。

T 条动作对应 T+1 条观测：

- `O[0]` 从 `source.states[0]` 正向运动学恢复本体；两路初始图像缺失，`image_valid[0]=false`，读取器返回 `None`。
- `O[t+1]` 来自 `source.obs[t]` 的实测本体和图像；每个动作明确对应 `before/action/after`。
- 最后一个动作后的本体和图像保留；发布版没有终端仿真状态，`simulation_state_valid[-1]=false`。

时间步长由仿真时间与控制频率交叉核验；时间缺口或间隔不一致会拒绝转换。线速度由实际位置和时间微分；角速度使用相邻 SO(3) 旋转差，避免轴角跨 ±π 的跳变。归一化 OSC 原动作保留为命令，不能当作实测速度。

位置是世界坐标下 `gripper0_grip_site` 的米制位置；观测姿态是 `robot0_right_hand` 的世界姿态，保存轴角、连续符号的 `xyzw` 四元数和旋转矩阵前两列。OSC 姿态使用 grip site，两者固定旋转保存为 `body_to_control_rotation`。角速度为世界坐标下 rad/s。

正向运动学使用保留全部自由度和坐标变换的精简 XML，仅用于运动学与时序抽检。它不能代替碰撞、动力学或任务成功验证。每回合抽检至多 8 个位置，报告移位对齐和同下标误差，以及姿态、关节与夹爪差异；不满足默认 1 cm / 5° 容差的回合保留失败标记。

## 命令区间、事件和拟合候选

本版本只接受已核验的二值夹爪命令 `OPEN=-1 / CLOSE=+1`；非二值或非有限命令显式拒绝，不通过滞回猜测目标。每次原始命令变化都保留。`command_hold_seconds` 默认 0.15 s，仅判断持续性：短命令的 `command_confirmed=None`、`command_persistent=false`，不会因此删除命令或动作。

`command_intervals` 在切换动作开始前的观测处划界，覆盖动作 `[start,end)`，拟合轨迹含观测 `[start,end]`。每条记录包含区间编号、当前命令、起止索引及操作事件引用。全部区间按时间连接，原始动作恰好覆盖一次。夹爪闭合、打开、并行机械臂运动和静止等待均留在对应区间内。

事件的命令持续性、反馈响应、反馈停稳和耦合运动分别记录，均不代表抓取成功。反馈停稳默认要求开度响应至少 2 mm、开度速度低于 0.01 m/s 并持续 0.15 s；运动超过 5 mm 或 5° 记为耦合。未停稳过程保留到下一命令或回合末尾。`phases` 仍提供 `motion / operation / coupled_operation` 的诊断划分，**不再用于排除拟合输入**。

每个命令区间必须有一个完整候选，短区间和等待亦然。在区间内由持续低速、方向变化及 0.5 s 网格提出其他边界；非完整候选最短默认 0.5 s。候选不能跨命令边界。先保留全部完整区间，再按优先级填充至每回合默认 128 个，裁剪数量进入清单。低速和转弯只是边界建议，末帧不是已核验的收敛目标。

`proposal_windows/H1,H3,H4,H5` 按命令区间排列端点位置/四元数、**当前及后续区间的保持命令**、区间引用、有效长度和后缀填充掩码。它们来自源示教查询，尚未经过 F 细分与执行；`field_supervision_mask / goal_supervision_mask / terminal_label_mask` 全为 false。`source_end_in_window` 仅说明源示教最后一个命令区间进入窗口，不是任务完成标签，也不是已验证场计划的终点标签。

## 文件与读取接口

```text
data/processed/libero_v2/
  manifest.json / normalization.json / selection.json
  run.json / report.md / integrity_audit.json
  preview.png / quality_summary.png / quality_analysis.json
  implementation/                  # 本次处理的包源码快照
  <suite>/<task>.hdf5
    environment / provenance / summary
    episodes/<demo>/
      time / actions / robot_state/*
      images/{agentview_rgb,eye_in_hand_rgb}
      image_valid / source_image_index
      simulation_state_valid / simulation_state_index
      body_to_control_rotation / source
      gripper_command / command_intervals / events
      phases / candidates / quality
      proposal_windows/H{1,3,4,5}/*
```

两路图像采用相对路径 HDF5 Virtual Dataset，`source` 使用相对外链，避免重复存储约 100 GB 原始数据。移动整个仓库可保持引用；单独分发结果须同时提供来源或实体化图像。读取器在转换进行中、来源缺失或发生变化时抛错，防止断开虚拟链接被静默读成全零。单独迁移数据子目录后应重新生成链接。

```python
from openpi.datasets.libero.reader import ProcessedLiberoDataset

dataset = ProcessedLiberoDataset("data/processed/libero_v2", split="train")
transition = dataset.transition(episode_index=0, action_index=3, images=True)
# before / action / after / language / split / task_success
# 图像可用性以 image_valid 为准，task_success 仍未知。

fit_input = dataset.candidate(episode_index=0, candidate_index=0)
# trajectory/time/actions: T+1 个物理单位状态与 T 个原始动作
# command_interval/candidate/operation_events: 命令、索引和离线事件信息
# frame/body_to_control_rotation/source: 坐标与溯源契约
# goal_candidate_verified=False，不能直接把末帧当作已核验目标。

proposals = dataset.proposals(episode_index=0, horizon=3)
# 仅用于离线分段分析和 F 输入准备，未来窗口不是策略输入。
```

以完整回合内容签名和固定种子划分，重复轨迹及其全部派生产物跟随同一划分。默认约 80/10/10，为同任务内示教划分；训练回合按帧加权估计均值、标准差、min/max 和受保护尺度，验证/测试不参与。独立 `audit` 核对来源、时序、命令原样保留、完整候选覆盖、窗口引用/命令/端点/掩码、首末可用图像、重复轨迹划分及训练集统计。

## 版本与后续接入

历史 `libero_v1/` 和 `libero_pilot/` 保留原有产物、清单及源码快照，读取器兼容 v1；新入口不覆盖这些历史结果。v1 的纯运动候选和过滤短命令的统计不能与 v2 混用。旧的散落模块和绘图脚本已收拢到此包，运行命令以本文为准。

三维 SEDS 参考 F、有限候选/DAG 选段、完整环境的连续诊断执行与激活前查询采集已于 2026-11-22 实现并小规模运行，见[参考动力系统说明](field_teachers.md)及其来源归档。当前新增输出为 `vlds.libero.offline_fields.v1`；语义确认边界、在线事实观察器、最终 `vlds.fields.v1` 多 H 主集和受扰标签仍未完成，v2 字段及监督掩码保持原状态。简单吸引子作为明确对照，不冒充已验证的复杂 SEDS 场。

本地实测规模与研究含义见论文项目的[实验记录](../../VLDS_paper/实验记录.md)，可重跑产物以各版本本地清单为准。

首版公式与已安装控制器接口可单独复查，不启动仿真或训练：

```bash
examples/libero/.venv/bin/python scripts/check_vlds_design.py --output docs/validation/vlds_design_2026-11-21.json
```

该脚本验证稳定解码的数值样本、参考步长、坐标/旋转和动作缩放，并核对零旋转的控制器行为；输出包含脚本及依赖源码哈希。它不是F拟合、数据转换或完整机器人执行测试，不会启用任何数据监督掩码。
