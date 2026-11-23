# 项目启动命令

以下命令均在代码仓库根目录执行；若当前位于包含三个项目的工作空间，先运行 `cd VLDS`。当前主要入口用于交互式模型观察、单场景验证和数据集演示浏览。目录职责见[项目结构说明](docs/project_structure.md)，其他文档见[文档索引](docs/README.md)。

移动现有项目目录后，先按[目录迁移检查](docs/project_structure.md#目录迁移检查)确认两套 Python 环境与本地资源路径有效。

## 环境准备（首次）

```bash
bash examples/libero/setup_franka.sh
```

项目主环境包含 ALOHA Sim 依赖；LIBERO 使用独立的仿真 Python 环境。

## 模型观察台（主要入口）

```bash
env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py
```

等价的模块入口：`env -u PYTHONPATH .venv/bin/python -m openpi.simulation`。
查看已安装环境和任务目录：`env -u PYTHONPATH .venv/bin/python -m openpi.simulation --catalog`。

在选择面板中依次选择：**仿真环境 → 任务集合 → 操作类型 → 具体任务 → 初始状态**。
点击“打开所选场景”后在观察台内显示 MuJoCo 三维视图，默认暂停；模型加载完成后点击“开始 / 继续”或“单步”。
每次只运行所选任务，结束后保留现场，等待手动重置或切换。

| 环境 | 可选任务 |
| --- | --- |
| LIBERO / Franka Panda | 空间关系、目标物体、目标要求、多步骤、综合任务；可进一步按操作类型筛选 |
| ALOHA Sim / 双臂 | 方块交接、精细插接；默认模型适配方块交接，插接推理需填写兼容检查点 |

- “重置”恢复同一初态；“换一个初始状态”只更换当前任务的初态。
- “打开下一个任务”是手动切换；同一模型保持加载，避免每个场景重新加载权重。
- 取消“打开场景时同时加载模型”，可以先只看场景，再点击“加载 / 切换模型”。
- 默认使用软件绘制界面，模型推理仍使用 GPU；图形驱动正常的机器可取消。
- 内嵌三维视图：左键旋转、右键平移、滚轮缩放；点击视图后空格暂停 / 继续、右方向键单步，可点击“复位视角”。
- 关闭选择面板会清理它启动的场景和模型服务。运行日志位于 `data/sim_inspector/`。

更多设计说明、模型适配边界及扩展建议见 `docs/simulation_inspector.md`。

## 完整数据集下载与训练共用

```bash
env -u PYTHONPATH .venv/bin/python scripts/download_datasets.py --endpoint https://hf-mirror.com --workers 12
```

正式数据统一保存于 `data/datasets/<组织>/<数据集>/`，原始 LIBERO 保存于 `data/datasets/raw/libero/`。完整范围、固定版本、续传、校验状态及训练环境变量见[完整数据集说明](docs/datasets.md)。

## 观看数据集演示

在观察台中切换到 **“数据集浏览”** 页签：选择本地数据集 → 操作类型 → 任务 → 回合 → 相机。
可暂停、逐帧、拖动进度、调整速度，并查看对应动作与机器人状态。含 XML 和仿真状态的 LIBERO HDF5 可打开完整三维回放。

观察台自动发现正式数据目录中的完整数据集，并在校验完成后优先显示完整副本。原单回合缓存保留；“添加一个演示回合”可用于额外预览，官方源连接失败时切换到镜像源。
也可加载自己的 LeRobot v2.x 根目录或 LIBERO / ALOHA HDF5 文件。

直接打开 LIBERO 演示：

```bash
env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py \
  --dataset data/dataset_previews/physical-intelligence--libero
```

回放每次只观看所选回合，末尾暂停；不加载模型，也不启动数据集性能评测。格式边界见 `docs/simulation_inspector.md`。

## 原命令行单任务入口

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_franka_sim.py \
  --viewer --software-rendering --suite libero_spatial --task-ids 0
```

`--models` 支持 `pi05`（默认）、`pi0`、`pi0_fast`；`--checkpoint` 可指定兼容检查点。
额外加 `--camera-preview` 可打开场景与腕部相机预览。启动终端按 `Ctrl+C` 停止。
旧的批量评测参数保留在该命令行脚本中，模型观察台不会调用批量模式。

## 独立推理服务（不启动仿真）

观察台会自动管理服务，通常无需单独启动：

```bash
env -u PYTHONPATH .venv/bin/python scripts/serve_policy.py \
  --host 127.0.0.1 --port 8000 policy:checkpoint \
  --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

`--policy.config` 与 `--policy.dir` 必须匹配。服务只监听本机。

## 数据处理结果实时观看

```bash
# 原始演示：选择回合后自动加载内嵌 MuJoCo 视图，并检测文件更新。
env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py \
  --dataset data/datasets/raw/libero --replay --watch

# 本机已生成的降采样示例：三维来源帧与处理后的数值、录制画面同步。
env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py \
  --dataset data/processed_examples/libero_stride2 --replay --watch

# 只包含完整场景状态的 HDF5 也可以播放。
env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py \
  --dataset data/processed_examples/libero_states_only.hdf5 --replay --watch
```

文件更新时优先使用临时文件加 `os.replace` 发布；界面的“重新载入当前数据”可以手动刷新。处理后的 LeRobot 数据可以通过 `meta/replay.json` 关联原始场景，具体约定见 `docs/simulation_inspector.md` 的“数据处理时实时观看”。

## LIBERO 示教预处理

处理后的 LIBERO 数据默认自动加载三维回放，切换任务或回合时自动更新；“演示回合”选择器在三维视图上方的通栏中。

```bash
# 观看已处理数据：MuJoCo 内彩色末端轨迹、当前段高亮，与录制画面同步
env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py \
  --dataset data/processed/libero_v2 --replay

# 导出彩色分段的三维末端轨迹及时间线
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero inspect \
  data/processed/libero_v2 --episode 0 --segments
```


```bash
env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero process \
  --workers 4 --output data/processed/libero_v2

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero audit \
  data/processed/libero_v2

env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero inspect \
  data/processed/libero_v2 --episode 0
```

输出规范化示教、完整夹爪命令区间、拟合候选与 H=1/3/4/5 诊断窗口；候选覆盖全部动作，含耦合运动和等待。运行状态、监督可用性和 F 接入边界见 [LIBERO 数据处理](docs/data_processing.md)。

## 参考动力系统拟合与完整环境诊断

SEDS 参考 F、完整动作覆盖的有限选段、简单吸引子对照、控制偏移校准和连续执行，统一使用 `python -m openpi.datasets.libero` 的 `fit-fields / calibrate-fields / rescore-fields / execute-fields / inspect-fields / audit-fields` 子命令。完整命令与真实结果边界见[参考动力系统说明](docs/field_teachers.md)。新增离线池不是 `vlds.fields.v1` 主训练集；旧 v2 候选掩码保持 false。

### 动力系统场与分段轨迹对照

已处理 LIBERO 的观察台浏览默认自动分段、后台预取场快照并随时间轴切换场，分段边界不会在播放主线程中重新采样；也可用“诊断整命令段”或“载入拟合结果…”手动检查，在 MuJoCo 中显示速度箭头、参考轨迹、名义积分和目标，支持聚焦、显示开关及导出。三维渲染进程与数据时间轴异步运行，软件渲染较慢时只跳过中间三维帧，不拖住录制画面播放。独立导出已有场：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.libero inspect-vector-field data/processed/libero_v2 \
  --episode 6000 --fields data/experiments/fields_20261122/pilot_c/episode_06000.json \
  --sequence 0 --field 1 --output data/experiments/field_visualization/saved_field.png
```

省略 `--fields` 并用 `--interval 1` 可重新拟合整个命令段。索引从 0 开始；名义轨迹不代表物理执行。参数与参考来源说明见[场可视化](docs/field_teachers.md#场可视化与逐段拟合诊断)。

## 全量数据集动力系统分段与播放

从本仓库根目录执行；省略回合选择即处理所有回合及划分。代码/配置变化后使用新输出目录。完整说明及控制合规限制见 [docs/dataset_dynamics.md](docs/dataset_dynamics.md)。

```bash
# 全部完整数据集：离线构建并登记为播放默认值
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics --output data/processed/dynamics_new --workers 10

# 复核本次已完成的 LIBERO 回放场（不启动机器人执行）
OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.audit data/processed/dynamics_20261122_final/libero

# 在已有回放结果旁生成 LIBERO 原生产控制配置：0.3 m/s、0.5 rad/s、姿态增益 2/s
# 此复建示例写入新的 experiments/production_libero_recheck；失败叶子保留并记录原因
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.profiles \
  data/processed/dynamics_20261122_final/libero --profile production_libero --workers 10 \
  --output data/experiments/production_libero_recheck

# 独立复核生产配置的覆盖、保存轨迹和约束上限
OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.audit \
  data/processed/dynamics_20261122_final/libero/profiles/production_libero

# 固定生产约束、保持原控制周期，通过明确放慢示教时间拟合全量回合
# 播放时在“配置”下选择“生产约束/放慢时间”
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.profiles \
  data/processed/dynamics_20261122_final/libero --profile production_retimed --workers 10

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.audit \
  data/processed/dynamics_20261122_final/libero/profiles/production_retimed --workers 10

# 浏览保存的分段与场，切换回合不会调用拟合优化器
env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py --dataset data/processed/libero_v2
```

### 长片段与非线性稳定场

原始片段至少 10 个观测转移且源时长至少 0.5 s；失败长段和短命令事件保留。此配置使用基础回放速度上限，不能与固定生产控制配置混称。某类全量结果通过独立完整性复核后，观察台会默认选择它。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.coherent \
  data/processed/dynamics_20261122_final/libero --workers 8

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.coherent_audit \
  data/processed/dynamics_20261122_final/libero/profiles/coherent --workers 8

OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.coherent_report data/processed/dynamics_20261122_final \
  --output data/experiments/dynamics_long_segments_20261123/analysis
```

对其他集合将 `libero` 替换为 `lerobot_libero`、`aloha`、`pusht` 或 `xarm`。夜间执行状态、失败与复现边界见[长片段复核记录](docs/validation/dataset_dynamics_long_segments_2026-11-23.md)。

### 分段末端减速诊断

名义积分在段末收敛会造成速度下降；状态连续不等于速度连续。新增 `--goal-lookahead-steps 4` 试验独立过点场配置，完整性审计覆盖全量后观察台会自动发现“长片段/过点场”。已审计样例可提前手动检查：选择处理后 LIBERO v2 的回合 6000，点击“载入拟合结果…”，选择 `data/experiments/dynamics_passthrough_20261123/v5/libero/episode_06000.json`。其他四类数据也支持载入对应回合，失败状态会保留。本轮样例、真实停止保护、整回合回退和验收边界见[段末减速与过点场](docs/dataset_dynamics.md#段末减速与过点场)。已有窗口可继续使用，新增入口在下次启动时生效。不要直接覆盖已有结果目录。

```bash
OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.continuity \
  data/experiments/dynamics_passthrough_20261123/v5/libero/episode_06000.json \
  --baseline data/processed/dynamics_20261122_final/libero/profiles/coherent/episode_06000.json \
  --output data/experiments/dynamics_passthrough_20261123/analysis/libero_06000
```
