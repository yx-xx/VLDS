# 项目启动命令

均在项目根目录执行。当前主要入口用于交互式模型观察、单场景验证和数据集演示浏览。目录职责见[项目结构说明](docs/project_structure.md)，其他文档见[文档索引](docs/README.md)。

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
点击“打开所选场景”后显示 MuJoCo 完整三维界面，默认暂停；模型加载完成后点击“开始 / 继续”或“单步”。
每次只运行所选任务，结束后保留现场，等待手动重置或切换。

| 环境 | 可选任务 |
| --- | --- |
| LIBERO / Franka Panda | 空间关系、目标物体、目标要求、多步骤、综合任务；可进一步按操作类型筛选 |
| ALOHA Sim / 双臂 | 方块交接、精细插接；默认模型适配方块交接，插接推理需填写兼容检查点 |

- “重置”恢复同一初态；“换一个初始状态”只更换当前任务的初态。
- “打开下一个任务”是手动切换；同一模型保持加载，避免每个场景重新加载权重。
- 取消“打开场景时同时加载模型”，可以先只看场景，再点击“加载 / 切换模型”。
- 默认使用软件绘制界面，模型推理仍使用 GPU；图形驱动正常的机器可取消。
- 三维窗口：左键旋转、右键平移、滚轮缩放、空格暂停 / 继续、右方向键单步。
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
# 原始演示：选择回合后自动打开 MuJoCo，并检测文件更新。
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
