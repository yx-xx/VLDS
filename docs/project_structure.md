# 项目结构与开发约定

本项目保留 openpi 的模型、训练和客户端组织方式，将交互式观察功能按职责分层。观察台用于观看和人工验证模型，每次运行一个所选场景；批量性能评测通过显式命令行入口执行。

## 顶层目录

```text
VLDS/
├── README.md                  # 项目入口与上游安装说明
├── COMMANDS.md                # 常用启动命令
├── pyproject.toml             # 依赖、工作区、代码检查与 pytest 配置
├── uv.lock                    # 依赖锁定文件
├── src/openpi/                # 模型、策略、训练、共享工具和仿真应用
│   ├── datasets/              # 完整数据集版本管理、下载与内容校验
│   └── simulation/            # 交互式观察台与仿真运行控制
├── packages/openpi-client/    # 轻量客户端、通信及共享 MuJoCo 查看器
├── scripts/                   # 用户命令入口、训练工具和 Docker 脚本
├── tests/                     # 仿真应用与命令脚本测试
│   ├── datasets/
│   ├── simulation/
│   └── scripts/
├── examples/                  # 环境和硬件适配示例及安装脚本
├── docs/                      # 使用说明与开发文档
├── third_party/               # 第三方源码；按原有依赖管理方式维护
├── data/                      # 完整数据集、预览、处理结果与运行输出（不提交）
├── assets/                    # 模型所需资源（不提交）
└── checkpoints/               # 本地模型检查点（不提交）
```

`src/openpi/` 中原有的 `models/`、`models_pytorch/`、`policies/`、`training/`、`shared/` 等目录继续负责各自功能。其已有单元测试和客户端单元测试保留在源码旁，由同一个 pytest 配置发现。

## 仿真模块

```text
src/openpi/simulation/
├── __init__.py
├── __main__.py                # python -m openpi.simulation
├── cli.py                     # 参数解析；按需加载图形界面
├── paths.py                   # 仓库、LIBERO 资源和工作进程路径
├── catalog.py                 # 环境、任务类型、任务和模型匹配规则
├── ui/
│   ├── inspector.py           # 主窗口、导航与仿真控件
│   ├── dataset_browser.py     # 任务目录、回合筛选与播放控件
│   ├── dataset_library.py     # 数据集总览卡片
│   └── tk_runtime.py          # 系统 Tk 选择与字体配置
├── data/
│   ├── reader.py              # 数据集索引、格式读取及帧访问
│   ├── replay.py              # 处理后数据与来源仿真帧的明确关联
│   ├── watch.py               # 文件更新检测与稳定发布识别
│   ├── discovery.py           # 正式数据与预览发现、完整副本优先
│   ├── descriptions.py        # 数据集说明与中文任务标签
│   └── download.py            # 按需下载指定演示回合
├── runtime/
│   ├── controller.py          # 场景进程、模型服务与生命周期
│   ├── replay.py              # 记录状态三维回放的进程控制
│   └── franka.py              # 命令行单任务及显式批量运行
└── workers/
    ├── scene.py               # 独立进程中的实时场景与模型动作执行
    └── replay.py              # 独立进程中的记录状态回放
```

- `ui/` 负责展示和交互；`data/` 负责数据格式、索引与下载，不依赖图形界面。
- `runtime/` 管理异步任务、子进程与模型服务，具体的仿真执行放在 `workers/`。
- `catalog.py` 读取任务目录时不导入仿真器或模型；`--catalog` 和 `--help` 可在无桌面环境运行。
- `paths.py` 统一定位仓库和工作脚本，业务模块不各自根据目录层级推导仓库根路径。
- `scripts/view_simulation.py` 和 `scripts/run_franka_sim.py` 保留为薄入口，实际逻辑由包内模块维护。上游训练、服务和下载工具仍保留原有入口。

## 两套 Python 环境

主环境 `.venv/` 使用 Python 3.11，运行控制面板、数据读取、模型服务和 ALOHA。LIBERO 使用 `examples/libero/.venv/` 中的独立 Python 3.8 与对应 MuJoCo 依赖。

`workers/` 中的脚本通过绝对路径交给相应解释器直接执行，以隔离两套依赖。此目录应保持 Python 3.8 兼容，不导入主环境的 `openpi` 模型、界面或运行控制模块；可使用独立环境中安装的 `openpi_client`。

图形模块必须通过 `ui/tk_runtime.py` 导入 Tk，并在 Tk 初始化后按需导入 `PIL.ImageTk`。提前导入 `tkinter` 或 `ImageTk` 会使系统 Tk 和中文字体选择失效。包的 `__init__.py` 保持轻量，避免导入时启动图形环境。

## 数据、缓存与运行输出

| 路径 | 用途 |
| --- | --- |
| `data/dataset_previews/<组织>--<数据集>/` | 按需下载的演示回合、元数据与来源记录 |
| `data/datasets/<组织>/<数据集>/` | 浏览、训练和后续离线评测共用的完整 LeRobot 数据 |
| `data/datasets/raw/libero/` | 原始 LIBERO HDF5 演示 |
| `data/datasets/download_status.json` | 完整下载进度、校验计数与失败文件 |
| `data/sim_inspector/` | 观察台模型服务日志；本地调试截图也可放在这里 |
| `data/franka_sim/` | 命令行仿真导出的录像、结果与服务日志 |
| `examples/libero/.cache/` | LIBERO 源码、资源与本机配置 |
| `.cache/`、`.pytest_cache/`、`.ruff_cache/` | 工具缓存 |
| `assets/` | 本地归一化统计等模型配套资源 |
| `checkpoints/<配置>/<实验>/` | 训练输出的模型参数、优化器状态和检查点 |
| `~/.cache/openpi/`（仓库外） | 下载的预训练模型和公共资源；可由 `OPENPI_DATA_HOME` 指定位置 |

上述仓库内的本地资源目录由 `.gitignore` 排除，默认预训练缓存位于仓库外。本机数据、模型配套资源和下载缓存分别存放在项目内对应的普通目录中。新增数据集的读取逻辑放在 `data/`；新增仿真适配放在 `workers/` 并在 `catalog.py` 注册；新增控件放在 `ui/`。

## 本地资源与 Git 管理

本机数据、模型配套资源、检查点和缓存分别放在项目内对应的普通目录中，由 `.gitignore` 排除。

```text
VLDS/                            # Git 代码仓库
├── src/、scripts/、tests/        # 源码、命令入口与测试
├── data/                        # 完整数据集、预览、处理结果和原有运行输出
├── assets/                      # 模型配套资源
├── checkpoints/                 # 训练输出的模型参数与检查点
├── .cache/                      # 项目下载工具缓存
└── examples/libero/.cache/       # LIBERO 源码、仿真资源及本机配置
```

- 完整数据集位于 `data/datasets/`，训练、下载和仿真直接访问原路径。
- 已下载的预训练模型默认位于 `~/.cache/openpi/openpi-assets/checkpoints/`；`OPENPI_DATA_HOME` 可指定其他缓存位置。
- `.venv/` 与 `examples/libero/.venv/` 保留为本地运行环境，继续被 Git 忽略。
- `.gitignore` 按资源目录排除内容；`src/openpi/models/`、`src/openpi/datasets/` 等源码目录仍正常纳入 Git。
- 数据来源、版本和校验记录随本地数据保留；需要共享的实验配置、版本说明和结果摘要写入代码或文档。

新克隆的仓库不会包含这些本机资源，可按下载说明在默认目录中重新准备。移动或备份本机项目时，应同时保留需要的数据、检查点和缓存目录。

## 测试与检查

应用与脚本测试统一放在 `tests/`，文件名使用 `test_*.py`。pytest 默认扫描 `tests/`、`src/` 和 `packages/`；CI 无需指定旧脚本目录。`tests/scripts/test_train.py` 会执行小规模训练，完整测试所需资源按上游环境要求准备。

在仓库根目录运行：

```bash
# 仿真与数据浏览回归；有桌面 DISPLAY 时同时检查字体和页面交互
env -u PYTHONPATH .venv/bin/python -m pytest tests/simulation -q

# 下载工具回归
env -u PYTHONPATH .venv/bin/python -m pytest tests/scripts/test_download_checkpoints.py -q

# 检查本次整理涉及的 Python 代码
env -u PYTHONPATH .venv/bin/ruff check src/openpi/simulation tests scripts/view_simulation.py scripts/run_franka_sim.py
env -u PYTHONPATH .venv/bin/ruff format --check src/openpi/simulation tests scripts/view_simulation.py scripts/run_franka_sim.py

# 按上游约定运行完整自动测试
env -u PYTHONPATH .venv/bin/python -m pytest --strict-markers -m 'not manual'
```

图形测试在没有 `DISPLAY` 时跳过。工作进程入口测试分别使用主环境和已安装的 LIBERO 解释器，验证移动后的脚本可直接执行。测试不自动下载权重或启动整套数据集性能评测。
