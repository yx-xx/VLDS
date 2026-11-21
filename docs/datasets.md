# 本地完整数据集

数据统一保存于 `data/datasets/`，供观察台、LeRobot 训练加载器和后续离线评测共用。之前的 `data/dataset_previews/` 是单回合预览缓存；完整副本校验完成后，观察台默认优先显示完整副本。

## 下载范围与固定版本

| 名称 | 数据源 | 选择的版本 | 本地目录 |
| --- | --- | --- | --- |
| LIBERO / LeRobot | `physical-intelligence/libero` | `v2.0` | `data/datasets/physical-intelligence/libero/` |
| ALOHA 方块交接 | `lerobot/aloha_sim_transfer_cube_human` | `v2.1` | `data/datasets/lerobot/aloha_sim_transfer_cube_human/` |
| PushT | `lerobot/pusht` | `v2.1` | `data/datasets/lerobot/pusht/` |
| xArm 抬升 | `lerobot/xarm_lift_medium` | `v2.1` | `data/datasets/lerobot/xarm_lift_medium/` |
| 原始 LIBERO | `yifengzhu-hf/LIBERO-datasets`，LIBERO 自带下载工具使用的源 | 首次下载时固定 `main` 的提交 | `data/datasets/raw/libero/` |

下载器将标签解析到具体提交并保存清单，此后使用提交号下载。实际提交、来源、字节数和完成状态记录在各数据集的 `dataset_source.json`，远程逐文件清单保存在 `.vlds/manifest.json`。不将浮动的 `main` 作为已下载版本标识。

此次解析的五个快照合计 3,961 个文件、135,485,585,963 字节，约 135.49 GB（126.18 GiB）。规模与完成状态分别是两个概念；是否完成以校验记录为准。

## 下载与续传

在项目根目录执行：

```bash
env -u PYTHONPATH .venv/bin/python scripts/download_datasets.py \
  --endpoint https://hf-mirror.com --workers 8 --parts 8 --libero-mirror modelscope
```

默认下载表中所有数据集。只补某个数据集可加 `--datasets libero`、`--datasets aloha` 等。官方源使用 `--endpoint https://huggingface.co`。

`--parts` 设置每个大文件的分段并发数，范围片段单独保存在隐藏的 `.ranges/` 目录中；中断后保留已完成的片段，合并后仍以原始清单的完整文件哈希为准。`--libero-mirror modelscope` 允许 LIBERO 使用已核对内容的国内传输镜像；镜像读取失败时回退到固定提交的 Hugging Face 文件。镜像不改变数据版本或校验标准，移除此参数即可只使用 Hugging Face 源。

文件以 `.part` 形式接收，支持 HTTP Range 续传；只有长度和内容哈希均匹配清单后才发布正式文件。Git LFS 文件校验 SHA-256，普通文件校验 Git blob 哈希。已有预览文件只有校验匹配后才复用，避免重复保存。

每个数据集的全部文件校验通过后，才将其 `dataset_source.json` 的 `complete` 设为 `true`。全局进度与失败文件记录在 `data/datasets/download_status.json`。中断或失败后执行同一命令即可复核已有文件并继续下载；请等前一个下载进程退出后重试。

本次终端任务日志保存在 `data/dataset_download.log`。运行中的进度可查看：

```bash
cat data/datasets/download_status.json
tail -n 20 data/dataset_download.log
```

## 浏览与三维回放

观察台会自动发现上述 LeRobot 目录，以及原始 LIBERO 目录中的 HDF5 文件。刷新数据集总览即可读取新下载的内容。显式打开预览路径仍查看该预览副本。

```bash
env -u PYTHONPATH .venv/bin/python -m openpi.simulation \
  --dataset data/datasets/physical-intelligence/libero
```

原始 HDF5 中的图像、动作、机器人状态和仿真状态以实际字段为准；回合同时包含模型 XML 与记录状态时，可在原生三维窗口中回放。LeRobot 格式用于统一的训练与数据读取，不保证包含重建完整三维现场所需的信息。

## 供训练与后续评测使用

本地目录采用 LeRobot 的 `<根目录>/<组织>/<数据集>` 布局。启动训练、计算归一化统计或其他数据处理前，设置：

```bash
export HF_LEROBOT_HOME="$PWD/data/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

这几项变量必须在 Python 进程启动前设置。之后已有的 `pi05_libero`、`pi0_libero` 和 `pi0_aloha_sim` 等配置按原 repo_id 读取本地完整数据。

例如，只计算 LIBERO 的归一化统计：

```bash
env -u PYTHONPATH .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_libero
```

PushT、xArm 下载后可进行数据浏览和通用数据分析；将它们用于当前 VLA 模型训练，仍需配置对应的观测字段、动作维度及机器人适配。原始 LIBERO 可用于检查仿真记录、构造离线实验和三维回放。

完整演示数据可供后续建立离线指标。闭环成功率由模型在仿真任务中执行后的结果计算；训练集、验证集与评分数据的划分应在具体实验中固定并单独记录。

## 示教预处理与拟合输入

完整原始 LIBERO 已提供可复现的处理入口，输出到 `data/processed/libero_v1/`。时序约定、缺失图像、夹爪耦合事件、分段候选、回合划分、归一化与读取接口见 [LIBERO 数据处理](data_processing.md)。候选数据与通过物理执行的场监督分别管理，是否完成以各自的清单字段为准。
