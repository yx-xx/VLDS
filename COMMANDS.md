# 项目启动命令

项目启动命令统一维护在本文件，均在项目根目录执行。

## 环境准备（首次）
```bash
bash examples/libero/setup_franka.sh
```

## 仿真

默认使用 π₀.₅，结果自动保存到 `data/franka_sim/<模式>_<时间戳>/`。

### 1. 全量评测：5 个任务集，全部任务，每任务 50 次

无窗口运行，耗时较长；这里评测的是全部仿真任务，不是重放训练数据集。

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_franka_sim.py --mode full
```

### 2. 实时观看：空间、物体、目标三类场景，各选 3 个任务

每任务运行 1 次，共 9 回合；依次打开场景和腕部相机预览。

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_franka_sim.py --mode view
```

### 关键参数

| 参数 | 含义 |
| --- | --- |
| `--mode` | `full`：全量评测；`view`：多场景实时观看。 |
| `--models` | 模型：`pi05`（默认）、`pi0`、`pi0_fast`；多个模型按顺序运行。 |
| `--trials` | 每任务次数，可覆盖模式默认值：全量 50 次、观看 1 次。 |
| `--seed` | 随机种子，默认 `7`；模型比较时保持一致。 |
| `--output-dir` | 自定义保存目录；不传则按模式自动创建时间戳目录。 |

- FAST 需要先下载动作 tokenizer；实时窗口在模型服务和仿真初始化后出现。
- 预览中按 `Q` / `Esc` 只隐藏窗口；在启动终端按 `Ctrl+C` 停止运行。
- 视频、结果、日志位于 `<输出目录>/<模型>/<任务集>/`：`*.mp4`、`results.json`、`server.log`。
- 所有任务集共用一份 `summary.json` 汇总。本入口只运行仿真，不连接真机。
- 端口占用时加 `--port 8001`；更多参数执行 `env -u PYTHONPATH .venv/bin/python scripts/run_franka_sim.py --help` 查看。

## 独立推理服务（不启动仿真）

仿真启动器会自动管理服务，通常不需要单独启动；不要与仿真同时占用相同端口。

```bash
env -u PYTHONPATH .venv/bin/python scripts/serve_policy.py \
  --host 127.0.0.1 --port 8000 policy:checkpoint \
  --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

`--policy.config` 与 `--policy.dir` 必须匹配；后者也可指定兼容的本地检查点目录。服务只监听本机，不直接控制机器人。
