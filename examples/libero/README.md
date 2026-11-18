# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

For local Franka Panda simulation with π₀, π₀-FAST, and π₀.₅, see [the local simulation instructions](#franka-panda-local-simulation).

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

## Franka Panda local simulation

此入口使用 LIBERO / MuJoCo 的 Franka Panda，不连接真机，不需要训练数据集或额外微调。默认离屏渲染并保存 MP4；添加 `--viewer` 可实时查看场景和腕部相机。

### 安装与启动

在仓库根目录执行安装命令；已安装的机器不必重复安装：

```bash
bash examples/libero/setup_franka.sh
```

安装脚本使用 `uv.lock` 准备 Python 3.11 模型环境，并在 `examples/libero/.venv` 创建 Python 3.8 仿真环境。仿真使用 CPU 版 PyTorch，模型仍使用 GPU / JAX 推理。
LIBERO 固定到 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，源码位于 `examples/libero/.cache/LIBERO`，不依赖 Git 子模块。源码获取失败最多尝试 3 次；安装使用 setuptools 兼容模式支持上游命名空间布局。

运行 π₀.₅ 的一个完整任务（默认 `libero_spatial`、任务 0、一次测试）：

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_franka_sim.py --models pi05
```

启动器自动管理推理服务和仿真客户端。清除继承的 `PYTHONPATH` 可避免 ROS 等外部 Python 环境干扰。首次运行需要加载权重并编译 JAX，可查看下方的 `server.log`。

### 实时观看

在本机图形桌面的终端执行：

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_franka_sim.py \
    --models pi05 --viewer --output-dir data/franka_sim/live
```

窗口左侧是场景相机，右侧是腕部相机，并显示任务编号、回合、控制步数和推理状态。预览只读取已有观测，不修改相机、物体、动作或任务判定；原有视频和 JSON 仍会保存。
按 `Q` / `Esc` 或关闭窗口只隐藏预览，仿真继续运行；在启动终端按 `Ctrl+C` 停止运行。预览在模型服务就绪、仿真初始化后出现，首次推理编译期间可能暂时停留在等待画面。
这里的“实时”指运行过程中更新画面，不保证模拟时间与墙钟时间相同。无桌面 `DISPLAY` 的 Linux 环境应省略 `--viewer`；正式批量评测也建议关闭预览，避免显示开销影响耗时比较。

建议按“短流程检查 → 单任务实时观察 → 无窗口批量评测”的顺序运行。比较模型时保持任务、官方初始状态、随机种子、步数上限和重规划设置一致，使用多个初始状态，而不是只观察一次成功。
每次实验使用独立的 `--output-dir`，保留视频、结果和日志。正式性能分析还应单独测量预热后的推理延迟和墙钟耗时，不能用 MP4 的播放帧率代替控制频率或实时性能。

### 短流程检查

仅验证推理、控制和渲染链路时，可缩短为 5 步；这不代表任务完成，也不能用于报告基准成功率：

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_franka_sim.py \
    --models pi05 --max-steps 5 --output-dir data/franka_sim/smoke
```

### 模型与任务

`--models` 支持 `pi0`、`pi0_fast`、`pi05`，分别加载 `pi0_libero`、`pi0_fast_libero`、`pi05_libero`。这些检查点包含 LIBERO 适配权重和归一化统计，不应替换成 ALOHA / DROID 权重进行 LIBERO 比较。
多个模型顺序运行，不会同时占用显存。π₀-FAST 还需要从 Hugging Face 下载动作 tokenizer。

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_franka_sim.py \
    --models pi0 pi0_fast pi05 --suite libero_spatial --task-ids 0 1 2 --trials 3
```

支持 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 和 `libero_90`。任务编号从 0 开始；`--all-tasks` 运行整个任务集。同一命令使用相同任务、初始状态编号和随机种子。
使用自己的兼容检查点时，只选择一个模型，并传入 `--checkpoint /absolute/path/to/checkpoint`。完整参数可通过 `--help` 查看。

模型缓存默认为 `~/.cache/openpi`，可用 `OPENPI_DATA_HOME` 修改；三套权重合计约 33 GiB。网络较慢时可先使用断点下载工具（需要 `aria2c`，或通过 `--aria2c` 指定路径）：

```bash
env -u PYTHONPATH .venv/bin/python scripts/download_checkpoints.py \
    gs://openpi-assets/checkpoints/pi05_libero \
    gs://openpi-assets/checkpoints/pi0_libero \
    gs://openpi-assets/checkpoints/pi0_fast_libero
```

下载器使用两个官方 GCS HTTPS 入口并固定对象版本，校验文件大小、CRC32C 和可用时的 MD5；全部通过后才发布完整缓存。中断后运行相同命令继续，不要将 `.partial` 目录用于推理。上游对象版本改变时会拒绝混用旧分片。

### 输出与排查

- `data/franka_sim/<model>/<suite>/*.mp4`：每个回合的视频。
- `data/franka_sim/<model>/<suite>/results.json`：任务结果、成功率和运行错误。
- `data/franka_sim/<model>/<suite>/server.log`：模型加载、编译和服务日志。
- `data/franka_sim/summary.json`：本次运行汇总。

重复运行会覆盖对应输出；保留实验时使用不同的 `--output-dir`。环境或推理错误会使命令失败，不会混同为普通的任务失败。
端口 8000 被占用时使用 `--port 8001`；启动等待时间可用 `--startup-timeout` 调整，默认为 3600 秒。默认使用 EGL；有可用 X11 显示服务时可尝试 `MUJOCO_GL=glx`。

### 真机部署边界

LIBERO 的 7 维动作是末端控制量和夹爪命令，不是 Franka 的 7 个关节角，不能直接转发给真机关节接口。真机迁移仍需确认相机标定、状态定义、坐标系、动作单位、控制周期、夹爪方向、归一化统计与匹配的检查点。
本入口没有实现或验证真机驱动的工作空间、速度限制、通信超时保护和硬件急停；仿真成功不构成真机效果或安全性保证。
仿真策略服务只监听 `127.0.0.1`；远程部署需使用受控网络或隧道，现有 WebSocket 服务没有身份认证，不能直接暴露到公网。
