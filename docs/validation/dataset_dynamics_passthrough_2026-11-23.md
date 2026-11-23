# 分段末端收敛与过点目标复核

2026-11-23。分段处传递位置不代表速度连续；旧场将片段末观测设为稳定吸引点，会提前减速。本轮只改进位置场的过点目标选择，保留源数据、真实停留、夹爪事件和原生坐标定义。

## 最终版本代表回合

| 数据集 | 样例回合 | 原终点配置通过 | 过点配置通过 | 整回合回退 |
| --- | --- | ---: | ---: | ---: |
| libero | 0, 1000, 3000, 6000 | 4/4 | 4/4 | 0 |
| lerobot_libero | 0, 846, 1692 | 3/3 | 3/3 | 1 |
| aloha | 0 | 0/1 | 0/1 | 0 |
| pusht | 0 | 0/1 | 0/1 | 0 |
| xarm | 0, 400, 799 | 0/3 | 0/3 | 0 |

所有表内样例均完成参数逐步独立重算、源状态一致性和实际目标校验。“通过”仍指原有最大位置/姿态误差与目标收敛验收，不等于速度连续或物理操作成功。样例没有覆盖全部 9,249 回合，尚未发布全量 `coherent_flow` 默认配置。

## LIBERO episode 6000

前后均为 6 个长段，覆盖 98 个源观测转移，分段边界相同。

- 五个运动边界的末步速度/示教速度中位数：0.483 → 0.789。
- 末步速度不足示教一半的边界提示：3 → 0。
- 最大位置误差：9.239 → 8.967 mm；原门槛 10 mm。

![原示教、终点场与过点场的速度和误差](../../data/experiments/dynamics_passthrough_20261123/analysis/libero_06000/boundary_speed.png)

橙色是原始示教，紫色是终点吸引场，蓝色是过点场；竖虚线标记相同的切换时刻。紫色曲线在边界附近的速度低谷支持场自身减速的诊断，图中仍可能有速度跳变，不宣称 C1 连续。

## 实现与保留边界

`contractive.fit` 支持显式吸引目标，回归和自由积分使用同一个目标；`coherent.py` 在未来有限源观测中搜索目标，以原片段最大误差和末步速度误差选择，按原时刻切换，延续上段状态。终端、零速度和反向边界保留原目标。全轨迹退化时，只有重拟合的原终点整条方案能通过才回退，并保留失败尝试统计。

源状态三维回放与紫色动力系统积分是两个数据来源。渲染/预取调度的阻塞修复不能解决场本身的提前收敛；场目标修正也不会改变原始机器人记录状态。姿态场仍使用原终点，整条切换系统的全局稳定性、速度连续性和物理执行尚未验证。

## 在观察台观看

选择处理后 LIBERO v2 的回合索引 6000，点击“载入拟合结果…”，选择 `data/experiments/dynamics_passthrough_20261123/v5/libero/episode_06000.json`。配置中出现“长片段/过点场”，时间轴、场和名义积分同步更新。其他四类数据可载入各自样例，原生状态投影支持区间播放；失败场保留验收原因。手动载入会重新核验源数据和参数积分，样例只加入当前会话，不改全量默认登记。

已用独立隐藏 Tk 窗口，对五类真实源数据完成文件载入、时间轴同步及按钮检查。LIBERO 检查了三维快照准备，其余类型检查了原生画布图元与区间播放按钮。该检查使用模拟回放接口，没有新建机器人 worker，也没有操作用户已开的仿真窗口。新增源码需要在方便时重新启动观察台后使用。

## 复现与验证

相关测试 192 passed；Ruff、Python 编译与 `git diff --check` 通过。测试范围见同名 JSON。

从仓库根目录执行：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  data/experiments/dynamics_passthrough_20261123/run_pilots.py libero lerobot_libero aloha pusht xarm --workers 2
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  data/experiments/dynamics_passthrough_20261123/summarize.py
```

本地输出位于 `data/experiments/dynamics_passthrough_20261123/v5/`；各目录的 `run.json`、`source_snapshot/`、`validation_snapshot/`、`audit.json` 保存配置、代码、独立复核及数据散列。已有目录只在代码和配置指纹一致时复用，后续修改应使用新的输出目录。可分享的摘要和逐边界数值在[同名 JSON](dataset_dynamics_passthrough_2026-11-23.json)。
