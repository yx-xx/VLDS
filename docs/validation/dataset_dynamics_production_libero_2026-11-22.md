# LIBERO 原生产控制配置复验（2026-11-22）

本记录把已有 `dynamics_20261122_final/libero` 的源状态和命令区间重新拟合到独立的 `production_libero` 配置，默认回放结果没有被覆盖。固定约束为位置限速 `0.3 m/s`、角速度上限 `0.5 rad/s`、姿态反馈增益 `2/s`；位置/姿态误差阈值、额外收敛时间、连续状态传递和失败叶子保留规则与回放构建一致。

## 结果

| 指标 | 数值 |
| --- | ---: |
| 回合 | 6,500 / 6,500 |
| 全部场 | 676,614 |
| 通过场 | 425,084 |
| 未通过场 | 251,530 |
| 单转移场 | 624,842 |
| 通过回合 | 2,787 |
| 未通过回合 | 3,713 |
| `rotation_maximum` 失败叶子 | 251,226 |
| `position_maximum` 失败叶子 | 337 |
| `nominal_arrival_timeout` 失败叶子 | 0 |
| 最大归一化位置误差 | 7.068887 |
| 保存轨迹与独立重算最大差异 | 2.22e-15 |

失败场保存在逐回合 JSON 中，失败回合没有从索引中删除。`rotation_maximum` 是主要瓶颈：在固定 `2/s` 姿态增益和 `0.5 rad/s` 角速度上限下，当前固定分段末姿态目标与源时序组合出现大量滞后，细分到单转移仍有失败；这个结果不证明其他目标、分段或模型都无解。位置约束只造成 337 个失败叶子。

后续 v2 独立审计为 `integrity_passed=true`、`all_nominal_passed=false`；兼容字段 `passed=true` 表示产物完整性，即覆盖、边界、参数上限、保存轨迹和逐步重算一致；它不表示 6,500 个回合都通过生产配置。生产配置真正通过的是 2,787 个回合。结果仍保持 `physical_execution_validated=false` 和 `training_ready=false`。

## 播放验证

浏览器通过动力系统面板的“配置”选择器读取 `replay` 或 `production_libero`。episode 6000 已验证两套配置都能加载，回放配置为 20 个场，生产配置为 98 个场；生产配置中的失败场仍显示其具体原因，时间轴和场叠加随配置同步切换。生产配置的紫色轨迹是固定约束下的名义积分，仍不代表机器人实际执行。

相关测试 **202 passed in 20.91s**；`DISPLAY=:1` 下两项真实 Tk 浏览器导航/刷新测试通过。Ruff、编译检查和两个仓库的 `git diff --check` 通过。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.profiles \
  data/processed/dynamics_20261122_final/libero --profile production_libero --workers 10 \
  --output data/experiments/production_libero_recheck

OPENBLAS_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python \
  -m openpi.datasets.dynamics.audit \
  data/processed/dynamics_20261122_final/libero/profiles/production_libero
```

机器可读结果见同目录的 `dataset_dynamics_production_libero_2026-11-22.json`；逐回合产物和源/实现散列留在代码项目的 `data/processed/dynamics_20261122_final/libero/profiles/production_libero/`。

后续修复已完成：[保持生产限速、显式放慢时间的全量结果](dataset_dynamics_production_retimed_2026-11-23.md)。原速产物继续作为诊断对照保留；不能将放慢后的通过结果改写为原速已通过。
