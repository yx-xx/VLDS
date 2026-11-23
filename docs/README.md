# 文档索引

## 日常使用

- [项目首页](../README.md)：项目定位与安装说明。
- [常用命令](../COMMANDS.md)：观察台、模型服务及命令行入口。
- [完整数据集](datasets.md)：下载、续传、校验和训练共用目录。
- [LIBERO 数据处理](data_processing.md)：统一入口、时序对齐、完整命令区间与拟合候选、划分归一化及 F 接口。
- [模型观察台](simulation_inspector.md)：模型仿真验证、数据集浏览和记录状态回放。
- [项目结构与开发约定](project_structure.md)：源码、脚本、测试、数据与运行环境。

## 模型、训练与部署

- [LIBERO / Franka 环境](../examples/libero/README.md)：独立环境安装、推理与显式批量评测。
- [ALOHA 仿真](../examples/aloha_sim/README.md)：双臂环境运行方式。
- [归一化统计](norm_stats.md)：训练数据统计计算。
- [远程推理](remote_inference.md)：模型服务与客户端。
- [Docker](docker.md)：容器安装与运行。
- [贡献指南](../CONTRIBUTING.md)：上游代码风格与提交要求。

各硬件和数据适配示例的具体步骤保留在 `examples/<名称>/README.md` 中。

- [参考动力系统拟合与连续执行](field_teachers.md)：SEDS、有限选段、完整环境诊断、强基线及实际结果来源。

- [全量数据集动力系统分段与播放](dataset_dynamics.md)：离线构建、连续拟合验收、跨坐标场投影、全量结果和控制配置边界。
- [分段末端收敛与过点目标复核](validation/dataset_dynamics_passthrough_2026-11-23.md)：区分位置连续和速度衔接，以前视目标减少提前减速，保留真实停止和整回合回退；仅代表样例。
- [VLDS 结构化模型与训练实现](vlds_training.md)：结构化模型、训练损失和代码侧实现顺序。
- [LIBERO 原生产控制配置复验](validation/dataset_dynamics_production_libero_2026-11-22.md)：固定速度、角速度和姿态增益下的独立重拟合、失败原因与回放切换。

- [LIBERO 生产约束与放慢时间复核](validation/dataset_dynamics_production_retimed_2026-11-23.md)：6,500 回合的显式时间映射、连续场验收、逐步审计与真实三维显示复核。

- [长片段与非线性场复核](validation/dataset_dynamics_long_segments_2026-11-23.md)：原始观测长度、闭环拟合、失败边界细搜与夜间全量进度。
