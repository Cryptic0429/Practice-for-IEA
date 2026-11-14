# 工作区导航

更新：2026-11-14。工作区已围绕**多轮对话修图 Agent**与**核心调色引擎**完成整理，并保留消融版与历史基线用于对照验证。

| 任务 | 当前入口 |
|---|---|
| 了解项目总体架构 | [README.md](../README.md)、[主线技术报告](../output/pdf/iea-mainline-technical-20261113.pdf) |
| 查看或扩展 Agent 编排层 | `scripts/iea_agent/`（状态管理、双模式后端、硬指标护栏与 Critic 自检闭环） |
| 运行重构后的 Web 工作台 | `scripts/iea_studio/`（对话修图主视图、示例图库弹窗、基线对照视图；未自动部署到服务器） |
| 访问核心与对照模型 | `models/core/`、`models/baselines/`；身份见 [mainline.json](../models/mainline.json)、[模型说明](../models/README.md) |
| 查看底层冻结引擎源码 | `scripts/functional_v15/`（语义主干与阴影编辑器）、`scripts/functional_v21/`（幅度与风险控制头） |
| 校验本地完整性与单测 | `make verify`、`make test`；不使用 GPU |
| 查完整历史实验 | [原报告（保留）](../output/pdf/iea-model-review-20261112.pdf)、SJTU1 清理前快照 |
| 恢复旁支／完整证据 | [复现说明](REPRODUCIBILITY.md)、[归档收据](archive/archive-receipt.json) |
| 智能体接手规范 | [AGENTS.md](../AGENTS.md)、[模板](AGENT_HANDOFF_TEMPLATE.md)、[交接记录](handoffs/mainline-20261113.md) |

## 目录组织原则

1. **物理冻结，增量扩展**：`models/from-*`、`scripts/functional_v*`、`experiments/` 与 `data/training/` 中的冻结路径保持原位与原字节不变，确保 `make verify` 全部通过；多轮对话、状态栈与自检闭环均在独立的 `scripts/iea_agent/` 中实现。
2. **紧凑本地包**：完整 7B 底座、大图缓存与非主线权重留在 `SJTU1`，需要复现全量训练时按 [复现说明](REPRODUCIBILITY.md) 恢复到独立目录。
