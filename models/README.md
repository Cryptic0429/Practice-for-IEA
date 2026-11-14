# 模型资产说明

本地保留 **3 份 LoRA ＋ 1 份控制头**，权重真实路径与 SHA256 以 [mainline.json](mainline.json)（与 [catalog.json](catalog.json) 同步）为准。

为兼顾**冻结路径哈希校验**与**直观语义命名**，物理权重继续保存在 `from-*` 原始目录中，同时通过 `core/` 与 `baselines/` 提供清晰入口：

| 分层 | 展示名称 | 语义化目录 | 真实物理路径 | 角色与说明 |
|---|---|---|---|---|
| **核心主干** | **语义决策主干** | `models/core/semantic_backbone` | `from-functional-v11/sft-shadow-correction/seed-01` | 负责结构化调色决策与阴影增益编辑器，与控制头共享同一适配器 |
| **核心控制头** | **幅度与风险控制头** | `models/core/controller_head` | `from-functional-v15/semantic-preserving-controller/seed-01` | 冻结主干语义后校准 9 维参数幅度并预测画质风险 |
| **消融对照** | **无控制头消融版** | `models/baselines/ablation_no_controller` | `from-functional-v11/sft-shadow-correction/seed-01` | 单独运行语义主干（不挂载控制头），用于对照验证控制头增益 |
| **功能基线** | **早期功能基线** | `models/baselines/early_functional_baseline` | `from-sft-calibrated/sft-functional/seed-01` | 初代双语功能基线对照 |
| **数据基线** | **早期校准基线** | `models/baselines/early_calibrated_sft` | `from-sft-standard/sft-calibrated/seed-02` | 早期 JSON 标签校准复现与初始化基准 |

- `current` 保留为早期校准基线的历史兼容软链接，不代表当前核心默认。
- 完整 `Qwen2.5-VL-7B-Instruct` 底座及非主线旁支留在 `SJTU1`（归档快照：`/mnt/sdc/zhc/IEA-local-archives/20261113-pre-mainline/tree/models/`），恢复见 [复现说明](../docs/REPRODUCIBILITY.md)。
- 分层命名属于研究维护选择，不代表生产准入：主力方案换种子复核失败，`production_qualified: false`。
