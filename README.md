# IEA：多轮对话驱动的全局智能图像编辑系统

基于 `Qwen2.5-VL-7B-Instruct`，将自然语言对话规划、结构化参数决策与确定性数学调色引擎结合，支持多轮无损修图、硬指标护栏与视觉自检回退。当前交付定位为**研究与交互原型**，尚未通过稳定实用性验收（`production_qualified: false`）。

## 核心系统架构

系统围绕**主力智能调色核心**与外层 **Agent 编排层（`scripts/iea_agent/`）** 构建，采用不重写冻结权重、零显存增量双模式切换的设计：

```text
用户照片 + 多轮自然语言对话
   │
   ▼
[1. 对话规划层（原生视觉语言底座，关闭适配器）]
   │  • 理解上下文（“再暖一点”、“退回一半”、“撤销上一步”）
   │  • 识别超出 9 类全局调色范围的请求并友好解释
   │  • 将模糊意图改写为标准单轮修图指令
   ▼
[2. 参数与风险决策层（开启语义决策主干 + 幅度风险控制头）]
   │  • 语义主干锁定动作与调色方向
   │  • 轻量化控制头结合图像统计与纹理特征校准 9 维滑块幅度并预测风险
   ▼
[3. 无损重算渲染（editor.py）]
   │  • 始终基于初始原图 source_rgb + 累计参数 current_params 一次性渲染
   ▼
[4. 两级自检闭环（CPU 硬指标护栏 + 视觉复检 Critic）]
   │  • 检查高光死白/暗部死黑比例及视觉自然度
   │  • 若未通过则按 0.6× 衰减幅度重试（最多 2 次）；仍有风险则自动回退保护画质
```

- **核心指标与边界**：主力方案首种子（`seed 20261205 / e20`）在原始开发集达 `16/20`、受控开发集由 `75/100` 提升至 `82/100`（重大问题由 1 降为 0）；但换种子复核失败（夜景提亮仍出现严重噪点），因此目前定位为研究与交互原型。

## 核心引擎与对照模型

本地保留 **3 份 LoRA ＋ 1 份控制头**（核心引擎的两部分共享同一份主干适配器；完整 7B 底座、非主线旁支与大体积历史数据保存在 `SJTU1`）：

| 分层 | 展示名称 | 目录入口 | 定位与主要基准结果 |
|---|---|---|---|
| **核心主干** | **语义决策主干** | `models/core/semantic_backbone` | 负责结构化指令决策与阴影增益编辑器 |
| **核心控制头** | **幅度与风险控制头** | `models/core/controller_head` | 冻结主干语义后校准幅度与质量风险；首种子受控 `82/100` |
| **消融对照** | **无控制头消融版** | `models/baselines/ablation_no_controller` | 单独运行主干以对照控制头增益（原始 `16/20`、受控 `75/100`） |
| **功能基线** | **早期功能基线** | `models/baselines/early_functional_baseline` | 初代双语功能基线对照（冻结验收 `31/50`，旧基线 `15/50`） |
| **数据基线** | **早期校准基线** | `models/baselines/early_calibrated_sft` | 早期标签校准复现与初始化基准（71 图相对 Stage A 降 9.46% L1，但 NoEdit 更低） |

> 注：真实物理路径与 SHA256 以 [主线清单](models/mainline.json) 为准，不改动冻结协议中的原始路径。

## 目录架构

```text
IEA/
├── models/
│   ├── core/            核心方案入口（semantic_backbone、controller_head）
│   ├── baselines/       对照基线入口（ablation_no_controller、early_functional_baseline、early_calibrated_sft）
│   ├── from-*/          冻结物理权重目录（供 SHA256 校验）
│   └── mainline.json    核心与对照模型唯一身份清单
├── scripts/
│   ├── iea_agent/       多轮对话 Agent 编排层（SessionState、双模式后端、CPU 护栏与 Critic 自检闭环）
│   ├── iea_studio/      重构后的暗房风格交互工作室（对话修图主视图、示例图库弹窗与基线对照视图）
│   ├── project/         本地完整性校验（assets.py verify）与归档按需恢复
│   ├── functional_v15/  核心主干冻结协议、阴影编辑器与训练入口
│   ├── functional_v21/  控制头网络（head.py）、纹理特征与训练/评测入口
│   ├── functional_v1/   早期功能基线冻结协议与编辑器
│   └── goal_v2_*        早期校准基线历史复现脚本
├── configs/             主线恢复 profiles 与 CPU 测试依赖（requirements-cpu.txt）
├── data/                主线标签定义与 6 张交互示例图（大图按需从 SJTU1 恢复）
├── experiments/         核心与对照实验冻结协议、日志、指标与反证
├── reports/ & output/   新旧技术报告源码与 PDF 成品
├── docs/                复现说明、工作区导航、交接记录与 SJTU1 归档核验清单
└── tests/               不回写历史证据的 CPU 编辑器与接口契约测试
```

## 快速开始与校验

```bash
make verify        # 校验核心/对照权重、冻结协议代码、保留证据与旧报告完整性（无需 GPU）
make test          # 运行资产布局单测 + 多轮 Agent 编排单测（纯标准库，无需 GPU）
make report        # 编译主线技术报告 PDF
```

CPU 编辑器及 Web 工作台接口测试：安装 [requirements-cpu.txt](configs/mainline/requirements-cpu.txt) 后运行 `make test-runtime`。

- **核心技术文档**：[主线技术报告](output/pdf/iea-mainline-technical-20261113.pdf) · [原完整报告（保留）](output/pdf/iea-model-review-20261112.pdf)
- **工程与复现指南**：[工作区导航](docs/WORKSPACE_GUIDE.md) · [复现与恢复](docs/REPRODUCIBILITY.md) · [工作台说明](scripts/iea_studio/README.md) · [智能体约定](AGENTS.md)

## 服务器与交互边界

`SJTU1` 保留完整底座、原实验目录（`/mnt/sdc/zhc/IEA`）与清理前独立快照（`/mnt/sdc/zhc/IEA-local-archives/20261113-pre-mainline/tree`），移出文件均经 SHA256 核验，详见 [归档收据](docs/archive/archive-receipt.json)。

本地已完成 `scripts/iea_agent/` 与 `scripts/iea_studio/` 的重构，未自动部署或重启 `SJTU1` 既有服务；转发连接既有服务器入口可执行 `bash scripts/iea_studio/open_local.sh`。GPU 推理须在 `SJTU1` 独立工作目录按实时预算账本运行，不触碰 GPU 1 既有服务。
