# 代码目录说明

## 核心 Agent 与交互工作台

- `iea_agent/`：多轮对话修图 Agent 编排层，包含会话状态与参数栈（`state.py`）、CPU 硬指标护栏（`guardrails.py`）、Planner/Critic 提示词（`prompts.py`）、PEFT 零开销双模式后端及 CPU 规则回退后端（`backend.py`）、自检重试闭环（`orchestrator.py`）与单元测试（`test_agent.py`）。
- `iea_studio/`：重构后的交互工作室（默认展示多轮对话修图工作台，支持示例图库弹窗、版本时间轴无损回溯与核心/基线对照视图；本地版本未自动部署到服务器）。
- `functional_v15/`：核心语义决策主干的冻结协议、独立阴影增益编辑器（`editor.py`）与训练/评测入口。
- `functional_v21/`：幅度与风险控制头的网络定义（`head.py`）、纹理残差特征（`texture.py`）、融合组合（`compose()`）与训练/评测入口。
- `project/assets.py`：本地完整性校验（`verify`）、布局单测（`test_assets.py`）与 `SJTU1` 归档按需恢复工具。

## 对照基线与冻结依赖

- `functional_v1/`：早期功能基线的冻结协议、编辑器与训练入口。
- `goal_v2_*`：早期校准基线的历史复现脚本（部分含原服务器绝对路径）。
- 其余 `functional_v*` 模块属于主线导入闭包或冻结协议哈希依赖。

> 冻结代码不移动、不重命名、不重写；日常验证运行 `make verify`、`make test` 与 `make test-runtime`。
