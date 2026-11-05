# IEA 模型名称与训练谱系

第五轮未晋级的连续参数头保存在 `experiments/functional-v5-20261105/runs/`，其父图文模型冻结。最好4/6但有严重反向编辑，未注册为试用候选；实际训练数据、结构与全部失败结果见第五轮报告。

第四轮诊断权重：`from-functional-v2/sft-input-plain/seed-01` 和 `from-functional-v2/sft-input-measured/seed-01`，均为第1轮，未通过升级。测量模型必须使用 `functional_v4.runtime` 的measured路线，普通对照使用plain路线，不能混用输入。详见各目录metadata及第四轮报告。

第三轮另登记 `from-functional-v2/dpo-correction/seed-01` 和 `from-functional-v2/sft-correction/seed-01`，均取第1轮供诊断试用，从第二轮课程e3初始化。两路线仍有开发严重过曝，原始请求均3/6，没有升级默认模型；权重身份、实际数据和失败决定见目录内metadata及第三轮报告。


第二轮新增 `from-functional-v1/sft-curriculum/seed-01`（第3轮）与 `from-functional-v1/sft-control/seed-01`（第2轮）。两者从第一轮 `from-sft-calibrated/sft-functional/seed-01` 初始化，目录内 `metadata.json` 记录实际数据、父权重与开发结果。加载原Qwen底座加当前一个adapter，不叠加父adapter。新课程有新增严重过曝，仅供实验试用，未替换默认模型；第二轮验收未开封。


`from-base/` 表示从原始 Qwen 开始训练；`from-sft-standard/` 表示从标准监督训练模型（原 A）继续训练。

| 历史标识 | 新名称与目录 | 训练起点 |
|---|---|---|
| A | [标准监督训练](from-base/sft-standard) | 基础 Qwen |
| B | [重采样监督训练](from-base/sft-resampled) | 基础 Qwen |
| SFT | [继续监督训练](from-sft-standard/sft-continued) | 标准监督训练（原 A） |
| GRPO | [L1 奖励强化学习](from-sft-standard/grpo-l1) | 标准监督训练（原 A） |
| Seed20261103 | [校准监督训练·复现1](from-sft-standard/sft-calibrated/seed-01) | 标准监督训练（原 A） |
| Seed20261104 | [校准监督训练·复现2](from-sft-standard/sft-calibrated/seed-02) | 标准监督训练（原 A） |
| Seed20261105 | [校准监督训练·复现3](from-sft-standard/sft-calibrated/seed-03) | 标准监督训练（原 A） |

当前选定入口是 [current](current)，指向校准监督训练·复现2。
每个模型的 `model-info.json` 记录父模型名称、父权重身份、数据、实际随机种子和导出方式；完整机器清单见 [catalog.json](catalog.json)。三个 `seed-01/02/03` 是复现实验编号，实际种子分别为 20261103、20261104、20261105。

历史实验、冻结模型锁及匿名图对应表继续使用 A/B/SFT/GRPO/Seed 标识。`legacy/` 集中保存旧名称入口，旧 `stage1/`、`stage2/`、`quality-v2-20261103/` 路径仍可访问同一模型。

完整基础模型 Qwen2.5-VL-7B-Instruct 和作者模型 Official IEA 保留在服务器原目录；本地仅保存自训练 adapter。Official runtime 为兼容加载目录。

## 功能优先实验，2026-11-04

新功能试用入口默认使用 `functional-current-20261104`；历史 `current` 链接与原作者服务保留。全部为实验模型，未达到高实用性目标。

| 模型 | 目录 | 父模型 | 开发所选检查点 |
|---|---|---|---|
| 基础起点，20261104 | [from-base/sft-functional/seed-01](from-base/sft-functional/seed-01) | 原始 Qwen | 第3轮／330步 |
| 继续训练，20261104，试用默认 | [from-sft-calibrated/sft-functional/seed-01](from-sft-calibrated/sft-functional/seed-01) | 校准监督训练·复现2 | 第1轮／110步 |
| 继续训练，20261105，复核 | [from-sft-calibrated/sft-functional/seed-02](from-sft-calibrated/sft-functional/seed-02) | 同上 | 固定同为第1轮 |

三份 adapter 均直接加载于固定原始 Qwen 底座，父 adapter 仅说明训练谱系，不能再次叠加。实际数据 440 条、来源与哈希、功能结果和限制见[本轮报告](../experiments/functional-v1-20261104/REPORT.md)。


第六轮 `functional-v6-20261105` 训练和原始开发逐图评审已完成，未过实用性门槛。基础起点末轮 `models/from-base/sft-native-coverage/seed-01`、继续训练末轮 `models/from-functional-v2/sft-native-coverage/seed-01` 已登记供诊断试用；它们分别12/20、15/20，均有3例严重副作用，旧回归均3/6。各含metadata、真实父模型、数据哈希与未通过状态；不是选出的合格冠军，默认未改，验收未开封。实际数据160原图组／460语义／920条不同中英文记录；全部8个检查点保留在服务器对应runs目录。
