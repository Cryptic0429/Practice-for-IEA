# 目录梳理与GitHub范围

本次沿用已有分组，没有移动、删除或覆盖原训练产物。以目录用途和证据状态区分内容；失败实验仍是判断下一步方法的重要证据。

| 目录 | 有效内容 | GitHub处理 |
|---|---|---|
| `scripts/functional_v1/` | 共用输入、解析器、编辑器、预算账本、推理工作进程和Flask页面 | 纳入代码；这是当前试用入口的依赖核心 |
| `scripts/functional_v2/` 至 `functional_v6/` | 各轮独立数据准备、训练、评测、标签及视觉核验 | 纳入；保持冻结文件不变以便复核 |
| `scripts/functional_v7/` | 后续编辑器、降噪可行性诊断 | 纳入本次快照，不作为已通过的生产流程 |
| `scripts/` 其他脚本 | 历史Stage 1/2、校准、交付与目录整理 | 保留历史代码，主入口优先看functional模块 |
| `configs/` | 超参数、环境依赖及模板 | 纳入文本配置；凭据通过环境变量提供 |
| `data/training/functional-*/` | 实际训练JSON、分组案例、准入与冻结记录 | 纳入JSON；照片、目标图不上传 |
| `experiments/functional-*/` | 协议、选模/未升级决定、逐条参数、AI评审、报告 | 纳入文本；图片、特征缓存、权重、日志及用户上传排除 |
| `models/` | 模型身份、父权重、数据哈希、adapter配置 | 纳入元数据；权重本地/服务器保存，见ARTIFACTS.json |
| `archives/` | 完整历史交付包 | 仅纳入功能实验收据，压缩包继续保留本地/服务器 |
| `docs/`与三份主文档 | 总览、原理、进度和来源 | 纳入Markdown与整理清单；论文PDF保留本地 |
| `artifacts/`、`metadata/`、`.downloads` | 旧路径兼容链接 | 不纳入；不是另一套实验，也不是更多训练数据 |
| `.venv-report/`、环境及缓存 | 运行依赖和本机缓存 | 不纳入 |

## 克隆后的使用边界

这是“代码＋文本证据”的仓库，不是包含全部图片和7B权重的一键离线包。报告中的图片路径、数据JSON中的绝对路径、旧兼容路径需要在具有原始资产的环境恢复。不要用缺失图片的文本评分代替视觉核验。

现有完整工作区：Mac `/Users/cryptic/Desktop/Project/IEA`；服务器 `SJTU1:/mnt/sdc/zhc/IEA`。服务器保存底座、完整运行检查点和环境。本地含已下载adapter、图像子集和已核验归档；没有将未下载检查点说成已经在本地。

恢复流程：

1. 按 `docs/ARTIFACTS.json` 和 `archives/functional-*/receipt.json` 确认所需权重/归档及哈希，从已有工作区取得这些资产；GitHub没有承诺提供权重下载。
2. 恢复固定Qwen2.5-VL-7B-Instruct底座及对应adapter。底座revision为 `cc594898137f460bfe9f0759e9844b3ce807cfb5`。父adapter仅说明训练来源，加载时只加最终一份adapter。
3. 恢复图像、目标和所需模型最终运行目录。注册表会检查文件是否实际存在；不能只复制metadata后宣称候选可用。历史路径映射见原目录整理记录。
4. 使用原环境依赖记录和复用指南；`envs/iea-rl`承担模型推理/训练，`envs/iea`承担Flask。指定 `IEA_ROOT` 和实际GPU设备，先确认现有服务和预算，避免重复启动训练。
5. 在项目根用 `PYTHONPATH=scripts` 启动对应模块。试用页面是 `functional_v1.web`，模型进程是 `functional_v1.worker`，端口分别12346和18282。离线与网页采用同一版本输入、解析与编辑器。

训练数据中的路径和冻结协议保持原字节，不能为让仓库“看起来可运行”静默改写标签或哈希。迁移到新环境时，应记录路径映射和新的运行环境。

## 可见性与数据来源

本次目标是用户指定的公开仓库 `Cryptic0429/Practice-for-IEA`。原图、参考图、模型权重、个人试用上传和反馈没有上传。训练/评审文本保留来源及AI构造标记，不把第三方GIER和Qwen资产重新声明为本项目所有；原资产的使用和再分发应遵循其各自条款。来源位置见 `experiments/shared/metadata/data-sources.json`。

`.gitignore`采用明确范围：保留上述代码及文本记录，阻止大文件、虚拟环境、兼容链接和请求目录误入版本管理。`docs/REPOSITORY_SNAPSHOT.json`记录首次提交的文件和SHA，便于检查本次实际上传了什么；不包含用户上传/反馈。
