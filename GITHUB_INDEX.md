# IEA 功能训练：当前可用部分与进度

这份仓库保存代码、实际训练清单、模型身份和可核对的实验记录。**截至2026-11-05，已跑通训练、逐图评审与本地试用；尚未得到经过验证的高实用性模型。** 后续实验仍在进行，本次提交是一个明确时间点的版本快照。

先读 [进度总结](docs/PROJECT_STATUS.md)，再看 [目录与有效内容](docs/REPOSITORY_MAP.md)。三份完整主文档是 [README](README.md)、[实验记录与结论](实验记录与结论.md)、[原理与复用指南](原理与复用指南.md)。

| 最常用内容 | 入口 |
|---|---|
| 第六轮实际训练数据 | [920条记录](data/training/functional-v6-20261105/train-accepted.json)、[460个语义案例](data/training/functional-v6-20261105/cases.json)、[准入记录](data/training/functional-v6-20261105/admission.json) |
| 第六轮结果和失败原因 | [报告](experiments/functional-v6-20261105/REPORT.md)、[未升级决定](experiments/functional-v6-20261105/development-selection.json) |
| 当前试用服务 | [Flask页面与API](scripts/functional_v1/web.py)、[本地模型工作进程](scripts/functional_v1/worker.py)、[模型注册表](scripts/functional_v1/catalog.py) |
| 固定模型输入与编辑器 | [runtime](scripts/functional_v1/runtime.py)、[提示与解析](scripts/functional_v1/common.py)、[v1.2编辑器](scripts/functional_v1/editor.py) |
| 训练、评测和直接视觉评审 | [第六轮代码](scripts/functional_v6)、[协议](experiments/functional-v6-20261105/training-protocol.json) |
| 权重和大文件在哪里 | [本地权重哈希及归档清单](docs/ARTIFACTS.json)、[恢复说明](docs/REPOSITORY_MAP.md) |

实际试用入口是服务器的 `127.0.0.1:12346`，不是本仓库提供的公网网站。有服务器权限时，运行 `ssh -N -L 127.0.0.1:12346:127.0.0.1:12346 SJTU1`，浏览器访问 `http://127.0.0.1:12346/`。模型下拉框中的第六轮两个候选已标明“未通过”；默认仍是第一轮候选。

GitHub中不含原图、目标图片、权重、虚拟环境、完整压缩归档、上传及反馈记录。报告中的原图链接是本地证据路径，需要恢复对应产物后查看。**仅克隆此仓库不能直接启动7B模型，也不能凭文本评审代替实际看图。**
