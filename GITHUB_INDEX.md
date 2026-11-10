# IEA GitHub 内容索引

本次更新日期：2026-11-10。仓库保存实验代码、文本数据清单、模型身份、逐轮结果和可复核证据，功能实验记录已补齐至 R33；新增 R1、R15、R23、R15 + R21 四候选交互对比台。

**当前结论：原 24 GPU 小时实验已结算，仍未达到稳定实用标准。** 新对比台用于个人交互体验，另设 2 GPU 小时验证及试用预算，不代表模型已升级或通过最终验收。

| 需要查看的内容 | 入口 |
|---|---|
| 项目总览与完整历史 | [README](README.md)、[实验记录与结论](实验记录与结论.md)、[原理与复用指南](原理与复用指南.md) |
| 当前进展与目录范围 | [进度总结](docs/PROJECT_STATUS.md)、[目录与恢复说明](docs/REPOSITORY_MAP.md) |
| 四模型交互体验与连接恢复 | [使用说明](scripts/comparison_studio/README.md)、[源码](scripts/comparison_studio/)、[验证结果](docs/COMPARISON_STUDIO.md) |
| R1 默认候选 | [首轮报告](experiments/functional-v1-20261104/REPORT.md)、[首轮代码](scripts/functional_v1/) |
| R15 阴影修正候选 | [第十五轮报告](experiments/functional-v15-20261106/REPORT.md)、[代码](scripts/functional_v15/) |
| R15 + R21 辅助控制头 | [第二十一轮报告](experiments/functional-v21-20261107/REPORT.md)、[代码](scripts/functional_v21/) |
| R23 直接 LoRA 候选 | [第二十三轮报告](experiments/functional-v23-20261107/REPORT.md)、[代码](scripts/functional_v23/) |
| 预算结算与后续 CPU 诊断 | [训练预算结算](experiments/functional-v23-20261107/FINAL_SUMMARY.md)、[第三十三轮报告](experiments/functional-v33-20261107/REPORT.md) |
| 最终交付与尚存差距 | [交付总结](experiments/functional-final-delivery-20261107/SUMMARY.md)、[归档说明](archives/functional-final-delivery-20261107/) |
| 训练数据与准入证据 | [各轮文本数据](data/training/)、[后期实际训练池](data/training/functional-v22-20261107/README.md) |
| 模型配置、身份与权重清单 | [模型说明](models/README.md)、[注册表](models/catalog.json)、[归档与权重信息](docs/ARTIFACTS.json) |

## 打开交互对比台

在已配置现有实验服务器 SSH 访问权限的本机，从项目根目录运行：

```bash
bash scripts/comparison_studio/open_local.sh
```

然后访问 <http://127.0.0.1:12348/>。这会建立到现有实验服务器的 SSH 转发；网页不是 GitHub 上的公网部署。服务器启动和依赖恢复见[运行说明](scripts/comparison_studio/README.md)。

六张历史开发图的 48 项固定比较记录为：42 项生成结果、4 项拒绝、2 项执行失败。生成结果数不是画质通过率；提示和编辑器存在历史差异，这次比较面向完整方案。详细口径见[验证说明](docs/COMPARISON_STUDIO.md)。

## 仓库范围

GitHub 不包含模型权重、原图、编辑结果图、虚拟环境、完整压缩归档以及个人上传和反馈。报告中的本地图片链接需恢复实验文件后才能查看；仅克隆此仓库不足以直接运行模型。

`experiments/comparison-studio-20261110/` 是被 Git 忽略的运行目录；公开内容仅为源码与筛选后的汇总验证信息。历次报告和账本保留对应时点的结果，`docs/REPOSITORY_SNAPSHOT.json` 是 2026-11-05 的初始快照，不能当作本次更新清单。
