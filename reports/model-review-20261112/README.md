# IEA 模型实验完整报告

交付日期：2026-11-12。正文由LaTeX编译，模型谱系和系统架构图使用TikZ矢量图，不使用生成图替代实验结果。

- [最终PDF](../../output/pdf/iea-model-review-20261112.pdf)
- [LaTeX正文](report.tex)
- [原始证据提取脚本](extract_evidence.py)
- [数据与来源哈希快照](generated/evidence_snapshot.json)
- [版面与交付检查](validation.json)
- [智能体工作约定](../../AGENTS.md)

报告涵盖35份登记LoRA、6份登记辅助头、R5/R29额外诊断说明、像素指标、双语功能指标、训练loss、候选训练方法、后期CPU实验、证据限制和复现入口。它汇总历史证据，不是新评测；没有新增训练或GPU推理。

## 重建

需要Tectonic（或相应修改为XeLaTeX）、系统中文字体Songti SC / Heiti SC，以及Times New Roman、Arial、Menlo。Linux等环境可将正文的fontspec字体替换为已安装的Noto CJK字体；字体替换后需要重新检查版面。

```bash
# 从已有冻结快照编译，默认不重新提取或改变实验结果。
bash reports/model-review-20261112/build.sh

# 仅当明确要更新报告数据快照时运行；只读历史证据。
python3 reports/model-review-20261112/extract_evidence.py
```

成品位于 `output/pdf/`，源文件与必要矢量图、表格、JSON快照保留在本目录。构建临时文件位于 `tmp/pdfs/model-review-20261112/`。必须先完成PDF渲染检查，再清理该任务目录；不要删除任何历史实验训练日志。

## 数据口径

训练loss表取本轮开始至导出检查点的逐优化器步日志均值，日志经过舍入，因此是近似值；不是最后一步loss，也不是最终模型在统一留出集上的loss。三份模型日志本次本地未取得，保持缺失。像素L1统一榜只覆盖早期7模型的独立71图；不虚构后续模型的同集分数。

功能表按原始请求20、旧回归6、受控100分别呈现；R1验收50另列。原始请求20与旧6可能反复使用；原图、翻译、合成条件和任务数不能互换。拒绝、未运行、不确定、已确认重大问题按原报告解释。
