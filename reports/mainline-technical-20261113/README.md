# IEA 主线技术报告

成品：`../../output/pdf/iea-mainline-technical-20261113.pdf`。旧版`../model-review-20261112/`及其PDF保留原字节。

```bash
bash reports/mainline-technical-20261113/build.sh
```

编译器为Tectonic（XeTeX），中文全文Heiti SC黑体，拉丁字母Arial；所有章节连续排版，无手工`\clearpage`／`\newpage`。公式用正文符号表达，避免数学字体混入宋体。两张架构图为TikZ矢量源，不需外部图片。

字体来自当前macOS；在Linux重建需安装许可允许的相同字体，或明确修改字体设置并重新做版面检查。正文、图源和历史证据快照保留；编译和逐页PNG位于`tmp/pdfs/mainline-technical-20261113`，检查后清理。

本报告不包含简历文本，不新增训练或推理数据。`generated/evidence.json`列出主线保留证据的哈希。远端归档及本地精简结果见`../../docs/archive/`；最终PDF验证见`validation.json`。
