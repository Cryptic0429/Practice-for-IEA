# 第 29 轮权重

[heads.pt](heads.pt)保存两个线性输出头及各自的训练归一化。一个来自 R22 去重子集，一个使用同一子集再加本轮方向和范围对照。

[adapter.pt](adapter.pt)保存 Qwen2.5-VL-7B 最后一层 q/v 的 rank 4、alpha 8 LoRA。四个张量共 45,056 个参数，需与服务器上固定版本的基础模型和请求提示一起加载。该文件是本项目 CPU 诊断格式。[训练凭证](receipt.json) · [完整加载和编辑路径](../runtime-verification.json) · [结论](../REPORT.md)
