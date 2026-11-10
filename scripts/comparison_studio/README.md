# IEA 四候选交互对比台

入口：http://127.0.0.1:12348/ 。部署在现有 SJTU1 实验服务器，应用仅绑定回环地址，通过 SSH 转发在本机访问。

支持上传图片、原样输入中英文要求、四候选共八项推理、逐项进度、原图/结果滑杆、双语并看、隐藏模型身份、原尺寸检查、PNG/ZIP下载和用户偏好记录。预置六张历史开发图片；这些例子不代表新独立测试。上传与反馈保存在独立目录，不自动参加训练。

## 固定候选与比较口径

- R1：`models/from-sft-calibrated/sft-functional/seed-01`，首轮默认检查点。
- R15：`models/from-functional-v11/sft-shadow-correction/seed-01`，实际第十五轮第1轮权重。
- R23：`models/from-functional-v15/sft-quality-coverage/seed-20261208`。
- R15 + R21：相同R15权重，附加 `models/from-functional-v15/semantic-preserving-controller/seed-01/controller.pt`，第20轮、质量策略。

每次统一输入的EXIF方向和RGB像素，四候选共用一份PNG。中文和英文各自原样输入，不自动翻译。R1保持首轮提示、严格参数协议和首轮编辑器；其他方案使用R15提示、决策协议和编辑器。比较对象是完整方案，不能把差异全部归因于模型权重。底座精度、图片输入上限262144像素、确定性生成和输出上限256 token保持一致；实际编辑和下载在统一原图尺寸上执行。

三个LoRA作为互斥适配器加载，共用同一个原始Qwen底座；逐张量验证加载值，一次只激活一份。R15 + R21精确复用本次R15生成的决策和冻结表示，再计算控制头。耗时分别记录主模型与追加部分，不将追加耗时说成完整推理耗时。R21保持父模型的工具、方向、保持和拒绝决定。风险拒绝、非法输出和运行错误无结果图，不用原图作为失败回退。

## 运行与恢复

服务器根目录 `/mnt/sdc/zhc/IEA`：

```bash
bash scripts/comparison_studio/launch.sh --device cuda --gpu 0 --gpu-budget-hours 2
```

本机重建转发：

```bash
bash scripts/comparison_studio/open_local.sh
```

CPU模式使用 `--device cpu`；所有候选同时改用FP32，可能与历史NF4结果不同。Flask使用服务器既有 `envs/iea`，推理子进程使用既有 `envs/iea-rl`，不需要安装前端依赖。

本次用户明确授权新增不超过2 GPU小时，用于验证和后续交互。只使用GPU0，原GPU1服务保留；GPU被占用时排队。预算记录在 `experiments/comparison-studio-20261110/resource-ledger.json`，与已结算的24小时账本分开。模型加载、推理、进程退出时间纳入计账；完成一批最多6项请求后退出子进程释放GPU。父进程计时和子进程独立截止定时器共同限制预算，启动预留5秒退出余量。重启读取已有账本，不重置预算。

任务位于 `experiments/comparison-studio-20261110/jobs/<id>/`，保存统一原图、八项状态/结果、参数、精度、权重身份和反馈。API反馈追加审计历史，ZIP导出当前记录和已有图片。服务和推理日志保留；开发期间的加载校验失败也保留。`?qa=1` 仅用于浏览器联调，将新请求和反馈标记为agent测试；正常用户入口不加此参数。

## 验证

```bash
PYTHONPATH=scripts envs/iea/bin/python -m unittest comparison_studio.test_app -v
```

接口测试验证四模型双语绑定、无结果文件时404、上传/反馈隔离、非法投票、跨站提交拒绝和ZIP内容。真实GPU验证、逐图结果、浏览器交互及下载校验记录见实验目录。
