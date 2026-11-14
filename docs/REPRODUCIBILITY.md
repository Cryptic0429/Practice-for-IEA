# 主线复现与历史恢复

更新：2026-11-13。紧凑本地工作区保留3份LoRA＋1份控制头；完整底座、环境、训练图像、缓存、中间检查点和非主线权重在SJTU1。报告中的结果是历史证据，本次没有重新训练或进行GPU推理。

## 1. 本地完整性与报告

```bash
make verify
make test
make report
```

`make verify`校验四个组件的所有原始文件及R15/R21协议要求的代码SHA256；不导入torch，不访问GPU。`make test`验证恢复选择规则。编辑器和Web接口测试使用`make test-runtime`，需要`configs/mainline/requirements-cpu.txt`中的依赖。完整环境记录保留在`experiments/shared/metadata/environment-installed-packages.txt`，训练环境另见`configs/stage2-rl-installed.txt`；不要把CPU测试依赖当训练环境锁。

新报告使用Tectonic、Heiti SC和Arial；旧报告继续使用原字体和自身`build.sh`。编译使用`tmp/pdfs/`；检查版面后可用`make clean-report`删除本次新报告的编译临时目录。旧PDF和源码不删除。

## 2. 归档位置与安全恢复

清理前快照：`SJTU1:/mnt/sdc/zhc/IEA-local-archives/20261113-pre-mainline/tree/`。

`docs/archive/manifest-before.json`包含清理前每个普通文件的大小、SHA256及符号链接目标；`archive-receipt.json`记录远端逐项验证。快照不是运行目录的硬链接，后续修改运行代码不会改变快照内容。服务器原始项目`/mnt/sdc/zhc/IEA`保留不变。

默认预览，不下载：

```bash
python3 scripts/project/assets.py restore \
  --prefix models/from-functional-v15/sft-quality-coverage \
  --destination /tmp/iea-r23-evidence
```

恢复时显式加`--execute`。先下载到临时目录、核验SHA256再交付；遇到不同的已有文件会拒绝覆盖。不要把全部历史重新下载进紧凑主线目录。

按需恢复某阶段的本地证据及能从JSON解析到的引用输入：

```bash
python3 scripts/project/assets.py restore \
  --profile r15-evidence --destination /tmp/iea-r15-evidence --execute
```

可选profile：`calibrated-evidence`、`r1-evidence`、`r15-evidence`、`r21-evidence`。profile是**证据恢复集合**，包含历史代码、配置、相关数据和祖先身份；不是已经打包全部环境的可直接训练镜像。原本没有下载到Mac的完整底座、缓存或优化器状态仍需从服务器原项目取得。历史绝对路径／符号链接可能带原机器前缀；证据核验保留原字节，不自动改写冻结文件。

## 3. 推理重放

- 底座：`/mnt/sdc/zhc/IEA/models/Qwen2.5-VL-7B-Instruct`，固定revision与分片哈希见模型身份及`origin-audit.json`。
- R1：原底座＋登记R1一个适配器；首轮提示、参数解析和编辑器。
- R15：原底座＋登记R15一个适配器；R15决策协议与阴影编辑器。
- R15＋R21：精确复用R15输出／特征，再用seed20261205、e20的头，质量策略，阈值0.5。
- 校准SFT：历史JSON参数任务，用`goal_v2_infer_delivered.py`，不是直接加入后期三方案对比台。

本地主线交互代码（`scripts/iea_studio/`）已整理为以 `IEA-Core (R15 + R21)` 为核心、以 `无控制头消融版 (R15)` 和 `早期功能基线 (R1)` 为对照的三方案对比，服务器已有四候选服务保持原样，本次未重新部署。`scripts/iea_studio/open_local.sh`只连接已有远端入口；它不把本地更新自动部署到服务器。新部署应在独立工作目录中完成依赖恢复，并沿用实际预算账本，不能复制旧账本当作新授权。

原推理采用NF4双量化、BF16、最大输入262144像素、确定性生成。CPU FP32与历史GPU精度不同。原底座审计还包含文件mtime；把底座复制到其他机器后需要重新做完整SHA审计并生成独立运行身份，不能为了通过检查而伪造mtime。运行归档loader可能检查R11父权重身份；主线Web worker直接验证最终适配器，不需要叠加父权重。

## 4. 训练配方与重放边界

| 阶段 | 初始化 | 固定配方 | 原始训练入口（只列出，不在本次执行） |
|---|---|---|---|
| 校准SFT | Stage A | 3000条、lr1e-4、1024步，seed20261104，512/1024步LoRA A/B等权平均导出 | `scripts/goal_v2_train_json3k_long.py`；worker为`goal_v2_3k_replication_worker.py` |
| R1 | 校准SFT seed20261104 | 440条、4轮、lr1e-4、batch4；保留e1 | `python -m functional_v1.train --start current --gpu 0 --seed 20261104` |
| R15 | 实际R11 final | 1850条、2轮、lr5e-5、batch4；保留e1 | `python -m functional_v15.train --route language_merger --gpu 0 --seed 20261127` |
| R21 | 冻结R15 | 925语义配对、CPU、80轮、lr1e-3；保留e20 | `python -m functional_v21.train --seed 20261205` |

训练入口依赖冻结协议、准备好的batch／隐藏特征缓存、绝对路径、底座身份和已有输出目录保护，不是“下载几个权重即可训练”。服务器原始环境是历史重放基准；在独立运行目录恢复实际输入与父权重，并核对冻结协议内所有哈希后再运行。R15父权重实际位于R11 final，不能用R1替代。

历史脚本会拒绝覆盖已有run，有些旧worker写死服务器路径。因此新训练需要为新任务创建独立运行配置和预算记录，并单独记录代码改动／运行根目录；不得改写历史协议来假装是原实验。本次只验证恢复及CPU通路，没有新增训练授权，也没有证明不同硬件上的位级一致。

## 5. 报告来源与旧路径

旧完整报告及全部源文件原字节保留，预生成表格可直接编译。旧`extract_evidence.py`重提取全量统计需要先在独立目录恢复完整历史证据；紧凑工作区不承诺所有旧路径仍然存在。当前入口以`models/mainline.json`、本文件和`docs/WORKSPACE_GUIDE.md`为准。
