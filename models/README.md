# IEA 模型名称与训练谱系

**24 GPU 小时实验已结算，未达到稳定实用标准。** 实际占用23.99923538698091 GPU小时（24.00，退出余量2.75秒）。第22轮原始开发15/20、13/20；第23轮两组原始均16/20、旧回归3/6及4/6，只一组通过第一阶段，整条路线未通过，最终验收未开封，默认未升级。实际训练池979图像条件／1958中英／240原图组（人像、风景各120）；第23轮每种子实际1000条，不等于整池遍历。AI评审和试用联调已记录，独立用户反馈0。

[逐项交付总结](../experiments/functional-v23-20261107/FINAL_SUMMARY.md) · [第22轮图和结果](../experiments/functional-v22-20261107/REPORT.md) · [第23轮图和结果](../experiments/functional-v23-20261107/REPORT.md) · [逐例结果页面](http://127.0.0.1:12346/v23?kind=evaluation) · [实际训练池](../data/training/functional-v22-20261107/README.md)。新实验推理按预算暂停，已保存结果仍可看和下载；[原作者 IEA](http://127.0.0.1:12345/)保留独立运行。

| 最新诊断模型 | 实际起点 | 权重与身份 |
|---|---|---|
| 第22轮两组幅度／质量小头 | 冻结实际R15e1，CPU头各1160步 | [小头和登记](from-functional-v15/quality-coverage-controller/) |
| 第23轮两组直接LoRA | 从实际R15e1继续训练，各250步 | [LoRA和登记](from-functional-v15/sft-quality-coverage/) |
| 第29轮方向和范围输出头及末层 LoRA | 冻结 Qwen2.5-VL-7B，CPU 监督输出头及末层 q/v 125 步 | [诊断权重与运行身份](../experiments/functional-v29-20261107/run/README.md) |

原默认 `functional-current-20261104` 保留。模型目录名中的早期起点不改变其实际父权重身份，详情看各 `metadata.json`。

第二十一轮真实训练、AI评审和试用联调已完成，**锁定的换种子复核失败，默认未升级**。首种子质量策略原始16/20、旧5/6、受控82/100且重大0；换种子后夜景提亮又出现重大噪点，因此不能宣布模型实用。925语义／1850中英／240原训练原图组（人像／风景各120），本轮新增照片／请求／正标签0；两个独立CPU头各80轮4640步，首种子两个策略共用权重。最终验收未开封。[数据、实际图片、失败原因与权重](../experiments/functional-v21-20261107/REPORT.md) · [诊断试用](http://127.0.0.1:12346/?model=functional-v21-quality-20261205) · [逐例数据及结果](http://127.0.0.1:12346/v21)。31次实际HTTP／网页调用、4条agent反馈，独立用户反馈0。累计23.265153517/24GPU小时，余0.734846483；实验模型已卸载，GPU1原服务保留。

第十五轮诊断权重位于 `from-functional-v11/sft-shadow-correction/seed-01`，第1轮463步、种子20261127，SHA256 `97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90`。实际训练925语义／1850双语记录、240原图组；原始开发16/20、旧回归4/6、受控75/100且1例重大问题，未晋升默认。第二轮原始开发16/20但旧回归3/6，完整检查点和优化器保留在实验目录。运行时使用原底座加一个116张量LoRA、本轮独立阴影编辑器；详见 [完整报告](../experiments/functional-v15-20261106/REPORT.md) 和 [实际试用](http://127.0.0.1:12346/?model=functional-v15-shadow-correction-20261127)。第十六／十七轮仅做诊断，没有训练或模型。

第十四轮已完成，视觉末段微调仍未达到实用性门槛：[报告与实际图片](experiments/functional-v14-20261106/REPORT.md)。完全复用925语义／1850中英文行、240原图组（人像／风景各120），新增照片和标签0；同一R11父权重新增40视觉LoRA张量，合计156，2轮926步。第1轮原始开发16/20、第2轮15/20，旧回归均4/6、保持／不支持均20/20；第1轮受控73/100且1例重大问题，与第十三轮原范围控制相同，低于父模型77/100。未升级默认、未复核、最终验收关闭。[诊断试用](http://127.0.0.1:12346/?model=functional-v14-visual-late4-20261127) · [实际数据及逐例结果](http://127.0.0.1:12346/v14)。11次真实HTTP与2次浏览器请求通过运行核验，2条反馈均为agent联调、独立用户0。本轮1.116173078GPU小时，含独立精度核查的累计18.565750220/24；GPU1原服务保留。训练完成、AI评审和用户认为可用分别记录，不能称为已得到实用模型。 适配器位于 `models/from-functional-v11/sft-visual-late4/seed-01`，SHA `04804b0ff29e3d9f23f32cddc395dd833c6447b4579f31735a20482a47c7fa47`，第1轮仅供失败诊断，完整底座在服务器。

第十三轮已完成，未达到实用性门槛：[报告与实际图片](experiments/functional-v13-20261106/REPORT.md)。实际训练925语义案例／1850中英文行、240原图组（人像／风景各120），新增原训练池80组／253核验案例；154原始英文、771AI请求，73对完整同图相反要求。两条微调范围第1轮均16/20原始开发，但受控均73/100，父模型77/100；原范围重大问题1例，扩展范围0例。第2轮均15/20，未升级默认、未启动复核、最终验收仍关闭。[诊断试用](http://127.0.0.1:12346/?model=functional-v13-real-coverage-wide-20261127) · [实际数据及成功／失败／不确定结果](http://127.0.0.1:12346/v13)。22次真实HTTP与2次浏览器请求已核验，3条反馈全为agent联调，独立用户反馈0。累计17.437052395/24 GPU小时，后续占用继续累计；GPU1原服务保留。不能称为已得到实用模型。 两个适配器位于 `models/from-functional-v11/sft-real-coverage/{language_merger,all_language_merger}/seed-01`；元数据登记父权重、数据哈希、实际功能结果及未通过状态。

第十二轮诊断模型：`from-functional-v11/sft-manual-calibration/seed-01`，父模型为第十一轮最终116张量LoRA，250语义／500双语、60原图组30/30，本阶段无新原图、父累计160。固定1轮125步seed20261125，NF4/BF16、r8a16、lr2e-5；同第十一轮编辑器，新一般工具说明。适配器SHA256 `6342f7a2af74a31bf84331426ada1c4a11109ab50b88f8af8bcb7bd3e2b0f7ab`。新说明未训练对照13/20，本模型14/20，上一轮15/20；均未过本轮16/20门槛，默认不变、验收未开。仅供诊断试用，拒绝时无结果图，不能称为实用性升级。详见[本轮报告](../experiments/functional-v12-20261106/REPORT.md)。

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


外部工具探查：`models/external/SCUNet`是作者发布的冻结降噪权重，不是本项目训练的指令模型，未注册到试用页。4张原训练池照片的细节取舍与未准入结果见`experiments/functional-v7-20261105/REPORT.md`。不得把外部预训练工具收益算成本项目LoRA收益。


第七轮两条第4轮诊断权重（均未通过）位于 `from-functional-v6/sft-outcome-language/seed-01` 和 `from-functional-v6/sft-outcome-language-merger/seed-01`。二者同用924条中英文记录，从同一第六轮继续训练末轮初始化；后者额外训练视觉连接层。必须使用 `functional_v7.runtime` 和 `functional-outcome-v1`，不能按旧参数JSON或旧编辑器推理。训练谱系、数据哈希和逐图结果见各目录 metadata；默认仍是第一轮功能候选。


第九轮（20261106）新增两份训练终点诊断，各10,108,960字节，均未通过功能门槛：`from-base/sft-fresh-outcome/seed-01`（原始Qwen完整权重起点）与 `from-author-iea/sft-fresh-outcome/seed-01`（作者IEA完整权重起点）。两者使用同一份462语义案例／924条中英文记录，160原图组、人像风景各80；同种子20261115、4轮924步。实际本轮数据来源为 `data/training/functional-v7-20261105/train-accepted.json`，没有另造新标签。末轮分别13/20与12/20合格，均有2例重大副作用；不升级默认、未开封验收。每个目录的metadata.json记录完整权重身份、数据哈希、代码与逐图结果。**必须按origin加载原始或作者完整权重再加唯一LoRA，禁止将作者起点适配器挂到原始底座。** 作者存在历史来源重叠。详见 `experiments/functional-v9-20261106/REPORT.md`。


## 第十轮照片条件纠错（未通过，仅诊断）

`from-functional-v9/sft-condition-language/seed-01`与`from-functional-v9/sft-condition-language-merger/seed-01`为固定第2轮终点，分别13/20、14/20原始请求双语合格，旧回归均4/6，未升级默认。240语义／480双语、60组各30；父检查点已训练160组。27原始英文／213AI，人工合成条件与用户反馈分开。每份metadata给出实际父SHA、数据SHA、完整底座身份、单适配器载入方式及功能短板。两者均只叠加在原始Qwen完整底座上，不能再附加父LoRA。


## 第十一轮负曝光与指令纠错（未通过，仅诊断）

模型：[from-functional-v10/sft-negative-gain-correction/seed-01](from-functional-v10/sft-negative-gain-correction/seed-01)。SHA `12243ea5f08ab9e2e75b082fd6e6a1e6e8b1efbd7eb4e102be3563ba084c4a56`。原始Qwen完整底座＋一个116张量LoRA，全部父张量（包括4个视觉连接张量）精确继承第十轮merger终点`772ead8e…ebbd`；没有叠加。672语义／1344双语、160原图组各80，97原始英文／575AI；没有新增独立原图或用户照片。

lr2e-5、有效batch4、1轮336步、种子20261123，最终一轮预先固定，不按损失挑选。新负曝光编辑器允许亮部一起降低，其他工具与提示不变。模型15/20原始请求、旧回归4/6、控制20/20；旧父参数只换编辑器也15/20，所以不能把新增合格归因于新训练。仍有非法负sepia、复合要求副作用及力度不足；历史不确定性不升级。默认未改，验收封存，复核未启动。身份／数据／来源详见metadata.json及[本轮报告](../experiments/functional-v11-20261106/REPORT.md)。


第十八轮两路固定末轮仅供诊断：`from-functional-v15/sft-counterfactual-ranking/seed-01`和`from-functional-v15/sft-repeated-positive-control/seed-01`。均继承R15第一轮一个适配器，116张量、2轮1002步；925语义／1850双语／240原图组，另19偏好语义（38双语），没有新照片或正例标签。原始开发16/20、旧回归3/6、控制20/20，四个检查点均失败，固定末轮不是选模赢家。推理使用R18身份包装和原样R15运行时，提示／解析／编辑器与离线一致；默认不变，最终验收关闭。实际SHA及父模型、数据、协议见各metadata.json。


第十九轮：`from-functional-v15/sft-photometric-bridge/seed-01`是固定末轮失败诊断模型；一个120张量PEFT适配器含原116LoRA与4个输入RGB分支张量，必须用`functional_v19.runtime`加载，SHA `9be16f1e8e42ca54dd356c59052ea487c291878b2f128925eb8227c40a05b78d`。925语义／1850双语／240组；16/20原始、4/6旧回归、20/20控制，但新增重大噪声1例，未晋升、未受控／复核／验收。原始7B底座另需保留，不把适配器当完整模型。

第二十轮：`from-functional-v15/continuous-controller-last/seed-01/controller.pt`与`continuous-controller-request-mean/seed-01/controller.pt`分别为466088参数的独立CPU预测头，需冻结实际R15父适配器97c84f...与原7B；不是LoRA单适配器，也不是完整底座。两个固定e80均未通过，仅诊断试用，必须functional_v20.runtime；最后位置头SHA24184a36443d4ba0a69ae8a9bfeb108d0a9d01c8f884f3eddfe3118a0f7fa0df，请求均值SHA57c79e7f3a907103ee1c00f134d8e4a38b5ad4eccad422ba6b1f2725e545ad50。
