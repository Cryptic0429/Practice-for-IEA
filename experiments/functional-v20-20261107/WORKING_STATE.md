# R20交接：4检查点失败，真实试用完成，目标仍active

累计23.06369779215919/24 GPU小时，余0.9363022078408108。仅GPU0/2/3，原作者GPU1 PID3378799/12345保留；watchdog3879191在23.9保存停owned。实验worker4152225/18282与Flask4152226/12346，GPU模型已卸载。操作前核实PID和服务。总预算不扩张，继续保留至少0.25h试用余量。无subagent授权、无skill、无git提交/推送。

R19完整包已经双端907文件/155硬链验证，377174479bytes，SHA921ce3e620be9d91c4e8ce1305f411ce1d45c21c686ea9921445a48a7191257c，13依赖。R18原包不改，其报告同步旧稿问题已更正并有独立9文件更正包；详见R19 REPORT。绝不可从服务器拉取旧Markdown覆盖本地作者结论：pull排除*.md，push最终正文，封存前要求两端authoritative-documents SHA一致。

R20实际完成：冻结R15e1父97c84f...（真实存放models/from-functional-v11/sft-shadow-correction/seed-01；目录按旧起点命名，不是R11权重），原Qwen NF4+116LoRA不更新。1942条允许输入单前向各取两表示：1850TRAIN+12old+40native+40controls，缓存同时last/request_mean及16当前RGB统计，0target/label/caseID编码。全部实际前缀匹配字面请求跨度并断言图像token展开；无验收提取。特征GPU0 PID4146205已结束，.181433574h。CPU训练两个466088参数128GELU/drop.1头，4动作+9x3方向+9正幅度，双语一致性，80轮4640步固定20/80；各16.27/15.98秒。数据925语义1850双语240组P120L120原样继承，无新照片/指令/标签；154原始英文771AI。780edit80keep44quality decline21unsupported。每轮1850呈现，不重复19偏好，不排名不reward。实际source cases.json，SFT格式train-accepted仅留存核对。TRAIN-only标准化与类别权重，四checkpoint优化器/RNG/有限性/权重改变/归一化已CPU核验。所有已冻结20训练/特征/runtime/encoder/controller源码不能再改，辅助评审、页面、归档脚本不在初始表可改。

实际old6匿名看43新结果+1继承，13原尺寸结果+2原图，历史精确像素争议保留U。last-e20/e80均6/6 M0；request-mean-e20/e80均3/6 M0，mean不扩大native/controls。last两ckpt actualnative20与controls20完成。native72unique中65new联系表直接看，7继承。对070两个新结果和1输入实际看整幅960x640，均C/M（彩色噪声及色块遍布天空/前景，遮盖地面山体纹理）。其他联系表原尺寸未决保留U，因两ckpt必然失败停止进一步细节扩展，非自动pass。最终native已确认10/20与9/20、M1均，controls20/20均。不能把未决较多的10/9同过去16/20作等精度比较。英文070质量拒绝无图；拒绝正确性不能从完成数推断，未建立可行性oracle。未controlled100、replicate或accept，全部4ckpt failed_practical_gate。

两个固定末轮e80仅失败诊断试用，不选赢家/不改默认：functional-v20-last-20261203模型头SHA24184a36443d4ba0a69ae8a9bfeb108d0a9d01c8f884f3eddfe3118a0f7fa0df，request-mean SHA57c79e7f3a907103ee1c00f134d8e4a38b5ad4eccad422ba6b1f2725e545ad50。模型头位于models/from-functional-v15/continuous-controller-{last,request-mean}/seed-01/controller.pt，必须functional_v20.runtime+实际R15父+原7B，不是LoRA适配器或完整底座。

21真实HTTP+3真实browser已核验，12编辑PNG离线字节完全一致，2模型错误（mean e80 old019EN/033EN空edit）422/noimage与离线一致，明确拒绝404、keep原像素、无外部改写。mean的2控制调用仅runtime探针不是20条分数。browser原case12EN last请求4b9e5f394b194919b73f3664550da77f，mean f7502593281d4f6782e04d5e65aaea89；切回last上传070EN，2ecb4fa21bf44d88a5afeb0299e9c266，质量拒绝无图。实际下载Downloads/output (4).png已复制demo/browser-download.png并校验。3agent反馈单独training_use=false，独立用户0，不能报用户认可。

CUA本阶段已completion“验证了训练头试用”，勿重复；IAB3/tab1已markDeliverable。最终回复必须内嵌demo/browser-case12-result.jpg或拒绝图等真实网页截图。页面/v20公开真实训练标签/已评审success fail U，accept/unreviewed after404，CPU6HTTP协议与records边界通过；native表已注明未决范围，模型身份明确。当前API30模型，默认functional-current-20261104未升级。

主3文档、REPORT、dataREADME、modelsREADME、config/status/map已更新最终失败叙述与实际GPU。当前准备封存R20完整包；是否已完成以archives/functional-v20-20261107/complete-receipt.json与complete-download-receipt.json为准，切勿重复创建stage或把已封存包当可写目录。

下一步先完成本轮封存。实用目标未达，不能因训练/交付结束或预算将满而complete。若再试，只做有新依据且预算内的对照：可考虑保留R15 LM语义动作/工具方向、让已缓存特征只学连续幅度的hybrid，以避免R20均值头的语义/空编辑和R5反向编辑；这是未冻结、未实施的想法，需先检查已有TRAIN父预测可用性、定义有效监督范围、冻结方案/数据/解码后才用GPU。不能针对DEV070、case12手工设参数或加阈值，不把CPU拟合当功能改善。至少保留用户试用资源；预算不是扩大训练的理由。

封存已实际完成：两端2895文件/90硬链验证，723097059bytes，SHA bf21b83f22f8f5c0138bd3fcd24b8aae945887b313d0fe8f16c4ba98f38b4d7c，manifest 5ea55386b78a178064aea648b82aa0ac08c05d14261401482b775c50c56f4c8b，14依赖。不要重建/修改frozen archive；未来阶段另目录。该后记在封存后追加。
