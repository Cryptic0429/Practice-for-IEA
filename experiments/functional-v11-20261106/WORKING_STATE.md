# 当前工作状态

截至2026-11-06，第十轮训练、逐图评审、试用和两端归档验证均完成，失败未升级。13/20与14/20、旧4/6、0标记重大但历史U保留，controls20/20。24HTTP+2浏览器请求，3agent反馈0独立用户。R10包SHA2de9ad0e438997f206acb8270ddd94ed41d8233c05566b61f5366b5f9a996f27，1,165,062,687字节，1338文件，两端全部核验；不含R11。3主文档已维护并上传。CUA本轮已发送一次“验证了本轮试用”完成摘要，勿重做旧阶段；tab1仍为已标记交付的/v10模型结果页。

R11仅CPU候选阶段，没有任何新训练、模型推理、选模或验收开封。预算14.330370567109844/16GPUh，余1.669629432890156，GPU1原服务PID3378799保留。试用worker/web最新PIDs4027009/4027010，restart receipt在EXP10/demo/records-final-service-restart.json，worker已卸载。只准GPU0/2/3；watchdog3879191继续。

发现R7负曝光有界曲线固定白端为1，导致亮部面积大的强压暗不充分。CPU4训练源照片(05/07/12/15)对比旧、半速负V乘法、全速乘法；4张半速新图已实际原尺寸查看。半速候选保留正曝光旧曲线，负E倍率2^(value/100)，负B2^(value/70)，中灰零附近响应匹配旧曲线，白端可下降；不是物理RAW曝光。半速拟用于更广标签核验，还未采用到服务。旧4个强压暗同像素曾AI判C/A，新对比提出全图完成度疑问；保留历史与新疑问，未来不能把旧C/A直接搬过来。EXP11/negative-exposure-probe/direct-review.json记录。

scripts11/editor.py为独立半速候选，outcomes.py复制R7严格协议但editor身份改v11、调用新editor。原system_prompt字节不改，正曲线与其他工具不改；旧7/9/10文件均保持冻结。

prepare_labels.py正执行(最新exec session43012)：合并R7全部462+R10新增200(不重复40锚点)=662，再加10个AI棕黄暖偏色纠正，暂拟672/1344，160组各80；548edit60keep64decline、97原始英文/575AI均为拟议数、未准入。负E/B改变的11旧标签+40强压暗=51新执行图，加10cool-cast新目标，共61待逐图看。其他611重放完全同像素或同无输出；其中44个quality-decline不能仅靠同null重放沿用，必须在新工具下重核可行性（下段）。只有547编辑/keep像素相同+20unsupported可明确继承，共567；未解决历史U仍U。

新增cool-cast-01..10从R10已核验原训练照片索引1,2,5,7,9(P)及11,12,13,17,20(L)。先round(.84*RGB+20)预留通道范围，再加temperature24，标签temp-24，实际执行往返≤1LSB；新请求明确减少棕黄偏色、保自然内容，10种AI双语措辞，人工条件非相机照片。仍须真的看每个输入/目标，不能以像素往返替代视觉合格。背景：R7纯白平衡28案例仅12负色温来自6组、4原始英文全加暖；R10降低棕色输出非法负sepia，两语言失败。

重要：新的负E+正B组合可能改善原本拒绝的大幅提亮，尤其正常白衣合照07。因此44个质量拒绝需重核：40个强合成块/条带案例(20组×两请求)、R7真实训练质量拒绝3例、R10白衣正常强提亮1例。计划额外执行合理的负E+正B方案；强噪声2要求各1合理新尝试，原3真实噪声至少2尝试，白衣至少2。看图后可行者优先改成编辑标签，不能为凑原decline数拒绝。尚未创建此quality-probe脚本。无验收图可用于调参。

preflight记录：probe第一次误读style['images']['canonical']，实际源在counterfactual manifest，已修复；CPU没有图片写入前失败。prepare_labels第一次缺Path import，在写完全部未核验图片、candidate-cases后构造manifest时失败；已修复并在43012重跑，验证已有图像文件SHA不变，新增manifest标记44个quality决策必须重核。不要把这些候选当数据准入。DATA11不存在实际训练文件。

未来草案（未冻结）：单一路线从R10merger父772ead8e694bdf3d4fee2a189cc6e1222f7c2ebe539c7795364fbbfc7341ebbd开始，PeftModel.from_pretrained trainable仅一个116张量adapter（旧transfer_parent helper只支持112，不可误用）。NF4/BF16、r8a16drop0，lr2e-5、batch4，1epoch约336steps（按实际准入数量更新），seed20261123，条件rep20261124。固定最终一轮，不选损失。先CPU重放R10旧参数在新编辑器上作为editor-only对照（不是新模型推理，实际改图也须匿名看图）；再新模型相同新编辑器。门槛16/20、旧4/6、0重大、controls20/20，然后controlled81/100、复核、锁模、验收；原16h内预留约1.2h给完整有条件路径，不无限试验。实际标签数量/决定/训练protocol必须等核验完再冻。

没有subagents，禁止主动spawn。goal仍active未达成实用模型，不可因训练/预算接近完成而mark complete。

最新补充：prepare_labels重跑成功(672候选，51改图+10cool待核验，611同像素或同null)，未准入。10个cool面板均已实际查看，5个人像结果也看了实际原尺寸，全部C/A；已写label-candidates/direct-observations.jsonl（10条）。51个负曝光改图尚未查看（只有4张normal新图在前述probe另有原尺寸C/A记录，可按确切SHA继承，bright输入目标仍未看）。

44个质量拒绝的新可行性探针已生成：scripts11/probe_quality_decisions.py，quality-revalidation/manifest.json，49实际执行候选、24面板。20合成强退化组各2要求各1新尝试；n070/n127各2，n190强对比强色彩浪花3，正常白衣07强提亮2。四个非合成组的面板已实际看了，但未写评审；n070原图与两新结果均960x720实际查看，仍明显肤色/块噪/灰雾问题；n127原图与两新结果也查看，但工具将4608x3072缩至3888x2592，需原像素crop核验后写结论。n190/白衣新结果还未看完整实际尺寸。20组强合成噪声的新探针面板全部尚未查看。

export_quality_details.py已执行(38109)，输出quality-revalidation/details/manifest.json及13个源/新结果的整数原像素裁块，无缩放无调色。n070脸部ROI370x380；n127头/手臂ROI1024x1280；n190波浪ROI1200x800；白衣07整个483x542。rsync正在12488?注意之前12488已结束；最新details下载session见工具。必须实际view这些图后才能把这四组quality判定写到notes。


## 2026-11-06 08:30 中国时间更新（以上待办已被本段实际结果取代）
672语义/1344双语已准入并冻结，160原图80/80、548edit60keep64decline，97原始英文575AI。61新/变动目标全部实际查看记录；51负E/B有2例原要求提亮未完成：n048、x250，分别选择原尺寸复核的新repair3(E5C15Sh60H-8)、repair1(E10C20Sh65H-8)。失败初目标/repair1/2保留不入训练。44quality旧拒绝重试53执行图均无C/A：n070/n127各2(major)、n1907次(暗岩层次丢失，alt2major其余minor)、白衣07两次minor，20强合成噪声组各2均major。细节裁块实际看过，不仅面板。10cool均C/A。567原样edit/keep/unsupported证据明确继承。所有数据/输入/原图隔离/目标重放检查通过。
CPU缓存1344条7.6GB已生成服务器，当前尚未下载本机；排除batches.pt做rsync。CPU/meta116张量父参数(含4个已训练视觉连接张量)精确拷贝通过。编辑器方向/零值/中灰小信号连续、78正向与其他工具精确对照、严格拒绝/keep/非法输出与1344输入白名单检查通过。protocol已冻结，代码不可改。单路线R10merger父772ead...，lr2e-5/1ep336stepsseed20261123/条件rep20261124。
训练PID4034886 GPU0已启动，lateststep59/336 at00:30:13UTC，保护GPU1原作者3378799，budgetwatch3879191仍在。freeze资源14.3303705671/16h。liveworker4027009/web4027010无重启未接11。新web11/gpu_child11已准备本地但未部署，稍后需实测路径身份无回退。
CPU冻结R10merger旧参数重放新editor完成92条：native40中8改图，old12中2改图，controls40不改。明确非新模型推理。parent-editor-replay，native4匿名新图29/32/40/161及old030两新图实际查看。29C/A、40C/A、161C/A；32C/U待原像素星点/岩石裁块（full-resolution已导出5616x3744，details刚生成/rsync；必须view4crops后补充评审）。old030左969944...C/A、右ef2...P/A已记录。其余精确结果继承R10冻结最终证据（含历史U），不是重看。
R11 finalizer/reuse已准备，R10最终derivation为前置覆盖历史，避免旧raw U被当新分歧；不升级真正final U。尚未finalize或评分CPU对照，待32补看。最终训练模型尚未评测，验收保持关闭。R11没有新用户反馈。

## 第十一轮最终状态（归档前）
AI功能评审：本轮固定终点native15/20、0标注重大（历史U保留）、old4/6、controls20/20。和父参数仅换编辑器的CPU对照15/20相同，无新增训练收益证据。stage1失败，未controlled、未rep、未验收，defaultR1不变。已registered diagnostic functional-v11-negative-gain-20261123，models/from-functional-v10/sft-negative-gain-correction/seed-01，SHA12243ea5...4c4a56。其余评审精确继承R10最终library，R11新增8唯一结果（6CPU、2训练终点）和032两个区域原像素crop。所有raw/derived文件保留，0新历史分歧。
真实HTTP12+browser2全部完成，网页上传/模型切换/下载/反馈/无图拒绝/8条未合格筛选已操作核验。sourceagent11明确，2条agent反馈，不是用户反馈(0)。browser请求001ZH7bf7f222782842269e8aaee21fec8c7b，严重合成噪声拒绝aef2e3a29a9848f99e277bc2ba3ed6de。下载/Users/cryptic/Downloads/output (3).png SHA018d4db...84765与offline完全相同。CUA本阶段已按要求发送completion_summary“验证了本轮入口”，不得重复旧阶段；若以后新CUA阶段先rewriteDocumentation。浏览器tab1/iea9留在v11失败筛选页，已markDeliverable。3PNG截图本地+server demo，最终回复须内嵌至少一个已保存proof截图。
服务live新worker4040202/web4040203，receiptEXP11/demo/fixed-final-service-restart.json，保护原GPU1PID3378799。worker已显式unload，GPU0/2/3均1MiB空闲。budgetwatch3879191仍有效。资源14.761292240023609/16，共余1.238707759976391；本轮.43092167291376327。3次trialchild所有驻留空闲均计入，非利用率折扣。R11REPORT及三主文档和modelsREADME已写最终负结果与实际数据。
正在准备R11冻结归档（脚本package_delivery/verify_download），依赖前七份已核验archive，含缓存7.6GB；本机暂未下载batches.pt，打包后逐文件verify不必解压覆盖新目录。不要修改既有归档。R11归档两端核验完成前不要把它写成已完成。
后续有限路线想法尚未执行：先冻结一个一般化工具语义提示对照，R11权重不变，输入仍原图+原始请求+工具说明。重点是有符号滑块并非目标外观、sepia只能添加棕色(负值非法)、曝光负值实际整体倍率与轻/强力度的区别；不能硬修JSON或针对case006特判，不用验收。若新提示通过原既有功能门槛再controlled与条件复核/锁模/验收，仍在原16h内；不再盲目重复相同低lrSFT。应在新独立functional_v12目录实施、旧11冻结保持原样。尚未决定/编写新提示，未产生12CPU/GPU结果。goal仍active未达成高实用性，无子agents、不主动spawn。

归档范围前检：初包在server字节核验通过但漏了14agent11请求目录（筛选model-id仍v10前缀）。初包7ab2049fa03e7743fbf5f0fc5e80dfd39edf566348f1e39b65676803dd538b7a／1945065053bytes、866files，保留不可改，不能称最终完整交付。已修订packager另写complete-snapshot、functional-v11-20261106-complete.tar.gz、complete-receipt.json，必须14明确HTTP/browser requestIDs/input/request metadata/所有executed output均在清单才能打包。verify_download读complete-receipt并写complete-download-receipt，不能误用初receipt。完整包正在重打，尚未本机核验。范围审计delivery-audit-v1.json透明保留原因；无GPU新增。
