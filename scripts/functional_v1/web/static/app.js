let specs=[],activeRequest=null,selectedVerdict=null;
const $=id=>document.getElementById(id);
function appendPicture(container,src,label){if(!src)return;const figure=document.createElement('figure'),caption=document.createElement('figcaption'),link=document.createElement('a'),im=document.createElement('img');caption.textContent=label;link.href=src;link.target='_blank';link.rel='noopener';im.loading='lazy';im.src=src;im.alt=label;link.append(im);figure.append(caption,link);container.append(figure)}
async function loadModels(){const data=await(await fetch('/api/models')).json();specs=data.models;$('model').replaceChildren(...specs.map(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=m.display_name;return o}));$('model').value=data.default;modelNote()}
function modelNote(){const item=specs.find(m=>m.id===$('model').value);$('model-note').textContent=item?.status||''}
$('model').addEventListener('change',()=>{modelNote();activeRequest=null;for(const id of ['after','download','feedback','technical'])$(id).classList.add('hidden');$('empty-after').classList.remove('hidden');$('message').textContent='已切换模型，请重新生成结果。';});
$('image').addEventListener('change',()=>{const f=$('image').files[0];if(!f)return;$('before').src=URL.createObjectURL(f);$('before').classList.remove('hidden');$('empty-before').classList.add('hidden');$('after').classList.add('hidden');$('empty-after').classList.remove('hidden');$('download').classList.add('hidden');$('feedback').classList.add('hidden');$('technical').classList.add('hidden');activeRequest=null;});
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('instruction').value=b.dataset.prompt;$('instruction').focus()});
$('edit-form').addEventListener('submit',async event=>{event.preventDefault();const payload=new FormData(event.target);$('submit').disabled=true;for(const id of ['model','image','instruction'])$(id).disabled=true;document.querySelectorAll('[data-prompt]').forEach(b=>b.disabled=true);activeRequest=null;$('comment').value='';$('message').textContent='模型正在处理；首次加载可能需要约一分钟。';$('after').classList.add('hidden');$('empty-after').classList.remove('hidden');$('download').classList.add('hidden');$('feedback').classList.add('hidden');try{const response=await fetch('/api/edit',{method:'POST',body:payload});const data=await response.json();if(!response.ok)throw new Error(data.error+(data.detail?' '+data.detail:''));$('before').src=data.before;$('after').src=data.after;$('after').classList.remove('hidden');$('empty-after').classList.add('hidden');$('download').href=data.after;$('download').classList.remove('hidden');$('message').textContent=data.message;$('parameters').textContent=JSON.stringify({parameters:data.parameters,...data.identity},null,2);$('technical').classList.remove('hidden');activeRequest=data.request_id;selectedVerdict=null;document.querySelectorAll('[data-verdict]').forEach(b=>b.classList.remove('active'));$('feedback').classList.remove('hidden');$('feedback-status').textContent='反馈单独保存，不会自动加入训练集。';}catch(error){$('message').textContent=error.message;$('technical').classList.add('hidden')}finally{$('submit').disabled=false;for(const id of ['model','image','instruction'])$(id).disabled=false;document.querySelectorAll('[data-prompt]').forEach(b=>b.disabled=false)}});
document.querySelectorAll('[data-verdict]').forEach(button=>button.onclick=async()=>{if(!activeRequest)return;try{const response=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request_id:activeRequest,verdict:button.dataset.verdict,comment:$('comment').value})});const result=await response.json();if(!response.ok)throw new Error(result.error);document.querySelectorAll('[data-verdict]').forEach(b=>b.classList.remove('active'));button.classList.add('active');$('feedback-status').textContent=result.message;}catch(error){$('feedback-status').textContent=error.message}});
document.querySelectorAll('[data-tab]').forEach(button=>button.onclick=async()=>{document.querySelectorAll('.tab,.nav').forEach(e=>e.classList.remove('active'));$(button.dataset.tab).classList.add('active');button.classList.add('active');if(button.dataset.tab==='data')await loadExamples();if(button.dataset.tab==='results')await loadExperiment()});
async function loadExamples(){const round=$('record-round').value,v2=round==='v2',v3=round==='v3',v4=round==='v4',v6=round==='v6';$('v6-data-filter').classList.toggle('hidden',!v6);$('data-description').textContent=v6?'第六轮：160个原图组，人像／风景各80；460个语义案例、920条不同中英文记录，无重复采样。其中100例保留原始英文并配AI中文翻译，360例为受控构造。下方是实际采用标签；模型表现和训练状态见实验记录，标签合格不等于模型已学会。':v4?'第四轮：60张照片，人像／风景各30张。370个语义任务、740条不同中英文记录；其中10条原始自然请求各重复10次，实际训练920条。下方展示370个实际采用的标签，重复与翻译不计独立照片。':v3?'第三轮：37条偏好记录＋56条监督保持记录，50个语义任务、38张照片，人像／风景各19张。偏好样本依次为输入、采用的正例、被拒绝的模型结果；同一任务有多个合理正例时分别展示。点击图片查看原始尺寸。':v2?'第二轮：60张照片，人像、风景各30张；360个语义任务，720条中英文记录，100% AI构造请求。下方是实际采用的输入和标签。':'第一轮：160个原图组，含照片、扫描图和插画；220个语义任务，440条记录，100% AI构造请求。';const data=await(await fetch(v6?'/api/v6/training-examples':v4?'/api/v4/training-examples':v3?'/api/v3/training-examples':v2?'/api/v2/training-examples':'/api/training-examples')).json();const kind=$('training-kind').value;const shown=v6&&kind!=='all'?data.examples.filter(x=>kind==='native'?x.native:!x.native):data.examples;const cards=shown.map(item=>{const card=document.createElement('article');card.className='data-card'+(item.rejected?' preference-card':'');const title=document.createElement('h2');title.textContent=item.id+' · '+({brown_tone:'棕褐色调',compound:'复合要求',contrast:'对比度',saturation:'饱和度',lighting:'光照',white_balance:'白平衡',identity:'保持原样'}[item.family]||item.family);const text=document.createElement('p');text.textContent=item.instructions.zh+' / '+item.instructions.en;const images=document.createElement('div');images.className='example-images'+(item.rejected?' preference-images':'');appendPicture(images,item.before,'训练原图');appendPicture(images,item.after,'实际采用的正例');appendPicture(images,item.rejected,'被拒绝的模型结果');const note=document.createElement('small');note.textContent=item.status+'；来源：'+(item.instruction_origin||'可控双语任务')+'；原始记录 '+item.original_record_id;card.append(title,text,images,note);return card});$('data-cards').replaceChildren(...cards)}
async function loadExperiment(){
 if($('record-round').value==='v6')return loadRound6Experiment();
 if($('record-round').value==='v4')return loadRound4Experiment();
 if($('record-round').value==='v3')return loadRound3Experiment();
 if($('record-round').value==='v2')return loadRound2Experiment();
 $('review-filters').classList.add('hidden');
 const data=await(await fetch('/api/experiment')).json();const div=$('experiment');div.replaceChildren();
 const add=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;div.append(e)};
 add('p',`${data.review_mode}。训练数据：${data.semantic_cases}个语义案例、${data.source_groups}个原图组，共${data.training_records}条中英文记录。用户实际试用反馈另行保存。`);
 if(Object.keys(data.acceptance_results||{}).length)add('p','首轮已完成，高实用性目标尚未达成。已知问题包括力度不分、复合要求漏项，以及对比度任务退步。请结合下方图片和自己的照片判断。');
 const names={base:'原始 Qwen',current:'旧校准模型','functional-base-20261104':'基础起点候选','functional-current-20261104':'继续训练候选（默认）','functional-current-20261105':'继续训练换种子复核'};
 const dev={base:data.development_results.base,current:data.development_results.current};
 for(const[id,item]of Object.entries(data.selection?.models||{}))dev[id]=item.functional_summary;
 const families=[['lighting','光照'],['white_balance','白平衡'],['saturation','饱和度'],['contrast','对比度'],['brown_tone','棕褐色']];
 for(const[split,results]of [['开发集',dev],['冻结验收集',data.acceptance_results]]){
  add('h2',split);if(!Object.values(results||{}).some(Boolean)){add('p','尚未完成直接视觉评审。');continue}
  const table=document.createElement('table');const head=document.createElement('tr');for(const text of ['模型','中英文均合格 / 50',...families.map(x=>x[1]+' / 10')]){const th=document.createElement('th');th.textContent=text;head.append(th)}table.append(head);
  for(const[id,r]of Object.entries(results||{})){if(!r)continue;const row=document.createElement('tr');for(const text of [names[id]||id,r.fully_completed_both_languages,...families.map(([f])=>r.per_family[f]?.completed??'—')]){const td=document.createElement('td');td.textContent=text;row.append(td)}table.append(row)}div.append(table);
 }
 if(data.resources)add('p',`计算上限 ${data.resources.limit_gpu_hours} GPU小时；已结算 ${Number(data.resources.consumed_gpu_hours||0).toFixed(3)} GPU小时，运行中占用继续累计。`);
 const samples=await(await fetch('/api/evaluation-examples')).json();
 const verdicts={completed:'完成',partial:'部分完成',failed:'失败',uncertain:'不确定'};
 $('evaluation-cards').replaceChildren(...samples.examples.map(item=>{
  const card=document.createElement('article');card.className='data-card';
  const h=document.createElement('h2');h.textContent=item.case_id+' · '+(item.assessment.completion==='completed'&&item.assessment.quality!=='acceptable'?'动作完成，整体未合格':verdicts[item.assessment.completion]);
  const p=document.createElement('p');p.textContent=item.instruction;
  const images=document.createElement('div');images.className='example-images';appendPicture(images,item.before,'评审原图');appendPicture(images,item.after,'模型结果');
  const n=document.createElement('p');n.textContent=item.assessment.evidence;
  const small=document.createElement('small');small.textContent=(item.split==='acceptance'?'冻结验收集':'开发集')+' · '+(names[item.model_id]||item.model_id)+' · AI辅助评审';
  card.append(h,p,images,n,small);return card;
 }));
}
loadModels().catch(error=>$('message').textContent='暂时无法加载模型列表：'+error.message);

$('record-round').addEventListener('change',async()=>{if($('data').classList.contains('active'))await loadExamples();if($('results').classList.contains('active'))await loadExperiment()});
async function loadRound2Experiment(){
 const data=await(await fetch('/api/v2/experiment')).json();const div=$('experiment');div.replaceChildren();
 const add=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;div.append(e)};
 add('h2','第二轮完成开发评审，实用目标尚未达成');
 add('p','训练使用60张照片，人像、风景各30张，360个语义任务、720条中英文记录。所有请求均由AI构造，标签均经过直接看图核验。旧数据对照为440条不同语言记录加280条抽样重复，同样训练720步。');
 add('p','开发集为20张照片（人像／风景各10），每张5个任务，共100个语义任务。完成要求中英文均合格、无质量副作用，并区分同图强弱要求。翻译版本不作独立样本；以下均为AI辅助评审。');
 const table=document.createElement('table');const head=document.createElement('tr');
 for(const t of ['模型','完整完成 / 100','人像 / 50','风景 / 50','严重副作用案例','五项均完成原图 / 20']){const th=document.createElement('th');th.textContent=t;head.append(th)}table.append(head);
 const order=['parent-v1','control-e1','control-e2','control-e3','control-e4','curriculum-e1','curriculum-e2','curriculum-e3','curriculum-e4'];
 for(const id of order){const r=data.development_results[id];if(!r)continue;const tr=document.createElement('tr');const name=id==='parent-v1'?'第一轮父模型':(id.startsWith('control')?'旧数据对照':'新课程')+' · 第'+id.at(-1)+'轮';for(const text of [name,r.fully_completed_both_languages,r.per_domain.portrait.completed,r.per_domain.landscape.completed,r.major_quality_issues,r.all_tasks_complete_photo_groups]){const td=document.createElement('td');td.textContent=text;tr.append(td)}table.append(tr)}div.append(table);
 add('p','开发集固定新课程第3轮、对照第2轮供试用。新课程仍出现两例英文曝光过强，未达到升级门槛；默认模型保持第一轮候选。验收集未开封，也未启动换种子复核。L1不参与以上选择。');
 add('p','6张开发原图另用原始英文请求及中文翻译作补充检查：三个模型均为3/6中英文合格；新课程出现1例严重过曝。它不是独立测试，也不是用户反馈。');
 add('p',`共用计算上限16 GPU小时；截至记录已结算 ${Number(data.resources.consumed_gpu_hours||0).toFixed(3)} GPU小时。运行中的占用继续累计。`);
 setReviewModels('v2');$('review-filters').classList.remove('hidden');await loadRound2Evaluation();
}
async function loadRound2Evaluation(){
 const model=$('review-model').value,split=$('review-split').value;
 const data=await(await fetch('/api/'+(['v3','v4','v6'].includes($('record-round').value)?$('record-round').value:'v2')+'/evaluation-examples?'+new URLSearchParams({model,split}))).json();
 const verdicts={completed:'完成',partial:'部分完成',failed:'失败',uncertain:'不确定'},quality={acceptable:'质量合格',minor_issue:'有质量副作用',major_issue:'严重副作用',uncertain:'质量不确定'};
 $('evaluation-cards').replaceChildren(...data.examples.map(item=>{
  const card=document.createElement('article');card.className='data-card';const h=document.createElement('h2');h.textContent=item.case_id+' · '+item.language.toUpperCase()+' · '+(item.assessment.completion==='completed'&&item.assessment.quality!=='acceptable'?'动作完成，整体未合格':verdicts[item.assessment.completion]);
  const p=document.createElement('p');p.textContent=item.instruction;const imgs=document.createElement('div');imgs.className='example-images';appendPicture(imgs,item.before,'评审原图');appendPicture(imgs,item.after,'模型实际结果');
  const note=document.createElement('p');note.textContent=quality[item.assessment.quality]+'：'+item.assessment.evidence;const small=document.createElement('small');small.textContent=item.model_id+' · '+({'development':'受控开发集','native-development':'20张原始请求开发','old-native':'6张历史原始请求回归','natural-use':'6张原始请求补充检查'}[split]||split)+' · AI辅助评审，无用户认证';card.append(h,p,imgs,note,small);return card;
 }));
}
$('review-model').addEventListener('change',loadRound2Evaluation);$('review-split').addEventListener('change',loadRound2Evaluation);

function setReviewModels(round){const oldSplit=$('review-split').value,splitChoices=round==='v6'?[['native-development','20张原始请求开发'],['old-native','6张历史原始请求回归'],['development','100项受控开发']]:[['development','开发集'],['natural-use','6张原始请求补充检查']];$('review-split').replaceChildren(...splitChoices.map(([id,name])=>{const o=document.createElement('option');o.value=id;o.textContent=name;return o}));if(splitChoices.some(x=>x[0]===oldSplit))$('review-split').value=oldSplit;const value=$('review-model').value,choices=round==='v6'?[['raw-base','原始Qwen'],['current-default','现有默认模型'],['round2-parent','第二轮父模型'],...[1,2,3,4].flatMap(e=>[['base-e'+e,'第六轮基础起点 · 第'+e+'轮'],['continued-e'+e,'第六轮继续训练 · 第'+e+'轮']])]:round==='v4'?[['measured-e1','第四轮观测输入 · 第1轮'],['measured-e2','第四轮观测输入 · 第2轮'],['plain-e1','第四轮普通输入 · 第1轮'],['plain-e2','第四轮普通输入 · 第2轮'],['parent-plain','未训练父模型 · 普通输入'],['parent-measured','未训练父模型 · 观测输入']]:round==='v3'?[['dpo-e1','第三轮偏好优化'],['sft-e1','第三轮匹配普通微调']]:[['curriculum-e3','第二轮人像／风景课程'],['parent-v1','第一轮父模型'],['control-e2','第二轮旧数据对照']];$('review-model').replaceChildren(...choices.map(([id,name])=>{const o=document.createElement('option');o.value=id;o.textContent=name;return o}));if(choices.some(x=>x[0]===value))$('review-model').value=value;}
async function loadRound3Experiment(){
 const data=await(await fetch('/api/v3/experiment')).json(),div=$('experiment');div.replaceChildren();const add=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;div.append(e)};
 add('h2','第三轮纠错实验未通过升级门槛');
 add('p','两路线均从第二轮课程e3出发，用同样93条记录（37条偏好、56条监督保持），38张照片、人像／风景各19张，训练2轮共48步。偏好优化和普通微调都未证明达到实用目标。');
 const table=document.createElement('table'),head=document.createElement('tr');for(const t of ['模型','开发完整合格 / 100','开发严重副作用','原始请求合格 / 6','原始请求严重副作用']){const th=document.createElement('th');th.textContent=t;head.append(th)}table.append(head);
 for(const id of ['dpo-e1','dpo-e2','sft-e1','sft-e2']){const d=data.development_results[id],n=data.natural_results[id],tr=document.createElement('tr');for(const t of [(id.startsWith('dpo')?'偏好优化':'匹配普通微调')+' · 第'+id.at(-1)+'轮',d.fully_completed_both_languages,d.major_quality_issues,n.fully_completed_both_languages,n.major_quality_issues]){const td=document.createElement('td');td.textContent=t;tr.append(td)}table.append(tr)}div.append(table);
 add('p','两路线第1轮提供诊断试用；都未升级默认模型。配对复核中6个旧标签被修订，20组没有明确优劣的结果不建立偏好。验收仍未开封。这仅是小规模纠错结果，不能泛称偏好优化无效。');
 add('p',`AI辅助直接视觉评审，无用户认证。最初16 GPU小时共用，已结算 ${Number(data.resources.consumed_gpu_hours||0).toFixed(3)} GPU小时。`);
 setReviewModels('v3');$('review-filters').classList.remove('hidden');await loadRound2Evaluation();
}

async function loadRound4Experiment(){
 const data=await(await fetch('/api/v4/experiment')).json(),div=$('experiment');div.replaceChildren();const add=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;div.append(e)};
 add('h2','第四轮图像观测实验未通过升级门槛');
 add('p','两路线从同一第二轮父模型开始，使用相同920条记录、60张照片（人像／风景各30）、相同种子与460步训练。370个语义任务中360个请求由AI构造，10个保留原始英文并配AI中文翻译；这10个各重复10次。观测路线额外读取当前照片的亮度和色彩统计；参考图、标签及评审意见不进入输入。');
 const table=document.createElement('table'),head=document.createElement('tr');for(const t of ['模型','开发完整合格 / 100','开发严重副作用','原始请求合格 / 6','原始请求严重副作用']){const th=document.createElement('th');th.textContent=t;head.append(th)}table.append(head);
 const names={'parent-plain':'未训练父模型 · 普通输入','parent-measured':'未训练父模型 · 观测输入','plain-e1':'普通输入 · 第1轮','plain-e2':'普通输入 · 第2轮','measured-e1':'观测输入 · 第1轮','measured-e2':'观测输入 · 第2轮'};
 for(const[id,name]of Object.entries(names)){const d=data.development_results[id],n=data.natural_results[id],tr=document.createElement('tr');for(const t of [name,d.fully_completed_both_languages,d.major_quality_issues,n.fully_completed_both_languages,n.major_quality_issues]){const td=document.createElement('td');td.textContent=t;tr.append(td)}table.append(tr)}div.append(table);
 add('p','完整合格要求同一任务的中英文都完成且质量合格，并区分强弱要求。未训练父模型的观测输入另有开发2条、原始请求2条非法输出，均计失败。两训练路线仍有过曝和复合要求漏项；第1轮权重仅供诊断试用，没有合格冠军。默认模型未更换，未启动换种子复核，验收未开封。');
 add('p',`均为AI辅助直接视觉评审，无用户认证。共享16 GPU小时上限，已结算 ${Number(data.resources.consumed_gpu_hours||0).toFixed(3)} GPU小时。`);
 setReviewModels('v4');$('review-filters').classList.remove('hidden');await loadRound2Evaluation();
}

$('training-kind').addEventListener('change',loadExamples);
async function loadRound6Experiment(){
 const data=await(await fetch('/api/v6/experiment')).json(),div=$('experiment');div.replaceChildren();$('review-filters').classList.add('hidden');$('evaluation-cards').replaceChildren();const add=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;div.append(e)};
 add('h2','第六轮：原始请求与照片覆盖');
 add('p','实际训练160个原图组，人像／风景各80；460个语义案例、920条不同中英文记录，无重复采样。100例保留原始英文，配对应中文AI翻译；360例为受控构造。标签全部实际执行并直接看图。3个未获得合格标签的来源已留档排除，在训练前从已审计训练池补入3张。');
 add('p','新开发与验收各20张原始请求照片，均为10人像／10风景，要求在本轮模型输出前冻结。中英文均完成且质量合格才计一例；新开发须至少16/20、旧6张回归至少4/6且无严重副作用，再看100项受控开发。以下为AI评审，不是用户认证。');
 for(const[id,r]of Object.entries(data.training)){add('p',(id.startsWith('base')?'基础起点':'继续训练')+'：'+(r.status==='completed'?(data.selection?'训练与开发评审完成，未通过功能门槛':'训练完成，等待逐图功能评审'):r.status==='running'?'正在训练，约完成'+Number(r.epoch||0).toFixed(1)+'/4轮':r.status==='starting'?'正在准备训练':r.status));}
 const ids=Object.keys(data.summaries['native-development']);
 if(!ids.length)add('p','尚无本轮完整逐图评审结果。训练损失与可执行率不代表修图成功，默认试用模型保持原候选。');
 else {const table=document.createElement('table'),head=document.createElement('tr');for(const t of ['模型','原始开发合格 / 20','严重副作用','旧回归合格 / 6','旧回归严重副作用','受控开发合格 / 100']){const th=document.createElement('th');th.textContent=t;head.append(th)}table.append(head);for(const id of ids){const n=data.summaries['native-development'][id],o=data.summaries['old-native'][id],c=data.summaries.development[id],tr=document.createElement('tr');for(const t of [id,n.fully_completed_both_languages,n.major_quality_issues,o?.fully_completed_both_languages??'待评审',o?.major_quality_issues??'待评审',c?.fully_completed_both_languages??(data.selection?.controlled_development_status?.startsWith('not_run')?'未运行（原始请求门槛未过）':'待评审')]){const td=document.createElement('td');td.textContent=t;tr.append(td)}table.append(tr)}div.append(table);setReviewModels('v6');if(data.selection?.controlled_development_status?.startsWith('not_run')&&$('review-split').value==='development')$('review-split').value='native-development';if(!ids.includes($('review-model').value))$('review-model').value=ids[0];$('review-filters').classList.remove('hidden');await loadRound2Evaluation();}
 add('p',`共享16 GPU小时上限，当前累计占用约 ${Number(data.resources.occupied_gpu_hours_live).toFixed(3)} GPU小时，运行中的时间继续累计。`);
 if(data.selection)add('p',data.selection.qualified?'开发选定候选，仍须独立验收和用户实际试用。':'未通过开发功能门槛，未升级默认模型。');
}
