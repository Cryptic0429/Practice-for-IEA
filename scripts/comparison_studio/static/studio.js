'use strict';
const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let config, job = null, jobs = [], sampleId = null, upload = null, objectURL = null;
const qaHeaders = new URLSearchParams(location.search).get('qa') === '1' ? {'X-IEA-Source':'agent-smoke'} : {};
let language = 'zh', blind = false, signature = '', busy = false, feedback = {ratings:{}, preferred:{}, comment:''};
const statusNames = {queued:'等待运行',loading:'载入模型',running:'正在生成',completed:'已完成',failed:'未完成',edited:'已编辑',unchanged:'保持原样',declined:'未交付'};
let toastTimer;
function toast(text) { $('#toast').textContent = text; $('#toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').hidden = true, 5000); }
async function api(url, options) {
  const r = await fetch(url, {...options, headers:{...qaHeaders,...options?.headers}}); let data;
  try { data = await r.json(); } catch { throw new Error('服务器暂时无法返回结果，请稍后刷新。'); }
  if (!r.ok) throw new Error(data.error || '请求未完成');
  return data;
}
function setSource(url, name, size = '') {
  $('#source-preview').src = url; $('#source-preview').hidden = false;
  $('#upload-empty').hidden = true; $('#upload-overlay').hidden = false;
  $('#upload-zone').classList.add('has-image'); $('#image-name').textContent = name; $('#image-size').textContent = size;
}
function selectSample(id, loadSaved = true) {
  const sample = config.samples.find(s => s.id === id); if (!sample) return;
  sampleId = id; upload = null; if (objectURL) URL.revokeObjectURL(objectURL); objectURL = null;
  $('#image-input').value = ''; $('#instruction-zh').value = sample.instructions.zh; $('#instruction-en').value = sample.instructions.en;
  setSource(sample.image, sample.title, '历史样例');
  document.querySelectorAll('.sample').forEach(e => e.classList.toggle('active', e.dataset.sample === id));
  job = null; feedback = {ratings:{},preferred:{},comment:''}; signature = ''; render();
  const saved = jobs.find(j => j.sample_id === id && j.status === 'completed');
  if (loadSaved && saved) loadJob(saved.id);
}
function setUpload(file) {
  if (!file) return;
  if (file.size > 22 * 1024 * 1024) {toast('请选择22 MB以内的图片。'); return;}
  if (!file.type.startsWith('image/')) {toast('请选择图片文件。'); return;}
  sampleId = null; upload = file; job = null; signature = '';
  if (objectURL) URL.revokeObjectURL(objectURL); objectURL = URL.createObjectURL(file);
  setSource(objectURL, file.name, `${(file.size / 1048576).toFixed(1)} MB`);
  document.querySelectorAll('.sample').forEach(e => e.classList.remove('active'));
  feedback = {ratings:{},preferred:{},comment:''}; render();
}
function modelOrder() {
  const order = Object.keys(config.models);
  if (!blind) return order;
  let seed = parseInt((job?.id || 'a3f92c11').slice(0, 8), 16);
  for (let i = order.length - 1; i > 0; i--) {seed = (Math.imul(seed,1664525) + 1013904223) >>> 0; const j = seed % (i + 1); [order[i],order[j]] = [order[j],order[i]];}
  return order;
}
function nameFor(id) { const i = modelOrder().indexOf(id); return blind ? `候选 ${String.fromCharCode(65+i)}` : config.models[id].name; }
function renderSamples() {
  $('#samples').innerHTML = config.samples.map(s => `<button class="sample" data-sample="${s.id}"><img src="${s.image}" alt="${escapeHTML(s.title)}"><span><strong>${escapeHTML(s.title)}</strong><small>${escapeHTML(s.category)}</small></span></button>`).join('');
  document.querySelectorAll('.sample').forEach(e => e.addEventListener('click', () => selectSample(e.dataset.sample)));
}
function stageHTML(row, model) {
  const state = row?.status || 'empty';
  if (row?.after) return `<div class="compare-stage"><img src="${row.after}" alt="${escapeHTML(nameFor(model))}编辑后的图片"><img class="before-layer" src="${job.before}" alt="原图"><div class="compare-divider"></div><input class="compare-slider" type="range" min="0" max="100" value="50" aria-label="拖动比较原图与${escapeHTML(nameFor(model))}结果"><span class="stage-tag left">原图</span><span class="stage-tag right">结果</span><button class="stage-open" data-open="${row.after}" data-title="${escapeHTML(nameFor(model))} · ${row.language === 'zh' ? '中文' : 'English'}">放大 ↗</button></div>`;
  let title = '等待你的照片', description = '运行后，拖动滑杆比较前后效果。', icon = '＋';
  if (state === 'queued' || state === 'running') {title = state === 'running' ? '正在生成' : '等待运行'; description = state === 'running' ? '正在执行这条原样指令。' : '每个模型都会处理同一张图片。'; icon = '◌';}
  if (state === 'declined') {title = '模型选择不交付'; description = row.reason === 'unreliable_quality' ? '模型判断无法可靠保证画质。没有生成结果图。' : '这项要求超出当前工具支持范围。没有生成结果图。'; icon = '—';}
  if (state === 'failed') {title = '这次运行未完成'; description = '输出或执行未通过检查。没有生成替代图片。'; icon = '!';}
  return `<div class="compare-stage"><div class="stage-empty"><span class="placeholder-icon ${state === 'running' ? 'loading-dot' : ''}">${icon}</span><strong>${title}</strong><span>${description}</span>${state==='declined'?'<small>拒绝不等于完成修图</small>':''}</div></div>`;
}
function outputHTML(row, model, lang) {
  const status = row?.status || 'empty', key = `${model}-${lang}`;
  const timing = row?.seconds !== undefined ? (model === 'r21' ? `追加 ${row.seconds.toFixed(1)}s · 复用主模型 ${Number(row.parent_seconds || 0).toFixed(1)}s` : `${row.seconds.toFixed(1)}s`) : '同一原图 · 原样指令';
  const score = feedback.ratings?.[key] || '';
  const options = [['','评价这张结果…'],['completed','完成要求，画质可接受'],['partial','部分完成 / 力度不合适'],['failed','失败 / 明显副作用'],['uncertain','暂时无法判断']];
  return `${language === 'both' ? `<div class="language-row"><span>${lang==='zh'?'中文指令':'ENGLISH INSTRUCTION'}</span><span>${statusNames[status] || '待比较'}</span></div>` : ''}${stageHTML(row, model)}<div class="card-meta"><span>${escapeHTML(timing)}</span>${row?.after ? `<a href="${row.after}?download=1" download>下载 PNG ↓</a>` : '<span>—</span>'}</div>${row?.decision || row?.error ? `<details class="params"><summary>查看${row?.error?'运行信息':'编辑参数'}</summary><pre>${escapeHTML(row?.error || JSON.stringify(row.decision,null,2))}</pre></details>` : ''}<div class="card-bottom"><select aria-label="${escapeHTML(nameFor(model))} ${lang}结果评价" data-rating="${key}" ${!row || ['queued','running'].includes(status) ? 'disabled' : ''}>${options.map(([v,t])=>`<option value="${v}" ${v===score?'selected':''}>${t}</option>`).join('')}</select></div>`;
}
function render() {
  if (!config) return;
  $('#result-title').textContent = job ? job.title : '四个候选，同场对照';
  $('#download-all').classList.toggle('disabled', !job);
  $('#download-all').setAttribute('aria-disabled', String(!job));
  if(job) $('#download-all').href = `/api/jobs/${job.id}/download`; else $('#download-all').removeAttribute('href');
  $('#original-button').disabled = !job && !sampleId && !upload;
  const active = job && ['queued','loading','running'].includes(job.status);
  $('#progress-box').hidden = !job;
  if(job) { $('#progress-message').textContent = job.message; $('#progress-count').textContent = `${job.finished || 0} / 8`; $('#progress-fill').style.width = `${(job.finished||0)/8*100}%`; }
  const dirty = job && (job.instructions.zh !== $('#instruction-zh').value.trim() || job.instructions.en !== $('#instruction-en').value.trim());
  if(job) {
    const source = job.sample_id ? '历史开发图片' : '本次上传';
    $('#context-note').textContent = `${dirty ? '输入已修改；下方仍是上一组要求的结果。 ' : ''}${source} · ${job.image_size.join(' × ')} · ${job.id.slice(0,8)} ｜ 本次中文：${job.instructions.zh} ｜ EN: ${job.instructions.en}`;
  } else $('#context-note').textContent = '四个候选各自使用原配套方案。生成结果后，可拖动分界线看前后差异。';
  $('#run-button span').textContent = dirty ? '按新要求重新对比' : job ? '重新运行四模型对比' : '开始四模型对比';
  $('#run-button').disabled = busy || Boolean(active && !dirty);
  const nextSignature = JSON.stringify([job?.id,job?.results,language,blind]);
  if(signature !== nextSignature) {
    signature = nextSignature;
    const langs = language === 'both' ? ['zh','en'] : [language];
    $('#model-grid').innerHTML = modelOrder().map((model,index) => {
      const spec = config.models[model], row = job?.results.find(r => r.model === model && r.language === langs[0]);
      const state = row?.status || 'empty';
      return `<article class="model-card"><div class="card-heading"><div class="card-title"><span class="model-dot">${String.fromCharCode(65+index)}</span><div><h3>${escapeHTML(nameFor(model))}</h3><p>${blind ? '模型身份已隐藏' : escapeHTML(spec.subtitle)}</p></div></div><span class="card-status ${['edited','unchanged'].includes(state)?'good': ['failed','declined'].includes(state)?'bad':''}">${statusNames[state] || '待比较'}</span></div>${langs.map(l=>outputHTML(job?.results.find(r=>r.model===model&&r.language===l),model,l)).join('')}</article>`;
    }).join('');
    document.querySelectorAll('.compare-slider').forEach(slider => slider.addEventListener('input', () => { const stage=slider.parentElement; stage.querySelector('.before-layer').style.clipPath=`inset(0 ${100-slider.value}% 0 0)`; stage.querySelector('.compare-divider').style.left=`${slider.value}%`; }));
    document.querySelectorAll('[data-open]').forEach(b => b.addEventListener('click',()=>openImage(b.dataset.open,b.dataset.title)));
    document.querySelectorAll('[data-rating]').forEach(s => s.addEventListener('change',()=>{feedback.ratings[s.dataset.rating]=s.value; $('#feedback-message').textContent='有未保存的评价';}));
  }
  $('#verdict-panel').hidden = !job || !(job.finished > 0);
  for(const l of ['zh','en']) {
    const el=$(`#preferred-${l}`), value=feedback.preferred?.[l] || '';
    const opts=[['','请选择…'],...modelOrder().map(m=>[m,nameFor(m)]),['tie','没有明显差别'],['none','都不适合交付']];
    el.innerHTML=opts.map(([v,t])=>`<option value="${v}" ${v===value?'selected':''}>${escapeHTML(t)}</option>`).join('');
  }
}
async function loadHistory() {
  const data = await api('/api/jobs'); jobs=data.jobs.filter(j => !(j.source === 'agent_smoke_test' && (!j.sample_id || j.status === 'failed')));
  $('#history-list').innerHTML = jobs.length ? jobs.slice(0,30).map(j=>`<button class="history-item ${j.id===job?.id?'active':''}" data-job="${j.id}"><span>${escapeHTML(j.title)}</span><small>${statusNames[j.status]} · ${new Date(j.created_at).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',timeZone:'Asia/Shanghai'})}</small></button>`).join('') : '<p class="muted">你的对比记录会保存在这里。</p>';
  document.querySelectorAll('[data-job]').forEach(b=>b.addEventListener('click',()=>loadJob(b.dataset.job)));
}
async function loadJob(id) {
  try {
    const data = await api(`/api/jobs/${id}`); job=data; sampleId=data.sample_id || null; upload=null;
    feedback=structuredClone(data.feedback || {}); feedback.ratings ||= {}; feedback.preferred ||= {};
    $('#feedback-comment').value=feedback.comment || ''; $('#feedback-message').textContent=data.feedback?.at ? '已保存的评价' : '反馈不自动用于训练。';
    $('#instruction-zh').value=data.instructions.zh; $('#instruction-en').value=data.instructions.en;
    setSource(data.before,data.title,data.image_size.join(' × '));
    document.querySelectorAll('.sample').forEach(e=>e.classList.toggle('active',e.dataset.sample===sampleId));
    signature=''; render(); loadHistory().catch(()=>{});
  } catch(e) {toast(e.message);}
}
function openImage(url,title) {$('#dialog-image').src=url;$('#dialog-title').textContent=title;$('.dialog-image-wrap').classList.remove('pixel');$('#pixel-toggle').textContent='原始像素 1:1';$('#image-dialog').showModal();}
async function refreshRuntime() {
  const h=await api('/health');
  $('#connection').textContent=h.device==='cuda'?'GPU 服务已连接':'CPU 服务已连接';$('#connection').classList.remove('offline');
  const left=Math.max(0,h.budget_gpu_hours-h.used_gpu_hours);
  $('#compute-note').textContent=h.device==='cuda'?`GPU 加速 · 本次预算剩余约 ${(left*60).toFixed(0)} 分钟`:'CPU 推理 · 一组对比可能需要数分钟';
  $('#runtime-detail').textContent=`本次精度：${h.precision}。${h.device==='cuda'?`新增 GPU 已占用 ${h.used_gpu_hours.toFixed(3)} / ${h.budget_gpu_hours} 小时，包含载入与推理；空闲时释放显存。`:'CPU FP32 与历史 GPU NF4 评测可能产生不同决策。'}`;
}
$('#compare-form').addEventListener('submit',async e=>{
  e.preventDefault(); if(busy)return;
  if(!upload && !sampleId && !job) {toast('先选择一张图片。');return;}
  busy=true;render();
  try {
    const data=new FormData(); data.set('zh',$('#instruction-zh').value.trim());data.set('en',$('#instruction-en').value.trim());
    if(upload) data.set('image',upload); else if(sampleId) data.set('sample_id',sampleId); else if(job) {const r=await fetch(job.before);data.set('image',await r.blob(),job.title+'.png');}
    const newJob=await api('/api/jobs',{method:'POST',headers:qaHeaders,body:data});
    job=newJob;feedback={ratings:{},preferred:{},comment:''};$('#feedback-comment').value='';$('#feedback-message').textContent='反馈不自动用于训练。';signature='';
    await loadHistory();toast('已加入队列，8项结果会逐个显示。');
  } catch(e) {toast(e.message);} finally {busy=false;render();}
});
$('#image-input').addEventListener('change',e=>setUpload(e.target.files[0]));
$('#upload-zone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#image-input').click();}});
['dragenter','dragover'].forEach(name=>$('#upload-zone').addEventListener(name,e=>{e.preventDefault();$('#upload-zone').classList.add('dragover');}));
['dragleave','drop'].forEach(name=>$('#upload-zone').addEventListener(name,e=>{e.preventDefault();$('#upload-zone').classList.remove('dragover');if(name==='drop')setUpload(e.dataTransfer.files[0]);}));
document.querySelectorAll('[data-language]').forEach(b=>b.addEventListener('click',()=>{language=b.dataset.language;document.querySelectorAll('[data-language]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-selected',String(x===b));});render();}));
$('#blind-toggle').addEventListener('change',e=>{blind=e.target.checked;render();});
$('#original-button').addEventListener('click',()=>openImage(job?.before || $('#source-preview').src,'编辑前 · 原图'));
$('#close-image').addEventListener('click',()=>$('#image-dialog').close());
$('#pixel-toggle').addEventListener('click',()=>{const on=$('.dialog-image-wrap').classList.toggle('pixel');$('#pixel-toggle').textContent=on?'适应窗口':'原始像素 1:1';});
$('#about-button').addEventListener('click',()=>$('#about-dialog').showModal());$('#close-about').addEventListener('click',()=>$('#about-dialog').close());
$('#refresh-history').addEventListener('click',()=>loadHistory().catch(e=>toast(e.message)));
for(const l of ['zh','en']) {$(`#preferred-${l}`).addEventListener('change',e=>{feedback.preferred[l]=e.target.value;$('#feedback-message').textContent='有未保存的评价';});$(`#instruction-${l}`).addEventListener('input',()=>render());}
$('#feedback-comment').addEventListener('input',e=>{feedback.comment=e.target.value;$('#feedback-message').textContent='有未保存的评价';});
$('#save-feedback').addEventListener('click',async()=>{
  if(!job)return;const b=$('#save-feedback');b.disabled=true;
  try {await api(`/api/jobs/${job.id}/feedback`,{method:'POST',headers:{'Content-Type':'application/json',...qaHeaders},body:JSON.stringify(feedback)});$('#feedback-message').textContent='已保存 · 不自动用于训练';toast('评价已保存。');}catch(e){toast(e.message);}finally{b.disabled=false;}
});
$('#reset-button').addEventListener('click',()=>{sampleId=null;upload=null;job=null;signature='';$('#image-input').value='';$('#source-preview').hidden=true;$('#upload-empty').hidden=false;$('#upload-overlay').hidden=true;$('#upload-zone').classList.remove('has-image');$('#image-name').textContent='JPG、PNG、WebP 等';$('#image-size').textContent='≤ 22 MB';$('#instruction-zh').value='';$('#instruction-en').value='';document.querySelectorAll('.sample').forEach(e=>e.classList.remove('active'));render();});
async function poll() {
  try {
    if(job && ['queued','loading','running'].includes(job.status)) {
      const id=job.id, latest=await api(`/api/jobs/${id}`); if(job?.id===id){job=latest;render();if(['completed','failed'].includes(job.status))await loadHistory();}
    }
  } catch(e){$('#connection').textContent='连接中断，正在重试';$('#connection').classList.add('offline');}
  setTimeout(poll,1300);
}
async function init() {
  try {
    config=await api('/api/config');renderSamples();await loadHistory();
    $('#model-details').innerHTML=Object.values(config.models).map(m=>`<div class="about-model"><strong>${escapeHTML(m.name)} · ${escapeHTML(m.subtitle)}</strong><span>${escapeHTML(m.note)}</span></div>`).join('');
    selectSample('native-dev-011');await refreshRuntime();poll();
    setInterval(()=>{refreshRuntime().catch(()=>{$('#connection').textContent='服务暂未连接';$('#connection').classList.add('offline');});loadHistory().catch(()=>{});},10000);
  }catch(e){toast(e.message);$('#connection').textContent='服务暂未连接';$('#connection').classList.add('offline');}
}
init();
