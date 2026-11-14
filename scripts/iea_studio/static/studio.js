'use strict';
const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const TOOL_ORDER = [
  ['exposure', '曝光'],
  ['brightness', '亮度'],
  ['contrast', '对比度'],
  ['highlights', '高光'],
  ['shadows', '阴影'],
  ['saturation', '饱和度'],
  ['temperature', '色温'],
  ['tint', '色调'],
  ['sepia', '复古'],
];

let config = null;
let appMode = 'agent';
let session = null;
let sessionsList = [];
let agentBusy = false;

let job = null;
let jobs = [];
let sampleId = null;
let upload = null;
let objectURL = null;
const qaHeaders = new URLSearchParams(location.search).get('qa') === '1' ? {'X-IEA-Source': 'agent-smoke'} : {};
let language = 'zh';
let blind = false;
let signature = '';
let busy = false;
let feedback = {ratings: {}, preferred: {}, comment: ''};

const statusNames = {
  queued: '等待运行', loading: '载入模型', running: '正在生成', completed: '已完成',
  failed: '未完成', edited: '已编辑', unchanged: '保持原样', declined: '未交付'
};

let toastTimer;
function toast(text) {
  $('#toast').textContent = text;
  $('#toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('#toast').hidden = true, 4500);
}

async function api(url, options) {
  const r = await fetch(url, {...options, headers: {...qaHeaders, ...options?.headers}});
  let data;
  try { data = await r.json(); } catch { throw new Error('服务暂时无法返回结果，请稍后刷新。'); }
  if (!r.ok) throw new Error(data.error || '请求未完成');
  return data;
}

/* ==================== TOP MODE & SAMPLES MODAL ==================== */
function switchMode(mode) {
  appMode = mode;
  document.querySelectorAll('.mode-tab').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));
  $('#agent-view').hidden = mode !== 'agent';
  $('#compare-view').hidden = mode !== 'compare';
}

function renderSamplesGallery() {
  if (!config) return;
  $('#samples').innerHTML = config.samples.map(s => `
    <button type="button" class="sample-card" data-pick-sample="${s.id}">
      <img src="${s.image}" alt="${escapeHTML(s.title)}">
      <div class="sample-card-body">
        <small>${escapeHTML(s.category)}</small>
        <strong>${escapeHTML(s.title)}</strong>
        <p>${escapeHTML(s.instructions.zh)}</p>
      </div>
    </button>
  `).join('');
  document.querySelectorAll('[data-pick-sample]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.pickSample;
      $('#samples-dialog').close();
      if (appMode === 'agent') {
        await createAgentSession({sampleId: id});
      } else {
        selectSample(id);
      }
    });
  });
}

/* ==================== MODE 1: CONVERSATIONAL AGENT ==================== */
async function refreshSessionsList() {
  const data = await api('/api/sessions');
  sessionsList = data.sessions || [];
  const select = $('#agent-session-select');
  const currentId = session?.session_id || '';
  select.innerHTML = `<option value="">选择或新建会话 (${sessionsList.length})</option>` +
    sessionsList.map(s => `<option value="${s.session_id}" ${s.session_id === currentId ? 'selected' : ''}>${escapeHTML(s.title)} · 第 ${s.active_turn} 轮</option>`).join('');
}

async function createAgentSession({file = null, sampleId: pickedSample = null} = {}) {
  const form = new FormData();
  if (file) form.set('image', file);
  else if (pickedSample) form.set('sample_id', pickedSample);
  else return;

  try {
    agentBusy = true;
    renderAgent();
    session = await api('/api/sessions', {method: 'POST', body: form});
    await refreshSessionsList();
    renderAgent();
    if (pickedSample) {
      const sample = config?.samples?.find(s => s.id === pickedSample);
      if (sample) $('#agent-chat-input').value = sample.instructions.zh;
    }
    toast('已创建新的修图会话，您可以开始多轮对话。');
  } catch (e) {
    toast(e.message);
  } finally {
    agentBusy = false;
    renderAgent();
  }
}

async function loadAgentSession(sessionId) {
  if (!sessionId) return;
  try {
    session = await api(`/api/sessions/${sessionId}`);
    renderAgent();
  } catch (e) {
    toast(e.message);
  }
}

async function sendAgentMessage(text) {
  if (!session || !text.trim() || agentBusy) return;
  const sid = session.session_id;
  agentBusy = true;
  session.messages.push({role: 'user', text: text.trim()});
  renderAgent();
  try {
    const res = await api(`/api/sessions/${sid}/messages`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text.trim()}),
    });
    session = res.session;
    $('#agent-chat-input').value = '';
    await refreshSessionsList();
  } catch (e) {
    toast(e.message);
  } finally {
    agentBusy = false;
    renderAgent();
  }
}

async function checkoutAgentTurn(payload) {
  if (!session || agentBusy) return;
  agentBusy = true;
  renderAgent();
  try {
    const res = await api(`/api/sessions/${session.session_id}/checkout`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    session = res.session;
    await refreshSessionsList();
  } catch (e) {
    toast(e.message);
  } finally {
    agentBusy = false;
    renderAgent();
  }
}

function renderAgent() {
  const hasSession = Boolean(session);
  $('#agent-empty-state').hidden = hasSession;
  $('#agent-split-viewer').hidden = !hasSession;
  $('#agent-pixel-open').disabled = !hasSession;
  $('#agent-undo-btn').disabled = !hasSession || agentBusy || (session?.active_turn ?? 0) <= 0;
  $('#agent-reset-btn').disabled = !hasSession || agentBusy || (session?.active_turn ?? 0) <= 0;
  $('#agent-send-btn').disabled = !hasSession || agentBusy;

  const dlImg = $('#agent-download-image');
  const dlZip = $('#agent-download-zip');
  dlImg.classList.toggle('disabled', !hasSession);
  dlZip.classList.toggle('disabled', !hasSession);
  dlImg.setAttribute('aria-disabled', String(!hasSession));
  dlZip.setAttribute('aria-disabled', String(!hasSession));

  const mode = session?.backend_mode || config?.agent_mode || 'cpu_rule_fallback';
  const chip = $('#agent-mode-chip');
  chip.textContent = mode === 'gpu_dual_mode' ? '双模式视觉底座' : 'CPU 规则演示模式';
  chip.classList.toggle('gpu', mode === 'gpu_dual_mode');

  if (!hasSession) {
    $('#agent-photo-title').textContent = '载入一张照片开始修图';
    $('#agent-image-meta').textContent = '未载入图像';
    $('#agent-params-count').textContent = '0 项生效';
    $('#agent-params-grid').innerHTML = TOOL_ORDER.map(([, label]) => `<div class="param-pill zero"><span>${label}</span><strong>0</strong></div>`).join('');
    $('#agent-guard-badge').textContent = '待运行';
    $('#agent-guard-badge').className = 'guard-badge';
    ['lum', 'white', 'black', 'warmth'].forEach(k => $(`#metric-${k}`).textContent = '—');
    return;
  }

  $('#agent-photo-title').textContent = session.title;
  $('#agent-image-meta').textContent = `${session.image_size.join(' × ')} · 当前第 ${session.active_turn} 轮`;
  $('#agent-before-img').src = session.source_url;
  $('#agent-after-img').src = session.active_image_url;
  $('#agent-active-tag').textContent = session.active_turn === 0 ? '原图状态' : `第 ${session.active_turn} 轮效果`;
  dlImg.href = `${session.active_image_url}?download=1`;
  dlZip.href = `/api/sessions/${session.session_id}/download`;

  // Timeline
  $('#agent-timeline').innerHTML = (session.history || []).map(h => `
    <button type="button" class="turn-chip ${h.turn === session.active_turn ? 'active' : ''}" data-checkout-turn="${h.turn}">
      <span>${escapeHTML(h.label)}</span>
    </button>
  `).join('');
  document.querySelectorAll('[data-checkout-turn]').forEach(btn => {
    btn.addEventListener('click', () => checkoutAgentTurn({turn: Number(btn.dataset.checkoutTurn)}));
  });

  // Parameters
  const params = session.current_params || {};
  const activeCount = Object.keys(params).length;
  $('#agent-params-count').textContent = `${activeCount} 项生效`;
  $('#agent-params-grid').innerHTML = TOOL_ORDER.map(([key, label]) => {
    const val = params[key] ?? 0;
    const sign = val > 0 ? '+' : '';
    return `<div class="param-pill ${val ? 'active' : 'zero'}"><span>${label}</span><strong>${sign}${val}</strong></div>`;
  }).join('');

  // Telemetry & guardrail
  const activeSnap = (session.history || []).find(h => h.turn === session.active_turn) || session.history?.[0];
  const stats = activeSnap?.stats || session.source_stats || {};
  const critic = activeSnap?.critic || {};
  const badge = $('#agent-guard-badge');
  badge.textContent = critic.passed === false ? '触发风险保护' : (critic.retries > 0 ? `自检微调 ${critic.retries} 次后通过` : '硬指标与自检正常');
  badge.className = `guard-badge ${critic.passed === false ? 'warn' : ''}`;

  $('#metric-lum').textContent = stats.mean_luminance !== undefined ? `${(stats.mean_luminance * 100).toFixed(1)}%` : '—';
  $('#metric-white').textContent = stats.white_clip_fraction !== undefined ? `${(stats.white_clip_fraction * 100).toFixed(2)}%` : '—';
  $('#metric-black').textContent = stats.black_clip_fraction !== undefined ? `${(stats.black_clip_fraction * 100).toFixed(2)}%` : '—';
  $('#metric-warmth').textContent = stats.warmth !== undefined ? `${stats.warmth >= 0 ? '+' : ''}${(stats.warmth * 100).toFixed(1)}` : '—';

  // Chat stream
  const stream = $('#agent-chat-stream');
  stream.innerHTML = (session.messages || []).map(m => {
    const deltaEntries = Object.entries(m.delta || {});
    const deltaHTML = deltaEntries.map(([k, v]) => {
      const label = TOOL_ORDER.find(([id]) => id === k)?.[1] || k;
      return `<span class="delta-tag">${escapeHTML(label)} ${v > 0 ? '+' : ''}${v}</span>`;
    }).join('');
    let criticHTML = '';
    if (m.critic?.passed) {
      criticHTML = `<span class="critic-tag">${m.critic.retries > 0 ? `自检衰减重试 ${m.critic.retries} 次后通过` : '自检通过'}</span>`;
    } else if (m.action === 'rollback' || m.action === 'decline') {
      criticHTML = `<span class="critic-tag warn">已保护性回退/拒修</span>`;
    }
    const badges = (deltaHTML || criticHTML) ? `<div class="bubble-badges">${deltaHTML}${criticHTML}</div>` : '';
    return `<div class="chat-bubble ${m.role === 'user' ? 'user' : 'assistant'}"><p>${escapeHTML(m.text)}</p>${badges}</div>`;
  }).join('');
  if (agentBusy) {
    stream.innerHTML += `<div class="chat-bubble assistant"><p>正在规划调色参数并执行画质自检…</p></div>`;
  }
  stream.scrollTop = stream.scrollHeight;
}

/* ==================== MODE 2: BASELINES COMPARISON ==================== */
function setSource(url, name, size = '') {
  $('#source-preview').src = url;
  $('#source-preview').hidden = false;
  $('#upload-empty').hidden = true;
  $('#upload-overlay').hidden = false;
  $('#image-name').textContent = name;
  $('#image-size').textContent = size;
}

function selectSample(id, loadSaved = true) {
  const sample = config.samples.find(s => s.id === id);
  if (!sample) return;
  sampleId = id;
  upload = null;
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = null;
  $('#image-input').value = '';
  $('#instruction-zh').value = sample.instructions.zh;
  $('#instruction-en').value = sample.instructions.en;
  setSource(sample.image, sample.title, '示例图库');
  job = null;
  feedback = {ratings: {}, preferred: {}, comment: ''};
  signature = '';
  render();
  const saved = jobs.find(j => j.sample_id === id && j.status === 'completed');
  if (loadSaved && saved) loadJob(saved.id);
}

function setUpload(file) {
  if (!file) return;
  if (file.size > 22 * 1024 * 1024) { toast('请选择 22 MB 以内的图片。'); return; }
  if (!file.type.startsWith('image/')) { toast('请选择图片文件。'); return; }
  sampleId = null;
  upload = file;
  job = null;
  signature = '';
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = URL.createObjectURL(file);
  setSource(objectURL, file.name, `${(file.size / 1048576).toFixed(1)} MB`);
  feedback = {ratings: {}, preferred: {}, comment: ''};
  render();
}

function modelOrder() {
  const order = Object.keys(config.models);
  if (!blind) return order;
  let seed = parseInt((job?.id || 'a3f92c11').slice(0, 8), 16);
  for (let i = order.length - 1; i > 0; i--) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const j = seed % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

function nameFor(id) {
  const i = modelOrder().indexOf(id);
  return blind ? `候选 ${String.fromCharCode(65 + i)}` : config.models[id].name;
}

function stageHTML(row, model) {
  const state = row?.status || 'empty';
  if (row?.after) {
    return `<div class="compare-stage"><img src="${row.after}" alt="${escapeHTML(nameFor(model))}结果"><img class="before-layer" src="${job.before}" alt="原图"><div class="compare-divider"></div><input class="compare-slider" type="range" min="0" max="100" value="50" aria-label="对比原图与结果"><span class="stage-tag left">原图</span><span class="stage-tag right">结果</span><button type="button" class="stage-open" data-open="${row.after}" data-title="${escapeHTML(nameFor(model))} · ${row.language === 'zh' ? '中文' : 'English'}">放大 ↗</button></div>`;
  }
  let title = '等待运行', description = '提交后拖动分界线对比前后变化。';
  if (state === 'queued' || state === 'running') { title = state === 'running' ? '正在生成' : '排队中'; description = '正在处理当前原样指令。'; }
  if (state === 'declined') { title = '安全拒修'; description = row.reason === 'unreliable_quality' ? '评估认为存在较高画质风险，未交付成片。' : '需求超出当前全局工具范围。'; }
  if (state === 'failed') { title = '运行未完成'; description = '本次未能通过输出检查。'; }
  return `<div class="compare-stage"><div class="stage-empty"><strong>${title}</strong><span>${description}</span></div></div>`;
}

function outputHTML(row, model, lang) {
  const status = row?.status || 'empty';
  const key = `${model}-${lang}`;
  const timing = row?.seconds !== undefined
    ? (model === 'r21' ? `控制头追加 ${row.seconds.toFixed(1)}s · 复用主干 ${Number(row.parent_seconds || 0).toFixed(1)}s` : `${row.seconds.toFixed(1)}s`)
    : '同一原图 · 原样指令';
  const score = feedback.ratings?.[key] || '';
  const options = [['', '评价这张结果…'], ['completed', '完成要求，画质自然'], ['partial', '部分完成 / 力度略偏'], ['failed', '失败 / 存在明显副作用'], ['uncertain', '暂无法判断']];
  return `${language === 'both' ? `<div class="language-row"><span>${lang === 'zh' ? '中文指令' : 'ENGLISH'}</span><span>${statusNames[status] || '待比较'}</span></div>` : ''}${stageHTML(row, model)}<div class="card-meta"><span>${escapeHTML(timing)}</span>${row?.after ? `<a href="${row.after}?download=1" download>下载 PNG ↓</a>` : '<span>—</span>'}</div>${row?.decision || row?.error ? `<details class="params"><summary>查看${row?.error ? '运行信息' : '决策参数'}</summary><pre>${escapeHTML(row?.error || JSON.stringify(row.decision, null, 2))}</pre></details>` : ''}<div class="card-bottom"><select aria-label="结果评价" data-rating="${key}" ${!row || ['queued', 'running'].includes(status) ? 'disabled' : ''}>${options.map(([v, t]) => `<option value="${v}" ${v === score ? 'selected' : ''}>${t}</option>`).join('')}</select></div>`;
}

function cardHTML(model, index, langs, extraClass = '') {
  const spec = config.models[model];
  const row = job?.results.find(r => r.model === model && r.language === langs[0]);
  const state = row?.status || 'empty';
  const badge = !blind && spec.badge ? `<span class="tier-badge ${spec.tier === 'core' ? 'core' : 'baseline'}">${escapeHTML(spec.badge)}</span>` : '';
  const body = `<div class="outputs-wrap ${extraClass === 'core-card' && langs.length === 2 ? 'bilingual-split' : ''}">${langs.map(l => `<div class="lang-output">${outputHTML(job?.results.find(r => r.model === model && r.language === l), model, l)}</div>`).join('')}</div>`;
  return `<article class="model-card ${extraClass}"><div class="card-heading"><div class="card-title"><span class="model-dot">${blind ? String.fromCharCode(65 + index) : (spec.tier === 'core' ? '★' : String(index))}</span><div><h3>${escapeHTML(nameFor(model))}${badge}</h3><p>${blind ? '盲测模式 · 身份已隐藏' : escapeHTML(spec.subtitle)}</p></div></div><span class="card-status ${['edited', 'unchanged'].includes(state) ? 'good' : ['failed', 'declined'].includes(state) ? 'bad' : ''}">${statusNames[state] || '待比较'}</span></div>${body}</article>`;
}

function render() {
  if (!config) return;
  const totalResults = Object.keys(config.models).length * 2;
  $('#result-title').textContent = job ? job.title : '核心引擎输出与基线对照';
  $('#download-all').classList.toggle('disabled', !job);
  $('#download-all').setAttribute('aria-disabled', String(!job));
  if (job) $('#download-all').href = `/api/jobs/${job.id}/download`; else $('#download-all').removeAttribute('href');
  $('#original-button').disabled = !job && !sampleId && !upload;

  const active = job && ['queued', 'loading', 'running'].includes(job.status);
  $('#progress-box').hidden = !job;
  if (job) {
    $('#progress-message').textContent = job.message;
    $('#progress-count').textContent = `${job.finished || 0} / ${job.results.length}`;
    $('#progress-fill').style.width = `${(job.finished || 0) / job.results.length * 100}%`;
  }
  const dirty = job && (job.instructions.zh !== $('#instruction-zh').value.trim() || job.instructions.en !== $('#instruction-en').value.trim());
  $('#context-note').textContent = job
    ? `${dirty ? '指令已修改；当前仍展示上一组结果。 ' : ''}${job.image_size.join(' × ')} · ${job.id.slice(0, 8)}`
    : `共 ${totalResults} 项双语结果，同图同指令对照检验。`;
  $('#run-button span').textContent = dirty ? '按新要求重新对照' : job ? '重新运行对比' : '运行核心与对照方案';
  $('#run-button').disabled = busy || Boolean(active && !dirty);

  const nextSignature = JSON.stringify([job?.id, job?.results, language, blind]);
  if (signature !== nextSignature) {
    signature = nextSignature;
    const langs = language === 'both' ? ['zh', 'en'] : [language];
    if (blind) {
      $('#model-grid').className = 'model-grid';
      $('#model-grid').innerHTML = modelOrder().map((model, index) => cardHTML(model, index, langs)).join('');
    } else {
      const coreId = config.core_model || Object.keys(config.models)[0];
      const baselineIds = config.baselines || Object.keys(config.models).filter(m => m !== coreId);
      $('#model-grid').className = 'tiered-grid';
      $('#model-grid').innerHTML = `
        <div class="tier-block">
          <div class="tier-header"><span class="eyebrow">CORE PIPELINE · 主力核心方案</span><span class="muted">语义决策锁定 ＋ 幅度与画质风险控制</span></div>
          ${cardHTML(coreId, 0, langs, 'core-card')}
        </div>
        <div class="tier-block">
          <div class="tier-header"><span class="eyebrow">BASELINES · 对照基线</span><span class="muted">用于消融对照与历史演进基准</span></div>
          <div class="model-grid">${baselineIds.map((model, idx) => cardHTML(model, idx + 1, langs)).join('')}</div>
        </div>`;
    }
    document.querySelectorAll('.compare-slider').forEach(slider => slider.addEventListener('input', () => {
      const stage = slider.parentElement;
      stage.querySelector('.before-layer').style.clipPath = `inset(0 ${100 - slider.value}% 0 0)`;
      stage.querySelector('.compare-divider').style.left = `${slider.value}%`;
    }));
    document.querySelectorAll('[data-open]').forEach(b => b.addEventListener('click', () => openImage(b.dataset.open, b.dataset.title)));
    document.querySelectorAll('[data-rating]').forEach(s => s.addEventListener('change', () => {
      feedback.ratings[s.dataset.rating] = s.value;
      $('#feedback-message').textContent = '有未保存的评价';
    }));
  }

  $('#verdict-panel').hidden = !job || !(job.finished > 0);
  for (const l of ['zh', 'en']) {
    const el = $(`#preferred-${l}`);
    const value = feedback.preferred?.[l] || '';
    const opts = [['', '请选择…'], ...modelOrder().map(m => [m, nameFor(m)]), ['tie', '没有明显差别'], ['none', '都不适合交付']];
    el.innerHTML = opts.map(([v, t]) => `<option value="${v}" ${v === value ? 'selected' : ''}>${escapeHTML(t)}</option>`).join('');
  }
}

async function loadHistory() {
  const data = await api('/api/jobs');
  jobs = data.jobs.filter(j => !(j.source === 'agent_smoke_test' && (!j.sample_id || j.status === 'failed')));
  $('#history-list').innerHTML = jobs.length
    ? jobs.slice(0, 30).map(j => `<button type="button" class="history-item ${j.id === job?.id ? 'active' : ''}" data-job="${j.id}"><span>${escapeHTML(j.title)}</span><small>${statusNames[j.status]}</small></button>`).join('')
    : '<p class="muted">暂无历史对比记录</p>';
  document.querySelectorAll('[data-job]').forEach(b => b.addEventListener('click', () => loadJob(b.dataset.job)));
}

async function loadJob(id) {
  try {
    const data = await api(`/api/jobs/${id}`);
    job = data;
    sampleId = data.sample_id || null;
    upload = null;
    feedback = structuredClone(data.feedback || {});
    feedback.ratings ||= {};
    feedback.preferred ||= {};
    $('#feedback-comment').value = feedback.comment || '';
    $('#feedback-message').textContent = data.feedback?.at ? '评价已保存' : '评价数据独立隔离存储。';
    $('#instruction-zh').value = data.instructions.zh;
    $('#instruction-en').value = data.instructions.en;
    setSource(data.before, data.title, data.image_size.join(' × '));
    signature = '';
    render();
    loadHistory().catch(() => {});
  } catch (e) { toast(e.message); }
}

function openImage(url, title) {
  $('#dialog-image').src = url;
  $('#dialog-title').textContent = title;
  $('.dialog-image-wrap').classList.remove('pixel');
  $('#pixel-toggle').textContent = '原始像素 1:1';
  $('#image-dialog').showModal();
}

async function refreshRuntime() {
  const h = await api('/health');
  const pill = $('#connection');
  pill.textContent = h.device === 'cuda' ? 'GPU 服务就绪' : 'CPU 本地就绪';
  pill.classList.remove('offline');
  const left = Math.max(0, (h.budget_gpu_hours || 0) - (h.used_gpu_hours || 0));
  $('#compute-note').textContent = h.device === 'cuda'
    ? `GPU 加速 · 剩余可用预算约 ${(left * 60).toFixed(0)} 分钟`
    : 'CPU 模式 · 对照推理耗时较长；对话预览模式实时可用';
  $('#runtime-detail').textContent = `当前计算精度：${h.precision || 'CPU FP32'}。对话引擎状态：${h.agent_mode === 'gpu_dual_mode' ? '双模式视觉语言底座' : 'CPU 规则演示模式'}。`;
}

/* ==================== EVENT WIRING ==================== */
document.querySelectorAll('.mode-tab').forEach(btn => btn.addEventListener('click', () => switchMode(btn.dataset.mode)));
$('#open-samples-btn').addEventListener('click', () => $('#samples-dialog').showModal());
$('#empty-samples-btn').addEventListener('click', () => $('#samples-dialog').showModal());
$('#agent-samples-inline').addEventListener('click', () => $('#samples-dialog').showModal());
$('#compare-open-samples').addEventListener('click', () => $('#samples-dialog').showModal());
$('#close-samples').addEventListener('click', () => $('#samples-dialog').close());

$('#empty-upload-btn').addEventListener('click', () => $('#agent-image-input').click());
$('#agent-image-input').addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) createAgentSession({file: f});
  e.target.value = '';
});
['dragenter', 'dragover'].forEach(ev => $('#agent-canvas-wrap').addEventListener(ev, e => {
  e.preventDefault(); $('#agent-canvas-wrap').classList.add('dragover');
}));
['dragleave', 'drop'].forEach(ev => $('#agent-canvas-wrap').addEventListener(ev, e => {
  e.preventDefault(); $('#agent-canvas-wrap').classList.remove('dragover');
  if (ev === 'drop' && e.dataTransfer.files[0]) createAgentSession({file: e.dataTransfer.files[0]});
}));

$('#agent-split-slider').addEventListener('input', e => {
  const v = e.target.value;
  $('#agent-before-img').style.clipPath = `inset(0 ${100 - v}% 0 0)`;
  $('#agent-split-divider').style.left = `${v}%`;
});
$('#agent-pixel-open').addEventListener('click', () => {
  if (session) openImage(session.active_image_url, `${session.title} · 第 ${session.active_turn} 轮`);
});
$('#agent-session-select').addEventListener('change', e => loadAgentSession(e.target.value));
$('#agent-undo-btn').addEventListener('click', () => checkoutAgentTurn({action: 'undo'}));
$('#agent-reset-btn').addEventListener('click', () => checkoutAgentTurn({action: 'reset'}));

document.querySelectorAll('[data-prompt]').forEach(btn => btn.addEventListener('click', () => {
  if (!session) { toast('请先上传照片或从示例图库选择一张图片。'); return; }
  sendAgentMessage(btn.dataset.prompt);
}));
$('#agent-chat-form').addEventListener('submit', e => {
  e.preventDefault();
  sendAgentMessage($('#agent-chat-input').value);
});
$('#agent-chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!agentBusy && session) sendAgentMessage($('#agent-chat-input').value);
  }
});

$('#compare-form').addEventListener('submit', async e => {
  e.preventDefault();
  if (busy) return;
  if (!upload && !sampleId && !job) { toast('请先选择一张图片。'); return; }
  busy = true;
  render();
  try {
    const data = new FormData();
    data.set('zh', $('#instruction-zh').value.trim());
    data.set('en', $('#instruction-en').value.trim());
    if (upload) data.set('image', upload);
    else if (sampleId) data.set('sample_id', sampleId);
    else if (job) {
      const r = await fetch(job.before);
      data.set('image', await r.blob(), `${job.title}.png`);
    }
    const newJob = await api('/api/jobs', {method: 'POST', headers: qaHeaders, body: data});
    job = newJob;
    feedback = {ratings: {}, preferred: {}, comment: ''};
    $('#feedback-comment').value = '';
    signature = '';
    await loadHistory();
    toast(`已加入队列，${Object.keys(config.models).length * 2} 项双语结果会陆续显示。`);
  } catch (e) { toast(e.message); }
  finally { busy = false; render(); }
});

$('#image-input').addEventListener('change', e => setUpload(e.target.files[0]));
['dragenter', 'dragover'].forEach(n => $('#upload-zone').addEventListener(n, e => { e.preventDefault(); $('#upload-zone').classList.add('dragover'); }));
['dragleave', 'drop'].forEach(n => $('#upload-zone').addEventListener(n, e => {
  e.preventDefault(); $('#upload-zone').classList.remove('dragover');
  if (n === 'drop') setUpload(e.dataTransfer.files[0]);
}));
document.querySelectorAll('[data-language]').forEach(b => b.addEventListener('click', () => {
  language = b.dataset.language;
  document.querySelectorAll('[data-language]').forEach(x => {
    x.classList.toggle('active', x === b);
    x.setAttribute('aria-selected', String(x === b));
  });
  render();
}));
$('#blind-toggle').addEventListener('change', e => { blind = e.target.checked; render(); });
$('#original-button').addEventListener('click', () => openImage(job?.before || $('#source-preview').src, '编辑前 · 原图'));
$('#close-image').addEventListener('click', () => $('#image-dialog').close());
$('#pixel-toggle').addEventListener('click', () => {
  const on = $('.dialog-image-wrap').classList.toggle('pixel');
  $('#pixel-toggle').textContent = on ? '适应窗口' : '原始像素 1:1';
});
$('#about-button').addEventListener('click', () => $('#about-dialog').showModal());
$('#close-about').addEventListener('click', () => $('#about-dialog').close());
$('#refresh-history').addEventListener('click', () => loadHistory().catch(e => toast(e.message)));
for (const l of ['zh', 'en']) {
  $(`#preferred-${l}`).addEventListener('change', e => { feedback.preferred[l] = e.target.value; $('#feedback-message').textContent = '有未保存的评价'; });
  $(`#instruction-${l}`).addEventListener('input', () => render());
}
$('#feedback-comment').addEventListener('input', e => { feedback.comment = e.target.value; $('#feedback-message').textContent = '有未保存的评价'; });
$('#save-feedback').addEventListener('click', async () => {
  if (!job) return;
  const b = $('#save-feedback');
  b.disabled = true;
  try {
    await api(`/api/jobs/${job.id}/feedback`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json', ...qaHeaders},
      body: JSON.stringify(feedback),
    });
    $('#feedback-message').textContent = '评价已保存（独立隔离存储）';
    toast('评价已保存。');
  } catch (e) { toast(e.message); }
  finally { b.disabled = false; }
});
$('#reset-button').addEventListener('click', () => {
  sampleId = null; upload = null; job = null; signature = '';
  $('#image-input').value = '';
  $('#source-preview').hidden = true; $('#upload-empty').hidden = false; $('#upload-overlay').hidden = true;
  $('#image-name').textContent = '未选择图片'; $('#image-size').textContent = '≤ 22 MB';
  $('#instruction-zh').value = ''; $('#instruction-en').value = '';
  render();
});

async function poll() {
  try {
    if (job && ['queued', 'loading', 'running'].includes(job.status)) {
      const id = job.id, latest = await api(`/api/jobs/${id}`);
      if (job?.id === id) {
        job = latest;
        render();
        if (['completed', 'failed'].includes(job.status)) await loadHistory();
      }
    }
  } catch {
    $('#connection').textContent = '连接中断，正在重试';
    $('#connection').classList.add('offline');
  }
  setTimeout(poll, 1300);
}

async function init() {
  try {
    config = await api('/api/config');
    renderSamplesGallery();
    $('#model-details').innerHTML = Object.values(config.models).map(m =>
      `<div class="about-model"><strong>${escapeHTML(m.name)} · ${escapeHTML(m.subtitle)}</strong><span>${escapeHTML(m.note)}</span></div>`
    ).join('');
    await Promise.all([loadHistory(), refreshSessionsList()]);
    if (sessionsList.length > 0) {
      await loadAgentSession(sessionsList[0].session_id);
    } else {
      renderAgent();
    }
    selectSample('native-dev-011');
    await refreshRuntime();
    poll();
    setInterval(() => {
      refreshRuntime().catch(() => {
        $('#connection').textContent = '服务暂未连接';
        $('#connection').classList.add('offline');
      });
    }, 10000);
  } catch (e) {
    toast(e.message);
    $('#connection').textContent = '服务暂未连接';
    $('#connection').classList.add('offline');
  }
}
init();
