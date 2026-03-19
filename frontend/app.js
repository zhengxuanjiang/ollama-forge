/* Ollama GUI v4.0 — Security + Progress + Step Locking + VEnv */
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const S = {
  chats: JSON.parse(localStorage.getItem('og3-c') || '[]'),
  cid: localStorage.getItem('og3-cid') || null,
  models: [], streaming: false, ac: null, pimg: null,
  pullAC: null, pullName: '', pullPaused: false, prevModel: null,
  selectedPython: '', // selected python path for training
  ftBusy: false, // fine-tuning step lock
  cfg: JSON.parse(localStorage.getItem('og3-s') || JSON.stringify({
    model: '', dev: 'auto', layers: 20, think: false, temp: 0.7, topp: 0.9,
    ctx: 4096, rp: 1.1, seed: -1, sysp: '', showSys: false, ka: '5m'
  }))
};

const PRESETS = {
  assistant: '你是一个有用的AI助手。简洁清晰地回答问题。',
  coder: '你是资深编程专家。提供简洁高效有注释的代码。',
  translator: '你是专业翻译。中英互译，保持风格。',
  writer: '你是专业写作助手。高质量有创意的内容。',
  analyst: '你是数据分析专家。结构化分析，数据驱动。'
};

function sav() { localStorage.setItem('og3-c', JSON.stringify(S.chats)); localStorage.setItem('og3-cid', S.cid || ''); }
function savS() { localStorage.setItem('og3-s', JSON.stringify(S.cfg)); }
let _tt;
function toast(m) {
  let t = $('.toast');
  if (!t) { t = document.createElement('div'); t.className = 'toast'; document.body.appendChild(t); }
  t.textContent = m; t.classList.add('show');
  clearTimeout(_tt); _tt = setTimeout(() => t.classList.remove('show'), 2500);
}
function scrollB() { requestAnimationFrame(() => { $('#chat-area').scrollTop = $('#chat-area').scrollHeight; }); }

// ======================= Improved Markdown Renderer =======================
let _codeId = 0;
function md(raw) {
  if (!raw) return '';
  let t = raw;

  // Phase 1: Extract protected content (code blocks, inline code, math)
  const codeBlocks = [];
  // Handle code blocks with optional language — more permissive regex
  t = t.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const id = _codeId++;
    codeBlocks.push({ id, lang: lang || 'plaintext', code: code.trimEnd() });
    return `\x00CODE${id}\x00`;
  });

  const inlines = [];
  t = t.replace(/`([^`\n]+)`/g, (_, c) => {
    const id = inlines.length; inlines.push(c); return `\x00IL${id}\x00`;
  });

  const mathBlocks = [];
  t = t.replace(/\$\$([\s\S]*?)\$\$/g, (_, m) => {
    const id = mathBlocks.length; mathBlocks.push({ tex: m.trim(), block: true }); return `\x00MATH${id}\x00`;
  });
  t = t.replace(/\$([^\s$][^$]*?[^\s$])\$/g, (_, m) => {
    const id = mathBlocks.length; mathBlocks.push({ tex: m.trim(), block: false }); return `\x00MATH${id}\x00`;
  });
  t = t.replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => {
    const id = mathBlocks.length; mathBlocks.push({ tex: m.trim(), block: true }); return `\x00MATH${id}\x00`;
  });
  t = t.replace(/\\\(([\s\S]*?)\\\)/g, (_, m) => {
    const id = mathBlocks.length; mathBlocks.push({ tex: m.trim(), block: false }); return `\x00MATH${id}\x00`;
  });

  // Phase 2: HTML escape
  t = t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Phase 3: Block-level parsing (process line by line for lists)
  const lines = t.split('\n');
  const result = [];
  let inList = false, listType = 'ul', listItems = [];

  function flushList() {
    if (listItems.length > 0) {
      result.push(`<${listType}>${listItems.map(li => `<li>${li}</li>`).join('')}</${listType}>`);
      listItems = [];
      inList = false;
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Headings
    const h3 = line.match(/^### (.+)$/);
    if (h3) { flushList(); result.push(`<h3>${h3[1]}</h3>`); continue; }
    const h2 = line.match(/^## (.+)$/);
    if (h2) { flushList(); result.push(`<h2>${h2[1]}</h2>`); continue; }
    const h1 = line.match(/^# (.+)$/);
    if (h1) { flushList(); result.push(`<h1>${h1[1]}</h1>`); continue; }

    // HR
    if (/^---+$/.test(line)) { flushList(); result.push('<hr>'); continue; }

    // Blockquote
    const bq = line.match(/^&gt; (.+)$/);
    if (bq) { flushList(); result.push(`<blockquote>${bq[1]}</blockquote>`); continue; }

    // Unordered list items
    const ul = line.match(/^[\-\*] (.+)$/);
    if (ul) {
      if (!inList || listType !== 'ul') { flushList(); inList = true; listType = 'ul'; }
      listItems.push(ul[1]);
      continue;
    }

    // Ordered list items
    const ol = line.match(/^\d+\. (.+)$/);
    if (ol) {
      if (!inList || listType !== 'ol') { flushList(); inList = true; listType = 'ol'; }
      listItems.push(ol[1]);
      continue;
    }

    // Not a list item — flush any pending list
    flushList();

    // Empty line = paragraph break
    if (line.trim() === '') {
      result.push('');
      continue;
    }

    // Code block placeholder on its own line (don't wrap in p)
    if (/^\x00CODE\d+\x00$/.test(line.trim())) {
      result.push(line.trim());
      continue;
    }

    // Regular text
    result.push(line);
  }
  flushList();

  t = result.join('\n');

  // Phase 4: Tables (must be done on joined text)
  t = t.replace(/^(\|.+\|)\n(\|[\s\-:|]+\|)\n((?:\|.+\|\n?)+)/gm, (_, hdr, _sep, body) => {
    const ths = hdr.split('|').filter(Boolean).map(h => `<th>${h.trim()}</th>`).join('');
    const rows = body.trim().split('\n').map(r =>
      '<tr>' + r.split('|').filter(Boolean).map(c => `<td>${c.trim()}</td>`).join('') + '</tr>'
    ).join('');
    return `<table><thead><tr>${ths}</tr></thead><tbody>${rows}</tbody></table>`;
  });

  // Phase 5: Inline formatting
  t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/\*(.+?)\*/g, '<em>$1</em>');
  t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // Phase 6: Paragraph wrapping — group consecutive text lines
  const finalLines = t.split('\n');
  let out = '';
  let pBuf = [];
  function flushP() {
    if (pBuf.length > 0) {
      out += '<p>' + pBuf.join('<br>') + '</p>';
      pBuf = [];
    }
  }
  for (const line of finalLines) {
    if (line === '') { flushP(); continue; }
    if (line.startsWith('<h') || line.startsWith('<table') || line.startsWith('<ul') ||
        line.startsWith('<ol') || line.startsWith('<hr') || line.startsWith('<blockquote') ||
        /^\x00CODE\d+\x00$/.test(line)) {
      flushP();
      out += line;
    } else {
      pBuf.push(line);
    }
  }
  flushP();
  t = out;

  // Phase 7: Restore inline code
  t = t.replace(/\x00IL(\d+)\x00/g, (_, id) => `<code>${esc(inlines[+id])}</code>`);

  // Phase 8: Restore math
  t = t.replace(/\x00MATH(\d+)\x00/g, (_, id) => {
    const m = mathBlocks[+id];
    try {
      return m.block
        ? `<div class="katex-display">${katex.renderToString(m.tex, { displayMode: true, throwOnError: false })}</div>`
        : katex.renderToString(m.tex, { displayMode: false, throwOnError: false });
    } catch (e) { return `<code>${esc(m.tex)}</code>`; }
  });

  // Phase 9: Restore code blocks
  t = t.replace(/\x00CODE(\d+)\x00/g, (_, id) => {
    const b = codeBlocks.find(x => x.id === +id);
    if (!b) return '';
    let highlighted;
    try {
      highlighted = b.lang && b.lang !== 'plaintext' && hljs.getLanguage(b.lang)
        ? hljs.highlight(b.code, { language: b.lang }).value
        : hljs.highlightAuto(b.code).value;
    } catch { highlighted = esc(b.code); }
    const lines = b.code.split('\n').length;
    const needCollapse = lines > 15;
    const uid = 'cb' + b.id;
    return `<div class="code-block"><div class="code-header"><span class="code-lang">${esc(b.lang || 'code')}</span><div class="code-actions">${needCollapse ? `<button class="code-btn" onclick="window._toggleCode('${uid}')" id="tog-${uid}">展开</button>` : ''}<button class="code-btn" onclick="window._cpBlock('${uid}')">复制</button></div></div><div class="code-body${needCollapse ? ' collapsed' : ''}" id="${uid}"><pre><code>${highlighted}</code></pre></div></div>`;
  });

  return t;
}

// Code block actions
window._toggleCode = id => {
  const el = document.getElementById(id); const btn = document.getElementById('tog-' + id);
  if (!el) return;
  if (el.classList.contains('collapsed')) { el.classList.remove('collapsed'); if (btn) btn.textContent = '收起'; }
  else { el.classList.add('collapsed'); if (btn) btn.textContent = '展开'; }
};
window._cpBlock = id => {
  const el = document.getElementById(id); if (!el) return;
  const code = el.querySelector('code');
  if (code) navigator.clipboard.writeText(code.textContent).then(() => toast('已复制'));
};

// ======================= Collapsible user message =======================
function renderUserContent(text, image) {
  let html = '';
  if (image) html += `<img class="c-img" src="data:image/png;base64,${image}">`;
  const escaped = esc(text);
  const lineCount = text.split('\n').length;
  if (lineCount > 6 || text.length > 400) {
    const uid = 'um' + Date.now() + Math.random().toString(36).slice(2, 5);
    html += `<div class="msg-collapse" id="${uid}">${escaped.replace(/\n/g, '<br>')}</div><button class="msg-expand-btn" onclick="window._expandMsg('${uid}',this)">▼ 展开全部 (${lineCount}行)</button>`;
  } else {
    html += escaped.replace(/\n/g, '<br>');
  }
  return html;
}
window._expandMsg = (id, btn) => {
  const el = document.getElementById(id); if (!el) return;
  if (el.classList.contains('msg-collapse')) { el.classList.remove('msg-collapse'); btn.innerHTML = '▲ 收起'; }
  else { el.classList.add('msg-collapse'); btn.innerHTML = '▼ 展开全部'; }
};

// ======================= Chat CRUD =======================
function newChat() {
  const c = { id: Date.now().toString(), title: '新对话', msgs: [], ts: Date.now(), pin: false };
  S.chats.unshift(c); S.cid = c.id; sav(); rList(); rMsgs(); return c;
}
function cur() { return S.chats.find(c => c.id === S.cid); }
function sortC() {
  S.chats.sort((a, b) => {
    if (a.pin && !b.pin) return -1; if (!a.pin && b.pin) return 1; return b.ts - a.ts;
  });
}
function rList() {
  sortC();
  $('#chat-list').innerHTML = S.chats.map(c =>
    `<div class="chat-item${c.id === S.cid ? ' active' : ''}" data-id="${c.id}" onclick="window._sw('${c.id}')" oncontextmenu="window._ctx(event,'${c.id}')">${c.pin ? '<span class="ci-pin">📌</span>' : ''}<span class="ci-text">${esc(c.title)}</span><div class="ci-acts"><button class="btn-i" style="width:22px;height:22px" onclick="event.stopPropagation();window._ctx2('${c.id}',this)">⋯</button></div></div>`
  ).join('');
}
let _ctxId = null;
window._ctx = function(e, id) {
  e.preventDefault(); _ctxId = id; const m = $('#ctx-menu');
  m.style.display = 'block'; m.style.left = Math.min(e.clientX, innerWidth - 160) + 'px';
  m.style.top = Math.min(e.clientY, innerHeight - 130) + 'px';
};
window._ctx2 = function(id, btn) {
  _ctxId = id; const m = $('#ctx-menu'); const r = btn.getBoundingClientRect();
  m.style.display = 'block'; m.style.left = r.left + 'px'; m.style.top = (r.bottom + 4) + 'px';
};
document.addEventListener('click', () => $('#ctx-menu').style.display = 'none');
window._sw = id => { S.cid = id; sav(); rList(); rMsgs(); };

function rMsgs() {
  const c = cur();
  if (!c || !c.msgs.length) {
    $('#welcome').style.display = 'flex'; $('#msgs').innerHTML = ''; $('#gen-stats').style.display = 'none'; return;
  }
  $('#welcome').style.display = 'none';
  $('#msgs').innerHTML = c.msgs.map((m, i) => {
    if (m.role === 'system') return '';
    const u = m.role === 'user';
    let thk = '';
    if (m.thinking) {
      const tid = 't' + i;
      thk = `<div class="thk"><div class="thk-head" onclick="document.getElementById('${tid}').classList.toggle('col')"><span class="thk-lbl">💭 思考过程</span><button class="thk-tog">${m.thinking.length > 200 ? '展开' : '收起'}</button></div><div class="thk-c${m.thinking.length > 200 ? ' col' : ''}" id="${tid}">${md(m.thinking)}</div></div>`;
    }
    const body = u ? renderUserContent(m.content, m.image) : md(m.content);
    return `<div class="msg"><div class="msg-head"><div class="av ${u ? 'av-u' : 'av-a'}">${u ? '👤' : '🦙'}</div><span class="msg-r">${u ? '你' : 'AI'}</span></div><div class="msg-b">${thk}${body}</div></div>`;
  }).join('');
  scrollB();
}

// ======================= Models =======================
async function loadM() {
  try {
    const r = await fetch('/api/tags'); const d = await r.json(); S.models = d.models || [];
    const sel = $('#model-select');
    sel.innerHTML = S.models.length
      ? S.models.map(m => `<option value="${m.name}">${m.name}</option>`).join('')
      : '<option value="">未找到模型</option>';
    if (S.cfg.model && S.models.find(m => m.name === S.cfg.model)) sel.value = S.cfg.model;
    else if (S.models.length) { S.cfg.model = S.models[0].name; sel.value = S.models[0].name; }
    S.prevModel = sel.value;
    $('#conn-dot').className = 'dot on';
    $('#conn-text').textContent = `已连接 · ${S.models.length} 模型`;
  } catch {
    $('#model-select').innerHTML = '<option>连接失败</option>';
    $('#conn-dot').className = 'dot off'; $('#conn-text').textContent = '未连接';
  }
}
function rModelList() {
  if (!S.models.length) { $('#m-list').innerHTML = '<p class="desc">暂无</p>'; return; }
  $('#m-list').innerHTML = S.models.map(m => {
    const sz = m.size ? (m.size / 1e9).toFixed(1) + ' GB' : '';
    const ps = m.details?.parameter_size || '';
    const q = m.details?.quantization_level || '';
    return `<div class="m-item" onclick="this.classList.toggle('exp')"><div class="m-info"><div class="m-name">${esc(m.name)}</div><div class="m-meta">${[ps, q, sz].filter(Boolean).join(' · ')}</div></div><button class="btn-dm" onclick="event.stopPropagation();window._delM('${m.name}')">删除</button><div class="m-detail"><b>大小:</b> ${sz} · <b>参数:</b> ${ps || 'N/A'} · <b>量化:</b> ${q || 'N/A'} · <b>格式:</b> ${m.details?.format || 'N/A'}</div></div>`;
  }).join('');
}
window._delM = async n => {
  if (!confirm(`确定删除 ${n}？`)) return;
  await fetch('/api/delete', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: n }) });
  toast(`已删除 ${n}`); await loadM(); rModelList();
};

// ======================= Pull =======================
async function doPull(name) {
  if (!name) { toast('请输入模型名称'); return; }
  S.pullName = name; S.pullPaused = false;
  $('#pull-prog').style.display = 'block'; $('#pfill').style.width = '0%'; $('#ptxt').textContent = '连接中...';
  $('#btn-pull').disabled = true; $('#btn-pull-pause').style.display = 'inline-flex'; $('#btn-pull-resume').style.display = 'none';
  S.pullAC = new AbortController();
  try {
    const r = await fetch('/api/pull', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }), signal: S.pullAC.signal });
    const rd = r.body.getReader(); const dc = new TextDecoder();
    while (true) {
      const { value, done } = await rd.read(); if (done) break;
      for (const l of dc.decode(value).split('\n').filter(x => x.startsWith('data: '))) {
        const d = l.slice(6); if (d === '[DONE]') continue;
        try {
          const j = JSON.parse(d);
          if (j.error) { $('#ptxt').textContent = '❌ ' + j.error; $('#btn-pull').disabled = false; return; }
          if (j.status) $('#ptxt').textContent = j.status;
          if (j.total && j.completed) {
            const p = Math.round(j.completed / j.total * 100); $('#pfill').style.width = p + '%';
            const mb = s => (s / 1048576).toFixed(1);
            $('#ptxt').textContent = `${j.status} ${mb(j.completed)}/${mb(j.total)} MB (${p}%)`;
          }
          if (j.status === 'success') {
            $('#pfill').style.width = '100%'; $('#ptxt').textContent = '✅ 完成！';
            toast(`${name} 拉取成功`); await loadM(); rModelList();
            setTimeout(() => $('#pull-prog').style.display = 'none', 3000);
          }
        } catch {}
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      if (S.pullPaused) $('#ptxt').textContent = '⏸ 已暂停'; else $('#ptxt').textContent = '已取消';
    } else { $('#ptxt').textContent = '❌ ' + e.message; }
  }
  S.pullAC = null; $('#btn-pull').disabled = false;
}

// ======================= Running models =======================
let _runTimer = null;
async function loadRun() {
  clearInterval(_runTimer);
  try {
    const r = await fetch('/api/ps'); const d = await r.json(); const ms = d.models || [];
    if (!ms.length) { $('#run-list').innerHTML = '<p class="desc">暂无</p>'; return; }
    window._runModels = ms; renderRun(); _runTimer = setInterval(renderRun, 1000);
  } catch { $('#run-list').innerHTML = '<p class="desc">无法获取</p>'; }
}
function renderRun() {
  const ms = window._runModels; if (!ms) return; const now = new Date();
  $('#run-list').innerHTML = ms.map(m => {
    const sz = m.size ? (m.size / 1e9).toFixed(1) + ' GB' : ''; let timerHtml = '';
    if (m.expires_at) {
      const exp = new Date(m.expires_at); const diff = exp - now;
      if (diff <= 0) timerHtml = '<span class="rm-timer">已过期</span>';
      else if (diff > 3.15e10) timerHtml = '<span class="rm-timer forever">♾ 永久</span>';
      else { const min = Math.floor(diff / 60000); const sec = Math.floor((diff % 60000) / 1000); timerHtml = `<span class="rm-timer">⏱ ${min}:${String(sec).padStart(2, '0')}</span>`; }
    }
    return `<div class="rm-item"><div class="rm-left"><span class="rm-name">${esc(m.name)}</span><br><span class="rm-meta">${sz}</span></div>${timerHtml}<button class="btn-ul" onclick="window._ul('${m.name}')">卸载</button></div>`;
  }).join('');
}
window._ul = async n => { await fetch('/api/unload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: n }) }); toast(`${n} 已卸载`); loadRun(); };

// ======================= Model switch dialog =======================
async function checkModelSwitch(nv) {
  try {
    const r = await fetch('/api/ps'); const d = await r.json(); const running = d.models || [];
    const old = running.find(m => m.name === S.prevModel);
    if (old && S.prevModel !== nv) {
      $('#switch-msg').textContent = `模型 "${S.prevModel}" 仍在内存中。是否卸载？`;
      $('#ov-switch').style.display = 'flex';
      return new Promise(res => {
        const cl = () => $('#ov-switch').style.display = 'none';
        $('#sw-unload').onclick = async () => { cl(); await fetch('/api/unload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: S.prevModel }) }); toast(`${S.prevModel} 已卸载`); S.prevModel = nv; res(); };
        $('#sw-keep').onclick = () => { cl(); S.prevModel = nv; res(); };
        $('#cl-switch').onclick = () => { cl(); S.prevModel = nv; res(); };
      });
    }
    S.prevModel = nv;
  } catch { S.prevModel = nv; }
}

// ======================= Export =======================
function doExport(fmt) {
  const c = cur(); if (!c || !c.msgs.length) return toast('无对话可导出');
  let content, ext, mime;
  if (fmt === 'json') { content = JSON.stringify({ title: c.title, messages: c.msgs }, null, 2); ext = '.json'; mime = 'application/json'; }
  else { const lines = [`# ${c.title}\n`]; for (const m of c.msgs) { lines.push(`\n## ${({ user: '👤 用户', assistant: '🦙 AI', system: '⚙️ 系统' })[m.role] || m.role}\n`); if (m.thinking) lines.push(`<details><summary>💭 思考</summary>\n\n${m.thinking}\n</details>\n`); lines.push(`\n${m.content || ''}\n`); } content = lines.join('\n'); ext = '.md'; mime = 'text/markdown'; }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: mime + ';charset=utf-8' }));
  a.download = c.title + ext; document.body.appendChild(a); a.click(); document.body.removeChild(a);
  $('#ov-export').style.display = 'none'; toast('导出成功');
}

// ======================= API Key & Sharing =======================
async function loadShareState() {
  try {
    const r = await fetch('/api/sharing-info'); const d = await r.json();
    $('#sw-sharing').checked = d.sharing;
    $('#share-info').style.display = d.sharing ? 'block' : 'none';
    const dot = $('#api-status-dot');
    if (dot) { dot.classList.toggle('active', d.sharing); }
    if (d.sharing) {
      $('#share-url').textContent = d.base_url;
      $('#share-v1').textContent = d.base_url + '/v1/chat/completions';
      if (d.admin_token) $('#admin-token-display').textContent = d.admin_token;
    }
  } catch {}
}
async function toggleSharing() {
  const en = $('#sw-sharing').checked;
  try {
    await fetch('/api/sharing', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: en }) });
    loadShareState(); toast(en ? '✅ 共享已开启' : '共享已关闭');
  } catch { toast('操作失败'); }
}
async function loadKeys() {
  try {
    const r = await fetch('/api/keys'); const d = await r.json(); const list = $('#keys-list');
    let totalToday = 0, totalAll = 0;
    if (!d.keys.length) {
      list.innerHTML = '<p class="desc" style="text-align:center;padding:20px 0;color:var(--t3)">暂无密钥，在左侧创建第一个 API 密钥</p>';
    } else {
      list.innerHTML = '<table class="api-key-table"><thead><tr><th>名称</th><th>属性</th><th>用量</th><th>操作</th></tr></thead><tbody>' + d.keys.map(k => {
        totalToday += k.today_usage || 0;
        totalAll += k.total_usage || 0;
        const badges = [];
        if (k.rpm > 0) badges.push(`<span class="key-badge">${k.rpm}/分</span>`);
        if (k.daily_limit > 0) {
          const pct = Math.round(k.today_usage / k.daily_limit * 100);
          badges.push(`<span class="key-badge${pct > 80 ? ' warn' : ''}">${k.today_usage}/${k.daily_limit}日</span>`);
        }
        if (k.expires_at) {
          const left = new Date(k.expires_at) - new Date();
          badges.push(`<span class="key-badge${left < 86400000 ? ' err' : ''}">${left > 0 ? Math.ceil(left / 86400000) + '天' : '已过期'}</span>`);
        }
        if (k.allowed_models?.length) badges.push(`<span class="key-badge">${k.allowed_models.length}模型</span>`);
        return `<tr><td style="font-weight:500;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(k.name)}">${esc(k.name)}</td><td><div class="key-badges">${badges.join('') || '<span style="color:var(--t3);font-size:10px">无限制</span>'}</div></td><td style="font-family:var(--fm);font-size:11px;color:var(--t2)">${k.total_usage}次</td><td><button class="btn-revoke" onclick="window._revokeKey('${k.id}')">吊销</button></td></tr>`;
      }).join('') + '</tbody></table>';
    }
    const el = id => $(`#${id}`);
    if (el('api-stat-keys')) el('api-stat-keys').textContent = d.keys.length;
    if (el('api-stat-today')) el('api-stat-today').textContent = totalToday;
    if (el('api-stat-total')) el('api-stat-total').textContent = totalAll;
  } catch {}
}
async function genKey() {
  const name = $('#key-name').value.trim() || 'API Key';
  const rpm = parseInt($('#key-rpm').value) || 0;
  const daily = parseInt($('#key-daily').value) || 0;
  const expDays = parseInt($('#key-expire').value) || 0;
  const modelsStr = $('#key-models').value.trim();
  const allowed_models = modelsStr ? modelsStr.split(',').map(s => s.trim()).filter(Boolean) : [];
  const expires_at = expDays > 0 ? new Date(Date.now() + expDays * 86400000).toISOString() : '';
  try {
    const r = await fetch('/api/keys', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, rpm, daily_limit: daily, allowed_models, expires_at }) });
    const d = await r.json();
    $('#new-key-show').style.display = 'block'; $('#new-key-val').textContent = d.key;
    $('#key-name').value = ''; $('#key-rpm').value = '0'; $('#key-daily').value = '0'; $('#key-expire').value = '0'; $('#key-models').value = '';
    loadKeys(); toast('✅ 密钥已生成');
  } catch { toast('生成失败'); }
}
window._revokeKey = async id => { if (!confirm('确定吊销？')) return; await fetch(`/api/keys/${id}`, { method: 'DELETE' }); toast('已吊销'); loadKeys(); };

function renderAPIExamples() {
  const url = $('#share-url')?.textContent || 'http://YOUR_IP:8765';
  const model = S.cfg.model || 'llama3.2';
  $('#api-examples').innerHTML = `<div class="api-examples-grid">
<div class="api-example-block"><h4><span class="api-lang-dot" style="background:#4ecdc4"></span>cURL</h4><pre>curl ${url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ogk-你的密钥" \\
  -d '{
    "model": "${model}",
    "messages": [{"role":"user","content":"你好"}]
  }'</pre></div>
<div class="api-example-block"><h4><span class="api-lang-dot" style="background:#3b82f6"></span>Python (OpenAI SDK)</h4><pre>from openai import OpenAI

client = OpenAI(
    base_url="${url}/v1",
    api_key="ogk-你的密钥"
)

r = client.chat.completions.create(
    model="${model}",
    messages=[{"role":"user","content":"你好"}]
)
print(r.choices[0].message.content)</pre></div>
<div class="api-example-block"><h4><span class="api-lang-dot" style="background:#f59e0b"></span>JavaScript (fetch)</h4><pre>const res = await fetch("${url}/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer ogk-你的密钥"
  },
  body: JSON.stringify({
    model: "${model}",
    messages: [{ role: "user", content: "你好" }],
    stream: true
  })
});</pre></div>
<div class="api-example-block"><h4><span class="api-lang-dot" style="background:#a78bfa"></span>流式调用 (Python)</h4><pre>r = client.chat.completions.create(
    model="${model}",
    messages=[{"role":"user","content":"你好"}],
    stream=True
)
for chunk in r:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")</pre></div>
</div>
<div class="api-note"><b>🔒 安全机制</b><br>• 管理操作（创建/删除密钥）需要本机访问或管理员令牌<br>• 每个密钥独立的 RPM 速率限制 + 每日配额<br>• 可设密钥过期时间和模型白名单<br>• 完全兼容 OpenAI SDK，支持流式输出</div>`;
}
async function loadAPIModelCount() {
  try {
    const r = await fetch('/api/tags'); const d = await r.json();
    const el = $('#api-stat-models');
    if (el && d.models) el.textContent = d.models.length;
  } catch {}
}
function showAPI() { loadShareState(); loadKeys(); renderAPIExamples(); loadAPIModelCount(); $('#ov-api').style.display = 'flex'; }

// ======================= Send & Stream =======================
async function send() {
  const txt = $('#inp').value.trim(); if (!txt || S.streaming) return;
  const model = $('#model-select').value; if (!model) return toast('请先选择模型');
  let c = cur(); if (!c) c = newChat();
  const um = { role: 'user', content: txt }; if (S.pimg) um.image = S.pimg.b64;
  c.msgs.push(um);
  if (c.msgs.filter(m => m.role === 'user').length === 1) c.title = txt.slice(0, 32) + (txt.length > 32 ? '...' : '');
  c.ts = Date.now(); sav(); rList();
  $('#inp').value = ''; autoR(); S.pimg = null; $('#img-prev').style.display = 'none';
  $('#welcome').style.display = 'none'; rMsgs();
  $('#msgs').insertAdjacentHTML('beforeend', `<div class="msg" id="ai-ld"><div class="msg-head"><div class="av av-a">🦙</div><span class="msg-r">AI</span></div><div class="msg-b"><div class="ldots"><span></span><span></span><span></span></div></div></div>`);
  scrollB();
  let numGpu; if (S.cfg.dev === 'gpu') numGpu = 999; else if (S.cfg.dev === 'cpu') numGpu = 0; else if (S.cfg.dev === 'hybrid') numGpu = S.cfg.layers;
  const apiMsgs = c.msgs.filter(m => m.role !== 'system').map(m => { const r = { role: m.role, content: m.content }; if (m.image) r.images = [m.image]; return r; });
  const payload = { model, messages: apiMsgs, system_prompt: S.cfg.sysp || '', thinking_enabled: S.cfg.think, temperature: S.cfg.temp, top_p: S.cfg.topp, num_ctx: S.cfg.ctx, repeat_penalty: S.cfg.rp, seed: S.cfg.seed, num_gpu: numGpu, keep_alive: S.cfg.ka };
  S.streaming = true; S.ac = new AbortController();
  $('#btn-send').style.display = 'none'; $('#btn-stop').style.display = 'flex';
  let full = '', thk = '', replyTc = 0, thinkTc = 0; const t0 = performance.now();
  $('#gen-stats').style.display = 'flex'; $('#st-think-tok').textContent = '0'; $('#st-tok').textContent = '0'; $('#st-speed').textContent = '0'; $('#st-time').textContent = '0';
  try {
    const resp = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: S.ac.signal });
    const rd = resp.body.getReader(); const dc = new TextDecoder();
    document.getElementById('ai-ld')?.remove();
    const el = document.createElement('div'); el.className = 'msg';
    el.innerHTML = `<div class="msg-head"><div class="av av-a">🦙</div><span class="msg-r">AI</span></div><div class="msg-b"><div class="thk" id="s-thk" style="display:none"><div class="thk-head" onclick="document.getElementById('s-thk-c').classList.toggle('col');"><span class="thk-lbl">💭 思考中...</span><button class="thk-tog">展开/收起</button></div><div class="thk-c" id="s-thk-c"></div></div><div class="streaming" id="s-out"></div></div>`;
    $('#msgs').appendChild(el); scrollB();
    let renderQueued = false;
    function queueRender() {
      if (!renderQueued) { renderQueued = true; requestAnimationFrame(() => {
        if (thk) $('#s-thk-c').innerHTML = md(thk);
        $('#s-out').innerHTML = md(full); scrollB(); renderQueued = false;
      }); }
    }
    while (true) {
      const { value, done } = await rd.read(); if (done) break;
      for (const l of dc.decode(value).split('\n').filter(x => x.startsWith('data: '))) {
        const d = l.slice(6); if (d === '[DONE]') continue;
        try {
          const j = JSON.parse(d);
          if (j.error) { full += `\n\n**错误:** ${j.error}`; queueRender(); continue; }
          const m = j.message; if (!m) continue;
          if (m.thinking) { thk += m.thinking; thinkTc++; $('#s-thk').style.display = 'block'; }
          if (m.content) { full += m.content; replyTc++; }
          if (j.eval_count) replyTc = j.eval_count;
          const elapsed = (performance.now() - t0) / 1000; const totalTc = thinkTc + replyTc;
          $('#st-think-tok').textContent = thinkTc; $('#st-tok').textContent = replyTc;
          $('#st-time').textContent = elapsed.toFixed(1);
          $('#st-speed').textContent = totalTc > 0 ? (totalTc / elapsed).toFixed(1) : '0';
          queueRender();
        } catch {}
      }
    }
  } catch (e) { if (e.name === 'AbortError') full += '\n\n*（已停止）*'; else full += `\n\n**错误:** ${e.message}`; }
  const elapsed = (performance.now() - t0) / 1000; const totalTc = thinkTc + replyTc;
  $('#st-time').textContent = elapsed.toFixed(1); $('#st-speed').textContent = totalTc > 0 ? (totalTc / elapsed).toFixed(1) : '0';
  S.streaming = false; S.ac = null; $('#btn-send').style.display = 'flex'; $('#btn-stop').style.display = 'none';
  const sOut = $('#s-out'); if (sOut) { sOut.classList.remove('streaming'); sOut.innerHTML = md(full); }
  if (thk) {
    const tc = $('#s-thk-c'); if (tc) { tc.innerHTML = md(thk); tc.classList.add('col'); }
    const lbl = $('#s-thk .thk-lbl'); if (lbl) lbl.textContent = `💭 思考 (${thinkTc} tokens)`;
  }
  const ai = { role: 'assistant', content: full }; if (thk) ai.thinking = thk; c.msgs.push(ai); sav();
}
function autoR() { const i = $('#inp'); i.style.height = 'auto'; i.style.height = Math.min(i.scrollHeight, 140) + 'px'; }

// ======================= Settings Sync =======================
function syncUI() {
  const s = S.cfg;
  const dr = document.querySelector(`input[name="dev"][value="${s.dev}"]`); if (dr) dr.checked = true; updDev();
  $('#sw-think').checked = s.think; $('#chip-think').classList.toggle('on', s.think);
  $('#sys-text').value = s.sysp; $('#chip-sys').classList.toggle('on', s.showSys);
  $('#sys-panel').style.display = s.showSys ? 'block' : 'none';
  $('#r-temp').value = s.temp; $('#v-temp').textContent = s.temp;
  $('#r-topp').value = s.topp; $('#v-topp').textContent = s.topp;
  $('#r-ctx').value = s.ctx; $('#v-ctx').textContent = s.ctx;
  $('#r-layers').value = s.layers; $('#v-layers').textContent = s.layers;
  $('#r-rp').value = s.rp; $('#v-rp').textContent = s.rp;
  $('#r-seed').value = s.seed; $('#v-seed').textContent = s.seed < 0 ? '随机' : s.seed;
  const ka = document.querySelector(`input[name="ka"][value="${s.ka}"]`); if (ka) ka.checked = true;
}
function updDev() {
  const d = S.cfg.dev;
  $('#hybrid-cfg').style.display = d === 'hybrid' ? 'block' : 'none';
  $('#dev-badge').className = 'dev-badge ' + d;
  $('#dev-label').textContent = { auto: '自动', gpu: 'GPU', cpu: 'CPU', hybrid: '混合' }[d] || d;
}

// ======================= Fine-Tuning System =======================
const FMT_EXAMPLES = {
  alpaca: `[\n  {\n    "instruction": "翻译以下句子为英文",\n    "input": "今天天气真好",\n    "output": "The weather is really nice today."\n  }\n]`,
  sharegpt: `[\n  {\n    "conversations": [\n      {"from": "human", "value": "如何学习Python？"},\n      {"from": "gpt", "value": "学习Python可以按以下路径：..."}\n    ]\n  }\n]`,
  openai: `[\n  {\n    "messages": [\n      {"role": "system", "content": "你是一个专业的客服助手"},\n      {"role": "user", "content": "我的订单什么时候到？"},\n      {"role": "assistant", "content": "请提供您的订单号。"}\n    ]\n  }\n]`
};

// Dataset links with dual options
const DS_LINKS = [
  { name: "📘 Alpaca 52K", desc: "通用指令 · Alpaca格式 · 52K条", url: "https://huggingface.co/datasets/tatsu-lab/alpaca" },
  { name: "📗 Alpaca-GPT4", desc: "GPT-4生成 · 高质量 · 52K条", url: "https://huggingface.co/datasets/vicgalle/alpaca-gpt4" },
  { name: "📙 Alpaca Cleaned", desc: "去噪版本 · 推荐 · 52K条", url: "https://huggingface.co/datasets/yahma/alpaca-cleaned" },
  { name: "📕 FineTome 100K", desc: "ShareGPT · 高质量筛选", url: "https://huggingface.co/datasets/mlabonne/FineTome-100k" },
  { name: "🐋 OpenOrca", desc: "推理增强 · 多格式 · 4M条", url: "https://huggingface.co/datasets/Open-Orca/OpenOrca" },
  { name: "💻 Code-Feedback", desc: "代码对话 · 66K条", url: "https://huggingface.co/datasets/m-a-p/Code-Feedback" },
  { name: "🏥 Medical QA (中文)", desc: "中文医疗 · 800K条", url: "https://huggingface.co/datasets/shibing624/medical" },
  { name: "🇨🇳 Alpaca-GPT4 中文", desc: "中文指令 · 48K条", url: "https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-chinese" },
];

function renderDSLinks() {
  $('#ft-ds-links').innerHTML = DS_LINKS.map((ds, i) =>
    `<div class="ft-ds-link-wrap"><div class="ft-ds-link-info"><b>${ds.name}</b><span>${ds.desc}</span></div><div class="ft-ds-link-actions"><a href="${ds.url}" target="_blank" class="btn-ds-action">🔗 官网</a><button class="btn-ds-action btn-ds-dl" onclick="downloadDS(${i})" ${S.ftBusy ? 'disabled' : ''}>📥 下载</button></div></div>`
  ).join('');
}

async function downloadDS(idx) {
  if (S.ftBusy) return toast('有操作正在进行中');
  const ds = DS_LINKS[idx]; const name = ds.name.replace(/[^\w\u4e00-\u9fff]/g, '') + '.jsonl';
  _doDownloadDS({ url: ds.url, filename: name });
}

async function downloadCustomDS() {
  if (S.ftBusy) return toast('有操作正在进行中');
  const input = $('#ds-hf-input').value.trim();
  if (!input) return toast('请输入数据集 ID 或链接');

  // Determine if it's an ID or URL
  let hf_id = '', url = '';
  if (input.startsWith('http')) {
    url = input;
  } else {
    hf_id = input;
  }
  // Generate filename from ID
  const safeName = (hf_id || input.split('/datasets/').pop() || 'dataset')
    .replace(/[^a-zA-Z0-9\u4e00-\u9fff_-]/g, '_').substring(0, 60) + '.jsonl';
  _doDownloadDS({ url, hf_id, filename: safeName });
}

async function _doDownloadDS(params) {
  S.ftBusy = true;
  showOp(`📥 正在下载数据集...`);
  $('#ds-download-progress').style.display = 'block'; $('#dl-pfill').style.width = '0%';
  $('#dl-log').innerHTML = '<div>📥 连接中...</div>';
  renderDSLinks();
  try {
    const r = await fetch('/api/finetune/download-dataset', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (!r.ok) {
      const errData = await r.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${r.status}`);
    }
    await readSSE(r, j => {
      if (j.progress != null) $('#dl-pfill').style.width = j.progress + '%';
      if (j.step) { updateOp(j.step); $('#dl-log').innerHTML += `<div>${esc(j.step)}</div>`; }
      $('#dl-log').scrollTop = $('#dl-log').scrollHeight;
      if (j.done) { toast('✅ 下载完成'); loadDatasets(); loadDatasetsForTrain(); }
      if (j.error) toast('❌ ' + j.step);
    });
  } catch (e) {
    toast('❌ ' + e.message);
    $('#dl-log').innerHTML += `<div style="color:var(--err)">❌ ${esc(e.message)}</div>`;
  }
  S.ftBusy = false; renderDSLinks(); hideOp();
  setTimeout(() => { $('#ds-download-progress').style.display = 'none'; }, 5000);
}

let _ftMonTimer = null;
async function loadHardware() {
  const el = $('#ft-hw-info'); el.innerHTML = '<p class="desc">🔍 正在检测硬件...</p>';
  try {
    const [hwRes, cfgRes] = await Promise.all([fetch('/api/hardware'), fetch('/api/config')]);
    const d = await hwRes.json();
    const cfg = await cfgRes.json();
    const hw = d.hardware; const recs = d.recommendations;
    const gpuHtml = hw.gpus.length
      ? hw.gpus.map(g => `<div class="ft-hw-card"><span class="val">${g.name}</span><span class="lbl">${g.vram_total_mb ? Math.round(g.vram_total_mb / 1024) + 'GB VRAM' : '检测中'}</span></div>`).join('')
      : '<div class="ft-hw-card"><span class="val">未检测到</span><span class="lbl">NVIDIA GPU</span></div>';
    el.innerHTML = `<div class="ft-hw-grid">${gpuHtml}<div class="ft-hw-card"><span class="val">${hw.ram_gb || '?'}GB</span><span class="lbl">系统内存</span></div><div class="ft-hw-card"><span class="val">${hw.disk_free_gb || '?'}GB</span><span class="lbl">可用磁盘</span></div><div class="ft-hw-card"><span class="val">${hw.cuda ? '✅ 可用' : '❌ 不可用'}</span><span class="lbl">CUDA</span></div>${hw.apple_silicon ? `<div class="ft-hw-card"><span class="val">Apple Silicon</span><span class="lbl">${hw.unified_memory_gb || '?'}GB 统一内存</span></div>` : ''}</div>`;

    // Populate path inputs
    $('#cfg-ft-dir').value = cfg.ft_dir || '';
    $('#cfg-hf-home').value = cfg.hf_home || '';

    // Show file structure detail
    const p = cfg.paths || {};
    $('#ft-paths-detail').innerHTML = `
      <div><b>📁 微调数据根目录:</b> <code>${esc(cfg.ft_dir)}</code></div>
      <div>├── <b>datasets/</b> — 数据集 <code>${esc(p.datasets||'')}</code></div>
      <div>├── <b>outputs/</b> — 训练 checkpoint <code>${esc(p.outputs||'')}</code></div>
      <div>├── <b>merged_model/</b> — 合并后的最终模型 (safetensors) <code>${esc(p.merged_model||'')}</code></div>
      <div>├── <b>imports/</b> — 导入的 GGUF 文件 <code>${esc(p.imports||'')}</code></div>
      <div>├── <b>venv/</b> — 虚拟环境 <code>${esc(p.venv||'')}</code></div>
      <div>└── <b>train_script.py</b> — 生成的训练脚本</div>
      <div style="margin-top:6px"><b>🤗 HuggingFace 模型缓存:</b> <code>${esc(p.hf_cache||'~/.cache/huggingface')}</code></div>
      <div style="color:var(--t3);margin-top:4px;font-size:10px">💡 基座模型首次下载后会缓存在 HF 缓存目录，后续训练无需重新下载</div>
    `;

    $('#ft-recs').innerHTML = recs.map(r => `<div class="ft-rec"><h4>${r.title}</h4><p>${r.desc}</p><div class="ft-rec-meta"><span>显存要求: ${r.vram_need}</span><span>难度: ${r.difficulty}</span><span>耗时: ${r.time}</span></div><div class="ft-rec-models">${r.models.map(m => `<span class="ft-rec-tag">${m}</span>`).join('')}</div></div>`).join('');
  } catch (e) { el.innerHTML = `<p class="desc">❌ 检测失败: ${e.message}</p>`; }
}

async function savePaths() {
  const msg = $('#cfg-save-msg');
  const ftDir = $('#cfg-ft-dir').value.trim();
  const hfHome = $('#cfg-hf-home').value.trim();
  if (!ftDir) { msg.style.display = 'block'; msg.innerHTML = '⚠️ 微调数据目录不能为空'; return; }
  msg.style.display = 'block'; msg.innerHTML = '⏳ 保存中...';
  try {
    const body = { ft_dir: ftDir };
    if (hfHome) body.hf_home = hfHome;
    const r = await fetch('/api/config', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) { msg.innerHTML = `❌ ${d.detail || '保存失败'}`; return; }
    msg.innerHTML = `✅ 已保存！${d.needs_restart ? '⚠️ HF 缓存路径修改需要重启生效' : ''}`;
    loadHardware(); // refresh
  } catch (e) { msg.innerHTML = `❌ ${e.message}`; }
}

// ======================= Model Catalog =======================
const MODEL_CATALOG = [
  // Meta Llama
  {id:"meta-llama/Llama-3.2-1B-Instruct",   name:"Llama 3.2 1B Instruct",   vendor:"meta",      params:1,   vram:6,  gated:true},
  {id:"meta-llama/Llama-3.2-3B-Instruct",   name:"Llama 3.2 3B Instruct",   vendor:"meta",      params:3,   vram:8,  gated:true},
  {id:"meta-llama/Llama-3.1-8B-Instruct",   name:"Llama 3.1 8B Instruct",   vendor:"meta",      params:8,   vram:16, gated:true},
  {id:"meta-llama/Llama-3.3-70B-Instruct",  name:"Llama 3.3 70B Instruct",  vendor:"meta",      params:70,  vram:48, gated:true},
  // Qwen (全部开放)
  {id:"Qwen/Qwen2.5-0.5B-Instruct",        name:"Qwen 2.5 0.5B Instruct",  vendor:"qwen",      params:0.5, vram:4,  gated:false},
  {id:"Qwen/Qwen2.5-1.5B-Instruct",        name:"Qwen 2.5 1.5B Instruct",  vendor:"qwen",      params:1.5, vram:6,  gated:false},
  {id:"Qwen/Qwen2.5-3B-Instruct",          name:"Qwen 2.5 3B Instruct",    vendor:"qwen",      params:3,   vram:8,  gated:false},
  {id:"Qwen/Qwen2.5-7B-Instruct",          name:"Qwen 2.5 7B Instruct",    vendor:"qwen",      params:7,   vram:16, gated:false},
  {id:"Qwen/Qwen2.5-14B-Instruct",         name:"Qwen 2.5 14B Instruct",   vendor:"qwen",      params:14,  vram:24, gated:false},
  {id:"Qwen/Qwen2.5-32B-Instruct",         name:"Qwen 2.5 32B Instruct",   vendor:"qwen",      params:32,  vram:48, gated:false},
  {id:"Qwen/Qwen2.5-72B-Instruct",         name:"Qwen 2.5 72B Instruct",   vendor:"qwen",      params:72,  vram:80, gated:false},
  {id:"Qwen/Qwen2.5-Coder-7B-Instruct",    name:"Qwen 2.5 Coder 7B",       vendor:"qwen",      params:7,   vram:16, gated:false},
  // Google Gemma
  {id:"google/gemma-2-2b-it",              name:"Gemma 2 2B Instruct",      vendor:"google",    params:2,   vram:6,  gated:true},
  {id:"google/gemma-2-9b-it",              name:"Gemma 2 9B Instruct",      vendor:"google",    params:9,   vram:16, gated:true},
  {id:"google/gemma-2-27b-it",             name:"Gemma 2 27B Instruct",     vendor:"google",    params:27,  vram:40, gated:true},
  // Microsoft Phi
  {id:"microsoft/Phi-3-mini-4k-instruct",  name:"Phi-3 Mini 3.8B",          vendor:"microsoft", params:3.8, vram:8,  gated:false},
  {id:"microsoft/Phi-3.5-mini-instruct",   name:"Phi-3.5 Mini 3.8B",        vendor:"microsoft", params:3.8, vram:8,  gated:false},
  {id:"microsoft/phi-4",                   name:"Phi-4 14B",                vendor:"microsoft", params:14,  vram:24, gated:false},
  // Mistral
  {id:"mistralai/Mistral-7B-Instruct-v0.3",name:"Mistral 7B v0.3",          vendor:"mistral",   params:7,   vram:16, gated:false},
  {id:"mistralai/Mixtral-8x7B-Instruct-v0.1",name:"Mixtral 8x7B",           vendor:"mistral",   params:47,  vram:48, gated:false},
  // DeepSeek
  {id:"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",name:"DeepSeek R1 1.5B (Qwen蒸馏)",vendor:"deepseek",params:1.5,vram:6, gated:false},
  {id:"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  name:"DeepSeek R1 7B (Qwen蒸馏)",  vendor:"deepseek",params:7,  vram:16,gated:false},
  {id:"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  name:"DeepSeek R1 14B (Qwen蒸馏)", vendor:"deepseek",params:14, vram:24,gated:false},
  {id:"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  name:"DeepSeek R1 8B (Llama蒸馏)", vendor:"deepseek",params:8,  vram:16,gated:false},
  // Yi
  {id:"01-ai/Yi-1.5-6B-Chat",              name:"Yi 1.5 6B Chat",           vendor:"01ai",      params:6,   vram:12, gated:false},
  {id:"01-ai/Yi-1.5-9B-Chat",              name:"Yi 1.5 9B Chat",           vendor:"01ai",      params:9,   vram:16, gated:false},
  // TinyLlama / Small models
  {id:"TinyLlama/TinyLlama-1.1B-Chat-v1.0",name:"TinyLlama 1.1B Chat",     vendor:"other",     params:1.1, vram:4,  gated:false},
  {id:"stabilityai/stablelm-zephyr-3b",    name:"StableLM Zephyr 3B",       vendor:"other",     params:3,   vram:8,  gated:false},
];

const VENDOR_NAMES = {meta:'Meta',qwen:'阿里 Qwen',google:'Google',microsoft:'Microsoft',mistral:'Mistral',deepseek:'DeepSeek','01ai':'零一万物',other:'其他'};

function filterModels() {
  const v = $('#mb-vendor').value;
  const s = $('#mb-size').value;
  const a = $('#mb-access').value;
  const q = ($('#mb-search').value || '').toLowerCase();
  return MODEL_CATALOG.filter(m => {
    if (v && m.vendor !== v) return false;
    if (s === '3' && m.params > 3) return false;
    if (s === '8' && m.params > 8) return false;
    if (s === '14' && m.params > 14) return false;
    if (s === '72' && m.params <= 14) return false;
    if (a === 'open' && m.gated) return false;
    if (a === 'gated' && !m.gated) return false;
    if (q && !m.name.toLowerCase().includes(q) && !m.id.toLowerCase().includes(q)) return false;
    return true;
  });
}

function renderModelBrowser() {
  const models = filterModels();
  const sel = $('#tr-base').value;
  const list = $('#mb-list');
  if (!models.length) {
    list.innerHTML = '<div class="mb-empty">没有符合条件的模型</div>';
  } else {
    list.innerHTML = models.map(m =>
      `<div class="mb-item${m.id === sel ? ' selected' : ''}" data-id="${esc(m.id)}" onclick="selectModel('${esc(m.id)}')">` +
        `<span class="mb-name">${esc(m.name)}</span>` +
        `<div class="mb-tags">` +
          `<span class="mb-tag size">${m.params}B</span>` +
          `<span class="mb-tag vram">≈${m.vram}GB</span>` +
          `<span class="mb-tag ${m.gated ? 'gated' : 'open'}">${m.gated ? '🔒 受限' : '🟢 开放'}</span>` +
        `</div>` +
      `</div>`
    ).join('');
  }
  updateSelectedDisplay();
}

function selectModel(id) {
  $('#tr-base').value = id;
  renderModelBrowser();
  updateSelectedDisplay();
}

function updateSelectedDisplay() {
  const sel = $('#tr-base').value.trim();
  const m = MODEL_CATALOG.find(x => x.id === sel);
  const el = $('#mb-selected');
  if (m) {
    el.innerHTML = `<span class="mb-sel-tag">✅ ${esc(m.name)} · ${m.params}B · ≈${m.vram}GB${m.gated ? ' · <span style="color:#fbbf24">🔒 需登录</span>' : ''}</span>`;
  } else if (sel) {
    el.innerHTML = `<span class="mb-sel-tag">📦 自定义: ${esc(sel)}</span>`;
  } else {
    el.innerHTML = '';
  }
}

function initModelBrowser() {
  ['mb-vendor','mb-size','mb-access'].forEach(id => {
    $(`#${id}`).onchange = renderModelBrowser;
  });
  $('#mb-search').oninput = renderModelBrowser;
  // Toggle browser expand/collapse
  $('#btn-mb-toggle').onclick = () => {
    const browser = $('#model-browser');
    const btn = $('#btn-mb-toggle');
    browser.classList.toggle('collapsed');
    btn.textContent = browser.classList.contains('collapsed') ? '📋 展开推荐模型列表' : '📋 收起推荐模型列表';
    if (!browser.classList.contains('collapsed')) renderModelBrowser();
  };
  // Sync manual input to display
  $('#tr-base').oninput = () => {
    updateSelectedDisplay();
    // Highlight matching catalog item if any
    const val = $('#tr-base').value.trim();
    $$('.mb-item').forEach(el => el.classList.toggle('selected', el.dataset.id === val));
  };
  renderModelBrowser();
  updateSelectedDisplay();
}

// ======================= HuggingFace Login =======================
async function checkHFStatus() {
  const el = $('#hf-status');
  try {
    const python_path = S.selectedPython || '';
    const r = await fetch('/api/hf/status', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ python_path: python_path || undefined })
    });
    const d = await r.json();
    if (d.logged_in) {
      el.className = 'ft-hf-status logged-in';
      el.innerHTML = `<span class="hf-ok-icon">✅</span> 已登录: <span class="hf-user">${esc(d.username)}</span> ${d.fullname ? `(${esc(d.fullname)})` : ''} <button class="btn-s" id="btn-hf-logout" style="margin-left:8px;font-size:10px;padding:2px 8px">退出登录</button>`;
      $('#hf-login-form').style.display = 'none';
      $('#btn-hf-logout').onclick = hfLogout;
    } else {
      el.className = 'ft-hf-status';
      el.innerHTML = '⚠️ 未登录 — 受限模型 (Llama, Gemma) 无法下载';
      $('#hf-login-form').style.display = 'block';
    }
  } catch (e) {
    el.className = 'ft-hf-status';
    el.innerHTML = '❓ 无法检测登录状态';
    $('#hf-login-form').style.display = 'block';
  }
}

async function hfLogin() {
  const token = $('#hf-token').value.trim();
  const msg = $('#hf-login-msg');
  if (!token) { msg.style.display = 'block'; msg.innerHTML = '⚠️ 请输入 Token'; return; }
  if (!token.startsWith('hf_')) { msg.style.display = 'block'; msg.innerHTML = '⚠️ Token 应以 hf_ 开头'; return; }
  msg.style.display = 'block'; msg.innerHTML = '⏳ 登录中...';
  try {
    const python_path = S.selectedPython || '';
    const r = await fetch('/api/hf/login', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({token, python_path}) });
    const d = await r.json();
    if (d.status === 'ok') {
      msg.innerHTML = `✅ 登录成功！用户: ${esc(d.username)}`;
      $('#hf-token').value = '';
      setTimeout(() => { checkHFStatus(); msg.style.display = 'none'; }, 1500);
    } else {
      msg.innerHTML = `❌ 登录失败: ${esc(d.error || '未知错误')}`;
    }
  } catch (e) { msg.innerHTML = `❌ ${e.message}`; }
}

async function hfLogout() {
  if (!confirm('确定退出 HuggingFace 登录？')) return;
  try {
    const python_path = S.selectedPython || '';
    await fetch('/api/hf/logout', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({python_path}) });
    checkHFStatus();
  } catch (e) { toast('退出失败: ' + e.message); }
}

function showFT() {
  $('#ov-ft').style.display = 'flex';
  document.body.style.overflow = 'hidden';
  loadHardware(); loadDatasets(); showFmtExample('alpaca'); renderDSLinks(); initModelBrowser();
}

function switchFTTab(tab) {
  $$('.ft-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  $$('.ft-pane').forEach(p => p.classList.toggle('active', p.id === 'ft-' + tab));
  if (tab === 'train') { startMonitor(); loadProjects(); }
  if (tab === 'env') { loadEnvs(); checkHFStatus(); }
  if (tab === 'history') loadHistory();
}

function selectMethod(m) {
  $$('.ft-mcard').forEach(c => c.classList.toggle('selected', c.dataset.m === m));
  $$('.ft-sub').forEach(s => s.style.display = 'none');
  if (m === 'modelfile') {
    $('#ft-sub-modelfile').style.display = 'block';
    const sel = $('#mf-base');
    sel.innerHTML = S.models.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
  } else if (m === 'qlora' || m === 'lora') {
    $('#ft-sub-train').style.display = 'block';
    $('#ft-train-title').textContent = m === 'qlora' ? '🧬 QLoRA 4-bit 微调配置' : '🔬 LoRA 16-bit 微调配置';
    checkTrainDeps(); loadDatasetsForTrain(); loadEnvsForSelect();
  } else if (m === 'import') {
    $('#ft-sub-import').style.display = 'block';
  }
}

// ======================= Environment Management =======================
async function loadEnvs() {
  const el = $('#env-list'); el.innerHTML = '<p class="desc">🔍 检测中...</p>';
  try {
    const r = await fetch('/api/venv/detect', { method: 'POST' }); const d = await r.json();
    if (!d.envs.length) { el.innerHTML = '<p class="desc">未找到环境</p>'; return; }
    el.innerHTML = d.envs.map(e =>
      `<div class="ft-env-item${S.selectedPython === e.path ? ' selected' : ''}" onclick="selectEnv('${esc(e.path)}')" data-path="${esc(e.path)}"><div class="ft-env-info"><b>${esc(e.name)}</b><span>${esc(e.version)} · ${esc(e.type)}</span></div><code class="ft-env-path">${esc(e.path)}</code></div>`
    ).join('');
  } catch (e) { el.innerHTML = `<p class="desc">❌ ${e.message}</p>`; }
}
window.selectEnv = function(path) {
  S.selectedPython = path;
  $$('.ft-env-item').forEach(el => el.classList.toggle('selected', el.dataset.path === path));
  toast(`✅ 已选择: ${path}`);
};

async function loadEnvsForSelect() {
  try {
    const r = await fetch('/api/venv/detect', { method: 'POST' }); const d = await r.json();
    const sel = $('#tr-python');
    sel.innerHTML = '<option value="">系统 Python</option>' +
      d.envs.map(e => `<option value="${esc(e.path)}" ${S.selectedPython === e.path ? 'selected' : ''}>${esc(e.name)} (${e.version})</option>`).join('');
  } catch {}
}

// ======================= Operation Progress Helpers =======================
function showOp(text) {
  const b = $('#ft-op-banner'); if (b) { b.style.display = 'block'; $('#ft-op-text').textContent = text; }
}
function updateOp(text) { const t = $('#ft-op-text'); if (t) t.textContent = text; }
function hideOp() { const b = $('#ft-op-banner'); if (b) b.style.display = 'none'; }
function btnLoad(id, loading, origText) {
  const b = $(id); if (!b) return;
  if (loading) { b.dataset.origText = b.textContent; b.classList.add('loading'); b.disabled = true; }
  else { b.classList.remove('loading'); b.disabled = false; b.textContent = origText || b.dataset.origText || b.textContent; }
}
// Helper: read SSE stream and call handler for each parsed JSON event
async function readSSE(response, onEvent) {
  const rd = response.body.getReader(); const dc = new TextDecoder(); let buf = '';
  while (true) {
    const { value, done } = await rd.read(); if (done) break;
    buf += dc.decode(value, { stream: true });
    const lines = buf.split('\n'); buf = lines.pop() || '';
    for (const l of lines) {
      if (!l.startsWith('data: ')) continue;
      const d = l.slice(6).trim(); if (d === '[DONE]') continue;
      try { onEvent(JSON.parse(d)); } catch {}
    }
  }
  // Process any remaining buffer
  if (buf.startsWith('data: ')) {
    const d = buf.slice(6).trim(); if (d && d !== '[DONE]') {
      try { onEvent(JSON.parse(d)); } catch {}
    }
  }
}

async function createVenv() {
  if (S.ftBusy) return toast('有操作正在进行中');
  S.ftBusy = true;
  const path = $('#venv-path').value.trim();
  const body = {}; if (path) body.path = path;
  btnLoad('#btn-create-venv', true);
  showOp('📦 正在创建虚拟环境并安装依赖...');
  $('#venv-progress').style.display = 'block'; $('#venv-pfill').style.width = '0%';
  $('#venv-log').innerHTML = '<div>⏳ 正在连接服务器...</div>';
  try {
    const r = await fetch('/api/venv/create', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    await readSSE(r, j => {
      if (j.progress) $('#venv-pfill').style.width = j.progress + '%';
      updateOp(j.step || '安装中...');
      $('#venv-log').innerHTML += `<div>${esc(j.step)}</div>`;
      $('#venv-log').scrollTop = $('#venv-log').scrollHeight;
      if (j.done) {
        toast('✅ 环境创建完成');
        if (j.python_path) S.selectedPython = j.python_path;
        loadEnvs();
      }
      if (j.error) toast('❌ 创建失败: ' + j.step);
    });
  } catch (e) { toast('❌ ' + e.message); $('#venv-log').innerHTML += `<div style="color:var(--err)">❌ ${esc(e.message)}</div>`; }
  S.ftBusy = false; btnLoad('#btn-create-venv', false, '📦 创建并安装依赖'); hideOp();
}

// ======================= Training Deps & Execution =======================
async function checkTrainDeps() {
  const el = $('#ft-deps-check'); el.className = 'ft-deps'; el.textContent = '🔍 检查训练依赖...';
  const python_path = $('#tr-python')?.value || S.selectedPython || '';
  try {
    const r = await fetch('/api/finetune/check-deps', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ python_path: python_path || undefined })
    });
    const d = await r.json();
    const miss = Object.entries(d.deps).filter(([k, v]) => !v).map(([k]) => k);
    if (d.all_ok) { el.className = 'ft-deps ok'; el.textContent = '✅ 所有依赖已安装'; }
    else { el.className = 'ft-deps miss'; el.innerHTML = `⚠️ 缺少: ${miss.join(', ')} <button class="btn-p" onclick="installDeps()" style="margin-left:8px;padding:3px 10px;font-size:10px" ${S.ftBusy ? 'disabled' : ''}>一键安装</button>`; }
    // GPU / PyTorch CUDA check
    const gpuEl = $('#ft-gpu-check');
    gpuEl.style.display = 'block';
    if (d.gpu_ok) {
      gpuEl.className = 'ft-deps ok';
      gpuEl.textContent = d.gpu_msg;
    } else if (d.fix_cmd) {
      gpuEl.className = 'ft-deps miss';
      gpuEl.innerHTML = `⚠️ ${esc(d.gpu_msg)}<br>` +
        `<small style="color:var(--t3)">检测到 CUDA ${esc(d.system_cuda)}，但 PyTorch 是 CPU 版本</small><br>` +
        `<button class="btn-p" id="btn-fix-torch" onclick="fixTorch()" style="margin-top:6px;padding:4px 14px;font-size:11px;background:var(--warn);color:var(--bg0)" ${S.ftBusy ? 'disabled' : ''}>🔧 修复 PyTorch（安装 GPU 版）</button>` +
        `<br><small style="color:var(--t3);margin-top:4px;display:inline-block">或手动: <code style="background:var(--bg0);padding:1px 4px;border-radius:3px">${esc(d.fix_cmd)}</code></small>`;
    } else {
      gpuEl.className = 'ft-deps miss';
      gpuEl.innerHTML = `⚠️ ${esc(d.gpu_msg)}`;
    }
    // Hide progress areas when re-checking
    $('#deps-install-progress').style.display = 'none';
    $('#gpu-fix-progress').style.display = 'none';
  } catch (e) { el.className = 'ft-deps miss'; el.textContent = '⚠️ 检查失败: ' + e.message; }
}

async function fixTorch() {
  if (S.ftBusy) return toast('有操作正在进行中');
  S.ftBusy = true;
  const gpuEl = $('#ft-gpu-check');
  gpuEl.className = 'ft-deps'; gpuEl.innerHTML = '<span class="ft-op-spinner" style="display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:6px"></span> 🔧 正在修复 PyTorch...';
  showOp('🔧 正在卸载 CPU 版 PyTorch 并安装 GPU 版...');
  const prog = $('#gpu-fix-progress'); prog.style.display = 'block';
  $('#gpu-fix-pfill').style.width = '0%'; $('#gpu-fix-log').innerHTML = '<div>⏳ 正在连接...</div>';
  const python_path = $('#tr-python')?.value || S.selectedPython || '';
  try {
    const r = await fetch('/api/finetune/fix-torch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ python_path: python_path || undefined })
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await readSSE(r, j => {
      updateOp(j.step || '修复中...');
      if (j.progress) $('#gpu-fix-pfill').style.width = j.progress + '%';
      $('#gpu-fix-log').innerHTML += `<div>${esc(j.step)}</div>`;
      $('#gpu-fix-log').scrollTop = $('#gpu-fix-log').scrollHeight;
      gpuEl.innerHTML = '<span class="ft-op-spinner" style="display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:6px"></span> ' + esc(j.step);
      if (j.error) { gpuEl.className = 'ft-deps miss'; }
      if (j.done) { S.ftBusy = false; hideOp(); checkTrainDeps(); }
    });
  } catch (e) { gpuEl.className = 'ft-deps miss'; gpuEl.textContent = '❌ 修复失败: ' + e.message; $('#gpu-fix-log').innerHTML += `<div style="color:var(--err)">❌ ${esc(e.message)}</div>`; }
  S.ftBusy = false; hideOp();
}

async function installDeps() {
  if (S.ftBusy) return toast('有操作正在进行中');
  S.ftBusy = true;
  const el = $('#ft-deps-check');
  el.className = 'ft-deps';
  el.innerHTML = '<span class="ft-op-spinner" style="display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:6px"></span> 📦 安装依赖中...';
  showOp('📦 正在安装训练依赖 (torch, peft, trl...)...');
  const prog = $('#deps-install-progress'); prog.style.display = 'block';
  $('#deps-pfill').style.width = '0%'; $('#deps-log').innerHTML = '<div>⏳ 正在连接...</div>';
  const python_path = $('#tr-python')?.value || S.selectedPython || '';
  try {
    const r = await fetch('/api/finetune/install-deps', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ python_path: python_path || undefined })
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    await readSSE(r, j => {
      if (j.progress) $('#deps-pfill').style.width = j.progress + '%';
      updateOp(j.step || '安装中...');
      el.innerHTML = '<span class="ft-op-spinner" style="display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:6px"></span> ' + esc(j.step);
      $('#deps-log').innerHTML += `<div>${esc(j.step)}</div>`;
      $('#deps-log').scrollTop = $('#deps-log').scrollHeight;
      if (j.done) { toast('✅ 依赖安装完成'); checkTrainDeps(); }
    });
  } catch (e) { el.className = 'ft-deps miss'; el.textContent = '❌ 安装失败: ' + e.message; $('#deps-log').innerHTML += `<div style="color:var(--err)">❌ ${esc(e.message)}</div>`; }
  S.ftBusy = false; hideOp();
}

async function createModelfile() {
  if (S.ftBusy) return toast('有操作正在进行中');
  const base = $('#mf-base').value; const name = $('#mf-name').value.trim();
  const sys = $('#mf-sys').value;
  if (!base || !name) return toast('请填写基础模型和名称');
  S.ftBusy = true;
  const exLines = $('#mf-examples').value.trim().split('\n').filter(Boolean);
  const examples = []; for (const l of exLines) { try { examples.push(JSON.parse(l)); } catch {} }
  const params = {};
  if ($('#mf-temp').value) params.temperature = $('#mf-temp').value;
  if ($('#mf-topp').value) params.top_p = $('#mf-topp').value;
  if ($('#mf-topk').value) params.top_k = $('#mf-topk').value;
  if ($('#mf-rp').value) params.repeat_penalty = $('#mf-rp').value;
  const res = $('#mf-result'); res.style.display = 'block'; res.className = 'ft-result'; res.textContent = '⏳ 创建中...';
  try {
    const r = await fetch('/api/finetune/modelfile', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ base_model: base, name, system_prompt: sys, parameters: params, examples }) });
    const d = await r.json();
    if (d.status === 'ok') { res.className = 'ft-result ok'; res.innerHTML = `✅ 模型 <b>${d.model_name}</b> 创建成功！`; toast('✅ 模型创建成功'); await loadM(); }
    else { res.className = 'ft-result err'; res.textContent = '❌ ' + d.error; }
  } catch (e) { res.className = 'ft-result err'; res.textContent = '❌ ' + e.message; }
  S.ftBusy = false;
}

// ======================= Upload with progress =======================
async function uploadDataset() {
  if (S.ftBusy) return toast('有操作正在进行中');
  const file = $('#ds-file').files[0]; if (!file) return toast('请选择文件');
  S.ftBusy = true; btnLoad('#btn-upload-ds', true);
  showOp(`📤 正在上传数据集: ${file.name}...`);
  const progEl = $('#ds-upload-progress');
  progEl.style.display = 'block'; $('#ds-pfill').style.width = '0%'; $('#ds-ptxt').textContent = '上传中...';

  const fd = new FormData(); fd.append('file', file);
  try {
    const xhr = new XMLHttpRequest();
    await new Promise((resolve, reject) => {
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) {
          const pct = Math.round(e.loaded / e.total * 100);
          $('#ds-pfill').style.width = pct + '%';
          const txt = `上传中... ${(e.loaded / 1024).toFixed(0)}/${(e.total / 1024).toFixed(0)} KB (${pct}%)`;
          $('#ds-ptxt').textContent = txt;
          updateOp('📤 ' + txt);
        }
      };
      xhr.onload = () => {
        if (xhr.status === 200) {
          const d = JSON.parse(xhr.responseText);
          if (d.errors?.length) toast('⚠️ ' + d.errors[0]); else toast('✅ 数据集上传成功');
          const pv = $('#ds-preview'); pv.style.display = 'block';
          pv.innerHTML = `<h4>📊 ${d.filename} — ${d.total}条 · 格式: ${d.detected_format}</h4>${d.errors.length ? `<p style="color:var(--err)">⚠️ ${d.errors.join('; ')}</p>` : ''}${d.preview.length ? `<pre>${esc(JSON.stringify(d.preview[0], null, 2))}</pre>` : ''}`;
          loadDatasets(); loadDatasetsForTrain();
          resolve();
        } else { reject(new Error('HTTP ' + xhr.status)); }
      };
      xhr.onerror = () => reject(new Error('网络错误'));
      xhr.open('POST', '/api/finetune/upload-dataset');
      xhr.send(fd);
    });
  } catch (e) { toast('❌ 上传失败: ' + e.message); }
  S.ftBusy = false; btnLoad('#btn-upload-ds', false, '上传'); hideOp();
  setTimeout(() => { progEl.style.display = 'none'; }, 2000);
}

async function loadDatasets() {
  try {
    const r = await fetch('/api/finetune/datasets'); const d = await r.json();
    S._dsInfo = d.datasets; // cache for format auto-detection
    const el = $('#ds-list');
    if (!d.datasets.length) { el.innerHTML = '<p class="desc">暂无</p>'; return; }
    const fmtLabels = { alpaca: '📘 Alpaca', sharegpt: '📗 ShareGPT', openai: '📙 OpenAI', text: '📄 Text', csv: '📊 CSV', other: '📎 其他', unknown: '❓ 未知' };
    const fmtColors = { alpaca: '#3b82f6', sharegpt: '#10b981', openai: '#f59e0b', text: '#8b5cf6', csv: '#06b6d4', other: '#6b7280', unknown: '#9ca3af' };
    el.innerHTML = d.datasets.map(f => {
      const fmt = f.format || 'unknown';
      const fmtLabel = fmtLabels[fmt] || fmt;
      const fmtColor = fmtColors[fmt] || '#6b7280';
      const sizeStr = f.size > 1024*1024 ? (f.size/1024/1024).toFixed(1)+'MB' : (f.size/1024).toFixed(1)+'KB';
      const rowsStr = f.rows ? f.rows.toLocaleString() + ' 条' : '';
      const colsStr = f.columns ? f.columns.join(', ') : '';
      let sampleHtml = '';
      if (f.sample) {
        const s = esc(f.sample.length > 200 ? f.sample.substring(0, 200) + '...' : f.sample);
        sampleHtml = `<div class="ds-sample" style="margin-top:4px;padding:4px 6px;background:var(--bg);border-radius:4px;font-size:10px;color:var(--t3);font-family:var(--fm);word-break:break-all;max-height:48px;overflow:hidden;cursor:pointer" title="点击展开/收起" onclick="this.style.maxHeight=this.style.maxHeight==='48px'?'none':'48px'">${s}</div>`;
      }
      return `<div class="ds-info-card" style="padding:8px 10px;margin-bottom:6px;border:1px solid var(--bd);border-radius:8px;background:var(--bg2)">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
          <div style="flex:1;min-width:0">
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
              <span style="font-weight:600;font-size:12px">${esc(f.name)}</span>
              <span style="font-size:10px;padding:1px 6px;border-radius:10px;color:#fff;background:${fmtColor}">${fmtLabel}</span>
            </div>
            <div style="font-size:10px;color:var(--t3);margin-top:2px">${sizeStr}${rowsStr ? ' · ' + rowsStr : ''}${colsStr ? ' · 字段: ' + colsStr : ''}</div>
          </div>
          <button class="btn-revoke" onclick="deleteDS('${esc(f.name)}')" style="flex-shrink:0">删除</button>
        </div>${sampleHtml}
      </div>`;
    }).join('');
  } catch {}
}
async function deleteDS(name) { if (!confirm('删除？')) return; await fetch(`/api/finetune/datasets/${name}`, { method: 'DELETE' }); loadDatasets(); loadDatasetsForTrain(); toast('已删除'); }
async function loadDatasetsForTrain() {
  try {
    const r = await fetch('/api/finetune/datasets'); const d = await r.json();
    S._dsInfo = d.datasets; // cache for format auto-detection
    const fmtMap = { alpaca: 'Alpaca', sharegpt: 'ShareGPT', openai: 'OpenAI' };
    $('#tr-dataset').innerHTML = d.datasets.length
      ? d.datasets.map(f => {
          const fmtHint = fmtMap[f.format] ? ` [${fmtMap[f.format]}]` : '';
          const rowHint = f.rows ? ` (${f.rows.toLocaleString()}条)` : '';
          return `<option value="${f.name}" data-fmt="${f.format || ''}">${f.name}${fmtHint}${rowHint}</option>`;
        }).join('')
      : '<option value="">请先上传数据集</option>';
    // Auto-select format for current selection
    _autoSelectDatasetFormat();
  } catch {}
}
function _autoSelectDatasetFormat() {
  const sel = $('#tr-dataset');
  const opt = sel.options[sel.selectedIndex];
  if (!opt) return;
  const fmt = opt.getAttribute('data-fmt');
  if (fmt === 'alpaca' || fmt === 'sharegpt' || fmt === 'openai') {
    $('#tr-fmt').value = fmt;
  }
}
function showFmtExample(fmt) {
  $$('.ft-fmt').forEach(b => b.classList.toggle('active', b.dataset.f === fmt));
  $('#fmt-example').textContent = FMT_EXAMPLES[fmt] || '';
}

async function startTraining() {
  if (S.ftBusy) return toast('有操作正在进行中');
  const base = $('#tr-base').value.trim();
  if (!base) return toast('请先选择或输入基座模型');
  if (!base.includes('/')) return toast('模型格式应为 "组织/模型名"，如 Qwen/Qwen2.5-1.5B-Instruct');
  const ds = $('#tr-dataset').value; if (!ds) return toast('请先上传数据集');
  S.ftBusy = true;
  btnLoad('#btn-start-train', true);
  showOp('🚀 正在启动训练...');
  const method = $('.ft-mcard.selected')?.dataset?.m || 'qlora';
  const python_path = $('#tr-python')?.value || S.selectedPython || '';
  const cfg = {
    base_model: $('#tr-base').value.trim(), dataset_path: ds, dataset_format: $('#tr-fmt').value,
    output_name: $('#tr-name').value || 'my-finetune', method,
    lora_r: parseInt($('#tr-r').value) || 16, lora_alpha: parseInt($('#tr-alpha').value) || 16,
    epochs: parseInt($('#tr-epochs').value) || 3, batch_size: parseInt($('#tr-bs').value) || 2,
    learning_rate: parseFloat($('#tr-lr').value) || 2e-4, max_seq_length: parseInt($('#tr-seq').value) || 2048,
    quant_method: $('#tr-quant').value || 'q8_0', export_ollama: $('#tr-ollama').value === '1',
    python_path: python_path || undefined
  };
  try {
    const r = await fetch('/api/finetune/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg) });
    const d = await r.json();
    if (d.status === 'started') {
      toast('🚀 训练已启动'); $('#train-progress').style.display = 'block';
      updateOp('🔄 训练进行中...');
      switchFTTab('train'); startMonitor();
      btnLoad('#btn-start-train', false, '🚀 开始训练');
    } else { toast('❌ ' + JSON.stringify(d)); S.ftBusy = false; btnLoad('#btn-start-train', false, '🚀 开始训练'); hideOp(); }
  } catch (e) { toast('❌ ' + e.message); S.ftBusy = false; btnLoad('#btn-start-train', false, '🚀 开始训练'); hideOp(); }
}

function startMonitor() {
  clearInterval(_ftMonTimer);
  _ftMonTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/finetune/status'); const d = await r.json();
      const card = $('#ft-status-card');
      const icons = { idle: '⏸', training: '🔄', completed: '✅', failed: '❌', stopped: '⏹' };
      card.className = 'ft-status-card' + (d.status === 'training' ? ' running' : '');
      card.querySelector('.ft-st-icon').textContent = icons[d.status] || '?';
      $('#ft-st-text').textContent = { idle: '空闲', training: '训练中...', completed: '已完成', failed: '失败', stopped: '已停止' }[d.status] || d.status;
      $('#ft-mon-fill').style.width = d.progress + '%';
      $('#ft-mon-pct').textContent = d.progress;
      $('#ft-mon-loss').textContent = d.last_loss != null ? d.last_loss.toFixed(4) : '-';
      const logEl = $('#ft-mon-log');
      logEl.innerHTML = d.logs.map(l => {
        let cls = '';
        if (l.includes('STEP:')) cls = 'log-step';
        if (l.includes('ERROR') || l.includes('❌')) cls = 'log-err';
        if (l.includes('WARN')) cls = 'log-warn';
        return `<div class="${cls}">${esc(l)}</div>`;
      }).join('');
      logEl.scrollTop = logEl.scrollHeight;
      const tp = $('#train-progress'); if (tp.style.display !== 'none') {
        $('#tr-pfill').style.width = d.progress + '%';
        const tl = $('#train-log'); tl.innerHTML = logEl.innerHTML; tl.scrollTop = tl.scrollHeight;
      }
      $('#btn-mon-stop').style.display = d.status === 'training' ? 'inline-flex' : 'none';
      if (d.status === 'completed') { clearInterval(_ftMonTimer); toast('🎉 训练完成！'); S.ftBusy = false; hideOp(); await loadM(); }
      else if (d.status === 'failed' || d.status === 'stopped') { clearInterval(_ftMonTimer); S.ftBusy = false; hideOp(); if (d.error) toast('⚠️ ' + d.error); }
    } catch {}
  }, 2000);
}

async function importGGUF() {
  if (S.ftBusy) return toast('有操作正在进行中');
  const file = $('#gguf-file').files[0]; if (!file) return toast('请选择GGUF文件');
  S.ftBusy = true;
  btnLoad('#btn-import-gguf', true);
  showOp(`📦 正在导入 GGUF 模型: ${file.name}...`);
  const name = $('#gguf-name').value.trim() || 'imported-model';
  const sys = $('#gguf-sys').value;
  const res = $('#import-result'); res.style.display = 'block'; res.className = 'ft-result'; res.textContent = '⏳ 导入中...';
  const progEl = $('#gguf-upload-progress'); progEl.style.display = 'block';

  const fd = new FormData(); fd.append('file', file); fd.append('name', name); fd.append('system_prompt', sys);
  try {
    const xhr = new XMLHttpRequest();
    await new Promise((resolve, reject) => {
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) {
          const pct = Math.round(e.loaded / e.total * 100);
          $('#gguf-pfill').style.width = pct + '%';
          const txt = `上传中... ${(e.loaded / 1e6).toFixed(1)}/${(e.total / 1e6).toFixed(1)} MB (${pct}%)`;
          $('#gguf-ptxt').textContent = txt;
          updateOp('📦 ' + txt);
        }
      };
      xhr.onload = () => {
        const d = JSON.parse(xhr.responseText);
        if (d.status === 'ok') { res.className = 'ft-result ok'; res.textContent = `✅ 模型 ${d.model_name} 导入成功 (${(d.size / 1e9).toFixed(2)}GB)`; toast('✅ 导入成功'); loadM(); }
        else { res.className = 'ft-result err'; res.textContent = '❌ ' + d.error; }
        resolve();
      };
      xhr.onerror = () => reject(new Error('网络错误'));
      xhr.open('POST', '/api/finetune/import-gguf'); xhr.send(fd);
    });
  } catch (e) { res.className = 'ft-result err'; res.textContent = '❌ ' + e.message; }
  S.ftBusy = false; btnLoad('#btn-import-gguf', false, '📦 导入模型'); hideOp(); progEl.style.display = 'none';
}

// ======================= Training History =======================
async function loadHistory() {
  const el = $('#ft-history-list');
  try {
    const r = await fetch('/api/finetune/history'); const d = await r.json();
    if (!d.records || !d.records.length) { el.innerHTML = '<p class="desc">暂无训练记录</p>'; return; }
    el.innerHTML = d.records.map(r => {
      const dur = r.duration_seconds ? (r.duration_seconds >= 3600
        ? `${Math.floor(r.duration_seconds/3600)}h${Math.floor((r.duration_seconds%3600)/60)}m`
        : `${Math.floor(r.duration_seconds/60)}m${r.duration_seconds%60}s`) : '-';
      const ts = r.timestamp ? new Date(r.timestamp).toLocaleString('zh-CN') : '';
      return `<div class="hist-item">
        <div class="hist-head"><b>${esc(r.output_name||'-')}</b><span class="hist-time">${ts}</span></div>
        <div class="hist-meta">
          <div class="hist-kv"><span class="hk">基座模型</span><span class="hv">${esc(r.base_model||'-')}</span></div>
          <div class="hist-kv"><span class="hk">方式</span><span class="hv">${esc(r.method||'-')}</span></div>
          <div class="hist-kv"><span class="hk">数据集</span><span class="hv">${esc(r.dataset||'-')}</span></div>
          <div class="hist-kv"><span class="hk">训练时长</span><span class="hv">${dur}</span></div>
          <div class="hist-kv"><span class="hk">最终 Loss</span><span class="hv">${r.final_loss != null ? Number(r.final_loss).toFixed(4) : '-'}</span></div>
          <div class="hist-kv"><span class="hk">总步数</span><span class="hv">${r.total_steps || '-'}</span></div>
          <div class="hist-kv"><span class="hk">Epochs</span><span class="hv">${r.epochs||'-'}</span></div>
          <div class="hist-kv"><span class="hk">LR</span><span class="hv">${r.learning_rate||'-'}</span></div>
          <div class="hist-kv"><span class="hk">LoRA r/α</span><span class="hv">${r.lora_r||'-'}/${r.lora_alpha||'-'}</span></div>
          <div class="hist-kv"><span class="hk">序列长度</span><span class="hv">${r.max_seq_length||'-'}</span></div>
          <div class="hist-kv"><span class="hk">Ollama注册</span><span class="hv">${r.ollama_registered ? '✅' : '❌'}</span></div>
          <div class="hist-kv"><span class="hk">断点续训</span><span class="hv">${r.resumed ? '是' : '否'}</span></div>
        </div>
        <div class="hist-actions">
          <button class="btn-s" onclick="navigator.clipboard.writeText('${esc(r.output_path||'')}');toast('已复制路径')">📋 复制输出路径</button>
          <button class="btn-s ws-btn-del" onclick="deleteHistRecord('${esc(r.id)}')">🗑️ 删除记录</button>
        </div>
      </div>`;
    }).join('');
  } catch (e) { el.innerHTML = `<p class="desc">❌ ${e.message}</p>`; }
}
window.deleteHistRecord = async function(id) {
  if (!confirm('删除此训练记录？')) return;
  await fetch(`/api/finetune/history/${id}`, { method: 'DELETE' });
  loadHistory(); toast('已删除');
};

// ======================= Project-Based Checkpoint Management =======================
async function loadProjects() {
  const el = $('#project-list');
  try {
    const r = await fetch('/api/finetune/projects'); const d = await r.json();
    if (!d.projects || !d.projects.length) {
      el.innerHTML = '<p class="desc">暂无训练工程（开始训练后自动创建）</p>'; return;
    }
    el.innerHTML = d.projects.map(p => {
      const status = {'completed':'✅','training':'🔄','failed':'❌','stopped':'⏹'}[p.status] || '📦';
      return `<div class="proj-item" data-name="${esc(p.dir_name)}">
        <div class="proj-head" onclick="toggleProject('${esc(p.dir_name)}')">
          <div class="proj-info">
            <b>${status} ${esc(p.label || p.name)}</b>
            <span>${esc(p.base_model||'')} · ${esc(p.dataset||'')} · ${p.method||''} · ${p.checkpoint_count||0} checkpoints</span>
          </div>
          <div class="proj-actions">
            <button class="btn-s" onclick="event.stopPropagation();renameProject('${esc(p.dir_name)}')" title="重命名">✏️</button>
            <button class="btn-s ws-btn-del" onclick="event.stopPropagation();deleteProject('${esc(p.dir_name)}')" title="删除">🗑️</button>
          </div>
        </div>
        <div class="proj-cps" id="proj-cps-${esc(p.dir_name)}" style="display:none"><p class="desc">加载中...</p></div>
      </div>`;
    }).join('');
  } catch (e) { el.innerHTML = `<p class="desc">❌ ${e.message}</p>`; }
}

window.toggleProject = async function(name) {
  const el = $(`#proj-cps-${name}`);
  if (el.style.display !== 'none') { el.style.display = 'none'; return; }
  el.style.display = 'block';
  el.innerHTML = '<p class="desc">🔍 加载 Checkpoints...</p>';
  try {
    const r = await fetch(`/api/finetune/projects/${encodeURIComponent(name)}/checkpoints`);
    const d = await r.json();
    if (!d.checkpoints || !d.checkpoints.length) {
      el.innerHTML = '<p class="desc">暂无 Checkpoint</p>'; return;
    }
    el.innerHTML = d.checkpoints.map(cp =>
      `<div class="cp-item">
        <div class="cp-info">
          <b>${esc(cp.name)}</b>
          <span>Step ${cp.step} · ${cp.size_mb}MB${cp.loss != null ? ' · Loss: '+Number(cp.loss).toFixed(4) : ''}${cp.epoch != null ? ' · Epoch: '+Number(cp.epoch).toFixed(1) : ''}</span>
        </div>
        <button class="btn-p" style="padding:4px 14px;font-size:10px;flex-shrink:0" onclick="resumeProject('${esc(name)}','${esc(cp.name)}')">▶ 从此继续训练</button>
      </div>`
    ).join('');
  } catch (e) { el.innerHTML = `<p class="desc">❌ ${e.message}</p>`; }
};

window.resumeProject = async function(project, checkpoint) {
  if (!confirm(`从 ${checkpoint} 断点恢复训练？\n\n将使用原始训练配置继续训练。`)) return;
  const python_path = S.selectedPython || '';
  try {
    const r = await fetch(`/api/finetune/projects/${encodeURIComponent(project)}/resume`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ checkpoint, python_path: python_path || undefined })
    });
    const d = await r.json();
    if (d.status === 'started') {
      toast('🚀 断点续训已启动！');
      switchFTTab('train');
      startMonitor();
    } else {
      toast('❌ ' + (d.detail || d.error || JSON.stringify(d)));
    }
  } catch (e) { toast('❌ ' + e.message); }
};

window.renameProject = async function(name) {
  const newName = prompt('新工程名称:', name);
  if (!newName || newName === name) return;
  try {
    const r = await fetch(`/api/finetune/projects/${encodeURIComponent(name)}/rename`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ new_name: newName })
    });
    const d = await r.json();
    if (d.status === 'ok') { toast('✅ 已重命名'); loadProjects(); }
    else toast('❌ ' + (d.detail || '重命名失败'));
  } catch (e) { toast('❌ ' + e.message); }
};

window.deleteProject = async function(name) {
  if (!confirm(`确定删除工程 ${name}？包含所有 Checkpoints，不可撤销。`)) return;
  try {
    await fetch(`/api/finetune/projects/${encodeURIComponent(name)}`, { method: 'DELETE' });
    toast('已删除'); loadProjects();
  } catch (e) { toast('❌ ' + e.message); }
};

// ======================= Model Workshop =======================
let _wsActiveModel = '';

function showWorkshop() {
  $('#ov-workshop').style.display = 'flex';
  document.body.style.overflow = 'hidden';
  wsLoadModels();
}

function switchWSTab(tab) {
  $$('.ws-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  $$('.ws-pane').forEach(p => p.classList.toggle('active', p.id === tab));
  if (tab === 'ws-quant' || tab === 'ws-variant' || tab === 'ws-test') wsPopulateSelects();
}

async function wsHFImport() {
  const hfModel = $('#ws-hf-model').value.trim();
  const ollamaName = $('#ws-hf-name').value.trim();
  const qtype = document.querySelector('input[name="hf-qtype"]:checked')?.value || 'q4_k_m';
  if (!hfModel) return toast('请输入 HuggingFace 模型 ID');
  if (!ollamaName) return toast('请输入 Ollama 模型名');
  if (!hfModel.includes('/')) return toast('模型 ID 格式应为 "组织/模型名"');

  const resEl = $('#ws-hf-result'); resEl.style.display = 'block';
  const logEl = $('#ws-hf-log'); logEl.innerHTML = '';
  const fillEl = $('#ws-hf-fill'); fillEl.style.width = '5%';
  $('#btn-ws-hf-import').disabled = true;

  try {
    const python_path = S.selectedPython || '';
    const r = await fetch('/api/workshop/import-hf', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ hf_model: hfModel, ollama_name: ollamaName, quant_type: qtype, python_path: python_path || undefined })
    });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const d = JSON.parse(line.slice(6));
          if (d.step) logEl.innerHTML += `<div>${esc(d.step)}</div>`;
          if (d.progress) fillEl.style.width = d.progress + '%';
          if (d.done) { toast('✅ 导入完成！'); wsLoadModels(); loadM(); }
          if (d.error) toast('❌ 导入失败');
          logEl.scrollTop = logEl.scrollHeight;
        } catch {}
      }
    }
  } catch (e) {
    logEl.innerHTML += `<div style="color:var(--err)">❌ ${esc(e.message)}</div>`;
  }
  $('#btn-ws-hf-import').disabled = false;
}

async function wsLoadModels() {
  const el = $('#ws-model-list');
  el.innerHTML = '<p class="desc">🔍 加载中...</p>';
  try {
    const r = await fetch('/api/workshop/models'); const d = await r.json();
    if (d.error) { el.innerHTML = `<p class="desc">❌ ${esc(d.error)}</p>`; return; }
    if (!d.models.length) { el.innerHTML = '<p class="desc">暂无已安装模型</p>'; return; }
    el.innerHTML = d.models.map(m =>
      `<div class="ws-m-item${_wsActiveModel===m.name?' active':''}" data-name="${esc(m.name)}" onclick="wsSelectModel('${esc(m.name)}')">
        <div class="ws-m-info"><b>${esc(m.name)}</b><span>${esc(m.size)} · ${esc(m.modified)}</span></div>
        <div class="ws-m-actions">
          <button onclick="event.stopPropagation();wsExportMF('${esc(m.name)}')" title="导出 Modelfile">📄</button>
          <button onclick="event.stopPropagation();wsCopyModel('${esc(m.name)}')" title="复制">📋</button>
          <button class="ws-btn-del" onclick="event.stopPropagation();wsDeleteModel('${esc(m.name)}')" title="删除">🗑️</button>
        </div>
      </div>`
    ).join('');
    wsPopulateSelects();
  } catch (e) { el.innerHTML = `<p class="desc">❌ ${e.message}</p>`; }
}

function wsPopulateSelects() {
  const items = $$('.ws-m-item');
  const names = Array.from(items).map(el => el.dataset.name).filter(Boolean);
  ['ws-q-source','ws-v-base','ws-t-model'].forEach(id => {
    const sel = $(`#${id}`);
    const cur = sel.value;
    sel.innerHTML = '<option value="">选择模型...</option>' + names.map(n => `<option value="${esc(n)}"${n===cur?' selected':''}>${esc(n)}</option>`).join('');
  });
  // Auto-suggest quantize target name
  const suggestQName = () => {
    const src = $('#ws-q-source').value;
    const qt = document.querySelector('input[name="qtype"]:checked')?.value || 'q4_k_m';
    if (src && !$('#ws-q-name')._userEdited) {
      const base = src.replace(/:.*$/, '').replace(/[^a-zA-Z0-9_-]/g, '-');
      $('#ws-q-name').value = `${base}-${qt.replace(/_/g, '')}`;
    }
  };
  $('#ws-q-source').onchange = suggestQName;
  document.querySelectorAll('input[name="qtype"]').forEach(r => r.onchange = suggestQName);
  // Track if user manually edited the name
  $('#ws-q-name').oninput = () => { $('#ws-q-name')._userEdited = true; };
  $('#ws-q-source').addEventListener('change', () => { $('#ws-q-name')._userEdited = false; });
}

window.wsSelectModel = async function(name) {
  _wsActiveModel = name;
  $$('.ws-m-item').forEach(el => el.classList.toggle('active', el.dataset.name === name));
  const sec = $('#ws-detail-section'); sec.style.display = 'block';
  const el = $('#ws-model-detail');
  el.innerHTML = '<p class="desc">🔍 加载详情...</p>';
  try {
    const r = await fetch(`/api/workshop/model-info/${encodeURIComponent(name)}`); const d = await r.json();
    el.innerHTML = `<div class="ws-detail-grid">
      <div class="ws-detail-card"><h4>基本信息</h4><pre>${esc(d.show || '无信息')}</pre></div>
      <div class="ws-detail-card"><h4>Modelfile</h4><pre>${esc(d.modelfile || '无')}</pre></div>
    </div>`;
  } catch (e) { el.innerHTML = `<p class="desc">❌ ${e.message}</p>`; }
};

window.wsExportMF = async function(name) {
  try {
    const r = await fetch('/api/workshop/export-modelfile', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({model: name}) });
    const d = await r.json();
    if (d.error) return toast('❌ ' + d.error);
    const blob = new Blob([d.modelfile], { type: 'text/plain' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `Modelfile-${name.replace(/[:/]/g, '-')}`; a.click();
    toast('✅ Modelfile 已下载');
  } catch (e) { toast('❌ ' + e.message); }
};

window.wsCopyModel = async function(name) {
  const dest = prompt('新模型名称:', name + '-copy');
  if (!dest) return;
  try {
    const r = await fetch('/api/workshop/copy-model', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({source: name, dest}) });
    const d = await r.json();
    if (d.status === 'ok') { toast('✅ 复制成功'); wsLoadModels(); loadM(); } else toast('❌ ' + (d.error || '失败'));
  } catch (e) { toast('❌ ' + e.message); }
};

window.wsDeleteModel = async function(name) {
  if (!confirm(`确定删除模型 ${name}？此操作不可撤销。`)) return;
  try {
    const r = await fetch(`/api/workshop/model/${encodeURIComponent(name)}`, { method: 'DELETE' });
    const d = await r.json();
    if (d.status === 'ok') { toast('✅ 已删除'); if (_wsActiveModel === name) { _wsActiveModel = ''; $('#ws-detail-section').style.display = 'none'; } wsLoadModels(); loadM(); }
    else toast('❌ ' + (d.error || '删除失败'));
  } catch (e) { toast('❌ ' + e.message); }
};

async function wsQuantize() {
  const source = $('#ws-q-source').value, newName = $('#ws-q-name').value.trim();
  const qtype = document.querySelector('input[name="qtype"]:checked')?.value || 'q4_k_m';
  const res = $('#ws-q-result');
  if (!source) return toast('请选择源模型');
  if (!newName) return toast('请输入新模型名称');
  res.style.display = 'block'; res.className = 'ft-result'; res.innerHTML = '⏳ 量化中，请稍候（大模型可能需要几分钟）...';
  $('#btn-ws-quantize').disabled = true;
  try {
    const r = await fetch('/api/workshop/quantize', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ source_model: source, new_name: newName, quant_type: qtype }) });
    const d = await r.json();
    if (d.status === 'ok') { res.className = 'ft-result ok'; res.innerHTML = `✅ 量化完成！<b>${esc(d.name)}</b> (${esc(d.quant)})`; wsLoadModels(); loadM(); toast('✅ 量化完成'); }
    else { res.className = 'ft-result err'; res.innerHTML = `❌ ${esc(d.error||'量化失败')}${d.hint ? '<div style="margin-top:8px;padding:8px 10px;background:var(--bg0);border-radius:6px;color:var(--t2);font-size:11px;line-height:1.6;white-space:pre-wrap">💡 '+esc(d.hint)+'</div>':''}`; }
  } catch (e) { res.className = 'ft-result err'; res.innerHTML = '❌ ' + e.message; }
  $('#btn-ws-quantize').disabled = false;
}

async function wsCreateVariant() {
  const base = $('#ws-v-base').value, newName = $('#ws-v-name').value.trim(), sys = $('#ws-v-sys').value.trim();
  const params = {};
  const temp = parseFloat($('#ws-v-temp').value); if (!isNaN(temp)) params.temperature = temp;
  const topp = parseFloat($('#ws-v-topp').value); if (!isNaN(topp)) params.top_p = topp;
  const rp = parseFloat($('#ws-v-rp').value); if (!isNaN(rp)) params.repeat_penalty = rp;
  const ctx = parseInt($('#ws-v-ctx').value); if (!isNaN(ctx)) params.num_ctx = ctx;
  const res = $('#ws-v-result');
  if (!base) return toast('请选择基础模型');
  if (!newName) return toast('请输入新模型名称');
  res.style.display = 'block'; res.className = 'ft-result'; res.innerHTML = '⏳ 创建中...';
  try {
    const r = await fetch('/api/workshop/create-variant', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ base_model: base, new_name: newName, system_prompt: sys, parameters: params }) });
    const d = await r.json();
    if (d.status === 'ok') { res.className = 'ft-result ok'; res.innerHTML = `✅ 变体 <b>${esc(d.name)}</b> 创建成功！<details style="margin-top:6px"><summary style="cursor:pointer;font-size:10px">查看 Modelfile</summary><pre style="margin-top:4px;font-size:10px">${esc(d.modelfile)}</pre></details>`; wsLoadModels(); loadM(); toast('✅ 变体创建成功'); }
    else { res.className = 'ft-result err'; res.innerHTML = `❌ 失败: ${esc(d.error||'')}`; }
  } catch (e) { res.className = 'ft-result err'; res.innerHTML = '❌ ' + e.message; }
}

async function wsQuickTest() {
  const model = $('#ws-t-model').value, prompt = $('#ws-t-prompt').value.trim();
  if (!model) return toast('请选择模型');
  if (!prompt) return toast('请输入测试提示词');
  const resEl = $('#ws-t-result'); resEl.style.display = 'block';
  const respEl = $('#ws-t-response'); respEl.textContent = '⏳ 模型思考中...';
  const statsEl = $('#ws-t-stats'); statsEl.innerHTML = '';
  $('#btn-ws-test').disabled = true;
  try {
    const r = await fetch('/api/workshop/quick-test', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ model, prompt }) });
    const d = await r.json();
    if (d.error) { respEl.textContent = '❌ ' + d.error; }
    else {
      respEl.textContent = d.response || '(空回复)';
      const dur = d.total_duration ? (d.total_duration / 1e9).toFixed(2) : '-';
      const tps = d.eval_count && d.eval_duration ? (d.eval_count / (d.eval_duration / 1e9)).toFixed(1) : '-';
      statsEl.innerHTML = `<span>⏱️ 总耗时: <b>${dur}s</b></span><span>📝 Token: <b>${d.eval_count||'-'}</b></span><span>⚡ 速度: <b>${tps} tok/s</b></span>`;
    }
  } catch (e) { respEl.textContent = '❌ ' + e.message; }
  $('#btn-ws-test').disabled = false;
}

// ======================= Init =======================
function init() {
  $('#btn-send').onclick = send; $('#btn-stop').onclick = () => S.ac?.abort();
  $('#inp').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } };
  $('#inp').oninput = autoR;
  $('#btn-new-chat').onclick = () => { newChat(); $('#inp').focus(); };
  $('#btn-clear-all').onclick = () => { if (!confirm('确定清除所有对话？')) return; S.chats = []; S.cid = null; sav(); rList(); rMsgs(); };
  $('#btn-toggle-sb').onclick = () => $('#sidebar').classList.toggle('open');
  // Model
  $('#model-select').onchange = async function() { const nv = this.value; await checkModelSwitch(nv); S.cfg.model = nv; savS(); };
  $('#btn-preload').onclick = async () => { const m = $('#model-select').value; if (!m) return toast('请先选择模型'); toast(`⚡ 预热 ${m}...`); try { await fetch('/api/preload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: m, keep_alive: S.cfg.ka }) }); toast(`✅ ${m} 已加载`); } catch (e) { toast('失败: ' + e.message); } };
  // Context menu
  $('#ctx-pin').onclick = () => { const c = S.chats.find(x => x.id === _ctxId); if (c) { c.pin = !c.pin; sav(); rList(); } };
  $('#ctx-rename').onclick = () => { $('#ov-rename').style.display = 'flex'; $('#rename-input').value = S.chats.find(x => x.id === _ctxId)?.title || ''; setTimeout(() => $('#rename-input').focus(), 50); };
  $('#ctx-delete').onclick = () => { const c = S.chats.find(x => x.id === _ctxId); if (c && confirm(`确定删除「${c.title}」？`)) { S.chats = S.chats.filter(x => x.id !== _ctxId); if (S.cid === _ctxId) S.cid = S.chats[0]?.id || null; sav(); rList(); rMsgs(); } };
  $('#btn-rename-ok').onclick = () => { const c = S.chats.find(x => x.id === _ctxId); if (c) { c.title = $('#rename-input').value.trim() || c.title; sav(); rList(); } $('#ov-rename').style.display = 'none'; };
  $('#cl-rename').onclick = () => $('#ov-rename').style.display = 'none';
  $('#ov-rename').onclick = e => { if (e.target.id === 'ov-rename') $('#ov-rename').style.display = 'none'; };
  $('#rename-input').onkeydown = e => { if (e.key === 'Enter') $('#btn-rename-ok').click(); };
  // Settings
  $('#btn-settings').onclick = () => { $('#ov-settings').style.display = 'flex'; loadRun(); };
  $('#cl-settings').onclick = () => { $('#ov-settings').style.display = 'none'; clearInterval(_runTimer); };
  $('#ov-settings').onclick = e => { if (e.target.id === 'ov-settings') { $('#ov-settings').style.display = 'none'; clearInterval(_runTimer); } };
  // Model mgmt
  $('#btn-model-mgmt').onclick = () => { rModelList(); $('#ov-model').style.display = 'flex'; };
  $('#cl-model').onclick = () => $('#ov-model').style.display = 'none';
  $('#ov-model').onclick = e => { if (e.target.id === 'ov-model') $('#ov-model').style.display = 'none'; };
  $('#btn-pull').onclick = () => doPull($('#pull-name').value.trim());
  $('#pull-name').onkeydown = e => { if (e.key === 'Enter') doPull($('#pull-name').value.trim()); };
  $$('.tag[data-m]').forEach(b => b.onclick = () => { $('#pull-name').value = b.dataset.m; });
  $('#btn-pull-pause').onclick = () => { if (S.pullAC) { S.pullPaused = true; S.pullAC.abort(); S.pullAC = null; $('#btn-pull-pause').style.display = 'none'; $('#btn-pull-resume').style.display = 'inline-flex'; } };
  $('#btn-pull-resume').onclick = () => { if (S.pullName && S.pullPaused) { $('#btn-pull-pause').style.display = 'inline-flex'; $('#btn-pull-resume').style.display = 'none'; doPull(S.pullName); } };
  $('#btn-pull-cancel').onclick = async () => { if (S.pullAC) { S.pullPaused = false; S.pullAC.abort(); S.pullAC = null; } if (S.pullName && confirm('放弃并删除已下载数据？')) { try { await fetch('/api/delete', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: S.pullName }) }); } catch {} toast('已清理'); S.pullName = ''; $('#pull-prog').style.display = 'none'; $('#btn-pull').disabled = false; await loadM(); rModelList(); } };
  // Think & system
  $('#chip-think').onclick = () => { S.cfg.think = !S.cfg.think; $('#chip-think').classList.toggle('on', S.cfg.think); $('#sw-think').checked = S.cfg.think; savS(); };
  $('#sw-think').onchange = () => { S.cfg.think = $('#sw-think').checked; $('#chip-think').classList.toggle('on', S.cfg.think); savS(); };
  $('#chip-sys').onclick = () => { S.cfg.showSys = !S.cfg.showSys; $('#chip-sys').classList.toggle('on', S.cfg.showSys); $('#sys-panel').style.display = S.cfg.showSys ? 'block' : 'none'; savS(); };
  $('#sys-text').oninput = () => { S.cfg.sysp = $('#sys-text').value; savS(); };
  $$('.pre').forEach(b => b.onclick = () => { const p = PRESETS[b.dataset.p]; if (p) { $('#sys-text').value = p; S.cfg.sysp = p; savS(); } });
  // Device & params
  $$('input[name="dev"]').forEach(r => r.onchange = () => { S.cfg.dev = r.value; updDev(); savS(); });
  $('#r-layers').oninput = function() { S.cfg.layers = +this.value; $('#v-layers').textContent = this.value; savS(); };
  $$('input[name="ka"]').forEach(r => r.onchange = () => { S.cfg.ka = r.value; savS(); });
  $('#r-temp').oninput = function() { S.cfg.temp = +this.value; $('#v-temp').textContent = this.value; savS(); };
  $('#r-topp').oninput = function() { S.cfg.topp = +this.value; $('#v-topp').textContent = this.value; savS(); };
  $('#r-ctx').oninput = function() { S.cfg.ctx = +this.value; $('#v-ctx').textContent = this.value; savS(); };
  $('#r-rp').oninput = function() { S.cfg.rp = +this.value; $('#v-rp').textContent = this.value; savS(); };
  $('#r-seed').oninput = function() { S.cfg.seed = +this.value; $('#v-seed').textContent = this.value < 0 ? '随机' : this.value; savS(); };
  $('#btn-ref-run').onclick = loadRun;
  // Image
  $('#img-in').onchange = e => { const f = e.target.files[0]; if (!f) return; const r = new FileReader(); r.onload = ev => { S.pimg = { b64: ev.target.result.split(',')[1] }; $('#prev-img').src = ev.target.result; $('#img-prev').style.display = 'inline-block'; }; r.readAsDataURL(f); e.target.value = ''; };
  $('#img-rm').onclick = () => { S.pimg = null; $('#img-prev').style.display = 'none'; };
  // Export
  $('#btn-export').onclick = () => { if (cur()?.msgs.length) $('#ov-export').style.display = 'flex'; else toast('无对话可导出'); };
  $('#cl-export').onclick = () => $('#ov-export').style.display = 'none';
  $('#ov-export').onclick = e => { if (e.target.id === 'ov-export') $('#ov-export').style.display = 'none'; };
  $('#exp-md').onclick = () => doExport('markdown'); $('#exp-json').onclick = () => doExport('json');
  // API
  $('#btn-api-page').onclick = showAPI; $('#cl-api').onclick = () => $('#ov-api').style.display = 'none';
  $('#sw-sharing').onchange = toggleSharing; $('#btn-gen-key').onclick = genKey;
  $('#key-name').onkeydown = e => { if (e.key === 'Enter') genKey(); };
  $('#btn-copy-url').onclick = () => { navigator.clipboard.writeText($('#share-url').textContent); toast('已复制'); };
  $('#btn-copy-key').onclick = () => { navigator.clipboard.writeText($('#new-key-val').textContent); toast('已复制密钥'); };
  $('#btn-copy-admin').onclick = () => { navigator.clipboard.writeText($('#admin-token-display').textContent); toast('已复制管理员令牌'); };
  // Fine-Tuning
  $('#btn-finetune').onclick = showFT; $('#cl-ft').onclick = () => { $('#ov-ft').style.display = 'none'; document.body.style.overflow = ''; clearInterval(_ftMonTimer); };
  $$('.ft-tab').forEach(t => t.onclick = () => switchFTTab(t.dataset.tab));
  $$('.ft-mcard').forEach(c => c.onclick = () => selectMethod(c.dataset.m));
  $$('.ft-fmt').forEach(b => b.onclick = () => showFmtExample(b.dataset.f));
  $('#btn-mf-create').onclick = createModelfile;
  $('#btn-start-train').onclick = startTraining;
  $('#btn-stop-train').onclick = async () => { await fetch('/api/finetune/stop', { method: 'POST' }); toast('正在停止...'); };
  $('#btn-mon-stop').onclick = async () => { await fetch('/api/finetune/stop', { method: 'POST' }); toast('正在停止...'); };
  $('#btn-upload-ds').onclick = uploadDataset;
  if ($('#tr-dataset')) $('#tr-dataset').onchange = _autoSelectDatasetFormat;
  if ($('#btn-ds-custom-dl')) $('#btn-ds-custom-dl').onclick = downloadCustomDS;
  // Enter key on custom HF input triggers download
  if ($('#ds-hf-input')) $('#ds-hf-input').onkeydown = e => { if (e.key === 'Enter') downloadCustomDS(); };
  $('#btn-import-gguf').onclick = importGGUF;
  $('#btn-create-venv').onclick = createVenv;
  $('#btn-save-paths').onclick = savePaths;
  $('#btn-hf-login').onclick = hfLogin;
  if ($('#btn-refresh-projects')) $('#btn-refresh-projects').onclick = loadProjects;
  // Model Workshop
  $('#btn-workshop').onclick = showWorkshop;
  $('#cl-workshop').onclick = () => { $('#ov-workshop').style.display = 'none'; document.body.style.overflow = ''; };
  $$('.ws-tab').forEach(t => t.onclick = () => switchWSTab(t.dataset.tab));
  // Pretrain Lab (initialized in pretrain-lab.js)
  if ($('#btn-pretrain')) $('#btn-pretrain').onclick = () => { if (typeof showPretrainLab === 'function') showPretrainLab(); };
  if ($('#btn-ws-refresh')) $('#btn-ws-refresh').onclick = wsLoadModels;
  if ($('#btn-ws-quantize')) $('#btn-ws-quantize').onclick = wsQuantize;
  if ($('#btn-ws-variant')) $('#btn-ws-variant').onclick = wsCreateVariant;
  if ($('#btn-ws-test')) $('#btn-ws-test').onclick = wsQuickTest;
  if ($('#btn-ws-hf-import')) $('#btn-ws-hf-import').onclick = wsHFImport;
  // Auto-suggest ollama name from HF model ID
  if ($('#ws-hf-model')) $('#ws-hf-model').oninput = () => {
    const hf = $('#ws-hf-model').value.trim();
    if (hf.includes('/') && !$('#ws-hf-name')._userEdited) {
      $('#ws-hf-name').value = hf.split('/').pop().toLowerCase().replace(/[^a-z0-9_-]/g, '-');
    }
  };
  if ($('#ws-hf-name')) $('#ws-hf-name').oninput = () => { $('#ws-hf-name')._userEdited = true; };
  // Close overlays
  $('#cl-switch').onclick = () => $('#ov-switch').style.display = 'none';
  // Shortcuts
  document.onkeydown = e => { if ((e.ctrlKey || e.metaKey) && e.key === 'n') { e.preventDefault(); newChat(); $('#inp').focus(); } };
  syncUI(); rList();
  if (S.cid) rMsgs(); else if (S.chats.length) { S.cid = S.chats[0].id; rList(); rMsgs(); }
  loadM(); $('#inp').focus();
}
document.addEventListener('DOMContentLoaded', init);
