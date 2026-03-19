/* ============================================================
   Pretrain Lab — 训练实验室 (架构编辑器 + 项目管理)
   ============================================================ */
const PT = {
  project: null,        // current project meta
  projects: [],         // all projects list
  arch: {},             // current architecture config
  blocks: [],           // canvas blocks [{id, type, label, icon, locked}]
  dragSrc: null,        // drag source element
  dragFromPalette: false,
  componentOptions: null, // component options from backend
  statsTimer: null,     // debounce timer for stats calculation
  selectedPython: '',   // selected python env path
  trainMode: 'pretrain', // 'pretrain' or 'sft'
  columnConfig: {},     // {filename: [col1, col2, ...]} for pretrain column selection
  layerGroups: {},     // {groupId: {label, repeat}} for layer group definitions
};

/* Get selected python path for API calls */
function ptPy() { return _$('#pt-tr-python')?.value || _$('#pt-tok-python')?.value || PT.selectedPython || ''; }

const _$ = s => document.querySelector(s);
const _$$ = s => document.querySelectorAll(s);

// ======================= Block definitions =======================
const BLOCK_CATS = {
  core:     { label: '核心层', icon: '🧱' },
  norm:     { label: '归一化', icon: '📏' },
  connect:  { label: '连接与流控', icon: '🔗' },
  expert:   { label: '专家/扩展', icon: '🧪' },
};

const BLOCK_DEFS = {
  // Core
  embedding:    { icon: '📦', label: 'Embedding',         color: '#c9a24d', cat: 'core', desc: '将 token ID 映射为向量', configKeys: ['vocab_size','max_seq_len','hidden_dim','pos_encoding','rope_theta'] },
  attention:    { icon: '👁️', label: 'Self-Attention',    color: '#a78bfa', cat: 'core', desc: '自注意力机制（Q/K/V 投影 + 缩放点积 + 输出投影）', configKeys: ['attention_type','num_heads','num_kv_heads','hidden_dim'] },
  cross_attn:   { icon: '🔀', label: 'Cross-Attention',   color: '#c084fc', cat: 'core', desc: '交叉注意力，用 Q 查询另一个序列的 K/V。Encoder-Decoder 架构使用' },
  ffn:          { icon: '⚡', label: 'FFN',                color: '#4ade80', cat: 'core', desc: '前馈神经网络（两层全连接 + 激活函数）', configKeys: ['activation','intermediate_dim'] },
  linear:       { icon: '📐', label: 'Linear Projection',  color: '#38bdf8', cat: 'core', desc: '自定义全连接层。可用于维度变换、瓶颈层等' },
  lm_head:      { icon: '🎯', label: 'LM Head',            color: '#f87171', cat: 'core', desc: '将隐藏状态映射回词表空间进行预测', configKeys: ['tie_word_embeddings'] },
  // Norm
  norm_pre:     { icon: '📏', label: 'Pre-Norm',    color: '#60a5fa', cat: 'norm', desc: '注意力前的归一化', configKeys: ['norm_type','norm_eps'] },
  norm_post:    { icon: '📏', label: 'Post-Norm',   color: '#60a5fa', cat: 'norm', desc: 'FFN 前的归一化', configKeys: ['norm_type','norm_eps'] },
  norm_final:   { icon: '📏', label: 'Final Norm',  color: '#60a5fa', cat: 'norm', desc: '输出前的最终归一化', configKeys: ['norm_type','norm_eps'] },
  // Connect & Flow
  residual:     { icon: '➰', label: 'Residual Add',       color: '#fbbf24', cat: 'connect', desc: '残差连接 (x + sublayer(x))。Transformer 的核心稳定机制' },
  dropout:      { icon: '💧', label: 'Dropout',             color: '#94a3b8', cat: 'connect', desc: '训练时随机丢弃，防止过拟合', configKeys: ['dropout'] },
  skip_connect: { icon: '⏭️', label: 'Skip Connection',    color: '#fb923c', cat: 'connect', desc: '跨多层的跳跃连接。DenseNet/U-Net 风格' },
  gate:         { icon: '🚪', label: 'Gating Layer',        color: '#a3e635', cat: 'connect', desc: '门控机制 (σ(x) ⊙ y)。控制信息流通' },
  // Expert / Extension
  moe:          { icon: '🧪', label: 'MoE Layer',           color: '#e879f9', cat: 'expert', desc: 'Mixture of Experts。Router 选择 top-k 专家处理 token，大幅扩展容量' },
  parallel_attn:{ icon: '⚡👁️', label: 'Parallel Attn+FFN', color: '#2dd4bf', cat: 'expert', desc: '并行注意力+FFN（PaLM 风格）。Attn 和 FFN 同时计算后相加，减少延迟' },
  kv_cache:     { icon: '💾', label: 'KV Cache',            color: '#a78bfa', cat: 'expert', desc: '推理时缓存 Key/Value，避免重复计算。标注推理优化' },
};

// Which block types are inside a standard layer
const LAYER_INNER_TYPES = new Set(['norm_pre','attention','cross_attn','norm_post','ffn','residual','dropout','moe','parallel_attn','gate','linear']);

// Standard transformer structure
function defaultBlocks() {
  return [
    { id: 'b-emb',      type: 'embedding',  label: 'Token Embedding', icon: '📦', locked: true },
    { id: 'b-norm-pre',  type: 'norm_pre',   label: 'RMSNorm',        icon: '📏', locked: false },
    { id: 'b-attn',     type: 'attention',  label: 'Attention',       icon: '👁️', locked: false },
    { id: 'b-norm-post', type: 'norm_post',  label: 'RMSNorm',        icon: '📏', locked: false },
    { id: 'b-ffn',      type: 'ffn',        label: 'FFN',             icon: '⚡', locked: false },
    { id: 'b-norm-f',   type: 'norm_final', label: 'Final Norm',      icon: '📏', locked: false },
    { id: 'b-head',     type: 'lm_head',    label: 'LM Head',         icon: '🎯', locked: true },
  ];
}

// ======================= Initialization =======================
function showPretrainLab() {
  _$('#ov-pretrain').style.display = 'flex';
  document.body.style.overflow = 'hidden';
  loadComponents();
  ptLoadProjectList();
  if (!PT.project) {
    loadTemplates();
    resetToDefault();
  }
}

async function loadComponents() {
  if (PT.componentOptions) return;
  try {
    const r = await fetch('/api/pretrain/arch/components');
    PT.componentOptions = await r.json();
  } catch (e) { console.error('Failed to load components:', e); }
}

// ======================= Templates =======================
async function loadTemplates() {
  try {
    const r = await fetch('/api/pretrain/arch/templates');
    const d = await r.json();
    const el = _$('#pt-templates');
    if (!el) return;
    el.innerHTML = d.templates.map(t => `
      <div class="pt-tpl-card" data-tid="${t.id}" onclick="ptSelectTemplate('${t.id}')">
        <div class="pt-tpl-head">
          <span class="pt-tpl-family">${esc(t.family)}</span>
          <span class="pt-tpl-params">${esc(t.total_params_fmt)}</span>
        </div>
        <div class="pt-tpl-name">${esc(t.name)}</div>
        <div class="pt-tpl-desc">${esc(t.description)}</div>
        <div class="pt-tpl-vram">训练显存 ~${esc(t.vram_train)}</div>
      </div>
    `).join('');
  } catch (e) { console.error(e); }
}

window.ptSelectTemplate = async function(tid) {
  // highlight
  _$$('.pt-tpl-card').forEach(c => c.classList.toggle('active', c.dataset.tid === tid));
  try {
    const r = await fetch(`/api/pretrain/arch/template/${tid}`);
    const t = await r.json();
    PT.arch = { ...t.architecture };
    updateBlocksFromArch();
    renderCanvas();
    updateStats();
  } catch (e) { console.error(e); }
};

// ======================= Canvas Rendering =======================
function updateBlocksFromArch() {
  const a = PT.arch;
  const normLabel = a.norm_type === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
  const attnLabel = ({ mha: 'Multi-Head Attention', gqa: 'Grouped-Query Attention', mqa: 'Multi-Query Attention' })[a.attention_type] || 'Attention';
  const ffnLabel = a.activation === 'swiglu' ? 'SwiGLU FFN' : `${(a.activation||'gelu').toUpperCase()} FFN`;
  const posLabel = ({ rope: '+ RoPE', alibi: '+ ALiBi', absolute: '+ 绝对位置编码', none: '' })[a.pos_encoding] || '';

  PT.blocks = [
    { id: 'b-emb',       type: 'embedding',   label: `Token Embedding ${posLabel}`, icon: '📦', locked: true },
    // === Repeating Layer ===
    { id: 'b-norm-pre',  type: 'norm_pre',    label: normLabel,  icon: '📏', locked: false, layerGroup: 'main' },
    { id: 'b-attn',      type: 'attention',   label: attnLabel,  icon: '👁️', locked: false, layerGroup: 'main' },
    { id: 'b-res1',      type: 'residual',    label: 'Residual Add', icon: '➰', locked: false, layerGroup: 'main' },
    { id: 'b-norm-post', type: 'norm_post',   label: normLabel,  icon: '📏', locked: false, layerGroup: 'main' },
    { id: 'b-ffn',       type: 'ffn',         label: ffnLabel,   icon: '⚡', locked: false, layerGroup: 'main' },
    { id: 'b-res2',      type: 'residual',    label: 'Residual Add', icon: '➰', locked: false, layerGroup: 'main' },
    // === End Layer ===
    { id: 'b-norm-f',    type: 'norm_final',  label: `Final ${normLabel}`, icon: '📏', locked: false },
    { id: 'b-head',      type: 'lm_head',     label: `LM Head ${a.tie_word_embeddings ? '(共享权重)' : ''}`, icon: '🎯', locked: true },
  ];
  // Store layer groups info
  PT.layerGroups = {
    main: { label: 'Transformer Layer', repeat: a.num_layers || 8 }
  };
}

function resetToDefault() {
  PT.arch = {
    vocab_size: 32000, max_seq_len: 512, hidden_dim: 512, num_layers: 8,
    num_heads: 8, num_kv_heads: 4, intermediate_dim: 1376,
    norm_type: 'rmsnorm', norm_eps: 1e-5, pos_encoding: 'rope',
    rope_theta: 10000, activation: 'swiglu', attention_type: 'gqa',
    tie_word_embeddings: true, dropout: 0.0,
  };
  updateBlocksFromArch();
  renderCanvas();
  updateStats();
}

function renderCanvas() {
  const canvas = _$('#pt-canvas');
  if (!canvas) return;
  const a = PT.arch;
  let html = '';
  let currentGroup = null;
  const hd = Math.floor(a.hidden_dim / (a.num_heads || 1));

  // Tensor shape annotation helper
  function shapeLabel(block) {
    switch (block.type) {
      case 'embedding': return `[B, S] → [B, S, ${a.hidden_dim}]`;
      case 'attention': case 'cross_attn': case 'parallel_attn': return `[B, S, ${a.hidden_dim}] → [B, S, ${a.hidden_dim}]`;
      case 'ffn': return `[B, S, ${a.hidden_dim}] → [B, S, ${a.intermediate_dim}] → [B, S, ${a.hidden_dim}]`;
      case 'moe': return `[B, S, ${a.hidden_dim}] → TopK experts → [B, S, ${a.hidden_dim}]`;
      case 'lm_head': return `[B, S, ${a.hidden_dim}] → [B, S, ${a.vocab_size}]`;
      case 'norm_pre': case 'norm_post': case 'norm_final': return `[B, S, ${a.hidden_dim}]`;
      case 'residual': return `x + sublayer(x)`;
      case 'dropout': return `p=${a.dropout || 0}`;
      case 'linear': return `[B, S, d_in] → [B, S, d_out]`;
      default: return '';
    }
  }

  // Insert button between blocks
  function insertBtn(idx) {
    return `<div class="pt-insert-zone" data-insert-idx="${idx}">
      <button class="pt-insert-btn" onclick="ptInsertBlockAt(${idx})" title="在此处插入组件">＋</button>
    </div>`;
  }

  // Insert button at top
  html += insertBtn(0);

  PT.blocks.forEach((block, idx) => {
    const def = BLOCK_DEFS[block.type] || { icon: '❓', label: block.type, color: '#666', desc: '' };
    const groupId = block.layerGroup;

    // Open layer group
    if (groupId && groupId !== currentGroup) {
      const grp = PT.layerGroups[groupId] || { label: 'Layer', repeat: a.num_layers };
      html += `<div class="pt-layer-group" data-group="${groupId}">
        <div class="pt-layer-badge" onclick="ptEditLayerGroup('${groupId}')" title="点击编辑层组">
          <span class="pt-lg-repeat">× ${grp.repeat}</span>
          <span class="pt-lg-label">${esc(grp.label)}</span>
        </div>
        <div class="pt-layer-bracket">`;
      currentGroup = groupId;
    }

    // Block element
    const shape = shapeLabel(block);
    const activeClass = block._active ? ' active' : '';
    html += `<div class="pt-block ${block.type}${activeClass}" draggable="${!block.locked}"
                  data-idx="${idx}" data-type="${block.type}" data-group="${groupId || ''}"
                  onclick="ptClickBlock(${idx})"
                  style="--block-color:${def.color || '#666'}">
      <div class="pt-block-grip">${block.locked ? '🔒' : '⠿'}</div>
      <div class="pt-block-icon">${def.icon || block.icon}</div>
      <div class="pt-block-info">
        <div class="pt-block-label">${esc(block.label)}</div>
        <div class="pt-block-desc">${esc(def.desc || '')}</div>
      </div>
      <div class="pt-block-badge">${_blockBadge(block, a)}</div>
      <div class="pt-block-actions">
        <button class="pt-block-act" onclick="event.stopPropagation();ptDuplicateBlock(${idx})" title="复制">📋</button>
        ${!block.locked ? `<button class="pt-block-act pt-block-del" onclick="event.stopPropagation();ptRemoveBlock(${idx})" title="移除">✕</button>` : ''}
      </div>
    </div>`;

    // Close layer group if next block is different group
    const nextBlock = PT.blocks[idx + 1];
    const nextGroup = nextBlock?.layerGroup;
    if (groupId && groupId !== nextGroup) {
      html += `</div></div>`; // close bracket + group
      currentGroup = null;
    }

    // Tensor shape annotation + insert button between blocks
    if (idx < PT.blocks.length - 1) {
      html += `<div class="pt-flow-annotation">
        <span class="pt-flow-shape">${shape}</span>
      </div>`;
      html += insertBtn(idx + 1);
    }
  });

  canvas.innerHTML = html;
  _setupCanvasDrag();
}

// Insert block at specific position
window.ptInsertBlockAt = function(idx) {
  // Show a quick picker
  const types = Object.entries(BLOCK_DEFS).filter(([k]) => !['embedding','lm_head'].includes(k));
  const catOrder = ['core','norm','connect','expert'];
  let html = '<div class="pt-quick-picker">';
  for (const catId of catOrder) {
    const cat = BLOCK_CATS[catId];
    if (!cat) continue;
    const items = types.filter(([,d]) => d.cat === catId);
    if (!items.length) continue;
    html += `<div class="pt-qp-cat">${cat.icon} ${cat.label}</div>`;
    html += items.map(([k, d]) => `<div class="pt-qp-item" onclick="ptDoInsertBlock(${idx},'${k}')">${d.icon} ${d.label}</div>`).join('');
  }
  html += '</div>';

  // Position picker near the insert button
  const existingPicker = _$('.pt-quick-picker-wrap');
  if (existingPicker) existingPicker.remove();
  const wrap = document.createElement('div');
  wrap.className = 'pt-quick-picker-wrap';
  wrap.innerHTML = html;
  wrap.style.position = 'fixed';
  wrap.style.zIndex = '9999';
  // Position near mouse click
  const btn = _$(`.pt-insert-zone[data-insert-idx="${idx}"] .pt-insert-btn`);
  if (btn) {
    const rect = btn.getBoundingClientRect();
    wrap.style.left = (rect.right + 8) + 'px';
    wrap.style.top = (rect.top - 50) + 'px';
  }
  document.body.appendChild(wrap);
  // Close on outside click
  setTimeout(() => {
    document.addEventListener('click', function _closePicker(e) {
      if (!wrap.contains(e.target)) { wrap.remove(); document.removeEventListener('click', _closePicker); }
    });
  }, 10);
};

window.ptDoInsertBlock = function(idx, ctype) {
  const def = BLOCK_DEFS[ctype];
  if (!def) return;
  // Determine if inserting inside a layer group
  const prevBlock = PT.blocks[idx - 1];
  const nextBlock = PT.blocks[idx];
  const groupId = prevBlock?.layerGroup || nextBlock?.layerGroup || null;

  const newBlock = {
    id: `b-${ctype}-${Date.now()}`,
    type: ctype,
    label: def.label,
    icon: def.icon,
    locked: false,
    layerGroup: LAYER_INNER_TYPES.has(ctype) ? groupId : null,
  };
  PT.blocks.splice(idx, 0, newBlock);
  renderCanvas();
  scheduleStats();
  // Remove picker
  const picker = _$('.pt-quick-picker-wrap');
  if (picker) picker.remove();
};

window.ptDuplicateBlock = function(idx) {
  const src = PT.blocks[idx];
  if (!src || src.locked) return;
  const newBlock = { ...src, id: `b-${src.type}-${Date.now()}`, locked: false };
  PT.blocks.splice(idx + 1, 0, newBlock);
  renderCanvas();
  scheduleStats();
};

window.ptEditLayerGroup = function(groupId) {
  const grp = PT.layerGroups[groupId];
  if (!grp) return;
  const newRepeat = prompt(`设置 "${grp.label}" 的重复次数 (当前: ${grp.repeat}):`, grp.repeat);
  if (newRepeat !== null && +newRepeat >= 1) {
    grp.repeat = +newRepeat;
    PT.arch.num_layers = +newRepeat;
    renderCanvas();
    scheduleStats();
  }
};

function _blockBadge(block, a) {
  switch (block.type) {
    case 'embedding': return `${a.vocab_size?.toLocaleString()} × ${a.hidden_dim}`;
    case 'attention': case 'cross_attn': {
      const hd = Math.floor(a.hidden_dim / (a.num_heads || 1));
      return `${a.num_heads}h ${a.num_kv_heads}kv d${hd}`;
    }
    case 'ffn': return `${a.hidden_dim} → ${a.intermediate_dim} → ${a.hidden_dim}`;
    case 'moe': return `${a.moe_num_experts || 8}E top${a.moe_top_k || 2}`;
    case 'parallel_attn': return `Attn ∥ FFN`;
    case 'norm_pre': case 'norm_post': case 'norm_final': return `d=${a.hidden_dim}`;
    case 'lm_head': return `${a.hidden_dim} → ${a.vocab_size?.toLocaleString()}`;
    case 'residual': return `x + f(x)`;
    case 'dropout': return `p=${a.dropout || 0}`;
    case 'gate': return `σ(x)⊙y`;
    case 'skip_connect': return `skip`;
    case 'linear': return `d → d`;
    case 'kv_cache': return `L=${a.num_layers} S=${a.max_seq_len}`;
    default: return '';
  }
}

// ======================= Block Click / Config Panel =======================
window.ptClickBlock = function(idx) {
  const block = PT.blocks[idx];
  if (!block) return;
  const section = _$('#pt-config-section');
  const title = _$('#pt-config-title');
  const body = _$('#pt-config-body');
  section.style.display = 'block';
  title.textContent = `⚙️ ${block.icon} ${block.label} 配置`;

  const a = PT.arch;
  const opts = PT.componentOptions || { options: {}, labels: {}, descriptions: {} };
  let html = '';

  // Helper to create a select with descriptions
  function sel(field, label, options, descriptions) {
    const curVal = a[field] || '';
    let h = `<div class="pt-cfg-row"><label class="ft-label">${label}</label><select class="ft-input pt-cfg-input" data-field="${field}" onchange="ptUpdateArch('${field}', this.value)">`;
    const labels = opts.labels[field] || {};
    for (const o of options) {
      h += `<option value="${o}" ${curVal === o ? 'selected' : ''}>${labels[o] || o}</option>`;
    }
    h += `</select>`;
    if (descriptions && descriptions[curVal]) {
      h += `<div class="pt-cfg-hint">${descriptions[curVal]}</div>`;
    }
    h += `</div>`;
    return h;
  }

  function num(field, label, min, max, step, hint) {
    const v = a[field] ?? 0;
    return `<div class="pt-cfg-row"><label class="ft-label">${label}</label>
      <input type="number" class="ft-input pt-cfg-input" data-field="${field}"
             value="${v}" min="${min}" max="${max}" step="${step||1}"
             onchange="ptUpdateArch('${field}', +this.value)">
      ${hint ? `<div class="pt-cfg-hint">${hint}</div>` : ''}
    </div>`;
  }

  function toggle(field, label, hint) {
    const v = a[field] ?? false;
    return `<div class="pt-cfg-row"><label class="ft-label" style="display:flex;align-items:center;gap:8px">
      <input type="checkbox" ${v ? 'checked' : ''} onchange="ptUpdateArch('${field}', this.checked)"> ${label}
    </label>${hint ? `<div class="pt-cfg-hint">${hint}</div>` : ''}</div>`;
  }

  // Show different config depending on block type
  switch (block.type) {
    case 'embedding':
      html += `<div class="pt-cfg-group-title">Embedding 层配置</div>`;
      html += num('vocab_size', '词表大小 (vocab_size)', 100, 200000, 1000, '词表越大，模型能表示的 token 种类越多，但 Embedding 层参数也越多');
      html += num('max_seq_len', '最大序列长度 (max_seq_len)', 32, 32768, 64, '模型能处理的最大 token 数。越长显存越多');
      html += num('hidden_dim', '隐藏维度 (hidden_dim)', 32, 8192, 64, '所有层的核心维度。必须能被注意力头数整除');
      html += sel('pos_encoding', '位置编码方式', opts.options.pos_encoding || ['rope','alibi','absolute','none'], opts.descriptions?.pos_encoding);
      if (a.pos_encoding === 'rope') {
        html += num('rope_theta', 'RoPE θ (rope_theta)', 1000, 1000000, 1000, '控制 RoPE 的频率基数。默认 10000，增大可改善长序列');
      }
      break;
    case 'attention': case 'cross_attn': case 'norm_pre':
      html += `<div class="pt-cfg-group-title">注意力层配置</div>`;
      html += sel('attention_type', '注意力类型', opts.options.attention_type || ['mha','gqa','mqa'], opts.descriptions?.attention_type);
      html += num('num_heads', '注意力头数 (num_heads)', 1, 128, 1, `当前 head_dim = ${a.hidden_dim}/${a.num_heads} = ${Math.floor(a.hidden_dim/(a.num_heads||1))}`);
      html += num('num_kv_heads', 'KV 头数 (num_kv_heads)', 1, 128, 1, 'GQA: 多个 Q 头共享同组 K/V。MQA: KV 头数=1。MHA: KV 头数=注意力头数');
      html += num('hidden_dim', '隐藏维度 (hidden_dim)', 32, 8192, 64, '改变此值会影响所有层');
      if (block.type === 'cross_attn') {
        html += `<div class="pt-cfg-hint" style="color:var(--ac)">💡 Cross-Attention 是架构标注，实际训练脚本需要提供 encoder 输出。适用于 Encoder-Decoder (T5/BART) 架构。</div>`;
      }
      break;
    case 'ffn': case 'norm_post':
      html += `<div class="pt-cfg-group-title">前馈网络配置</div>`;
      html += sel('activation', '激活函数', opts.options.activation || ['swiglu','gelu','relu','silu'], opts.descriptions?.activation);
      html += num('intermediate_dim', 'FFN 中间维度 (intermediate_dim)', 64, 32768, 64, `当前扩展比 = ${(a.intermediate_dim / a.hidden_dim).toFixed(1)}x。SwiGLU 通常用 ~2.7x，标准 FFN 通常用 4x`);
      break;
    case 'norm_final':
      html += `<div class="pt-cfg-group-title">归一化配置</div>`;
      html += sel('norm_type', '归一化方式', opts.options.norm_type || ['rmsnorm','layernorm'], opts.descriptions?.norm_type);
      html += num('norm_eps', 'Epsilon (norm_eps)', 1e-8, 1e-3, 1e-6, '防止除零的小常数');
      break;
    case 'lm_head':
      html += `<div class="pt-cfg-group-title">输出层配置</div>`;
      html += toggle('tie_word_embeddings', '共享 Embedding 权重 (tie_word_embeddings)', '将 LM Head 的权重矩阵与 Embedding 层共享，减少参数量。大多数现代模型都启用此选项');
      break;
    case 'moe':
      html += `<div class="pt-cfg-group-title">Mixture of Experts 配置</div>`;
      html += num('moe_num_experts', '专家数量', 2, 128, 1, '总共有多少个独立的 FFN 专家。常见值: 8, 16, 64');
      html += num('moe_top_k', 'Top-K 路由', 1, 8, 1, '每个 token 选择 top-k 个专家处理。通常 k=1 或 k=2');
      html += sel('activation', '专家 FFN 激活函数', opts.options.activation || ['swiglu','gelu','relu','silu'], opts.descriptions?.activation);
      html += num('intermediate_dim', '每个专家的 FFN 中间维度', 64, 32768, 64, '每个专家的独立 FFN 维度');
      html += `<div class="pt-cfg-hint" style="color:var(--ac)">💡 MoE 用 Router 选择 top-k 专家。参数量 ≈ num_experts × FFN_params，但每个 token 只激活 top-k 个。DeepSeek-MoE、Mixtral 使用此架构。</div>`;
      break;
    case 'parallel_attn':
      html += `<div class="pt-cfg-group-title">并行 Attention+FFN (PaLM 风格)</div>`;
      html += `<div class="pt-cfg-hint">Attention 和 FFN 同时对输入计算，结果相加。减少了顺序依赖，在大规模训练中可提升 ~15% 吞吐量。</div>`;
      html += sel('attention_type', '注意力类型', opts.options.attention_type || ['mha','gqa','mqa'], opts.descriptions?.attention_type);
      html += sel('activation', '激活函数', opts.options.activation || ['swiglu','gelu','relu','silu'], opts.descriptions?.activation);
      break;
    case 'residual':
      html += `<div class="pt-cfg-group-title">残差连接</div>`;
      html += `<div class="pt-cfg-hint">残差连接 (x + sublayer(x)) 是 Transformer 的核心机制，帮助梯度流过深层网络，避免梯度消失。每个 Attention 和 FFN 后都应有残差连接。</div>`;
      break;
    case 'dropout':
      html += `<div class="pt-cfg-group-title">Dropout 配置</div>`;
      html += num('dropout', 'Dropout 概率', 0, 0.5, 0.05, '训练时随机丢弃的比例。预训练通常设为 0，微调可设 0.1');
      break;
    case 'gate':
      html += `<div class="pt-cfg-group-title">门控层配置</div>`;
      html += `<div class="pt-cfg-hint">门控层 σ(Wx) ⊙ y 控制信息流通。常用于 Gated Transformer、GLU 变体等。</div>`;
      break;
    case 'skip_connect':
      html += `<div class="pt-cfg-group-title">跳跃连接</div>`;
      html += `<div class="pt-cfg-hint">跨多层的跳跃连接，将较浅层的输出直接传递给较深层。类似 DenseNet 的设计，有助于特征复用。</div>`;
      break;
    case 'linear':
      html += `<div class="pt-cfg-group-title">线性投影层</div>`;
      html += `<div class="pt-cfg-hint">自定义全连接层 (nn.Linear)。可用于：维度变换、瓶颈层、Adapter 层等。</div>`;
      html += num('hidden_dim', '输入/输出维度', 32, 8192, 64, '自定义线性层的维度');
      break;
    case 'kv_cache':
      html += `<div class="pt-cfg-group-title">KV Cache 标注</div>`;
      html += `<div class="pt-cfg-hint">这是推理优化标注。训练时会被忽略，推理时缓存每层的 K/V，避免对历史 token 重复计算注意力。</div>`;
      break;
    default:
      html += `<div class="pt-cfg-hint">此组件暂无额外配置选项</div>`;
  }

  // Global settings always visible
  html += `<div class="pt-cfg-group-title" style="margin-top:16px">全局设置</div>`;
  html += num('num_layers', 'Transformer 层数 (num_layers)', 1, 128, 1, '核心超参数。更多层 = 更强表达力 = 更多参数和显存');
  html += num('dropout', 'Dropout', 0, 0.5, 0.05, '训练时随机丢弃的比例。预训练通常设为 0');

  body.innerHTML = html;

  // Scroll to config
  section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

window.ptUpdateArch = function(field, value) {
  PT.arch[field] = value;

  // Auto-adjust kv_heads if attention_type changed
  if (field === 'attention_type') {
    if (value === 'mha') PT.arch.num_kv_heads = PT.arch.num_heads;
    else if (value === 'mqa') PT.arch.num_kv_heads = 1;
  }
  if (field === 'num_heads' && PT.arch.attention_type === 'mha') {
    PT.arch.num_kv_heads = PT.arch.num_heads;
  }

  updateBlocksFromArch();
  renderCanvas();
  scheduleStats();

  // Re-render the config panel for the same block type to update hints
  // Find which block is open
  const title = _$('#pt-config-title');
  if (title && _$('#pt-config-section').style.display !== 'none') {
    const openType = PT.blocks.find(b => title.textContent.includes(b.icon));
    if (openType) {
      const idx = PT.blocks.indexOf(openType);
      if (idx >= 0) ptClickBlock(idx);
    }
  }
};

// ======================= Drag & Drop =======================
function _setupCanvasDrag() {
  const blocks = _$$('#pt-canvas .pt-block[draggable="true"]');
  blocks.forEach(el => {
    el.addEventListener('dragstart', _onBlockDragStart);
    el.addEventListener('dragend', _onBlockDragEnd);
  });
  const canvas = _$('#pt-canvas');
  if (canvas) {
    canvas.addEventListener('dragover', _onCanvasDragOver);
    canvas.addEventListener('drop', _onCanvasDrop);
  }
}

function _onBlockDragStart(e) {
  PT.dragSrc = e.currentTarget;
  PT.dragFromPalette = false;
  e.currentTarget.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', e.currentTarget.dataset.idx);
}

function _onBlockDragEnd(e) {
  e.currentTarget.classList.remove('dragging');
  _$$('.pt-block').forEach(b => b.classList.remove('drag-over'));
  PT.dragSrc = null;
}

function _onCanvasDragOver(e) {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  const target = e.target.closest('.pt-block');
  _$$('.pt-block').forEach(b => b.classList.remove('drag-over'));
  if (target && target !== PT.dragSrc) {
    target.classList.add('drag-over');
  }
}

function _onCanvasDrop(e) {
  e.preventDefault();
  const target = e.target.closest('.pt-block');
  if (!target) return;

  if (PT.dragFromPalette) {
    // Add new block from palette
    const ctype = e.dataTransfer.getData('text/plain');
    const def = BLOCK_DEFS[ctype];
    if (!def) return;
    const newBlock = {
      id: `b-${ctype}-${Date.now()}`,
      type: ctype,
      label: def.label,
      icon: def.icon,
      locked: false,
    };
    const targetIdx = +target.dataset.idx;
    PT.blocks.splice(targetIdx + 1, 0, newBlock);
    renderCanvas();
    scheduleStats();
  } else if (PT.dragSrc) {
    // Reorder
    const fromIdx = +PT.dragSrc.dataset.idx;
    const toIdx = +target.dataset.idx;
    if (fromIdx === toIdx) return;
    const [moved] = PT.blocks.splice(fromIdx, 1);
    PT.blocks.splice(toIdx, 0, moved);
    renderCanvas();
  }
}

// Palette drag
function _setupPaletteDrag() {
  _$$('.pt-palette-item[draggable]').forEach(el => {
    el.addEventListener('dragstart', e => {
      PT.dragFromPalette = true;
      PT.dragSrc = null;
      e.dataTransfer.effectAllowed = 'copy';
      e.dataTransfer.setData('text/plain', el.dataset.ctype);
      el.classList.add('dragging');
    });
    el.addEventListener('dragend', e => {
      el.classList.remove('dragging');
      PT.dragFromPalette = false;
    });
  });
}

// ============ Pattern Insertion ============
window.ptInsertPattern = function(patternId) {
  const a = PT.arch;
  const normLabel = a.norm_type === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
  const t = Date.now();

  const patterns = {
    pre_norm_attn: [
      { type: 'norm_pre', label: normLabel, icon: '📏' },
      { type: 'attention', label: 'Self-Attention', icon: '👁️' },
      { type: 'residual', label: 'Residual Add', icon: '➰' },
    ],
    post_norm_block: [
      { type: 'attention', label: 'Self-Attention', icon: '👁️' },
      { type: 'norm_pre', label: `Post-${normLabel}`, icon: '📏' },
      { type: 'residual', label: 'Residual Add', icon: '➰' },
      { type: 'ffn', label: 'FFN', icon: '⚡' },
      { type: 'norm_post', label: `Post-${normLabel}`, icon: '📏' },
      { type: 'residual', label: 'Residual Add', icon: '➰' },
    ],
    moe_layer: [
      { type: 'norm_pre', label: normLabel, icon: '📏' },
      { type: 'attention', label: 'Self-Attention', icon: '👁️' },
      { type: 'residual', label: 'Residual Add', icon: '➰' },
      { type: 'norm_post', label: normLabel, icon: '📏' },
      { type: 'moe', label: 'MoE Layer', icon: '🧪' },
      { type: 'residual', label: 'Residual Add', icon: '➰' },
    ],
  };

  const blocks = patterns[patternId];
  if (!blocks) return;

  // Find insertion point (before norm_final or lm_head)
  let insertIdx = PT.blocks.findIndex(b => b.type === 'norm_final' || b.type === 'lm_head');
  if (insertIdx < 0) insertIdx = PT.blocks.length;

  // Detect if we're inserting inside a layer group
  const prevBlock = PT.blocks[insertIdx - 1];
  const groupId = prevBlock?.layerGroup || null;

  blocks.forEach((b, i) => {
    PT.blocks.splice(insertIdx + i, 0, {
      id: `b-${b.type}-${t}-${i}`,
      type: b.type,
      label: b.label,
      icon: b.icon,
      locked: false,
      layerGroup: groupId,
    });
  });

  renderCanvas();
  scheduleStats();
  ptToast(`✅ 已插入 ${blocks.length} 个组件`);
};

// ============ Architecture JSON Import/Export ============
window.ptExportArchJSON = function() {
  const data = {
    architecture: PT.arch,
    blocks: PT.blocks.map(b => ({ type: b.type, label: b.label, locked: b.locked, layerGroup: b.layerGroup })),
    layerGroups: PT.layerGroups,
    _exported: new Date().toISOString(),
  };
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `arch_${PT.arch.hidden_dim}d_${PT.arch.num_layers}L.json`;
  a.click();
  URL.revokeObjectURL(url);
  ptToast('📤 架构已导出');
};

window.ptImportArchJSON = function() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      if (data.architecture) {
        PT.arch = { ...PT.arch, ...data.architecture };
        if (data.blocks?.length) {
          PT.blocks = data.blocks.map((b, i) => ({
            id: `b-${b.type}-imp-${i}`,
            type: b.type,
            label: b.label || BLOCK_DEFS[b.type]?.label || b.type,
            icon: BLOCK_DEFS[b.type]?.icon || '❓',
            locked: b.locked || false,
            layerGroup: b.layerGroup || null,
          }));
        } else {
          updateBlocksFromArch();
        }
        if (data.layerGroups) PT.layerGroups = data.layerGroups;
        renderCanvas();
        updateStats();
        ptToast('📥 架构已导入');
      } else {
        ptToast('❌ 无效的架构文件');
      }
    } catch (err) {
      ptToast('❌ 解析失败: ' + err.message);
    }
  };
  input.click();
};

window.ptRemoveBlock = function(idx) {
  const b = PT.blocks[idx];
  if (b && !b.locked) {
    PT.blocks.splice(idx, 1);
    renderCanvas();
    scheduleStats();
  }
};

// ======================= Stats =======================
function scheduleStats() {
  clearTimeout(PT.statsTimer);
  PT.statsTimer = setTimeout(updateStats, 150);
}

async function updateStats() {
  try {
    const r = await fetch('/api/pretrain/arch/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ architecture: PT.arch }),
    });
    const d = await r.json();
    const e = d.estimated;

    _$('#pt-s-params').textContent = e.total_params_fmt;
    _$('#pt-s-vram-train').textContent = e.vram.train_fp16_fmt;
    _$('#pt-s-vram-infer').textContent = e.vram.inference_fp16_fmt;
    _$('#pt-s-emb').textContent = `${e.breakdown.embedding_fmt} (${e.info.embedding_pct})`;
    _$('#pt-s-perlayer').textContent = `${e.breakdown.per_layer_fmt} (${e.info.params_per_layer_pct})`;
    _$('#pt-s-headdim').textContent = e.info.head_dim;

    // Issues
    const issuesEl = _$('#pt-issues');
    if (issuesEl) {
      if (d.issues.length === 0) {
        issuesEl.innerHTML = '<div class="pt-issue-ok">✅ 架构配置有效</div>';
      } else {
        issuesEl.innerHTML = d.issues.map(i =>
          `<div class="pt-issue pt-issue-${i.level}">${i.level === 'error' ? '❌' : '⚠️'} ${esc(i.msg)}</div>`
        ).join('');
      }
    }

    // Color the params based on size
    const paramsEl = _$('#pt-s-params');
    if (paramsEl) {
      const total = e.total_params;
      if (total < 10_000_000) paramsEl.style.color = 'var(--ok)';
      else if (total < 200_000_000) paramsEl.style.color = 'var(--ac)';
      else if (total < 1_000_000_000) paramsEl.style.color = 'var(--warn)';
      else paramsEl.style.color = 'var(--err)';
    }
  } catch (e) { console.error('Stats error:', e); }
}

// ======================= Project Management =======================
async function ptLoadProjectList() {
  try {
    const r = await fetch('/api/pretrain/projects');
    const d = await r.json();
    PT.projects = d.projects;
    const sel = _$('#pt-project-sel');
    if (!sel) return;
    sel.innerHTML = '<option value="">— 选择项目 —</option>' +
      d.projects.map(p => `<option value="${p.id}" ${PT.project?.id === p.id ? 'selected' : ''}>${esc(p.name)} (${p.total_params_fmt})</option>`).join('');
  } catch (e) { console.error(e); }
}

async function ptLoadProject(pid) {
  if (!pid) { PT.project = null; _ptResetAllUI(); resetToDefault(); return; }
  try {
    const r = await fetch(`/api/pretrain/projects/${pid}`);
    const d = await r.json();
    PT.project = d;
    PT.arch = { ...d.architecture };
    updateBlocksFromArch();
    renderCanvas();
    updateStats();
    // Update select
    const sel = _$('#pt-project-sel');
    if (sel) sel.value = pid;
    // Show delete button
    const delBtn = _$('#btn-pt-del-proj');
    if (delBtn) delBtn.style.display = 'inline-flex';
    // Reset tab-specific content
    _ptResetAllUI();
    ptUpdateWorkflowStatus();
  } catch (e) { console.error(e); ptToast('加载项目失败'); }
}

/* Reset all tab UI when switching projects */
function _ptResetAllUI() {
  // Clear monitor
  const mon = _$('#pt-monitor-section'); if (mon) mon.style.display = 'none';
  const samp = _$('#pt-samples-section'); if (samp) samp.style.display = 'none';
  // Clear chat
  const chatMsgs = _$('#pt-chat-messages');
  if (chatMsgs) chatMsgs.innerHTML = '<div class="pt-chat-placeholder">选择一个项目和 checkpoint，然后在下方输入文本开始对话</div>';
  // Clear play result
  const playResult = _$('#pt-play-result'); if (playResult) playResult.style.display = 'none';
  const cmpResult = _$('#pt-cmp-result'); if (cmpResult) cmpResult.innerHTML = '';
  const expResult = _$('#pt-export-result'); if (expResult) expResult.style.display = 'none';
  // Clear tokenizer status
  const tokStatus = _$('#pt-tok-status'); if (tokStatus) tokStatus.style.display = 'none';
  // Clear dataset preview
  const dsPreview = _$('#pt-ds-preview-section'); if (dsPreview) dsPreview.style.display = 'none';
  // Clear processed info
  const procInfo = _$('#pt-processed-info'); if (procInfo) procInfo.innerHTML = '';
  // Clear proc result
  const procResult = _$('#pt-proc-result'); if (procResult) procResult.style.display = 'none';
  // Clear training history
  const histEl = _$('#pt-train-history'); if (histEl) histEl.innerHTML = '<p class="desc">暂无训练记录</p>';
  // Clear timeline
  const timeline = _$('#pt-timeline'); if (timeline) timeline.innerHTML = '';
  // Clear config section
  const cfgSection = _$('#pt-config-section'); if (cfgSection) cfgSection.style.display = 'none';
  // Reset loss history
  _lossHistory.length = 0;
  _valLossHistory.length = 0;
  // Reset column config
  PT.columnConfig = {};
  // Reset chat history
  if (typeof _chatHistory !== 'undefined') _chatHistory = [];
  // Reset train buttons
  const startBtn = _$('#btn-pt-start-train'); if (startBtn) { startBtn.style.display = 'inline-flex'; startBtn.disabled = false; }
  const stopBtn = _$('#btn-pt-stop-train'); if (stopBtn) stopBtn.style.display = 'none';
  // Clear train log
  const trainLog = _$('#pt-train-log'); if (trainLog) trainLog.textContent = '';
  // Clear ckpt list
  const ckptList = _$('#pt-ckpt-list'); if (ckptList) ckptList.innerHTML = '<p class="desc">尚无 checkpoint</p>';
}

// ======================= Training Mode Switch (Pretrain / SFT) =======================
function ptSwitchTrainMode(mode) {
  PT.trainMode = mode;
  const isSFT = mode === 'sft';

  // Toggle button active state via class (CSS handles styling)
  const btnPT = document.getElementById('pt-mode-pretrain');
  const btnSFT = document.getElementById('pt-mode-sft');
  if (btnPT) btnPT.classList.toggle('active', !isSFT);
  if (btnSFT) btnSFT.classList.toggle('active', isSFT);

  // Update hint text
  const hint = document.getElementById('pt-mode-hint');
  if (hint) {
    hint.innerHTML = isSFT
      ? '当前模式：<strong>SFT 对话微调</strong> — 教已预训练的模型学会一问一答的对话能力'
      : '当前模式：<strong>预训练</strong> — 用大量文本教模型学习语言规律';
  }

  // Data tab: switch headers
  const dph = _$('#pt-data-pretrain-head'); if (dph) dph.style.display = isSFT ? 'none' : '';
  const dsh = _$('#pt-data-sft-head'); if (dsh) dsh.style.display = isSFT ? '' : 'none';
  const sfg = _$('#pt-sft-format-guide'); if (sfg) sfg.style.display = isSFT ? '' : 'none';

  // Data tab: switch processing sections
  const pp = _$('#pt-proc-pretrain'); if (pp) pp.style.display = isSFT ? 'none' : '';
  const ps = _$('#pt-proc-sft'); if (ps) ps.style.display = isSFT ? '' : 'none';

  // Data tab: switch recommended datasets
  const rp = _$('#pt-rec-pretrain-section'); if (rp) rp.style.display = isSFT ? 'none' : '';
  const rs = _$('#pt-rec-sft-section'); if (rs) rs.style.display = isSFT ? '' : 'none';

  // Training tab: SFT config panel
  const sc = _$('#pt-tr-sft-cfg'); if (sc) sc.style.display = isSFT ? '' : 'none';

  // Chat tab: SFT badge and descriptions
  const cbd = _$('#pt-chat-sft-badge'); if (cbd) cbd.style.display = isSFT ? '' : 'none';
  const cdp = _$('#pt-chat-desc-pretrain'); if (cdp) cdp.style.display = isSFT ? 'none' : '';
  const cds = _$('#pt-chat-desc-sft'); if (cds) cds.style.display = isSFT ? '' : 'none';

  // Update chat template display
  if (isSFT) {
    const tmpl = _$('#pt-sft-template')?.value || 'chatml';
    const names = { chatml: 'ChatML', llama: 'Llama', simple: 'Simple' };
    const tn = _$('#pt-chat-template-name'); if (tn) tn.textContent = names[tmpl] || tmpl;
    const td = _$('#pt-tr-sft-template-display'); if (td) td.textContent = names[tmpl] || tmpl;
    // Load SFT recommended datasets if not loaded yet
    ptLoadSFTRecommendedDatasets();
    // Load checkpoints for base checkpoint selector
    if (PT.project) ptLoadBaseCheckpoints();
  }

  // Update training defaults for SFT
  if (isSFT) {
    const lr = _$('#pt-tr-lr'); if (lr && (lr.value === '3e-4' || lr.value === '0.0003')) lr.value = '2e-5';
    const steps = _$('#pt-tr-steps'); if (steps && steps.value === '5000') steps.value = '1000';
    const bs = _$('#pt-tr-bs'); if (bs && bs.value === '32') bs.value = '8';
    const prompts = _$('#pt-tr-prompts'); if (prompts && prompts.value.includes('Once upon')) prompts.value = '你好,请介绍一下你自己,What is AI?';
  } else {
    const lr = _$('#pt-tr-lr'); if (lr && (lr.value === '2e-5' || lr.value === '0.00002')) lr.value = '3e-4';
    const steps = _$('#pt-tr-steps'); if (steps && steps.value === '1000') steps.value = '5000';
    const bs = _$('#pt-tr-bs'); if (bs && bs.value === '8') bs.value = '32';
  }

  console.log('[PretrainLab] mode switched to:', mode);

  // SFT mode: hide Architecture & Tokenizer tabs (not needed for SFT)
  _$$('.pt-tab').forEach(tab => {
    const tabId = tab.dataset.tab;
    if (tabId === 'pt-arch' || tabId === 'pt-token') {
      tab.style.display = isSFT ? 'none' : '';
    }
  });
  // If currently on a hidden tab, switch to data tab
  if (isSFT) {
    const activeTab = _$('.pt-tab.active');
    if (activeTab && (activeTab.dataset.tab === 'pt-arch' || activeTab.dataset.tab === 'pt-token')) {
      const dataTab = _$('.pt-tab[data-tab="pt-data"]');
      if (dataTab) dataTab.click();
    }
  }

  // Reload checkpoints for the new mode
  if (PT.project) ptLoadCheckpoints();
}

async function ptLoadSFTRecommendedDatasets() {
  const grid = _$('#pt-rec-sft-datasets');
  if (!grid || grid.children.length > 0) return; // already loaded
  try {
    const r = await fetch('/api/pretrain/datasets/recommended-sft');
    const d = await r.json();
    grid.innerHTML = (d.datasets || []).map(ds => `
      <div class="pt-rec-card">
        <div class="pt-rec-name">${ds.name}</div>
        <div class="pt-rec-desc">${ds.description}</div>
        <div class="pt-rec-meta">
          <span>${ds.lang === 'zh' ? '🇨🇳' : '🇬🇧'} ${ds.lang}</span>
          <span>📦 ${ds.size}</span>
          <span>📊 ${ds.rows}</span>
          <span class="pt-rec-diff pt-diff-${ds.difficulty === '入门' ? 'easy' : ds.difficulty === '中等' ? 'med' : 'hard'}">${ds.difficulty}</span>
        </div>
        ${ds.format_hint ? `<div style="font-size:10px;color:var(--t3);margin-top:4px">格式: ${ds.format_hint}</div>` : ''}
        <div style="font-size:10px;color:var(--t3);margin-top:2px">${ds.recommended_for}</div>
        <button class="btn-p pt-rec-dl" style="margin-top:6px;font-size:11px;padding:3px 10px" onclick="ptQuickDownloadHF('${ds.id}')">📥 下载</button>
      </div>
    `).join('');
  } catch (e) {
    grid.innerHTML = '<p class="desc">加载推荐数据集失败</p>';
  }
}

async function ptLoadBaseCheckpoints() {
  const sel = _$('#pt-tr-base-ckpt');
  if (!sel || !PT.project) return;
  try {
    // Always load PRETRAIN checkpoints for base model selection
    const r = await fetch(`/api/pretrain/train/checkpoints/${PT.project.id}?mode=pretrain`);
    const d = await r.json();
    sel.innerHTML = '<option value="">不加载（从随机初始化开始）</option>';
    (d.checkpoints || []).forEach(ck => {
      sel.innerHTML += `<option value="${ck.path}">[预训练] ${ck.name} (step ${ck.step})</option>`;
    });
  } catch (e) {}
}

// SFT data processing
async function ptProcessSFT() {
  if (!PT.project) { ptToast('请先选择项目'); return; }

  // Check tokenizer first
  try {
    const tr = await fetch(`/api/pretrain/tokenizer/project/${PT.project.id}`);
    const td = await tr.json();
    if (!td.has_tokenizer) {
      ptToast('⚠️ 请先到「词表构建」Tab 配置 Tokenizer');
      return;
    }
  } catch (e) { /* proceed anyway */ }

  const seqLen = parseInt(_$('#pt-sft-proc-seqlen')?.value || '512');
  const template = _$('#pt-sft-template')?.value || 'chatml';

  // Collect selected files
  const selectedFiles = [];
  _$$('#pt-sft-proc-file-list input[type=checkbox]:checked').forEach(cb => {
    selectedFiles.push(cb.value);
  });
  if (!selectedFiles.length) { ptToast('请至少选择一个对话数据文件'); return; }

  const resultEl = _$('#pt-proc-result');
  resultEl.style.display = 'block';
  resultEl.innerHTML = '<div class="pt-progress-box"><div class="pt-progress-bar"><div class="pt-progress-fill" id="pt-sft-proc-fill"></div></div><div id="pt-sft-proc-log" class="pt-progress-log">开始 SFT 预处理...</div></div>';

  try {
    const res = await fetch(`/api/pretrain/datasets/${PT.project.id}/process-sft`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_seq_len: seqLen, python_path: ptPy(), chat_template: template, files: selectedFiles }),
    });
    // Handle HTTP errors (not SSE)
    if (!res.ok) {
      let errMsg = `HTTP ${res.status}`;
      try {
        const errBody = await res.json();
        errMsg = errBody.detail || errBody.error || errMsg;
      } catch (e) {}
      resultEl.innerHTML = `<div style="padding:12px;background:rgba(239,68,68,.1);border-radius:8px;border:1px solid rgba(239,68,68,.3);color:var(--err)">❌ ${errMsg}</div>`;
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        const m = line.match(/^data:\s*(.+)/);
        if (!m) continue;
        try {
          const d = JSON.parse(m[1]);
          const fill = _$('#pt-sft-proc-fill');
          const log = _$('#pt-sft-proc-log');
          if (fill && d.progress !== undefined) fill.style.width = d.progress + '%';
          if (log && d.step) log.textContent = d.step;
          if (d.done) {
            const vocabInfo = d.vocab_expanded
              ? `<br><span style="color:var(--accent)">✨ 已自动注入特殊 token: ${(d.added_special_tokens||[]).join(', ')} (词表 ${d.original_vocab_size} → ${d.vocab_size})</span>`
              : '';
            resultEl.innerHTML = `<div style="padding:12px;background:rgba(74,222,128,.1);border-radius:8px;border:1px solid rgba(74,222,128,.3)">
              <strong>✅ SFT 数据预处理完成</strong><br>
              <span class="desc">对话数: ${d.total_conversations?.toLocaleString() || '?'} | 训练 tokens: ${d.assistant_tokens?.toLocaleString() || '?'} (${d.assistant_pct || '?'}%) | 词表: ${d.vocab_size} | 模板: ${template}</span>
              ${vocabInfo}
            </div>`;
            ptLoadProcessedInfo();
          }
          if (d.error) {
            resultEl.innerHTML = `<div style="padding:12px;background:rgba(239,68,68,.1);border-radius:8px;border:1px solid rgba(239,68,68,.3);color:var(--err)">❌ ${d.step || d.error || '处理失败'}</div>`;
          }
        } catch (e) {}
      }
    }
  } catch (e) {
    resultEl.innerHTML = `<div style="color:var(--err)">❌ ${e.message}</div>`;
  }
}

// Convert dataset to SFT format
function ptLoadProcessedInfo() { ptLoadDatasets(); }

async function ptConvertToSFT() {
  if (!PT.project) { ptToast('请先选择项目'); return; }

  // Get all dataset files
  const dsListEl = _$('#pt-ds-list');
  const files = dsListEl?.querySelectorAll('[data-filename]');
  if (!files || files.length === 0) { ptToast('没有可转换的数据集文件'); return; }

  // Build list of convertible files (json, jsonl, txt, csv)
  const convertible = [];
  files.forEach(f => {
    const name = f.dataset.filename;
    if (name && !name.includes('_sft.jsonl')) {
      convertible.push(name);
    }
  });
  if (!convertible.length) { ptToast('没有可转换的文件'); return; }

  // Let user pick
  const targetFile = convertible.length === 1
    ? convertible[0]
    : prompt(`选择要转换的文件:\n${convertible.map((f,i) => `${i+1}. ${f}`).join('\n')}\n\n输入编号:`, '1');

  let fileName;
  if (convertible.length === 1) {
    fileName = convertible[0];
  } else {
    const idx = parseInt(targetFile) - 1;
    if (isNaN(idx) || idx < 0 || idx >= convertible.length) { ptToast('无效选择'); return; }
    fileName = convertible[idx];
  }

  try {
    const r = await fetch(`/api/pretrain/datasets/${PT.project.id}/convert-to-sft`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: fileName }),
    });
    const d = await r.json();
    if (d.status === 'ok') {
      ptToast(`✅ 转换成功！${d.total_conversations} 条对话 → ${d.output_file}`);
      ptLoadDatasets();
    } else {
      ptToast('❌ ' + (d.detail || '转换失败'), 'error');
    }
  } catch (e) {
    ptToast('❌ ' + e.message, 'error');
  }
}

async function ptCreateProject() {
  const name = _$('#pt-new-name').value.trim();
  if (!name) { ptToast('请输入项目名称'); return; }
  const tid = _$('#pt-new-template').value;
  try {
    const r = await fetch('/api/pretrain/projects/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, template_id: tid }),
    });
    const d = await r.json();
    PT.project = d.project;
    PT.arch = { ...d.project.architecture };
    _ptResetAllUI();
    updateBlocksFromArch();
    renderCanvas();
    updateStats();
    _$('#ov-pt-new').style.display = 'none';
    await ptLoadProjectList();
    // Show delete button
    const delBtn = _$('#btn-pt-del-proj');
    if (delBtn) delBtn.style.display = 'inline-flex';
    ptToast(`✅ 项目 "${name}" 已创建`);
    ptUpdateWorkflowStatus();
  } catch (e) { console.error(e); ptToast('创建失败'); }
}

async function ptSaveProjectArch() {
  if (!PT.project) { ptToast('请先创建或选择一个项目'); return; }
  try {
    const r = await fetch(`/api/pretrain/projects/${PT.project.id}/arch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ architecture: PT.arch }),
    });
    const d = await r.json();
    if (d.saved) {
      PT.project.architecture = { ...PT.arch };
      ptToast('✅ 架构已保存');
      ptLoadProjectList();
    }
  } catch (e) { console.error(e); ptToast('保存失败'); }
}

// Auto-save without toast (silent save on arch changes)
async function ptAutoSaveArch() {
  if (!PT.project) return;
  try {
    await fetch(`/api/pretrain/projects/${PT.project.id}/arch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ architecture: PT.arch }),
    });
    PT.project.architecture = { ...PT.arch };
  } catch (e) { console.error('Auto-save failed:', e); }
}

// ============ Workflow Status (progress indicator) ============
async function ptUpdateWorkflowStatus() {
  if (!PT.project) return;
  const el = _$('#pt-workflow-status');
  if (!el) return;

  // Check each step
  let tokOk = false, dataOk = false, procOk = false, trainOk = false;

  try {
    // Check tokenizer
    const tokR = await fetch(`/api/pretrain/tokenizer/project/${PT.project.id}`);
    const tokD = await tokR.json();
    tokOk = !!tokD.has_tokenizer;

    // Check datasets & processed
    const dsR = await fetch(`/api/pretrain/datasets/${PT.project.id}`);
    const dsD = await dsR.json();
    dataOk = dsD.datasets?.length > 0;
    procOk = dsD.processed?.length > 0;

    // Check checkpoints
    const mode = PT.trainMode;
    const ckR = await fetch(`/api/pretrain/train/checkpoints/${PT.project.id}?mode=${mode}`);
    const ckD = await ckR.json();
    trainOk = ckD.checkpoints?.length > 0;
  } catch (e) { /* silent */ }

  const steps = [
    { key: 'arch', label: '架构', ok: true, tab: 'pt-arch', icon: '🏗️' },
    { key: 'tok', label: '分词器', ok: tokOk, tab: 'pt-token', icon: '📝' },
    { key: 'data', label: '数据', ok: dataOk, tab: 'pt-data', icon: '📊' },
    { key: 'proc', label: '预处理', ok: procOk, tab: 'pt-data', icon: '⚙️' },
    { key: 'train', label: '训练', ok: trainOk, tab: 'pt-train', icon: '🚀' },
  ];

  el.innerHTML = steps.map((s, i) => {
    const statusClass = s.ok ? 'done' : 'pending';
    const arrow = i < steps.length - 1 ? '<span class="pt-wf-arrow">→</span>' : '';
    return `<span class="pt-wf-step ${statusClass}" onclick="switchPTTab('${s.tab}')" title="${s.label}: ${s.ok ? '已完成' : '待完成'}">
      ${s.icon} <span class="pt-wf-label">${s.label}</span>${s.ok ? ' ✓' : ''}
    </span>${arrow}`;
  }).join('');
}

// ============ Pre-flight Checks Before Training ============
async function ptPreflightCheck() {
  const issues = [];

  // 1. Tokenizer?
  try {
    const r = await fetch(`/api/pretrain/tokenizer/project/${PT.project.id}`);
    const d = await r.json();
    if (!d.has_tokenizer) {
      issues.push('❌ 未配置 Tokenizer。请先到「词表构建」Tab 下载或训练分词器。');
    } else if (d.config?.vocab_size && d.config.vocab_size !== PT.arch.vocab_size) {
      issues.push(`⚠️ Tokenizer 词表大小 (${d.config.vocab_size.toLocaleString()}) 与架构 vocab_size (${PT.arch.vocab_size.toLocaleString()}) 不一致。将自动同步。`);
      PT.arch.vocab_size = d.config.vocab_size;
      ptAutoSaveArch();
    }
  } catch (e) { /* skip */ }

  // 2. Processed data?
  try {
    const r = await fetch(`/api/pretrain/datasets/${PT.project.id}`);
    const d = await r.json();
    if (!d.processed?.length) {
      issues.push('❌ 没有预处理过的训练数据。请先到「数据准备」Tab 上传数据并预处理。');
    } else {
      // Check seq_len match
      const proc = d.processed[0];
      if (proc.max_seq_len && proc.max_seq_len !== PT.arch.max_seq_len) {
        issues.push(`⚠️ 预处理的 max_seq_len (${proc.max_seq_len}) 与架构 (${PT.arch.max_seq_len}) 不一致。训练将使用预处理的值。`);
      }
    }
  } catch (e) { /* skip */ }

  // 3. Architecture validation errors?
  try {
    const r = await fetch('/api/pretrain/arch/validate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ architecture: PT.arch }),
    });
    const d = await r.json();
    const errors = (d.issues || []).filter(i => i.level === 'error');
    if (errors.length > 0) {
      issues.push(`❌ 架构配置有误: ${errors[0].msg}`);
    }
  } catch (e) { /* skip */ }

  return issues;
}

// ======================= Tab Switching =======================
let _ptLastTab = 'pt-arch';
function switchPTTab(tab) {
  // Auto-save arch when leaving arch tab
  if (_ptLastTab === 'pt-arch' && tab !== 'pt-arch' && PT.project) {
    ptAutoSaveArch();
  }
  _ptLastTab = tab;
  _$$('.pt-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  _$$('.pt-pane').forEach(p => p.classList.toggle('active', p.id === tab));
}

// ======================= Escape helper =======================
// esc() is provided by app.js (global scope) — do NOT redeclare here

// Re-use toast from app.js
function ptToast(m) {
  // app.js defines a global toast — but we can't call it directly since it's in function scope
  let t = document.querySelector('.toast');
  if (!t) { t = document.createElement('div'); t.className = 'toast'; document.body.appendChild(t); }
  t.textContent = m; t.classList.add('show');
  clearTimeout(ptToast._tt); ptToast._tt = setTimeout(() => t.classList.remove('show'), 2500);
}

// ======================= Init Event Listeners =======================
document.addEventListener('DOMContentLoaded', () => {
  // Tab switching
  _$$('.pt-tab').forEach(t => t.addEventListener('click', () => {
    switchPTTab(t.dataset.tab);
    // Load data when switching to specific tabs
    if (t.dataset.tab === 'pt-token') ptInitTokenTab();
    if (t.dataset.tab === 'pt-data') ptInitDataTab();
    if (t.dataset.tab === 'pt-train') ptInitTrainTab();
    if (t.dataset.tab === 'pt-play') ptInitPlayTab();
    if (t.dataset.tab === 'pt-chat') ptInitChatTab();
  }));

  // Close button
  const clBtn = _$('#cl-pretrain');
  if (clBtn) clBtn.onclick = () => { _$('#ov-pretrain').style.display = 'none'; document.body.style.overflow = ''; };

  // Project selector
  const pSel = _$('#pt-project-sel');
  if (pSel) pSel.onchange = () => ptLoadProject(pSel.value);

  // New project dialog
  const newBtn = _$('#btn-pt-new-proj');
  if (newBtn) newBtn.onclick = () => { _$('#ov-pt-new').style.display = 'flex'; _$('#pt-new-name').value = ''; _$('#pt-new-name').focus(); };
  const clNew = _$('#cl-pt-new');
  if (clNew) clNew.onclick = () => { _$('#ov-pt-new').style.display = 'none'; };
  const createBtn = _$('#btn-pt-create');
  if (createBtn) createBtn.onclick = ptCreateProject;
  const nameInput = _$('#pt-new-name');
  if (nameInput) nameInput.onkeydown = e => { if (e.key === 'Enter') ptCreateProject(); };

  // Save architecture
  const saveBtn = _$('#btn-pt-save-arch');
  if (saveBtn) saveBtn.onclick = ptSaveProjectArch;

  // Palette drag setup
  _setupPaletteDrag();

  // Close overlay on click outside
  const ovNew = _$('#ov-pt-new');
  if (ovNew) ovNew.onclick = e => { if (e.target === ovNew) ovNew.style.display = 'none'; };

  // === Tokenizer tab ===
  const tokDl = _$('#btn-pt-tok-download');
  if (tokDl) tokDl.onclick = ptDownloadTokenizer;
  const tokTrain = _$('#btn-pt-tok-train');
  if (tokTrain) tokTrain.onclick = ptTrainTokenizer;
  const tokPreview = _$('#btn-pt-tok-preview');
  if (tokPreview) tokPreview.onclick = ptPreviewTokens;

  // === Dataset tab ===
  const dataFile = _$('#pt-data-file');
  if (dataFile) dataFile.onchange = ptUploadFiles;
  const pasteBtn = _$('#btn-pt-paste');
  if (pasteBtn) pasteBtn.onclick = () => { _$('#pt-paste-panel').style.display = 'block'; _$('#pt-hf-panel').style.display = 'none'; };
  const pasteSave = _$('#btn-pt-paste-save');
  if (pasteSave) pasteSave.onclick = ptSavePastedText;
  const pasteCancel = _$('#btn-pt-paste-cancel');
  if (pasteCancel) pasteCancel.onclick = () => { _$('#pt-paste-panel').style.display = 'none'; };
  const hfBtn = _$('#btn-pt-hf-dl');
  if (hfBtn) hfBtn.onclick = () => { _$('#pt-hf-panel').style.display = 'block'; _$('#pt-paste-panel').style.display = 'none'; };
  const hfStart = _$('#btn-pt-hf-start');
  if (hfStart) hfStart.onclick = ptDownloadHFDataset;
  const dsRefresh = _$('#btn-pt-ds-refresh');
  if (dsRefresh) dsRefresh.onclick = ptLoadDatasets;
  const procBtn = _$('#btn-pt-process');
  if (procBtn) procBtn.onclick = ptProcessDataset;
  const appendBtn = _$('#btn-pt-process-append');
  if (appendBtn) appendBtn.onclick = () => {
    if (confirm('将使用所有勾选的数据文件重新预处理。之前的训练 Checkpoint 仍然保留，可以继续训练（断点续训）。\n\n确定继续？')) {
      ptProcessDataset();
    }
  };

  // SFT data processing
  const procSftBtn = _$('#btn-pt-process-sft');
  if (procSftBtn) procSftBtn.onclick = ptProcessSFT;
  const appendSftBtn = _$('#btn-pt-process-sft-append');
  if (appendSftBtn) appendSftBtn.onclick = () => {
    if (confirm('将使用所有勾选的对话数据文件重新预处理。之前的 SFT Checkpoint 仍然保留，可以继续训练（断点续训）。\n\n确定继续？')) {
      ptProcessSFT();
    }
  };
  const convertBtn = _$('#btn-pt-convert-sft');
  if (convertBtn) convertBtn.onclick = ptConvertToSFT;
  // SFT template change → update displays
  const sftTmpl = _$('#pt-sft-template');
  if (sftTmpl) sftTmpl.onchange = () => {
    const names = { chatml: 'ChatML', llama: 'Llama', simple: 'Simple' };
    const name = names[sftTmpl.value] || sftTmpl.value;
    const tn = _$('#pt-chat-template-name'); if (tn) tn.textContent = name;
    const td = _$('#pt-tr-sft-template-display'); if (td) td.textContent = name;
  };

  // === Training tab ===
  const startTrain = _$('#btn-pt-start-train');
  if (startTrain) startTrain.onclick = ptStartTraining;
  const stopTrain = _$('#btn-pt-stop-train');
  if (stopTrain) stopTrain.onclick = ptStopTraining;
  const refreshCkpts = _$('#btn-pt-refresh-ckpts');
  if (refreshCkpts) refreshCkpts.onclick = ptLoadCheckpoints;

  // === Playground tab ===
  const genBtn = _$('#btn-pt-generate');
  if (genBtn) genBtn.onclick = ptGenerate;
  const cmpBtn = _$('#btn-pt-compare');
  if (cmpBtn) cmpBtn.onclick = ptCompare;
  const expGguf = _$('#btn-pt-export-gguf');
  if (expGguf) expGguf.onclick = ptExportGGUF;

  // === Chat tab ===
  const chatSend = _$('#btn-pt-chat-send');
  if (chatSend) chatSend.onclick = ptChatSend;
  const chatClear = _$('#btn-pt-chat-clear');
  if (chatClear) chatClear.onclick = ptChatClear;

  // === Delete project ===
  const delBtn = _$('#btn-pt-del-proj');
  if (delBtn) delBtn.onclick = ptDeleteProject;
});

// Make showPretrainLab globally accessible
window.showPretrainLab = showPretrainLab;
window.ptSwitchTrainMode = ptSwitchTrainMode;
window.ptQuickDownloadHF = ptQuickDownloadHF;


/* ============================================================
   TOKENIZER TAB
   ============================================================ */
const TOKEN_COLORS = [
  '#e74c3c33','#3498db33','#2ecc7133','#f39c1233','#9b59b633',
  '#1abc9c33','#e67e2233','#2980b933','#27ae6033','#c0392b33',
  '#8e44ad33','#d3541333','#16a08533','#f1c40f33','#e84393aa',
];

async function ptInitTokenTab() {
  await ptLoadTokenizerList();
  ptCheckTokenizerStatus();
  // Populate python env selector (same as train tab)
  ptLoadEnvsInto('#pt-tok-python');
}

async function ptLoadTokenizerList() {
  try {
    const r = await fetch('/api/pretrain/tokenizer/list');
    const d = await r.json();
    const el = _$('#pt-tok-list');
    if (!el) return;
    el.innerHTML = d.tokenizers.map(t => `
      <div class="pt-tok-card" data-source="${esc(t.source)}" onclick="ptSelectPretrained('${esc(t.source)}', '${esc(t.id)}')">
        <div class="pt-tok-card-name">${esc(t.name)}</div>
        <div class="pt-tok-card-info">
          <span>词表: ${t.vocab_size.toLocaleString()}</span>
        </div>
        <div class="pt-tok-card-desc">${esc(t.description)}</div>
        <div class="pt-tok-card-langs">${t.languages.map(l => `<span class="pt-tok-lang">${l}</span>`).join('')}</div>
      </div>
    `).join('');
  } catch (e) { console.error(e); }
}

window.ptSelectPretrained = function(source, id) {
  _$$('.pt-tok-card').forEach(c => c.classList.toggle('active', c.dataset.source === source));
  const inp = _$('#pt-tok-custom-src');
  if (inp) inp.value = source;
};

window.ptSwitchTokMode = function(mode) {
  _$$('.pt-tok-mode').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  _$('#pt-tok-pretrained').style.display = mode === 'pretrained' ? 'block' : 'none';
  _$('#pt-tok-custom').style.display = mode === 'custom' ? 'block' : 'none';
};

/* ============ Shared SSE reader for progress bars ============ */
async function _ptReadSSE(url, body, containerEl, fillId, logId, onDone) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let errMsg = `HTTP ${r.status}`;
      try { const errBody = await r.json(); errMsg = errBody.detail || errBody.error || errMsg; } catch {}
      containerEl.className = 'pt-tok-status error';
      const log = document.getElementById(logId);
      if (log) log.innerHTML += `<div style="color:var(--err)">❌ ${esc(errMsg)}</div>`;
      return;
    }
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
          const ev = JSON.parse(line.slice(6));
          // Update progress bar
          const fill = document.getElementById(fillId);
          if (fill && ev.progress >= 0) fill.style.width = ev.progress + '%';
          // Update log
          const log = document.getElementById(logId);
          if (log && ev.step) {
            log.innerHTML += `<div>${esc(ev.step)}</div>`;
            log.scrollTop = log.scrollHeight;
          }
          // Error
          if (ev.error) {
            containerEl.className = 'pt-tok-status error';
            if (fill) fill.style.background = 'var(--err)';
          }
          // Done callback
          if (ev.done && onDone) onDone(ev);
        } catch {}
      }
    }
  } catch (e) {
    containerEl.className = 'pt-tok-status error';
    const log = document.getElementById(logId);
    if (log) log.innerHTML += `<div style="color:var(--err)">❌ ${esc(e.message)}</div>`;
  }
}

async function ptDownloadTokenizer() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const source = _$('#pt-tok-custom-src')?.value?.trim();
  if (!source) { ptToast('请选择或输入 tokenizer 来源'); return; }

  const statusEl = _$('#pt-tok-status');
  statusEl.style.display = 'block';
  statusEl.className = 'pt-tok-status';
  statusEl.innerHTML = `<div class="pt-progress-bar"><div class="pt-progress-fill" id="pt-tok-fill"></div></div><div id="pt-tok-log" class="pt-progress-log">⏳ 准备下载...</div>`;

  await _ptReadSSE('/api/pretrain/tokenizer/download',
    { project_id: PT.project.id, source, python_path: ptPy() },
    statusEl, 'pt-tok-fill', 'pt-tok-log',
    (ev) => {
      if (ev.done) {
        statusEl.className = 'pt-tok-status ok';
        let html = `✅ Tokenizer 已下载！词表大小: <b>${(ev.vocab_size||0).toLocaleString()}</b>`;
        if (ev.test_results?.length) {
          html += `<div style="margin-top:8px"><b>分词测试:</b></div>`;
          ev.test_results.forEach(t => {
            html += `<div style="margin-top:4px;font-size:11px;color:var(--t3)">"${esc(t.text)}" → <b>${t.length}</b> tokens</div>`;
          });
        }
        // Auto-sync vocab_size to architecture
        if (ev.vocab_size && PT.arch && ev.vocab_size !== PT.arch.vocab_size) {
          const oldV = PT.arch.vocab_size;
          PT.arch.vocab_size = ev.vocab_size;
          html += `<div style="margin-top:8px;padding:8px;background:rgba(74,222,128,.08);border-radius:6px;border:1px solid rgba(74,222,128,.2)">
            <span style="color:var(--ac)">🔄 已自动同步：</span> 架构 vocab_size ${oldV?.toLocaleString()} → <b>${ev.vocab_size.toLocaleString()}</b>
          </div>`;
          ptAutoSaveArch();
          updateBlocksFromArch();
          renderCanvas();
          scheduleStats();
        }
        statusEl.innerHTML = html;
        ptToast('✅ Tokenizer 下载成功');
        ptUpdateWorkflowStatus();
      }
    }
  );
}

async function ptTrainTokenizer() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const vocabSize = +((_$('#pt-tok-vocab')?.value) || 8000);
  const minFreq = +((_$('#pt-tok-minfreq')?.value) || 2);

  const statusEl = _$('#pt-tok-status');
  statusEl.style.display = 'block';
  statusEl.className = 'pt-tok-status';
  statusEl.innerHTML = `<div class="pt-progress-bar"><div class="pt-progress-fill" id="pt-tok-train-fill"></div></div><div id="pt-tok-train-log" class="pt-progress-log">⏳ 准备训练...</div>`;

  await _ptReadSSE('/api/pretrain/tokenizer/train',
    { project_id: PT.project.id, vocab_size: vocabSize, min_frequency: minFreq, python_path: ptPy() },
    statusEl, 'pt-tok-train-fill', 'pt-tok-train-log',
    (ev) => {
      if (ev.done) {
        statusEl.className = 'pt-tok-status ok';
        let html = `✅ Tokenizer 训练完成！词表大小: <b>${(ev.vocab_size||0).toLocaleString()}</b>`;
        if (ev.sample_tokens?.length) {
          html += `<div style="margin-top:8px"><b>词表样本 (前50):</b></div>`;
          html += `<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">${ev.sample_tokens.map(t => `<code style="font-size:10px;padding:1px 4px;background:var(--bg0);border-radius:3px">${esc(t)}</code>`).join('')}</div>`;
        }
        if (ev.test_results?.length) {
          html += `<div style="margin-top:8px"><b>分词测试:</b></div>`;
          ev.test_results.forEach(t => {
            html += `<div style="margin-top:4px;font-size:11px;color:var(--t3)">"${esc(t.text)}" → <b>${t.length}</b> tokens</div>`;
          });
        }
        statusEl.innerHTML = html;
        ptToast('✅ Tokenizer 训练成功');
        // Auto-sync vocab_size to architecture
        if (ev.vocab_size && PT.arch && ev.vocab_size !== PT.arch.vocab_size) {
          const oldV = PT.arch.vocab_size;
          PT.arch.vocab_size = ev.vocab_size;
          statusEl.innerHTML += `<div style="margin-top:8px;padding:8px;background:rgba(74,222,128,.08);border-radius:6px;border:1px solid rgba(74,222,128,.2)">
            <span style="color:var(--ac)">🔄 已自动同步：</span> 架构 vocab_size ${oldV?.toLocaleString()} → <b>${ev.vocab_size.toLocaleString()}</b>
          </div>`;
          ptAutoSaveArch();
          updateBlocksFromArch();
          renderCanvas();
          scheduleStats();
        }
        ptUpdateWorkflowStatus();
      }
    }
  );
}

async function ptPreviewTokens() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const text = _$('#pt-tok-input')?.value || '';
  if (!text.trim()) { ptToast('请输入文本'); return; }

  try {
    const r = await fetch('/api/pretrain/tokenizer/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: PT.project.id, text, python_path: ptPy() }),
    });
    const d = await r.json();

    const resultEl = _$('#pt-tok-preview-result');
    resultEl.style.display = 'block';

    if (d.error) {
      _$('#pt-tok-tokens').innerHTML = `<div style="color:var(--t3);font-size:12px">${esc(d.error)}</div>`;
      if (d.simple_stats) {
        _$('#pt-tok-preview-stats').innerHTML = `<span>字符数: <b>${d.simple_stats.chars}</b></span><span>估算 tokens: <b>~${d.simple_stats.est_tokens_rough}</b></span><span>语言: <b>${d.simple_stats.primary_lang}</b></span>`;
      }
      return;
    }

    // Render colored tokens
    const tokens = d.tokens || [];
    const tokensHtml = tokens.map((tok, i) => {
      const color = TOKEN_COLORS[i % TOKEN_COLORS.length];
      const displayTok = tok.replace(/Ġ/g, '⎵').replace(/▁/g, '⎵').replace(/Ċ/g, '↵');
      return `<span class="pt-tok-chip" style="background:${color}" title="ID: ${d.ids?.[i] ?? '?'}">${esc(displayTok)}</span>`;
    }).join('');
    _$('#pt-tok-tokens').innerHTML = tokensHtml;

    _$('#pt-tok-preview-stats').innerHTML = `
      <span>Token 数: <b>${d.length}</b></span>
      <span>字符数: <b>${text.length}</b></span>
      <span>压缩率: <b>${(text.length / (d.length || 1)).toFixed(2)}</b> 字符/token</span>
    `;
  } catch (e) {
    console.error(e);
    ptToast('预览失败');
  }
}

async function ptCheckTokenizerStatus() {
  if (!PT.project) return;
  try {
    const r = await fetch(`/api/pretrain/tokenizer/project/${PT.project.id}`);
    const d = await r.json();
    if (d.has_tokenizer) {
      const statusEl = _$('#pt-tok-status');
      statusEl.style.display = 'block';
      statusEl.className = 'pt-tok-status ok';
      const cfg = d.config || {};
      statusEl.innerHTML = `✅ 已配置 Tokenizer: <b>${esc(cfg.type === 'custom_bpe' ? '自定义 BPE' : cfg.source || '未知')}</b>，词表大小: <b>${cfg.vocab_size?.toLocaleString() || '?'}</b>`;
    }
  } catch (e) { console.error(e); }
}


/* ============================================================
   DATASET TAB
   ============================================================ */
async function ptInitDataTab() {
  ptLoadDatasets();
  ptLoadRecommendedDatasets();
  // Sync seq_len from architecture to processing inputs
  if (PT.arch?.max_seq_len) {
    const archSeq = String(PT.arch.max_seq_len);
    const seqEl = _$('#pt-proc-seqlen');
    if (seqEl && (seqEl.value === '512' || seqEl.value !== archSeq)) {
      seqEl.value = PT.arch.max_seq_len;
    }
    const sftSeqEl = _$('#pt-sft-proc-seqlen');
    if (sftSeqEl && (sftSeqEl.value === '512' || sftSeqEl.value !== archSeq)) {
      sftSeqEl.value = PT.arch.max_seq_len;
    }
  }
}

async function ptLoadDatasets() {
  if (!PT.project) {
    _$('#pt-ds-list').innerHTML = '<p class="desc">请先选择或创建一个项目</p>';
    return;
  }
  try {
    const r = await fetch(`/api/pretrain/datasets/${PT.project.id}`);
    const d = await r.json();
    const el = _$('#pt-ds-list');

    if (!d.datasets?.length) {
      el.innerHTML = '<p class="desc">暂无数据集，请上传文件或粘贴文本</p>';
    } else {
      el.innerHTML = d.datasets.map(ds => {
        const icon = ds.ext === '.txt' ? '📄' : ds.ext === '.jsonl' ? '📋' : '📊';
        return `
          <div class="pt-ds-item" data-filename="${esc(ds.name)}" onclick="ptPreviewDataset('${esc(ds.name)}')">
            <div class="pt-ds-icon">${icon}</div>
            <div class="pt-ds-info">
              <div class="pt-ds-name">${esc(ds.name)}</div>
              <div class="pt-ds-meta"><span>${esc(ds.size_fmt)}</span><span>${esc(ds.ext)}</span></div>
            </div>
            <div class="pt-ds-acts">
              <button class="pt-ds-btn" onclick="event.stopPropagation();ptDatasetStats('${esc(ds.name)}')">📊 统计</button>
              <button class="pt-ds-btn danger" onclick="event.stopPropagation();ptDeleteDataset('${esc(ds.name)}')">🗑️</button>
            </div>
          </div>`;
      }).join('');
    }

    // Show processed data
    if (d.processed?.length) {
      const procEl = _$('#pt-processed-info');
      procEl.innerHTML = d.processed.map(p => {
        if (p.mode === 'sft') {
          return `<div class="pt-proc-info"><h4>✅ 已处理的 SFT 训练数据</h4><div class="pt-proc-grid">
            <div class="pt-proc-stat"><div class="label">名称</div><div class="val">${esc(p.name)}</div></div>
            <div class="pt-proc-stat"><div class="label">对话数</div><div class="val">${(p.total_conversations||0).toLocaleString()}</div></div>
            <div class="pt-proc-stat"><div class="label">训练 Tokens</div><div class="val">${(p.assistant_tokens||0).toLocaleString()}</div></div>
            <div class="pt-proc-stat"><div class="label">Token 总数</div><div class="val">${(p.total_tokens||0).toLocaleString()}</div></div>
            <div class="pt-proc-stat"><div class="label">助手占比</div><div class="val">${p.assistant_pct||0}%</div></div>
            <div class="pt-proc-stat"><div class="label">模板</div><div class="val">${p.chat_template||'chatml'}</div></div>
          </div></div>`;
        }
        return `<div class="pt-proc-info"><h4>✅ 已处理的训练数据</h4><div class="pt-proc-grid">
          <div class="pt-proc-stat"><div class="label">名称</div><div class="val">${esc(p.name)}</div></div>
          <div class="pt-proc-stat"><div class="label">Token 总数</div><div class="val">${(p.usable_tokens||0).toLocaleString()}</div></div>
          <div class="pt-proc-stat"><div class="label">序列数</div><div class="val">${(p.n_sequences||0).toLocaleString()}</div></div>
          <div class="pt-proc-stat"><div class="label">序列长度</div><div class="val">${p.max_seq_len}</div></div>
          <div class="pt-proc-stat"><div class="label">压缩率</div><div class="val">${p.compression_ratio} 字/tok</div></div>
        </div></div>`;
      }).join('');
    }

    // Populate file selectors for processing
    const allFiles = d.datasets || [];
    // Pretrain: show .txt .csv .tsv .jsonl
    const ptFileList = _$('#pt-proc-file-list');
    if (ptFileList) {
      const ptFiles = allFiles.filter(f => ['.txt','.csv','.tsv','.jsonl','.json'].includes(f.ext));
      if (ptFiles.length) {
        ptFileList.innerHTML = ptFiles.map(f => {
          const isStructured = ['.jsonl','.json','.csv','.tsv'].includes(f.ext);
          const hasColConfig = PT.columnConfig[f.name]?.length > 0;
          const colBadge = hasColConfig ? `<span style="font-size:9px;color:var(--ac);background:rgba(74,222,128,.1);padding:1px 5px;border-radius:3px">📋 ${PT.columnConfig[f.name].join(', ')}</span>` : '';
          const colBtn = isStructured ? `<button class="pt-col-btn" onclick="event.preventDefault();event.stopPropagation();ptShowColumnSelector('${esc(f.name)}')" title="选择列">⚙️ 选列</button>` : '';
          return `
          <label class="pt-proc-file-item" onclick="this.classList.toggle('checked')">
            <input type="checkbox" value="${esc(f.name)}" checked>
            <span>${esc(f.name)}</span>
            ${colBadge}
            ${colBtn}
            <span style="color:var(--t4);font-size:10px">${esc(f.size_fmt)}</span>
          </label>`;
        }).join('');
      } else {
        ptFileList.innerHTML = '<span class="desc">请先添加数据集</span>';
      }
    }
    // SFT: show .jsonl .json
    const sftFileList = _$('#pt-sft-proc-file-list');
    if (sftFileList) {
      const sftFiles = allFiles.filter(f => ['.jsonl','.json'].includes(f.ext));
      if (sftFiles.length) {
        sftFileList.innerHTML = sftFiles.map(f => `
          <label class="pt-proc-file-item" onclick="this.classList.toggle('checked')">
            <input type="checkbox" value="${esc(f.name)}" checked>
            <span>${esc(f.name)}</span>
            <span style="color:var(--t4);font-size:10px">${esc(f.size_fmt)}</span>
          </label>`).join('');
      } else {
        sftFileList.innerHTML = '<span class="desc">请先添加对话数据集（JSONL 格式），或使用「格式转换」</span>';
      }
    }
  } catch (e) { console.error(e); }
  ptCheckSupplementary();
}

async function ptLoadRecommendedDatasets() {
  try {
    const r = await fetch('/api/pretrain/datasets/recommended');
    const d = await r.json();
    const el = _$('#pt-rec-datasets');
    if (!el) return;
    el.innerHTML = d.datasets.map(ds => `
      <div class="pt-rec-card">
        <div class="pt-rec-head">
          <span class="pt-rec-name">${esc(ds.name)}</span>
          <span class="pt-rec-lang">${esc(ds.lang)}</span>
        </div>
        <div class="pt-rec-desc">${esc(ds.description)}</div>
        <div class="pt-rec-meta">
          <span>${esc(ds.size)}</span>
          <span>${esc(ds.rows)} 条</span>
          <span>${esc(ds.difficulty)}</span>
        </div>
        <div class="pt-rec-desc" style="font-size:10px;color:var(--ac)">💡 ${esc(ds.recommended_for)}</div>
        <div class="pt-rec-use" onclick="ptUseRecommended('${esc(ds.id)}','${esc(ds.name)}')">📥 使用此数据集</div>
      </div>
    `).join('');
  } catch (e) { console.error(e); }
}

window.ptUseRecommended = function(hfId, name) {
  // 填充到 HF 下载面板
  _$('#pt-hf-panel').style.display = 'block';
  _$('#pt-paste-panel').style.display = 'none';
  _$('#pt-hf-id').value = hfId;
  const ext = PT.trainMode === 'sft' ? '.jsonl' : '.txt';
  _$('#pt-hf-fname').value = name.toLowerCase().replace(/\s+/g, '_') + ext;
  _$('#pt-hf-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

function ptQuickDownloadHF(hfId) {
  // Same as ptUseRecommended but auto-detect extension for SFT
  _$('#pt-hf-panel').style.display = 'block';
  _$('#pt-paste-panel').style.display = 'none';
  _$('#pt-hf-id').value = hfId;
  const ext = PT.trainMode === 'sft' ? '.jsonl' : '.txt';
  const name = hfId.split('/').pop().toLowerCase().replace(/\s+/g, '_');
  _$('#pt-hf-fname').value = name + ext;
  _$('#pt-hf-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function ptUploadFiles() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const files = _$('#pt-data-file').files;
  if (!files.length) return;

  for (const file of files) {
    const form = new FormData();
    form.append('file', file);
    try {
      const r = await fetch(`/api/pretrain/datasets/${PT.project.id}/upload`, { method: 'POST', body: form });
      const d = await r.json();
      if (d.status === 'ok') {
        ptToast(`✅ 已上传: ${d.name} (${d.size_fmt})`);
      } else {
        ptToast(`❌ 上传失败: ${d.detail || '未知错误'}`);
      }
    } catch (e) { ptToast(`❌ 上传失败: ${e.message}`); }
  }
  _$('#pt-data-file').value = '';
  ptLoadDatasets();
}

async function ptSavePastedText() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const text = _$('#pt-paste-text')?.value || '';
  const name = _$('#pt-paste-name')?.value || '';
  if (!text.trim()) { ptToast('请输入文本'); return; }

  try {
    const r = await fetch(`/api/pretrain/datasets/${PT.project.id}/paste`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, name }),
    });
    const d = await r.json();
    if (d.status === 'ok') {
      ptToast(`✅ 已保存: ${d.name}`);
      _$('#pt-paste-panel').style.display = 'none';
      _$('#pt-paste-text').value = '';
      ptLoadDatasets();
    } else {
      ptToast(`❌ ${d.detail || '保存失败'}`);
    }
  } catch (e) { ptToast(`❌ ${e.message}`); }
}

async function ptDownloadHFDataset() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const hfId = _$('#pt-hf-id')?.value?.trim();
  const fname = _$('#pt-hf-fname')?.value?.trim() || 'dataset.txt';
  const maxRows = +((_$('#pt-hf-maxrows')?.value) || 0);
  if (!hfId) { ptToast('请输入数据集 ID'); return; }

  const progEl = _$('#pt-hf-progress');
  progEl.style.display = 'block';
  progEl.className = 'pt-progress-box';
  progEl.innerHTML = `<div class="pt-progress-bar"><div class="pt-progress-fill" id="pt-hf-fill"></div></div><div id="pt-hf-log" class="pt-progress-log">⏳ 准备下载 ${esc(hfId)}...</div>`;
  const btn = _$('#btn-pt-hf-start');
  if (btn) btn.disabled = true;

  await _ptReadSSE(`/api/pretrain/datasets/${PT.project.id}/download-hf`,
    { hf_id: hfId, filename: fname, max_rows: maxRows, python_path: ptPy(), mode: PT.trainMode },
    progEl, 'pt-hf-fill', 'pt-hf-log',
    (ev) => {
      ptToast('✅ 数据集下载完成');
      ptLoadDatasets();
    }
  );

  if (btn) btn.disabled = false;
}

window.ptPreviewDataset = async function(filename) {
  if (!PT.project) return;
  const section = _$('#pt-ds-preview-section');
  const nameEl = _$('#pt-ds-preview-name');
  const previewEl = _$('#pt-ds-preview');
  const statsEl = _$('#pt-ds-stats');
  section.style.display = 'block';
  nameEl.textContent = filename;
  previewEl.innerHTML = '<span class="desc">加载中...</span>';
  statsEl.innerHTML = '';

  try {
    const ext = filename.split('.').pop().toLowerCase();
    const isStructured = ['jsonl', 'json', 'csv', 'tsv'].includes(ext);

    if (isStructured) {
      // For structured files: show table + column info
      const [previewRes, colRes] = await Promise.all([
        fetch(`/api/pretrain/datasets/${PT.project.id}/preview/${encodeURIComponent(filename)}`).then(r => r.json()),
        fetch(`/api/pretrain/datasets/${PT.project.id}/columns/${encodeURIComponent(filename)}`).then(r => r.json()),
      ]);

      if (colRes.columns?.length) {
        const cols = colRes.columns.slice(0, 8);
        let tableHtml = `<div style="margin-bottom:8px;font-size:12px;color:var(--ac)">📊 检测到 ${colRes.columns.length} 列 · ${colRes.total_rows.toLocaleString()} 行</div>`;
        tableHtml += '<div style="overflow-x:auto"><table style="width:100%;font-size:11px;border-collapse:collapse">';
        tableHtml += '<tr>';
        for (const c of cols) {
          const typeIcon = c.is_conversation ? '💬' : c.is_text ? '📝' : '🔢';
          tableHtml += `<th style="padding:4px 8px;border-bottom:2px solid var(--ac);text-align:left;white-space:nowrap">${typeIcon} ${esc(c.name)} <span style="font-size:9px;color:var(--t4);font-weight:400">${c.fill_rate}%</span></th>`;
        }
        tableHtml += '</tr>';
        for (const row of (colRes.sample_rows || []).slice(0, 5)) {
          tableHtml += '<tr>';
          for (const c of cols) {
            const val = String(row[c.name] || '').substring(0, 120);
            tableHtml += `<td style="padding:4px 8px;border-bottom:1px solid var(--bg0);max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--t2)">${esc(val)}</td>`;
          }
          tableHtml += '</tr>';
        }
        tableHtml += '</table></div>';

        if (PT.trainMode === 'pretrain') {
          const suggested = colRes.suggested_columns || [];
          tableHtml += `<div style="margin-top:10px;padding:8px;background:rgba(74,222,128,.05);border-radius:6px;border:1px solid rgba(74,222,128,.2)">
            <span style="font-size:11px;font-weight:600;color:var(--ac)">💡 推荐用于预训练的列：</span>
            <span style="font-size:11px;color:var(--t2)">${suggested.length ? suggested.join(', ') : '未检测到适合的文本列'}</span>
            <button class="btn-p" style="font-size:10px;padding:2px 8px;margin-left:8px;background:var(--ac);color:#fff" onclick="ptShowColumnSelector('${esc(filename)}')">⚙️ 配置列选择</button>
          </div>`;
        }
        previewEl.innerHTML = tableHtml;
      } else {
        previewEl.innerHTML = `<pre style="white-space:pre-wrap;font-size:11px;color:var(--t2);max-height:300px;overflow:auto">${esc(previewRes.preview?.substring(0, 3000) || '')}</pre>`;
      }

      if (previewRes.stats) {
        const s = previewRes.stats || {};
        statsEl.innerHTML = `
          <div class="pt-ds-stat-item"><span class="label">总字符</span><span class="val">${(s.chars||0).toLocaleString()}</span></div>
          <div class="pt-ds-stat-item"><span class="label">总行数</span><span class="val">${(previewRes.total_lines||0).toLocaleString()}</span></div>
          <div class="pt-ds-stat-item"><span class="label">检测语言</span><span class="val">${esc(s.detected_lang||'?')}</span></div>
        `;
      }
    } else {
      // Plain text: raw preview
      const r = await fetch(`/api/pretrain/datasets/${PT.project.id}/preview/${encodeURIComponent(filename)}`);
      const d = await r.json();
      if (d.error) { ptToast(d.error); return; }

      _$('#pt-ds-preview-name').textContent = `${d.name} — ${d.total_lines} 行，显示前 ${d.shown_lines} 行`;
      previewEl.textContent = d.preview;

      const s = d.stats || {};
      const ct = s.char_types || {};
      statsEl.innerHTML = `
        <div class="pt-ds-stat-item"><span class="label">总字符</span><span class="val">${(s.chars||0).toLocaleString()}</span></div>
        <div class="pt-ds-stat-item"><span class="label">总行数</span><span class="val">${(s.lines||0).toLocaleString()}</span></div>
        <div class="pt-ds-stat-item"><span class="label">非空行</span><span class="val">${(s.non_empty_lines||0).toLocaleString()}</span></div>
        <div class="pt-ds-stat-item"><span class="label">词数 (近似)</span><span class="val">${(s.words||0).toLocaleString()}</span></div>
        <div class="pt-ds-stat-item"><span class="label">中文字符</span><span class="val">${(ct.cjk||0).toLocaleString()}</span></div>
        <div class="pt-ds-stat-item"><span class="label">英文字符</span><span class="val">${(ct.ascii_letters||0).toLocaleString()}</span></div>
        <div class="pt-ds-stat-item"><span class="label">检测语言</span><span class="val">${esc(s.detected_lang||'?')}</span></div>
        <div class="pt-ds-stat-item"><span class="label">估算 tokens</span><span class="val">~${(s.est_tokens_rough||0).toLocaleString()}</span></div>
      `;
    }
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) { console.error(e); ptToast('预览失败'); }
};

window.ptDatasetStats = async function(filename) {
  if (!PT.project) return;
  try {
    const r = await fetch(`/api/pretrain/datasets/${PT.project.id}/stats/${encodeURIComponent(filename)}`);
    const d = await r.json();
    if (d.error) { ptToast(d.error); return; }

    // Show in preview section with detailed stats
    _$('#pt-ds-preview-section').style.display = 'block';
    _$('#pt-ds-preview-name').textContent = `📊 ${d.name} 详细统计`;

    // Build rich stats view
    let html = '<div class="pt-ds-stats">';
    const s = d.stats || {};
    const ct = s.char_types || {};
    html += `
      <div class="pt-ds-stat-item"><span class="label">文件大小</span><span class="val">${esc(d.size_fmt)}</span></div>
      <div class="pt-ds-stat-item"><span class="label">总字符</span><span class="val">${(s.chars||0).toLocaleString()}</span></div>
      <div class="pt-ds-stat-item"><span class="label">行数</span><span class="val">${(d.line_count||0).toLocaleString()}</span></div>
      <div class="pt-ds-stat-item"><span class="label">行长 (均)</span><span class="val">${d.line_lengths?.avg||0}</span></div>
      <div class="pt-ds-stat-item"><span class="label">行长 (最短)</span><span class="val">${d.line_lengths?.min||0}</span></div>
      <div class="pt-ds-stat-item"><span class="label">行长 (最长)</span><span class="val">${d.line_lengths?.max||0}</span></div>
    `;
    html += '</div>';

    // Top words
    if (d.top_words?.length) {
      html += '<div style="margin-top:12px"><b style="font-size:11px;color:var(--t3)">高频词 (Top 20):</b><div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">';
      d.top_words.slice(0, 20).forEach(w => {
        html += `<code style="font-size:10px;padding:2px 6px;background:var(--bg0);border-radius:3px">${esc(w.word)} <b style="color:var(--ac)">${w.count}</b></code>`;
      });
      html += '</div></div>';
    }

    // Histogram
    if (d.histogram?.length) {
      const maxCount = Math.max(...d.histogram.map(h => h.count));
      html += '<div style="margin-top:12px"><b style="font-size:11px;color:var(--t3)">行长度分布:</b><div style="margin-top:4px">';
      d.histogram.forEach(h => {
        const pct = maxCount > 0 ? (h.count / maxCount * 100) : 0;
        html += `<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px">
          <span style="font-size:9px;color:var(--t3);width:60px;text-align:right;font-family:var(--fm)">${esc(h.range)}</span>
          <div style="flex:1;height:10px;background:var(--bg0);border-radius:3px;overflow:hidden"><div style="height:100%;width:${pct}%;background:var(--ac);border-radius:3px"></div></div>
          <span style="font-size:9px;color:var(--t3);width:40px;font-family:var(--fm)">${h.count}</span>
        </div>`;
      });
      html += '</div></div>';
    }

    _$('#pt-ds-preview').innerHTML = '';
    _$('#pt-ds-stats').innerHTML = html;
    _$('#pt-ds-preview-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) { console.error(e); ptToast('获取统计失败'); }
};

window.ptDeleteDataset = async function(filename) {
  if (!PT.project) return;
  if (!confirm(`确定删除 ${filename}？`)) return;
  try {
    await fetch(`/api/pretrain/datasets/${PT.project.id}/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    ptToast('已删除');
    ptLoadDatasets();
  } catch (e) { ptToast('删除失败'); }
};

async function ptProcessDataset() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }

  // Check tokenizer first
  try {
    const tr = await fetch(`/api/pretrain/tokenizer/project/${PT.project.id}`);
    const td = await tr.json();
    if (!td.has_tokenizer) {
      ptToast('⚠️ 请先到「词表构建」Tab 配置 Tokenizer');
      return;
    }
  } catch (e) { /* proceed anyway */ }

  const seqLen = +((_$('#pt-proc-seqlen')?.value) || 512);

  // Collect selected files from checkboxes
  const selectedFiles = [];
  _$$('#pt-proc-file-list input[type=checkbox]:checked').forEach(cb => {
    selectedFiles.push(cb.value);
  });
  if (!selectedFiles.length) { ptToast('请至少选择一个数据文件'); return; }

  // Build column_config for selected files
  const colConfig = {};
  for (const fname of selectedFiles) {
    if (PT.columnConfig[fname]?.length) {
      colConfig[fname] = PT.columnConfig[fname];
    }
  }

  const resultEl = _$('#pt-proc-result');
  resultEl.style.display = 'block';
  resultEl.className = 'pt-tok-status';
  resultEl.innerHTML = `<div class="pt-progress-bar"><div class="pt-progress-fill" id="pt-proc-fill"></div></div><div id="pt-proc-log" class="pt-progress-log">⏳ 准备预处理...</div>`;

  await _ptReadSSE(`/api/pretrain/datasets/${PT.project.id}/process`,
    { max_seq_len: seqLen, python_path: ptPy(), files: selectedFiles, column_config: colConfig },
    resultEl, 'pt-proc-fill', 'pt-proc-log',
    (ev) => {
      const rowsInfo = ev.rows_extracted > 0 ? `<div class="pt-proc-stat"><div class="label">提取行数</div><div class="val">${ev.rows_extracted.toLocaleString()}</div></div>` : '';
      resultEl.className = 'pt-tok-status ok';
      resultEl.innerHTML = `<div class="pt-proc-info"><h4>✅ 预处理完成！</h4><div class="pt-proc-grid">
        <div class="pt-proc-stat"><div class="label">原始字符</div><div class="val">${(ev.total_chars||0).toLocaleString()}</div></div>
        <div class="pt-proc-stat"><div class="label">Token 总数</div><div class="val">${(ev.total_tokens||0).toLocaleString()}</div></div>
        <div class="pt-proc-stat"><div class="label">有效 tokens</div><div class="val">${(ev.usable_tokens||0).toLocaleString()}</div></div>
        ${rowsInfo}
        <div class="pt-proc-stat"><div class="label">训练序列数</div><div class="val">${(ev.n_sequences||0).toLocaleString()}</div></div>
        <div class="pt-proc-stat"><div class="label">序列长度</div><div class="val">${ev.max_seq_len}</div></div>
        <div class="pt-proc-stat"><div class="label">压缩率</div><div class="val">${ev.compression_ratio} 字/tok</div></div>
      </div></div>`;
      ptToast('✅ 数据预处理完成');
      ptLoadDatasets();
      ptUpdateWorkflowStatus();
    }
  );
}

// ============ Column Selector ============
window.ptShowColumnSelector = async function(filename) {
  if (!PT.project) return;
  const panel = _$('#pt-col-selector-panel');
  const listEl = _$('#pt-col-list');
  const previewEl = _$('#pt-col-preview');
  const fnameEl = _$('#pt-col-filename');
  fnameEl.textContent = filename;
  listEl.innerHTML = '<span class="desc">🔍 检测列...</span>';
  previewEl.style.display = 'none';
  panel.style.display = 'block';

  try {
    const r = await fetch(`/api/pretrain/datasets/${PT.project.id}/columns/${encodeURIComponent(filename)}`);
    const d = await r.json();
    if (d.error) {
      listEl.innerHTML = `<span class="desc" style="color:var(--err)">${esc(d.error)}</span>`;
      return;
    }

    const currentCols = PT.columnConfig[filename] || d.suggested_columns || [];

    listEl.innerHTML = d.columns.map(col => {
      const checked = currentCols.includes(col.name) ? 'checked' : '';
      const isCheckedClass = checked ? ' checked' : '';
      const typeIcon = col.is_conversation ? '💬' : col.is_text ? '📝' : '🔢';
      const fillColor = col.fill_rate > 80 ? 'var(--ok)' : col.fill_rate > 50 ? '#f59e0b' : 'var(--err)';
      return `
        <label class="pt-proc-file-item${isCheckedClass}" onclick="this.classList.toggle('checked');ptUpdateColConfig('${esc(filename)}')">
          <input type="checkbox" value="${esc(col.name)}" ${checked} data-filename="${esc(filename)}">
          <span style="font-weight:500">${typeIcon} ${esc(col.name)}</span>
          <span style="color:var(--t4);font-size:10px">
            填充率 <span style="color:${fillColor}">${col.fill_rate}%</span> · 
            平均 ${col.avg_len} 字符
          </span>
        </label>`;
    }).join('');

    // Show sample data
    if (d.sample_rows?.length) {
      previewEl.style.display = 'block';
      const cols = d.columns.map(c => c.name).slice(0, 6);
      let html = '<table style="width:100%;font-size:10px;border-collapse:collapse"><tr>';
      html += cols.map(c => `<th style="padding:3px 6px;border-bottom:1px solid var(--br);text-align:left;color:var(--ac)">${esc(c)}</th>`).join('');
      html += '</tr>';
      for (const row of d.sample_rows.slice(0, 3)) {
        html += '<tr>';
        html += cols.map(c => {
          const val = String(row[c] || '').substring(0, 80);
          return `<td style="padding:3px 6px;border-bottom:1px solid var(--bg0);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(val)}</td>`;
        }).join('');
        html += '</tr>';
      }
      html += `</table><p style="margin:4px 0 0;color:var(--t4);font-size:10px">共 ${d.total_rows.toLocaleString()} 行</p>`;
      previewEl.innerHTML = html;
    }
  } catch (e) {
    listEl.innerHTML = `<span class="desc" style="color:var(--err)">检测失败: ${e.message}</span>`;
  }
};

window.ptUpdateColConfig = function(filename) {
  const checkboxes = _$$(`#pt-col-list input[type=checkbox][data-filename="${filename}"]`);
  const selected = [];
  checkboxes.forEach(cb => { if (cb.checked) selected.push(cb.value); });
  PT.columnConfig[filename] = selected;
  // Update badge in file list
  ptLoadDatasets();
};

// ============ Supplementary Training ============
function ptCheckSupplementary() {
  // Show "补充数据" button if data is already processed
  const appendBtn = _$('#btn-pt-process-append');
  const appendBtnSft = _$('#btn-pt-process-sft-append');
  const procInfo = _$('#pt-processed-info');
  const hasProcessed = procInfo && procInfo.innerHTML.trim().length > 0;

  if (appendBtn) appendBtn.style.display = hasProcessed && PT.trainMode === 'pretrain' ? 'inline-block' : 'none';
  if (appendBtnSft) appendBtnSft.style.display = hasProcessed && PT.trainMode === 'sft' ? 'inline-block' : 'none';
}


/* ============================================================
   TRAINING TAB
   ============================================================ */
let _ptMonTimer = null;
const _lossHistory = [];
const _valLossHistory = [];

function ptInitTrainTab() {
  ptLoadCheckpoints();
  ptLoadSamplesTimeline();
  ptCheckTrainStatus();
  ptLoadEnvs();
  ptDetectHardware();
  ptLoadTrainingHistory();

  // Env change → re-detect hardware with that python
  const envSel = _$('#pt-tr-python');
  if (envSel) envSel.onchange = () => {
    PT.selectedPython = envSel.value;
    ptDetectHardware();
  };

  // DDP toggle
  const ddpCheck = _$('#pt-tr-ddp');
  if (ddpCheck) ddpCheck.onchange = () => {
    _$('#pt-tr-ddp-cfg').style.display = ddpCheck.checked ? 'block' : 'none';
  };
}

/* Load Python environments from shared /api/venv/detect */
async function ptLoadEnvs() {
  ptLoadEnvsInto('#pt-tr-python');
}

/* Reusable: populate any python env selector */
async function ptLoadEnvsInto(selector) {
  const sel = _$(selector);
  if (!sel) return;
  try {
    const r = await fetch('/api/venv/detect', { method: 'POST' });
    const d = await r.json();
    sel.innerHTML = '<option value="">系统 Python</option>';
    if (d.envs?.length) {
      d.envs.forEach(e => {
        const opt = document.createElement('option');
        opt.value = e.path;
        opt.textContent = `${e.name} (${e.version}) [${e.type}]`;
        sel.appendChild(opt);
      });
    }
    // Restore previous selection
    if (PT.selectedPython) sel.value = PT.selectedPython;
  } catch (e) { console.error('ptLoadEnvsInto:', e); }
}

/* Detect GPU hardware — uses shared /api/hardware (nvidia-smi + torch check) */
async function ptDetectHardware() {
  const infoEl = _$('#pt-hw-info');
  const devSel = _$('#pt-tr-device');
  if (!infoEl) return;
  infoEl.innerHTML = '🔍 正在检测硬件...';
  infoEl.style.color = '';

  try {
    const r = await fetch('/api/hardware');
    const data = await r.json();
    const hw = data.hardware || data;

    // Also check CUDA via selected python env
    const pyPath = _$('#pt-tr-python')?.value || '';
    let torchInfo = null;
    if (pyPath) {
      try {
        const tr = await fetch('/api/pretrain/hardware?python_path=' + encodeURIComponent(pyPath));
        torchInfo = await tr.json();
      } catch {}
    }

    // Merge: prefer torchInfo if we have selected env
    const gpus = hw.gpus || [];
    const hasCuda = torchInfo ? torchInfo.cuda_available : hw.cuda;
    const cudaVer = torchInfo?.cuda_version || '';
    const torchVer = torchInfo?.torch_version || '';

    // Remove old GPU options (keep auto + cpu)
    if (devSel) {
      while (devSel.options.length > 2) devSel.remove(2);
    }

    if (hasCuda && gpus.length) {
      gpus.forEach(g => {
        if (!devSel) return;
        const opt = document.createElement('option');
        const idx = g.index ?? gpus.indexOf(g);
        opt.value = `cuda:${idx}`;
        const vram = g.vram_total_mb ? ` (${(g.vram_total_mb/1024).toFixed(1)} GB)` : '';
        opt.textContent = `🎮 GPU ${idx}: ${g.name}${vram}`;
        devSel.appendChild(opt);
      });
      // Auto-select first GPU
      if (devSel && devSel.value === 'auto') devSel.value = 'cuda:0';

      const names = gpus.map(g => g.name).join(', ');
      const vramTotal = gpus.reduce((s, g) => s + (g.vram_total_mb || 0), 0);
      let info = `✅ <b>${gpus.length} GPU</b>: ${esc(names)}`;
      if (vramTotal) info += ` | <b>${(vramTotal/1024).toFixed(1)} GB</b> 显存`;
      if (cudaVer) info += ` | CUDA ${esc(cudaVer)}`;
      if (torchVer) info += ` | PyTorch ${esc(torchVer)}`;
      info += ` | RAM ${hw.ram_gb || '?'} GB`;
      infoEl.innerHTML = info;
      infoEl.style.color = 'var(--ok)';
    } else {
      let info = '⚠️ 未检测到可用的 GPU';
      if (torchVer) info += ` (PyTorch ${esc(torchVer)}, CUDA 不可用)`;
      info += ` | RAM ${hw.ram_gb || '?'} GB`;
      info += '<br><span style="font-size:11px;color:var(--t4)">💡 建议在微调中心 → 训练环境 标签页创建带 CUDA 的虚拟环境</span>';
      infoEl.innerHTML = info;
      infoEl.style.color = 'var(--warn)';
      if (devSel) devSel.value = 'cpu';
    }
  } catch (e) {
    infoEl.innerHTML = '⚠️ 硬件检测失败: ' + esc(e.message);
    infoEl.style.color = 'var(--t4)';
  }
}

function ptCollectTrainConfig() {
  const prompts = (_$('#pt-tr-prompts')?.value || '').split(',').map(s => s.trim()).filter(Boolean);
  const ddpEnabled = _$('#pt-tr-ddp')?.checked || false;
  const gpuStr = _$('#pt-tr-gpus')?.value || '0';
  const gpuIds = gpuStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const device = _$('#pt-tr-device')?.value || 'auto';

  const cfg = {
    device,
    train_mode: PT.trainMode,
    learning_rate: parseFloat(_$('#pt-tr-lr')?.value || '3e-4'),
    batch_size: +(_$('#pt-tr-bs')?.value || 32),
    grad_accum_steps: +(_$('#pt-tr-accum')?.value || 4),
    max_steps: +(_$('#pt-tr-steps')?.value || 5000),
    warmup_steps: +(_$('#pt-tr-warmup')?.value || 200),
    weight_decay: +(_$('#pt-tr-wd')?.value || 0.1),
    grad_clip: +(_$('#pt-tr-clip')?.value || 1.0),
    lr_scheduler: _$('#pt-tr-sched')?.value || 'cosine',
    fp16: _$('#pt-tr-fp16')?.checked ?? true,
    save_every_steps: +(_$('#pt-tr-save')?.value || 500),
    sample_every_steps: +(_$('#pt-tr-sample')?.value || 200),
    eval_every_steps: +(_$('#pt-tr-eval')?.value || 500),
    val_split: 0.05,
    sample_prompts: prompts.length ? prompts : ['Once upon a time', '从前有一个', '人工智能的未来', 'The meaning of life is'],
    distributed: { enabled: ddpEnabled, type: 'ddp', gpu_ids: gpuIds },
  };

  // SFT-specific config
  if (PT.trainMode === 'sft') {
    cfg.chat_template = _$('#pt-sft-template')?.value || 'chatml';
    cfg.base_checkpoint = _$('#pt-tr-base-ckpt')?.value || '';
  }

  return cfg;
}

async function ptStartTraining() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }

  // Pre-flight checks
  const issues = await ptPreflightCheck();
  const errors = issues.filter(i => i.startsWith('❌'));
  const warnings = issues.filter(i => i.startsWith('⚠️'));
  if (errors.length > 0) {
    alert('无法开始训练：\n\n' + errors.join('\n'));
    return;
  }
  if (warnings.length > 0) {
    if (!confirm('检测到以下问题：\n\n' + warnings.join('\n') + '\n\n是否仍然继续训练？')) return;
  }

  const training = ptCollectTrainConfig();

  _$('#btn-pt-start-train').disabled = true;
  try {
    const r = await fetch('/api/pretrain/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: PT.project.id, training, python_path: ptPy() }),
    });
    const d = await r.json();
    if (d.status === 'ok') {
      ptToast('🚀 训练已启动');
      _lossHistory.length = 0;
      _valLossHistory.length = 0;
      ptStartMonitor();
    } else {
      ptToast(`❌ ${d.error || '启动失败'}`);
      _$('#btn-pt-start-train').disabled = false;
    }
  } catch (e) {
    ptToast(`❌ ${e.message}`);
    _$('#btn-pt-start-train').disabled = false;
  }
}

async function ptStopTraining() {
  try {
    await fetch('/api/pretrain/train/stop', { method: 'POST' });
    ptToast('正在停止训练...');
  } catch (e) { ptToast('停止失败'); }
}

function ptStartMonitor() {
  _$('#pt-monitor-section').style.display = 'block';
  _$('#pt-samples-section').style.display = 'block';
  _$('#btn-pt-start-train').style.display = 'none';
  _$('#btn-pt-stop-train').style.display = 'inline-flex';
  if (_ptMonTimer) clearInterval(_ptMonTimer);
  _ptMonTimer = setInterval(ptPollStatus, 2000);
  ptPollStatus();
}

function ptStopMonitor() {
  if (_ptMonTimer) { clearInterval(_ptMonTimer); _ptMonTimer = null; }
  _$('#btn-pt-start-train').style.display = 'inline-flex';
  _$('#btn-pt-start-train').disabled = false;
  _$('#btn-pt-stop-train').style.display = 'none';
}

async function ptPollStatus() {
  try {
    const r = await fetch('/api/pretrain/train/status');
    const d = await r.json();

    _$('#pt-mon-step').textContent = `${d.current_step} / ${d.total_steps}`;
    _$('#pt-mon-loss').textContent = d.loss > 0 ? d.loss.toFixed(4) : '—';
    _$('#pt-mon-val-loss').textContent = d.val_loss > 0 ? d.val_loss.toFixed(4) : '—';
    if (d.val_loss > 0) {
      const vlEl = _$('#pt-mon-val-loss');
      vlEl.style.color = (d.best_val_loss && d.val_loss <= d.best_val_loss * 1.05) ? 'var(--ok)' : 'var(--warn)';
    }
    _$('#pt-mon-lr').textContent = d.lr > 0 ? d.lr.toExponential(2) : '—';
    _$('#pt-mon-speed').textContent = d.tokens_per_sec > 0 ? `${d.tokens_per_sec.toFixed(0)} tok/s` : '—';
    _$('#pt-mon-device').textContent = d.device || '—';
    _$('#pt-mon-device').style.color = (d.device && d.device.startsWith('cuda')) ? 'var(--ok)' : 'var(--warn)';
    _$('#pt-mon-status').textContent = d.running ? '🟢 训练中' : '⏹ 已停止';
    _$('#pt-mon-status').style.color = d.running ? 'var(--ok)' : 'var(--t3)';

    // Progress
    const pct = d.total_steps > 0 ? (d.current_step / d.total_steps * 100) : 0;
    _$('#pt-mon-fill').style.width = pct + '%';

    // Loss history
    if (d.loss > 0 && d.current_step > 0) {
      if (!_lossHistory.length || _lossHistory[_lossHistory.length - 1].step !== d.current_step) {
        _lossHistory.push({ step: d.current_step, loss: d.loss, lr: d.lr });
      }
      if (d.val_loss > 0) {
        if (!_valLossHistory.length || _valLossHistory[_valLossHistory.length - 1].step !== d.current_step) {
          _valLossHistory.push({ step: d.current_step, loss: d.val_loss });
        }
      }
      ptDrawLossChart();
    }

    // Samples
    if (d.samples?.length) {
      ptRenderLiveSamples(d.samples);
    }

    // Log
    if (d.log_lines?.length) {
      const logEl = _$('#pt-train-log');
      logEl.textContent = d.log_lines.slice(-50).join('\n');
      logEl.scrollTop = logEl.scrollHeight;
    }

    if (!d.running && _ptMonTimer) {
      ptStopMonitor();
      ptLoadCheckpoints();
      ptLoadSamplesTimeline();
      ptLoadTrainingHistory();
      ptUpdateWorkflowStatus();
      // Show completion guidance
      const guide = document.createElement('div');
      guide.style.cssText = 'margin-top:12px;padding:12px 16px;background:linear-gradient(135deg,rgba(74,222,128,.08),rgba(56,189,248,.08));border-radius:10px;border:1px solid rgba(74,222,128,.2)';
      guide.innerHTML = `<div style="font-size:13px;font-weight:600;color:var(--ok);margin-bottom:6px">🎉 训练完成！</div>
        <div style="font-size:12px;color:var(--t2)">接下来可以：
          <span class="pt-guide-link" onclick="switchPTTab('pt-chat')" style="color:var(--ac);cursor:pointer;text-decoration:underline;margin-left:6px">💬 去对话测试</span>
          <span class="pt-guide-link" onclick="switchPTTab('pt-play')" style="color:var(--ac);cursor:pointer;text-decoration:underline;margin-left:10px">🔬 对比不同 checkpoint</span>
          <span class="pt-guide-link" onclick="switchPTTab('pt-play')" style="color:var(--ac);cursor:pointer;text-decoration:underline;margin-left:10px">📦 导出模型</span>
        </div>`;
      _$('#pt-monitor-section').appendChild(guide);
      ptToast('🎉 训练已完成！');
    }
  } catch (e) { console.error(e); }
}

async function ptCheckTrainStatus() {
  try {
    const r = await fetch('/api/pretrain/train/status');
    const d = await r.json();
    if (d.running) ptStartMonitor();
  } catch (e) {}
}

function ptDrawLossChart() {
  const canvas = _$('#pt-loss-chart');
  if (!canvas || !_lossHistory.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const data = _lossHistory;
  const valData = _valLossHistory;
  const allLosses = [...data.map(d => d.loss), ...valData.map(d => d.loss)];
  const minLoss = Math.min(...allLosses);
  const maxLoss = Math.max(...allLosses);
  const range = maxLoss - minLoss || 1;
  const minStep = data[0].step;
  const maxStep = data[data.length - 1].step;
  const stepRange = maxStep - minStep || 1;

  // Grid
  ctx.strokeStyle = '#282838';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = H * 0.1 + (H * 0.8) * i / 4;
    ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(W - 10, y); ctx.stroke();
    ctx.fillStyle = '#666';
    ctx.font = '10px monospace';
    ctx.fillText((maxLoss - range * i / 4).toFixed(2), 2, y + 3);
  }

  // Train loss curve
  ctx.strokeStyle = '#c9a24d';
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((d, i) => {
    const x = 40 + (d.step - minStep) / stepRange * (W - 50);
    const y = H * 0.1 + (1 - (d.loss - minLoss) / range) * H * 0.8;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Val loss curve
  if (valData.length > 1) {
    ctx.strokeStyle = '#4ade80';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    valData.forEach((d, i) => {
      const x = 40 + (d.step - minStep) / stepRange * (W - 50);
      const y = H * 0.1 + (1 - (d.loss - minLoss) / range) * H * 0.8;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Labels
  ctx.fillStyle = '#999';
  ctx.font = '10px monospace';
  ctx.fillText(`Step ${minStep}`, 40, H - 2);
  ctx.fillText(`Step ${maxStep}`, W - 70, H - 2);
  ctx.fillStyle = '#c9a24d';
  ctx.fillText('Train', 2, 12);
  if (valData.length) { ctx.fillStyle = '#4ade80'; ctx.fillText('Val', 45, 12); }
}

function ptRenderLiveSamples(samples) {
  const el = _$('#pt-samples-list');
  if (!el) return;
  // Show latest 5
  const recent = samples.slice(-5).reverse();
  el.innerHTML = recent.map(s => `
    <div class="pt-sample-card">
      <div class="pt-sample-step">Step ${s.step}</div>
      ${(s.samples || []).map(item => `
        <div class="pt-sample-text"><span class="pt-sample-prompt">${esc(item.prompt)}</span>${esc(item.text.slice(item.prompt.length))}</div>
      `).join('')}
    </div>
  `).join('');
}

async function ptLoadCheckpoints() {
  if (!PT.project) return;
  try {
    // Load checkpoints for current mode (for checkpoint list, play, chat)
    const r = await fetch(`/api/pretrain/train/checkpoints/${PT.project.id}?mode=${PT.trainMode}`);
    const d = await r.json();
    const el = _$('#pt-ckpt-list');
    const ckpts = d.checkpoints || [];

    const modeLabel = PT.trainMode === 'sft' ? 'SFT' : '预训练';
    if (!ckpts.length) {
      el.innerHTML = `<p class="desc">尚无${modeLabel} checkpoint</p>`;
    } else {
      el.innerHTML = ckpts.map(c => `
        <div class="pt-ckpt-item">
          <span class="name">${esc(c.name)}</span>
          <span class="meta">${esc(c.size_fmt)}</span>
          ${c.is_final ? '<span class="final-badge">FINAL</span>' : ''}
        </div>
      `).join('');
    }

    // Update checkpoint selectors in playground and chat (current mode)
    const opts = '<option value="">最新</option>' + ckpts.map(c => `<option value="${esc(c.path)}">${esc(c.name)} (step ${c.step})</option>`).join('');
    ['#pt-play-ckpt', '#pt-exp-ckpt', '#pt-chat-ckpt'].forEach(sel => {
      const e = _$(sel);
      if (e) e.innerHTML = opts;
    });

    // SFT base checkpoint selector: always load PRETRAIN checkpoints
    if (PT.trainMode === 'sft') {
      const r2 = await fetch(`/api/pretrain/train/checkpoints/${PT.project.id}?mode=pretrain`);
      const d2 = await r2.json();
      const ptCkpts = d2.checkpoints || [];
      const baseSel = _$('#pt-tr-base-ckpt');
      if (baseSel) {
        baseSel.innerHTML = '<option value="">不加载（从随机初始化开始）</option>' +
          ptCkpts.map(c => `<option value="${esc(c.path)}">[预训练] ${esc(c.name)} (step ${c.step})</option>`).join('');
      }
    }
  } catch (e) { console.error(e); }
}


/* ============================================================
   PLAYGROUND TAB
   ============================================================ */
function ptInitPlayTab() {
  ptLoadCheckpoints();
  ptLoadSamplesTimeline();
}

async function ptGenerate() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const prompt = _$('#pt-play-prompt')?.value?.trim();
  if (!prompt) { ptToast('请输入提示词'); return; }

  const resultEl = _$('#pt-play-result');
  resultEl.style.display = 'block';
  _$('#pt-play-output').innerHTML = '<span style="color:var(--t3)">⏳ 生成中...</span>';
  _$('#pt-play-stats').innerHTML = '';
  _$('#btn-pt-generate').disabled = true;

  try {
    const r = await fetch('/api/pretrain/inference/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project_id: PT.project.id,
        checkpoint: _$('#pt-play-ckpt')?.value || '',
        prompts: [prompt],
        max_tokens: +(_$('#pt-play-max')?.value || 100),
        temperature: +(_$('#pt-play-temp')?.value || 0.8),
        top_k: +(_$('#pt-play-topk')?.value || 50),
        python_path: ptPy(),
      }),
    });
    const d = await r.json();
    if (d.status === 'ok' && d.results?.length) {
      const res = d.results[0];
      _$('#pt-play-output').innerHTML = `<span class="prompt-part">${esc(res.prompt)}</span><span class="gen-part">${esc(res.text.slice(res.prompt.length))}</span>`;
      _$('#pt-play-stats').innerHTML = `<span>Step: <b>${d.step || '?'}</b></span><span>新 tokens: <b>${res.new_tokens}</b></span><span>耗时: <b>${res.elapsed}s</b></span><span>速度: <b>${res.tok_s} tok/s</b></span>`;
    } else {
      _$('#pt-play-output').innerHTML = `<span style="color:var(--err)">❌ ${esc(d.error || '生成失败')}</span>`;
    }
  } catch (e) {
    _$('#pt-play-output').innerHTML = `<span style="color:var(--err)">❌ ${esc(e.message)}</span>`;
  }
  _$('#btn-pt-generate').disabled = false;
}

async function ptCompare() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const prompt = _$('#pt-cmp-prompt')?.value?.trim();
  if (!prompt) { ptToast('请输入对比提示词'); return; }

  const resultEl = _$('#pt-cmp-result');
  resultEl.innerHTML = '<p class="desc">⏳ 正在对比生成...</p>';

  try {
    // Get all checkpoints
    const r1 = await fetch(`/api/pretrain/train/checkpoints/${PT.project.id}?mode=${PT.trainMode}`);
    const d1 = await r1.json();
    const ckpts = (d1.checkpoints || []).filter((c, i, arr) => {
      // Pick evenly spaced checkpoints (max 4)
      if (arr.length <= 4) return true;
      const step = Math.floor(arr.length / 4);
      return i % step === 0 || i === arr.length - 1;
    }).slice(0, 4);

    if (!ckpts.length) { resultEl.innerHTML = '<p class="desc">没有可用的 checkpoint</p>'; return; }

    resultEl.innerHTML = ckpts.map(c => `<div class="pt-cmp-card"><div class="pt-cmp-card-head">${esc(c.name)}</div><div class="pt-cmp-card-text">⏳ 生成中...</div></div>`).join('');

    // Generate for each (sequentially)
    const cards = resultEl.querySelectorAll('.pt-cmp-card');
    for (let i = 0; i < ckpts.length; i++) {
      try {
        const r = await fetch('/api/pretrain/inference/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: PT.project.id,
            checkpoint: ckpts[i].path,
            prompts: [prompt],
            max_tokens: 80,
            temperature: 0.8, top_k: 50,
            python_path: ptPy(),
          }),
        });
        const d = await r.json();
        if (d.status === 'ok' && d.results?.length) {
          cards[i].querySelector('.pt-cmp-card-text').innerHTML = `<span style="color:var(--ac);font-weight:600">${esc(prompt)}</span>${esc(d.results[0].text.slice(prompt.length))}`;
        } else {
          cards[i].querySelector('.pt-cmp-card-text').textContent = `❌ ${d.error || '生成失败'}`;
        }
      } catch (e) {
        cards[i].querySelector('.pt-cmp-card-text').textContent = `❌ ${e.message}`;
      }
    }
  } catch (e) { resultEl.innerHTML = `<p class="desc">❌ ${esc(e.message)}</p>`; }
}

async function ptExportGGUF() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const el = _$('#pt-export-result');
  el.style.display = 'block';
  el.className = 'pt-tok-status';
  el.innerHTML = `<div class="pt-progress-bar"><div class="pt-progress-fill" id="pt-exp-fill"></div></div><div id="pt-exp-log" class="pt-progress-log">⏳ 准备导出...</div>`;

  await _ptReadSSE('/api/pretrain/export/gguf',
    {
      project_id: PT.project.id,
      checkpoint: _$('#pt-exp-ckpt')?.value || '',
      output_name: _$('#pt-exp-name')?.value || 'model',
      python_path: ptPy(),
    },
    el, 'pt-exp-fill', 'pt-exp-log',
    (ev) => {
      el.className = 'pt-tok-status ok';
      el.innerHTML = `✅ HuggingFace 格式导出完成！<br>路径: <code>${esc(ev.hf_dir || '')}</code>`;
      ptToast('✅ 模型导出成功');
    }
  );
}


/* ============================================================
   DELETE PROJECT
   ============================================================ */
async function ptDeleteProject() {
  if (!PT.project) return;
  if (!confirm(`确定要删除项目「${PT.project.name}」吗？\n此操作将删除所有数据集、checkpoint 和训练记录，且不可恢复！`)) return;
  try {
    await fetch(`/api/pretrain/projects/${PT.project.id}`, { method: 'DELETE' });
    ptToast('🗑️ 项目已删除');
    PT.project = null;
    _ptResetAllUI();
    resetToDefault();
    const delBtn = _$('#btn-pt-del-proj');
    if (delBtn) delBtn.style.display = 'none';
    const wfEl = _$('#pt-workflow-status');
    if (wfEl) wfEl.innerHTML = '';
    await ptLoadProjectList();
    const sel = _$('#pt-project-sel');
    if (sel) sel.value = '';
  } catch (e) { ptToast('删除失败'); }
}


/* ============================================================
   TRAINING HISTORY
   ============================================================ */
async function ptLoadTrainingHistory() {
  if (!PT.project) return;
  try {
    const r = await fetch(`/api/pretrain/train/history/${PT.project.id}`);
    const d = await r.json();
    const records = d.records || [];
    const el = _$('#pt-train-history');
    if (!el) return;

    if (!records.length) {
      el.innerHTML = '<p class="desc">暂无训练记录。训练完成后将自动记录</p>';
      return;
    }

    el.innerHTML = records.slice().reverse().map(r => {
      const elapsed = r.elapsed_sec ? (r.elapsed_sec >= 3600 ?
        `${(r.elapsed_sec/3600).toFixed(1)}h` :
        `${(r.elapsed_sec/60).toFixed(1)}min`) : '?';
      const fmtParams = r.params >= 1e6 ? `${(r.params/1e6).toFixed(1)}M` : `${(r.params/1e3).toFixed(0)}K`;
      return `
        <div class="pt-hist-card">
          <div class="pt-hist-head">
            <span class="time">${esc(r.timestamp?.slice(0, 19).replace('T', ' ') || '?')}</span>
            <span class="badge">✅ ${r.steps} steps</span>
          </div>
          <div class="pt-hist-grid">
            <div class="pt-hist-stat">参数量: <span class="val">${esc(fmtParams)}</span></div>
            <div class="pt-hist-stat">训练时长: <span class="val">${esc(elapsed)}</span></div>
            <div class="pt-hist-stat">Final Loss: <span class="val">${r.final_loss?.toFixed(4) || '—'}</span></div>
            <div class="pt-hist-stat">Val Loss: <span class="val">${r.final_val_loss != null ? r.final_val_loss.toFixed(4) : '—'}</span></div>
            <div class="pt-hist-stat">Best Val: <span class="val">${r.best_val_loss != null ? r.best_val_loss.toFixed(4) : '—'}</span></div>
            <div class="pt-hist-stat">设备: <span class="val">${esc(r.device || '?')}</span></div>
          </div>
          ${r.datasets?.length ? `<div class="pt-hist-datasets">📁 数据集: ${r.datasets.map(d => esc(d)).join(', ')}</div>` : ''}
        </div>`;
    }).join('');
  } catch (e) { console.error(e); }
}


/* ============================================================
   CHAT TAB (Streaming)
   ============================================================ */
function ptInitChatTab() {
  ptLoadCheckpoints();
}

let _chatAbort = null;
let _chatHistory = []; // [{role: 'user'|'assistant', content: '...'}]

async function ptChatSend() {
  if (!PT.project) { ptToast('请先选择一个项目'); return; }
  const input = _$('#pt-chat-input');
  const prompt = input?.value?.trim();
  if (!prompt) { ptToast('请输入文本'); return; }

  const msgsEl = _$('#pt-chat-messages');
  // Remove placeholder
  const ph = msgsEl.querySelector('.pt-chat-placeholder');
  if (ph) ph.remove();

  // Add user message
  msgsEl.innerHTML += `
    <div class="pt-chat-msg user">
      <div class="pt-chat-avatar">👤</div>
      <div class="pt-chat-bubble">${esc(prompt)}</div>
    </div>`;

  // Add model message (streaming)
  const msgId = 'pt-chat-' + Date.now();
  msgsEl.innerHTML += `
    <div class="pt-chat-msg model" id="${msgId}">
      <div class="pt-chat-avatar">🤖</div>
      <div>
        <div class="pt-chat-bubble" id="${msgId}-text"><span class="pt-chat-cursor"></span></div>
        <div class="pt-chat-stats" id="${msgId}-stats"></div>
      </div>
    </div>`;
  msgsEl.scrollTop = msgsEl.scrollHeight;

  input.value = '';
  const sendBtn = _$('#btn-pt-chat-send');
  sendBtn.disabled = true;
  sendBtn.textContent = '⏳ 生成中...';

  try {
    // In SFT mode, build multi-turn prompt from history
    let actualPrompt = prompt;
    if (PT.trainMode === 'sft') {
      _chatHistory.push({ role: 'user', content: prompt });
      const tmpl = _$('#pt-sft-template')?.value || 'chatml';
      actualPrompt = _buildSFTPrompt(_chatHistory, tmpl);
    }

    const body = {
      project_id: PT.project.id,
      checkpoint: _$('#pt-chat-ckpt')?.value || '',
      prompt: actualPrompt,
      max_tokens: +(_$('#pt-chat-max')?.value || 200),
      temperature: +(_$('#pt-chat-temp')?.value || 0.8),
      top_k: +(_$('#pt-chat-topk')?.value || 50),
      python_path: ptPy(),
    };
    // In SFT mode, pass chat_template for stop tokens + raw_prompt since we built the prompt ourselves
    if (PT.trainMode === 'sft') {
      body.chat_template = _$('#pt-sft-template')?.value || 'chatml';
      body.raw_prompt = true;
    }

    const r = await fetch('/api/pretrain/inference/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let generated = '';
    const textEl = document.getElementById(`${msgId}-text`);
    const statsEl = document.getElementById(`${msgId}-stats`);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.event === 'token') {
            generated += ev.text;
            textEl.innerHTML = esc(generated) + '<span class="pt-chat-cursor"></span>';
            msgsEl.scrollTop = msgsEl.scrollHeight;
          } else if (ev.event === 'done') {
            textEl.innerHTML = esc(generated);
            statsEl.textContent = `${ev.total_tokens} tokens · ${ev.elapsed}s · ${ev.tok_s} tok/s · step ${ev.step}`;
            // Store assistant response in history for multi-turn
            if (PT.trainMode === 'sft' && generated) {
              _chatHistory.push({ role: 'assistant', content: generated });
            }
          } else if (ev.event === 'error') {
            textEl.innerHTML = `<span style="color:var(--err)">❌ ${esc(ev.error)}</span>`;
          }
        } catch {}
      }
    }

    // Ensure cursor is removed
    textEl.innerHTML = esc(generated) || textEl.innerHTML;

  } catch (e) {
    const textEl = document.getElementById(`${msgId}-text`);
    if (textEl) textEl.innerHTML = `<span style="color:var(--err)">❌ ${esc(e.message)}</span>`;
  }

  sendBtn.disabled = false;
  sendBtn.textContent = '✨ 生成';
}

function ptChatClear() {
  const msgsEl = _$('#pt-chat-messages');
  if (msgsEl) msgsEl.innerHTML = '<div class="pt-chat-placeholder">选择一个项目和 checkpoint，然后在下方输入文本开始对话</div>';
  _chatHistory = [];
}

// Build full multi-turn prompt for SFT mode
function _buildSFTPrompt(history, template) {
  let prompt = '';
  for (const msg of history) {
    if (template === 'chatml') {
      prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
    } else if (template === 'llama') {
      if (msg.role === 'user') prompt += `[INST] ${msg.content} [/INST]\n`;
      else prompt += `${msg.content}</s>\n`;
    } else { // simple
      if (msg.role === 'user') prompt += `### User:\n${msg.content}\n\n`;
      else prompt += `### Assistant:\n${msg.content}\n\n`;
    }
  }
  // Add assistant prefix for next response
  if (template === 'chatml') prompt += '<|im_start|>assistant\n';
  else if (template === 'simple') prompt += '### Assistant:\n';
  return prompt;
}

function _getSFTStopTokens(template) {
  if (template === 'chatml') return ['<|im_end|>', '<|im_start|>'];
  if (template === 'llama') return ['[INST]', '</s>'];
  return ['### User:', '###'];
}

async function ptLoadSamplesTimeline() {
  if (!PT.project) return;
  try {
    const r = await fetch(`/api/pretrain/train/samples/${PT.project.id}?mode=${PT.trainMode}`);
    const d = await r.json();
    const all = [...(d.saved_samples || [])];
    const el = _$('#pt-timeline');
    if (!el) return;

    if (!all.length) {
      el.innerHTML = '<p class="desc">训练完成后，模型生成的文本样本将显示在这里</p>';
      return;
    }

    el.innerHTML = all.map(s => `
      <div class="pt-tl-item">
        <div class="pt-tl-step">Step ${s.step}</div>
        ${(s.samples || []).map(item => `
          <div class="pt-tl-text"><span style="color:var(--ac);font-weight:600">${esc(item.prompt)}</span>${esc(item.text.slice(item.prompt.length))}</div>
        `).join('')}
      </div>
    `).join('');
  } catch (e) { console.error(e); }
}
