"""streaming.py — 统一的流式子进程执行器，支持 SSE 进度推送"""
import json, subprocess, sys, os
from pathlib import Path
from typing import Generator


def run_streaming(
    script: str,
    python_path: str | None = None,
    timeout: int = 600,
    env: dict | None = None,
) -> Generator[str, None, None]:
    """执行 Python 脚本并逐行 yield SSE 数据。
    脚本中应通过 print(json.dumps({...}), flush=True) 输出进度。
    每个 JSON 对象应包含:
      - step: str  (显示文字)
      - progress: int  (0-100)
      - done: bool  (是否完成，可选)
      - error: bool  (是否出错，可选)
      - 其他自定义字段
    """
    py = python_path or sys.executable
    proc_env = os.environ.copy()
    proc_env["PYTHONIOENCODING"] = "utf-8"
    proc_env["PYTHONUTF8"] = "1"
    if env:
        proc_env.update(env)

    try:
        proc = subprocess.Popen(
            [py, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=proc_env,
            encoding="utf-8",
            errors="replace",
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                yield f"data: {line}\n\n"
            else:
                # Non-JSON output → wrap as step message
                yield f'data: {json.dumps({"step": line, "progress": -1})}\n\n'

        proc.wait(timeout=30)
        if proc.returncode != 0:
            err = proc.stderr.read().strip() if proc.stderr else ""
            yield f'data: {json.dumps({"step": f"❌ 进程退出码 {proc.returncode}: {err[:300]}", "error": True})}\n\n'

    except subprocess.TimeoutExpired:
        proc.kill()
        yield f'data: {json.dumps({"step": "❌ 操作超时", "error": True})}\n\n'
    except Exception as e:
        yield f'data: {json.dumps({"step": f"❌ {e}", "error": True})}\n\n'


# ======================= Tokenizer 下载脚本 =======================
def script_download_tokenizer(source: str, output_dir: str) -> str:
    return f'''
import json, sys, os

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def p(step, progress=0, **kw):
    d = json.dumps({{"step": step, "progress": progress, **kw}}, ensure_ascii=False)
    sys.stdout.buffer.write((d + chr(10)).encode("utf-8", "replace"))
    sys.stdout.buffer.flush()

p("📦 准备下载 tokenizer: {source}", 5)

try:
    from transformers import AutoTokenizer
except ImportError:
    p("❌ 需要安装 transformers: pip install transformers", 0, error=True)
    sys.exit(0)

p("📥 正在从 HuggingFace 下载...", 20)
try:
    tok = AutoTokenizer.from_pretrained("{source}", trust_remote_code=True)
    p("💾 正在保存到本地...", 70)
    tok.save_pretrained("{output_dir}")

    p("🔍 测试分词效果...", 85)
    test_texts = [
        "Hello, world! This is a test.",
        "你好世界！这是一个测试。",
        "The quick brown fox jumps over the lazy dog.",
        "人工智能正在改变我们的生活方式。",
    ]
    test_results = []
    for t in test_texts:
        ids = tok.encode(t)
        tokens = tok.convert_ids_to_tokens(ids)
        test_results.append({{
            "text": t,
            "tokens": tokens[:30],
            "ids": ids[:30],
            "length": len(ids),
        }})

    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)
    p("✅ Tokenizer 下载完成！", 100, done=True,
      vocab_size=vocab_size, test_results=test_results)

except Exception as e:
    p(f"❌ 下载失败: {{e}}", 0, error=True)
'''


# ======================= Tokenizer 训练脚本 =======================
def script_train_tokenizer(
    data_files: list[str],
    output_dir: str,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> str:
    if special_tokens is None:
        special_tokens = ["<|endoftext|>", "<pad>", "<unk>", "<s>", "</s>"]

    return f'''
import json, sys, os

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def p(step, progress=0, **kw):
    d = json.dumps({{"step": step, "progress": progress, **kw}}, ensure_ascii=False)
    sys.stdout.buffer.write((d + chr(10)).encode("utf-8", "replace"))
    sys.stdout.buffer.flush()

p("📦 准备训练 BPE tokenizer", 5)

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
except ImportError:
    p("❌ 需要安装 tokenizers: pip install tokenizers", 0, error=True)
    sys.exit(0)

p("🔧 初始化 BPE 模型...", 10)
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size={vocab_size},
    min_frequency={min_frequency},
    special_tokens={json.dumps(special_tokens)},
    show_progress=False,
)

files = {json.dumps(data_files)}
total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
p(f"📊 训练数据: {{len(files)}} 个文件, {{total_size/1024/1024:.1f}} MB", 20)

p("🚀 开始训练...(这可能需要几分钟)", 30)
tokenizer.train(files, trainer)

p("💾 保存 tokenizer...", 80)
os.makedirs("{output_dir}", exist_ok=True)
tokenizer.save(os.path.join("{output_dir}", "tokenizer.json"))

p("🔍 测试分词效果...", 90)
vocab = tokenizer.get_vocab()
sample_tokens = sorted(vocab.items(), key=lambda x: x[1])[:50]

test_texts = [
    "Hello, world! This is a test.",
    "你好世界！这是一个测试。",
    "The quick brown fox jumps over the lazy dog.",
    "人工智能正在改变我们的生活方式。",
]
test_results = []
for t in test_texts:
    enc = tokenizer.encode(t)
    test_results.append({{
        "text": t,
        "tokens": enc.tokens[:30],
        "ids": enc.ids[:30],
        "length": len(enc.ids),
    }})

p("✅ Tokenizer 训练完成！", 100, done=True,
  vocab_size=tokenizer.get_vocab_size(),
  sample_tokens=[t[0] for t in sample_tokens],
  test_results=test_results)
'''


# ======================= 数据预处理脚本 =======================
def script_process_dataset(
    dataset_files: list[str],
    tokenizer_dir: str,
    max_seq_len: int,
    output_bin: str,
    output_meta: str,
    column_config: dict | None = None,
    column_separator: str = "\n",
) -> str:
    if column_config is None:
        column_config = {}
    return f'''
import json, sys, os, array
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def p(step, progress=0, **kw):
    d = json.dumps({{"step": step, "progress": progress, **kw}}, ensure_ascii=False)
    sys.stdout.buffer.write((d + chr(10)).encode("utf-8", "replace"))
    sys.stdout.buffer.flush()

p("📦 加载 tokenizer...", 5)

tok_dir = {json.dumps(tokenizer_dir)}
tok_json = os.path.join(tok_dir, "tokenizer.json")

if os.path.exists(tok_json):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_json)
    def encode(text):
        return tok.encode(text).ids
    vocab_size = tok.get_vocab_size()
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    def encode(text):
        return tok.encode(text)
    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)

p(f"✅ Tokenizer 加载完成 (词表: {{vocab_size}})", 10)

files = {json.dumps(dataset_files)}
max_seq = {max_seq_len}
column_config = {json.dumps(column_config)}
col_sep = {json.dumps(column_separator)}
all_ids = []
total_chars = 0
total_rows_extracted = 0

def extract_text_from_structured(fpath, columns):
    """从结构化文件中按列提取文本"""
    ext = os.path.splitext(fpath)[1].lower()
    texts = []

    if ext in ('.jsonl',):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        parts = []
                        for col in columns:
                            val = obj.get(col, "")
                            if val and isinstance(val, str):
                                parts.append(val.strip())
                        if parts:
                            texts.append(col_sep.join(parts))
                except json.JSONDecodeError:
                    continue

    elif ext in ('.json',):
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    parts = []
                    for col in columns:
                        val = obj.get(col, "")
                        if val and isinstance(val, str):
                            parts.append(val.strip())
                    if parts:
                        texts.append(col_sep.join(parts))

    elif ext in ('.csv', '.tsv'):
        import csv
        delimiter = chr(9) if ext == '.tsv' else ','
        with open(fpath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                parts = []
                for col in columns:
                    val = row.get(col, "")
                    if val and val.strip():
                        parts.append(val.strip())
                if parts:
                    texts.append(col_sep.join(parts))

    return texts

for fi, fpath in enumerate(files):
    pct = 10 + int(60 * fi / len(files))
    fname = os.path.basename(fpath)
    p(f"📖 处理文件 {{fi+1}}/{{len(files)}}: {{fname}}", pct)

    ext = os.path.splitext(fpath)[1].lower()
    file_columns = column_config.get(fname, [])

    # If columns specified for this file, extract from structured format
    if file_columns and ext in ('.jsonl', '.json', '.csv', '.tsv'):
        p(f"  📋 按列提取: {{', '.join(file_columns)}}", pct)
        texts = extract_text_from_structured(fpath, file_columns)
        total_rows_extracted += len(texts)
        text = (chr(10) + chr(10)).join(texts)
        total_chars += len(text)
        p(f"  📊 提取 {{len(texts)}} 条记录, {{len(text):,}} 字符", pct + 2)
    else:
        # Plain text: read entire file
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        total_chars += len(text)

    # Tokenize in chunks
    chunk_size = 100000
    chunks = range(0, len(text), chunk_size)
    for ci, i in enumerate(chunks):
        chunk = text[i:i+chunk_size]
        ids = encode(chunk)
        all_ids.extend(ids)
        if ci % 5 == 0 and len(chunks) > 5:
            sub_pct = pct + int(60 / len(files) * ci / len(chunks))
            p(f"  ✏️ 分词中... {{len(all_ids):,}} tokens", min(sub_pct, 70))

total_tokens = len(all_ids)
p(f"📊 分词完成: {{total_chars:,}} 字符 → {{total_tokens:,}} tokens", 75)

# Pack
dtype = "H" if vocab_size < 65536 else "I"
bytes_per_token = 2 if dtype == "H" else 4
n_seqs = total_tokens // max_seq
usable_tokens = n_seqs * max_seq
all_ids = all_ids[:usable_tokens]

p(f"📦 打包为 {{n_seqs:,}} 个序列 (长度 {{max_seq}})...", 85)
arr = array.array(dtype, all_ids)

out_bin = {json.dumps(output_bin)}
os.makedirs(os.path.dirname(out_bin), exist_ok=True)
with open(out_bin, "wb") as f:
    f.write(arr.tobytes())

file_size = os.path.getsize(out_bin)

meta = {{
    "total_chars": total_chars,
    "total_tokens": total_tokens,
    "usable_tokens": usable_tokens,
    "n_sequences": n_seqs,
    "max_seq_len": max_seq,
    "vocab_size": vocab_size,
    "dtype": dtype,
    "bytes_per_token": bytes_per_token,
    "file_size": file_size,
    "compression_ratio": round(total_chars / total_tokens, 2) if total_tokens > 0 else 0,
    "files_processed": files,
    "column_config": column_config,
    "rows_extracted": total_rows_extracted,
}}
meta_path = {json.dumps(output_meta)}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

p(f"✅ 预处理完成！{{usable_tokens:,}} tokens, {{n_seqs:,}} 序列, {{file_size/1024/1024:.1f}} MB",
  100, done=True, **meta)
'''


# ======================= SFT 对话数据预处理脚本 =======================
def script_process_sft_dataset(
    dataset_files: list[str],
    tokenizer_dir: str,
    max_seq_len: int,
    output_tokens: str,
    output_masks: str,
    output_meta: str,
    chat_template: str = "chatml",
) -> str:
    return f'''
import json, sys, os, array, struct
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def p(step, progress=0, **kw):
    d = json.dumps({{"step": step, "progress": progress, **kw}}, ensure_ascii=False)
    sys.stdout.buffer.write((d + chr(10)).encode("utf-8", "replace"))
    sys.stdout.buffer.flush()

p("📦 加载 tokenizer...", 5)

tok_dir = {json.dumps(tokenizer_dir)}
tok_json = os.path.join(tok_dir, "tokenizer.json")
is_hf_tok = False

if os.path.exists(tok_json):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_json)
    vocab_size = tok.get_vocab_size()
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)
    is_hf_tok = True

original_vocab_size = vocab_size
p(f"✅ Tokenizer 加载完成 (词表: {{vocab_size}})", 8)

# ============ Inject chat template special tokens ============
template = "{chat_template}"
SPECIAL_TOKENS_MAP = {{
    "chatml": ["<|im_start|>", "<|im_end|>"],
    "llama": ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "</s>"],
    "simple": [],
}}
needed_tokens = SPECIAL_TOKENS_MAP.get(template, [])
added_tokens = []

if needed_tokens:
    if is_hf_tok:
        # transformers AutoTokenizer
        existing = set(tok.get_vocab().keys())
        to_add = [t for t in needed_tokens if t not in existing]
        if to_add:
            tok.add_special_tokens({{"additional_special_tokens": to_add}})
            vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)
            added_tokens = to_add
            tok.save_pretrained(tok_dir)
            p(f"✨ 注入 {{len(to_add)}} 个特殊 token: {{', '.join(to_add)}} → 词表: {{vocab_size}}", 9)
    else:
        # tokenizers library (custom BPE)
        from tokenizers import AddedToken
        existing_vocab = tok.get_vocab()
        to_add = [t for t in needed_tokens if t not in existing_vocab]
        if to_add:
            added = tok.add_special_tokens([AddedToken(t, special=True) for t in to_add])
            vocab_size = tok.get_vocab_size()
            added_tokens = to_add
            tok.save(tok_json)
            p(f"✨ 注入 {{len(to_add)}} 个特殊 token: {{', '.join(to_add)}} → 词表: {{vocab_size}}", 9)

if not added_tokens:
    p(f"✅ 特殊 token 已存在，无需注入", 9)

# Build encode function (must be after token injection)
if is_hf_tok:
    def encode(text):
        return tok.encode(text)
else:
    def encode(text):
        return tok.encode(text).ids

p(f"📝 准备处理对话数据 (词表: {{vocab_size}}, 模板: {{template}})", 10)
def format_conversation(conv):
    """Convert conversation list to ChatML formatted string, returning (full_text, assistant_ranges)"""
    parts = []
    assistant_ranges = []  # (start_char, end_char) of assistant content
    pos = 0
    for msg in conv:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if template == "chatml":
            header = f"<|im_start|>{{role}}\\n"
            footer = "<|im_end|>\\n"
        elif template == "llama":
            if role == "system":
                header = "<<SYS>>\\n"
                footer = "\\n<</SYS>>\\n\\n"
            elif role == "user":
                header = "[INST] "
                footer = " [/INST]\\n"
            else:
                header = ""
                footer = "\\n"
        else:  # simple
            header = f"### {{role.capitalize()}}:\\n"
            footer = "\\n\\n"

        part = header + content + footer
        start_pos = pos + len(header)
        end_pos = pos + len(header) + len(content) + len(footer)
        if role == "assistant":
            assistant_ranges.append((start_pos, end_pos))
        pos += len(part)
        parts.append(part)
    return "".join(parts), assistant_ranges

files = {json.dumps(dataset_files)}
max_seq = {max_seq_len}

# Read all conversations
all_convs = []
for fi, fpath in enumerate(files):
    pct = 10 + int(20 * fi / max(len(files), 1))
    p(f"📖 读取文件 {{fi+1}}/{{len(files)}}: {{os.path.basename(fpath)}}", pct)
    with open(fpath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                conv = obj.get("conversations") or obj.get("messages") or obj.get("conversation")
                if conv and isinstance(conv, list) and len(conv) >= 2:
                    all_convs.append(conv)
            except json.JSONDecodeError:
                pass

p(f"📋 共 {{len(all_convs)}} 条对话", 35)
if not all_convs:
    p("❌ 没有找到有效的对话数据。JSONL 格式应为: {{\\\"conversations\\\": [{{\\\"role\\\": \\\"user\\\", \\\"content\\\": \\\"...\\\"}}, ...]}}", 0, error=True)
    sys.exit(0)

# Process each conversation into tokens + loss mask
all_tokens = []
all_masks = []
total_convs_used = 0
total_assistant_tokens = 0

for ci, conv in enumerate(all_convs):
    if (ci + 1) % max(1, len(all_convs) // 20) == 0:
        pct = 35 + int(50 * (ci + 1) / len(all_convs))
        p(f"🔄 处理对话 {{ci+1}}/{{len(all_convs)}}", min(pct, 85))

    full_text, asst_ranges = format_conversation(conv)

    # Tokenize the full text
    token_ids = encode(full_text)
    if len(token_ids) < 4:
        continue

    # Build character-to-token mapping by tokenizing progressively
    # Simpler approach: tokenize each part separately and track boundaries
    # Instead, use a mask based on role tokens
    # We re-tokenize each segment to build the mask accurately
    mask = [0] * len(token_ids)

    # Token-level approach: tokenize up to assistant start, then mark the rest
    char_pos = 0
    for msg in conv:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if template == "chatml":
            header = f"<|im_start|>{{role}}\\n"
            footer = "<|im_end|>\\n"
        elif template == "llama":
            if role == "system":
                header = "<<SYS>>\\n"
                footer = "\\n<</SYS>>\\n\\n"
            elif role == "user":
                header = "[INST] "
                footer = " [/INST]\\n"
            else:
                header = ""
                footer = "\\n"
        else:
            header = f"### {{role.capitalize()}}:\\n"
            footer = "\\n\\n"

        # Tokens for this part
        pre_text = full_text[:char_pos + len(header)]
        pre_tokens = encode(pre_text)
        post_text = full_text[:char_pos + len(header) + len(content) + len(footer)]
        post_tokens = encode(post_text)

        if role == "assistant":
            # Mark tokens from pre_tokens length to post_tokens length
            start_tok = len(pre_tokens)
            end_tok = len(post_tokens)
            for ti in range(start_tok, min(end_tok, len(mask))):
                mask[ti] = 1
                total_assistant_tokens += 1

        char_pos += len(header) + len(content) + len(footer)

    # Truncate or pad to max_seq_len
    if len(token_ids) > max_seq:
        token_ids = token_ids[:max_seq]
        mask = mask[:max_seq]
    elif len(token_ids) < max_seq:
        pad_len = max_seq - len(token_ids)
        token_ids = token_ids + [0] * pad_len
        mask = mask + [0] * pad_len

    all_tokens.extend(token_ids)
    all_masks.extend(mask)
    total_convs_used += 1

p(f"✅ 处理完成: {{total_convs_used}} 条对话, {{total_assistant_tokens}} 个 assistant tokens", 88)

if total_convs_used == 0:
    p("❌ 没有生成有效的训练数据", 0, error=True)
    sys.exit(0)

# Write binary files
dtype = "H" if vocab_size < 65536 else "I"
tokens_arr = array.array(dtype, all_tokens)
masks_arr = array.array("B", all_masks)  # uint8

tokens_path = {json.dumps(output_tokens)}
masks_path = {json.dumps(output_masks)}
meta_path = {json.dumps(output_meta)}

with open(tokens_path, "wb") as f:
    f.write(tokens_arr.tobytes())
with open(masks_path, "wb") as f:
    f.write(masks_arr.tobytes())

total_tokens = len(all_tokens)
file_size = os.path.getsize(tokens_path)

meta = {{
    "mode": "sft",
    "chat_template": template,
    "total_conversations": total_convs_used,
    "total_tokens": total_tokens,
    "assistant_tokens": total_assistant_tokens,
    "assistant_pct": round(total_assistant_tokens / max(total_tokens, 1) * 100, 1),
    "n_sequences": total_convs_used,
    "max_seq_len": max_seq,
    "vocab_size": vocab_size,
    "original_vocab_size": original_vocab_size,
    "added_special_tokens": added_tokens,
    "vocab_expanded": vocab_size != original_vocab_size,
    "dtype": dtype,
    "file_size": file_size,
}}

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

p(f"✅ SFT 数据预处理完成！{{total_convs_used}} 条对话, {{total_assistant_tokens:,}} 个训练 tokens",
  100, done=True, **meta)
'''


# ======================= GGUF 导出脚本 =======================
def script_export_hf(checkpoint_path: str, hf_dir: str, project_dir: str) -> str:
    export_dir = str(Path(hf_dir).parent)  # hf_dir is export/hf_model, we want export/
    return f'''
import json, sys, os, torch, shutil, struct, subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def p(step, progress=0, **kw):
    d = json.dumps({{"step": step, "progress": progress, **kw}}, ensure_ascii=False)
    sys.stdout.buffer.write((d + chr(10)).encode("utf-8", "replace"))
    sys.stdout.buffer.flush()

p("📦 加载 checkpoint...", 10)
ckpt = torch.load({json.dumps(checkpoint_path)}, map_location="cpu", weights_only=False)
arch = ckpt["arch"]
state = ckpt["model"]
step = ckpt.get("step", 0)

p(f"📐 模型架构: {{arch['num_layers']}}层, d={{arch['hidden_dim']}}, step={{step}}", 20)

# ===== Step 1: Save HuggingFace format =====
p("💾 保存为 HuggingFace 格式...", 30)
os.makedirs({json.dumps(hf_dir)}, exist_ok=True)

tie = arch.get("tie_word_embeddings", True)

# Remap keys to HF Llama format
mapped = {{}}
for k, v in state.items():
    nk = k
    if nk.startswith("norm."):
        nk = "model.norm." + nk[len("norm."):]
    nk = nk.replace("tok_emb.weight", "model.embed_tokens.weight")
    nk = nk.replace("pos_emb.weight", "model.embed_positions.weight")
    nk = nk.replace(".attn.q.", ".self_attn.q_proj.")
    nk = nk.replace(".attn.k.", ".self_attn.k_proj.")
    nk = nk.replace(".attn.v.", ".self_attn.v_proj.")
    nk = nk.replace(".attn.o.", ".self_attn.o_proj.")
    nk = nk.replace(".ffn.gate.", ".mlp.gate_proj.")
    nk = nk.replace(".ffn.up.", ".mlp.up_proj.")
    nk = nk.replace(".ffn.down.", ".mlp.down_proj.")
    nk = nk.replace(".norm1.", ".input_layernorm.")
    nk = nk.replace(".norm2.", ".post_attention_layernorm.")
    nk = nk.replace("layers.", "model.layers.")
    mapped[nk] = v

# Clone all tensors to break storage sharing + convert to float16
mapped = {{k: v.detach().clone().contiguous().to(torch.float16) for k, v in mapped.items()}}

# Ensure lm_head exists
if "lm_head.weight" not in mapped and "model.embed_tokens.weight" in mapped:
    mapped["lm_head.weight"] = mapped["model.embed_tokens.weight"].clone()

try:
    from safetensors.torch import save_file
    save_file(mapped, os.path.join({json.dumps(hf_dir)}, "model.safetensors"))
    p("✅ safetensors 已保存", 40)
except ImportError:
    torch.save(mapped, os.path.join({json.dumps(hf_dir)}, "pytorch_model.bin"))
    p("✅ pytorch_model.bin 已保存", 40)

# Config
act = arch.get("activation", "swiglu")
hf_act = "silu" if act == "swiglu" else act
hf_config = {{
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": arch["hidden_dim"],
    "intermediate_size": arch["intermediate_dim"],
    "num_hidden_layers": arch["num_layers"],
    "num_attention_heads": arch["num_heads"],
    "num_key_value_heads": arch.get("num_kv_heads", arch["num_heads"]),
    "hidden_act": hf_act,
    "vocab_size": arch["vocab_size"],
    "max_position_embeddings": arch["max_seq_len"],
    "rms_norm_eps": arch.get("norm_eps", 1e-5),
    "rope_theta": arch.get("rope_theta", 10000.0),
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
}}
with open(os.path.join({json.dumps(hf_dir)}, "config.json"), "w") as f:
    json.dump(hf_config, f, indent=2)

# Copy tokenizer
tok_dir = os.path.join({json.dumps(project_dir)}, "tokenizer")
if os.path.exists(tok_dir):
    for fn in os.listdir(tok_dir):
        src = os.path.join(tok_dir, fn)
        dst = os.path.join({json.dumps(hf_dir)}, fn)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

# Generate tokenizer_config.json if missing
tok_cfg_path = os.path.join({json.dumps(hf_dir)}, "tokenizer_config.json")
if not os.path.exists(tok_cfg_path):
    tok_cfg = {{
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": arch["max_seq_len"],
    }}
    if os.path.exists(os.path.join({json.dumps(hf_dir)}, "tokenizer.json")):
        tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    with open(tok_cfg_path, "w") as f:
        json.dump(tok_cfg, f, indent=2)

p("📝 HF 格式完成，开始 GGUF 转换...", 45)

# ===== Step 2: Convert to GGUF =====
gguf_path = os.path.join({json.dumps(export_dir)}, "model-pretrain.gguf")
hf_model_dir = {json.dumps(hf_dir)}

# ---------- Manual GGUF construction (primary method) ----------
p("🔧 构建 GGUF...", 50)
try:
    try:
        import gguf
    except ImportError:
        p("📦 安装 gguf 库...", 52)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf", "-q"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import gguf
    import numpy as np

    p(f"📦 gguf 版本: {{getattr(gguf, '__version__', 'unknown')}}", 54)

    # ----- Read tokenizer -----
    tok_json_path = os.path.join(hf_model_dir, "tokenizer.json")
    vocab_txt_path = os.path.join(hf_model_dir, "vocab.txt")

    raw_vocab = {{}}   # token_str -> token_id
    tok_type = "BPE"   # default
    bpe_merges = []

    # Try tokenizer.json first
    if os.path.exists(tok_json_path):
        with open(tok_json_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        tok_type = tok_data.get("model", {{}}).get("type", "BPE")
        raw_vocab = tok_data.get("model", {{}}).get("vocab", {{}})
        bpe_merges = tok_data.get("model", {{}}).get("merges", [])

    # Fallback to vocab.txt (BERT)
    if not raw_vocab and os.path.exists(vocab_txt_path):
        tok_type = "WordPiece"
        with open(vocab_txt_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                raw_vocab[line.strip()] = i

    is_bpe = (tok_type == "BPE" and len(bpe_merges) > 0)
    p(f"📝 Tokenizer: {{tok_type}}, vocab={{len(raw_vocab)}}, merges={{len(bpe_merges)}}", 56)

    # ----- Build vocab arrays -----
    sorted_vocab = sorted(raw_vocab.items(), key=lambda x: x[1])

    # Detect special tokens from added_tokens
    special_ids = set()
    if os.path.exists(tok_json_path):
        with open(tok_json_path, "r", encoding="utf-8") as f:
            for at in json.load(f).get("added_tokens", []):
                special_ids.add(at.get("id", -1))

    # Known special token names
    SPECIAL_NAMES = {{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                     "<pad>", "<unk>", "<s>", "</s>", "<|endoftext|>",
                     "<|padding|>", "<mask>"}}

    vocab_tokens = []
    vocab_scores = []
    vocab_types = []
    for token_str, token_id in sorted_vocab:
        vocab_tokens.append(token_str.encode("utf-8", errors="replace"))
        vocab_scores.append(-float(token_id))
        if token_id in special_ids or token_str in SPECIAL_NAMES:
            vocab_types.append(3)  # CONTROL
        else:
            vocab_types.append(1)  # NORMAL

    # Pad/trim to model vocab_size
    model_vocab_size = arch["vocab_size"]
    while len(vocab_tokens) < model_vocab_size:
        idx = len(vocab_tokens)
        vocab_tokens.append(f"<pad{{idx}}>".encode("utf-8"))
        vocab_scores.append(-100000.0)
        vocab_types.append(3)
    vocab_tokens = vocab_tokens[:model_vocab_size]
    vocab_scores = vocab_scores[:model_vocab_size]
    vocab_types = vocab_types[:model_vocab_size]

    # ----- Create GGUF writer -----
    p("📐 写入 GGUF 元数据...", 60)

    # Delete old file if exists
    if os.path.exists(gguf_path):
        os.remove(gguf_path)

    writer = gguf.GGUFWriter(gguf_path, "llama")

    # Model metadata (these are the most basic, should exist in all gguf versions)
    n_heads = arch["num_heads"]
    n_kv = arch.get("num_kv_heads", n_heads)
    head_dim = arch["hidden_dim"] // n_heads

    writer.add_name("pretrain-lab-model")
    writer.add_context_length(arch["max_seq_len"])
    writer.add_embedding_length(arch["hidden_dim"])
    writer.add_block_count(arch["num_layers"])
    writer.add_feed_forward_length(arch["intermediate_dim"])
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv)
    writer.add_layer_norm_rms_epsilon(arch.get("norm_eps", 1e-5))
    writer.add_file_type(1)  # MOSTLY_F16

    # These might not exist in very old gguf versions — safe to skip
    for method_name, value in [
        ("add_rope_dimension_count", head_dim),
        ("add_rope_freq_base", arch.get("rope_theta", 10000.0)),
    ]:
        fn = getattr(writer, method_name, None)
        if fn:
            fn(value)

    # ----- Tokenizer -----
    # For llama.cpp: BPE needs merges, SPM doesn't.
    # WordPiece has no merges → use "llama" (SPM) which only needs token+score.
    gguf_tok = "gpt2" if is_bpe else "llama"
    writer.add_tokenizer_model(gguf_tok)
    writer.add_token_list(vocab_tokens)
    writer.add_token_scores(np.array(vocab_scores, dtype=np.float32))
    writer.add_token_types(np.array(vocab_types, dtype=np.int32))

    if is_bpe:
        merges_bytes = [m.encode("utf-8") if isinstance(m, str) else m for m in bpe_merges]
        writer.add_token_merges(merges_bytes)

    # Special token IDs — use try/except for each optional method
    bos_id = raw_vocab.get("<s>", raw_vocab.get("[CLS]", raw_vocab.get("<|endoftext|>", 0)))
    eos_id = raw_vocab.get("</s>", raw_vocab.get("[SEP]", raw_vocab.get("<|endoftext|>", 0)))
    writer.add_bos_token_id(bos_id)
    writer.add_eos_token_id(eos_id)

    for method_name, value in [
        ("add_unk_token_id", raw_vocab.get("<unk>", raw_vocab.get("[UNK]", 0))),
        ("add_pad_token_id", raw_vocab.get("<pad>", raw_vocab.get("[PAD]", 0))),
        ("add_add_bos_token", False),
        ("add_add_eos_token", False),
    ]:
        fn = getattr(writer, method_name, None)
        if fn:
            try:
                fn(value)
            except Exception:
                pass

    p(f"📝 词表: {{len(vocab_tokens)}}, tokenizer={{gguf_tok}}", 65)

    # ----- Tensors -----
    p("📐 写入权重张量...", 70)

    GGUF_MAP = {{
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }}
    for i in range(arch["num_layers"]):
        GGUF_MAP.update({{
            f"model.layers.{{i}}.input_layernorm.weight": f"blk.{{i}}.attn_norm.weight",
            f"model.layers.{{i}}.post_attention_layernorm.weight": f"blk.{{i}}.ffn_norm.weight",
            f"model.layers.{{i}}.self_attn.q_proj.weight": f"blk.{{i}}.attn_q.weight",
            f"model.layers.{{i}}.self_attn.k_proj.weight": f"blk.{{i}}.attn_k.weight",
            f"model.layers.{{i}}.self_attn.v_proj.weight": f"blk.{{i}}.attn_v.weight",
            f"model.layers.{{i}}.self_attn.o_proj.weight": f"blk.{{i}}.attn_output.weight",
            f"model.layers.{{i}}.mlp.gate_proj.weight": f"blk.{{i}}.ffn_gate.weight",
            f"model.layers.{{i}}.mlp.up_proj.weight": f"blk.{{i}}.ffn_up.weight",
            f"model.layers.{{i}}.mlp.down_proj.weight": f"blk.{{i}}.ffn_down.weight",
        }})

    written = 0
    skipped = []
    for hf_name, tensor in mapped.items():
        gguf_name = GGUF_MAP.get(hf_name)
        if gguf_name is None:
            skipped.append(hf_name)
            continue
        data = tensor.to(torch.float16).numpy()
        writer.add_tensor(gguf_name, data)
        written += 1

    if skipped:
        p(f"⚠️ 跳过 {{len(skipped)}} 个权重: {{', '.join(skipped[:3])}}", 78)

    expected = 3 + arch["num_layers"] * 9
    p(f"📦 写入 {{written}}/{{expected}} 个张量...", 82)

    if written < expected:
        p(f"❌ 张量数量不足: {{written}} < {{expected}}", 90)
        raise RuntimeError(f"张量数量不足: {{written}} < {{expected}}, 缺少权重映射")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    gguf_size = os.path.getsize(gguf_path)
    if gguf_size < 1000:
        raise RuntimeError(f"GGUF 文件异常小: {{gguf_size}} bytes")

    p(f"✅ GGUF 导出完成! ({{gguf_size / 1024 / 1024:.1f}} MB)", 100, done=True,
      hf_dir=hf_model_dir, gguf_path=gguf_path)

except Exception as e:
    import traceback
    err_detail = traceback.format_exc()
    p(f"❌ GGUF 构建失败: {{type(e).__name__}}: {{e}}", 92)
    p(f"📋 错误详情:", 93)
    # Print each line of traceback as separate progress step so it doesn't flash
    for i, line in enumerate(err_detail.strip().split(chr(10))[-8:]):
        p(f"   {{line.strip()}}", 93)
    p(f"💡 可手动转换: pip install llama-cpp-python 后用 convert_hf_to_gguf.py", 95)
    p(f"❌ 导出失败，请查看上方错误信息", 100, done=True,
      hf_dir=hf_model_dir, gguf_error=str(e))
'''


# ======================= HF 数据集下载脚本 =======================
def script_download_hf_dataset(
    hf_id: str,
    output_path: str,
    max_rows: int = 0,
) -> str:
    return f'''
import json, sys, os, time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def p(step, progress=0, **kw):
    d = json.dumps({{"step": step, "progress": progress, **kw}}, ensure_ascii=False)
    sys.stdout.buffer.write((d + chr(10)).encode("utf-8", "replace"))
    sys.stdout.buffer.flush()

try:
    from datasets import load_dataset, get_dataset_split_names
except ImportError:
    p("❌ 需要安装 datasets 库: pip install datasets", 0, error=True)
    sys.exit(0)

hf_id = {json.dumps(hf_id)}
out_path = {json.dumps(output_path)}
max_r = {max_rows}

try:
    p(f"🔍 查询数据集 {{hf_id}} 可用的 splits...", 5)

    # Auto-detect available splits
    try:
        available_splits = get_dataset_split_names(hf_id, trust_remote_code=True)
    except Exception:
        available_splits = None

    if available_splits:
        p(f"📋 可用 splits: {{', '.join(available_splits)}}", 8)
        if "train" in available_splits:
            split = "train"
        elif "all" in available_splits:
            split = "all"
        else:
            split = available_splits[0]
    else:
        split = "train"

    p(f"📥 正在下载 {{hf_id}} (split={{split}})... 这可能需要几分钟，取决于数据集大小", 10)

    # Track download progress with a background thread
    import threading
    dl_progress = {{"pct": 10, "msg": "下载中..."}}
    dl_done = threading.Event()
    def progress_reporter():
        pct = 12
        while not dl_done.is_set():
            if pct < 40:
                pct += 1
            dl_progress["pct"] = pct
            p(f"📥 下载中... ({{pct}}%)", pct)
            dl_done.wait(3.0)
    reporter = threading.Thread(target=progress_reporter, daemon=True)
    reporter.start()

    ds = load_dataset(hf_id, split=split, trust_remote_code=True)
    dl_done.set()
    total = len(ds)
    p(f"📋 数据集加载完成，共 {{total}} 条 (split={{split}})", 45)

    if max_r > 0 and max_r < total:
        ds = ds.select(range(max_r))
        total = max_r
        p(f"📦 已截取前 {{total}} 条", 50)

    # Detect text column
    text_cols = ["text", "content", "story", "document", "passage", "sentence",
                 "input", "output", "question", "answer", "prompt", "completion"]
    col = None
    for c in text_cols:
        if c in ds.column_names:
            col = c
            break
    if not col:
        for c in ds.column_names:
            sample = ds[0][c]
            if isinstance(sample, str) and len(sample) > 10:
                col = c
                break
    if not col:
        col = ds.column_names[0]

    p(f"📝 使用列: {{col}}", 52)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            txt = str(row.get(col, ""))
            if txt.strip():
                f.write(txt.strip() + "\\n\\n")
                written += 1
            if (i + 1) % max(1, total // 20) == 0 or i == total - 1:
                pct = 52 + int(45 * (i + 1) / total)
                p(f"✏️ 已写入 {{i+1}}/{{total}} 条", min(pct, 97))

    size = os.path.getsize(out_path)
    fname = os.path.basename(out_path)
    p(f"✅ 完成！保存到 {{fname}} ({{size/1024/1024:.1f}} MB, {{written}} 条有效数据)",
      100, done=True, filename=fname, total=written, size=size)

except Exception as e:
    p(f"❌ {{e}}", 0, error=True)
'''


def script_download_hf_dataset_sft(
    hf_id: str,
    output_path: str,
    max_rows: int = 0,
) -> str:
    """下载 HF 数据集并自动转换为 SFT JSONL（conversations 格式）"""
    return f'''
import json, sys, os, time, threading

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def p(step, progress=0, **kw):
    d = json.dumps({{"step": step, "progress": progress, **kw}}, ensure_ascii=False)
    sys.stdout.buffer.write((d + chr(10)).encode("utf-8", "replace"))
    sys.stdout.buffer.flush()

try:
    from datasets import load_dataset, get_dataset_split_names
except ImportError:
    p("❌ 需要安装 datasets 库: pip install datasets", 0, error=True)
    sys.exit(0)

hf_id = {json.dumps(hf_id)}
out_path = {json.dumps(output_path)}
max_r = {max_rows}

try:
    p(f"🔍 查询数据集 {{hf_id}} 可用的 splits...", 5)
    try:
        available_splits = get_dataset_split_names(hf_id, trust_remote_code=True)
    except Exception:
        available_splits = None

    if available_splits:
        p(f"📋 可用 splits: {{', '.join(available_splits)}}", 8)
        split = "train" if "train" in available_splits else (available_splits[0])
    else:
        split = "train"

    p(f"📥 正在下载 {{hf_id}} (split={{split}})...", 10)

    dl_done = threading.Event()
    def progress_reporter():
        pct = 12
        while not dl_done.is_set():
            if pct < 40: pct += 1
            p(f"📥 下载中... ({{pct}}%)", pct)
            dl_done.wait(3.0)
    threading.Thread(target=progress_reporter, daemon=True).start()

    ds = load_dataset(hf_id, split=split, trust_remote_code=True)
    dl_done.set()
    total = len(ds)
    p(f"📋 数据集加载完成，共 {{total}} 条 (split={{split}})", 45)
    p(f"📋 列名: {{', '.join(ds.column_names)}}", 46)

    if max_r > 0 and max_r < total:
        ds = ds.select(range(max_r))
        total = max_r

    cols = ds.column_names
    sample = ds[0]

    # ============ Auto-detect format and convert to conversations JSONL ============
    def detect_format():
        # 1. Already conversations/messages format
        if "conversations" in cols or "messages" in cols:
            key = "conversations" if "conversations" in cols else "messages"
            if isinstance(sample[key], list):
                return "conversations", key
        # 2. Alpaca format: instruction + (input) + output
        if "instruction" in cols and "output" in cols:
            return "alpaca", None
        # 3. ShareGPT format: conversation list with from/value
        if "conversation" in cols and isinstance(sample.get("conversation"), list):
            return "sharegpt", "conversation"
        # 4. OpenOrca format: system_prompt + question + response
        if "question" in cols and "response" in cols:
            return "openorca", None
        # 5. Prompt + completion
        if "prompt" in cols and "completion" in cols:
            return "prompt_completion", None
        # 6. Q&A style
        for qk in ["question", "query", "input", "prompt", "instruction"]:
            for ak in ["answer", "response", "output", "completion", "reply"]:
                if qk in cols and ak in cols:
                    return "qa", (qk, ak)
        return "unknown", None

    fmt, meta = detect_format()
    p(f"🔍 检测到格式: {{fmt}}", 50)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            conv = None

            if fmt == "conversations":
                raw = row[meta]
                if isinstance(raw, list) and len(raw) >= 2:
                    # Normalize role names
                    msgs = []
                    for m in raw:
                        if isinstance(m, dict):
                            role = m.get("role", m.get("from", "user"))
                            content = m.get("content", m.get("value", ""))
                            if role in ("human", "user"): role = "user"
                            elif role in ("gpt", "assistant", "bot"): role = "assistant"
                            if content: msgs.append({{"role": role, "content": str(content)}})
                    if len(msgs) >= 2:
                        conv = msgs

            elif fmt == "alpaca":
                instruction = str(row.get("instruction", "")).strip()
                inp = str(row.get("input", "")).strip()
                output = str(row.get("output", "")).strip()
                if instruction and output:
                    user_msg = f"{{instruction}}\\n{{inp}}" if inp else instruction
                    conv = [{{"role": "user", "content": user_msg}}, {{"role": "assistant", "content": output}}]

            elif fmt == "sharegpt":
                raw = row[meta]
                if isinstance(raw, list) and len(raw) >= 2:
                    msgs = []
                    for m in raw:
                        role = m.get("from", m.get("role", "user"))
                        content = m.get("value", m.get("content", ""))
                        if role in ("human", "user"): role = "user"
                        elif role in ("gpt", "assistant", "bot"): role = "assistant"
                        if content: msgs.append({{"role": role, "content": str(content)}})
                    if len(msgs) >= 2:
                        conv = msgs

            elif fmt == "openorca":
                sys_prompt = str(row.get("system_prompt", "")).strip()
                question = str(row.get("question", "")).strip()
                response = str(row.get("response", "")).strip()
                if question and response:
                    conv = []
                    if sys_prompt: conv.append({{"role": "system", "content": sys_prompt}})
                    conv.append({{"role": "user", "content": question}})
                    conv.append({{"role": "assistant", "content": response}})

            elif fmt == "prompt_completion":
                prompt = str(row.get("prompt", "")).strip()
                completion = str(row.get("completion", "")).strip()
                if prompt and completion:
                    conv = [{{"role": "user", "content": prompt}}, {{"role": "assistant", "content": completion}}]

            elif fmt == "qa":
                qk, ak = meta
                q = str(row.get(qk, "")).strip()
                a = str(row.get(ak, "")).strip()
                if q and a:
                    conv = [{{"role": "user", "content": q}}, {{"role": "assistant", "content": a}}]

            else:
                # Unknown format: try to find any two text columns
                texts = []
                for c in cols:
                    val = row.get(c, "")
                    if isinstance(val, str) and len(val.strip()) > 5:
                        texts.append(val.strip())
                if len(texts) >= 2:
                    conv = [{{"role": "user", "content": texts[0]}}, {{"role": "assistant", "content": texts[1]}}]

            if conv:
                f.write(json.dumps({{"conversations": conv}}, ensure_ascii=False) + "\\n")
                written += 1
            else:
                skipped += 1

            if (i + 1) % max(1, total // 20) == 0 or i == total - 1:
                pct = 52 + int(45 * (i + 1) / total)
                p(f"✏️ 已转换 {{i+1}}/{{total}} 条 (有效: {{written}}, 跳过: {{skipped}})", min(pct, 97))

    size = os.path.getsize(out_path)
    fname = os.path.basename(out_path)
    p(f"✅ 完成！保存到 {{fname}} ({{size/1024/1024:.1f}} MB, {{written}} 条对话)",
      100, done=True, filename=fname, total=written, size=size)

except Exception as e:
    import traceback
    p(f"❌ {{e}}\\n{{traceback.format_exc()}}", 0, error=True)
'''
