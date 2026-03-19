"""data_manager.py — 预训练数据集管理：上传、下载、统计、预处理"""
import json, os, shutil, time, re
from pathlib import Path
from datetime import datetime
from collections import Counter

# ======================= 推荐数据集（教育用途）=======================
RECOMMENDED_DATASETS = [
    {
        "id": "roneneldan/TinyStories",
        "name": "TinyStories",
        "description": "由 GPT-3.5/4 生成的简短英文故事，专为小模型训练设计。数据干净、质量高、体积小。",
        "lang": "en",
        "size": "~470MB",
        "rows": "~2.1M",
        "difficulty": "入门",
        "recommended_for": "1M-50M 参数模型，首次实验的最佳选择",
    },
    {
        "id": "wikitext/wikitext-103-raw-v1",
        "name": "WikiText-103",
        "description": "来自维基百科的长文本语料，经典的语言模型评测数据集。",
        "lang": "en",
        "size": "~500MB",
        "rows": "~1.8M",
        "difficulty": "入门",
        "recommended_for": "学习语言模型训练的经典选择",
    },
    {
        "id": "wikitext/wikitext-2-raw-v1",
        "name": "WikiText-2",
        "description": "WikiText 的小型版本，适合快速实验。",
        "lang": "en",
        "size": "~12MB",
        "rows": "~36K",
        "difficulty": "入门",
        "recommended_for": "快速验证训练流程，几分钟即可完成",
    },
    {
        "id": "silk-road/chinese-wikitexts",
        "name": "中文维基百科",
        "description": "中文维基百科提取的纯文本，中文预训练的基础语料。",
        "lang": "zh",
        "size": "~1.5GB",
        "rows": "~1M",
        "difficulty": "中等",
        "recommended_for": "中文模型训练",
    },
    {
        "id": "cerebras/SlimPajama-627B",
        "name": "SlimPajama (采样)",
        "description": "大规模高质量英文语料的清洗版本。注意：完整数据集非常大，建议只下载一小部分。",
        "lang": "en",
        "size": "~627GB (完整)",
        "rows": "~600B tokens",
        "difficulty": "高级",
        "recommended_for": "100M+ 参数模型，需要大量数据",
    },
    {
        "id": "HuggingFaceFW/fineweb-edu",
        "name": "FineWeb-Edu (采样)",
        "description": "经过教育质量过滤的高质量网页文本。",
        "lang": "en",
        "size": "~1.3TB (完整)",
        "rows": "~1.3T tokens",
        "difficulty": "高级",
        "recommended_for": "高质量英文预训练",
    },
]

# ======================= SFT 推荐数据集 =======================
RECOMMENDED_SFT_DATASETS = [
    {
        "id": "tatsu-lab/alpaca",
        "name": "Stanford Alpaca",
        "description": "经典的指令微调数据集，由 GPT-3.5 生成的 52K 条英文指令-回复对。",
        "lang": "en",
        "size": "~40MB",
        "rows": "~52K",
        "difficulty": "入门",
        "recommended_for": "首次 SFT 实验的最佳选择",
        "format_hint": "instruction/input/output 格式，需转换",
    },
    {
        "id": "shibing624/alpaca-zh",
        "name": "Alpaca 中文",
        "description": "中文版 Alpaca 指令微调数据集。",
        "lang": "zh",
        "size": "~30MB",
        "rows": "~51K",
        "difficulty": "入门",
        "recommended_for": "中文 SFT 实验",
        "format_hint": "instruction/input/output 格式",
    },
    {
        "id": "Open-Orca/OpenOrca",
        "name": "OpenOrca",
        "description": "大规模多样化的指令数据集，包含各种推理和知识任务。",
        "lang": "en",
        "size": "~3GB",
        "rows": "~4.2M",
        "difficulty": "中等",
        "recommended_for": "训练通用对话和推理能力",
        "format_hint": "system_prompt/question/response 格式",
    },
    {
        "id": "BAAI/COIG",
        "name": "COIG 中文指令",
        "description": "由 BAAI 整理的中文指令数据集，包含翻译、考试、人类价值观等多种类别。",
        "lang": "zh",
        "size": "~500MB",
        "rows": "~190K",
        "difficulty": "中等",
        "recommended_for": "中文多领域 SFT",
        "format_hint": "conversations 格式",
    },
]


def get_recommended_datasets() -> list:
    return RECOMMENDED_DATASETS


def get_recommended_sft_datasets() -> list:
    return RECOMMENDED_SFT_DATASETS


# ======================= 数据集文件管理 =======================
def datasets_dir(project_dir: Path) -> Path:
    d = project_dir / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_datasets(project_dir: Path) -> list:
    """列出项目下的所有数据集文件"""
    dd = datasets_dir(project_dir)
    result = []
    for f in sorted(dd.iterdir()):
        if f.is_file() and f.suffix.lower() in ('.txt', '.jsonl', '.json', '.csv', '.tsv', '.parquet'):
            stat = f.stat()
            result.append({
                "name": f.name,
                "path": str(f),
                "size": stat.st_size,
                "size_fmt": _fmt_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "ext": f.suffix.lower(),
            })
    return result


def save_uploaded_text(project_dir: Path, filename: str, content: bytes) -> dict:
    """保存上传的文本文件"""
    dd = datasets_dir(project_dir)
    safe_name = _safe_filename(filename)
    fpath = dd / safe_name

    # 避免覆盖
    if fpath.exists():
        stem = fpath.stem
        ext = fpath.suffix
        i = 1
        while fpath.exists():
            fpath = dd / f"{stem}_{i}{ext}"
            i += 1

    fpath.write_bytes(content)

    # 简单统计
    try:
        text = content.decode("utf-8")
        stats = _quick_text_stats(text)
    except UnicodeDecodeError:
        stats = {"error": "文件编码不是 UTF-8"}

    return {
        "status": "ok",
        "name": fpath.name,
        "path": str(fpath),
        "size": len(content),
        "size_fmt": _fmt_size(len(content)),
        "stats": stats,
    }


def save_pasted_text(project_dir: Path, text: str, name: str = "") -> dict:
    """保存用户粘贴的文本"""
    dd = datasets_dir(project_dir)
    if not name:
        name = f"pasted_{int(time.time())}.txt"
    safe_name = _safe_filename(name)
    if not safe_name.endswith('.txt'):
        safe_name += '.txt'
    fpath = dd / safe_name
    content = text.encode("utf-8")
    fpath.write_bytes(content)

    stats = _quick_text_stats(text)
    return {
        "status": "ok",
        "name": fpath.name,
        "path": str(fpath),
        "size": len(content),
        "size_fmt": _fmt_size(len(content)),
        "stats": stats,
    }


def delete_dataset(project_dir: Path, filename: str) -> bool:
    dd = datasets_dir(project_dir)
    fpath = dd / _safe_filename(filename)
    if fpath.exists() and fpath.is_file():
        fpath.unlink()
        return True
    return False


def get_dataset_preview(project_dir: Path, filename: str, max_lines: int = 30) -> dict:
    """获取数据集预览（前 N 行 + 统计）"""
    dd = datasets_dir(project_dir)
    fpath = dd / _safe_filename(filename)
    if not fpath.exists():
        return {"error": "文件不存在"}

    try:
        text = fpath.read_text("utf-8")
    except UnicodeDecodeError:
        return {"error": "文件编码不是 UTF-8"}

    lines = text.split('\n')
    preview_lines = lines[:max_lines]
    stats = _quick_text_stats(text)

    # Length distribution
    line_lengths = [len(l) for l in lines if l.strip()]
    if line_lengths:
        stats["line_length_distribution"] = {
            "min": min(line_lengths),
            "max": max(line_lengths),
            "avg": round(sum(line_lengths) / len(line_lengths), 1),
            "median": sorted(line_lengths)[len(line_lengths) // 2],
        }

    return {
        "name": fpath.name,
        "preview": '\n'.join(preview_lines),
        "total_lines": len(lines),
        "shown_lines": len(preview_lines),
        "stats": stats,
    }


def get_dataset_stats(project_dir: Path, filename: str) -> dict:
    """获取数据集的详细统计"""
    dd = datasets_dir(project_dir)
    fpath = dd / _safe_filename(filename)
    if not fpath.exists():
        return {"error": "文件不存在"}

    try:
        text = fpath.read_text("utf-8")
    except UnicodeDecodeError:
        return {"error": "文件编码不是 UTF-8"}

    stats = _quick_text_stats(text)

    # 更详细的统计
    lines = [l for l in text.split('\n') if l.strip()]
    line_lengths = [len(l) for l in lines]

    # Character frequency (top 30)
    char_freq = Counter(text)
    top_chars = char_freq.most_common(30)

    # Word frequency (top 30, simple split)
    words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())
    word_freq = Counter(words).most_common(30)

    # Line length histogram (10 bins)
    if line_lengths:
        mn, mx = min(line_lengths), max(line_lengths)
        bins = 10
        bin_width = max(1, (mx - mn + 1) // bins)
        hist = Counter()
        for l in line_lengths:
            bucket = min((l - mn) // bin_width, bins - 1)
            hist[bucket] += 1
        histogram = []
        for i in range(bins):
            lo = mn + i * bin_width
            hi = lo + bin_width - 1
            histogram.append({"range": f"{lo}-{hi}", "count": hist.get(i, 0)})
    else:
        histogram = []

    return {
        "name": fpath.name,
        "size": fpath.stat().st_size,
        "size_fmt": _fmt_size(fpath.stat().st_size),
        "stats": stats,
        "line_count": len(lines),
        "line_lengths": {
            "min": min(line_lengths) if line_lengths else 0,
            "max": max(line_lengths) if line_lengths else 0,
            "avg": round(sum(line_lengths) / len(line_lengths), 1) if line_lengths else 0,
        },
        "histogram": histogram,
        "top_chars": [{"char": c, "count": n} for c, n in top_chars],
        "top_words": [{"word": w, "count": n} for w, n in word_freq],
    }


# ======================= 结构化文件列检测 =======================
def detect_columns(project_dir: Path, filename: str, max_sample: int = 20) -> dict:
    """检测结构化文件（JSONL/JSON/CSV/TSV）的列信息"""
    dd = datasets_dir(project_dir)
    fpath = dd / _safe_filename(filename)
    if not fpath.exists():
        return {"error": "文件不存在"}

    ext = fpath.suffix.lower()
    columns = []
    total_rows = 0
    sample_rows = []

    try:
        text = fpath.read_text("utf-8")
    except UnicodeDecodeError:
        return {"error": "文件编码不是 UTF-8"}

    if ext in ('.jsonl',):
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        total_rows = len(lines)
        for line in lines[:max_sample]:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    sample_rows.append(obj)
            except json.JSONDecodeError:
                continue

    elif ext in ('.json',):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                total_rows = len(data)
                sample_rows = [r for r in data[:max_sample] if isinstance(r, dict)]
            elif isinstance(data, dict):
                total_rows = 1
                sample_rows = [data]
        except json.JSONDecodeError:
            return {"error": "JSON 解析失败"}

    elif ext in ('.csv', '.tsv'):
        import csv
        delimiter = '\t' if ext == '.tsv' else ','
        lines = text.split('\n')
        total_rows = max(0, len(lines) - 1)
        reader = csv.DictReader(lines, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= max_sample:
                break
            sample_rows.append(dict(row))

    else:
        return {"error": "不是结构化文件，无法检测列", "is_plain_text": True}

    if not sample_rows:
        return {"error": "文件为空或无法解析"}

    # Analyze columns from sample rows
    all_keys = {}
    for row in sample_rows:
        for k, v in row.items():
            if k not in all_keys:
                all_keys[k] = {"name": k, "samples": [], "types": Counter(), "non_empty": 0, "avg_len": 0}
            val_str = str(v) if v is not None else ""
            all_keys[k]["samples"].append(val_str[:200])
            all_keys[k]["types"][type(v).__name__] += 1
            if val_str.strip():
                all_keys[k]["non_empty"] += 1

    for k, info in all_keys.items():
        non_empty_samples = [s for s in info["samples"] if s.strip()]
        info["avg_len"] = round(sum(len(s) for s in non_empty_samples) / max(len(non_empty_samples), 1), 1)
        info["fill_rate"] = round(info["non_empty"] / max(len(info["samples"]), 1) * 100, 1)
        info["primary_type"] = info["types"].most_common(1)[0][0] if info["types"] else "str"
        info["samples"] = info["samples"][:5]  # Only keep 5 samples
        info["types"] = dict(info["types"])
        # Auto-detect: is this a good text column for pretraining?
        info["is_text"] = (info["primary_type"] == "str" and info["avg_len"] > 10 and info["fill_rate"] > 50)
        # Is this a conversation column?
        info["is_conversation"] = (info["primary_type"] == "list" or
            (info["primary_type"] == "str" and any(kw in k.lower() for kw in ["conversation", "messages", "dialog"])))

    columns = list(all_keys.values())

    # Auto-suggest best columns for pretraining
    text_cols = [c["name"] for c in columns if c["is_text"]]
    # Prioritize known column names
    priority_names = ["text", "content", "output", "instruction", "input", "question", "answer",
                      "response", "prompt", "completion", "story", "document", "passage", "sentence"]
    suggested = []
    for pn in priority_names:
        if pn in text_cols:
            suggested.append(pn)
    # Add remaining text columns
    for tc in text_cols:
        if tc not in suggested:
            suggested.append(tc)

    return {
        "filename": filename,
        "ext": ext,
        "total_rows": total_rows,
        "columns": columns,
        "suggested_columns": suggested,
        "sample_rows": sample_rows[:3],
        "is_structured": True,
    }


# ======================= 数据预处理（分词打包）=======================
def process_dataset_for_training(
    project_dir: Path,
    dataset_files: list[str],
    tokenizer_dir: str,
    max_seq_len: int = 512,
    output_name: str = "train_data",
    python_path: str | None = None,
) -> dict:
    """将文本数据集分词并打包为训练格式（token ids 的序列）。
    输出 .bin 格式的 token ID 数组。
    """
    import subprocess, sys

    out_dir = project_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_name}.bin"
    meta_path = out_dir / f"{output_name}_meta.json"

    # 构建处理脚本
    script = f'''
import json, sys, os, struct, array

# Load tokenizer
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

# Read and tokenize all files
all_ids = []
files = {json.dumps(dataset_files)}
max_seq = {max_seq_len}

total_chars = 0
total_tokens = 0

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()
    total_chars += len(text)

    # Tokenize in chunks to avoid memory issues
    chunk_size = 100000  # chars
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        ids = encode(chunk)
        all_ids.extend(ids)

total_tokens = len(all_ids)

# Pack into sequences of max_seq_len
# Write as uint16 if vocab_size < 65536, else uint32
dtype = "H" if vocab_size < 65536 else "I"  # H=uint16, I=uint32
bytes_per_token = 2 if dtype == "H" else 4

# Truncate to multiple of max_seq_len
n_seqs = total_tokens // max_seq
usable_tokens = n_seqs * max_seq
all_ids = all_ids[:usable_tokens]

# Write binary
arr = array.array(dtype, all_ids)
out_path = {json.dumps(str(out_path))}
with open(out_path, "wb") as f:
    f.write(arr.tobytes())

# Write metadata
meta = {{
    "total_chars": total_chars,
    "total_tokens": total_tokens,
    "usable_tokens": usable_tokens,
    "n_sequences": n_seqs,
    "max_seq_len": max_seq,
    "vocab_size": vocab_size,
    "dtype": dtype,
    "bytes_per_token": bytes_per_token,
    "file_size": os.path.getsize(out_path),
    "compression_ratio": round(total_chars / total_tokens, 2) if total_tokens > 0 else 0,
    "files_processed": files,
}}

meta_path = {json.dumps(str(meta_path))}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(json.dumps({{"status": "ok", **meta}}))
'''

    py = python_path or sys.executable
    try:
        r = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, timeout=600,
        )
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                    result["output_path"] = str(out_path)
                    result["meta_path"] = str(meta_path)
                    return result
                except json.JSONDecodeError:
                    pass
        err = r.stderr.strip() or r.stdout.strip()
        return {"status": "error", "error": f"处理失败: {err[:500]}"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "处理超时 (>10分钟)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_processed_info(project_dir: Path) -> list:
    """获取已处理的数据集信息"""
    proc_dir = project_dir / "processed"
    if not proc_dir.exists():
        return []
    result = []
    for f in sorted(proc_dir.glob("*_meta.json")):
        try:
            meta = json.loads(f.read_text("utf-8"))
            meta["name"] = f.stem.replace("_meta", "")
            result.append(meta)
        except Exception:
            pass
    return result


# ======================= 工具函数 =======================
def _safe_filename(name: str) -> str:
    """清理文件名，防止路径穿越"""
    name = os.path.basename(name)
    name = re.sub(r'[^\w.\-]', '_', name)
    return name or "file.txt"


def _fmt_size(n: int) -> str:
    if n >= 1e9: return f"{n/1e9:.2f} GB"
    if n >= 1e6: return f"{n/1e6:.1f} MB"
    if n >= 1e3: return f"{n/1e3:.1f} KB"
    return f"{n} B"


def _quick_text_stats(text: str) -> dict:
    """快速文本统计（纯 Python，无外部依赖）"""
    chars = len(text)
    lines = text.count('\n') + (1 if text and not text.endswith('\n') else 0)
    non_empty_lines = sum(1 for l in text.split('\n') if l.strip())
    words = len(text.split())

    # Character type distribution
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    spaces = sum(1 for c in text if c.isspace())
    punctuation = sum(1 for c in text if not c.isalnum() and not c.isspace())

    # Language detection (rough)
    if cjk > ascii_letters * 0.3:
        lang = "zh" if cjk > ascii_letters else "zh-en"
    elif ascii_letters > 0:
        lang = "en"
    else:
        lang = "other"

    return {
        "chars": chars,
        "chars_fmt": _fmt_size(chars) if chars > 10000 else str(chars),
        "lines": lines,
        "non_empty_lines": non_empty_lines,
        "words": words,
        "char_types": {
            "cjk": cjk,
            "ascii_letters": ascii_letters,
            "digits": digits,
            "punctuation": punctuation,
            "spaces": spaces,
        },
        "detected_lang": lang,
        "est_tokens_rough": cjk + len(text.split()),  # 非常粗略的 token 估算
    }
