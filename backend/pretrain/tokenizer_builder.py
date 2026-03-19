"""tokenizer_builder.py — Tokenizer 训练、预览、管理"""
import json, os, subprocess, sys, shutil, re, time
from pathlib import Path
from collections import Counter

# ======================= 预置 Tokenizer 列表 =======================
PRETRAINED_TOKENIZERS = [
    {
        "id": "gpt2",
        "name": "GPT-2",
        "vocab_size": 50257,
        "source": "openai-community/gpt2",
        "description": "OpenAI GPT-2 的 BPE tokenizer，英文为主，支持多语言但中文效率较低",
        "languages": ["en"],
    },
    {
        "id": "llama2",
        "name": "LLaMA 2",
        "vocab_size": 32000,
        "source": "meta-llama/Llama-2-7b-hf",
        "description": "Meta LLaMA 2 的 SentencePiece tokenizer，以英文为主",
        "languages": ["en"],
    },
    {
        "id": "llama3",
        "name": "LLaMA 3",
        "vocab_size": 128256,
        "source": "meta-llama/Meta-Llama-3-8B",
        "description": "Meta LLaMA 3 的 tiktoken tokenizer，大词表，多语言支持改善",
        "languages": ["en", "multi"],
    },
    {
        "id": "qwen2",
        "name": "Qwen 2",
        "vocab_size": 151936,
        "source": "Qwen/Qwen2-1.5B",
        "description": "阿里千问的 tokenizer，中英双语优化，词表很大",
        "languages": ["zh", "en"],
    },
    {
        "id": "mistral",
        "name": "Mistral",
        "vocab_size": 32000,
        "source": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 的 SentencePiece tokenizer，类似 LLaMA",
        "languages": ["en"],
    },
    {
        "id": "bert-base-chinese",
        "name": "BERT 中文",
        "vocab_size": 21128,
        "source": "google-bert/bert-base-chinese",
        "description": "BERT 中文版的 WordPiece tokenizer，逐字分词，词表较小",
        "languages": ["zh"],
    },
]

# ======================= 从项目数据训练新 tokenizer =======================
def train_bpe_tokenizer(
    data_files: list[str],
    output_dir: str,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
    python_path: str | None = None,
) -> dict:
    """使用 HuggingFace tokenizers 库训练 BPE tokenizer。
    返回 {"status": "ok"|"error", ...}
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>", "<pad>", "<unk>", "<s>", "</s>"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 构建训练脚本
    script = f'''
import json, sys
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
except ImportError:
    print(json.dumps({{"status": "error", "error": "请安装 tokenizers 库: pip install tokenizers"}}))
    sys.exit(0)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size={vocab_size},
    min_frequency={min_frequency},
    special_tokens={json.dumps(special_tokens)},
    show_progress=True,
)

files = {json.dumps(data_files)}
tokenizer.train(files, trainer)

# Save
tokenizer.save("{(out / 'tokenizer.json').as_posix()}")

# Stats
vocab = tokenizer.get_vocab()
sample_tokens = sorted(vocab.items(), key=lambda x: x[1])[:50]

# Test encoding
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

print(json.dumps({{
    "status": "ok",
    "vocab_size": tokenizer.get_vocab_size(),
    "sample_tokens": [t[0] for t in sample_tokens],
    "test_results": test_results,
}}))
'''

    py = python_path or sys.executable
    try:
        r = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            err = r.stderr.strip() or r.stdout.strip()
            return {"status": "error", "error": f"训练失败: {err[:500]}"}

        # Parse JSON output
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return {"status": "error", "error": "无法解析训练结果"}

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "训练超时 (>5分钟)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def download_pretrained_tokenizer(
    source: str,
    output_dir: str,
    python_path: str | None = None,
) -> dict:
    """从 HuggingFace 下载预训练 tokenizer"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    script = f'''
import json, sys, os
try:
    from transformers import AutoTokenizer
except ImportError:
    print(json.dumps({{"status": "error", "error": "请安装 transformers 库: pip install transformers"}}))
    sys.exit(0)

try:
    tok = AutoTokenizer.from_pretrained("{source}", trust_remote_code=True)
    tok.save_pretrained("{out.as_posix()}")

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

    print(json.dumps({{
        "status": "ok",
        "vocab_size": tok.vocab_size if hasattr(tok, 'vocab_size') else len(tok),
        "test_results": test_results,
    }}))
except Exception as e:
    print(json.dumps({{"status": "error", "error": str(e)}}))
'''

    py = python_path or sys.executable
    try:
        r = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, timeout=300,
        )
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        err = r.stderr.strip() or r.stdout.strip()
        return {"status": "error", "error": f"下载失败: {err[:500]}"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "下载超时"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def preview_tokenization(
    tokenizer_dir: str,
    text: str,
    python_path: str | None = None,
) -> dict:
    """使用指定的 tokenizer 对文本进行分词预览"""
    script = f'''
import json, sys, os
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
text = {json.dumps(text)}

tok_json = os.path.join({json.dumps(tokenizer_dir)}, "tokenizer.json")
if os.path.exists(tok_json):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_json)
    enc = tok.encode(text)
    ids = enc.ids
    raw_tokens = enc.tokens

    # Byte-level BPE (Qwen/GPT-2) stores tokens as UTF-8 byte sequences
    # that look garbled when displayed directly (e.g. "ä½łå¥½" for "你好").
    # The ONLY reliable way is tok.decode([id]) for each token.
    display_tokens = []
    for i, tid in enumerate(ids):
        try:
            decoded = tok.decode([tid])
            if decoded:
                display_tokens.append(decoded)
            else:
                display_tokens.append(raw_tokens[i] if i < len(raw_tokens) else f"[{{tid}}]")
        except Exception:
            display_tokens.append(raw_tokens[i] if i < len(raw_tokens) else f"[{{tid}}]")

    print(json.dumps({{
        "tokens": display_tokens,
        "ids": ids,
        "length": len(ids),
    }}, ensure_ascii=False))
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained({json.dumps(tokenizer_dir)}, trust_remote_code=True)
    ids = tok.encode(text)
    display_tokens = []
    for tid in ids:
        try:
            decoded = tok.decode([tid])
            display_tokens.append(decoded)
        except Exception:
            raw = tok.convert_ids_to_tokens([tid])
            display_tokens.append(raw[0] if raw else f"[{{tid}}]")

    print(json.dumps({{
        "tokens": display_tokens,
        "ids": ids,
        "length": len(ids),
    }}, ensure_ascii=False))
'''
    py = python_path or sys.executable
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        r = subprocess.run([py, "-c", script], capture_output=True, text=True, timeout=30,
                           env=env, encoding="utf-8", errors="replace")
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        err = r.stderr.strip() or r.stdout.strip()
        return {"error": f"预览失败: {err[:300]}"}
    except Exception as e:
        return {"error": str(e)}


# ======================= 简单字符级统计 (不需要外部库) =======================
def simple_text_stats(text: str) -> dict:
    """不依赖任何外部库的简单文本统计"""
    chars = len(text)
    lines = text.count('\n') + 1
    words_approx = len(text.split())

    # Character type distribution
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    spaces = sum(1 for c in text if c.isspace())
    other = chars - cjk - ascii_letters - digits - spaces

    # Simple "word" approximation for compression ratio
    # A rough heuristic: each CJK char ≈ 1 token, each English word ≈ 1-2 tokens
    est_tokens_basic = cjk + words_approx  # very rough

    return {
        "chars": chars,
        "lines": lines,
        "words_approx": words_approx,
        "char_types": {
            "cjk": cjk,
            "ascii_letters": ascii_letters,
            "digits": digits,
            "spaces": spaces,
            "other": other,
        },
        "est_tokens_rough": est_tokens_basic,
        "primary_lang": "zh" if cjk > ascii_letters else "en" if ascii_letters > 0 else "other",
    }
