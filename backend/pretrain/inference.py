"""inference.py — 从 checkpoint 加载模型并生成文本（支持流式）"""
import json, os, subprocess, sys
from pathlib import Path
from typing import Generator


def _build_model_script(tok_dir: str) -> str:
    """Build the shared model definition part used by both sync and streaming inference."""
    return f'''
import json, sys, os, torch, time
import torch.nn as nn
import torch.nn.functional as F

# ========== Model (exact copy from trainer.py) ==========

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def get_norm(dim, nt="rmsnorm", eps=1e-5):
    return RMSNorm(dim, eps) if nt == "rmsnorm" else nn.LayerNorm(dim, eps=eps)

def precompute_rope(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos[:x.shape[-2], :d//2].to(x.device)
    sin = sin[:x.shape[-2], :d//2].to(x.device)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["num_heads"]
        self.n_kv = cfg.get("num_kv_heads", self.n_heads)
        self.head_dim = cfg["hidden_dim"] // self.n_heads
        self.q = nn.Linear(cfg["hidden_dim"], self.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(cfg["hidden_dim"], self.n_kv * self.head_dim, bias=False)
        self.v = nn.Linear(cfg["hidden_dim"], self.n_kv * self.head_dim, bias=False)
        self.o = nn.Linear(self.n_heads * self.head_dim, cfg["hidden_dim"], bias=False)
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        if rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        if self.n_kv < self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.dropout(self.o(out))

class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, inter = cfg["hidden_dim"], cfg["intermediate_dim"]
        act = cfg.get("activation", "swiglu")
        self.is_gated = act == "swiglu"
        if self.is_gated:
            self.gate = nn.Linear(dim, inter, bias=False)
            self.up = nn.Linear(dim, inter, bias=False)
            self.down = nn.Linear(inter, dim, bias=False)
        else:
            self.up = nn.Linear(dim, inter, bias=False)
            self.down = nn.Linear(inter, dim, bias=False)
            acts = {{"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}}
            self.act = acts.get(act, nn.GELU())
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
    def forward(self, x):
        if self.is_gated:
            return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
        return self.dropout(self.down(self.act(self.up(x))))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = get_norm(cfg["hidden_dim"], cfg.get("norm_type", "rmsnorm"), cfg.get("norm_eps", 1e-5))
        self.attn = Attention(cfg)
        self.norm2 = get_norm(cfg["hidden_dim"], cfg.get("norm_type", "rmsnorm"), cfg.get("norm_eps", 1e-5))
        self.ffn = FFN(cfg)
    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x

class PretrainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_dim"])
        self.use_abs_pos = cfg.get("pos_encoding") == "absolute"
        if self.use_abs_pos:
            self.pos_emb = nn.Embedding(cfg["max_seq_len"], cfg["hidden_dim"])
        self.drop = nn.Dropout(cfg.get("dropout", 0.0))
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.norm = get_norm(cfg["hidden_dim"], cfg.get("norm_type", "rmsnorm"), cfg.get("norm_eps", 1e-5))
        self.lm_head = nn.Linear(cfg["hidden_dim"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", True):
            self.lm_head.weight = self.tok_emb.weight
        self.rope_cos, self.rope_sin = None, None
        pe = cfg.get("pos_encoding", "rope")
        if pe == "rope":
            hd = cfg["hidden_dim"] // cfg["num_heads"]
            self.rope_cos, self.rope_sin = precompute_rope(hd, cfg["max_seq_len"], cfg.get("rope_theta", 10000.0))

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.use_abs_pos:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new=100, temp=0.8, top_k=50):
        for _ in range(max_new):
            ctx = idx[:, -self.cfg["max_seq_len"]:]
            logits = self(ctx)[:, -1, :] / max(temp, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx

# ========== Load tokenizer ==========
tok_dir = {json.dumps(tok_dir)}
tok_json = os.path.join(tok_dir, "tokenizer.json")
if os.path.exists(tok_json):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_json)
    encode = lambda t: tok.encode(t).ids
    decode = lambda ids: tok.decode(ids)
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    encode = lambda t: tok.encode(t)
    decode = lambda ids: tok.decode(ids)
'''


def generate_text(
    project_dir: str,
    checkpoint_path: str,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    python_path: str | None = None,
) -> dict:
    """加载 checkpoint 并生成文本 (同步)"""
    tok_dir = str(Path(project_dir) / "tokenizer")
    model_code = _build_model_script(tok_dir)

    script = model_code + f'''

ckpt = torch.load({json.dumps(checkpoint_path)}, map_location="cpu", weights_only=False)
arch = ckpt["arch"]
model = PretrainModel(arch)
model.load_state_dict(ckpt["model"])
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

prompts = {json.dumps(prompts)}
results = []
for prompt in prompts:
    ids = encode(prompt)
    inp = torch.tensor([ids], device=device)
    t0 = time.time()
    out = model.generate(inp, max_new={max_new_tokens}, temp={temperature}, top_k={top_k})
    elapsed = time.time() - t0
    generated_ids = out[0].tolist()
    text = decode(generated_ids)
    new_tokens = len(generated_ids) - len(ids)
    results.append({{
        "prompt": prompt,
        "text": text,
        "new_tokens": new_tokens,
        "elapsed": round(elapsed, 2),
        "tok_s": round(new_tokens / elapsed, 1) if elapsed > 0 else 0,
    }})

print(json.dumps({{"status": "ok", "results": results, "step": ckpt.get("step", 0)}}))
'''

    py = python_path or sys.executable
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        r = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, timeout=120,
            env=env, encoding="utf-8", errors="replace",
        )
        for line in r.stdout.strip().split("\n"):
            if line.strip().startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        err = r.stderr.strip() or r.stdout.strip()
        return {"status": "error", "error": f"生成失败: {err[:500]}"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "生成超时 (>2分钟)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def generate_text_streaming(
    project_dir: str,
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    python_path: str | None = None,
    stop_tokens: list[str] | None = None,
) -> Generator[str, None, None]:
    """加载 checkpoint 并逐 token 流式生成文本 (SSE)"""
    tok_dir = str(Path(project_dir) / "tokenizer")
    model_code = _build_model_script(tok_dir)

    stop_tokens_json = json.dumps(stop_tokens or [])
    script = model_code + f'''

ckpt = torch.load({json.dumps(checkpoint_path)}, map_location="cpu", weights_only=False)
arch = ckpt["arch"]
model = PretrainModel(arch)
model.load_state_dict(ckpt["model"])
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

prompt = {json.dumps(prompt)}
stop_tokens = {stop_tokens_json}
ids = encode(prompt)
inp = torch.tensor([ids], device=device)
t0 = time.time()
gen_count = 0
generated_text = ""

# Stream token by token
with torch.no_grad():
    for i in range({max_new_tokens}):
        ctx = inp[:, -arch["max_seq_len"]:]
        logits = model(ctx)[:, -1, :] / max({temperature}, 1e-8)
        if {top_k} > 0:
            v, _ = torch.topk(logits, min({top_k}, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        inp = torch.cat([inp, nxt], 1)
        new_id = nxt[0, 0].item()
        full_text = decode(inp[0].tolist())
        prev_text = decode(inp[0, :-1].tolist())
        token_text = full_text[len(prev_text):]
        generated_text += token_text
        gen_count += 1
        elapsed = time.time() - t0
        tok_s = gen_count / elapsed if elapsed > 0 else 0
        # Check stop tokens
        should_stop = False
        for st in stop_tokens:
            if st in generated_text:
                # Trim the stop token from output
                idx = generated_text.find(st)
                if idx >= 0:
                    token_text = ""  # don't output the stop token part
                should_stop = True
                break
        if token_text:
            print(json.dumps({{
                "event": "token",
                "text": token_text,
                "token_id": new_id,
                "n": gen_count,
                "tok_s": round(tok_s, 1),
            }}, ensure_ascii=False), flush=True)
        if should_stop:
            break

elapsed = time.time() - t0
print(json.dumps({{
    "event": "done",
    "total_tokens": gen_count,
    "elapsed": round(elapsed, 2),
    "tok_s": round(gen_count / elapsed, 1) if elapsed > 0 else 0,
    "step": ckpt.get("step", 0),
}}), flush=True)
'''

    py = python_path or sys.executable
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        proc = subprocess.Popen(
            [py, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, bufsize=1,
            env=env, encoding="utf-8", errors="replace",
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                yield f"data: {line}\n\n"
            else:
                yield f'data: {json.dumps({"event": "log", "text": line})}\n\n'

        proc.wait(timeout=30)
        if proc.returncode != 0:
            err = proc.stderr.read().strip() if proc.stderr else ""
            yield f'data: {json.dumps({"event": "error", "error": err[:300]})}\n\n'

    except subprocess.TimeoutExpired:
        proc.kill()
        yield f'data: {json.dumps({"event": "error", "error": "生成超时"})}\n\n'
    except Exception as e:
        yield f'data: {json.dumps({"event": "error", "error": str(e)})}\n\n'
