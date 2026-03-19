"""trainer.py — 预训练脚本生成、进程管理、监控"""
import json, os, subprocess, sys, signal, time, threading, re
from pathlib import Path
from datetime import datetime

# ======================= Global train state =======================
_train_state = {
    "running": False,
    "pid": None,
    "project_id": None,
    "start_time": None,
    "current_step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "val_loss": 0.0,
    "lr": 0.0,
    "tokens_per_sec": 0.0,
    "samples": [],      # auto-generated text samples
    "log_lines": [],     # last N log lines
    "error": None,
    "checkpoint_dir": None,
}
_train_lock = threading.Lock()
_train_proc = None


def get_train_state() -> dict:
    with _train_lock:
        return dict(_train_state)


def _update_state(**kwargs):
    with _train_lock:
        _train_state.update(kwargs)


# ======================= Generate training script =======================
def generate_train_script(project_dir: Path, config: dict) -> str:
    """生成完整的 PyTorch 预训练脚本"""
    arch = config.get("architecture", {})
    training = config.get("training", {})
    tok_dir = (project_dir / "tokenizer").as_posix()
    data_dir = (project_dir / "processed").as_posix()
    ckpt_dir = (project_dir / "checkpoints").as_posix()
    samples_file = (project_dir / "samples.jsonl").as_posix()

    # Training params with defaults
    lr = training.get("learning_rate", 3e-4)
    bs = training.get("batch_size", 32)
    grad_accum = training.get("grad_accum_steps", 4)
    max_steps = training.get("max_steps", 5000)
    warmup = training.get("warmup_steps", 200)
    wd = training.get("weight_decay", 0.1)
    grad_clip = training.get("grad_clip", 1.0)
    use_fp16 = training.get("fp16", True)
    save_every = training.get("save_every_steps", 500)
    sample_every = training.get("sample_every_steps", 200)
    sample_prompts = training.get("sample_prompts", ["Once upon a time", "The meaning of life is"])
    lr_scheduler = training.get("lr_scheduler", "cosine")
    eval_every = training.get("eval_every_steps", save_every)
    val_split = training.get("val_split", 0.05)

    # DDP
    dist = training.get("distributed", {})
    use_ddp = dist.get("enabled", False)
    gpu_ids = dist.get("gpu_ids", [0])

    # Device selection
    device_cfg = training.get("device", "auto")

    script = f'''#!/usr/bin/env python3
"""Auto-generated pretraining script — Pretrain Lab"""
import os, sys, json, math, time, signal, struct
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ======================= Config =======================
ARCH = {repr(arch)}

TRAIN = {{
    "lr": {lr},
    "batch_size": {bs},
    "grad_accum_steps": {grad_accum},
    "max_steps": {max_steps},
    "warmup_steps": {warmup},
    "weight_decay": {wd},
    "grad_clip": {grad_clip},
    "fp16": {str(use_fp16)},
    "save_every": {save_every},
    "sample_every": {sample_every},
    "lr_scheduler": "{lr_scheduler}",
    "eval_every": {eval_every},
    "val_split": {val_split},
}}

SAMPLE_PROMPTS = {json.dumps(sample_prompts)}
TOK_DIR = "{tok_dir}"
DATA_DIR = "{data_dir}"
CKPT_DIR = "{ckpt_dir}"
SAMPLES_FILE = "{samples_file}"
USE_DDP = {str(use_ddp)}
DEVICE_CFG = "{device_cfg}"

os.makedirs(CKPT_DIR, exist_ok=True)

# ======================= Model Definition =======================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def get_norm(dim, norm_type="rmsnorm", eps=1e-5):
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps)
    return nn.LayerNorm(dim, eps=eps)

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

        # GQA: repeat k,v
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
        dim = cfg["hidden_dim"]
        inter = cfg["intermediate_dim"]
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
        self.norm1 = get_norm(cfg["hidden_dim"], cfg.get("norm_type","rmsnorm"), cfg.get("norm_eps",1e-5))
        self.attn = Attention(cfg)
        self.norm2 = get_norm(cfg["hidden_dim"], cfg.get("norm_type","rmsnorm"), cfg.get("norm_eps",1e-5))
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
        self.norm = get_norm(cfg["hidden_dim"], cfg.get("norm_type","rmsnorm"), cfg.get("norm_eps",1e-5))
        self.lm_head = nn.Linear(cfg["hidden_dim"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", True):
            self.lm_head.weight = self.tok_emb.weight

        # RoPE
        self.rope_cos, self.rope_sin = None, None
        if cfg.get("pos_encoding") == "rope":
            hd = cfg["hidden_dim"] // cfg["num_heads"]
            self.rope_cos, self.rope_sin = precompute_rope(hd, cfg["max_seq_len"], cfg.get("rope_theta", 10000.0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.use_abs_pos:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=100, temperature=0.8, top_k=50):
        for _ in range(max_new):
            ctx = idx[:, -self.cfg["max_seq_len"]:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ======================= Dataset (mmap) =======================
class TokenDataset(Dataset):
    """Memory-mapped dataset — does NOT load entire file into RAM."""
    def __init__(self, data_dir, seq_len, offset=0, length=None):
        self.seq_len = seq_len
        meta_files = list(Path(data_dir).glob("*_meta.json"))
        if not meta_files:
            raise FileNotFoundError(f"No processed data in {{data_dir}}")
        meta = json.loads(meta_files[0].read_text())
        bin_file = meta_files[0].with_name(meta_files[0].stem.replace("_meta", "") + ".bin")
        dtype_str = meta.get("dtype", "H")
        np_dtype = np.uint16 if dtype_str == "H" else np.uint32
        self.data = np.memmap(str(bin_file), dtype=np_dtype, mode='r')
        total_tokens = len(self.data)
        # Apply offset/length for train/val split
        if length is not None:
            end = min(offset + length, total_tokens)
        else:
            end = total_tokens
        self.start = offset
        self.end = end
        self.n_seqs = (end - offset) // seq_len

    def __len__(self):
        return max(0, self.n_seqs - 1)

    def __getitem__(self, idx):
        start = self.start + idx * self.seq_len
        x = torch.from_numpy(self.data[start : start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1 : start + self.seq_len + 1].astype(np.int64))
        return x, y


# ======================= Tokenizer loader =======================
def load_tokenizer(tok_dir):
    tok_json = os.path.join(tok_dir, "tokenizer.json")
    if os.path.exists(tok_json):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_json)
        return tok, lambda t: tok.encode(t).ids, lambda ids: tok.decode(ids)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        return tok, lambda t: tok.encode(t), lambda ids: tok.decode(ids)


# ======================= LR Scheduler =======================
def get_lr(step, warmup, max_steps, max_lr, scheduler="cosine"):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if scheduler == "cosine":
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return max_lr  # constant


# ======================= Eval =======================
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (bx, by) in enumerate(val_loader):
        if i >= max_batches:
            break
        bx, by = bx.to(device), by.to(device)
        _, loss = model(bx, by)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ======================= Training Loop =======================
def train():
    # Device selection
    if DEVICE_CFG == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif DEVICE_CFG == "cpu":
        device = "cpu"
    else:
        device = DEVICE_CFG
        if not torch.cuda.is_available():
            print(json.dumps({{"event": "warning", "msg": "CUDA 不可用，回退到 CPU"}}), flush=True)
            device = "cpu"

    rank = 0
    use_cuda = device.startswith("cuda")

    if USE_DDP and torch.cuda.device_count() > 1:
        import torch.distributed as dist_m
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist_m.init_process_group("nccl")
        rank = dist_m.get_rank()
        device = f"cuda:{{rank}}"
        torch.cuda.set_device(device)
        use_cuda = True

    model = PretrainModel(ARCH).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(json.dumps({{"event": "init", "params": n_params, "device": str(device)}}), flush=True)

    if USE_DDP and torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[rank])

    raw_model = model.module if hasattr(model, "module") else model

    # Dataset — split into train and val
    meta_files = list(Path(DATA_DIR).glob("*_meta.json"))
    if not meta_files:
        print(json.dumps({{"event": "error", "msg": "No processed data found"}}), flush=True)
        return
    meta = json.loads(meta_files[0].read_text())
    bin_file = meta_files[0].with_name(meta_files[0].stem.replace("_meta", "") + ".bin")
    np_dtype = np.uint16 if meta.get("dtype", "H") == "H" else np.uint32
    total_tokens = os.path.getsize(str(bin_file)) // np.dtype(np_dtype).itemsize
    val_tokens = int(total_tokens * TRAIN["val_split"])
    # Round down to seq_len boundary
    val_tokens = (val_tokens // ARCH["max_seq_len"]) * ARCH["max_seq_len"]
    train_tokens = total_tokens - val_tokens

    train_dataset = TokenDataset(DATA_DIR, ARCH["max_seq_len"], offset=0, length=train_tokens)
    val_dataset = None
    val_loader = None
    if val_tokens >= ARCH["max_seq_len"] * 2:
        val_dataset = TokenDataset(DATA_DIR, ARCH["max_seq_len"], offset=train_tokens, length=val_tokens)
        val_loader = DataLoader(val_dataset, batch_size=TRAIN["batch_size"], shuffle=False, pin_memory=use_cuda, num_workers=0)

    if rank == 0:
        print(json.dumps({{
            "event": "data",
            "sequences": len(train_dataset),
            "val_sequences": len(val_dataset) if val_dataset else 0,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
        }}), flush=True)

    sampler = None
    if USE_DDP and torch.cuda.device_count() > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_dataset, shuffle=True)

    loader = DataLoader(train_dataset, batch_size=TRAIN["batch_size"], shuffle=(sampler is None), sampler=sampler, pin_memory=use_cuda, num_workers=2 if use_cuda else 0)

    # Optimizer
    decay_params = [p for n, p in raw_model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in raw_model.named_parameters() if p.dim() < 2]
    optim = torch.optim.AdamW([
        {{"params": decay_params, "weight_decay": TRAIN["weight_decay"]}},
        {{"params": nodecay_params, "weight_decay": 0.0}},
    ], lr=TRAIN["lr"], betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN["fp16"] and use_cuda)

    # Tokenizer for sampling
    _, encode_fn, decode_fn = load_tokenizer(TOK_DIR)

    # Resume from checkpoint?
    start_step = 0
    ckpt_files = sorted(Path(CKPT_DIR).glob("step_*.pt"))
    if ckpt_files:
        latest = ckpt_files[-1]
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        if rank == 0:
            print(json.dumps({{"event": "resume", "step": start_step, "from": str(latest)}}), flush=True)

    # Training
    step = start_step
    t0 = time.time()
    tokens_processed = 0
    running = True
    best_val_loss = float("inf")

    def handle_stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    model.train()
    while step < TRAIN["max_steps"] and running:
        if sampler:
            sampler.set_epoch(step // len(loader))

        for batch_x, batch_y in loader:
            if step >= TRAIN["max_steps"] or not running:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            lr = get_lr(step, TRAIN["warmup_steps"], TRAIN["max_steps"], TRAIN["lr"], TRAIN["lr_scheduler"])
            for pg in optim.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=TRAIN["fp16"] and use_cuda):
                _, loss = model(batch_x, batch_y)
                loss = loss / TRAIN["grad_accum_steps"]

            scaler.scale(loss).backward()
            tokens_processed += batch_x.numel()

            if (step + 1) % TRAIN["grad_accum_steps"] == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            step += 1
            real_loss = loss.item() * TRAIN["grad_accum_steps"]
            elapsed = time.time() - t0
            tok_s = tokens_processed / elapsed if elapsed > 0 else 0

            if rank == 0 and step % 10 == 0:
                print(json.dumps({{
                    "event": "step",
                    "step": step,
                    "total": TRAIN["max_steps"],
                    "loss": round(real_loss, 4),
                    "lr": round(lr, 8),
                    "tok_s": round(tok_s, 1),
                    "elapsed": round(elapsed, 1),
                }}), flush=True)

            # Eval on validation set
            if rank == 0 and val_loader and step % TRAIN["eval_every"] == 0:
                val_loss = evaluate(model, val_loader, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                print(json.dumps({{
                    "event": "eval",
                    "step": step,
                    "val_loss": round(val_loss, 4),
                    "best_val_loss": round(best_val_loss, 4),
                }}), flush=True)

            # Save checkpoint
            if rank == 0 and step % TRAIN["save_every"] == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"step_{{step:06d}}.pt")
                torch.save({{
                    "model": raw_model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "step": step,
                    "loss": real_loss,
                    "arch": ARCH,
                }}, ckpt_path)
                print(json.dumps({{"event": "checkpoint", "step": step, "path": ckpt_path}}), flush=True)

            # Generate samples
            if rank == 0 and step % TRAIN["sample_every"] == 0:
                model.eval()
                samples = []
                for prompt in SAMPLE_PROMPTS:
                    ids = encode_fn(prompt)
                    inp = torch.tensor([ids], device=device)
                    out = raw_model.generate(inp, max_new=80, temperature=0.8, top_k=50)
                    text = decode_fn(out[0].tolist())
                    samples.append({{"prompt": prompt, "text": text}})
                print(json.dumps({{"event": "sample", "step": step, "samples": samples}}), flush=True)
                # Also write to file
                with open(SAMPLES_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({{"step": step, "samples": samples}}, ensure_ascii=False) + "\\n")
                model.train()

    # Final save
    if rank == 0:
        ckpt_path = os.path.join(CKPT_DIR, f"step_{{step:06d}}_final.pt")
        torch.save({{
            "model": raw_model.state_dict(),
            "optimizer": optim.state_dict(),
            "step": step,
            "arch": ARCH,
        }}, ckpt_path)
        elapsed = time.time() - t0
        # Final eval
        final_val_loss = -1.0
        if val_loader:
            final_val_loss = evaluate(raw_model, val_loader, device)
        print(json.dumps({{
            "event": "done",
            "step": step,
            "elapsed": round(elapsed, 1),
            "path": ckpt_path,
            "final_val_loss": round(final_val_loss, 4) if final_val_loss >= 0 else None,
            "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
            "params": n_params,
        }}), flush=True)

    if USE_DDP and torch.cuda.device_count() > 1:
        import torch.distributed as dist_m
        dist_m.destroy_process_group()


if __name__ == "__main__":
    train()
'''
    return script


# ======================= Generate SFT training script =======================
def generate_sft_train_script(project_dir: Path, config: dict) -> str:
    """生成 SFT（监督微调）训练脚本——只在 assistant 回复部分计算 loss"""
    arch = config.get("architecture", {})
    training = config.get("training", {})
    tok_dir = (project_dir / "tokenizer").as_posix()
    data_dir = (project_dir / "processed").as_posix()
    ckpt_dir = (project_dir / "checkpoints_sft").as_posix()
    samples_file = (project_dir / "samples_sft.jsonl").as_posix()

    lr = training.get("learning_rate", 2e-5)
    bs = training.get("batch_size", 8)
    grad_accum = training.get("grad_accum_steps", 4)
    max_steps = training.get("max_steps", 2000)
    warmup = training.get("warmup_steps", 100)
    wd = training.get("weight_decay", 0.01)
    grad_clip = training.get("grad_clip", 1.0)
    use_fp16 = training.get("fp16", True)
    save_every = training.get("save_every_steps", 500)
    sample_every = training.get("sample_every_steps", 200)
    eval_every = training.get("eval_every_steps", save_every)
    val_split = training.get("val_split", 0.05)
    lr_scheduler = training.get("lr_scheduler", "cosine")
    chat_template = training.get("chat_template", "chatml")
    base_checkpoint = training.get("base_checkpoint", "")
    sample_prompts = training.get("sample_prompts", ["你好", "请介绍一下你自己"])
    device_cfg = training.get("device", "auto")

    script = f'''#!/usr/bin/env python3
"""Auto-generated SFT training script — Pretrain Lab"""
import os, sys, json, math, time, signal
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ======================= Config =======================
ARCH = {repr(arch)}
TRAIN = {{
    "lr": {lr}, "batch_size": {bs}, "grad_accum_steps": {grad_accum},
    "max_steps": {max_steps}, "warmup_steps": {warmup},
    "weight_decay": {wd}, "grad_clip": {grad_clip},
    "fp16": {str(use_fp16)}, "save_every": {save_every},
    "sample_every": {sample_every}, "lr_scheduler": "{lr_scheduler}",
    "eval_every": {eval_every}, "val_split": {val_split},
}}
SAMPLE_PROMPTS = {json.dumps(sample_prompts)}
TOK_DIR = "{tok_dir}"
DATA_DIR = "{data_dir}"
CKPT_DIR = "{ckpt_dir}"
SAMPLES_FILE = "{samples_file}"
DEVICE_CFG = "{device_cfg}"
BASE_CKPT = {json.dumps(base_checkpoint)}
CHAT_TEMPLATE = "{chat_template}"

os.makedirs(CKPT_DIR, exist_ok=True)

# ======================= Model Definition =======================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def get_norm(dim, norm_type="rmsnorm", eps=1e-5):
    return RMSNorm(dim, eps) if norm_type == "rmsnorm" else nn.LayerNorm(dim, eps=eps)

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
        return self.dropout(self.o(out.transpose(1, 2).contiguous().view(B, T, -1)))

class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, inter = cfg["hidden_dim"], cfg["intermediate_dim"]
        self.is_gated = cfg.get("activation", "swiglu") == "swiglu"
        if self.is_gated:
            self.gate = nn.Linear(dim, inter, bias=False)
            self.up = nn.Linear(dim, inter, bias=False)
            self.down = nn.Linear(inter, dim, bias=False)
        else:
            self.up = nn.Linear(dim, inter, bias=False)
            self.down = nn.Linear(inter, dim, bias=False)
            acts = {{"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}}
            self.act = acts.get(cfg.get("activation","gelu"), nn.GELU())
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
    def forward(self, x):
        if self.is_gated:
            return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
        return self.dropout(self.down(self.act(self.up(x))))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = get_norm(cfg["hidden_dim"], cfg.get("norm_type","rmsnorm"), cfg.get("norm_eps",1e-5))
        self.attn = Attention(cfg)
        self.norm2 = get_norm(cfg["hidden_dim"], cfg.get("norm_type","rmsnorm"), cfg.get("norm_eps",1e-5))
        self.ffn = FFN(cfg)
    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        return x + self.ffn(self.norm2(x))

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
        self.norm = get_norm(cfg["hidden_dim"], cfg.get("norm_type","rmsnorm"), cfg.get("norm_eps",1e-5))
        self.lm_head = nn.Linear(cfg["hidden_dim"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", True):
            self.lm_head.weight = self.tok_emb.weight
        self.rope_cos, self.rope_sin = None, None
        if cfg.get("pos_encoding") == "rope":
            hd = cfg["hidden_dim"] // cfg["num_heads"]
            self.rope_cos, self.rope_sin = precompute_rope(hd, cfg["max_seq_len"], cfg.get("rope_theta", 10000.0))
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.use_abs_pos:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            if loss_mask is not None:
                # SFT masked loss: only compute on assistant tokens
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                flat_mask = loss_mask.view(-1).float()
                per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
                loss = (per_token_loss * flat_mask).sum() / flat_mask.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new=100, temperature=0.8, top_k=50):
        for _ in range(max_new):
            ctx = idx[:, -self.cfg["max_seq_len"]:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx

# ======================= SFT Dataset (mmap) =======================
class SFTDataset(Dataset):
    def __init__(self, data_dir, seq_len, offset=0, length=None):
        self.seq_len = seq_len
        meta_files = list(Path(data_dir).glob("*_meta.json"))
        if not meta_files:
            raise FileNotFoundError(f"No processed data in {{data_dir}}")
        meta = json.loads(meta_files[0].read_text())
        tokens_file = str(Path(data_dir) / "sft_tokens.bin")
        masks_file = str(Path(data_dir) / "sft_masks.bin")
        np_dtype = np.uint16 if meta.get("dtype","H") == "H" else np.uint32
        self.tokens = np.memmap(tokens_file, dtype=np_dtype, mode='r')
        self.masks = np.memmap(masks_file, dtype=np.uint8, mode='r')
        total = len(self.tokens)
        end = min(offset + length, total) if length else total
        self.start = offset
        self.n_seqs = (end - offset) // seq_len
    def __len__(self):
        return max(0, self.n_seqs - 1)
    def __getitem__(self, idx):
        s = self.start + idx * self.seq_len
        x = torch.from_numpy(self.tokens[s:s+self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.tokens[s+1:s+self.seq_len+1].astype(np.int64))
        m = torch.from_numpy(self.masks[s+1:s+self.seq_len+1].astype(np.int64))
        return x, y, m

# ======================= Tokenizer & LR =======================
def load_tokenizer(tok_dir):
    tok_json = os.path.join(tok_dir, "tokenizer.json")
    if os.path.exists(tok_json):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_json)
        return tok, lambda t: tok.encode(t).ids, lambda ids: tok.decode(ids)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        return tok, lambda t: tok.encode(t), lambda ids: tok.decode(ids)

def get_lr(step, warmup, max_steps, max_lr, scheduler="cosine"):
    if step < warmup: return max_lr * (step + 1) / warmup
    if scheduler == "cosine":
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return max_lr

def format_chat_prompt(prompt, template):
    if template == "chatml":
        return f"<|im_start|>user\\n{{prompt}}<|im_end|>\\n<|im_start|>assistant\\n"
    elif template == "llama":
        return f"[INST] {{prompt}} [/INST]\\n"
    return f"### User:\\n{{prompt}}\\n\\n### Assistant:\\n"

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss, count = 0.0, 0
    for i, (bx, by, bm) in enumerate(val_loader):
        if i >= max_batches: break
        bx, by, bm = bx.to(device), by.to(device), bm.to(device)
        _, loss = model(bx, by, bm)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)

# ======================= SFT Training Loop =======================
def train():
    if DEVICE_CFG == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif DEVICE_CFG == "cpu":
        device = "cpu"
    else:
        device = DEVICE_CFG
        if not torch.cuda.is_available():
            print(json.dumps({{"event": "warning", "msg": "CUDA 不可用，回退到 CPU"}}), flush=True)
            device = "cpu"
    use_cuda = device.startswith("cuda")

    # Read SFT data meta to check for vocab expansion (special token injection)
    sft_meta_files = list(Path(DATA_DIR).glob("sft_data_meta.json"))
    sft_meta = json.loads(sft_meta_files[0].read_text()) if sft_meta_files else {{}}
    sft_vocab_size = sft_meta.get("vocab_size", ARCH["vocab_size"])
    original_vocab_size = sft_meta.get("original_vocab_size", sft_vocab_size)
    added_tokens = sft_meta.get("added_special_tokens", [])
    vocab_expanded = sft_vocab_size != original_vocab_size

    if vocab_expanded:
        print(json.dumps({{"event": "vocab_expand", "original": original_vocab_size, "new": sft_vocab_size, "added": added_tokens}}), flush=True)
        ARCH["vocab_size"] = sft_vocab_size

    model = PretrainModel(ARCH).to(device)

    # Load base checkpoint if specified
    if BASE_CKPT and os.path.exists(BASE_CKPT):
        ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
        base_state = ckpt["model"]
        base_vocab = ckpt.get("arch", {{}}).get("vocab_size", original_vocab_size)

        if vocab_expanded and base_vocab < sft_vocab_size:
            # Embedding layer expanded: copy old weights, randomly init new tokens
            print(json.dumps({{"event": "resize_embeddings", "from": base_vocab, "to": sft_vocab_size, "added_tokens": len(added_tokens)}}), flush=True)

            # tok_emb.weight: [old_vocab, hidden] → [new_vocab, hidden]
            if "tok_emb.weight" in base_state:
                old_emb = base_state["tok_emb.weight"]
                new_emb = model.tok_emb.weight.data.clone()
                new_emb[:old_emb.shape[0]] = old_emb
                # Init new tokens as mean of existing embeddings + small noise
                mean_emb = old_emb.mean(dim=0)
                for i in range(old_emb.shape[0], sft_vocab_size):
                    new_emb[i] = mean_emb + torch.randn_like(mean_emb) * 0.01
                base_state["tok_emb.weight"] = new_emb

            # lm_head.weight (if not tied)
            if "lm_head.weight" in base_state and not ARCH.get("tie_word_embeddings", True):
                old_head = base_state["lm_head.weight"]
                new_head = model.lm_head.weight.data.clone()
                new_head[:old_head.shape[0]] = old_head
                mean_head = old_head.mean(dim=0)
                for i in range(old_head.shape[0], sft_vocab_size):
                    new_head[i] = mean_head + torch.randn_like(mean_head) * 0.01
                base_state["lm_head.weight"] = new_head

        model.load_state_dict(base_state, strict=False)
        print(json.dumps({{"event": "loaded_base", "from": BASE_CKPT, "vocab_expanded": vocab_expanded}}), flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({{"event": "init", "params": n_params, "device": str(device), "mode": "sft", "vocab_size": ARCH["vocab_size"]}}), flush=True)

    # Dataset
    meta_files = list(Path(DATA_DIR).glob("*_meta.json"))
    meta = json.loads(meta_files[0].read_text()) if meta_files else {{}}
    tokens_file = str(Path(DATA_DIR) / "sft_tokens.bin")
    np_dtype = np.uint16 if meta.get("dtype","H") == "H" else np.uint32
    total_tokens = os.path.getsize(tokens_file) // np.dtype(np_dtype).itemsize
    val_tokens = int(total_tokens * TRAIN["val_split"])
    val_tokens = (val_tokens // ARCH["max_seq_len"]) * ARCH["max_seq_len"]
    train_tokens = total_tokens - val_tokens

    train_ds = SFTDataset(DATA_DIR, ARCH["max_seq_len"], offset=0, length=train_tokens)
    val_loader = None
    if val_tokens >= ARCH["max_seq_len"] * 2:
        val_ds = SFTDataset(DATA_DIR, ARCH["max_seq_len"], offset=train_tokens, length=val_tokens)
        val_loader = DataLoader(val_ds, batch_size=TRAIN["batch_size"], shuffle=False, pin_memory=use_cuda)

    print(json.dumps({{"event": "data", "sequences": len(train_ds), "val_sequences": len(val_ds) if val_loader else 0}}), flush=True)

    loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True, pin_memory=use_cuda, num_workers=2 if use_cuda else 0)

    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim = torch.optim.AdamW([
        {{"params": decay_params, "weight_decay": TRAIN["weight_decay"]}},
        {{"params": nodecay_params, "weight_decay": 0.0}},
    ], lr=TRAIN["lr"], betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN["fp16"] and use_cuda)
    _, encode_fn, decode_fn = load_tokenizer(TOK_DIR)

    # Resume
    start_step = 0
    ckpt_files = sorted(Path(CKPT_DIR).glob("step_*.pt"))
    if ckpt_files:
        latest = ckpt_files[-1]
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(json.dumps({{"event": "resume", "step": start_step}}), flush=True)

    step, t0, tokens_processed, running, best_val = start_step, time.time(), 0, True, float("inf")
    signal.signal(signal.SIGTERM, lambda s,f: exec("nonlocal running; running = False", {{"running": running}}))

    model.train()
    while step < TRAIN["max_steps"] and running:
        for batch_x, batch_y, batch_m in loader:
            if step >= TRAIN["max_steps"] or not running: break
            batch_x, batch_y, batch_m = batch_x.to(device), batch_y.to(device), batch_m.to(device)
            lr = get_lr(step, TRAIN["warmup_steps"], TRAIN["max_steps"], TRAIN["lr"], TRAIN["lr_scheduler"])
            for pg in optim.param_groups: pg["lr"] = lr
            with torch.amp.autocast("cuda", enabled=TRAIN["fp16"] and use_cuda):
                _, loss = model(batch_x, batch_y, batch_m)
                loss = loss / TRAIN["grad_accum_steps"]
            scaler.scale(loss).backward()
            tokens_processed += batch_x.numel()
            if (step + 1) % TRAIN["grad_accum_steps"] == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
            step += 1
            real_loss = loss.item() * TRAIN["grad_accum_steps"]
            elapsed = time.time() - t0
            tok_s = tokens_processed / elapsed if elapsed > 0 else 0
            if step % 10 == 0:
                print(json.dumps({{"event": "step", "step": step, "total": TRAIN["max_steps"], "loss": round(real_loss, 4), "lr": round(lr, 8), "tok_s": round(tok_s, 1), "elapsed": round(elapsed, 1)}}), flush=True)
            if val_loader and step % TRAIN["eval_every"] == 0:
                vl = evaluate(model, val_loader, device)
                if vl < best_val: best_val = vl
                print(json.dumps({{"event": "eval", "step": step, "val_loss": round(vl, 4), "best_val_loss": round(best_val, 4)}}), flush=True)
            if step % TRAIN["save_every"] == 0:
                torch.save({{"model": model.state_dict(), "optimizer": optim.state_dict(), "step": step, "loss": real_loss, "arch": ARCH}}, os.path.join(CKPT_DIR, f"step_{{step:06d}}.pt"))
                print(json.dumps({{"event": "checkpoint", "step": step}}), flush=True)
            if step % TRAIN["sample_every"] == 0:
                model.eval()
                samples = []
                for prompt in SAMPLE_PROMPTS:
                    chat_prompt = format_chat_prompt(prompt, CHAT_TEMPLATE)
                    ids = encode_fn(chat_prompt)
                    inp = torch.tensor([ids], device=device)
                    out = model.generate(inp, max_new=80, temperature=0.8, top_k=50)
                    text = decode_fn(out[0].tolist())
                    # Extract only assistant part
                    reply = text[len(chat_prompt):]
                    samples.append({{"prompt": prompt, "text": reply}})
                print(json.dumps({{"event": "sample", "step": step, "samples": samples}}), flush=True)
                with open(SAMPLES_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({{"step": step, "samples": samples}}, ensure_ascii=False) + "\\n")
                model.train()

    # Final
    ckpt_path = os.path.join(CKPT_DIR, f"step_{{step:06d}}_final.pt")
    torch.save({{"model": model.state_dict(), "optimizer": optim.state_dict(), "step": step, "arch": ARCH}}, ckpt_path)
    elapsed = time.time() - t0
    final_vl = evaluate(model, val_loader, device) if val_loader else -1
    print(json.dumps({{
        "event": "done", "step": step, "elapsed": round(elapsed, 1), "path": ckpt_path,
        "final_val_loss": round(final_vl, 4) if final_vl >= 0 else None,
        "best_val_loss": round(best_val, 4) if best_val < float("inf") else None,
        "params": n_params,
    }}), flush=True)

if __name__ == "__main__":
    train()
'''
    return script


# ======================= Start Training =======================
def start_training(project_dir: Path, config: dict, python_path: str | None = None) -> dict:
    global _train_proc

    if _train_state["running"]:
        return {"status": "error", "error": "已有训练在进行中"}

    # Detect SFT mode: check if processed data has SFT metadata
    train_mode = config.get("training", {}).get("train_mode", "pretrain")
    sft_meta = project_dir / "processed" / "sft_data_meta.json"
    if train_mode == "sft" or sft_meta.exists():
        script = generate_sft_train_script(project_dir, config)
    else:
        script = generate_train_script(project_dir, config)

    script_path = project_dir / "train_script.py"
    script_path.write_text(script, "utf-8")

    py = python_path or sys.executable
    dist = config.get("training", {}).get("distributed", {})
    use_ddp = dist.get("enabled", False)
    gpu_ids = dist.get("gpu_ids", [0])

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        device_cfg = config.get("training", {}).get("device", "auto")
        if use_ddp and gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        elif device_cfg.startswith("cuda:"):
            # Single GPU selection — map to CUDA_VISIBLE_DEVICES
            gpu_idx = device_cfg.split(":")[1]
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx
        elif gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        if use_ddp and len(gpu_ids) > 1:
            cmd = [py, "-m", "torch.distributed.run",
                   f"--nproc_per_node={len(gpu_ids)}",
                   str(script_path)]
        else:
            cmd = [py, str(script_path)]

        _train_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, cwd=str(project_dir),
            encoding="utf-8", errors="replace",
        )

        _update_state(
            running=True,
            pid=_train_proc.pid,
            project_id=config.get("id", ""),
            start_time=datetime.now().isoformat(),
            current_step=0,
            total_steps=config.get("training", {}).get("max_steps", 5000),
            loss=0.0, lr=0.0, tokens_per_sec=0.0,
            samples=[], log_lines=[], error=None,
            checkpoint_dir=str(project_dir / "checkpoints"),
        )

        # Monitor thread
        t = threading.Thread(target=_monitor_training, daemon=True)
        t.start()

        return {"status": "ok", "pid": _train_proc.pid}

    except Exception as e:
        _update_state(running=False, error=str(e))
        return {"status": "error", "error": str(e)}


def _monitor_training():
    """读取训练进程的 stdout，解析 JSON 事件"""
    global _train_proc
    if not _train_proc:
        return

    try:
        for line in _train_proc.stdout:
            line = line.strip()
            if not line:
                continue

            # Append to log
            with _train_lock:
                _train_state["log_lines"].append(line)
                if len(_train_state["log_lines"]) > 500:
                    _train_state["log_lines"] = _train_state["log_lines"][-300:]

            # Try parse JSON
            if line.startswith("{"):
                try:
                    ev = json.loads(line)
                    event = ev.get("event", "")

                    if event == "step":
                        _update_state(
                            current_step=ev.get("step", 0),
                            total_steps=ev.get("total", 0),
                            loss=ev.get("loss", 0),
                            lr=ev.get("lr", 0),
                            tokens_per_sec=ev.get("tok_s", 0),
                        )
                    elif event == "eval":
                        _update_state(
                            val_loss=ev.get("val_loss", 0),
                            best_val_loss=ev.get("best_val_loss", 0),
                        )
                    elif event == "sample":
                        with _train_lock:
                            _train_state["samples"].append({
                                "step": ev.get("step", 0),
                                "samples": ev.get("samples", []),
                            })
                            # Keep last 20 sample events
                            if len(_train_state["samples"]) > 20:
                                _train_state["samples"] = _train_state["samples"][-20:]
                    elif event == "checkpoint":
                        pass  # logged
                    elif event == "done":
                        _update_state(running=False, error=None)
                        # Save training record
                        _save_training_record(ev)
                    elif event == "init":
                        _update_state(
                            device=ev.get("device", "unknown"),
                            total_params=ev.get("params", 0),
                        )
                    elif event == "warning":
                        with _train_lock:
                            _train_state["log_lines"].append(f"⚠️ {ev.get('msg', '')}")
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    finally:
        if _train_proc:
            _train_proc.wait()
        _update_state(running=False)
        _train_proc = None


def stop_training() -> dict:
    global _train_proc
    if not _train_state["running"] or not _train_proc:
        return {"status": "not_running"}
    try:
        _train_proc.terminate()
        _train_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _train_proc.kill()
    except Exception:
        pass
    _update_state(running=False)
    _train_proc = None
    return {"status": "stopped"}


def list_checkpoints(project_dir: Path, mode: str = "pretrain") -> list:
    subdir = "checkpoints_sft" if mode == "sft" else "checkpoints"
    ckpt_dir = project_dir / subdir
    if not ckpt_dir.exists():
        return []
    result = []
    for f in sorted(ckpt_dir.glob("step_*.pt")):
        stat = f.stat()
        step_match = re.search(r'step_(\d+)', f.stem)
        step = int(step_match.group(1)) if step_match else 0
        result.append({
            "name": f.name,
            "path": str(f),
            "step": step,
            "size": stat.st_size,
            "size_fmt": f"{stat.st_size/1e6:.1f} MB",
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_final": "final" in f.name,
            "mode": mode,
        })
    return result


def load_samples(project_dir: Path, mode: str = "pretrain") -> list:
    """Load auto-generated samples from file"""
    fname = "samples_sft.jsonl" if mode == "sft" else "samples.jsonl"
    f = project_dir / fname
    if not f.exists():
        return []
    result = []
    try:
        for line in f.read_text("utf-8").strip().split("\n"):
            if line.strip():
                result.append(json.loads(line))
    except Exception:
        pass
    return result


def _save_training_record(done_event: dict):
    """Save a training record to the project's history"""
    pid = _train_state.get("project_id", "")
    if not pid:
        return
    try:
        # Find project dir
        from backend.app import _get_ft_dir
        projects_root = _get_ft_dir() / "pretrain-lab"
        safe = "".join(c for c in pid if c.isalnum() or c in "-_")
        pdir = projects_root / safe
        meta_f = pdir / "project.json"
        if not meta_f.exists():
            return
        meta = json.loads(meta_f.read_text("utf-8"))

        # Build record
        record = {
            "id": f"run-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "steps": done_event.get("step", 0),
            "elapsed_sec": done_event.get("elapsed", 0),
            "final_loss": _train_state.get("loss", 0),
            "final_val_loss": done_event.get("final_val_loss"),
            "best_val_loss": done_event.get("best_val_loss"),
            "params": done_event.get("params", 0),
            "checkpoint_path": done_event.get("path", ""),
            "architecture": meta.get("architecture", {}),
            "training_config": meta.get("training", {}),
            "device": _train_state.get("device", "unknown"),
            "datasets": [f.name for f in (pdir / "datasets").iterdir() if f.is_file()] if (pdir / "datasets").exists() else [],
        }

        # Append to history
        if "train_history" not in meta:
            meta["train_history"] = []
        meta["train_history"].append(record)
        meta["status"] = "done"
        meta["updated"] = datetime.now().isoformat()
        meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    except Exception as e:
        print(f"[trainer] save_training_record error: {e}", flush=True)


def load_training_records(project_dir: Path) -> list:
    """Load training history from project metadata"""
    meta_f = project_dir / "project.json"
    if not meta_f.exists():
        return []
    try:
        meta = json.loads(meta_f.read_text("utf-8"))
        return meta.get("train_history", [])
    except Exception:
        return []
