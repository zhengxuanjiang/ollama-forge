"""architect.py — 模型架构定义、模板、参数计算"""
import math, copy, json
from pathlib import Path

# ======================= 组件选项定义 =======================
COMPONENT_OPTIONS = {
    "norm_type":     ["layernorm", "rmsnorm"],
    "pos_encoding":  ["rope", "alibi", "absolute", "none"],
    "activation":    ["swiglu", "gelu", "relu", "silu"],
    "attention_type": ["mha", "gqa", "mqa"],
}

COMPONENT_LABELS = {
    "norm_type":     {"layernorm": "LayerNorm", "rmsnorm": "RMSNorm"},
    "pos_encoding":  {"rope": "RoPE", "alibi": "ALiBi", "absolute": "绝对位置编码", "none": "无 (需自行处理)"},
    "activation":    {"swiglu": "SwiGLU", "gelu": "GELU", "relu": "ReLU", "silu": "SiLU"},
    "attention_type": {"mha": "Multi-Head (MHA)", "gqa": "Grouped-Query (GQA)", "mqa": "Multi-Query (MQA)"},
}

COMPONENT_DESCRIPTIONS = {
    "norm_type": {
        "layernorm": "经典归一化，GPT-2 使用。学习 γ 和 β 两组参数",
        "rmsnorm":   "LLaMA/Mistral 使用。去掉均值中心化，只保留缩放，更快更稳定",
    },
    "pos_encoding": {
        "rope":     "旋转位置编码，LLaMA/Mistral 使用。通过旋转变换编码相对位置，支持长度外推",
        "alibi":    "线性偏置，BLOOM 使用。在注意力分数上加线性偏置，天然支持长度外推",
        "absolute": "绝对位置编码，GPT-2 使用。可学习的位置向量，简单直接但不支持外推",
        "none":     "不使用位置编码，需要通过其他方式引入位置信息",
    },
    "activation": {
        "swiglu": "门控激活函数，LLaMA/Mistral 使用。FFN 会有 3 个权重矩阵而非 2 个，效果更好",
        "gelu":   "高斯误差线性单元，GPT-2/BERT 使用。平滑的近似 ReLU",
        "relu":   "最简单的激活函数。负值直接置零，计算快但可能有 dead neuron 问题",
        "silu":   "Sigmoid 加权线性单元。平滑版 ReLU，PaLM 等使用",
    },
    "attention_type": {
        "mha": "标准多头注意力。每个头有独立的 Q/K/V，参数最多但表达力最强",
        "gqa": "分组查询注意力，LLaMA2/Mistral 使用。多个 Q 头共享 K/V 头，节省显存和加速推理",
        "mqa": "多查询注意力，Falcon 使用。所有 Q 头共享同一组 K/V，最省显存但可能损失精度",
    },
}

# ======================= 预设模板 =======================
TEMPLATES = {
    "gpt2-small": {
        "id": "gpt2-small",
        "name": "GPT-2 Small",
        "description": "经典入门架构，OpenAI GPT-2 的最小版本。使用绝对位置编码和标准 LayerNorm。",
        "family": "GPT-2",
        "architecture": {
            "vocab_size": 50257,
            "max_seq_len": 1024,
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "num_kv_heads": 12,
            "intermediate_dim": 3072,
            "norm_type": "layernorm",
            "norm_eps": 1e-5,
            "pos_encoding": "absolute",
            "rope_theta": 10000,
            "activation": "gelu",
            "attention_type": "mha",
            "tie_word_embeddings": True,
            "dropout": 0.1,
        }
    },
    "gpt2-medium": {
        "id": "gpt2-medium",
        "name": "GPT-2 Medium",
        "description": "GPT-2 中等规模版本，适合有 8GB+ 显存的用户。",
        "family": "GPT-2",
        "architecture": {
            "vocab_size": 50257,
            "max_seq_len": 1024,
            "hidden_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "num_kv_heads": 16,
            "intermediate_dim": 4096,
            "norm_type": "layernorm",
            "norm_eps": 1e-5,
            "pos_encoding": "absolute",
            "rope_theta": 10000,
            "activation": "gelu",
            "attention_type": "mha",
            "tie_word_embeddings": True,
            "dropout": 0.1,
        }
    },
    "llama-tiny": {
        "id": "llama-tiny",
        "name": "LLaMA Tiny",
        "description": "LLaMA 风格的极小模型，~25M 参数。适合快速实验和学习现代架构设计。",
        "family": "LLaMA",
        "architecture": {
            "vocab_size": 32000,
            "max_seq_len": 512,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "num_kv_heads": 4,
            "intermediate_dim": 1376,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "pos_encoding": "rope",
            "rope_theta": 10000.0,
            "activation": "swiglu",
            "attention_type": "gqa",
            "tie_word_embeddings": True,
            "dropout": 0.0,
        }
    },
    "llama-small": {
        "id": "llama-small",
        "name": "LLaMA Small",
        "description": "LLaMA 风格的小型模型，~110M 参数。平衡训练成本与模型能力。",
        "family": "LLaMA",
        "architecture": {
            "vocab_size": 32000,
            "max_seq_len": 1024,
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "num_kv_heads": 4,
            "intermediate_dim": 2048,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "pos_encoding": "rope",
            "rope_theta": 10000.0,
            "activation": "swiglu",
            "attention_type": "gqa",
            "tie_word_embeddings": True,
            "dropout": 0.0,
        }
    },
    "mistral-tiny": {
        "id": "mistral-tiny",
        "name": "Mistral Tiny",
        "description": "Mistral 风格的小型模型。使用滑动窗口注意力（需训练脚本支持）。",
        "family": "Mistral",
        "architecture": {
            "vocab_size": 32000,
            "max_seq_len": 1024,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "num_kv_heads": 2,
            "intermediate_dim": 1376,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "pos_encoding": "rope",
            "rope_theta": 10000.0,
            "activation": "swiglu",
            "attention_type": "gqa",
            "tie_word_embeddings": True,
            "dropout": 0.0,
        }
    },
    "minimal": {
        "id": "minimal",
        "name": "极简入门",
        "description": "最小的可训练模型，~1M 参数。几分钟内就能看到训练效果，非常适合理解训练流程。",
        "family": "自定义",
        "architecture": {
            "vocab_size": 8000,
            "max_seq_len": 256,
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "num_kv_heads": 4,
            "intermediate_dim": 512,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "pos_encoding": "rope",
            "rope_theta": 10000.0,
            "activation": "swiglu",
            "attention_type": "mha",
            "tie_word_embeddings": True,
            "dropout": 0.0,
        }
    },
    "phi-tiny": {
        "id": "phi-tiny",
        "name": "Phi Tiny",
        "description": "Phi 风格的小型模型。使用并行 Attention+FFN 和 RoPE partial rotation。",
        "family": "Phi",
        "architecture": {
            "vocab_size": 32000,
            "max_seq_len": 2048,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "num_kv_heads": 4,
            "intermediate_dim": 1376,
            "norm_type": "layernorm",
            "norm_eps": 1e-5,
            "pos_encoding": "rope",
            "rope_theta": 10000.0,
            "activation": "gelu",
            "attention_type": "gqa",
            "tie_word_embeddings": True,
            "dropout": 0.0,
        }
    },
    "gemma-tiny": {
        "id": "gemma-tiny",
        "name": "Gemma Tiny",
        "description": "Google Gemma 风格的小型模型。使用 GeGLU 激活和 RMSNorm。",
        "family": "Gemma",
        "architecture": {
            "vocab_size": 32000,
            "max_seq_len": 1024,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "num_kv_heads": 1,
            "intermediate_dim": 2048,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-6,
            "pos_encoding": "rope",
            "rope_theta": 10000.0,
            "activation": "gelu",
            "attention_type": "mqa",
            "tie_word_embeddings": True,
            "dropout": 0.0,
        }
    },
    "moe-tiny": {
        "id": "moe-tiny",
        "name": "MoE Tiny",
        "description": "Mixture of Experts 风格的小型模型。8 个专家，top-2 路由。类似 Mixtral/DeepSeek-MoE。",
        "family": "MoE",
        "architecture": {
            "vocab_size": 32000,
            "max_seq_len": 512,
            "hidden_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "num_kv_heads": 2,
            "intermediate_dim": 1024,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "pos_encoding": "rope",
            "rope_theta": 10000.0,
            "activation": "swiglu",
            "attention_type": "gqa",
            "tie_word_embeddings": True,
            "dropout": 0.0,
            "moe_num_experts": 8,
            "moe_top_k": 2,
        }
    },
}

# ======================= 参数计算 =======================
def calc_params(arch: dict) -> dict:
    """计算模型参数量和显存估算"""
    V = arch.get("vocab_size", 32000)
    D = arch.get("hidden_dim", 512)
    L = arch.get("num_layers", 8)
    H = arch.get("num_heads", 8)
    Hkv = arch.get("num_kv_heads", H)
    I = arch.get("intermediate_dim", D * 4)
    S = arch.get("max_seq_len", 512)
    act = arch.get("activation", "gelu")
    norm = arch.get("norm_type", "layernorm")
    pos = arch.get("pos_encoding", "rope")
    tie = arch.get("tie_word_embeddings", True)
    moe_experts = arch.get("moe_num_experts", 0)
    moe_topk = arch.get("moe_top_k", 2)

    head_dim = D // H if H > 0 else D

    # === Embedding ===
    emb_params = V * D  # token embedding
    if pos == "absolute":
        emb_params += S * D  # position embedding

    # === Each Transformer Layer ===
    # Attention: Q, K, V projections + output projection
    q_params = D * (H * head_dim)       # W_q
    k_params = D * (Hkv * head_dim)     # W_k
    v_params = D * (Hkv * head_dim)     # W_v
    o_params = (H * head_dim) * D       # W_o
    attn_params = q_params + k_params + v_params + o_params

    # FFN
    if act == "swiglu":
        ffn_params = D * I + D * I + I * D  # gate, up, down (3 matrices)
    else:
        ffn_params = D * I + I * D  # up, down (2 matrices)

    # MoE: multiply FFN by num_experts, add router
    moe_router_params = 0
    if moe_experts > 0:
        moe_ffn_params = ffn_params * moe_experts
        moe_router_params = D * moe_experts  # router linear
        ffn_params = moe_ffn_params + moe_router_params

    # Norm (2 per layer: pre-attn and pre-ffn)
    if norm == "rmsnorm":
        norm_params = D  # just scale
    else:
        norm_params = D * 2  # scale + bias
    layer_norm_params = norm_params * 2

    layer_params = attn_params + ffn_params + layer_norm_params
    all_layer_params = layer_params * L

    # === LM Head ===
    # Final norm
    final_norm = norm_params
    # LM head (output projection)
    lm_head_params = 0 if tie else V * D

    total = emb_params + all_layer_params + final_norm + lm_head_params

    # === VRAM 估算 ===
    bytes_per_param_fp32 = 4
    bytes_per_param_fp16 = 2

    # 训练: params(fp16) + gradients(fp16) + optimizer_states(fp32, ×2 for Adam) ≈ param × (2+2+8) = 12 bytes
    vram_train_fp16 = total * 12
    # 再加上 activations 的粗略估算
    activation_mem = L * S * D * 2 * 4  # rough estimate per batch item
    vram_train_total = vram_train_fp16 + activation_mem

    # 推理: params only, fp16
    vram_infer_fp16 = total * bytes_per_param_fp16
    # 加上 KV cache
    kv_cache = 2 * L * S * Hkv * head_dim * bytes_per_param_fp16
    vram_infer_total = vram_infer_fp16 + kv_cache

    def fmt_params(n):
        if n >= 1e9: return f"{n/1e9:.1f}B"
        if n >= 1e6: return f"{n/1e6:.1f}M"
        if n >= 1e3: return f"{n/1e3:.1f}K"
        return str(n)

    def fmt_bytes(n):
        if n >= 1e9: return f"{n/1e9:.2f} GB"
        if n >= 1e6: return f"{n/1e6:.1f} MB"
        return f"{n/1e3:.1f} KB"

    return {
        "total_params": total,
        "total_params_fmt": fmt_params(total),
        "breakdown": {
            "embedding": emb_params,
            "embedding_fmt": fmt_params(emb_params),
            "per_layer": layer_params,
            "per_layer_fmt": fmt_params(layer_params),
            "per_layer_attn": attn_params,
            "per_layer_ffn": ffn_params,
            "per_layer_norm": layer_norm_params,
            "all_layers": all_layer_params,
            "all_layers_fmt": fmt_params(all_layer_params),
            "final_norm": final_norm,
            "lm_head": lm_head_params,
            "lm_head_fmt": fmt_params(lm_head_params) if lm_head_params else "共享 (0)",
            "moe_router": moe_router_params * L if moe_experts > 0 else 0,
            "moe_info": f"{moe_experts} experts × top-{moe_topk}" if moe_experts > 0 else "",
        },
        "vram": {
            "train_fp16": vram_train_total,
            "train_fp16_fmt": fmt_bytes(vram_train_total),
            "inference_fp16": vram_infer_total,
            "inference_fp16_fmt": fmt_bytes(vram_infer_total),
        },
        "info": {
            "head_dim": head_dim,
            "params_per_layer_pct": f"{layer_params/total*100:.1f}%" if total else "0%",
            "embedding_pct": f"{emb_params/total*100:.1f}%" if total else "0%",
            "is_moe": moe_experts > 0,
        }
    }


def validate_arch(arch: dict) -> list:
    """校验架构配置，返回错误/警告列表"""
    issues = []
    D = arch.get("hidden_dim", 0)
    H = arch.get("num_heads", 0)
    Hkv = arch.get("num_kv_heads", 0)
    V = arch.get("vocab_size", 0)
    L = arch.get("num_layers", 0)
    I = arch.get("intermediate_dim", 0)
    S = arch.get("max_seq_len", 0)

    # 基本范围检查
    if D < 32:
        issues.append({"level": "error", "msg": f"隐藏维度 ({D}) 太小，最少需要 32"})
    if D > 8192:
        issues.append({"level": "warn", "msg": f"隐藏维度 ({D}) 很大，训练可能需要大量显存"})
    if H < 1:
        issues.append({"level": "error", "msg": "注意力头数至少为 1"})
    if H > 0 and D % H != 0:
        issues.append({"level": "error", "msg": f"隐藏维度 ({D}) 必须能被注意力头数 ({H}) 整除"})
    if Hkv < 1:
        issues.append({"level": "error", "msg": "KV 头数至少为 1"})
    if Hkv > H:
        issues.append({"level": "error", "msg": f"KV 头数 ({Hkv}) 不能大于注意力头数 ({H})"})
    if H > 0 and H % Hkv != 0:
        issues.append({"level": "error", "msg": f"注意力头数 ({H}) 必须能被 KV 头数 ({Hkv}) 整除"})
    if V < 100:
        issues.append({"level": "error", "msg": f"词表大小 ({V}) 太小，最少需要 100"})
    if L < 1:
        issues.append({"level": "error", "msg": "层数至少为 1"})
    if I < D:
        issues.append({"level": "warn", "msg": f"FFN 中间维度 ({I}) 通常应大于隐藏维度 ({D})"})
    if S < 32:
        issues.append({"level": "error", "msg": f"最大序列长度 ({S}) 太小"})

    # GQA/MQA 检查
    att = arch.get("attention_type", "mha")
    if att == "mha" and Hkv != H:
        issues.append({"level": "warn", "msg": f"MHA 模式下 KV 头数通常等于注意力头数 (当前 {Hkv} vs {H})"})
    if att == "mqa" and Hkv != 1:
        issues.append({"level": "warn", "msg": f"MQA 模式下 KV 头数通常为 1 (当前 {Hkv})"})

    # 参数量警告
    info = calc_params(arch)
    total = info["total_params"]
    if total > 500_000_000:
        issues.append({"level": "warn", "msg": f"模型参数量 ({info['total_params_fmt']}) 较大，消费级 GPU 训练可能很慢"})
    if total > 1_000_000_000:
        issues.append({"level": "warn", "msg": f"模型超过 1B 参数，强烈建议多卡训练"})

    return issues


def get_templates() -> list:
    """返回所有模板的摘要信息"""
    result = []
    for tid, t in TEMPLATES.items():
        info = calc_params(t["architecture"])
        result.append({
            "id": tid,
            "name": t["name"],
            "family": t["family"],
            "description": t["description"],
            "total_params_fmt": info["total_params_fmt"],
            "vram_train": info["vram"]["train_fp16_fmt"],
        })
    return result


def get_template(tid: str) -> dict | None:
    """返回某个模板的完整配置"""
    t = TEMPLATES.get(tid)
    if not t:
        return None
    result = copy.deepcopy(t)
    result["estimated"] = calc_params(t["architecture"])
    return result


def get_component_options() -> dict:
    """返回所有组件选项、标签和描述"""
    return {
        "options": COMPONENT_OPTIONS,
        "labels": COMPONENT_LABELS,
        "descriptions": COMPONENT_DESCRIPTIONS,
    }
