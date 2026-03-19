"""exporter.py — 模型导出到 GGUF + 导入 Ollama (v2 - debug edition)"""
import json, os, subprocess, sys, shutil
from pathlib import Path


def export_to_gguf(
    project_dir: str,
    checkpoint_path: str,
    output_name: str = "model",
    quant_type: str = "f16",
    python_path: str | None = None,
) -> dict:
    """将 PyTorch checkpoint 转为 GGUF 格式 (通过 safetensors + llama.cpp)"""
    pdir = Path(project_dir)
    export_dir = pdir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = export_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert checkpoint to HuggingFace format
    # NOTE: This script is executed as a subprocess via `python -c`.
    # All {{ }} are escaped braces for the f-string (become { } in the final script).
    script = f'''
import json, sys, os, traceback
print("[DEBUG-EXPORT] exporter.py v2 — script started", flush=True)

try:
    import torch
    print(f"[DEBUG-EXPORT] torch version: {{torch.__version__}}", flush=True)

    ckpt_path = {json.dumps(checkpoint_path)}
    hf_out = {json.dumps(str(hf_dir))}
    tok_src = os.path.join({json.dumps(project_dir)}, "tokenizer")

    print(f"[DEBUG-EXPORT] Loading checkpoint: {{ckpt_path}}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    state = ckpt["model"]
    tie = arch.get("tie_word_embeddings", True)

    print(f"[DEBUG-EXPORT] arch keys: {{list(arch.keys())}}", flush=True)
    print(f"[DEBUG-EXPORT] tie_word_embeddings = {{tie}}", flush=True)
    print(f"[DEBUG-EXPORT] state_dict has {{len(state)}} keys:", flush=True)
    for k, v in state.items():
        print(f"  [DEBUG-EXPORT]   {{k}}: {{v.shape}} dtype={{v.dtype}} ptr={{v.data_ptr()}}", flush=True)

    # ===== Remap custom keys to HuggingFace Llama format =====
    KEY_MAP = [
        ("tok_emb.weight", "model.embed_tokens.weight"),
        ("pos_emb.weight", "model.embed_positions.weight"),
        (".attn.q.", ".self_attn.q_proj."),
        (".attn.k.", ".self_attn.k_proj."),
        (".attn.v.", ".self_attn.v_proj."),
        (".attn.o.", ".self_attn.o_proj."),
        (".ffn.gate.", ".mlp.gate_proj."),
        (".ffn.up.", ".mlp.up_proj."),
        (".ffn.down.", ".mlp.down_proj."),
        (".norm1.", ".input_layernorm."),
        (".norm2.", ".post_attention_layernorm."),
        ("layers.", "model.layers."),
    ]

    mapped = {{}}
    for k, v in state.items():
        nk = k
        # Special: top-level "norm." -> "model.norm." (must not match norm inside layers)
        if nk.startswith("norm."):
            nk = "model.norm." + nk[len("norm."):]
        for old, new in KEY_MAP:
            nk = nk.replace(old, new)
        # Clone tensor to NEW storage — this is critical for safetensors
        mapped[nk] = v.detach().clone().contiguous()

    print(f"[DEBUG-EXPORT] Mapped {{len(mapped)}} keys:", flush=True)
    for k, v in mapped.items():
        print(f"  [DEBUG-EXPORT]   {{k}}: {{v.shape}} ptr={{v.data_ptr()}}", flush=True)

    # ===== Handle tied weights =====
    # After clone(), all tensors are independent — safe for safetensors.
    # Ensure lm_head.weight exists (Ollama needs it even when originally tied).
    if "lm_head.weight" not in mapped and "model.embed_tokens.weight" in mapped:
        print("[DEBUG-EXPORT] lm_head.weight missing, copying from embed_tokens", flush=True)
        mapped["lm_head.weight"] = mapped["model.embed_tokens.weight"].clone()

    # ===== Double-check: no two tensors share storage =====
    ptrs = {{}}
    for k, v in mapped.items():
        p = v.data_ptr()
        if p in ptrs:
            print(f"[DEBUG-EXPORT] WARNING: {{k}} shares storage with {{ptrs[p]}}, re-cloning", flush=True)
            mapped[k] = v.detach().clone().contiguous()
        ptrs[p] = k

    print(f"[DEBUG-EXPORT] Final tensor count: {{len(mapped)}}", flush=True)

    # ===== Save =====
    safetensors_path = os.path.join(hf_out, "model.safetensors")
    try:
        from safetensors.torch import save_file
        print(f"[DEBUG-EXPORT] Saving safetensors to {{safetensors_path}} ...", flush=True)
        save_file(mapped, safetensors_path)
        print(f"[DEBUG-EXPORT] safetensors saved OK, size={{os.path.getsize(safetensors_path)}}", flush=True)
    except ImportError:
        bin_path = os.path.join(hf_out, "pytorch_model.bin")
        print(f"[DEBUG-EXPORT] safetensors not available, saving pytorch_model.bin", flush=True)
        torch.save(mapped, bin_path)
    except Exception as e:
        print(f"[DEBUG-EXPORT] safetensors save_file FAILED: {{e}}", flush=True)
        print(f"[DEBUG-EXPORT] Full traceback:", flush=True)
        traceback.print_exc()
        # Fallback: try torch.save
        bin_path = os.path.join(hf_out, "pytorch_model.bin")
        print(f"[DEBUG-EXPORT] Falling back to pytorch_model.bin", flush=True)
        torch.save(mapped, bin_path)

    # ===== Generate config.json =====
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
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
    }}
    cfg_path = os.path.join(hf_out, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"[DEBUG-EXPORT] config.json written", flush=True)

    # ===== Copy tokenizer files =====
    if os.path.exists(tok_src):
        import shutil
        for fn in os.listdir(tok_src):
            src = os.path.join(tok_src, fn)
            dst = os.path.join(hf_out, fn)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"[DEBUG-EXPORT] Tokenizer files copied", flush=True)

    print(json.dumps({{"status": "ok", "hf_dir": hf_out}}))

except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"status": "error", "error": str(e)}}))
'''

    py = python_path or sys.executable
    try:
        r = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            encoding="utf-8", errors="replace",
        )

        # Collect all debug lines + find JSON result
        all_output = (r.stdout or "") + "\n" + (r.stderr or "")
        debug_lines = [l for l in all_output.split("\n") if "[DEBUG-EXPORT]" in l]
        result = None
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    pass

        if r.returncode != 0 or not result or result.get("status") != "ok":
            err = r.stderr.strip() or r.stdout.strip()
            debug_info = "\n".join(debug_lines[-20:]) if debug_lines else ""
            return {
                "status": "error",
                "error": f"进程退出码 {r.returncode}: {err[:800]}",
                "debug": debug_info,
            }

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "导出超时 (>2分钟)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Step 2: Try GGUF conversion via llama.cpp's convert script
    gguf_path = export_dir / f"{output_name}.gguf"

    # Look for llama.cpp convert script
    convert_script = None
    for candidate in [
        shutil.which("convert_hf_to_gguf.py"),
        shutil.which("convert-hf-to-gguf.py"),
    ]:
        if candidate:
            convert_script = candidate
            break

    if convert_script:
        try:
            cmd = [py, convert_script, str(hf_dir), "--outfile", str(gguf_path), "--outtype", quant_type]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and gguf_path.exists():
                return {
                    "status": "ok",
                    "gguf_path": str(gguf_path),
                    "size": gguf_path.stat().st_size,
                    "size_fmt": f"{gguf_path.stat().st_size / 1e6:.1f} MB",
                }
        except Exception:
            pass

    # If llama.cpp not available, return HF format path
    return {
        "status": "ok",
        "note": "llama.cpp 未安装，已导出为 HuggingFace 格式。可手动使用 convert_hf_to_gguf.py 转换。",
        "hf_dir": str(hf_dir),
        "gguf_path": None,
    }


def import_to_ollama(
    project_dir: str,
    gguf_path: str | None = None,
    model_name: str = "my-pretrained-model",
    system_prompt: str = "",
) -> dict:
    """创建 Modelfile 并导入到 Ollama（支持 GGUF 或 safetensors 目录）"""
    pdir = Path(project_dir)
    export_dir = pdir / "export"
    hf_dir = export_dir / "hf_model"

    # ===== Find GGUF file =====
    model_from = None

    # Priority 1: explicit gguf_path
    if gguf_path and Path(gguf_path).exists():
        model_from = gguf_path

    # Priority 2: any .gguf in export dir
    if not model_from:
        gguf_files = sorted(export_dir.glob("*.gguf"), key=lambda f: f.stat().st_mtime, reverse=True)
        if gguf_files:
            model_from = str(gguf_files[0])

    if not model_from:
        return {
            "status": "error",
            "error": "未找到 GGUF 文件。请先点击「导出 GGUF」，确保导出成功（显示 ✅ GGUF 导出完成），然后再导入。",
        }

    # Create Modelfile
    modelfile = f'FROM {model_from}\n'
    if system_prompt:
        modelfile += f'SYSTEM """{system_prompt}"""\n'
    modelfile += 'PARAMETER temperature 0.7\n'
    modelfile += 'PARAMETER top_k 50\n'
    modelfile += 'PARAMETER top_p 0.9\n'

    modelfile_path = export_dir / "Modelfile"
    modelfile_path.write_text(modelfile, "utf-8")

    # Import via ollama create
    try:
        r = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0:
            return {
                "status": "ok",
                "model_name": model_name,
                "modelfile": modelfile,
                "message": f"模型 {model_name} 已导入 Ollama！可以在对话界面中使用。",
            }
        else:
            err = r.stderr.strip() or r.stdout.strip()
            return {"status": "error", "error": f"Ollama 导入失败: {err}"}
    except FileNotFoundError:
        return {"status": "error", "error": "未找到 ollama 命令。请确保 Ollama 已安装并在 PATH 中。"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Ollama 导入超时 (>10分钟)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
