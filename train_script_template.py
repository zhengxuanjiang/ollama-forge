# 训练脚本模板 — 标准 PEFT + TRL（不依赖 Unsloth，Windows 完全兼容）
def get_train_script(cfg):
    """Generate training script using standard HuggingFace PEFT + TRL."""
    is_qlora = cfg["method"] == "qlora"
    ds_fmt = cfg["dataset_format"]

    # ======================= Dataset loading per format =======================
    if ds_fmt == "alpaca":
        ds_code = f'''
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="{cfg['dataset_path']}", split="train")

    columns = dataset.column_names
    print(f"STEP: Dataset columns: {{columns}}", flush=True)

    inst_col = "instruction" if "instruction" in columns else None
    inp_col = "input" if "input" in columns else None
    out_col = "output" if "output" in columns else ("response" if "response" in columns else ("text" if "text" in columns else None))
    if not inst_col and inp_col:
        inst_col = inp_col
        inp_col = None
    if not out_col:
        print("ERROR: Cannot find output/response/text column in dataset", flush=True)
        sys.exit(1)

    def fmt(example):
        inst = example.get(inst_col, "") if inst_col else ""
        inp = example.get(inp_col, "") if inp_col else ""
        out = example[out_col]
        q = inst
        if inp:
            q = q + "\\n" + inp if q else inp
        messages = [
            {{"role": "user", "content": q}},
            {{"role": "assistant", "content": out}},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {{"text": text}}
    dataset = dataset.map(fmt)
'''
    elif ds_fmt == "sharegpt":
        ds_code = f'''
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="{cfg['dataset_path']}", split="train")

    ROLE_MAP = {{"human": "user", "gpt": "assistant", "system": "system"}}
    def fmt(example):
        convs = example.get("conversations", [])
        messages = []
        for c in convs:
            role = c.get("from", c.get("role", "user"))
            content = c.get("value", c.get("content", ""))
            messages.append({{"role": ROLE_MAP.get(role, role), "content": content}})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {{"text": text}}
    dataset = dataset.map(fmt)
'''
    else:  # openai / messages
        ds_code = f'''
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="{cfg['dataset_path']}", split="train")

    def fmt(example):
        key = "messages" if "messages" in example else "conversations" if "conversations" in example else None
        if key:
            text = tokenizer.apply_chat_template(example[key], tokenize=False, add_generation_prompt=False)
        else:
            text = str(example.get("text", example.get("output", "")))
        return {{"text": text}}
    dataset = dataset.map(fmt)
'''

    # ======================= Conditional code blocks =======================
    if is_qlora:
        bnb_block = '''
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )'''
        model_extra = 'quantization_config=bnb_config,'
        kbit_prep = '    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)'
        optim = 'adamw_8bit'
    else:
        bnb_block = '''
    bnb_config = None'''
        model_extra = ''
        kbit_prep = ''
        optim = 'adamw_torch'

    quant = cfg.get('quant_method', 'q8_0')

    # ======================= Safe merge block =======================
    # QLoRA: must reload base model in fp16 on CPU to avoid tensor corruption
    # LoRA: can merge directly but CPU merge is still safer
    if is_qlora:
        merge_block = f'''
    # ===== Safe Merge (reload in fp16 to avoid 4-bit tensor corruption) =====
    print("STEP: Merging LoRA adapter into base model...", flush=True)
    merged_path = os.path.join(os.getcwd(), "merged_model")
    adapter_path = os.path.join(os.getcwd(), "outputs")

    # Save adapter first
    print("STEP: Saving LoRA adapter...", flush=True)
    model.save_pretrained(adapter_path)

    # Free GPU memory
    del model
    del trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reload base model in fp16 on CPU — critical to avoid tensor shape corruption
    print("STEP: Reloading base model in fp16 for safe merge...", flush=True)
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        "{cfg['base_model']}",
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        token=_hf_token,
    )

    # Load and merge adapter cleanly
    from peft import PeftModel as PM
    base_model_reload = PM.from_pretrained(base_model_reload, adapter_path)
    base_model_reload = base_model_reload.merge_and_unload()

    # Save merged model
    base_model_reload.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    del base_model_reload
    gc.collect()
    print(f"STEP: Merged model saved to {{merged_path}}", flush=True)
'''
    else:
        merge_block = '''
    # ===== Merge & Save (LoRA fp16 — direct merge is safe) =====
    print("STEP: Merging LoRA adapter into base model...", flush=True)
    merged_path = os.path.join(os.getcwd(), "merged_model")

    # Move to CPU for safer merge
    model = model.cpu()
    model = model.merge_and_unload()
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"STEP: Merged model saved to {merged_path}", flush=True)
'''

    # ======================= Modelfile generation =======================
    # Build the Modelfile creation code — detects model family → correct template
    modelfile_block = _build_modelfile_code(cfg)

    # ======================= Ollama export =======================
    if cfg.get('export_ollama'):
        ollama_section = f'''
    print("STEP: Registering model with Ollama...", flush=True)
    try:
        import subprocess as sp
        mf = os.path.join(merged_path, "Modelfile")
        # Use Popen to stream progress (ollama create can take minutes for large models)
        proc = sp.Popen(["ollama", "create", "{cfg['output_name']}", "-f", mf],
                        stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1, cwd=merged_path,
                        encoding='utf-8', errors='replace')
        _ollama_timeout = 900  # 15 minutes
        _start = __import__("time").time()
        _last_print = _start
        for line in proc.stdout:
            line = line.strip()
            if line:
                # Strip ANSI codes
                import re as _re
                line = _re.sub(r"\\x1b\\[[0-9;?]*[a-zA-Z]|\\x1b\\].*?\\x07", "", line).strip()
                if line:
                    print(f"STEP: Ollama: {{line}}", flush=True)
            now = __import__("time").time()
            if now - _last_print > 30:
                elapsed = int(now - _start)
                print(f"STEP: Ollama 注册中... ({{elapsed}}秒)", flush=True)
                _last_print = now
            if now - _start > _ollama_timeout:
                proc.kill()
                print("WARN: Ollama registration timed out (>15min). Model saved locally.", flush=True)
                break
        proc.wait(timeout=30)
        if proc.returncode == 0:
            print(f"STEP: ✅ Ollama model \\'{cfg['output_name']}\\' created!", flush=True)
        else:
            print(f"WARN: ollama create exit code {{proc.returncode}}", flush=True)
            print(f"WARN: Manual: ollama create {cfg['output_name']} -f {{mf}}", flush=True)
    except FileNotFoundError:
        print("WARN: ollama not found in PATH. Import the merged_model folder manually.", flush=True)
    except Exception as e:
        print(f"WARN: Ollama creation: {{e}}", flush=True)
'''
    else:
        ollama_section = ''

    # ======================= Assemble script =======================
    hf_home = cfg.get('hf_home', '').replace('\\', '/')
    hf_env_line = f'    os.environ["HF_HOME"] = "{hf_home}"' if hf_home else '    # HF_HOME: using default (~/.cache/huggingface)'

    script = f'''#!/usr/bin/env python3
"""Auto-generated fine-tuning script — Standard PEFT + TRL (Windows compatible, no Unsloth)"""
import os, sys, json

# Fix Windows GBK encoding issue — ensure stdout uses UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
{hf_env_line}

def main():
    import torch

    # ===== GPU Check =====
    print("STEP: Checking GPU...", flush=True)
    if torch.cuda.is_available():
        print(f"STEP: GPU: {{torch.cuda.get_device_name(0)}} (CUDA {{torch.version.cuda}})", flush=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("STEP: GPU: Apple MPS", flush=True)
    elif torch.version.cuda is None:
        print("ERROR: PyTorch 是 CPU 版本！请在微调中心点击「修复 PyTorch」", flush=True)
        sys.exit(1)
    else:
        print("ERROR: PyTorch 有 CUDA 支持但未检测到 GPU 设备", flush=True)
        sys.exit(1)

    # ===== Load Model =====
    print("STEP: Loading model (may download on first run)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    # Load HF token for gated models (Llama, Gemma, etc.)
    _hf_token = None
    for _tp in [
        os.path.join(os.environ.get("HF_HOME", ""), "token"),
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token"),
    ]:
        if _tp and os.path.isfile(_tp):
            _hf_token = open(_tp).read().strip()
            if _hf_token:
                print("STEP: HuggingFace token loaded", flush=True)
                break
    if not _hf_token:
        _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token  # ensure env var is set for all HF calls

{bnb_block}

    model = AutoModelForCausalLM.from_pretrained(
        "{cfg['base_model']}",
        {model_extra}
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        token=_hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained("{cfg['base_model']}", trust_remote_code=True, token=_hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    print("STEP: Model loaded", flush=True)

    # ===== LoRA =====
    print("STEP: Applying LoRA adapters...", flush=True)
{kbit_prep}
    lora_config = LoraConfig(
        r={cfg['lora_r']},
        lora_alpha={cfg['lora_alpha']},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ===== Dataset =====
    print("STEP: Loading dataset...", flush=True)
{ds_code}
    print(f"STEP: Dataset loaded — {{len(dataset)}} samples", flush=True)

    # ===== Train =====
    print("STEP: Starting training...", flush=True)
    from trl import SFTTrainer, SFTConfig

    bf16_ok = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size={cfg['batch_size']},
            gradient_accumulation_steps=4,
            warmup_steps={cfg['warmup_steps']},
            num_train_epochs={cfg['epochs']},
            learning_rate={cfg['learning_rate']},
            fp16=not bf16_ok,
            bf16=bf16_ok,
            logging_steps=1,
            save_strategy="steps",
            save_steps={cfg['save_steps']},
            save_total_limit=5,
            output_dir="outputs",
            optim="{optim}",
            max_seq_length={cfg['max_seq_length']},
            seed=3407,
            dataloader_num_workers=0,
            dataset_num_proc=None,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={{"use_reentrant": False}},
        ),
    )

    stats = trainer.train({f'resume_from_checkpoint="{cfg["resume_from_checkpoint"]}"' if cfg.get("resume_from_checkpoint") else ''})
    print(f"STEP: Training complete! Loss: {{stats.training_loss:.4f}}", flush=True)

{merge_block}

    # ===== Create Modelfile with proper chat template =====
{modelfile_block}

{ollama_section}
    # Output training metadata for history tracking
    import json as _j
    _ollama_ok = False
    try:
        _ollama_ok = "{cfg['output_name']}" in str(__import__("subprocess").check_output(["ollama", "list"], text=True, timeout=10))
    except Exception:
        pass
    _meta = _j.dumps({{"loss": round(stats.training_loss, 6), "total_steps": stats.global_step, "ollama_ok": _ollama_ok}})
    print(f"TRAIN_META: {{_meta}}", flush=True)
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
'''
    return script


# ======================= Modelfile Template Detection =======================

# Known model families and their Ollama TEMPLATE / SYSTEM / stop tokens
_MODEL_TEMPLATES = {
    # ChatML family: Qwen, Qwen2, Qwen2.5, Qwen3, Phi-3, Phi-4, Yi
    "chatml": {
        "keywords": ["qwen", "phi-3", "phi-4", "phi3", "phi4", "yi-", "deepseek"],
        "template": (
            '<|im_start|>system\n'
            '{{ .System }}<|im_end|>\n'
            '<|im_start|>user\n'
            '{{ .Prompt }}<|im_end|>\n'
            '<|im_start|>assistant\n'
        ),
        "system": "You are a helpful assistant.",
        "stops": ["<|im_end|>", "<|im_start|>"],
    },
    # Llama 3 / 3.1 / 3.2 / 3.3
    "llama3": {
        "keywords": ["llama-3", "llama3"],
        "template": (
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
            '{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
            '{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        ),
        "system": "You are a helpful assistant.",
        "stops": ["<|eot_id|>"],
    },
    # Zephyr family: TinyLlama, Zephyr, StableLM, OpenChat
    "zephyr": {
        "keywords": ["tinyllama", "tiny-llama", "zephyr", "stablelm", "openchat", "rocket"],
        "template": (
            '<|system|>\n'
            '{{ .System }}</s>\n'
            '<|user|>\n'
            '{{ .Prompt }}</s>\n'
            '<|assistant|>\n'
        ),
        "system": "You are a helpful assistant.",
        "stops": ["</s>", "<|user|>"],
    },
    # Gemma / Gemma2
    "gemma": {
        "keywords": ["gemma"],
        "template": (
            '<start_of_turn>user\n'
            '{{ .Prompt }}<end_of_turn>\n'
            '<start_of_turn>model\n'
        ),
        "system": "",
        "stops": ["<end_of_turn>"],
    },
    # Mistral / Mixtral
    "mistral": {
        "keywords": ["mistral", "mixtral"],
        "template": '[INST] {{ .System }} {{ .Prompt }} [/INST]',
        "system": "You are a helpful assistant.",
        "stops": ["[INST]", "[/INST]"],
    },
    # Llama 2 / CodeLlama
    "llama2": {
        "keywords": ["llama-2", "llama2", "codellama"],
        "template": (
            '[INST] <<SYS>>\n{{ .System }}\n<</SYS>>\n\n'
            '{{ .Prompt }} [/INST]'
        ),
        "system": "You are a helpful assistant.",
        "stops": ["[INST]", "[/INST]"],
    },
    # Vicuna / LLaMA-based with Vicuna template
    "vicuna": {
        "keywords": ["vicuna"],
        "template": (
            '{{ .System }}\n\n'
            'USER: {{ .Prompt }}\n'
            'ASSISTANT:'
        ),
        "system": "A chat between a curious user and an artificial intelligence assistant.",
        "stops": ["USER:", "</s>"],
    },
}

# Fallback detection: tokenizer_config.json chat_template → family
_TOKENIZER_HINTS = {
    "<|im_start|>": "chatml",
    "<|start_header_id|>": "llama3",
    "<start_of_turn>": "gemma",
    "[INST]": "mistral",
    "<|system|>": "zephyr",
    "<|user|>": "zephyr",
    "<<SYS>>": "llama2",
}


def _detect_model_family(base_model_name):
    """Detect model family from the HuggingFace model name."""
    name = base_model_name.lower()
    for family, info in _MODEL_TEMPLATES.items():
        if any(kw in name for kw in info["keywords"]):
            return family
    return None


def _build_modelfile_code(cfg):
    """Generate the Python code block that writes a proper Modelfile.

    This code will be injected into the generated training script. It:
    1. Detects the model family from the base model name
    2. Falls back to reading tokenizer_config.json if unknown
    3. Writes a Modelfile with correct TEMPLATE, SYSTEM, stop tokens,
       and repeat_penalty (to prevent output loops)
    """
    base_model = cfg['base_model']
    max_seq = cfg['max_seq_length']

    # Try static detection first
    family = _detect_model_family(base_model)

    if family:
        info = _MODEL_TEMPLATES[family]
        # Escape template for embedding in Python string literal
        tmpl_escaped = info["template"].replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        system_escaped = info["system"].replace('"', '\\"')
        stops = info["stops"]

        lines = []
        lines.append(f'    print("STEP: Creating Modelfile (detected: {family} family)...", flush=True)')
        lines.append(f'    _mf_path = os.path.join(merged_path, "Modelfile")')
        lines.append(f'    with open(_mf_path, "w", encoding="utf-8") as _mf:')
        lines.append(f'        _mf.write("FROM .\\n")')
        lines.append(f'        _mf.write(\'TEMPLATE """' + tmpl_escaped + '"""\\n\')')
        if info["system"]:
            lines.append(f'        _mf.write(\'SYSTEM """{system_escaped}"""\\n\')')
        for s in stops:
            lines.append(f'        _mf.write(\'PARAMETER stop "{s}"\\n\')')
        lines.append(f'        _mf.write("PARAMETER temperature 0.7\\n")')
        lines.append(f'        _mf.write("PARAMETER num_ctx {max_seq}\\n")')
        lines.append(f'        _mf.write("PARAMETER repeat_penalty 1.1\\n")')
        lines.append(f'        _mf.write("PARAMETER repeat_last_n 64\\n")')
        lines.append(f'    print("STEP: Modelfile created", flush=True)')
        return '\n'.join(lines)

    # Unknown model — generate runtime detection code with LLM-assisted fallback
    # Note: base_model and max_seq are injected at code-generation time
    _bm = base_model
    _ms = max_seq
    code = '''    # ===== Detect chat template at runtime =====
    print("STEP: Detecting chat template...", flush=True)
    _family = None
    _tc_path = os.path.join(merged_path, "tokenizer_config.json")
    _tc_content = ""
    if os.path.exists(_tc_path):
        with open(_tc_path, "r", encoding="utf-8") as _f:
            _tc_content = _f.read()
            _tc = json.loads(_tc_content)
        _ct = _tc.get("chat_template", "")
        if "<|im_start|>" in _ct:
            _family = "chatml"
        elif "<|start_header_id|>" in _ct:
            _family = "llama3"
        elif "<start_of_turn>" in _ct:
            _family = "gemma"
        elif "<|system|>" in _ct or "<|user|>" in _ct:
            _family = "zephyr"
        elif "<<SYS>>" in _ct:
            _family = "llama2"
        elif "[INST]" in _ct:
            _family = "mistral"
        elif "USER:" in _ct and "ASSISTANT:" in _ct:
            _family = "vicuna"

    # ===== LLM-assisted Modelfile generation for unknown models =====
    _llm_modelfile = None
    if not _family:
        print("STEP: \\u1f916 使用 AI 分析模型模板...", flush=True)
        try:
            import urllib.request, urllib.error
            _api_key = "sk-191e8a488f80457c8d014c64301b7c35"
            _api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            _tc_summary = _tc_content[:3000] if _tc_content else "No tokenizer_config.json found"
            _prompt = """You are an Ollama Modelfile expert. Given model info, output ONLY Modelfile lines.

Model: """ + _BASE_MODEL_ + """
tokenizer_config.json:
""" + _tc_summary + """

Rules:
1. TEMPLATE uses Go syntax: {{ .System }}, {{ .Prompt }}
2. Stop tokens MUST match the model's special tokens
3. Include: PARAMETER temperature 0.7, num_ctx """ + str(_MAX_SEQ_) + """, repeat_penalty 1.1, repeat_last_n 64
4. Output ONLY Modelfile lines, NO explanations, NO markdown

Example:
TEMPLATE \\"\\"\\"<|system|>
{{ .System }}</s>
<|user|>
{{ .Prompt }}</s>
<|assistant|>
\\"\\"\\"
SYSTEM \\"\\"\\"You are a helpful assistant.\\"\\"\\"
PARAMETER stop "</s>"
PARAMETER temperature 0.7"""

            _body = json.dumps({
                "model": "qwen-turbo",
                "messages": [{"role": "user", "content": _prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            })
            _req = urllib.request.Request(_api_url,
                data=_body.encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {_api_key}"})

            for _attempt in range(3):
                try:
                    with urllib.request.urlopen(_req, timeout=30) as _resp:
                        _rdata = json.loads(_resp.read().decode("utf-8"))
                        _content = _rdata.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if _content and ("TEMPLATE" in _content or "PARAMETER" in _content):
                            _llm_modelfile = _content.strip()
                            print("STEP: \\u2705 AI 成功生成 Modelfile 模板", flush=True)
                            break
                except Exception as _e:
                    print(f"STEP: \\u26a0\\ufe0f AI 分析重试 ({_attempt+1}/3): {_e}", flush=True)
                    import time as _time
                    _time.sleep(2)
        except Exception as _e:
            print(f"STEP: \\u26a0\\ufe0f AI 分析跳过: {_e}", flush=True)

    # ===== Write Modelfile =====
    _mf_path = os.path.join(merged_path, "Modelfile")
    with open(_mf_path, "w", encoding="utf-8") as _mf:
        _mf.write("FROM .\\n")

        if _llm_modelfile:
            _has_template = "TEMPLATE" in _llm_modelfile
            _has_stop = "PARAMETER stop" in _llm_modelfile
            if _has_template and _has_stop:
                # Parse AI output — track multi-line """ blocks
                _in_block = False
                _valid_cmds = ("TEMPLATE", "SYSTEM", "PARAMETER", "LICENSE", "MESSAGE")
                _cleaned_ai = _llm_modelfile.replace("```modelfile", "").replace("```Modelfile", "").replace("```", "").strip()
                for _line in _cleaned_ai.split("\\n"):
                    _line_s = _line.strip()
                    if not _line_s:
                        continue
                    if _in_block:
                        _mf.write(_line_s + "\\n")
                        if '"""' in _line_s:
                            _in_block = False
                        continue
                    # Skip FROM lines
                    if _line_s.upper().startswith("FROM"):
                        continue
                    # Check if line starts with a valid command
                    if any(_line_s.startswith(c) for c in _valid_cmds):
                        _mf.write(_line_s + "\\n")
                        if '"""' in _line_s and _line_s.count('"""') == 1:
                            _in_block = True
                        continue
                if "num_ctx" not in _llm_modelfile:
                    _mf.write("PARAMETER num_ctx " + str(_MAX_SEQ_) + "\\n")
                if "repeat_penalty" not in _llm_modelfile:
                    _mf.write("PARAMETER repeat_penalty 1.1\\n")
                    _mf.write("PARAMETER repeat_last_n 64\\n")
                _family = "llm-generated"
            else:
                print("WARN: AI output incomplete, falling back to rules", flush=True)
                _llm_modelfile = None

        if not _llm_modelfile:
            if _family == "chatml":
                _mf.write('TEMPLATE """<|im_start|>system\\n{{ .System }}<|im_end|>\\n<|im_start|>user\\n{{ .Prompt }}<|im_end|>\\n<|im_start|>assistant\\n"""\\n')
                _mf.write('SYSTEM """You are a helpful assistant."""\\n')
                _mf.write('PARAMETER stop "<|im_end|>"\\n')
                _mf.write('PARAMETER stop "<|im_start|>"\\n')
            elif _family == "llama3":
                _mf.write('TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"""\\n')
                _mf.write('SYSTEM """You are a helpful assistant."""\\n')
                _mf.write('PARAMETER stop "<|eot_id|>"\\n')
            elif _family == "zephyr":
                _mf.write('TEMPLATE """<|system|>\\n{{ .System }}</s>\\n<|user|>\\n{{ .Prompt }}</s>\\n<|assistant|>\\n"""\\n')
                _mf.write('SYSTEM """You are a helpful assistant."""\\n')
                _mf.write('PARAMETER stop "</s>"\\n')
                _mf.write('PARAMETER stop "<|user|>"\\n')
            elif _family == "gemma":
                _mf.write('TEMPLATE """<start_of_turn>user\\n{{ .Prompt }}<end_of_turn>\\n<start_of_turn>model\\n"""\\n')
                _mf.write('PARAMETER stop "<end_of_turn>"\\n')
            elif _family == "mistral":
                _mf.write('TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""\\n')
                _mf.write('SYSTEM """You are a helpful assistant."""\\n')
                _mf.write('PARAMETER stop "[INST]"\\n')
                _mf.write('PARAMETER stop "[/INST]"\\n')
            elif _family == "llama2":
                _mf.write('TEMPLATE """[INST] <<SYS>>\\n{{ .System }}\\n<</SYS>>\\n\\n{{ .Prompt }} [/INST]"""\\n')
                _mf.write('SYSTEM """You are a helpful assistant."""\\n')
                _mf.write('PARAMETER stop "[INST]"\\n')
                _mf.write('PARAMETER stop "[/INST]"\\n')
            elif _family == "vicuna":
                _mf.write('TEMPLATE """{{ .System }}\\n\\nUSER: {{ .Prompt }}\\nASSISTANT:"""\\n')
                _mf.write('SYSTEM """A chat between a curious user and an artificial intelligence assistant."""\\n')
                _mf.write('PARAMETER stop "USER:"\\n')
                _mf.write('PARAMETER stop "</s>"\\n')
            else:
                _eos = ""
                if os.path.exists(_tc_path):
                    try:
                        _tc2 = json.loads(open(_tc_path, encoding="utf-8").read())
                        _eos = _tc2.get("eos_token", "")
                        if isinstance(_eos, dict):
                            _eos = _eos.get("content", "")
                    except Exception:
                        pass
                if _eos and _eos not in ("<|endoftext|>",):
                    _mf.write('TEMPLATE """{{ .System }}\\nUser: {{ .Prompt }}\\nAssistant: """\\n')
                    _mf.write('SYSTEM """You are a helpful assistant."""\\n')
                    _mf.write(f'PARAMETER stop "{_eos}"\\n')
                else:
                    _mf.write('TEMPLATE """<|system|>\\n{{ .System }}</s>\\n<|user|>\\n{{ .Prompt }}</s>\\n<|assistant|>\\n"""\\n')
                    _mf.write('SYSTEM """You are a helpful assistant."""\\n')
                    _mf.write('PARAMETER stop "</s>"\\n')

            _mf.write("PARAMETER temperature 0.7\\n")
            _mf.write("PARAMETER num_ctx " + str(_MAX_SEQ_) + "\\n")
            _mf.write("PARAMETER repeat_penalty 1.1\\n")
            _mf.write("PARAMETER repeat_last_n 64\\n")

    print(f"STEP: Modelfile created (family: {_family or 'fallback'})", flush=True)'''
    # Inject actual values
    code = code.replace('_BASE_MODEL_', repr(_bm))
    code = code.replace('_MAX_SEQ_', str(_ms))
    return code
