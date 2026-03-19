"""routes.py — 训练实验室 API 路由"""
import json, os, shutil, time, secrets, threading
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse

from .architect import (
    calc_params, validate_arch, get_templates, get_template,
    get_component_options, TEMPLATES,
)
from .tokenizer_builder import (
    PRETRAINED_TOKENIZERS, train_bpe_tokenizer,
    download_pretrained_tokenizer, preview_tokenization,
    simple_text_stats,
)
from .data_manager import (
    RECOMMENDED_DATASETS, get_recommended_datasets,
    get_recommended_sft_datasets,
    list_datasets, save_uploaded_text, save_pasted_text,
    delete_dataset, get_dataset_preview, get_dataset_stats,
    process_dataset_for_training, get_processed_info,
    detect_columns,
)
from .trainer import (
    get_train_state, start_training, stop_training,
    list_checkpoints, load_samples, load_training_records,
)
from .inference import generate_text, generate_text_streaming
from .exporter import export_to_gguf, import_to_ollama
from .streaming import (
    run_streaming,
    script_download_tokenizer, script_train_tokenizer,
    script_process_dataset, script_process_sft_dataset,
    script_export_hf,
    script_download_hf_dataset, script_download_hf_dataset_sft,
)

router = APIRouter(prefix="/api/pretrain", tags=["pretrain"])

# ======================= 项目存储 =======================
def _projects_root() -> Path:
    """训练实验室项目根目录，复用 FT_DIR"""
    from backend.app import _get_ft_dir
    p = _get_ft_dir() / "pretrain-lab"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _project_dir(project_id: str) -> Path:
    # 防路径穿越
    safe = "".join(c for c in project_id if c.isalnum() or c in "-_")
    return _projects_root() / safe


def _load_project_meta(project_id: str) -> dict | None:
    f = _project_dir(project_id) / "project.json"
    if f.exists():
        return json.loads(f.read_text("utf-8"))
    return None


def _save_project_meta(project_id: str, data: dict):
    d = _project_dir(project_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ======================= 架构相关 =======================
@router.get("/arch/templates")
async def list_templates():
    """返回所有预设模板的摘要"""
    return {"templates": get_templates()}


@router.get("/arch/template/{tid}")
async def get_template_detail(tid: str):
    """返回某个模板的完整配置"""
    t = get_template(tid)
    if not t:
        raise HTTPException(404, f"模板 '{tid}' 不存在")
    return t


@router.post("/arch/validate")
async def validate_architecture(request: Request):
    """校验架构配置，返回参数量计算和问题列表"""
    body = await request.json()
    arch = body.get("architecture", body)
    issues = validate_arch(arch)
    info = calc_params(arch)
    return {
        "valid": not any(i["level"] == "error" for i in issues),
        "issues": issues,
        "estimated": info,
    }


@router.get("/arch/components")
async def list_components():
    """返回所有组件选项和说明（供前端编辑器使用）"""
    return get_component_options()


# ======================= 项目管理 =======================
@router.post("/projects/create")
async def create_project(request: Request):
    """新建训练项目"""
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "项目名不能为空")
    # 生成 ID
    pid = f"{int(time.time())}-{secrets.token_hex(3)}"
    template_id = body.get("template_id", "")
    # 如果选了模板，用模板的架构作为初始值
    if template_id and template_id in TEMPLATES:
        arch = get_template(template_id)["architecture"]
    else:
        arch = body.get("architecture", get_template("llama-tiny")["architecture"])

    meta = {
        "id": pid,
        "name": name,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "status": "design",  # design / data / training / done
        "template_id": template_id,
        "architecture": arch,
        "tokenizer": body.get("tokenizer", {"type": "pretrained", "source": "", "vocab_size": arch.get("vocab_size", 32000)}),
        "training": body.get("training", {}),
        "history": [],
    }
    _save_project_meta(pid, meta)
    return {"project": meta, "estimated": calc_params(arch)}


@router.get("/projects")
async def list_projects():
    """列出所有项目"""
    root = _projects_root()
    projects = []
    for d in sorted(root.iterdir(), reverse=True):
        if d.is_dir():
            f = d / "project.json"
            if f.exists():
                try:
                    meta = json.loads(f.read_text("utf-8"))
                    info = calc_params(meta.get("architecture", {}))
                    projects.append({
                        "id": meta["id"],
                        "name": meta["name"],
                        "created": meta.get("created", ""),
                        "updated": meta.get("updated", ""),
                        "status": meta.get("status", "design"),
                        "template_id": meta.get("template_id", ""),
                        "total_params_fmt": info["total_params_fmt"],
                    })
                except Exception:
                    pass
    return {"projects": projects}


@router.get("/projects/{pid}")
async def get_project(pid: str):
    """获取项目详情"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    meta["estimated"] = calc_params(meta.get("architecture", {}))
    return meta


@router.put("/projects/{pid}")
async def update_project(pid: str, request: Request):
    """更新项目（架构、训练配置等）"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    body = await request.json()
    # 可更新的字段
    if "name" in body:
        meta["name"] = body["name"]
    if "architecture" in body:
        meta["architecture"] = body["architecture"]
    if "tokenizer" in body:
        meta["tokenizer"] = body["tokenizer"]
    if "training" in body:
        meta["training"] = body["training"]
    if "status" in body:
        meta["status"] = body["status"]
    meta["updated"] = datetime.now().isoformat()
    _save_project_meta(pid, meta)
    return {"project": meta, "estimated": calc_params(meta.get("architecture", {}))}


@router.delete("/projects/{pid}")
async def delete_project(pid: str):
    """删除项目"""
    d = _project_dir(pid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return {"deleted": True}
    raise HTTPException(404, "项目不存在")


@router.post("/projects/{pid}/arch")
async def save_project_arch(pid: str, request: Request):
    """单独保存项目的架构设计"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    body = await request.json()
    arch = body.get("architecture", {})

    # 校验
    issues = validate_arch(arch)
    has_error = any(i["level"] == "error" for i in issues)

    if not has_error:
        meta["architecture"] = arch
        meta["updated"] = datetime.now().isoformat()
        _save_project_meta(pid, meta)

    return {
        "saved": not has_error,
        "issues": issues,
        "estimated": calc_params(arch),
    }


# ======================= Tokenizer 相关 =======================
@router.get("/tokenizer/list")
async def list_tokenizers():
    """返回可用的预训练 tokenizer 列表"""
    return {"tokenizers": PRETRAINED_TOKENIZERS}


@router.get("/tokenizer/project/{pid}")
async def get_project_tokenizer_info(pid: str):
    """获取项目的 tokenizer 状态"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    tok_config = meta.get("tokenizer", {})
    tok_dir = _project_dir(pid) / "tokenizer"
    has_tokenizer = tok_dir.exists() and any(tok_dir.iterdir()) if tok_dir.exists() else False

    return {
        "config": tok_config,
        "has_tokenizer": has_tokenizer,
        "tokenizer_dir": str(tok_dir) if has_tokenizer else None,
    }


@router.post("/tokenizer/download")
async def download_tokenizer(request: Request):
    """下载预训练 tokenizer 到项目（SSE 流式）"""
    body = await request.json()
    pid = body.get("project_id", "")
    source = body.get("source", "")
    python_path = body.get("python_path", "")

    if not pid:
        raise HTTPException(400, "缺少 project_id")
    if not source:
        raise HTTPException(400, "缺少 tokenizer source")

    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    tok_dir = _project_dir(pid) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    script = script_download_tokenizer(source, tok_dir.as_posix())

    def gen():
        final_data = {}
        for chunk in run_streaming(script, python_path or None):
            yield chunk
            # Parse final result to update project meta
            if "data: " in chunk:
                try:
                    ev = json.loads(chunk.split("data: ")[1].strip())
                    if ev.get("done"):
                        final_data = ev
                except Exception:
                    pass
        # Update project meta after completion
        if final_data.get("done"):
            m = _load_project_meta(pid)
            if m:
                m["tokenizer"] = {
                    "type": "pretrained", "source": source,
                    "vocab_size": final_data.get("vocab_size", 0),
                }
                m["updated"] = datetime.now().isoformat()
                _save_project_meta(pid, m)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/tokenizer/train")
async def train_tokenizer(request: Request):
    """从项目数据集训练新 BPE tokenizer（SSE 流式）"""
    body = await request.json()
    pid = body.get("project_id", "")
    vocab_size = body.get("vocab_size", 8000)
    min_freq = body.get("min_frequency", 2)
    python_path = body.get("python_path", "")

    if not pid:
        raise HTTPException(400, "缺少 project_id")
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    ds_dir = _project_dir(pid) / "datasets"
    if not ds_dir.exists():
        raise HTTPException(400, "项目没有数据集，请先上传或添加数据")

    data_files = [str(f) for f in ds_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in ('.txt', '.jsonl')]
    if not data_files:
        raise HTTPException(400, "没有找到可用的文本文件 (.txt/.jsonl)")

    tok_dir = _project_dir(pid) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    script = script_train_tokenizer(data_files, tok_dir.as_posix(), vocab_size, min_freq)

    def gen():
        final_data = {}
        for chunk in run_streaming(script, python_path or None):
            yield chunk
            if "data: " in chunk:
                try:
                    ev = json.loads(chunk.split("data: ")[1].strip())
                    if ev.get("done"):
                        final_data = ev
                except Exception:
                    pass
        if final_data.get("done"):
            m = _load_project_meta(pid)
            if m:
                m["tokenizer"] = {
                    "type": "custom_bpe",
                    "vocab_size": final_data.get("vocab_size", vocab_size),
                    "source": "custom",
                }
                m["updated"] = datetime.now().isoformat()
                _save_project_meta(pid, m)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/tokenizer/preview")
async def preview_tokens(request: Request):
    """使用项目的 tokenizer 预览分词结果"""
    body = await request.json()
    pid = body.get("project_id", "")
    text = body.get("text", "")
    python_path = body.get("python_path", "")

    if not text:
        raise HTTPException(400, "请输入文本")
    if not pid:
        raise HTTPException(400, "缺少 project_id")

    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    tok_dir = _project_dir(pid) / "tokenizer"
    if not tok_dir.exists() or not any(tok_dir.iterdir()):
        # 如果没有下载/训练 tokenizer，返回简单统计
        return {
            "error": "尚未配置 tokenizer，请先下载或训练",
            "simple_stats": simple_text_stats(text),
        }

    return preview_tokenization(str(tok_dir), text, python_path or None)


# ======================= 数据集相关 =======================
@router.get("/datasets/recommended")
async def recommended_datasets():
    """返回推荐的预训练数据集列表"""
    return {"datasets": get_recommended_datasets()}


@router.get("/datasets/recommended-sft")
async def recommended_sft_datasets():
    """返回推荐的 SFT 对话数据集列表"""
    return {"datasets": get_recommended_sft_datasets()}


@router.get("/datasets/{pid}")
async def list_project_datasets(pid: str):
    """列出项目的数据集"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    pdir = _project_dir(pid)
    ds = list_datasets(pdir)
    processed = get_processed_info(pdir)
    return {"datasets": ds, "processed": processed}


@router.post("/datasets/{pid}/upload")
async def upload_dataset(pid: str, file: UploadFile = File(...)):
    """上传文本文件到项目"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    content = await file.read()
    fname = file.filename or "upload.txt"
    ext = Path(fname).suffix.lower()

    if ext not in ('.txt', '.jsonl', '.json', '.csv', '.tsv'):
        raise HTTPException(400, f"不支持的格式: {ext}，支持 .txt / .jsonl / .json / .csv / .tsv")

    if len(content) > 500 * 1024 * 1024:  # 500MB limit
        raise HTTPException(400, "文件过大，上限 500MB")

    result = save_uploaded_text(_project_dir(pid), fname, content)
    return result


@router.post("/datasets/{pid}/paste")
async def paste_text_dataset(pid: str, request: Request):
    """保存粘贴的文本为数据集"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    body = await request.json()
    text = body.get("text", "")
    name = body.get("name", "")

    if not text.strip():
        raise HTTPException(400, "文本内容为空")
    if len(text) > 50 * 1024 * 1024:  # 50MB limit for pasted text
        raise HTTPException(400, "文本过大")

    result = save_pasted_text(_project_dir(pid), text, name)
    return result


@router.delete("/datasets/{pid}/{filename}")
async def delete_project_dataset(pid: str, filename: str):
    """删除项目的数据集文件"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    ok = delete_dataset(_project_dir(pid), filename)
    if ok:
        return {"deleted": True}
    raise HTTPException(404, "文件不存在")


@router.get("/datasets/{pid}/preview/{filename}")
async def preview_dataset(pid: str, filename: str):
    """预览数据集内容"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    return get_dataset_preview(_project_dir(pid), filename)


@router.get("/datasets/{pid}/stats/{filename}")
async def dataset_stats(pid: str, filename: str):
    """获取数据集详细统计"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    return get_dataset_stats(_project_dir(pid), filename)


@router.get("/datasets/{pid}/columns/{filename}")
async def get_dataset_columns(pid: str, filename: str):
    """检测结构化数据文件的列信息"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    return detect_columns(_project_dir(pid), filename)


@router.post("/datasets/{pid}/download-hf")
async def download_hf_dataset(pid: str, request: Request):
    """从 HuggingFace 下载数据集（SSE 流式，自动检测 split）"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    body = await request.json()
    hf_id = body.get("hf_id", "").strip()
    filename = body.get("filename", "dataset.txt")
    python_path = body.get("python_path", "")
    max_rows = body.get("max_rows", 0)
    mode = body.get("mode", "pretrain")  # "pretrain" or "sft"

    if not hf_id:
        raise HTTPException(400, "请输入 HuggingFace 数据集 ID")

    ds_dir = _project_dir(pid) / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    out_path = (ds_dir / filename).as_posix()

    if mode == "sft":
        script = script_download_hf_dataset_sft(hf_id, out_path, max_rows)
    else:
        script = script_download_hf_dataset(hf_id, out_path, max_rows)
    return StreamingResponse(
        run_streaming(script, python_path or None, timeout=1800),
        media_type="text/event-stream",
    )


@router.post("/datasets/{pid}/process")
async def process_dataset(pid: str, request: Request):
    """将数据集分词打包为训练格式（SSE 流式）"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    body = await request.json()
    filenames = body.get("files", [])
    max_seq_len = body.get("max_seq_len", meta.get("architecture", {}).get("max_seq_len", 512))
    python_path = body.get("python_path", "")
    output_name = body.get("output_name", "train_data")
    column_config = body.get("column_config", {})  # {filename: [col1, col2, ...]}
    column_separator = body.get("column_separator", "\n")

    pdir = _project_dir(pid)
    ds_dir = pdir / "datasets"

    if not filenames:
        filenames = [f.name for f in ds_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in ('.txt', '.jsonl', '.json', '.csv', '.tsv')]

    data_files = [str(ds_dir / f) for f in filenames if (ds_dir / f).exists()]
    if not data_files:
        raise HTTPException(400, "没有可处理的数据文件")

    tok_dir = pdir / "tokenizer"
    if not tok_dir.exists() or not any(tok_dir.iterdir()):
        raise HTTPException(400, "请先配置 tokenizer")

    out_dir = pdir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bin = str(out_dir / f"{output_name}.bin")
    out_meta = str(out_dir / f"{output_name}_meta.json")

    script = script_process_dataset(data_files, str(tok_dir), max_seq_len, out_bin, out_meta,
                                     column_config=column_config, column_separator=column_separator)
    return StreamingResponse(
        run_streaming(script, python_path or None),
        media_type="text/event-stream",
    )


@router.post("/datasets/{pid}/process-sft")
async def process_sft_dataset(pid: str, request: Request):
    """将对话数据集分词打包为 SFT 训练格式（带 loss mask）（SSE 流式）"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    body = await request.json()
    filenames = body.get("files", [])
    max_seq_len = body.get("max_seq_len", meta.get("architecture", {}).get("max_seq_len", 512))
    python_path = body.get("python_path", "")
    chat_template = body.get("chat_template", "chatml")

    pdir = _project_dir(pid)
    ds_dir = pdir / "datasets"

    if not filenames:
        filenames = [f.name for f in ds_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in ('.jsonl', '.json')]

    data_files = [str(ds_dir / f) for f in filenames if (ds_dir / f).exists()]
    if not data_files:
        raise HTTPException(400, "没有可处理的对话数据文件。请上传 JSONL 格式的对话数据，或使用「格式转换」将其他格式转换为 JSONL。")

    tok_dir = pdir / "tokenizer"
    if not tok_dir.exists() or not any(tok_dir.iterdir()):
        raise HTTPException(400, "请先配置 tokenizer")

    out_dir = pdir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tokens = str(out_dir / "sft_tokens.bin")
    out_masks = str(out_dir / "sft_masks.bin")
    out_meta = str(out_dir / "sft_data_meta.json")

    script = script_process_sft_dataset(data_files, str(tok_dir), max_seq_len, out_tokens, out_masks, out_meta, chat_template)
    return StreamingResponse(
        run_streaming(script, python_path or None),
        media_type="text/event-stream",
    )


@router.post("/datasets/{pid}/convert-to-sft")
async def convert_to_sft_format(pid: str, request: Request):
    """将 Alpaca/ShareGPT/自定义格式转换为标准 SFT JSONL"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    body = await request.json()
    filename = body.get("filename", "")
    source_format = body.get("format", "auto")  # auto, alpaca, sharegpt

    pdir = _project_dir(pid)
    src = pdir / "datasets" / filename
    if not src.exists():
        raise HTTPException(400, "文件不存在")

    try:
        text = src.read_text("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "文件编码不是 UTF-8")

    # Try to parse
    conversations = []
    lines = text.strip().split('\n')

    # Try JSONL first (tolerant: skip non-JSON lines)
    parsed_lines = []
    json_lines_found = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed_lines.append(json.loads(line))
            json_lines_found += 1
        except json.JSONDecodeError:
            continue  # skip non-JSON lines instead of breaking

    if not parsed_lines:
        # Try as a single JSON array
        try:
            parsed_lines = json.loads(text)
            if not isinstance(parsed_lines, list):
                parsed_lines = [parsed_lines]
        except json.JSONDecodeError:
            parsed_lines = []

    # If we got JSON objects, process them
    for obj in parsed_lines:
        if not isinstance(obj, dict):
            continue
        # Already in conversations format?
        if "conversations" in obj or "messages" in obj:
            conv = obj.get("conversations") or obj.get("messages")
            if conv and isinstance(conv, list):
                conversations.append({"conversations": conv})
            continue

        # Alpaca format: instruction, input, output
        if "instruction" in obj:
            msgs = []
            instruction = obj.get("instruction", "")
            inp = obj.get("input", "")
            output = obj.get("output", "")
            if inp:
                instruction = f"{instruction}\n{inp}"
            msgs.append({"role": "user", "content": instruction})
            msgs.append({"role": "assistant", "content": output})
            conversations.append({"conversations": msgs})
            continue

        # ShareGPT format: from/value pairs
        if isinstance(obj, dict) and "conversations" not in obj:
            conv_list = obj.get("conversation") or obj.get("dialog") or []
            if isinstance(conv_list, list) and conv_list:
                msgs = []
                for turn in conv_list:
                    role = turn.get("from", turn.get("role", "user"))
                    content = turn.get("value", turn.get("content", ""))
                    if role in ("human", "user"):
                        role = "user"
                    elif role in ("gpt", "assistant", "bot"):
                        role = "assistant"
                    msgs.append({"role": role, "content": content})
                if msgs:
                    conversations.append({"conversations": msgs})
                continue

        # system_prompt / question / response format (OpenOrca)
        if "question" in obj and "response" in obj:
            msgs = []
            if obj.get("system_prompt"):
                msgs.append({"role": "system", "content": obj["system_prompt"]})
            msgs.append({"role": "user", "content": obj["question"]})
            msgs.append({"role": "assistant", "content": obj["response"]})
            conversations.append({"conversations": msgs})
            continue

    if not conversations:
        # Fallback: try plain text patterns (for .txt files)
        # Pattern 1: Q:/A: or User:/Assistant: or 问:/答: pairs
        import re
        qa_patterns = [
            (r'^(?:Q|User|Human|问|用户)[:\s：](.+)', r'^(?:A|Assistant|Bot|答|助手)[:\s：](.+)'),
        ]
        for q_pat, a_pat in qa_patterns:
            i = 0
            while i < len(lines):
                q_match = re.match(q_pat, lines[i].strip(), re.IGNORECASE)
                if q_match and i + 1 < len(lines):
                    a_match = re.match(a_pat, lines[i+1].strip(), re.IGNORECASE)
                    if a_match:
                        conversations.append({"conversations": [
                            {"role": "user", "content": q_match.group(1).strip()},
                            {"role": "assistant", "content": a_match.group(1).strip()},
                        ]})
                        i += 2
                        continue
                i += 1
            if conversations:
                break

        # Pattern 2: Tab-separated pairs (question\tanswer)
        if not conversations:
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and len(parts[0]) > 3 and len(parts[1]) > 3:
                    conversations.append({"conversations": [
                        {"role": "user", "content": parts[0].strip()},
                        {"role": "assistant", "content": parts[1].strip()},
                    ]})

    if not conversations:
        raise HTTPException(400,
            "未能从文件中提取对话数据。支持的格式：\n"
            "• JSONL/JSON (Alpaca, ShareGPT, conversations)\n"
            "• 纯文本 Q:/A: 对话格式\n"
            "• Tab 分隔的问答对"
        )

    # Write output
    out_name = src.stem + "_sft.jsonl"
    out_path = pdir / "datasets" / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Preview
    preview = []
    for c in conversations[:3]:
        preview.append(c["conversations"])

    return {
        "status": "ok",
        "output_file": out_name,
        "total_conversations": len(conversations),
        "preview": preview,
    }


# ======================= 硬件检测 =======================
@router.get("/hardware")
async def detect_hardware(python_path: str = ""):
    """检测 GPU 和 CUDA 环境"""
    import subprocess, sys, platform
    py = python_path or sys.executable

    info = {
        "os": platform.system(),
        "gpus": [],
        "cuda_available": False,
        "torch_version": "",
        "cuda_version": "",
    }

    # Try nvidia-smi first
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    info["gpus"].append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_total_mb": int(float(parts[2])),
                        "vram_free_mb": int(float(parts[3])),
                        "driver": parts[4],
                    })
    except Exception:
        pass

    # Check torch + CUDA
    try:
        script = (
            "import torch, json; print(json.dumps({"
            "'torch': torch.__version__,"
            "'cuda': torch.cuda.is_available(),"
            "'cuda_ver': torch.version.cuda or '',"
            "'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,"
            "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []"
            "}))"
        )
        r = subprocess.run([py, "-c", script], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            d = json.loads(r.stdout.strip())
            info["torch_version"] = d.get("torch", "")
            info["cuda_available"] = d.get("cuda", False)
            info["cuda_version"] = d.get("cuda_ver", "")
            # If nvidia-smi didn't work, use torch info
            if not info["gpus"] and d.get("devices"):
                for i, name in enumerate(d["devices"]):
                    info["gpus"].append({"index": i, "name": name, "vram_total_mb": 0, "vram_free_mb": 0, "driver": ""})
    except Exception:
        pass

    return info


# ======================= Training 相关 =======================
@router.post("/train/start")
async def start_train(request: Request):
    """启动预训练"""
    body = await request.json()
    pid = body.get("project_id", "")
    python_path = body.get("python_path", "")

    if not pid:
        raise HTTPException(400, "缺少 project_id")
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    # Merge training config from body into project
    if "training" in body:
        meta["training"] = {**meta.get("training", {}), **body["training"]}
        meta["updated"] = datetime.now().isoformat()
        _save_project_meta(pid, meta)

    meta["id"] = pid
    result = start_training(_project_dir(pid), meta, python_path or None)
    if result.get("status") == "ok":
        meta["status"] = "training"
        _save_project_meta(pid, meta)
    return result


@router.post("/train/stop")
async def stop_train():
    """停止训练"""
    return stop_training()


@router.get("/train/status")
async def train_status():
    """获取训练状态"""
    return get_train_state()


@router.get("/train/checkpoints/{pid}")
async def get_checkpoints(pid: str, mode: str = "pretrain"):
    """列出项目的 checkpoint（按模式区分）"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    return {"checkpoints": list_checkpoints(_project_dir(pid), mode)}


@router.get("/train/samples/{pid}")
async def get_samples(pid: str, mode: str = "pretrain"):
    """获取训练过程中自动生成的样本"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    state = get_train_state()
    return {
        "live_samples": state.get("samples", []),
        "saved_samples": load_samples(_project_dir(pid), mode),
    }


# ======================= Inference / Playground =======================
@router.post("/inference/generate")
async def infer_generate(request: Request):
    """使用指定 checkpoint 生成文本"""
    body = await request.json()
    pid = body.get("project_id", "")
    checkpoint = body.get("checkpoint", "")
    prompts = body.get("prompts", ["Hello"])
    max_tokens = body.get("max_tokens", 100)
    temperature = body.get("temperature", 0.8)
    top_k = body.get("top_k", 50)
    python_path = body.get("python_path", "")

    if not pid:
        raise HTTPException(400, "缺少 project_id")
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    # If no checkpoint specified, find the latest
    if not checkpoint:
        mode = body.get("mode", "pretrain")
        ckpts = list_checkpoints(_project_dir(pid), mode)
        if not ckpts:
            raise HTTPException(400, "没有可用的 checkpoint")
        checkpoint = ckpts[-1]["path"]

    return generate_text(
        project_dir=str(_project_dir(pid)),
        checkpoint_path=checkpoint,
        prompts=prompts,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        python_path=python_path or None,
    )


@router.post("/inference/stream")
async def infer_stream(request: Request):
    """流式生成文本 (SSE，逐 token)"""
    body = await request.json()
    pid = body.get("project_id", "")
    checkpoint = body.get("checkpoint", "")
    prompt = body.get("prompt", "Hello")
    max_tokens = body.get("max_tokens", 200)
    temperature = body.get("temperature", 0.8)
    top_k = body.get("top_k", 50)
    python_path = body.get("python_path", "")
    chat_template = body.get("chat_template", "")  # empty = raw, "chatml"/"llama"/"simple"
    raw_prompt = body.get("raw_prompt", False)  # if true, skip prompt wrapping but still use stop tokens

    if not pid:
        raise HTTPException(400, "缺少 project_id")
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    if not checkpoint:
        mode = "sft" if chat_template else "pretrain"
        ckpts = list_checkpoints(_project_dir(pid), mode)
        if not ckpts:
            raise HTTPException(400, "没有可用的 checkpoint")
        checkpoint = ckpts[-1]["path"]

    # Wrap prompt with chat template if specified (skip if raw_prompt)
    actual_prompt = prompt
    if chat_template and not raw_prompt:
        if chat_template == "chatml":
            actual_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif chat_template == "llama":
            actual_prompt = f"[INST] {prompt} [/INST]\n"
        elif chat_template == "simple":
            actual_prompt = f"### User:\n{prompt}\n\n### Assistant:\n"

    # Determine stop tokens for chat mode
    stop_tokens = []
    if chat_template == "chatml":
        stop_tokens = ["<|im_end|>", "<|im_start|>"]
    elif chat_template == "llama":
        stop_tokens = ["[INST]", "</s>"]
    elif chat_template == "simple":
        stop_tokens = ["### User:", "###"]

    return StreamingResponse(
        generate_text_streaming(
            project_dir=str(_project_dir(pid)),
            checkpoint_path=checkpoint,
            prompt=actual_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            python_path=python_path or None,
            stop_tokens=stop_tokens,
        ),
        media_type="text/event-stream",
    )


# ======================= Training Records =======================
@router.get("/train/history/{pid}")
async def get_training_history(pid: str):
    """获取项目的训练历史记录"""
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")
    return {"records": load_training_records(_project_dir(pid))}


# ======================= Export =======================
@router.post("/export/gguf")
async def export_gguf(request: Request):
    """导出模型为 GGUF 格式（SSE 流式）"""
    body = await request.json()
    pid = body.get("project_id", "")
    checkpoint = body.get("checkpoint", "")
    output_name = body.get("output_name", "model")
    python_path = body.get("python_path", "")

    if not pid:
        raise HTTPException(400, "缺少 project_id")
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    if not checkpoint:
        mode = body.get("mode", "pretrain")
        ckpts = list_checkpoints(_project_dir(pid), mode)
        if not ckpts:
            raise HTTPException(400, "没有可用的 checkpoint")
        checkpoint = ckpts[-1]["path"]

    pdir = _project_dir(pid)
    export_dir = pdir / "export"
    hf_dir = export_dir / "hf_model"
    script = script_export_hf(checkpoint, hf_dir.as_posix(), str(pdir))

    return StreamingResponse(
        run_streaming(script, python_path or None),
        media_type="text/event-stream",
    )


@router.post("/export/ollama")
async def export_ollama(request: Request):
    """导入模型到 Ollama"""
    body = await request.json()
    pid = body.get("project_id", "")
    model_name = body.get("model_name", "my-pretrained-model")
    system_prompt = body.get("system_prompt", "")
    gguf_path = body.get("gguf_path", "")

    if not pid:
        raise HTTPException(400, "缺少 project_id")
    meta = _load_project_meta(pid)
    if not meta:
        raise HTTPException(404, "项目不存在")

    return import_to_ollama(
        project_dir=str(_project_dir(pid)),
        gguf_path=gguf_path or None,
        model_name=model_name,
        system_prompt=system_prompt,
    )
