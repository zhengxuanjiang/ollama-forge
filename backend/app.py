"""Ollama GUI Backend v4.0 — Security + Concurrency + VEnv + Improved Fine-Tuning"""
import json, os, base64, secrets, time, socket, subprocess, sys, shutil, re, platform
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio, threading

import httpx
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Header, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# ======================= Constants =======================
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
FE = Path(__file__).parent.parent / "frontend"
KEYS_FILE = Path.home() / ".ollama-gui-keys.json"
CONFIG_FILE = Path(__file__).parent.parent / "config.json"

def _load_config():
    """Load persistent config from config.json."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {}

def _save_config(cfg):
    """Save config to config.json."""
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")

def _get_ft_dir():
    """Resolve FT_DIR: config.json > env var > default."""
    cfg = _load_config()
    p = cfg.get("ft_dir") or os.environ.get("OLLAMA_GUI_FT_DIR") or str(Path.home() / ".ollama-gui-finetune")
    return Path(p)

def _get_hf_home():
    """Resolve HF_HOME: config.json > env var > default."""
    cfg = _load_config()
    return cfg.get("hf_home") or os.environ.get("HF_HOME") or ""

# Initialize FT_DIR (global, can be updated at runtime via /api/config)
FT_DIR = _get_ft_dir()
FT_DIR.mkdir(parents=True, exist_ok=True)

# Set HF cache dir if configured (otherwise HF uses ~/.cache/huggingface)
_hf = _get_hf_home()
if _hf:
    os.environ["HF_HOME"] = _hf
    Path(_hf).mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Include pretrain lab router
from backend.pretrain.routes import router as pretrain_router
app.include_router(pretrain_router)

# ======================= Thread-Safe File Lock =======================
_file_lock = threading.Lock()

def _load():
    """Thread-safe load of keys data."""
    with _file_lock:
        if KEYS_FILE.exists():
            try:
                return json.loads(KEYS_FILE.read_text("utf-8"))
            except Exception:
                pass
        return {"keys": [], "sharing": False, "admin_token": ""}

def _save(data):
    """Thread-safe save of keys data (atomic write)."""
    with _file_lock:
        tmp = KEYS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(KEYS_FILE)

# ======================= Admin Auth =======================
def _ensure_admin_token():
    """Generate admin token on first run."""
    data = _load()
    if not data.get("admin_token"):
        data["admin_token"] = "oga-" + secrets.token_hex(24)
        _save(data)
    return data["admin_token"]

def _check_admin(request: Request):
    """Validate admin access: localhost bypasses, LAN requires token."""
    client_ip = request.client.host if request.client else ""
    # Localhost is trusted
    if client_ip in ("127.0.0.1", "::1", "localhost"):
        return True
    # LAN requires admin token
    auth = request.headers.get("X-Admin-Token", "")
    data = _load()
    if not auth or auth != data.get("admin_token", ""):
        raise HTTPException(403, "需要管理员权限。非本机访问请提供 X-Admin-Token")
    return True

# ======================= Rate Limit =======================
_rate_windows = defaultdict(list)
_rate_lock = threading.Lock()

# ======================= Input Sanitization =======================
_SAFE_NAME_RE = re.compile(r'^[a-zA-Z0-9_\-.:/ ]+$')

def _sanitize_name(name: str, label: str = "名称") -> str:
    """Sanitize model/file names to prevent injection."""
    name = name.strip()
    if not name:
        raise HTTPException(400, f"{label}不能为空")
    if len(name) > 200:
        raise HTTPException(400, f"{label}过长（最大200字符）")
    # Block path traversal
    if ".." in name or name.startswith("/") or name.startswith("\\"):
        raise HTTPException(400, f"{label}包含非法字符")
    return name

def _sanitize_model_name(name: str) -> str:
    """Stricter validation for model names used in subprocess calls."""
    name = name.strip().lower().replace(" ", "-")
    if not re.match(r'^[a-z0-9][a-z0-9_\-.:]*$', name):
        raise HTTPException(400, "模型名只能包含小写字母、数字、-_.: 且不能以特殊字符开头")
    if len(name) > 100:
        raise HTTPException(400, "模型名过长")
    return name

# ======================= Key Validation =======================
def _check_key(auth: str, model: str = None):
    """Validate API key + rate limit + quota + expiration + model whitelist."""
    data = _load()
    if not data.get("sharing"):
        raise HTTPException(403, "共享未开启")
    if not auth:
        raise HTTPException(401, "缺少 Authorization header")

    token = auth.replace("Bearer ", "").strip() if auth.startswith("Bearer ") else auth

    key = None
    for k in data["keys"]:
        if k["key"] == token and k.get("active", True):
            key = k
            break
    if not key:
        raise HTTPException(401, "无效或已吊销的 API Key")

    kid = key["id"]
    now = datetime.now()

    # Expiration
    if key.get("expires_at"):
        exp = datetime.fromisoformat(key["expires_at"])
        if now > exp:
            raise HTTPException(403, f"密钥已过期 ({key['expires_at']})")

    # Model whitelist
    allowed = key.get("allowed_models", [])
    if allowed and model and model not in allowed:
        raise HTTPException(403, f"此密钥不允许使用模型 {model}，允许: {', '.join(allowed)}")

    # Daily quota
    today = now.strftime("%Y-%m-%d")
    daily = key.get("daily_usage", {})
    used_today = daily.get(today, 0)
    limit = key.get("daily_limit", 0)
    if limit > 0 and used_today >= limit:
        raise HTTPException(429, f"已达今日配额 ({limit} 次)")

    # Rate limit (RPM) — thread safe
    rpm = key.get("rpm", 0)
    if rpm > 0:
        with _rate_lock:
            window = _rate_windows[kid]
            cutoff = time.time() - 60
            window[:] = [t for t in window if t > cutoff]
            if len(window) >= rpm:
                raise HTTPException(429, f"速率超限 ({rpm} 次/分钟)")
            window.append(time.time())

    # Increment usage
    if "daily_usage" not in key:
        key["daily_usage"] = {}
    key["daily_usage"][today] = used_today + 1
    key["total_usage"] = key.get("total_usage", 0) + 1

    # Clean old daily usage (keep 30 days)
    old_keys = [d for d in key.get("daily_usage", {})
                if d < (now - timedelta(days=30)).strftime("%Y-%m-%d")]
    for ok in old_keys:
        key["daily_usage"].pop(ok, None)

    _save(data)
    return key

# ======================= Key Management (Admin Protected) =======================
@app.get("/api/keys")
async def list_keys(request: Request):
    _check_admin(request)
    data = _load()
    keys = []
    today = datetime.now().strftime("%Y-%m-%d")
    for k in data.get("keys", []):
        keys.append({
            "id": k["id"], "name": k["name"], "active": k.get("active", True),
            "key_preview": k["key"][:12] + "..." + "*" * 16,
            "rpm": k.get("rpm", 0), "daily_limit": k.get("daily_limit", 0),
            "total_usage": k.get("total_usage", 0),
            "today_usage": k.get("daily_usage", {}).get(today, 0),
            "allowed_models": k.get("allowed_models", []),
            "expires_at": k.get("expires_at", ""),
            "created": k.get("created", ""),
        })
    return {"keys": keys, "sharing": data.get("sharing", False)}

@app.post("/api/keys")
async def create_key(request: Request):
    _check_admin(request)
    body = await request.json()
    name = _sanitize_name(body.get("name", "API Key"), "密钥名称")
    key = "ogk-" + secrets.token_hex(24)
    entry = {
        "id": secrets.token_hex(4), "name": name,
        "key": key, "active": True,
        "created": datetime.now().isoformat(),
        "rpm": max(0, int(body.get("rpm", 0))),
        "daily_limit": max(0, int(body.get("daily_limit", 0))),
        "allowed_models": body.get("allowed_models", []),
        "expires_at": body.get("expires_at", ""),
        "total_usage": 0, "daily_usage": {}
    }
    data = _load()
    data["keys"].append(entry)
    _save(data)
    return {"key": key, "id": entry["id"], "name": entry["name"]}

@app.put("/api/keys/{key_id}")
async def update_key(key_id: str, request: Request):
    _check_admin(request)
    body = await request.json()
    data = _load()
    for k in data["keys"]:
        if k["id"] == key_id:
            for f in ["name", "rpm", "daily_limit", "allowed_models", "expires_at", "active"]:
                if f in body:
                    k[f] = body[f]
            _save(data)
            return {"status": "ok"}
    raise HTTPException(404, "密钥未找到")

@app.delete("/api/keys/{key_id}")
async def revoke_key(key_id: str, request: Request):
    _check_admin(request)
    data = _load()
    data["keys"] = [k for k in data["keys"] if k["id"] != key_id]
    _save(data)
    return {"status": "ok"}

@app.post("/api/sharing")
async def toggle_sharing(request: Request):
    _check_admin(request)
    body = await request.json()
    data = _load()
    data["sharing"] = body.get("enabled", False)
    _save(data)
    return {"sharing": data["sharing"]}

def _get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

@app.get("/api/sharing-info")
async def sharing_info(request: Request):
    _check_admin(request)
    data = _load()
    ip = _get_ip()
    port = os.environ.get("PORT", "8765")
    return {
        "sharing": data.get("sharing", False),
        "lan_ip": ip, "port": port,
        "base_url": f"http://{ip}:{port}",
        "key_count": len([k for k in data.get("keys", []) if k.get("active", True)]),
        "admin_token": data.get("admin_token", "")
    }

# ======================= Ollama Proxy (GUI, no auth) =======================
async def _proxy(method, path, **kwargs):
    async with httpx.AsyncClient(timeout=300) as client:
        return await client.request(method, f"{OLLAMA}{path}", **kwargs)

@app.get("/api/tags")
async def tags():
    try:
        return (await _proxy("GET", "/api/tags")).json()
    except httpx.ConnectError:
        raise HTTPException(502, "Ollama 未运行")

@app.get("/api/ps")
async def ps():
    try:
        return (await _proxy("GET", "/api/ps")).json()
    except httpx.ConnectError:
        raise HTTPException(502, "Ollama 未运行")

@app.post("/api/show")
async def show(request: Request):
    try:
        return (await _proxy("POST", "/api/show", json=await request.json())).json()
    except httpx.ConnectError:
        raise HTTPException(502, "Ollama 未运行")

@app.post("/api/pull")
async def pull(request: Request):
    body = await request.json()
    async def gen():
        try:
            timeout = httpx.Timeout(connect=30, read=600, write=30, pool=30)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{OLLAMA}/api/pull", json={**body, "stream": True}) as resp:
                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'error': (await resp.aread()).decode()})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"
        except httpx.ConnectError:
            yield f'data: {json.dumps({"error": "连接失败"})}\n\n'
        except httpx.ReadTimeout:
            yield f'data: {json.dumps({"error": "超时，请重试（断点续传）"})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.delete("/api/delete")
async def delete_model(request: Request):
    try:
        await _proxy("DELETE", "/api/delete", json=await request.json())
        return {"status": "ok"}
    except httpx.ConnectError:
        raise HTTPException(502, "Ollama 未运行")

@app.post("/api/unload")
async def unload(request: Request):
    body = await request.json()
    try:
        await _proxy("POST", "/api/generate", json={
            "model": body["model"], "prompt": "", "keep_alive": 0
        })
        return {"status": "ok"}
    except httpx.ConnectError:
        raise HTTPException(502, "Ollama 未运行")

@app.post("/api/preload")
async def preload(request: Request):
    body = await request.json()
    try:
        await _proxy("POST", "/api/generate", json={
            "model": body["model"], "prompt": "",
            "keep_alive": body.get("keep_alive", "30m")
        })
        return {"status": "ok"}
    except httpx.ConnectError:
        raise HTTPException(502, "Ollama 未运行")

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    system_prompt = body.pop("system_prompt", None)
    think = body.pop("thinking_enabled", False)
    num_ctx = body.pop("num_ctx", 4096)
    temp = body.pop("temperature", 0.7)
    topp = body.pop("top_p", 0.9)
    num_gpu = body.pop("num_gpu", None)
    keep_alive = body.pop("keep_alive", "5m")
    repeat_penalty = body.pop("repeat_penalty", 1.1)
    seed = body.pop("seed", None)

    msgs = body.get("messages", [])
    if system_prompt and system_prompt.strip():
        msgs = [{"role": "system", "content": system_prompt}] + \
               [m for m in msgs if m.get("role") != "system"]

    opts = {"temperature": temp, "top_p": topp, "num_ctx": num_ctx, "repeat_penalty": repeat_penalty}
    if seed is not None and seed >= 0:
        opts["seed"] = seed
    if num_gpu is not None:
        opts["num_gpu"] = num_gpu

    payload = {
        "model": body.get("model", ""), "messages": msgs, "stream": True,
        "options": opts, "keep_alive": keep_alive, "think": think
    }

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA}/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'error': (await resp.aread()).decode()})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"
        except httpx.ConnectError:
            yield f'data: {json.dumps({"error": "Ollama 连接失败"})}\n\n'
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/upload-image")
async def upload_img(file: UploadFile = File(...)):
    content = await file.read()
    return {
        "filename": file.filename,
        "base64": base64.b64encode(content).decode(),
        "size": len(content)
    }

@app.get("/api/health")
async def health():
    try:
        resp = await _proxy("GET", "/api/tags")
        return {"status": "ok", "ollama": resp.status_code == 200}
    except Exception:
        return {"status": "ok", "ollama": False}

# ======================= OpenAI-Compatible (API Key Required) =======================
@app.get("/v1/models")
async def v1_models(authorization: str = Header(None)):
    _check_key(authorization)
    try:
        data = (await _proxy("GET", "/api/tags")).json()
        return {
            "object": "list",
            "data": [{"id": m["name"], "object": "model", "created": 0, "owned_by": "ollama"}
                     for m in data.get("models", [])]
        }
    except httpx.ConnectError:
        raise HTTPException(502, "Ollama 未运行")

@app.post("/v1/chat/completions")
async def v1_chat(request: Request, authorization: str = Header(None)):
    body = await request.json()
    model = body.get("model", "")
    _check_key(authorization, model)
    msgs = body.get("messages", [])
    stream = body.get("stream", False)
    opts = {"temperature": body.get("temperature", 0.7), "top_p": body.get("top_p", 0.9)}
    if body.get("max_tokens"):
        opts["num_predict"] = body["max_tokens"]
    payload = {
        "model": model, "messages": msgs, "stream": stream,
        "options": opts, "keep_alive": body.get("keep_alive", "5m")
    }

    if not stream:
        try:
            resp = await _proxy("POST", "/api/chat", json=payload)
            d = resp.json()
            return {
                "id": "chatcmpl-" + secrets.token_hex(6),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": d.get("message", {}), "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": d.get("prompt_eval_count", 0),
                    "completion_tokens": d.get("eval_count", 0),
                    "total_tokens": d.get("prompt_eval_count", 0) + d.get("eval_count", 0)
                }
            }
        except httpx.ConnectError:
            raise HTTPException(502, "Ollama 未运行")
    else:
        async def sgen():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{OLLAMA}/api/chat",
                                             json={**payload, "stream": True}) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                d = json.loads(line)
                                m = d.get("message", {})
                                chunk = {
                                    "id": "chatcmpl-x", "object": "chat.completion.chunk",
                                    "created": int(time.time()), "model": model,
                                    "choices": [{"index": 0, "delta": {"content": m.get("content", "")},
                                                 "finish_reason": None}]
                                }
                                yield f'data: {json.dumps(chunk)}\n\n'
                                if d.get("done"):
                                    done_chunk = {
                                        "id": "chatcmpl-x", "object": "chat.completion.chunk",
                                        "created": int(time.time()), "model": model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                                    }
                                    yield f'data: {json.dumps(done_chunk)}\n\n'
                            except Exception:
                                pass
            except httpx.ConnectError:
                yield f'data: {json.dumps({"error": "Ollama 连接失败"})}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(sgen(), media_type="text/event-stream")

# ======================= Settings / Config =======================
@app.get("/api/config")
async def get_config():
    """Return current config including all file paths."""
    cfg = _load_config()
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    # Normalize all paths to forward slashes for consistent display and to avoid escape issues
    ft = str(FT_DIR).replace("\\", "/")
    hf = hf_home.replace("\\", "/")
    return {
        "ft_dir": ft,
        "hf_home": hf,
        "paths": {
            "datasets":     str(FT_DIR / "datasets").replace("\\", "/"),
            "train_script": str(FT_DIR / "train_script.py").replace("\\", "/"),
            "outputs":      str(FT_DIR / "outputs").replace("\\", "/"),
            "merged_model": str(FT_DIR / "merged_model").replace("\\", "/"),
            "imports":      str(FT_DIR / "imports").replace("\\", "/"),
            "venv":         str(FT_DIR / "venv").replace("\\", "/"),
            "hf_cache":     hf,
        },
        "saved": cfg,  # raw saved config
    }

@app.post("/api/config")
async def set_config(request: Request):
    """Update config. Changes to ft_dir/hf_home take effect after restart."""
    global FT_DIR
    body = await request.json()
    cfg = _load_config()

    changed = []
    needs_restart = False

    if "ft_dir" in body and body["ft_dir"]:
        # Normalize path: resolve and convert to forward slashes to avoid escape issues
        new_dir = str(Path(body["ft_dir"]).resolve()).replace("\\", "/")
        old_dir = str(FT_DIR).replace("\\", "/")
        if new_dir != old_dir:
            # Validate: can we create/access this directory?
            try:
                Path(new_dir).mkdir(parents=True, exist_ok=True)
                (Path(new_dir) / "datasets").mkdir(exist_ok=True)
            except Exception as e:
                raise HTTPException(400, f"无法创建目录 {new_dir}: {e}")
            cfg["ft_dir"] = new_dir
            FT_DIR = Path(new_dir)
            changed.append("ft_dir")

    if "hf_home" in body and body["hf_home"]:
        new_hf = str(Path(body["hf_home"]).resolve()).replace("\\", "/")
        old_hf = os.environ.get("HF_HOME", "").replace("\\", "/")
        if new_hf != old_hf:
            try:
                Path(new_hf).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise HTTPException(400, f"无法创建目录 {new_hf}: {e}")
            cfg["hf_home"] = new_hf
            os.environ["HF_HOME"] = new_hf
            changed.append("hf_home")
            needs_restart = True  # HF libraries cache the path at import time

    _save_config(cfg)
    return {
        "status": "ok",
        "changed": changed,
        "ft_dir": str(FT_DIR),
        "hf_home": os.environ.get("HF_HOME", ""),
        "needs_restart": needs_restart,
        "message": "HF 缓存路径修改需要重启生效" if needs_restart else "已保存",
    }

# ======================= HuggingFace Account =======================
@app.post("/api/hf/status")
async def hf_status(request: Request):
    """Check HuggingFace login status."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    python_path = body.get("python_path") or None

    # Method 1: Try via huggingface_hub in the selected Python environment
    py = python_path or sys.executable
    try:
        cmd = "from huggingface_hub import HfApi; a=HfApi(); i=a.whoami(); print(i.get('name','')); print(i.get('fullname',''))"
        r = subprocess.run([py, "-c", cmd], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            lines = r.stdout.strip().split("\n")
            if lines and lines[0]:
                return {"logged_in": True, "username": lines[0], "fullname": lines[1] if len(lines) > 1 else ""}
    except Exception:
        pass

    # Method 2: Fallback — check token file directly (works even if huggingface_hub is not in system Python)
    for token_path in [
        Path.home() / ".cache" / "huggingface" / "token",
        Path(os.environ.get("HF_HOME", "")) / "token" if os.environ.get("HF_HOME") else None,
    ]:
        if token_path and token_path.exists():
            try:
                token = token_path.read_text("utf-8").strip()
                if token.startswith("hf_"):
                    return {"logged_in": True, "username": "(token file detected)", "fullname": ""}
            except Exception:
                pass

    return {"logged_in": False, "username": "", "fullname": ""}

@app.post("/api/hf/login")
async def hf_login(request: Request):
    """Login to HuggingFace with token."""
    body = await request.json()
    token = body.get("token", "").strip()
    python_path = body.get("python_path", sys.executable)
    if not token:
        raise HTTPException(400, "请输入 Token")
    if not token.startswith("hf_"):
        raise HTTPException(400, "Token 格式不正确，应以 hf_ 开头")

    # Method 1: Write token file directly (avoids [WinError 87] git credential issue on Windows)
    try:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        token_dir = Path(hf_home)
        token_dir.mkdir(parents=True, exist_ok=True)
        token_path = token_dir / "token"
        token_path.write_text(token, "utf-8")
    except Exception as e:
        return {"status": "error", "error": f"无法写入 token 文件: {e}"}

    # Method 2: Also try huggingface_hub login (non-critical, for full integration)
    try:
        cmd = (
            "import os; os.environ['HF_HUB_DISABLE_TELEMETRY']='1'; "
            "from huggingface_hub import login; "
            f"login(token='{token}', add_to_git_credential=False)"
        )
        subprocess.run([python_path, "-c", cmd], capture_output=True, text=True, timeout=30)
    except Exception:
        pass  # Token file was already written, this is just a bonus

    # Verify by checking whoami
    username = ""
    try:
        verify = 'from huggingface_hub import HfApi; i=HfApi().whoami(); print(i.get("name",""))'
        v = subprocess.run([python_path, "-c", verify], capture_output=True, text=True, timeout=15)
        if v.returncode == 0:
            username = v.stdout.strip()
    except Exception:
        username = "(token saved)"

    return {"status": "ok", "username": username or "(token saved)"}

@app.post("/api/hf/logout")
async def hf_logout(request: Request):
    """Logout from HuggingFace."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    python_path = body.get("python_path", sys.executable)
    try:
        subprocess.run([python_path, "-c", "from huggingface_hub import logout; logout()"],
                      capture_output=True, text=True, timeout=15)
    except Exception:
        pass
    # Remove token files from all possible locations
    for token_path in [
        Path.home() / ".cache" / "huggingface" / "token",
        Path(os.environ.get("HF_HOME", "")) / "token" if os.environ.get("HF_HOME") else None,
    ]:
        if token_path and token_path.exists():
            try:
                token_path.unlink()
            except Exception:
                pass
    return {"status": "ok"}

# ======================= Fine-Tuning System =======================
# Thread-safe training state
_train_lock = threading.Lock()
_train_state = {
    "status": "idle", "progress": 0, "logs": [],
    "error": "", "job_id": "", "config": {}, "pid": None,
    "start_time": None
}

def _get_train_state():
    with _train_lock:
        state = dict(_train_state)
        state["logs"] = list(_train_state["logs"])  # deep copy the mutable list
        return state

def _update_train_state(**kwargs):
    with _train_lock:
        _train_state.update(kwargs)

# ======================= Training History =======================
def _history_path():
    return FT_DIR / "training_history.json"

def _load_history():
    p = _history_path()
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            pass
    return []

def _save_history(records):
    with _file_lock:
        tmp = _history_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(_history_path())

def _add_history_record(record):
    records = _load_history()
    records.insert(0, record)  # newest first
    if len(records) > 100:
        records = records[:100]
    _save_history(records)

# ======================= Project-Based Training =======================
def _projects_dir():
    d = FT_DIR / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_project(name):
    pf = _projects_dir() / name / "project.json"
    if pf.exists():
        try: return json.loads(pf.read_text("utf-8"))
        except Exception: pass
    return None

def _save_project(name, data):
    pd = _projects_dir() / name
    pd.mkdir(parents=True, exist_ok=True)
    (pd / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

def _list_projects():
    pd = _projects_dir()
    projects = []
    if not pd.exists():
        return projects
    for d in sorted(pd.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "project.json").exists():
            try:
                info = json.loads((d / "project.json").read_text("utf-8"))
                info["dir_name"] = d.name
                out_dir = d / "outputs"
                cps = [x for x in out_dir.iterdir() if x.is_dir() and x.name.startswith("checkpoint-")] if out_dir.exists() else []
                info["checkpoint_count"] = len(cps)
                info["has_merged"] = (d / "merged_model").exists()
                projects.append(info)
            except Exception:
                pass
    return projects

def _project_checkpoints(project_name):
    out_dir = _projects_dir() / project_name / "outputs"
    checkpoints = []
    if out_dir.exists():
        for d in sorted(out_dir.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("checkpoint-"):
                try: step = int(d.name.split("-")[1])
                except (ValueError, IndexError): step = 0
                has_state = (d / "trainer_state.json").exists()
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
                info = {"name": d.name, "step": step, "path": str(d).replace("\\", "/"),
                        "size_mb": round(size_mb, 1)}
                if has_state:
                    try:
                        ts = json.loads((d / "trainer_state.json").read_text("utf-8"))
                        if ts.get("log_history"):
                            last = ts["log_history"][-1]
                            info["loss"] = last.get("loss") or last.get("train_loss")
                            info["epoch"] = last.get("epoch")
                    except Exception: pass
                checkpoints.append(info)
    return checkpoints

@app.get("/api/finetune/history")
async def get_history():
    return {"records": _load_history()}

@app.delete("/api/finetune/history/{record_id}")
async def delete_history_record(record_id: str):
    records = _load_history()
    records = [r for r in records if r.get("id") != record_id]
    _save_history(records)
    return {"status": "ok"}

# Project management endpoints
@app.get("/api/finetune/projects")
async def list_projects():
    return {"projects": _list_projects()}

@app.get("/api/finetune/projects/{name}/checkpoints")
async def get_project_checkpoints(name: str):
    p = _load_project(name)
    if not p:
        raise HTTPException(404, "项目不存在")
    return {"checkpoints": _project_checkpoints(name), "project": p}

@app.post("/api/finetune/projects/{name}/rename")
async def rename_project(name: str, request: Request):
    body = await request.json()
    new_name = re.sub(r'[^\w\-.]', '_', body.get("new_name", "").strip())
    if not new_name:
        raise HTTPException(400, "名称不能为空")
    src = _projects_dir() / name
    dst = _projects_dir() / new_name
    if not src.exists():
        raise HTTPException(404, "项目不存在")
    if dst.exists():
        raise HTTPException(409, "目标名称已存在")
    src.rename(dst)
    # Update project.json
    p = _load_project(new_name)
    if p:
        p["name"] = new_name
        _save_project(new_name, p)
    return {"status": "ok", "new_name": new_name}

@app.delete("/api/finetune/projects/{name}")
async def delete_project(name: str):
    pd = _projects_dir() / name
    if not pd.exists():
        raise HTTPException(404, "项目不存在")
    shutil.rmtree(pd, ignore_errors=True)
    return {"status": "ok"}

@app.post("/api/finetune/projects/{name}/resume")
async def resume_from_project(name: str, request: Request):
    """Resume training from a checkpoint — starts immediately."""
    body = await request.json()
    checkpoint_name = body.get("checkpoint", "")
    python_path = body.get("python_path", sys.executable)
    if not checkpoint_name:
        raise HTTPException(400, "请指定 checkpoint")

    state = _get_train_state()
    if state["status"] == "training":
        raise HTTPException(409, "已有训练任务在运行")

    p = _load_project(name)
    if not p:
        raise HTTPException(404, "项目不存在")

    project_dir = _projects_dir() / name
    cp_path = project_dir / "outputs" / checkpoint_name
    if not cp_path.exists():
        raise HTTPException(400, f"Checkpoint 不存在: {checkpoint_name}")

    # Rebuild config from project
    cfg = p.get("config", {})
    cfg["resume_from_checkpoint"] = str(cp_path).replace("\\", "/")

    job_id = secrets.token_hex(4)
    start_time = time.time()
    _update_train_state(
        status="training", progress=0, logs=["🔄 从断点恢复: " + checkpoint_name],
        error="", job_id=job_id, config=cfg, pid=None, last_loss=None,
        start_time=start_time, project_name=name
    )

    script = _gen_train_script(cfg)
    script_path = project_dir / "train_script.py"
    script_path.write_text(script, "utf-8")

    def run_train():
        _do_train(python_path, script_path, project_dir, cfg, job_id, start_time, name)
    threading.Thread(target=run_train, daemon=True).start()
    return {"status": "started", "job_id": job_id, "project": name}

# Legacy checkpoint endpoint (returns all across projects)
@app.get("/api/finetune/checkpoints")
async def list_checkpoints():
    """List checkpoints across all projects."""
    all_cps = []
    for p in _list_projects():
        pname = p["dir_name"]
        for cp in _project_checkpoints(pname):
            cp["project"] = pname
            cp["project_label"] = p.get("label", pname)
            all_cps.append(cp)
    return {"checkpoints": all_cps}

# ======================= Hardware Detection =======================
def _detect_hardware():
    info = {
        "gpus": [], "ram_gb": 0, "disk_free_gb": 0,
        "os": platform.system(), "cuda": False,
        "python": sys.version.split()[0]
    }
    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["disk_free_gb"] = round(psutil.disk_usage('/').free / 1e9, 1)
    except Exception:
        try:
            if platform.system() != "Windows":
                m = os.popen("free -b 2>/dev/null").read()
                if m:
                    info["ram_gb"] = round(int(m.split('\n')[1].split()[1]) / 1e9, 1)
        except Exception:
            pass
    # GPU via nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10
        )
        for line in out.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                info["gpus"].append({
                    "name": parts[0],
                    "vram_total_mb": int(parts[1]),
                    "vram_free_mb": int(parts[2]),
                    "driver": parts[3] if len(parts) > 3 else ""
                })
        info["cuda"] = True
    except Exception:
        pass
    # Check CUDA via torch
    if not info["cuda"]:
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 "import torch; print(torch.cuda.is_available()); "
                 "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                capture_output=True, text=True, timeout=15
            )
            if "True" in r.stdout:
                info["cuda"] = True
                nm = r.stdout.strip().split('\n')
                if len(nm) > 1 and nm[1]:
                    info["gpus"].append({"name": nm[1], "vram_total_mb": 0, "vram_free_mb": 0, "driver": ""})
        except Exception:
            pass
    # Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["apple_silicon"] = True
        try:
            m = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5)
            info["unified_memory_gb"] = round(int(m.strip()) / 1e9, 1)
        except Exception:
            pass
    return info

def _recommend_models(hw):
    vram = 0
    if hw.get("gpus"):
        vram = max(g.get("vram_total_mb", 0) for g in hw["gpus"])
    apple = hw.get("apple_silicon", False)
    um = hw.get("unified_memory_gb", 0)
    recs = []

    recs.append({
        "method": "modelfile", "title": "Modelfile 提示词定制",
        "desc": "通过系统提示词 + few-shot 示例创建定制模型，无需GPU，即时完成",
        "vram_need": "无要求", "models": ["所有已安装模型"],
        "difficulty": "⭐ 入门", "time": "< 1 分钟"
    })

    if apple and um >= 16:
        recs.append({
            "method": "qlora", "title": "QLoRA 微调 (Apple Silicon)",
            "desc": f"统一内存 {um}GB，可通过 MLX 框架微调",
            "vram_need": "≥16GB 统一内存",
            "models": ["Llama 3.2 1B/3B", "Qwen3 0.6B/1.7B", "Phi-4 Mini", "Gemma2 2B"],
            "difficulty": "⭐⭐⭐ 进阶", "time": "30分 - 数小时"
        })

    if vram >= 6000:
        models_3b = ["Qwen3 0.6B", "Llama 3.2 1B", "Phi-3 Mini 3.8B"]
        recs.append({
            "method": "qlora", "title": "QLoRA 4-bit 微调",
            "desc": f"GPU {vram}MB VRAM — 可微调小模型",
            "vram_need": "≥6GB", "models": models_3b,
            "difficulty": "⭐⭐⭐ 进阶", "time": "1-4 小时"
        })
    if vram >= 8000:
        recs[-1]["models"] = ["Qwen3 0.6B/1.7B", "Llama 3.2 1B/3B", "Phi-3 Mini", "Gemma2 2B"]
    if vram >= 12000:
        recs[-1]["models"] = ["Llama 3.2 3B", "Qwen3 4B", "Phi-4 Mini", "Gemma2 2B/9B"]
    if vram >= 16000:
        recs[-1]["models"] = ["Llama 3.1 8B", "Qwen3 8B", "Mistral 7B", "Gemma2 9B", "Phi-4"]
        recs[-1]["vram_need"] = "≥16GB"
    if vram >= 24000:
        recs[-1]["models"] = ["Llama 3.1 8B", "Qwen3 8B/14B", "Mistral 7B", "Gemma2 9B", "CodeLlama 13B"]
        recs[-1]["vram_need"] = "≥24GB"
        recs.append({
            "method": "lora", "title": "LoRA 16-bit 微调",
            "desc": "更高精度，训练质量更好",
            "vram_need": "≥24GB",
            "models": ["Llama 3.1 8B", "Qwen3 8B", "Mistral 7B"],
            "difficulty": "⭐⭐⭐⭐ 高级", "time": "2-8 小时"
        })
    if vram >= 48000 and recs:
        recs[-1]["models"] = ["Llama 3.1 8B/70B", "Qwen3 14B/30B", "Mixtral 8x7B"]

    recs.append({
        "method": "import", "title": "导入 GGUF / 适配器",
        "desc": "导入在其他环境训练好的 GGUF 模型或 LoRA 适配器",
        "vram_need": "无要求", "models": ["任何 GGUF 格式模型"],
        "difficulty": "⭐⭐ 简单", "time": "< 5 分钟"
    })
    return recs

@app.get("/api/hardware")
async def detect_hw():
    hw = _detect_hardware()
    recs = _recommend_models(hw)
    return {"hardware": hw, "recommendations": recs, "ft_dir": str(FT_DIR)}

# ======================= Virtual Environment Management =======================
@app.post("/api/venv/detect")
async def detect_python_envs():
    """Detect available Python environments on the system."""
    envs = []

    # Current interpreter
    envs.append({
        "name": "当前 Python",
        "path": sys.executable,
        "version": platform.python_version(),
        "type": "system"
    })

    # Common conda/venv locations
    search_dirs = []
    home = Path.home()

    # Conda envs
    for conda_dir in [home / "miniconda" / "envs", home / "anaconda3" / "envs",
                       home / "miniconda3" / "envs", Path("C:/") / "miniconda" / "envs",
                       Path("C:/") / "Users" / os.environ.get("USERNAME", "") / "miniconda" / "envs",
                       Path("E:/") / "miniconda" / "envs"]:
        if conda_dir.exists():
            for d in conda_dir.iterdir():
                if d.is_dir():
                    py = d / ("python.exe" if platform.system() == "Windows" else "bin/python")
                    if py.exists():
                        try:
                            r = subprocess.run([str(py), "--version"], capture_output=True, text=True, timeout=5)
                            ver = r.stdout.strip().replace("Python ", "") if r.returncode == 0 else "?"
                            envs.append({"name": f"conda: {d.name}", "path": str(py), "version": ver, "type": "conda"})
                        except Exception:
                            envs.append({"name": f"conda: {d.name}", "path": str(py), "version": "?", "type": "conda"})

    # Existing GUI venv
    gui_venv = FT_DIR / "venv"
    if gui_venv.exists():
        py = gui_venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
        if py.exists():
            try:
                r = subprocess.run([str(py), "--version"], capture_output=True, text=True, timeout=5)
                ver = r.stdout.strip().replace("Python ", "")
                envs.append({"name": "GUI 微调专用环境", "path": str(py), "version": ver, "type": "venv"})
            except Exception:
                pass

    return {"envs": envs}

@app.post("/api/venv/create")
async def create_venv(request: Request):
    """Create a virtual environment at specified path and install training deps."""
    body = await request.json()
    venv_path = body.get("path", str(FT_DIR / "venv"))
    base_python = body.get("base_python", sys.executable)

    async def gen():
        try:
            yield f'data: {json.dumps({"step": "📦 创建虚拟环境...", "phase": "venv"})}\n\n'
            r = subprocess.run(
                [base_python, "-m", "venv", venv_path],
                capture_output=True, text=True, timeout=120
            )
            if r.returncode != 0:
                yield f'data: {json.dumps({"step": "❌ 创建失败: " + r.stderr[:300], "error": True})}\n\n'
                yield "data: [DONE]\n\n"
                return

            py = os.path.join(venv_path, "Scripts" if platform.system() == "Windows" else "bin", "python")
            if platform.system() == "Windows":
                py += ".exe"

            yield f'data: {json.dumps({"step": "✅ 虚拟环境已创建", "phase": "venv"})}\n\n'
            yield f'data: {json.dumps({"step": "📦 升级 pip...", "phase": "pip"})}\n\n'
            subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"],
                         capture_output=True, text=True, timeout=120)

            # Install PyTorch with CUDA support first (pinned versions)
            cuda_ver = _detect_cuda_version()
            idx_url = _get_torch_cuda_index_url(cuda_ver)
            if idx_url:
                yield f'data: {json.dumps({"step": f"🔍 检测到 CUDA {cuda_ver}，安装 GPU 版 PyTorch...", "phase": "deps", "progress": 5})}\n\n'
                torch_cmd = [py, "-m", "pip", "install",
                             "torch==2.4.0", "torchvision==0.19.1", "torchaudio==2.4.0",
                             "--index-url", idx_url, "-q"]
            else:
                yield f'data: {json.dumps({"step": "📦 安装 PyTorch...", "phase": "deps", "progress": 5})}\n\n'
                torch_cmd = [py, "-m", "pip", "install",
                             "torch==2.4.0", "torchvision==0.19.1", "torchaudio==2.4.0", "-q"]
            r = subprocess.run(torch_cmd, capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                yield f'data: {json.dumps({"step": "✅ PyTorch", "phase": "deps", "progress": 20})}\n\n'
            else:
                yield f'data: {json.dumps({"step": f"⚠️ PyTorch: {r.stderr[:200]}", "phase": "deps"})}\n\n'

            # Install remaining packages with pinned versions (standard PEFT+TRL, no Unsloth)
            core_deps = [
                "numpy==1.26.4", "transformers==4.46.3", "datasets==3.2.0",
                "peft==0.13.2", "trl==0.12.2", "accelerate==1.2.1",
                "bitsandbytes==0.45.3", "sentencepiece==0.2.1", "protobuf==3.20.3",
                "huggingface_hub==0.27.1", "tokenizers==0.20.3", "safetensors==0.7.0",
            ]
            yield f'data: {json.dumps({"step": "📦 安装训练依赖 (PEFT+TRL)...", "phase": "deps", "progress": 30})}\n\n'
            r = subprocess.run(
                [py, "-m", "pip", "install"] + core_deps + ["-q"],
                capture_output=True, text=True, timeout=900
            )
            if r.returncode == 0:
                yield f'data: {json.dumps({"step": "✅ 训练依赖已安装", "phase": "deps", "progress": 95})}\n\n'
            else:
                yield f'data: {json.dumps({"step": f"⚠️ 依赖安装: {r.stderr[:200]}", "phase": "deps"})}\n\n'

            yield f'data: {json.dumps({"step": "✅ 环境配置完成！", "done": True, "python_path": py})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"step": f"❌ 错误: {str(e)}", "error": True})}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# ======================= Modelfile Creation =======================
@app.post("/api/finetune/modelfile")
async def create_modelfile(request: Request):
    body = await request.json()
    base = body.get("base_model", "")
    name = _sanitize_model_name(body.get("name", ""))
    system = body.get("system_prompt", "")
    params = body.get("parameters", {})
    examples = body.get("examples", [])

    if not base:
        raise HTTPException(400, "需要基础模型")

    lines = [f"FROM {base}"]
    if system:
        full_sys = system
        if examples:
            full_sys += "\n\n以下是示例：\n"
            for ex in examples:
                full_sys += f"\n用户: {ex.get('input', '')}\n助手: {ex.get('output', '')}\n"
        lines.append(f'SYSTEM """{full_sys}"""')
    for k, v in params.items():
        # Only allow known parameters
        if k in ("temperature", "top_p", "top_k", "repeat_penalty", "num_ctx", "seed"):
            lines.append(f"PARAMETER {k} {v}")

    mf_content = "\n".join(lines)
    mf_path = FT_DIR / f"Modelfile-{name}"
    mf_path.write_text(mf_content, "utf-8")

    try:
        r = subprocess.run(
            ["ollama", "create", name, "-f", str(mf_path)],
            capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace'
        )
        if r.returncode != 0:
            return {"status": "error", "error": r.stderr or r.stdout or "创建失败"}
        return {"status": "ok", "model_name": name, "modelfile": mf_content}
    except FileNotFoundError:
        return {"status": "error", "error": "找不到 ollama 命令，请确认 Ollama 已安装"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "创建超时"}

# ======================= Dataset Management =======================
@app.post("/api/finetune/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    fname = _sanitize_name(file.filename or "dataset", "文件名")
    ext = Path(fname).suffix.lower()

    if ext not in (".jsonl", ".json", ".csv"):
        raise HTTPException(400, f"不支持的格式: {ext}，支持 .jsonl / .json / .csv")

    save_path = FT_DIR / "datasets"
    save_path.mkdir(exist_ok=True)
    fpath = save_path / fname
    fpath.write_bytes(content)

    # Parse and validate
    preview = []
    total = 0
    fmt = "unknown"
    errors = []
    try:
        text = content.decode("utf-8")
        if ext == ".jsonl":
            fmt = "jsonl"
            for i, line in enumerate(text.strip().split('\n')):
                if not line.strip():
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                    if i < 5:
                        preview.append(obj)
                except Exception:
                    errors.append(f"行 {i + 1}: JSON解析失败")
        elif ext == ".json":
            data = json.loads(text)
            if isinstance(data, list):
                fmt = "json_array"
                total = len(data)
                preview = data[:5]
            elif isinstance(data, dict) and "data" in data:
                fmt = "json_data"
                total = len(data["data"])
                preview = data["data"][:5]
            else:
                errors.append("JSON格式无法识别，需要数组或含 data 字段的对象")
        elif ext == ".csv":
            fmt = "csv"
            lines_list = text.strip().split('\n')
            if lines_list:
                headers = lines_list[0].split(',')
                total = len(lines_list) - 1
                for line in lines_list[1:6]:
                    vals = line.split(',')
                    preview.append(dict(zip(headers, vals)))
    except Exception as e:
        errors.append(str(e))

    # Detect dataset format
    detected_format = "unknown"
    if preview:
        sample = preview[0]
        if isinstance(sample, dict):
            keys = set(sample.keys())
            if {"instruction", "output"} & keys:
                detected_format = "alpaca"
            elif "conversations" in keys:
                detected_format = "sharegpt"
            elif "messages" in keys:
                detected_format = "openai"
            elif {"input", "output"} & keys:
                detected_format = "input_output"
            elif {"question", "answer"} & keys:
                detected_format = "qa"
            elif {"prompt", "completion"} & keys:
                detected_format = "completion"
            elif {"text"} & keys:
                detected_format = "text_only"

    return {
        "status": "ok", "filename": fname, "path": str(fpath),
        "format": fmt, "detected_format": detected_format,
        "total": total, "preview": preview[:5], "errors": errors,
        "size": len(content)
    }

@app.get("/api/finetune/datasets")
async def list_datasets():
    dp = FT_DIR / "datasets"
    if not dp.exists():
        return {"datasets": []}
    ds = []
    for f in dp.iterdir():
        if f.is_file():
            info = {
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "format": "unknown",
                "rows": 0,
                "sample": None,
            }
            # Detect format and count rows
            try:
                if f.suffix in (".jsonl", ".json"):
                    lines = f.read_text("utf-8", errors="replace").strip().split("\n")
                    # Handle JSON arrays (single-line [...])
                    if len(lines) == 1 and lines[0].strip().startswith("["):
                        try:
                            arr = json.loads(lines[0])
                            info["rows"] = len(arr)
                            if arr:
                                first = arr[0]
                        except Exception:
                            info["rows"] = 1
                            first = json.loads(lines[0]) if lines else {}
                    else:
                        info["rows"] = len(lines)
                        first = json.loads(lines[0]) if lines else {}
                    # Detect format from first row
                    if isinstance(first, dict):
                        keys = set(first.keys())
                        if {"instruction", "output"}.issubset(keys) or {"instruction", "response"}.issubset(keys):
                            info["format"] = "alpaca"
                        elif "conversations" in keys:
                            info["format"] = "sharegpt"
                        elif "messages" in keys:
                            info["format"] = "openai"
                        elif "text" in keys:
                            info["format"] = "text"
                        else:
                            info["format"] = "other"
                        # Provide a truncated sample
                        sample_str = json.dumps(first, ensure_ascii=False)
                        info["sample"] = sample_str[:500] + ("..." if len(sample_str) > 500 else "")
                        info["columns"] = sorted(keys)
                elif f.suffix == ".csv":
                    lines = f.read_text("utf-8", errors="replace").strip().split("\n")
                    info["rows"] = max(0, len(lines) - 1)  # subtract header
                    info["format"] = "csv"
                    if len(lines) >= 2:
                        info["columns"] = [c.strip().strip('"') for c in lines[0].split(",")]
                        info["sample"] = lines[1][:500]
            except Exception:
                pass
            ds.append(info)
    return {"datasets": sorted(ds, key=lambda x: x["modified"], reverse=True)}

@app.delete("/api/finetune/datasets/{name}")
async def delete_dataset(name: str):
    # Sanitize to prevent path traversal
    name = Path(name).name  # Strip any directory components
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, "非法文件名")
    fp = FT_DIR / "datasets" / name
    if fp.exists():
        fp.unlink()
    return {"status": "ok"}

# ======================= HuggingFace Dataset Download =======================

def _ensure_pyarrow():
    """Ensure pyarrow is available for reading parquet files."""
    try:
        import pyarrow.parquet
        return True
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pyarrow", "-q"],
                capture_output=True, timeout=120
            )
            import pyarrow.parquet
            return True
        except Exception:
            return False

async def _stream_download_file(client, url, dest_path, headers=None, on_progress=None):
    """Download a file with streaming and progress callback. Returns total bytes."""
    async with client.stream("GET", url, headers=headers or {}) as resp:
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if on_progress and total > 0:
                    on_progress(downloaded, total)
    return downloaded

@app.post("/api/finetune/download-dataset")
async def download_dataset(request: Request):
    """Download dataset from HuggingFace and save locally (full download via parquet)."""
    body = await request.json()
    url = body.get("url", "").strip()
    hf_id = body.get("hf_id", "").strip()
    filename = _sanitize_name(body.get("filename", "dataset.jsonl"), "文件名")
    fmt = body.get("format", "auto")

    # Resolve repo_id from URL or direct ID
    repo_id = ""
    if hf_id:
        repo_id = hf_id.strip("/")
    elif url:
        if "/datasets/" in url:
            repo_id = url.split("/datasets/")[-1].split("?")[0].split("/tree/")[0].split("/viewer")[0].rstrip("/")
        elif url.endswith((".json", ".jsonl", ".csv", ".parquet")):
            repo_id = ""
        else:
            raise HTTPException(400, "无法解析数据集链接，请使用 HuggingFace 数据集链接或直接输入数据集 ID")
    else:
        raise HTTPException(400, "请提供数据集链接或 HuggingFace 数据集 ID")

    save_path = FT_DIR / "datasets"
    save_path.mkdir(exist_ok=True)

    async def gen():
        try:
            if not repo_id:
                # Direct file download
                yield f'data: {json.dumps({"step": "📥 直接下载文件...", "progress": 5})}\n\n'
                async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0), follow_redirects=True) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        yield f'data: {json.dumps({"step": f"❌ HTTP {resp.status_code}", "error": True})}\n\n'
                        return
                    fpath = save_path / filename
                    fpath.write_bytes(resp.content)
                    yield f'data: {json.dumps({"step": f"✅ 已保存 {filename} ({len(resp.content)/1024:.1f}KB)", "done": True, "filename": filename})}\n\n'
                return

            # ===== HuggingFace Dataset — Parquet Bulk Download =====
            yield f'data: {json.dumps({"step": f"🔍 查询数据集信息: {repo_id}...", "progress": 3})}\n\n'

            # Load HF token for gated datasets
            _hf_token = None
            for _tp in [
                os.path.join(os.environ.get("HF_HOME", ""), "token"),
                os.path.expanduser("~/.cache/huggingface/token"),
                os.path.expanduser("~/.huggingface/token"),
            ]:
                if _tp and os.path.isfile(_tp):
                    try:
                        _hf_token = open(_tp, encoding="utf-8").read().strip()
                        if _hf_token: break
                    except: pass
            if not _hf_token:
                _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

            headers = {}
            if _hf_token:
                headers["Authorization"] = f"Bearer {_hf_token}"

            # Use a long timeout for large file downloads
            async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0), follow_redirects=True) as client:

                # Step 1: Discover configs/splits
                config_name = "default"
                split_name = "train"
                num_rows = 0

                info_url = f"https://datasets-server.huggingface.co/info?dataset={repo_id}"
                try:
                    info_resp = await client.get(info_url, headers=headers)
                    if info_resp.status_code == 200:
                        info_data = info_resp.json()
                        ds_info = info_data.get("dataset_info", {})
                        if ds_info:
                            configs = list(ds_info.keys())
                            config_name = configs[0] if configs else "default"
                            splits = ds_info.get(config_name, {}).get("splits", {})
                            if "train" in splits:
                                split_name = "train"
                            elif splits:
                                split_name = list(splits.keys())[0]
                            num_rows = splits.get(split_name, {}).get("num_examples", 0)
                except Exception:
                    pass

                yield f'data: {json.dumps({"step": f"📋 配置: {config_name}/{split_name}" + (f" ({num_rows} 条)" if num_rows else ""), "progress": 8})}\n\n'

                # Step 2: Get parquet file URLs
                yield f'data: {json.dumps({"step": "📦 获取 Parquet 文件列表...", "progress": 10})}\n\n'

                parquet_files = []
                parquet_url = f"https://datasets-server.huggingface.co/parquet?dataset={repo_id}"
                try:
                    pq_resp = await client.get(parquet_url, headers=headers)
                    if pq_resp.status_code == 200:
                        pq_data = pq_resp.json()
                        all_pq = pq_data.get("parquet_files", [])
                        # Filter by config and split (prefer train)
                        for pf in all_pq:
                            if pf.get("config") == config_name and pf.get("split") == split_name:
                                parquet_files.append(pf)
                        # If nothing found for default config, try the first available config
                        if not parquet_files and all_pq:
                            # Find available configs
                            available_configs = list({pf.get("config") for pf in all_pq})
                            if available_configs:
                                config_name = available_configs[0]
                                available_splits = [pf.get("split") for pf in all_pq if pf.get("config") == config_name]
                                split_name = "train" if "train" in available_splits else (available_splits[0] if available_splits else "train")
                                parquet_files = [pf for pf in all_pq if pf.get("config") == config_name and pf.get("split") == split_name]
                                if parquet_files:
                                    yield f'data: {json.dumps({"step": f"📋 使用配置: {config_name}/{split_name}", "progress": 12})}\n\n'
                except Exception as e:
                    yield f'data: {json.dumps({"step": f"⚠️ Parquet 列表获取失败: {e}", "progress": 12})}\n\n'

                if not parquet_files:
                    yield f'data: {json.dumps({"step": "❌ 未找到可下载的 Parquet 文件，请检查数据集 ID 是否正确", "error": True})}\n\n'
                    return

                total_size = sum(pf.get("size", 0) for pf in parquet_files)
                size_mb = total_size / 1024 / 1024 if total_size else 0
                yield f'data: {json.dumps({"step": f"📦 找到 {len(parquet_files)} 个 Parquet 文件" + (f" ({size_mb:.1f}MB)" if size_mb > 0 else ""), "progress": 15})}\n\n'

                # Step 3: Ensure pyarrow is available
                if not _ensure_pyarrow():
                    yield f'data: {json.dumps({"step": "❌ 无法安装 pyarrow，请手动运行: pip install pyarrow", "error": True})}\n\n'
                    return

                # Step 4: Download parquet files
                import tempfile
                tmp_dir = tempfile.mkdtemp(prefix="ds_parquet_")
                downloaded_files = []
                total_downloaded = 0

                try:
                    for i, pf in enumerate(parquet_files):
                        pf_url = pf.get("url", "")
                        pf_name = pf.get("filename", f"part_{i}.parquet")
                        pf_size = pf.get("size", 0)
                        dest = os.path.join(tmp_dir, pf_name)

                        file_label = f"({i+1}/{len(parquet_files)})" if len(parquet_files) > 1 else ""
                        yield f'data: {json.dumps({"step": f"📥 下载 {pf_name} {file_label}...", "progress": 15 + int(i / len(parquet_files) * 55)})}\n\n'

                        _last_report = [time.time()]
                        _progress_msg = [None]

                        def _on_progress(dl, tot, _i=i):
                            now = time.time()
                            if now - _last_report[0] > 1.0:
                                pct_file = int(dl / tot * 100) if tot else 0
                                overall = 15 + int((_i + dl / max(tot, 1)) / len(parquet_files) * 55)
                                _progress_msg[0] = (pct_file, overall, dl, tot)
                                _last_report[0] = now

                        try:
                            nbytes = await _stream_download_file(client, pf_url, dest, headers=headers, on_progress=_on_progress)
                            downloaded_files.append(dest)
                            total_downloaded += nbytes
                        except Exception as e:
                            yield f'data: {json.dumps({"step": f"❌ 下载 {pf_name} 失败: {e}", "error": True})}\n\n'
                            return

                    yield f'data: {json.dumps({"step": f"✅ 下载完成 ({total_downloaded/1024/1024:.1f}MB)，正在转换...", "progress": 72})}\n\n'

                    # Step 5: Read parquet and convert to JSONL
                    import pyarrow.parquet as pq_mod

                    all_rows = []
                    for fi, fpath_pq in enumerate(downloaded_files):
                        yield f'data: {json.dumps({"step": f"🔄 读取 Parquet 文件 ({fi+1}/{len(downloaded_files)})...", "progress": 72 + int(fi / max(len(downloaded_files), 1) * 15)})}\n\n'
                        try:
                            table = pq_mod.read_table(fpath_pq)
                            # Convert to list of dicts
                            batch_rows = table.to_pydict()
                            n = table.num_rows
                            columns = list(batch_rows.keys())
                            for row_i in range(n):
                                row = {}
                                for col in columns:
                                    val = batch_rows[col][row_i]
                                    # Convert pyarrow types to native Python
                                    if hasattr(val, 'as_py'):
                                        val = val.as_py()
                                    row[col] = val
                                all_rows.append(row)
                        except Exception as e:
                            yield f'data: {json.dumps({"step": f"⚠️ 读取文件 {fi+1} 失败: {e}", "progress": 80})}\n\n'

                    if not all_rows:
                        yield f'data: {json.dumps({"step": "❌ 未读取到任何数据", "error": True})}\n\n'
                        return

                    yield f'data: {json.dumps({"step": f"🔄 处理数据格式 ({len(all_rows)} 条)...", "progress": 90})}\n\n'

                    # Step 6: Auto-detect and convert format
                    converted = _convert_dataset_rows(all_rows, fmt)
                    fname = filename if filename.endswith(".jsonl") else filename.rsplit(".", 1)[0] + ".jsonl"

                    if not converted:
                        content = "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows)
                        yield f'data: {json.dumps({"step": "⚠️ 未能自动识别格式，已保存原始数据", "progress": 95})}\n\n'
                    else:
                        content = "\n".join(json.dumps(r, ensure_ascii=False) for r in converted["rows"])
                        cfmt = converted["format"]
                        is_standard = cfmt in ("alpaca", "sharegpt", "openai")
                        msg = f"✅ 格式: {cfmt}" + (" (标准格式，无需转换)" if is_standard else " (已自动转换)")
                        yield f'data: {json.dumps({"step": msg, "progress": 95})}\n\n'

                    fpath = save_path / fname
                    fpath.write_text(content, "utf-8")
                    size_kb = len(content.encode("utf-8")) / 1024
                    yield f'data: {json.dumps({"step": f"✅ 下载完成！共 {len(all_rows)} 条 · {size_kb:.1f}KB · 已保存为 {fname}", "done": True, "filename": fname, "size": int(size_kb*1024), "rows": len(all_rows)})}\n\n'

                finally:
                    # Cleanup temp parquet files
                    try:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass

        except Exception as e:
            yield f'data: {json.dumps({"step": f"❌ {str(e)}", "error": True})}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _convert_dataset_rows(rows, target_fmt="auto"):
    """Auto-detect dataset format. Preserve if already standard; convert otherwise."""
    if not rows:
        return None
    sample = rows[0]
    keys = set(sample.keys()) if isinstance(sample, dict) else set()

    # ===== Already in standard training format — preserve as-is =====
    if {"instruction", "output"}.issubset(keys):
        return {"format": "alpaca", "rows": rows}
    if "conversations" in keys:
        # Validate it's real ShareGPT format
        convs = sample.get("conversations")
        if isinstance(convs, list) and len(convs) > 0 and isinstance(convs[0], dict):
            return {"format": "sharegpt", "rows": rows}
    if "messages" in keys:
        msgs = sample.get("messages")
        if isinstance(msgs, list) and len(msgs) > 0 and isinstance(msgs[0], dict):
            return {"format": "openai", "rows": rows}

    # ===== Try to detect and convert multi-turn conversation formats =====
    # Some datasets use "dialog"/"dialogue" fields
    for conv_key in ("dialog", "dialogue", "turns", "chat"):
        if conv_key in keys:
            val = sample.get(conv_key)
            if isinstance(val, list) and len(val) > 0:
                # Convert to ShareGPT format
                converted = []
                for r in rows:
                    turns = r.get(conv_key, [])
                    convs = []
                    for i, t in enumerate(turns):
                        if isinstance(t, dict):
                            # Has role/content structure
                            role = t.get("role") or t.get("from") or ("human" if i % 2 == 0 else "gpt")
                            content = t.get("content") or t.get("value") or t.get("text") or str(t)
                            convs.append({"from": role, "value": content})
                        elif isinstance(t, str):
                            convs.append({"from": "human" if i % 2 == 0 else "gpt", "value": t})
                    if convs:
                        converted.append({"conversations": convs})
                if converted:
                    return {"format": "sharegpt", "rows": converted}

    # ===== Try common Q&A field mappings → Alpaca =====
    field_maps = [
        (("question", "answer"), ("instruction", "output")),
        (("prompt", "completion"), ("instruction", "output")),
        (("prompt", "response"), ("instruction", "output")),
        (("query", "response"), ("instruction", "output")),
        (("human", "assistant"), ("instruction", "output")),
        (("user", "assistant"), ("instruction", "output")),
        (("input", "output"), ("instruction", "output")),
        (("context", "response"), ("instruction", "output")),
        (("question", "best_answer"), ("instruction", "output")),
    ]

    for src_fields, dst_fields in field_maps:
        if all(f in keys for f in src_fields):
            converted = []
            for r in rows:
                converted.append({
                    dst_fields[0]: str(r.get(src_fields[0], "")),
                    "input": "",
                    dst_fields[1]: str(r.get(src_fields[1], ""))
                })
            return {"format": "alpaca", "rows": converted}

    # ===== Single text field — try to detect structure =====
    if "text" in keys and len(keys) <= 3:
        sample_text = str(sample.get("text", ""))
        # Check if text contains conversation markers
        if any(m in sample_text for m in ["<|im_start|>", "<|user|>", "### Human:", "### Assistant:", "USER:", "ASSISTANT:"]):
            # Multi-turn in text — just preserve raw
            return {"format": "raw_text (内含对话)", "rows": rows}
        # Plain instruction text
        converted = []
        for r in rows:
            converted.append({"instruction": str(r.get("text", "")), "input": "", "output": ""})
        return {"format": "alpaca (从text转换)", "rows": converted}

    # Unknown format — return rows unchanged
    return {"format": "原始格式", "rows": rows}

# ======================= CUDA Version Detection =======================
def _detect_cuda_version():
    """Detect system CUDA version from nvidia-smi or nvcc."""
    # Try nvidia-smi first (works even without CUDA toolkit)
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], text=True, timeout=10, stderr=subprocess.DEVNULL
        )
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            return m.group(1)  # e.g. "12.4"
    except Exception:
        pass
    # Try nvcc
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], text=True, timeout=10, stderr=subprocess.DEVNULL
        )
        m = re.search(r"release ([\d.]+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None

def _get_torch_cuda_index_url(cuda_ver: str = None):
    """Get the correct PyTorch index URL for system CUDA version."""
    if not cuda_ver:
        cuda_ver = _detect_cuda_version()
    if not cuda_ver:
        return None  # Can't determine, will fallback to default

    major_minor = cuda_ver.split(".")
    ver = float(f"{major_minor[0]}.{major_minor[1]}" if len(major_minor) >= 2 else major_minor[0])

    # Map CUDA version to PyTorch index URL
    # PyTorch supports cu118, cu121, cu124, cu126
    if ver >= 12.6:
        return "https://download.pytorch.org/whl/cu126"
    elif ver >= 12.4:
        return "https://download.pytorch.org/whl/cu124"
    elif ver >= 12.1:
        return "https://download.pytorch.org/whl/cu121"
    elif ver >= 11.8:
        return "https://download.pytorch.org/whl/cu118"
    else:
        return "https://download.pytorch.org/whl/cu118"  # Best effort

# ======================= Dependency Check =======================
@app.post("/api/finetune/check-deps")
async def check_deps(request: Request):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    python_path = body.get("python_path", sys.executable)

    deps = {}
    for pkg in ["torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "accelerate"]:
        try:
            r = subprocess.run(
                [python_path, "-c", f"import {pkg}; print({pkg}.__version__)"],
                capture_output=True, text=True, timeout=15
            )
            deps[pkg] = r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            deps[pkg] = None

    # Detailed GPU / PyTorch CUDA diagnosis
    gpu_ok = False
    gpu_msg = ""
    torch_has_cuda = False
    torch_cuda_ver = ""
    system_cuda_ver = _detect_cuda_version()
    fix_cmd = ""

    if deps.get("torch"):
        # Torch is installed — check if it has CUDA support
        try:
            r = subprocess.run(
                [python_path, "-c",
                 "import torch; "
                 "print(torch.cuda.is_available()); "
                 "print(torch.version.cuda or 'none'); "
                 "print('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'no_mps'); "
                 "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"],
                capture_output=True, text=True, timeout=15
            )
            lines = r.stdout.strip().split('\n')
            cuda_avail = lines[0].strip() == "True" if lines else False
            torch_cuda_ver = lines[1].strip() if len(lines) > 1 else "none"
            mps_avail = len(lines) > 2 and lines[2].strip() == "mps"
            gpu_name = lines[3].strip() if len(lines) > 3 else ""

            if cuda_avail:
                gpu_ok = True
                gpu_msg = f"✅ CUDA 可用 — {gpu_name} (PyTorch CUDA {torch_cuda_ver})"
                torch_has_cuda = True
            elif mps_avail:
                gpu_ok = True
                gpu_msg = "✅ MPS (Apple Silicon) 可用"
            elif torch_cuda_ver == "none" or torch_cuda_ver == "None":
                # PyTorch installed WITHOUT CUDA — this is the user's exact problem
                idx_url = _get_torch_cuda_index_url(system_cuda_ver)
                if system_cuda_ver and idx_url:
                    fix_cmd = f"pip install torch torchvision torchaudio --index-url {idx_url}"
                    gpu_msg = (f"⚠️ PyTorch 已安装但是 CPU 版本！你的系统有 CUDA {system_cuda_ver}，"
                              f"但当前 PyTorch 没有 CUDA 支持。点击「修复 PyTorch」重新安装 GPU 版本")
                else:
                    gpu_msg = "⚠️ PyTorch 已安装但是 CPU 版本，且未检测到系统 CUDA。请安装 NVIDIA 驱动和 CUDA Toolkit"
            else:
                # Torch has CUDA compiled in but can't find device
                gpu_msg = (f"⚠️ PyTorch 有 CUDA {torch_cuda_ver} 支持，但未检测到可用 GPU 设备。"
                          f"请检查 NVIDIA 驱动是否正常")
        except Exception as e:
            gpu_msg = f"⚠️ PyTorch GPU 状态检测失败: {str(e)[:100]}"
    else:
        # Torch not installed at all
        if system_cuda_ver:
            gpu_msg = f"PyTorch 未安装。检测到系统 CUDA {system_cuda_ver}，安装依赖时将自动安装 GPU 版"
        else:
            gpu_msg = "PyTorch 未安装，未检测到 CUDA"

    return {
        "deps": deps,
        "all_ok": all(v is not None for v in deps.values()),
        "gpu_ok": gpu_ok,
        "gpu_msg": gpu_msg,
        "torch_has_cuda": torch_has_cuda,
        "system_cuda": system_cuda_ver or "",
        "fix_cmd": fix_cmd,
        "python_path": python_path,
    }

@app.post("/api/finetune/install-deps")
async def install_deps(request: Request):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    python_path = body.get("python_path", sys.executable)

    async def gen():
        sys_break = "venv" not in python_path and "conda" not in python_path

        # --- Step 1: Install PyTorch with correct CUDA support ---
        cuda_ver = _detect_cuda_version()
        idx_url = _get_torch_cuda_index_url(cuda_ver)
        if idx_url:
            yield f'data: {json.dumps({"step": f"🔍 检测到系统 CUDA {cuda_ver}，安装 GPU 版 PyTorch...", "progress": 5})}\n\n'
            cmd = [python_path, "-m", "pip", "install",
                   "torch==2.4.0", "torchvision==0.19.1", "torchaudio==2.4.0",
                   "--index-url", idx_url, "-q"]
        else:
            yield f'data: {json.dumps({"step": "⚠️ 未检测到 CUDA，安装默认 PyTorch（可能为 CPU 版）...", "progress": 5})}\n\n'
            cmd = [python_path, "-m", "pip", "install", "torch==2.4.0", "torchvision==0.19.1", "torchaudio==2.4.0", "-q"]
        if sys_break:
            cmd.append("--break-system-packages")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                yield f'data: {json.dumps({"step": "✅ PyTorch 已安装", "progress": 30})}\n\n'
            else:
                yield f'data: {json.dumps({"step": f"⚠️ PyTorch: {r.stderr[:300]}", "progress": 30})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"step": f"❌ PyTorch: {str(e)}", "progress": 30})}\n\n'

        # --- Step 2: Install core training dependencies ---
        yield f'data: {json.dumps({"step": "📦 安装训练依赖 (PEFT + TRL)...", "progress": 40})}\n\n'
        core_deps = [
            "numpy==1.26.4",
            "transformers==4.46.3",
            "datasets==3.2.0",
            "peft==0.13.2",
            "trl==0.12.2",
            "accelerate==1.2.1",
            "bitsandbytes==0.45.3",
            "sentencepiece==0.2.1",
            "protobuf==3.20.3",
            "huggingface_hub==0.27.1",
            "tokenizers==0.20.3",
            "safetensors==0.7.0",
        ]
        try:
            cmd = [python_path, "-m", "pip", "install"] + core_deps + ["-q"]
            if sys_break:
                cmd.append("--break-system-packages")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                yield f'data: {json.dumps({"step": "✅ 训练依赖已安装", "progress": 80})}\n\n'
            else:
                yield f'data: {json.dumps({"step": f"⚠️ 依赖安装: {r.stderr[:300]}", "progress": 80})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"step": f"⚠️ 依赖安装: {str(e)}", "progress": 80})}\n\n'

        # --- Step 3: Verify installation ---
        yield f'data: {json.dumps({"step": "🔍 验证安装...", "progress": 90})}\n\n'
        try:
            verify_cmd = [
                python_path, "-c",
                "import torch, transformers, peft, trl; "
                "print(f'PyTorch: {torch.__version__}'); "
                "print(f'CUDA: {torch.cuda.is_available()}'); "
                "print(f'PEFT: {peft.__version__}'); "
                "print(f'TRL: {trl.__version__}')"
            ]
            r = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                yield f'data: {json.dumps({"step": f"✅ 验证成功! {r.stdout.strip()}", "progress": 100})}\n\n'
            else:
                yield f'data: {json.dumps({"step": f"⚠️ 验证结果: {r.stdout} {r.stderr}", "progress": 100})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"step": f"⚠️ 验证失败: {str(e)}", "progress": 100})}\n\n'

        yield f'data: {json.dumps({"step": "✅ 全部完成（已使用标准 PEFT+TRL，无需 Unsloth）", "done": True})}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/finetune/fix-torch")
async def fix_torch(request: Request):
    """Reinstall PyTorch with GPU (CUDA) support."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    python_path = body.get("python_path", sys.executable)

    async def gen():
        cuda_ver = _detect_cuda_version()
        idx_url = _get_torch_cuda_index_url(cuda_ver)
        if not idx_url:
            yield f'data: {json.dumps({"step": "❌ 未检测到 NVIDIA GPU / CUDA 驱动。请先安装 NVIDIA 驱动", "error": True})}\n\n'
            yield "data: [DONE]\n\n"
            return

        yield f'data: {json.dumps({"step": f"🔍 检测到 CUDA {cuda_ver}，卸载 CPU 版 PyTorch...", "progress": 10})}\n\n'
        subprocess.run(
            [python_path, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
            capture_output=True, text=True, timeout=120
        )

        yield f'data: {json.dumps({"step": f"📦 安装 GPU 版 PyTorch (CUDA)...", "progress": 30})}\n\n'
        cmd = [python_path, "-m", "pip", "install",
               "torch==2.4.0", "torchvision==0.19.1", "torchaudio==2.4.0",
               "--index-url", idx_url]
        if "venv" not in python_path and "conda" not in python_path:
            cmd.append("--break-system-packages")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            yield f'data: {json.dumps({"step": "✅ GPU 版 PyTorch 安装成功！", "progress": 90})}\n\n'
            # Verify
            v = subprocess.run(
                [python_path, "-c", "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"],
                capture_output=True, text=True, timeout=15
            )
            lines = v.stdout.strip().split("\n")
            if lines and lines[0].strip() == "True":
                cuda_v = lines[1].strip() if len(lines) > 1 else "?"
                yield f'data: {json.dumps({"step": f"✅ 验证通过: torch.cuda.is_available() = True (CUDA {cuda_v})", "done": True})}\n\n'
            else:
                yield f'data: {json.dumps({"step": "⚠️ 安装完成但 CUDA 仍不可用，可能需要重启或检查驱动", "done": True})}\n\n'
        else:
            yield f'data: {json.dumps({"step": f"❌ 安装失败: {r.stderr[:400]}", "error": True})}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# ======================= Training =======================
@app.post("/api/finetune/train")
async def start_training(request: Request):
    state = _get_train_state()
    if state["status"] == "training":
        raise HTTPException(409, "已有训练任务在运行")

    body = await request.json()
    python_path = body.get("python_path", sys.executable)

    # Validate inputs that will be injected into the generated training script
    VALID_METHODS = {"qlora", "lora"}
    VALID_FORMATS = {"alpaca", "sharegpt", "openai"}
    VALID_QUANTS = {"q8_0", "q4_k_m", "q5_k_m", "q4_0", "q5_0", "q5_1", "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k_s", "q6_k", "f16", "f32"}

    method = body.get("method", "qlora")
    if method not in VALID_METHODS:
        raise HTTPException(400, f"不支持的训练方式: {method}")
    ds_format = body.get("dataset_format", "alpaca")
    if ds_format not in VALID_FORMATS:
        raise HTTPException(400, f"不支持的数据集格式: {ds_format}")
    quant = body.get("quant_method", "q8_0")
    if quant not in VALID_QUANTS:
        raise HTTPException(400, f"不支持的量化方法: {quant}")

    # Validate base_model: only allow safe HuggingFace model identifiers
    base_model = body.get("base_model", "meta-llama/Llama-3.2-1B-Instruct")
    if not re.match(r'^[A-Za-z0-9][A-Za-z0-9_\-./]*$', base_model) or len(base_model) > 200:
        raise HTTPException(400, "模型名格式不合法")

    # Validate dataset_path: must stay within FT_DIR/datasets
    ds_name = body.get("dataset_path", "")
    ds_name = Path(ds_name).name  # strip any directory components
    ds_full = (FT_DIR / "datasets" / ds_name).resolve()
    if not str(ds_full).startswith(str((FT_DIR / "datasets").resolve())):
        raise HTTPException(400, "数据集路径不合法")
    if not ds_full.exists():
        raise HTTPException(400, f"数据集文件不存在: {ds_name}")

    output_name = _sanitize_model_name(body.get("output_name", "my-finetune"))

    # Create project directory
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"{output_name}_{ts_str}"
    project_dir = _projects_dir() / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "outputs").mkdir(exist_ok=True)

    cfg = {
        "base_model": base_model,
        "dataset_path": str(ds_full).replace("\\", "/"),
        "dataset_format": ds_format,
        "output_name": output_name,
        "method": method,
        "lora_r": max(1, int(body.get("lora_r", 16))),
        "lora_alpha": max(1, int(body.get("lora_alpha", 16))),
        "epochs": max(1, int(body.get("epochs", 3))),
        "batch_size": max(1, int(body.get("batch_size", 2))),
        "learning_rate": float(body.get("learning_rate", 2e-4)),
        "max_seq_length": max(128, int(body.get("max_seq_length", 2048))),
        "warmup_steps": max(0, int(body.get("warmup_steps", 5))),
        "save_steps": max(1, int(body.get("save_steps", 50))),
        "quant_method": quant,
        "export_ollama": body.get("export_ollama", True),
        "hf_home": os.environ.get("HF_HOME", ""),
    }

    # Save project metadata
    _save_project(project_name, {
        "name": project_name,
        "label": output_name,
        "base_model": base_model,
        "dataset": ds_name,
        "method": method,
        "created": datetime.now().isoformat(),
        "status": "training",
        "config": cfg,
    })

    job_id = secrets.token_hex(4)
    start_time = time.time()
    _update_train_state(
        status="training", progress=0, logs=[], error="",
        job_id=job_id, config=cfg, pid=None, last_loss=None,
        start_time=start_time, project_name=project_name
    )

    script = _gen_train_script(cfg)
    script_path = project_dir / "train_script.py"
    script_path.write_text(script, "utf-8")

    def run_train():
        _do_train(python_path, script_path, project_dir, cfg, job_id, start_time, project_name)
    threading.Thread(target=run_train, daemon=True).start()
    return {"status": "started", "job_id": job_id, "project": project_name}


def _do_train(python_path, script_path, work_dir, cfg, job_id, start_time, project_name):
    """Shared training loop for both new training and checkpoint resume."""
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [python_path, str(script_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(work_dir),
            encoding='utf-8', errors='replace',
            env=env
        )
        _update_train_state(pid=proc.pid)

        for line in proc.stdout:
            line = line.rstrip()
            with _train_lock:
                _train_state["logs"].append(line)
                # Parse progress
                if "'loss'" in line:
                    try:
                        m = re.search(r"'loss':\s*([\d.]+)", line)
                        if m:
                            _train_state["last_loss"] = float(m.group(1))
                        m2 = re.search(r"'epoch':\s*([\d.]+)", line)
                        if m2:
                            ep = float(m2.group(1))
                            _train_state["progress"] = min(95, int(ep / cfg["epochs"] * 95))
                    except Exception:
                        pass
                if "EXPORT_DONE" in line:
                    _train_state["progress"] = 100
                # Detect common errors and add friendly messages
                line_lower = line.lower()
                if "gatedrepoerror" in line_lower or "gated repo" in line_lower or "authorized list" in line_lower:
                    _train_state["logs"].append(
                        "❌ 错误: 该模型是受限模型(Gated Repo)，需要先在 HuggingFace 申请访问权限。"
                    )
                elif "401" in line and ("unauthorized" in line_lower or "token" in line_lower):
                    _train_state["logs"].append(
                        "❌ 错误: HuggingFace 登录凭证无效或已过期，请在「训练环境」Tab 重新登录。"
                    )
                elif "out of memory" in line_lower or "cuda oom" in line_lower:
                    _train_state["logs"].append(
                        "❌ 错误: 显存不足(OOM)。建议: 减小 batch_size、减小 max_seq_length、或选择更小的模型。"
                    )
                # Parse TRAIN_META for history
                if "TRAIN_META:" in line:
                    try:
                        meta_str = line.split("TRAIN_META:")[1].strip()
                        meta = json.loads(meta_str)
                        _train_state["_meta"] = meta
                    except Exception:
                        pass
                # Keep log manageable
                if len(_train_state["logs"]) > 500:
                    _train_state["logs"] = _train_state["logs"][-300:]

        proc.wait()
        duration = time.time() - start_time
        if proc.returncode == 0:
            _update_train_state(status="completed", progress=100)
            # Save to training history
            meta = _train_state.get("_meta", {})
            record = {
                "id": job_id,
                "timestamp": datetime.now().isoformat(),
                "base_model": cfg["base_model"],
                "output_name": cfg["output_name"],
                "method": cfg["method"],
                "dataset": Path(cfg["dataset_path"]).name,
                "dataset_format": cfg["dataset_format"],
                "epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
                "learning_rate": cfg["learning_rate"],
                "lora_r": cfg["lora_r"],
                "lora_alpha": cfg["lora_alpha"],
                "max_seq_length": cfg["max_seq_length"],
                "final_loss": meta.get("loss") or _train_state.get("last_loss"),
                "total_steps": meta.get("total_steps"),
                "duration_seconds": round(duration),
                "output_path": str(work_dir / "merged_model").replace("\\", "/"),
                "adapter_path": str(work_dir / "outputs").replace("\\", "/"),
                "ollama_registered": meta.get("ollama_ok", False),
                "resumed": bool(cfg.get("resume_from_checkpoint")),
                "project": project_name,
            }
            _add_history_record(record)
            # Update project status
            p = _load_project(project_name)
            if p:
                p["status"] = "completed"
                p["completed"] = datetime.now().isoformat()
                p["final_loss"] = record["final_loss"]
                _save_project(project_name, p)
        else:
            logs_text = "\n".join(_train_state.get("logs", [])[-20:]).lower()
            if "gatedrepoerror" in logs_text or "gated repo" in logs_text:
                _update_train_state(status="failed", error="模型访问被拒绝(Gated Repo)，请先在 HuggingFace 模型页面申请权限")
            elif "out of memory" in logs_text:
                _update_train_state(status="failed", error="显存不足(OOM)，请减小 batch_size 或 max_seq_length")
            else:
                _update_train_state(status="failed", error=f"训练进程退出码 {proc.returncode}")
            p = _load_project(project_name)
            if p:
                p["status"] = "failed"
                _save_project(project_name, p)
    except Exception as e:
        _update_train_state(status="failed", error=str(e))

def _gen_train_script(cfg):
    """Generate training script with proper GPU detection and fallback."""
    # Import the template function
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_script_template import get_train_script
    return get_train_script(cfg)

@app.get("/api/finetune/status")
async def train_status():
    state = _get_train_state()
    return {
        "status": state.get("status", "idle"),
        "progress": state.get("progress", 0),
        "logs": state.get("logs", [])[-50:],
        "error": state.get("error", ""),
        "job_id": state.get("job_id", ""),
        "last_loss": state.get("last_loss"),
        "config": state.get("config", {})
    }

@app.post("/api/finetune/stop")
async def stop_training():
    pid = _train_state.get("pid")
    if pid:
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    _update_train_state(status="stopped", error="用户手动停止")
    return {"status": "stopped"}

@app.post("/api/finetune/import-gguf")
async def import_gguf(
    file: UploadFile = File(...),
    name: str = Form("imported-model"),
    system_prompt: str = Form("")
):
    name = _sanitize_model_name(name)
    content = await file.read()
    imp_dir = FT_DIR / "imports"
    imp_dir.mkdir(exist_ok=True)
    fname = _sanitize_name(file.filename or "model.gguf", "文件名")
    fpath = imp_dir / fname
    fpath.write_bytes(content)

    mf = f"FROM ./{fname}\n"
    if system_prompt:
        mf += f'SYSTEM """{system_prompt}"""\n'
    mf_path = imp_dir / "Modelfile"
    mf_path.write_text(mf, "utf-8")

    try:
        r = subprocess.run(
            ["ollama", "create", name, "-f", str(mf_path)],
            capture_output=True, text=True, cwd=str(imp_dir), timeout=300,
            encoding='utf-8', errors='replace'
        )
        if r.returncode != 0:
            return {"status": "error", "error": r.stderr or "创建失败"}
        return {"status": "ok", "model_name": name, "size": len(content)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ======================= Model Workshop (模型工坊) =======================

@app.get("/api/workshop/models")
async def workshop_list_models():
    """List all Ollama models with detailed info."""
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return {"models": [], "error": r.stderr}
        models = []
        for line in r.stdout.strip().split("\n")[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 4:
                name = parts[0]
                model_id = parts[1] if len(parts) > 1 else ""
                size = parts[2] + " " + parts[3] if len(parts) > 3 else ""
                modified = " ".join(parts[4:]) if len(parts) > 4 else ""
                models.append({"name": name, "id": model_id, "size": size, "modified": modified})
        return {"models": models}
    except FileNotFoundError:
        return {"models": [], "error": "ollama not found"}
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.get("/api/workshop/model-info/{name:path}")
async def workshop_model_info(name: str):
    """Get detailed model info including Modelfile."""
    result = {"name": name}
    try:
        r = subprocess.run(["ollama", "show", name], capture_output=True, text=True, timeout=15)
        result["show"] = r.stdout if r.returncode == 0 else r.stderr

        r2 = subprocess.run(["ollama", "show", name, "--modelfile"],
                           capture_output=True, text=True, timeout=15)
        result["modelfile"] = r2.stdout if r2.returncode == 0 else ""

        # Parse parameters from show output
        params = {}
        if r.returncode == 0:
            for line in r.stdout.split("\n"):
                line = line.strip()
                if line and not line.startswith("Model") and not line.startswith("---"):
                    kv = line.split(None, 1)
                    if len(kv) == 2:
                        params[kv[0]] = kv[1]
        result["parameters"] = params
    except Exception as e:
        result["error"] = str(e)
    return result

@app.post("/api/workshop/quick-test")
async def workshop_quick_test(request: Request):
    """Quick test a model with a prompt."""
    body = await request.json()
    model = body.get("model", "")
    prompt = body.get("prompt", "你好！请简单介绍一下你自己。")
    if not model:
        raise HTTPException(400, "请选择模型")

    try:
        import httpx
        ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{ollama_url}/api/generate", json={
                "model": model, "prompt": prompt, "stream": False,
                "options": {"num_predict": 200, "temperature": 0.7}
            })
            data = resp.json()
            return {
                "response": data.get("response", ""),
                "total_duration": data.get("total_duration", 0),
                "eval_count": data.get("eval_count", 0),
                "eval_duration": data.get("eval_duration", 0),
            }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/workshop/create-variant")
async def workshop_create_variant(request: Request):
    """Create a model variant with different system prompt / parameters."""
    body = await request.json()
    base = body.get("base_model", "")
    new_name = _sanitize_model_name(body.get("new_name", ""))
    system_prompt = body.get("system_prompt", "")
    params = body.get("parameters", {})

    if not base or not new_name:
        raise HTTPException(400, "需要基础模型和新名称")

    # Build Modelfile
    mf_lines = [f"FROM {base}"]
    if system_prompt:
        safe_sys = system_prompt.replace('"""', '\\"\\"\\"')
        mf_lines.append(f'SYSTEM """{safe_sys}"""')
    for k, v in params.items():
        if k in ("temperature", "top_p", "top_k", "repeat_penalty", "repeat_last_n",
                 "num_ctx", "num_predict", "stop", "seed"):
            mf_lines.append(f"PARAMETER {k} {v}")

    mf_content = "\n".join(mf_lines)
    tmp_dir = FT_DIR / "tmp_variant"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mf_path = tmp_dir / "Modelfile"
    mf_path.write_text(mf_content, "utf-8")

    try:
        r = subprocess.run(["ollama", "create", new_name, "-f", str(mf_path)],
                          capture_output=True, text=True, timeout=300,
                          encoding='utf-8', errors='replace')
        if r.returncode != 0:
            return {"status": "error", "error": _strip_ansi(r.stderr[:500] or "创建失败"), "modelfile": mf_content}
        return {"status": "ok", "name": new_name, "modelfile": mf_content}
    except Exception as e:
        return {"status": "error", "error": str(e), "modelfile": mf_content}

@app.post("/api/workshop/export-modelfile")
async def workshop_export_modelfile(request: Request):
    """Export a model's Modelfile."""
    body = await request.json()
    model = body.get("model", "")
    if not model:
        raise HTTPException(400, "请选择模型")
    try:
        r = subprocess.run(["ollama", "show", model, "--modelfile"],
                          capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return {"error": r.stderr}
        return {"modelfile": r.stdout, "model": model}
    except Exception as e:
        return {"error": str(e)}

def _strip_ansi(text):
    """Strip ANSI escape codes from terminal output."""
    return re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]|\x1b\].*?\x07', '', text).strip()

def _detect_model_quant(model_name):
    """Detect a model's quantization level. Returns (quant_str, is_f16_or_f32)."""
    try:
        r = subprocess.run(["ollama", "show", model_name],
                          capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            text = r.stdout.lower()
            # Look for quantization info in show output
            for line in text.split("\n"):
                line = line.strip()
                if "quantization" in line or "quant" in line:
                    if "f16" in line or "fp16" in line:
                        return "F16", True
                    if "f32" in line or "fp32" in line:
                        return "F32", True
                    # Extract quant type like Q4_0, Q4_K_M, etc.
                    m = re.search(r'(q\d[_\w]*)', line)
                    if m:
                        return m.group(1).upper(), False
                # Also check "architecture" or format lines
                if "f16" in line and ("format" in line or "type" in line or "size" in line):
                    return "F16", True
    except Exception:
        pass
    return "unknown", False

@app.post("/api/workshop/quantize")
async def workshop_quantize(request: Request):
    """Quantize a model — creates a new quantized variant via Ollama."""
    body = await request.json()
    source = body.get("source_model", "")
    new_name = _sanitize_model_name(body.get("new_name", ""))
    quant_type = body.get("quant_type", "q4_0")

    if not source or not new_name:
        raise HTTPException(400, "需要源模型和新名称")

    VALID_QUANTS = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q4_k_s", "q4_k_m",
                    "q5_k_s", "q5_k_m", "q6_k", "q3_k_s", "q3_k_m", "q3_k_l", "q2_k", "f16"}
    if quant_type not in VALID_QUANTS:
        raise HTTPException(400, f"不支持的量化类型: {quant_type}")

    tmp_dir = FT_DIR / "tmp_quant"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ===== Strategy 1: Check for local safetensors from fine-tuning =====
    merged_dir = FT_DIR / "merged_model"
    if merged_dir.exists() and any(merged_dir.glob("*.safetensors")):
        mf_lines = ["FROM ."]
        try:
            r = subprocess.run(["ollama", "show", source, "--modelfile"],
                              capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    if line.startswith(("TEMPLATE", "SYSTEM", "PARAMETER", "LICENSE")):
                        mf_lines.append(line)
        except Exception:
            pass
        mf_content = "\n".join(mf_lines)
        mf_path = merged_dir / "Modelfile.quant"
        mf_path.write_text(mf_content, "utf-8")
        try:
            r = subprocess.run(
                ["ollama", "create", new_name, "-f", str(mf_path), "--quantize", quant_type],
                capture_output=True, text=True, cwd=str(merged_dir), timeout=1800,
                encoding='utf-8', errors='replace'
            )
            if r.returncode != 0:
                err = _strip_ansi(r.stderr or r.stdout or "量化失败")
                return {"status": "error", "error": err[:500]}
            return {"status": "ok", "name": new_name, "quant": quant_type, "method": "safetensors"}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "量化超时(>30分钟)"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ===== Strategy 2: Re-quantize from existing Ollama model =====
    # Pre-check: detect source model quantization level
    src_quant, is_fp = _detect_model_quant(source)

    mf_lines = [f"FROM {source}"]
    mf_content = "\n".join(mf_lines)
    mf_path = tmp_dir / "Modelfile.quant"
    mf_path.write_text(mf_content, "utf-8")
    try:
        r = subprocess.run(
            ["ollama", "create", new_name, "-f", str(mf_path), "--quantize", quant_type],
            capture_output=True, text=True, cwd=str(tmp_dir), timeout=1800,
            encoding='utf-8', errors='replace'
        )
        if r.returncode != 0:
            err = _strip_ansi(r.stderr or r.stdout or "")
            # Detect the specific "only F16/F32" error
            if "f16" in err.lower() and ("f32" in err.lower() or "only supported" in err.lower()):
                hint = (f"Ollama 只能从 F16/F32 精度的模型进行量化。"
                        f"当前源模型 {source} 的精度为 {src_quant}，已经是量化过的格式。\n\n"
                        f"解决方案:\n"
                        f"1. 拉取该模型的 F16 版本再量化 (如果有的话)\n"
                        f"2. 从 HuggingFace 下载 safetensors 原始权重后通过微调中心导入\n"
                        f"3. 使用「模型变体」功能修改已有模型的参数（无需量化）")
                return {"status": "error", "error": f"源模型 {source} ({src_quant}) 不是 F16/F32 精度，无法量化", "hint": hint}
            return {"status": "error", "error": err[:500],
                    "hint": "量化失败，请检查源模型是否为 F16/F32 精度"}
        return {"status": "ok", "name": new_name, "quant": quant_type, "method": "re-quantize"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "量化超时(>30分钟)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.delete("/api/workshop/model/{name:path}")
async def workshop_delete_model(name: str):
    """Delete an Ollama model."""
    try:
        r = subprocess.run(["ollama", "rm", name], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return {"status": "error", "error": _strip_ansi(r.stderr or "删除失败")}
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/api/workshop/copy-model")
async def workshop_copy_model(request: Request):
    """Copy/clone a model with a new name."""
    body = await request.json()
    source = body.get("source", "")
    dest = _sanitize_model_name(body.get("dest", ""))
    if not source or not dest:
        raise HTTPException(400, "需要源模型和目标名称")
    try:
        r = subprocess.run(["ollama", "cp", source, dest],
                          capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return {"status": "error", "error": _strip_ansi(r.stderr or "复制失败")}
        return {"status": "ok", "name": dest}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/workshop/import-hf")
async def workshop_import_hf(request: Request):
    """Import a HuggingFace model into Ollama (download safetensors → ollama create)."""
    body = await request.json()
    hf_model = body.get("hf_model", "").strip()
    ollama_name = _sanitize_model_name(body.get("ollama_name", ""))
    quant_type = body.get("quant_type", "q4_k_m")
    python_path = body.get("python_path", sys.executable)

    if not hf_model or not ollama_name:
        raise HTTPException(400, "请指定 HuggingFace 模型 ID 和 Ollama 模型名")
    if not re.match(r'^[A-Za-z0-9][A-Za-z0-9_\-./]*$', hf_model):
        raise HTTPException(400, "模型 ID 格式不合法")

    VALID_QUANTS = {"q4_0", "q4_k_m", "q5_k_m", "q8_0", "q3_k_m", "f16"}
    if quant_type not in VALID_QUANTS:
        quant_type = "q4_k_m"

    work_dir = FT_DIR / "hf_import" / ollama_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Generate a download + import script
    hf_home = os.environ.get("HF_HOME", "").replace("\\", "/")
    script = f'''#!/usr/bin/env python3
"""Download HF model and prepare for Ollama import."""
import os, sys, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix Windows GBK encoding issue — ensure stdout uses UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

{"os.environ['HF_HOME'] = " + repr(hf_home) if hf_home else "# HF_HOME: default"}

def main():
    print("STEP: Downloading model from HuggingFace: {hf_model}", flush=True)
    from huggingface_hub import snapshot_download
    
    # Load HF token
    _hf_token = None
    for _tp in [
        os.path.join(os.environ.get("HF_HOME", ""), "token"),
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token"),
    ]:
        if _tp and os.path.isfile(_tp):
            _hf_token = open(_tp, encoding="utf-8").read().strip()
            if _hf_token: break
    if not _hf_token:
        _hf_token = os.environ.get("HF_TOKEN")
    
    local_path = snapshot_download(
        "{hf_model}",
        token=_hf_token,
        local_dir=os.path.join(os.getcwd(), "model_files"),
        ignore_patterns=["*.bin", "*.ot", "*.h5", "consolidated*"],
    )
    print(f"STEP: Downloaded to {{local_path}}", flush=True)
    
    # Check for safetensors
    import glob
    st_files = glob.glob(os.path.join(local_path, "*.safetensors"))
    if not st_files:
        print("ERROR: No safetensors files found. Model may use a different format.", flush=True)
        sys.exit(1)
    print(f"STEP: Found {{len(st_files)}} safetensors files", flush=True)
    
    # Build minimal Modelfile — only FROM + optional defaults.
    # Ollama's own Jinja→Go converter reads tokenizer_config.json
    # and auto-generates TEMPLATE / stop tokens, which is far more
    # reliable than any hardcoded heuristic we could write here.
    print("STEP: Creating Modelfile...", flush=True)
    mf_lines = ["FROM ."]
    mf_lines.append('SYSTEM """You are a helpful assistant."""')
    mf_lines.append('PARAMETER temperature 0.7')
    
    mf_content = "\\n".join(mf_lines) + "\\n"
    mf_path = os.path.join(local_path, "Modelfile")
    with open(mf_path, "w", encoding="utf-8") as f:
        f.write(mf_content)
    print("STEP: Modelfile created", flush=True)
    print(f"STEP: Modelfile content:\\n{{mf_content}}", flush=True)
    
    # Register with Ollama
    print("STEP: Importing to Ollama (this may take several minutes)...", flush=True)
    import subprocess as sp
    cmd = ["ollama", "create", "{ollama_name}", "-f", mf_path, "--quantize", "{quant_type}"]
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1, cwd=local_path, encoding='utf-8', errors='replace')
    import re as _re
    for line in proc.stdout:
        line = _re.sub(r"\\x1b\\[[0-9;?]*[a-zA-Z]|\\x1b\\].*?\\x07", "", line).strip()
        if line:
            print(f"STEP: Ollama: {{line}}", flush=True)
    proc.wait()
    if proc.returncode == 0:
        print(f"STEP: ✅ Model '{ollama_name}' imported successfully!", flush=True)
        print("IMPORT_DONE", flush=True)
    else:
        print(f"ERROR: ollama create failed (exit code {{proc.returncode}})", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    script_path = work_dir / "import_script.py"
    script_path.write_text(script, "utf-8")

    async def gen():
        yield f"data: {json.dumps({'step': f'开始导入 {hf_model}...', 'progress': 5})}\n\n"
        try:
            _env = os.environ.copy()
            _env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.Popen(
                [python_path, str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(work_dir),
                encoding='utf-8', errors='replace',
                env=_env
            )
            for line in proc.stdout:
                line = line.rstrip()
                progress = 30
                if "Downloaded" in line: progress = 40
                elif "safetensors" in line: progress = 50
                elif "Modelfile" in line: progress = 55
                elif "Importing" in line: progress = 60
                elif "Ollama:" in line: progress = 75
                elif "IMPORT_DONE" in line: progress = 100
                elif "ERROR" in line: progress = 0
                step_text = _strip_ansi(line.replace("STEP: ", ""))
                yield f"data: {json.dumps({'step': step_text, 'progress': progress})}\n\n"
            proc.wait()
            if proc.returncode == 0:
                yield f"data: {json.dumps({'step': f'✅ {ollama_name} 导入完成！', 'progress': 100, 'done': True, 'model_name': ollama_name})}\n\n"
            else:
                yield f"data: {json.dumps({'step': '❌ 导入失败', 'error': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'step': f'❌ {e}', 'error': True})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ======================= Static Files =======================
@app.get("/")
async def index():
    return HTMLResponse((FE / "index.html").read_text("utf-8"))

@app.get("/styles.css")
async def css():
    return Response((FE / "styles.css").read_text("utf-8"), media_type="text/css")

@app.get("/app.js")
async def js():
    return Response((FE / "app.js").read_text("utf-8"), media_type="application/javascript")

@app.get("/pretrain-lab.js")
async def pretrain_js():
    return Response((FE / "pretrain-lab.js").read_text("utf-8"), media_type="application/javascript")