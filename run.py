#!/usr/bin/env python3
"""Ollama GUI v4.0 — python run.py"""
import subprocess, sys, os, time, webbrowser, threading
from pathlib import Path

DEPS = ["fastapi", "uvicorn", "httpx", "python-multipart"]

def _ok(p):
    try:
        __import__(p.replace("-", "_"))
        return True
    except ImportError:
        return False

def main():
    # ===== Clean __pycache__ to prevent stale bytecode =====
    root = Path(__file__).parent
    for pc in root.rglob("__pycache__"):
        try:
            import shutil
            shutil.rmtree(pc)
        except Exception:
            pass

    miss = [p for p in DEPS if not _ok(p)]
    if miss:
        print(f"📦 Installing: {', '.join(miss)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + miss + ["-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    print("\n╔════════════════════════════════════════╗")
    print("║      🦙  Ollama GUI  v5.0              ║")
    print("║  标准 PEFT+TRL · 无需 Unsloth          ║")
    print("╚════════════════════════════════════════╝")

    from backend.app import FT_DIR
    print(f"\n📁 微调数据目录: {FT_DIR}")
    hf = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    print(f"🤗 HF模型缓存:   {hf}")
    print("   (在网页「硬件检测」标签页可修改路径)")


    try:
        import httpx
        assert httpx.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
        print("✅ Ollama 已连接")
    except Exception:
        print("⚠️  请先运行: ollama serve")

    # Ensure admin token exists
    from backend.app import _ensure_admin_token
    admin_token = _ensure_admin_token()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8751"))

    print(f"\n🚀 本地: http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            print(f"🌐 局域网: http://{ip}:{port}")
        except Exception:
            pass

    print(f"\n🔐 管理员令牌 (LAN访问需要): {admin_token}")
    print("   本机访问无需令牌\n")

    threading.Thread(
        target=lambda: (time.sleep(1.2), webbrowser.open(f"http://127.0.0.1:{port}")),
        daemon=True
    ).start()

    import uvicorn
    uvicorn.run("backend.app:app", host=host, port=port, log_level="warning")

if __name__ == "__main__":
    main()
