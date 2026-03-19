# 微调中心安装指南

## 架构说明

v5 使用**标准 HuggingFace PEFT + TRL** 进行微调，不依赖 Unsloth。

**优点**：
- Windows 完全兼容，无 multiprocessing/pickle 问题
- 无需安装 triton、xformers、unsloth 等复杂依赖
- 依赖少，安装快，出错概率低
- 模型保存为 safetensors，Ollama 直接导入并自动量化

## 自定义数据目录

默认数据目录为 `~/.ollama-gui-finetune`（Windows 上在 C 盘）。

**如需改到其他盘**，设置环境变量：

```bash
# Windows (PowerShell)
$env:OLLAMA_GUI_FT_DIR = "E:\ollama-gui-finetune"
python run.py

# Windows (CMD)
set OLLAMA_GUI_FT_DIR=E:\ollama-gui-finetune
python run.py

# Linux/Mac
export OLLAMA_GUI_FT_DIR=/data/ollama-gui-finetune
python run.py
```

## 安装方法

### 方法1: 使用 GUI 一键安装（推荐）

在微调中心界面中点击「一键安装」即可，会自动：
1. 安装 GPU 版 PyTorch（根据系统 CUDA 版本）
2. 安装 transformers、peft、trl、datasets 等训练依赖

### 方法2: 手动安装

```bash
# 1. 创建环境
conda create -n ollama-gui-ft python=3.11 -y
conda activate ollama-gui-ft

# 2. 安装 PyTorch GPU 版（先查 CUDA 版本: nvidia-smi）
pip install torch==2.4.0 torchvision==0.19.1 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装训练依赖
pip install numpy==1.26.4 transformers==4.46.3 datasets==3.2.0 peft==0.13.2 trl==0.12.2 accelerate==1.2.1 bitsandbytes==0.45.3 sentencepiece==0.2.1 protobuf==3.20.3 huggingface_hub==0.27.1 tokenizers==0.20.3 safetensors==0.7.0
```

### 方法3: 使用 requirements 文件

```bash
conda create -n ollama-gui-ft python=3.11 -y
conda activate ollama-gui-ft
pip install -r requirements-finetune.txt
```

## 验证安装

```python
import torch, transformers, peft, trl
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("PEFT:", peft.__version__)
print("TRL:", trl.__version__)
```

## 依赖列表

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 | 推荐 |
| PyTorch | 2.4.0 | 需 GPU 版 |
| transformers | 4.46.3 | HuggingFace 模型库 |
| datasets | 3.2.0 | 数据集处理 |
| peft | 0.13.2 | LoRA 适配器 |
| trl | 0.12.2 | SFT 训练器 |
| accelerate | 1.2.1 | 分布式训练 |
| bitsandbytes | 0.45.3 | 4-bit 量化 |
| numpy | 1.26.4 | 避免 2.x 兼容问题 |

## 训练流程

```
加载模型 → BitsAndBytesConfig 4-bit 量化 → 添加 LoRA 适配器
→ SFTTrainer 训练 → 合并 LoRA → 保存 safetensors
→ ollama create（自动量化为 Q8_0/Q4_K_M 等）
```

## CUDA 版本对应

| 系统 CUDA | PyTorch 索引 URL |
|---|---|
| 12.6+ | https://download.pytorch.org/whl/cu126 |
| 12.4-12.5 | https://download.pytorch.org/whl/cu124 |
| 12.1-12.3 | https://download.pytorch.org/whl/cu121 |
| 11.8 | https://download.pytorch.org/whl/cu118 |

## 常见问题

### "PyTorch 是 CPU 版本"
点击微调中心的「修复 PyTorch」按钮，或手动：
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.0 torchvision==0.19.1 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### "NumPy 版本不兼容"
```bash
pip install numpy==1.26.4
```

### Ollama 导入失败
训练完成后会在 `merged_model/` 目录生成 safetensors 和 Modelfile。可手动导入：
```bash
cd ~/.ollama-gui-finetune/merged_model
ollama create my-model -f Modelfile
```
