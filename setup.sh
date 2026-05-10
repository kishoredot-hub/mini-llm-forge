#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# LLMForge Setup Script — Ubuntu 24.04
# Run once: bash setup.sh
# ─────────────────────────────────────────────────────────────────
set -e

echo "═══════════════════════════════════════════════"
echo "  LLMForge Setup — Ubuntu 24.04"
echo "═══════════════════════════════════════════════"

# ── System deps ──────────────────────────────────────────────────
echo "[1/5] Installing system dependencies..."
sudo apt update -qq
sudo apt install -y python3 python3-pip python3-venv python3-dev \
    build-essential git curl wget ffmpeg libsndfile1 \
    poppler-utils -qq

# ── Virtual environment ──────────────────────────────────────────
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# ── PyTorch (auto-detect CUDA) ───────────────────────────────────
echo "[3/5] Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[\d.]+" | head -1 | cut -d. -f1,2 | tr -d .)
    echo "    Detected CUDA ${CUDA_VERSION}"
    if [ "$CUDA_VERSION" -ge "121" ] 2>/dev/null; then
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121 -q
    else
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu118 -q
    fi
else
    echo "    No GPU detected — installing CPU-only PyTorch"
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

# ── Python packages ──────────────────────────────────────────────
echo "[4/5] Installing Python packages..."
pip install -q \
    transformers==4.40.0 datasets accelerate peft bitsandbytes \
    sentencepiece tokenizers \
    chromadb sentence-transformers langchain langchain-community \
    PyMuPDF pypdf \
    nltk rouge-score evaluate \
    gradio>=4.0 \
    pandas numpy matplotlib seaborn \
    psutil tqdm rich pyyaml \
    python-dotenv genanki \
    jupyter jupyterlab ipywidgets

# Optional: Lion optimizer
pip install lion-pytorch -q 2>/dev/null || true

# ── Project structure ────────────────────────────────────────────
echo "[5/5] Creating project directories..."
mkdir -p data/{course_pdfs,checkpoints,chromadb} outputs/plots \
    logs notebooks

# ── Make __init__ files ──────────────────────────────────────────
touch src/__init__.py src/components/__init__.py \
      src/data/__init__.py src/training/__init__.py \
      src/rag/__init__.py src/agent/__init__.py \
      src/eval/__init__.py src/export/__init__.py \
      app/__init__.py tests/__init__.py

# ── Verify ───────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
python3 -c "
import torch
print(f'  PyTorch : {torch.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU     : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
import gradio; print(f'  Gradio  : {gradio.__version__}')
import transformers; print(f'  HF Trans: {transformers.__version__}')
"
echo "═══════════════════════════════════════════════"
echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To launch the app:"
echo "  python app/app.py"
echo ""
echo "To start JupyterLab:"
echo "  jupyter lab --port=8888"
