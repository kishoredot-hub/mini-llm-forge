# 🤖 LLMForge — End-to-End AI Assistant System

**IISc CCE — Large Language Models: A Hands-On Approach | Jan–May 2026**

A modular, educational LLM toolkit that covers every course module through
building a working AI assistant. Inspired by [buildanllm.com](https://buildanllm.com)
and [Unsloth Studio](https://unsloth.ai). Runs on free Colab T4 or Ubuntu 24.04.

---

## Course Module Coverage

| Module | What is Built | File |
|---|---|---|
| **Tokenization** | Custom BPE tokenizer + HF wrapper | `src/data/tokenizer.py` |
| **Architecture** | TinyGPT — all components swappable via config | `src/model.py` |
| **Training** | Trainer — swappable optimizers, schedules, precision | `src/training/trainer.py` |
| **Fine-tuning** | LoRA from scratch — inject, merge, save adapter | `src/training/lora.py` |
| **Inference Opt** | FP16/4-bit/KV-cache benchmarks | `notebooks/Part2_*.ipynb` |
| **RAG** | ChromaDB + MiniLM + gap detection | `src/rag/pipeline.py` |
| **Context Eng** | Zero-shot / few-shot / CoT / system prompt | `notebooks/Part3_*.ipynb` |
| **Agents** | ReAct agent + calculator/search/definition tools | `src/agent/react.py` |
| **Evaluation** | PPL / BLEU / ROUGE / faithfulness / speed | `src/eval/evaluator.py` |
| **Export** | safetensors / GGUF / LoRA adapter | `src/export/exporter.py` |

---

## Quick Start

```bash
# Clone and setup
git clone <your-repo>
cd LLMForge
bash setup.sh          # installs everything (Ubuntu 24.04)

# Activate environment
source venv/bin/activate

# Verify everything works
python scripts/quickstart.py

# Launch Gradio app (6 tabs)
python app/app.py
# Open: http://localhost:7860

# Run tests
pytest tests/ -v
```

---

## Modular Architecture — Swap Any Component

Every component is registered and swappable via config or code:

```python
from src.model import TransformerLM, ModelConfig

# GPT-2 style (learned pos + LayerNorm + GELU + standard MHA)
model = TransformerLM.from_preset("gpt2")

# LLaMA style (RoPE + RMSNorm + SwiGLU + GQA)
model = TransformerLM.from_preset("llama")

# OLMo style (ALiBi + LayerNorm + SwiGLU)
model = TransformerLM.from_preset("olmo")

# Custom mix — any combination
model = TransformerLM(ModelConfig(
    pos_encoding   = "rope",          # learned | rope | alibi | none
    norm_type      = "rmsnorm",       # layernorm | rmsnorm
    activation     = "swiglu",        # gelu | swiglu | relu | geglu
    attention_type = "standard",      # standard | grouped_query
    d_model=256, n_heads=8, n_layers=6, d_ff=1024,
))
```

### From YAML config

```yaml
# config/gpt2_style.yaml
model:
  pos_encoding:   learned
  norm_type:      layernorm
  activation:     gelu
  attention_type: standard
  d_model: 256
  n_heads: 8
  n_layers: 6

training:
  optimizer:    adamw    # adamw | lion | adafactor | sgd
  lr_schedule:  cosine   # cosine | linear | wsd | constant
  precision:    fp16     # fp32 | fp16 | bf16
```

```python
model = TransformerLM.from_yaml("config/gpt2_style.yaml")
```

---

## Datasets — Plug Any Source

```python
from src.data.loaders import make_dataloaders

# HuggingFace datasets
train_dl, val_dl = make_dataloaders("tinystories",   tokenizer)
train_dl, val_dl = make_dataloaders("wikitext",      tokenizer)
train_dl, val_dl = make_dataloaders("alpaca",        tokenizer)
train_dl, val_dl = make_dataloaders("dolly",         tokenizer)

# Local files
train_dl, val_dl = make_dataloaders("custom_text", tokenizer, path="my_corpus.txt")
train_dl, val_dl = make_dataloaders("custom_csv",  tokenizer, path="data.csv")
train_dl, val_dl = make_dataloaders("pdf",         tokenizer, path="data/pdfs/")
```

---

## Training

```python
from src.training.trainer import Trainer, TrainConfig

cfg = TrainConfig(
    optimizer    = "adamw",   # adamw | lion | adafactor | sgd
    lr_schedule  = "cosine",  # cosine | linear | wsd | constant
    precision    = "fp16",    # fp32 | fp16 | bf16
    batch_size   = 32,
    max_steps    = 5000,
    warmup_steps = 200,
)
trainer = Trainer(model, train_dl, val_dl, cfg)
metrics = trainer.train()
```

---

## LoRA Fine-tuning

```python
from src.training.lora import LoRAConfig, inject_lora, save_adapter, merge_lora

cfg   = LoRAConfig(rank=8, alpha=16, target_modules=["qkv", "out_proj"])
model = inject_lora(model, cfg)   # ~0.1% trainable params

# Fine-tune ...
trainer.train()

# Save only the adapter (tiny ~2MB file)
save_adapter(model, "checkpoints/adapter/", cfg)

# Or merge into base weights for deployment
model = merge_lora(model)
```

---

## RAG Pipeline

```python
from src.rag.pipeline import RAGPipeline

rag = RAGPipeline(embed_model="all-MiniLM-L6-v2", chunk_size=256)
rag.ingest_pdf("lecture_notes.pdf")
rag.ingest_text("Any text content...")

chunks = rag.retrieve("What is attention?", k=3)
```

---

## ReAct Agent

```python
from src.agent.react import MiniAgent

agent = MiniAgent(generate_fn=my_generate_fn, rag=rag)
answer, steps = agent.run("How much memory does GPT-2 Medium need in FP16?")
# Agent uses: calculator, doc_search, definition tools
```

---

## Export

```python
from src.export.exporter import ModelExporter

exporter = ModelExporter(model, tokenizer)
exporter.save_checkpoint("checkpoints/model.pt")      # PyTorch
exporter.save_safetensors("export/safetensors/")      # HF format
exporter.save_fp16("export/fp16/")                    # Half precision
exporter.export_gguf("export/gguf/", "q4_k_m")        # llama.cpp / Ollama
exporter.export_summary("export/final/")              # Full package
```

---

## Register Your Own Component

```python
from src.components.registry import Registry
import torch.nn as nn

@Registry.norm("my_custom_norm")
class MyNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x * self.scale

# Now use in config
model = TransformerLM(ModelConfig(norm_type="my_custom_norm", ...))
```

---

## Project Structure

```
LLMForge/
├── src/
│   ├── components/        # Swappable building blocks
│   │   ├── registry.py    # Central component registry
│   │   ├── norms.py       # LayerNorm, RMSNorm
│   │   ├── positional.py  # Learned, RoPE, ALiBi, None
│   │   ├── activations.py # GELU, SwiGLU, ReLU, GeGLU
│   │   └── attention.py   # MHA, Grouped Query Attention
│   ├── model.py           # TransformerLM — assembles from config
│   ├── data/
│   │   ├── tokenizer.py   # BPE trainer + HF wrapper
│   │   └── loaders.py     # Dataset registry (7 sources)
│   ├── training/
│   │   ├── trainer.py     # Training engine
│   │   └── lora.py        # LoRA / QLoRA from scratch
│   ├── rag/pipeline.py    # RAG pipeline
│   ├── agent/react.py     # ReAct agent
│   ├── eval/evaluator.py  # Evaluation suite
│   └── export/exporter.py # Export to pt/safetensors/GGUF
├── config/                # YAML configs for each architecture
├── app/app.py             # 6-tab Gradio app
├── notebooks/             # 4 Parts Colab notebooks
├── tests/                 # pytest suite (30+ tests)
├── scripts/quickstart.py  # Demo script
├── setup.sh               # Ubuntu 24.04 setup
└── requirements.txt
```

---

## Gradio App — 6 Tabs

| Tab | Description |
|---|---|
| 🏗️ **Architecture** | Build any model via dropdowns, see param count live |
| 🏋️ **Train** | Pre-train with live loss/PPL/grad norm plots |
| 📚 **RAG Q&A** | Upload PDFs, ask questions with source attribution |
| 🤖 **Agent** | ReAct agent with tool calling |
| ⚡ **Inference** | Generate text + attention heatmaps (glass-box) |
| 📊 **Evaluation** | Full eval dashboard — PPL, speed, VRAM |

---

## Inspired By

- [buildanllm.com](https://buildanllm.com) — Glass-box training visualization, modular architecture presets
- [Unsloth Studio](https://unsloth.ai) — Data Recipes, real-time observability, GGUF export, QLoRA
- [nanoGPT](https://github.com/karpathy/nanogpt) — Clean GPT implementation
- [LLaMA](https://arxiv.org/abs/2302.13971) — RoPE, RMSNorm, SwiGLU, GQA

---

*IISc CCE LLM Course | Jan–May 2026 | Runs on free Colab T4*
