"""
export/exporter.py — Export trained models to various formats.

Supported formats:
  - PyTorch checkpoint (.pt)
  - HuggingFace safetensors
  - GGUF (via llama.cpp, for Ollama / llama.cpp inference)
  - LoRA adapter only (tiny ~MB file)

Inspired by Unsloth Studio's export workflow.
"""
from __future__ import annotations
import os, json, shutil
from typing import Optional
import torch
import torch.nn as nn


class ModelExporter:
    """Export models to deployment-ready formats."""

    def __init__(self, model: nn.Module, tokenizer=None):
        self.model     = model
        self.tokenizer = tokenizer

    # ── PyTorch checkpoint ────────────────────────────────────────
    def save_checkpoint(self, path: str, metadata: dict = None):
        """Save full model checkpoint with config."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {"state_dict": self.model.state_dict()}

        if hasattr(self.model, "cfg"):
            ckpt["config"] = self.model.cfg

        if metadata:
            ckpt["metadata"] = metadata

        torch.save(ckpt, path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"✅ Checkpoint saved → {path} ({size_mb:.1f} MB)")

    # ── HuggingFace safetensors ───────────────────────────────────
    def save_safetensors(self, path: str):
        """
        Save model weights as safetensors (HuggingFace standard).
        Compatible with HF Hub, vLLM, text-generation-inference.
        """
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("pip install safetensors")

        os.makedirs(path, exist_ok=True)
        tensors = {k: v.contiguous().cpu()
                   for k, v in self.model.state_dict().items()}
        save_file(tensors, os.path.join(path, "model.safetensors"))

        # Save config
        if hasattr(self.model, "cfg"):
            cfg_dict = self.model.cfg.__dict__
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(cfg_dict, f, indent=2)

        # Save tokenizer if available
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "save_pretrained"):
                    self.tokenizer.save_pretrained(path)
                elif hasattr(self.tokenizer, "save"):
                    self.tokenizer.save(path)
            except Exception as e:
                print(f"  Warning: Could not save tokenizer: {e}")

        size = sum(os.path.getsize(os.path.join(path, f))
                   for f in os.listdir(path)) / 1e6
        print(f"✅ Safetensors saved → {path} ({size:.1f} MB)")

    # ── LoRA adapter only ─────────────────────────────────────────
    def save_lora_adapter(self, path: str):
        """
        Save only the LoRA adapter weights — very small file (~MB).
        Base model weights are NOT saved (must be re-loaded separately).
        """
        from src.training.lora import save_adapter, LoRAConfig
        cfg = LoRAConfig()   # default, overridden by model's actual config
        save_adapter(self.model, path, cfg)

    # ── Merge LoRA and save ───────────────────────────────────────
    def merge_and_save(self, path: str):
        """
        Merge LoRA adapters into base weights, then save as safetensors.
        The merged model has no LoRA layers — it's a plain dense model.
        """
        from src.training.lora import merge_lora
        import copy
        merged = merge_lora(copy.deepcopy(self.model))
        exporter = ModelExporter(merged, self.tokenizer)
        exporter.save_safetensors(path)
        print(f"✅ Merged model saved → {path}")

    # ── FP16 conversion ───────────────────────────────────────────
    def save_fp16(self, path: str):
        """Convert to FP16 and save (halves file size)."""
        import copy
        model_fp16 = copy.deepcopy(self.model).half()
        exporter   = ModelExporter(model_fp16, self.tokenizer)
        exporter.save_safetensors(path)
        print(f"✅ FP16 model saved → {path}")

    # ── GGUF export (via llama.cpp) ───────────────────────────────
    def export_gguf(self, path: str, quantization: str = "q4_k_m"):
        """
        Convert to GGUF format for use with Ollama / llama.cpp.

        Requires llama.cpp installed:
            git clone https://github.com/ggerganov/llama.cpp
            cd llama.cpp && make

        Args:
            path:         output directory
            quantization: q4_k_m | q5_k_m | q8_0 | f16
        """
        # First save as safetensors (conversion input)
        tmp_path = path + "_tmp_safetensors"
        self.save_safetensors(tmp_path)

        # Look for llama.cpp convert script
        convert_script = self._find_llama_cpp_convert()
        if convert_script is None:
            print("⚠️  llama.cpp not found. To export GGUF:")
            print("   git clone https://github.com/ggerganov/llama.cpp")
            print(f"   python llama.cpp/convert-hf-to-gguf.py {tmp_path} "
                  f"--outfile {path}/model.gguf")
            print(f"   ./llama.cpp/llama-quantize {path}/model.gguf "
                  f"{path}/model_{quantization}.gguf {quantization.upper()}")
            print("\nSafetensors saved at:", tmp_path)
            print("Run the above commands manually to complete GGUF export.")
            return

        import subprocess
        os.makedirs(path, exist_ok=True)
        gguf_path = os.path.join(path, "model.gguf")

        # Convert to GGUF
        result = subprocess.run([
            "python3", convert_script, tmp_path,
            "--outfile", gguf_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Conversion failed:\n{result.stderr}")
            return

        # Quantize
        if quantization != "f16":
            quant_path = os.path.join(path, f"model_{quantization}.gguf")
            quant_bin  = os.path.join(os.path.dirname(convert_script),
                                       "..", "llama-quantize")
            if os.path.exists(quant_bin):
                subprocess.run([quant_bin, gguf_path, quant_path,
                                 quantization.upper()])
                size = os.path.getsize(quant_path) / 1e6
                print(f"✅ GGUF ({quantization}) saved → {quant_path} "
                      f"({size:.1f} MB)")
            else:
                print(f"✅ GGUF (f16) saved → {gguf_path}")

        # Cleanup temp
        shutil.rmtree(tmp_path, ignore_errors=True)

    def _find_llama_cpp_convert(self) -> Optional[str]:
        """Search common locations for llama.cpp convert script."""
        candidates = [
            "llama.cpp/convert-hf-to-gguf.py",
            "~/llama.cpp/convert-hf-to-gguf.py",
            "/opt/llama.cpp/convert-hf-to-gguf.py",
        ]
        for path in candidates:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded
        return None

    # ── Summary ───────────────────────────────────────────────────
    def export_summary(self, output_dir: str):
        """
        Save a full export package:
          - checkpoint.pt
          - model.safetensors
          - config.json
          - README.md (model card)
        """
        os.makedirs(output_dir, exist_ok=True)
        self.save_checkpoint(os.path.join(output_dir, "checkpoint.pt"))
        self.save_safetensors(output_dir)

        # Model card
        total = sum(p.numel() for p in self.model.parameters())
        cfg   = getattr(self.model, "cfg", None)

        readme = f"""# LLMForge Model Export

## Model Details
- **Parameters:** {total/1e6:.2f}M
- **FP16 Memory:** {total*2/1e6:.1f} MB
"""
        if cfg:
            readme += f"""
## Architecture
- **Positional Encoding:** {cfg.pos_encoding}
- **Normalization:** {cfg.norm_type}
- **Activation:** {cfg.activation}
- **Attention:** {cfg.attention_type}
- **Layers:** {cfg.n_layers}
- **d_model:** {cfg.d_model}
- **n_heads:** {cfg.n_heads}
- **d_ff:** {cfg.d_ff}
- **Vocab Size:** {cfg.vocab_size}
"""
        readme += """
## Usage
```python
from src.model import TransformerLM
model = TransformerLM.load("checkpoint.pt")
```

## Training
Built with LLMForge — IISc LLM Course Project
"""
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(readme)

        print(f"\n📦 Export package saved → {output_dir}/")
        print(f"   checkpoint.pt")
        print(f"   model.safetensors")
        print(f"   config.json")
        print(f"   README.md")
