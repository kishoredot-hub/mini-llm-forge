"""
scripts/quickstart.py — Demonstrates all modular features.
Run: python scripts/quickstart.py

Shows:
  1. Build 4 different architectures from presets
  2. Compare parameter counts
  3. Run a forward pass
  4. Show all registered components
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from src.model import TransformerLM, ModelConfig
from src.components.registry import Registry

# Force all components to register
import src.components.norms
import src.components.positional
import src.components.activations
import src.components.attention


def separator(title=""):
    print(f"\n{'─'*55}")
    if title:
        print(f"  {title}")
        print(f"{'─'*55}")


def main():
    print("=" * 55)
    print("  LLMForge — Modular Architecture Demo")
    print("=" * 55)

    # ── Show all registered components ───────────────────────────
    separator("Registered Components")
    Registry.summary()

    # ── Build 4 different architectures ──────────────────────────
    separator("Architecture Presets")

    presets = ["gpt2", "llama", "olmo", "tiny"]
    models  = {}

    for preset in presets:
        print(f"\nBuilding model from preset: '{preset}'...")
        model       = TransformerLM.from_preset(preset)
        total       = sum(p.numel() for p in model.parameters())
        trainable   = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
        cfg         = model.cfg
        models[preset] = model
        print(f"\n  [{preset.upper()}]")
        print(f"    pos_encoding  : {cfg.pos_encoding}")
        print(f"    norm_type     : {cfg.norm_type}")
        print(f"    activation    : {cfg.activation}")
        print(f"    attention     : {cfg.attention_type}")
        print(f"    params        : {total/1e6:.2f}M")
        print(f"    FP16 memory   : {total*2/1e6:.1f} MB")

    # ── Custom mix-and-match ──────────────────────────────────────
    separator("Custom Architecture (RoPE + RMSNorm + SwiGLU + Standard MHA)")

    custom_cfg = ModelConfig(
        vocab_size=8000, d_model=256, n_heads=8,
        n_layers=6, d_ff=1024,
        pos_encoding="rope",      # LLaMA-style
        norm_type="rmsnorm",      # LLaMA-style
        activation="swiglu",      # LLaMA-style
        attention_type="standard" # GPT-2 style (not GQA)
    )
    custom_model = TransformerLM(custom_cfg)
    print(f"  Custom mix built successfully!")

    # ── Forward pass test ─────────────────────────────────────────
    separator("Forward Pass Test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = models["gpt2"].to(device)

    batch_size = 2
    seq_len    = 64
    x          = torch.randint(0, 8000, (batch_size, seq_len)).to(device)
    y          = torch.randint(0, 8000, (batch_size, seq_len)).to(device)

    with torch.no_grad():
        out = model(x, targets=y)

    print(f"  Input shape  : {x.shape}")
    print(f"  Logits shape : {out['logits'].shape}")
    print(f"  Loss         : {out['loss'].item():.4f}")
    print(f"  n_attn_maps  : {len(out['attn_weights'])} layers")
    print(f"  Attn map[0]  : {out['attn_weights'][0].shape}")

    # ── Generation test ───────────────────────────────────────────
    separator("Generation Test (KV Cache)")

    prompt   = torch.randint(0, 8000, (1, 10)).to(device)
    with torch.no_grad():
        generated = model.generate(
            prompt, max_new_tokens=20, temperature=1.0,
            top_k=50, do_sample=True
        )
    print(f"  Prompt tokens    : {prompt.shape[1]}")
    print(f"  Generated tokens : {generated.shape[1]}")

    # ── LoRA injection demo ───────────────────────────────────────
    separator("LoRA Injection Demo")

    from src.training.lora import LoRAConfig, inject_lora, save_adapter
    lora_cfg = LoRAConfig(rank=8, alpha=16, target_modules=["qkv", "out_proj"])
    model    = models["gpt2"]
    model    = inject_lora(model, lora_cfg)

    # ── Dataset registry demo ─────────────────────────────────────
    separator("Dataset Registry")
    print("  Available datasets:")
    for name in Registry.list("dataset"):
        print(f"    - {name}")

    print("\n  Usage examples:")
    print("    make_dataloaders('tinystories', tokenizer)")
    print("    make_dataloaders('alpaca', tokenizer)")
    print("    make_dataloaders('pdf', tokenizer, path='data/pdfs/')")
    print("    make_dataloaders('custom_csv', tokenizer, path='my_data.csv')")

    separator("✅ All systems working!")
    print("  Next steps:")
    print("    python app/app.py          # Launch Gradio app")
    print("    jupyter lab                # Open notebooks")
    print("    python -c 'from src.model import TransformerLM; m = TransformerLM.from_preset(\"llama\")'")


if __name__ == "__main__":
    main()
