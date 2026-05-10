"""
training/lora.py — LoRA and QLoRA from scratch.
Supports: inject, merge, save adapter, load adapter.

Inspired by Unsloth's approach: efficient fine-tuning
with optional quantization (QLoRA via bitsandbytes).
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank:           int   = 8
    alpha:          int   = 16
    dropout:        float = 0.05
    target_modules: List[str] = None   # None = auto-detect Linear layers
    bias:           str   = "none"     # none | all | lora_only
    # QLoRA settings
    quantize:       bool  = False
    quant_bits:     int   = 4

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["qkv", "out_proj", "q_proj",
                                    "k_proj", "v_proj"]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


# ── LoRA Layer ────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    Wraps a frozen Linear layer with trainable low-rank matrices.

    W_new = W_frozen + B @ A * scale
    where A ∈ R^(rank × d_in), B ∈ R^(d_out × rank)

    A initialized with Kaiming uniform (non-zero)
    B initialized to zero (so LoRA starts as identity)
    """

    def __init__(self, original: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        self.original = original
        self.cfg      = cfg
        d_out, d_in   = original.weight.shape

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

        # Trainable adapters
        self.lora_A   = nn.Parameter(
            torch.empty(cfg.rank, d_in)
        )
        self.lora_B   = nn.Parameter(
            torch.zeros(d_out, cfg.rank)
        )
        self.dropout  = nn.Dropout(cfg.dropout)

        # Kaiming init for A (fan_in mode)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base     = self.original(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)
        return base + lora_out * self.cfg.scale

    def merge_weights(self) -> nn.Linear:
        """Merge adapter into original weight → return plain Linear."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        merged.weight.data = (
            self.original.weight.data
            + (self.lora_B @ self.lora_A) * self.cfg.scale
        )
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data.clone()
        return merged

    def extra_repr(self) -> str:
        return (f"rank={self.cfg.rank}, alpha={self.cfg.alpha}, "
                f"scale={self.cfg.scale:.3f}")


# ── Inject / Remove LoRA ──────────────────────────────────────────
def inject_lora(model: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """
    Replace target Linear layers with LoRA-wrapped versions.
    Returns modified model.
    """
    replaced = 0

    def _inject(parent: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and _should_adapt(name, cfg):
                setattr(parent, name, LoRALinear(child, cfg))
                replaced += 1
            else:
                _inject(child, full_name)

    _inject(model)

    # Bias handling
    if cfg.bias == "none":
        for n, p in model.named_parameters():
            if "bias" in n and "lora" not in n:
                p.requires_grad = False
    elif cfg.bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif cfg.bias == "lora_only":
        for n, p in model.named_parameters():
            if "lora" in n.lower() and "bias" in n:
                p.requires_grad = True

    # Stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"LoRA injected: {replaced} layers | "
          f"Trainable: {trainable:,}/{total:,} "
          f"({100*trainable/total:.3f}%)")
    return model


def _should_adapt(name: str, cfg: LoRAConfig) -> bool:
    return any(t in name for t in cfg.target_modules)


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters into base weights (for inference)."""
    merged = 0

    def _merge(parent: nn.Module):
        nonlocal merged
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merge_weights())
                merged += 1
            else:
                _merge(child)

    _merge(model)
    print(f"Merged {merged} LoRA layers into base weights")
    return model


# ── Save / Load LoRA Adapter ──────────────────────────────────────
def save_adapter(model: nn.Module, path: str, cfg: LoRAConfig):
    """Save only the LoRA adapter weights (small file ~MB)."""
    os.makedirs(path, exist_ok=True)
    adapter_weights = {
        name: param.data
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    }
    torch.save(adapter_weights, os.path.join(path, "adapter.pt"))

    # Save config
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump({
            "rank": cfg.rank, "alpha": cfg.alpha,
            "dropout": cfg.dropout,
            "target_modules": cfg.target_modules,
            "bias": cfg.bias,
        }, f, indent=2)

    size_mb = sum(v.numel() * 2 for v in adapter_weights.values()) / 1e6
    print(f"Adapter saved → {path} ({size_mb:.1f} MB, "
          f"{len(adapter_weights)} tensors)")


def load_adapter(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA adapter weights into an already-injected model."""
    weights = torch.load(os.path.join(path, "adapter.pt"),
                         map_location="cpu")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"Adapter loaded from {path} | "
          f"missing={len(missing)} | unexpected={len(unexpected)}")
    return model


# ── QLoRA helper ─────────────────────────────────────────────────
def load_model_4bit(model_id: str, device_map: str = "auto"):
    """
    Load any HuggingFace model in 4-bit (QLoRA-ready).
    Requires: pip install bitsandbytes transformers
    """
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg,
            device_map=device_map, trust_remote_code=True,
        )
        print(f"Loaded {model_id} in 4-bit (QLoRA-ready)")
        return model
    except ImportError:
        raise ImportError(
            "QLoRA requires: pip install bitsandbytes transformers"
        )
