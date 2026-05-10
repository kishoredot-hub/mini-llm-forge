"""
norms.py — Swappable normalization layers.

Registered:
    layernorm  — standard LayerNorm  (GPT-2, OLMo)
    rmsnorm    — RMSNorm without mean subtraction (LLaMA, Mistral)
"""
import torch
import torch.nn as nn
from .registry import Registry


@Registry.norm("layernorm")
class LayerNorm(nn.Module):
    """Standard LayerNorm with learnable affine transform."""

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias   = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(
            x, x.shape[-1:], self.weight, self.bias, self.eps
        )


@Registry.norm("rmsnorm")
class RMSNorm(nn.Module):
    """
    RMSNorm — used in LLaMA, Mistral, Qwen.
    Faster than LayerNorm: no mean subtraction, no bias.
    Formula: x / RMS(x) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def build_norm(norm_type: str, d_model: int, **kwargs) -> nn.Module:
    """Build a norm layer from config string."""
    return Registry.build_norm(norm_type, d_model=d_model, **kwargs)
