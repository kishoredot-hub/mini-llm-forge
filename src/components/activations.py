"""
activations.py — Swappable MLP activation functions.

Registered:
    gelu    — Gaussian Error Linear Unit (GPT-2, BERT)
    swiglu  — SwiGLU gated activation (LLaMA, Mistral, PaLM)
    relu    — Classic ReLU
    geglu   — Gated GELU (T5, some variants)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import Registry


@Registry.activation("gelu")
class GELU(nn.Module):
    """Standard GELU — used in GPT-2."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


@Registry.activation("swiglu")
class SwiGLU(nn.Module):
    """
    SwiGLU — gated activation used in LLaMA, Mistral, PaLM.

    Instead of one linear layer, uses two gate projections:
        out = gate(x) * up(x)
        gate(x) = x * sigmoid(x)  [SiLU / Swish]
    Then projects down.

    Requires d_ff = 2/3 × 4d for same parameter count as GELU MLP.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Two parallel projections to d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) * up → element-wise gating
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


@Registry.activation("relu")
class ReLU(nn.Module):
    """Classic two-layer MLP with ReLU."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


@Registry.activation("geglu")
class GeGLU(nn.Module):
    """
    Gated GELU — used in T5 variants.
    out = GELU(gate(x)) * up(x)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x)) * self.up_proj(x)
        )


def build_activation(name: str, d_model: int, d_ff: int) -> nn.Module:
    return Registry.build_activation(name, d_model=d_model, d_ff=d_ff)
