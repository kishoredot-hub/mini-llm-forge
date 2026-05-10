"""
positional.py — Swappable positional encoding strategies.

Registered:
    learned  — Learned positional embeddings (GPT-2)
    rope     — Rotary Position Embedding (LLaMA, Mistral)
    alibi    — ALiBi: Attention with Linear Biases (OLMo, BLOOM)
    none     — No positional encoding (useful for ablations)
"""
import math
import torch
import torch.nn as nn
from .registry import Registry


@Registry.pos_encoding("learned")
class LearnedPositionalEncoding(nn.Module):
    """
    Learned position embeddings — GPT-2 style.
    Each position gets a trainable embedding vector.
    """

    def __init__(self, max_seq: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.emb  = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) — adds position embeddings."""
        T   = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return self.drop(x + self.emb(pos))

    def get_bias(self, T: int, H: int, device) -> None:
        return None   # No attention bias needed


@Registry.pos_encoding("rope")
class RotaryPositionalEncoding(nn.Module):
    """
    RoPE — Rotary Position Embedding (LLaMA, Mistral, Qwen).

    Applied to Q and K inside the attention layer.
    Does NOT add to embeddings — instead rotates Q/K vectors.

    Key insight: rotating Q and K by position-dependent angles
    makes dot-product QK^T depend only on relative position.
    """

    def __init__(self, d_head: int, max_seq: int = 2048,
                 theta: float = 10000.0, **kwargs):
        super().__init__()
        self.d_head = d_head
        # Precompute frequency bands
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_head, 2).float() / d_head)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Cache cos/sin for max_seq
        self._cache_cos_sin(max_seq)

    def _cache_cos_sin(self, max_seq: int):
        t     = torch.arange(max_seq, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)           # (T, d_head/2)
        emb   = torch.cat([freqs, freqs], dim=-1)       # (T, d_head)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate by splitting in half and negating second half."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                   seq_len: int) -> tuple:
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        q   = (q * cos) + (self._rotate_half(q) * sin)
        k   = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RoPE is applied inside attention, not to embeddings."""
        return x   # Pass-through; rotation happens in attention

    def get_bias(self, T: int, H: int, device) -> None:
        return None


@Registry.pos_encoding("alibi")
class ALiBiPositionalEncoding(nn.Module):
    """
    ALiBi — Attention with Linear Biases (OLMo, BLOOM, MPT).

    Instead of adding embeddings, subtracts a linear distance
    penalty from attention scores. No learned parameters.

    Key insight: attention(q_i, k_j) -= m_h * |i - j|
    where m_h is a head-specific slope (geometric sequence).
    """

    def __init__(self, n_heads: int, max_seq: int = 2048, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        slopes       = self._get_slopes(n_heads)
        # (H, 1, max_seq) bias — distance from each position
        alibi = slopes.unsqueeze(1) * torch.arange(max_seq).unsqueeze(0)
        self.register_buffer("alibi", alibi)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        """Geometric sequence of slopes: 2^(-8/n), ..."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        # Interpolate for non-power-of-2 head counts
        closest = 2 ** math.floor(math.log2(n_heads))
        base    = get_slopes_power_of_2(closest)
        extra   = get_slopes_power_of_2(2 * closest)[0::2]
        return torch.tensor((base + extra)[:n_heads])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x   # ALiBi adds to attention scores, not embeddings

    def get_bias(self, T: int, H: int, device) -> torch.Tensor:
        """Returns (H, T, T) ALiBi bias for attention scores."""
        bias = self.alibi[:H, :T].to(device)           # (H, T)
        # Build causal distance matrix: bias[i, j] = -slope * (i-j)
        positions = torch.arange(T, device=device)
        dist      = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T,T)
        dist      = dist.clamp(max=0).abs()             # only look back
        return -bias.unsqueeze(-1) * dist.unsqueeze(0)  # (H, T, T)


@Registry.pos_encoding("none")
class NoPositionalEncoding(nn.Module):
    """No positional encoding — useful for ablation studies."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_bias(self, T, H, device):
        return None


def build_pos_encoding(name: str, **kwargs) -> nn.Module:
    return Registry.build_pos_encoding(name, **kwargs)
