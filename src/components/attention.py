"""
attention.py — Swappable attention mechanisms.

Registered:
    standard       — Multi-Head Self-Attention (GPT-2)
    grouped_query  — Grouped Query Attention / GQA (LLaMA-2, Mistral)

Both support KV cache for efficient autoregressive generation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .registry import Registry


# ── Base class both implementations inherit ──────────────────────
class BaseAttention(nn.Module):
    """Shared utilities for attention implementations."""

    def _make_causal_mask(self, T: int, T_k: int,
                          device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask — prevents attending to future."""
        mask = torch.ones(T, T_k, device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=T_k - T + 1)   # handles cached k
        return mask   # True = block (set to -inf)

    def _split_heads(self, x: torch.Tensor, n_heads: int,
                     d_head: int) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, n_heads, d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)


@Registry.attention("standard")
class MultiHeadAttention(BaseAttention):
    """
    Standard Multi-Head Self-Attention (GPT-2, OLMo).
    n_kv_heads == n_heads (all heads have their own K and V).
    """

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1, bias: bool = True, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop     = nn.Dropout(dropout)
        self.scale    = self.d_head ** -0.5

    def forward(
        self,
        x:          torch.Tensor,
        attn_bias:  Optional[torch.Tensor] = None,
        past_kv:    Optional[Tuple]        = None,
        use_cache:  bool                   = False,
        rope:       Optional[nn.Module]    = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:

        B, T, C = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)

        Q = self._split_heads(Q, self.n_heads, self.d_head)
        K = self._split_heads(K, self.n_heads, self.d_head)
        V = self._split_heads(V, self.n_heads, self.d_head)

        # Apply RoPE to Q, K if provided
        if rope is not None and hasattr(rope, "apply_rope"):
            Q, K = rope.apply_rope(Q, K, seq_len=T)

        # Extend with KV cache for autoregressive generation
        if past_kv is not None:
            pk, pv = past_kv
            K = torch.cat([pk, K], dim=2)
            V = torch.cat([pv, V], dim=2)

        present_kv = (K, V) if use_cache else None
        T_k        = K.size(2)

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Add ALiBi / other positional bias if provided
        if attn_bias is not None:
            scores = scores + attn_bias[:, :T, :T_k]

        # Causal mask
        mask   = self._make_causal_mask(T, T_k, x.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn_weights = self.drop(scores.softmax(dim=-1))
        out          = self._merge_heads(torch.matmul(attn_weights, V))
        out          = self.out_proj(out)

        return out, attn_weights, present_kv


@Registry.attention("grouped_query")
class GroupedQueryAttention(BaseAttention):
    """
    Grouped Query Attention (LLaMA-2, Mistral, Falcon).

    Uses fewer K/V heads than Q heads.
    Each KV head is shared by (n_heads // n_kv_heads) Q heads.

    Benefits:
        - Reduces KV cache memory by n_heads/n_kv_heads
        - Faster inference (less KV bandwidth)
        - n_kv_heads=1 → Multi-Query Attention (MQA)
        - n_kv_heads=n_heads → standard MHA
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, bias: bool = False, **kwargs):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep      = n_heads // self.n_kv_heads  # repeats per KV head
        self.d_head     = d_model // n_heads
        self.scale      = self.d_head ** -0.5

        self.q_proj   = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.k_proj   = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.v_proj   = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=bias)
        self.drop     = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match Q heads by repeating."""
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        return (x.unsqueeze(3)
                 .expand(B, H, T, self.n_rep, D)
                 .reshape(B, H * self.n_rep, T, D))

    def forward(
        self,
        x:         torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        past_kv:   Optional[Tuple]        = None,
        use_cache: bool                   = False,
        rope:      Optional[nn.Module]    = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:

        B, T, _ = x.shape

        Q = self._split_heads(self.q_proj(x), self.n_heads, self.d_head)
        K = self._split_heads(self.k_proj(x), self.n_kv_heads, self.d_head)
        V = self._split_heads(self.v_proj(x), self.n_kv_heads, self.d_head)

        if rope is not None and hasattr(rope, "apply_rope"):
            Q, K = rope.apply_rope(Q, K, seq_len=T)

        if past_kv is not None:
            pk, pv = past_kv
            K = torch.cat([pk, K], dim=2)
            V = torch.cat([pv, V], dim=2)

        present_kv = (K, V) if use_cache else None
        T_k        = K.size(2)

        # Repeat KV heads to match Q head count
        K_full = self._repeat_kv(K)
        V_full = self._repeat_kv(V)

        scores = torch.matmul(Q, K_full.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            scores = scores + attn_bias[:, :T, :T_k]

        mask   = self._make_causal_mask(T, T_k, x.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn_weights = self.drop(scores.softmax(dim=-1))
        out          = self._merge_heads(torch.matmul(attn_weights, V_full))
        out          = self.out_proj(out)

        return out, attn_weights, present_kv


def build_attention(name: str, **kwargs) -> BaseAttention:
    return Registry.build_attention(name, **kwargs)
