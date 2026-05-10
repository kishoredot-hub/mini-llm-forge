"""
model.py — Assembles any transformer architecture from config.

Supports:
  - GPT-2 style  (learned pos + LayerNorm + GELU + standard MHA)
  - LLaMA style  (RoPE + RMSNorm + SwiGLU + GQA)
  - OLMo style   (ALiBi + LayerNorm + SwiGLU)
  - Mix & match any combination via config

Usage:
    cfg = ModelConfig.from_yaml("config/gpt2_style.yaml")
    model = TransformerLM(cfg)
    model = TransformerLM.from_preset("llama")
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .components.norms import build_norm
from .components.positional import build_pos_encoding
from .components.activations import build_activation
from .components.attention import build_attention


# ── Config dataclass ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Core dimensions
    vocab_size:   int   = 8000
    d_model:      int   = 256
    n_heads:      int   = 8
    n_kv_heads:   Optional[int] = None  # None = same as n_heads
    n_layers:     int   = 6
    d_ff:         int   = 1024
    max_seq:      int   = 512
    dropout:      float = 0.1

    # Swappable components
    pos_encoding:   str  = "learned"   # learned | rope | alibi | none
    norm_type:      str  = "layernorm" # layernorm | rmsnorm
    activation:     str  = "gelu"      # gelu | swiglu | relu | geglu
    attention_type: str  = "standard"  # standard | grouped_query
    rope_theta:     float = 10000.0

    # MoE (Mixture of Experts)
    use_moe:       bool = False
    num_experts:   int  = 8
    top_k:         int  = 2

    # ── Presets ───────────────────────────────────────────────────
    PRESETS = {
        "gpt2": dict(
            pos_encoding="learned", norm_type="layernorm",
            activation="gelu", attention_type="standard",
        ),
        "llama": dict(
            pos_encoding="rope", norm_type="rmsnorm",
            activation="swiglu", attention_type="grouped_query",
            dropout=0.0,
        ),
        "olmo": dict(
            pos_encoding="alibi", norm_type="layernorm",
            activation="swiglu", attention_type="standard",
        ),
        "tiny": dict(
            vocab_size=8000, d_model=128, n_heads=4,
            n_layers=4, d_ff=512, max_seq=256,
            pos_encoding="learned", norm_type="layernorm",
            activation="gelu", attention_type="standard",
        ),
    }

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        m = raw.get("model", raw)
        return cls(**{k: v for k, v in m.items() if hasattr(cls, k)})

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ModelConfig":
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{name}'. "
                             f"Available: {list(cls.PRESETS.keys())}")
        cfg = cls(**cls.PRESETS[name])
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# ── Transformer Block ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    Single transformer layer with fully swappable components.
    Supports Pre-LN (more stable training — used by default).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg  = cfg

        # Normalization
        self.ln1  = build_norm(cfg.norm_type, d_model=cfg.d_model)
        self.ln2  = build_norm(cfg.norm_type, d_model=cfg.d_model)

        # Attention
        self.attn = build_attention(
            cfg.attention_type,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            dropout=cfg.dropout,
        )

        # MLP / Feed-Forward
        self.mlp  = build_activation(cfg.activation,
                                      d_model=cfg.d_model,
                                      d_ff=cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:         torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        past_kv:   Optional[Tuple]        = None,
        use_cache: bool                   = False,
        rope:      Optional[nn.Module]    = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:

        # Pre-LN Attention (residual connection)
        attn_out, attn_weights, present_kv = self.attn(
            self.ln1(x),
            attn_bias=attn_bias,
            past_kv=past_kv,
            use_cache=use_cache,
            rope=rope,
        )
        x = x + self.drop(attn_out)

        # Pre-LN MLP (residual connection)
        x = x + self.drop(self.mlp(self.ln2(x)))

        return x, attn_weights, present_kv


# ── Full Transformer LM ───────────────────────────────────────────
class TransformerLM(nn.Module):
    """
    Decoder-only autoregressive language model.
    All components swappable via ModelConfig.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg      = cfg

        # Token embedding (always learned)
        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        # Positional encoding
        self.pos_enc  = build_pos_encoding(
            cfg.pos_encoding,
            d_model=cfg.d_model,
            d_head=cfg.d_model // cfg.n_heads,
            max_seq=cfg.max_seq,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            theta=cfg.rope_theta,
        )

        # Transformer blocks
        self.blocks   = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        # Final norm + LM head
        self.ln_f     = build_norm(cfg.norm_type, d_model=cfg.d_model)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying — share token embedding and output projection
        self.lm_head.weight = self.tok_emb.weight

        # Init weights
        self.apply(self._init_weights)

        total = sum(p.numel() for p in self.parameters())
        print(f"TransformerLM [{cfg.pos_encoding}|{cfg.norm_type}|"
              f"{cfg.activation}|{cfg.attention_type}] "
              f"— {total/1e6:.2f}M params")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        # Scale residual projections by 1/sqrt(2*n_layers) (GPT-2 trick)
        for name, p in module.named_parameters():
            if "out_proj.weight" in name or "down_proj.weight" in name:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    # ── RoPE reference for attention blocks ──────────────────────
    @property
    def rope(self):
        if self.cfg.pos_encoding == "rope":
            return self.pos_enc
        return None

    # ── Forward pass ─────────────────────────────────────────────
    def forward(
        self,
        idx:              torch.Tensor,
        targets:          Optional[torch.Tensor] = None,
        use_cache:        bool                   = False,
        past_key_values:  Optional[List]         = None,
    ) -> Tuple:

        B, T = idx.shape

        # Token embeddings
        x    = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)
        # Apply learned/ALiBi pos encoding to embeddings
        x    = self.pos_enc(x)
        x    = self.emb_drop(x)

        # ALiBi attention bias (if applicable)
        attn_bias = self.pos_enc.get_bias(T, self.cfg.n_heads, idx.device)

        # Run through transformer blocks
        all_attn_weights  = []
        all_present_kvs   = []

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            x, attn_w, present_kv = block(
                x,
                attn_bias=attn_bias,
                past_kv=past_kv,
                use_cache=use_cache,
                rope=self.rope,
            )
            all_attn_weights.append(attn_w)
            all_present_kvs.append(present_kv)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return dict(
            logits=logits,
            loss=loss,
            attn_weights=all_attn_weights,   # for visualization
            past_key_values=all_present_kvs if use_cache else None,
        )

    # ── Generation ────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        idx:          torch.Tensor,
        max_new_tokens: int   = 100,
        temperature:    float = 1.0,
        top_k:          int   = 50,
        top_p:          float = 0.9,
        do_sample:      bool  = True,
        eos_token_id:   Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache."""
        past_kvs = None

        for _ in range(max_new_tokens):
            # Only feed new token(s) when using cache
            idx_in = idx if past_kvs is None else idx[:, -1:]

            out      = self.forward(
                idx_in, use_cache=True, past_key_values=past_kvs
            )
            logits   = out["logits"][:, -1, :]   # last token
            past_kvs = out["past_key_values"]

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits = logits.masked_fill(logits < kth, -float("inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0]  = False
                sorted_logits[remove] = -float("inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs    = logits.softmax(-1)
            next_tok = (torch.multinomial(probs, 1) if do_sample
                        else logits.argmax(-1, keepdim=True))

            idx = torch.cat([idx, next_tok], dim=1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        return idx

    # ── Preset constructors ───────────────────────────────────────
    @classmethod
    def from_preset(cls, name: str, **overrides) -> "TransformerLM":
        cfg = ModelConfig.from_preset(name, **overrides)
        return cls(cfg)

    @classmethod
    def from_yaml(cls, path: str) -> "TransformerLM":
        cfg = ModelConfig.from_yaml(path)
        return cls(cfg)

    # ── Save / load ───────────────────────────────────────────────
    def save(self, path: str):
        torch.save({"cfg": self.cfg, "state": self.state_dict()}, path)
        print(f"Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TransformerLM":
        # Explicitly allow unpickling of ModelConfig
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt["cfg"])
        model.load_state_dict(ckpt["state"])
        return model.to(device)

