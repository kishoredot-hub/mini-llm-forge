"""
tests/test_components.py — Unit tests for all modular components.
Run: python -m pytest tests/ -v
"""
import pytest
import torch
import sys, os
import src.data.loaders
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Register all components
import src.components.norms
import src.components.positional
import src.components.activations
import src.components.attention
from src.components.registry import Registry
from src.model import TransformerLM, ModelConfig


# ── Fixtures ──────────────────────────────────────────────────────
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tiny_cfg():
    return ModelConfig(
        vocab_size=100, d_model=64, n_heads=4,
        n_layers=2, d_ff=128, max_seq=32, dropout=0.0
    )


@pytest.fixture
def sample_input():
    return torch.randint(0, 100, (2, 16))


# ── Registry tests ────────────────────────────────────────────────
class TestRegistry:
    def test_norm_registered(self):
        assert "layernorm" in Registry.list("norm")
        assert "rmsnorm"   in Registry.list("norm")

    def test_pos_encoding_registered(self):
        for name in ["learned", "rope", "alibi", "none"]:
            assert name in Registry.list("pos_encoding")

    def test_activation_registered(self):
        for name in ["gelu", "swiglu", "relu", "geglu"]:
            assert name in Registry.list("activation")

    def test_attention_registered(self):
        for name in ["standard", "grouped_query"]:
            assert name in Registry.list("attention")

    def test_dataset_registered(self):
        for name in ["tinystories", "wikitext", "alpaca",
                     "custom_text", "custom_csv", "pdf"]:
            assert name in Registry.list("dataset")

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError):
            Registry.build_norm("nonexistent_norm", d_model=64)


# ── Norm tests ────────────────────────────────────────────────────
class TestNorms:
    @pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
    def test_forward_shape(self, norm_type):
        norm = Registry.build_norm(norm_type, d_model=64)
        x    = torch.randn(2, 16, 64)
        y    = norm(x)
        assert y.shape == x.shape

    def test_layernorm_normalizes(self):
        from src.components.norms import LayerNorm
        norm = LayerNorm(64)
        x    = torch.randn(2, 16, 64) * 100 + 50
        y    = norm(x)
        # After layernorm, mean ≈ 0, std ≈ 1 per token
        assert y.mean().abs().item() < 0.1

    def test_rmsnorm_no_bias(self):
        from src.components.norms import RMSNorm
        norm = RMSNorm(64)
        # RMSNorm has only weight, no bias
        assert not hasattr(norm, "bias") or norm.bias is None


# ── Positional encoding tests ─────────────────────────────────────
class TestPositionalEncoding:
    def test_learned_adds_to_embeddings(self):
        pos = Registry.build_pos_encoding("learned", d_model=64,
                                           max_seq=32, dropout=0.0)
        x   = torch.zeros(2, 16, 64)
        y   = pos(x)
        assert y.shape == (2, 16, 64)
        assert not torch.allclose(y, x)   # embeddings were added

    def test_rope_passthrough(self):
        pos = Registry.build_pos_encoding("rope", d_head=16,
                                           max_seq=32)
        x   = torch.randn(2, 16, 64)
        y   = pos(x)
        assert torch.allclose(y, x)   # RoPE doesn't modify embeddings

    def test_rope_apply(self):
        from src.components.positional import RotaryPositionalEncoding
        rope = RotaryPositionalEncoding(d_head=16, max_seq=32)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        q_rot, k_rot = rope.apply_rope(q, k, seq_len=8)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.allclose(q_rot, q)

    def test_alibi_bias_shape(self):
        pos = Registry.build_pos_encoding("alibi", n_heads=4, max_seq=32)
        bias = pos.get_bias(T=16, H=4, device=torch.device("cpu"))
        assert bias.shape == (4, 16, 16)

    def test_none_passthrough(self):
        pos = Registry.build_pos_encoding("none")
        x   = torch.randn(2, 16, 64)
        assert torch.allclose(pos(x), x)


# ── Activation tests ──────────────────────────────────────────────
class TestActivations:
    @pytest.mark.parametrize("act_type", ["gelu", "swiglu", "relu", "geglu"])
    def test_forward_shape(self, act_type):
        act = Registry.build_activation(act_type, d_model=64, d_ff=128)
        x   = torch.randn(2, 16, 64)
        y   = act(x)
        assert y.shape == x.shape

    def test_swiglu_gated(self):
        from src.components.activations import SwiGLU
        act = SwiGLU(d_model=64, d_ff=128)
        # SwiGLU has gate_proj, up_proj, down_proj — 3 linears
        assert hasattr(act, "gate_proj")
        assert hasattr(act, "up_proj")
        assert hasattr(act, "down_proj")


# ── Attention tests ───────────────────────────────────────────────
class TestAttention:
    def test_standard_mha_shape(self):
        attn = Registry.build_attention("standard", d_model=64,
                                         n_heads=4, dropout=0.0)
        x    = torch.randn(2, 16, 64)
        out, weights, kv = attn(x)
        assert out.shape     == (2, 16, 64)
        assert weights.shape == (2, 4, 16, 16)

    def test_gqa_shape(self):
        attn = Registry.build_attention("grouped_query", d_model=64,
                                         n_heads=4, n_kv_heads=2, dropout=0.0)
        x    = torch.randn(2, 16, 64)
        out, weights, kv = attn(x)
        assert out.shape == (2, 16, 64)

    def test_kv_cache(self):
        attn = Registry.build_attention("standard", d_model=64,
                                         n_heads=4, dropout=0.0)
        x    = torch.randn(1, 10, 64)
        _, _, kv = attn(x, use_cache=True)
        assert kv is not None
        k, v = kv
        assert k.shape[2] == 10   # seq len in cache

    def test_causal_mask(self):
        """Future tokens should not affect past token outputs."""
        attn = Registry.build_attention("standard", d_model=64,
                                         n_heads=4, dropout=0.0)
        attn.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out1, _, _ = attn(x)
            # Modify future tokens
            x_mod = x.clone()
            x_mod[:, 4:] = torch.randn_like(x_mod[:, 4:])
            out2, _, _ = attn(x_mod)
        # Past token outputs should be identical
        assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)


# ── Full model tests ──────────────────────────────────────────────
class TestTransformerLM:
    @pytest.mark.parametrize("preset", ["gpt2", "llama", "tiny"])
    def test_presets_build(self, preset):
        model = TransformerLM.from_preset(preset)
        assert model is not None

    def test_forward_loss(self, tiny_cfg, sample_input):
        model = TransformerLM(tiny_cfg)
        model.eval()
        x = sample_input
        y = torch.randint(0, 100, x.shape)
        with torch.no_grad():
            out = model(x, targets=y)
        assert "loss"   in out
        assert "logits" in out
        assert out["logits"].shape == (*x.shape, 100)
        assert out["loss"].item()  >  0

    def test_generation(self, tiny_cfg):
        model = TransformerLM(tiny_cfg)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        with torch.no_grad():
            gen = model.generate(prompt, max_new_tokens=10,
                                  do_sample=False)
        assert gen.shape == (1, 15)   # 5 + 10

    def test_kv_cache_consistency(self, tiny_cfg):
        """KV cache should produce identical output to no-cache."""
        model = TransformerLM(tiny_cfg)
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            out_no_cache = model(x)["logits"]
            out_cached   = model(x, use_cache=True)["logits"]
        assert torch.allclose(out_no_cache, out_cached, atol=1e-5)

    def test_save_load(self, tiny_cfg, tmp_path):
        model = TransformerLM(tiny_cfg)
        path  = str(tmp_path / "model.pt")
        model.save(path)
        loaded = TransformerLM.load(path)
        # Weights should be identical
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p1, p2)

    def test_attention_weights_returned(self, tiny_cfg, sample_input):
        model = TransformerLM(tiny_cfg)
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert len(out["attn_weights"]) == tiny_cfg.n_layers


# ── LoRA tests ────────────────────────────────────────────────────
class TestLoRA:
    def test_inject_reduces_trainable(self, tiny_cfg):
        from src.training.lora import LoRAConfig, inject_lora
        model      = TransformerLM(tiny_cfg)
        total_full = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)
        cfg        = LoRAConfig(rank=4, alpha=8)
        model      = inject_lora(model, cfg)
        trainable  = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)
        assert trainable < total_full

    def test_merge_restores_linear(self, tiny_cfg):
        from src.training.lora import LoRAConfig, inject_lora, merge_lora
        from src.components.attention import MultiHeadAttention
        import torch.nn as nn
        model = TransformerLM(tiny_cfg)
        cfg   = LoRAConfig(rank=4)
        model = inject_lora(model, cfg)
        model = merge_lora(model)
        # All LoRALinear layers should be replaced with plain Linear
        for m in model.modules():
            from src.training.lora import LoRALinear
            assert not isinstance(m, LoRALinear)

    def test_lora_output_differs_from_base(self, tiny_cfg, sample_input):
        """LoRA should change outputs (adapter is non-zero after init)."""
        from src.training.lora import LoRAConfig, inject_lora
        import copy
        model_base = TransformerLM(tiny_cfg)
        model_lora = copy.deepcopy(model_base)
        cfg        = LoRAConfig(rank=4, alpha=8)
        model_lora = inject_lora(model_lora, cfg)

        # Manually set lora_B to non-zero to force difference
        for m in model_lora.modules():
            from src.training.lora import LoRALinear
            if isinstance(m, LoRALinear):
                torch.nn.init.normal_(m.lora_B, std=0.1)

        with torch.no_grad():
            out_base = model_base(sample_input)["logits"]
            out_lora = model_lora(sample_input)["logits"]
        assert not torch.allclose(out_base, out_lora, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
