"""
registry.py — Central registry for all swappable components.

Usage:
    from src.components.registry import Registry

    # Register your own norm
    @Registry.norm("my_norm")
    class MyNorm(nn.Module): ...

    # Build from config
    norm = Registry.build_norm("my_norm", d_model=256)
"""
from __future__ import annotations
from typing import Dict, Type, Any
import torch.nn as nn


class _ComponentRegistry:
    """Holds {name → class} maps for each component type."""

    def __init__(self):
        self._stores: Dict[str, Dict[str, Type]] = {}

    def _get_store(self, kind: str) -> Dict[str, Type]:
        if kind not in self._stores:
            self._stores[kind] = {}
        return self._stores[kind]

    def register(self, kind: str, name: str):
        """Decorator: @Registry.register('norm', 'layernorm')"""
        def decorator(cls):
            store = self._get_store(kind)
            if name in store:
                raise ValueError(f"'{name}' already registered under '{kind}'")
            store[name] = cls
            return cls
        return decorator

    def build(self, kind: str, name: str, **kwargs) -> nn.Module:
        store = self._get_store(kind)
        if name not in store:
            available = list(store.keys())
            raise ValueError(
                f"Unknown {kind} '{name}'. Available: {available}"
            )
        print(f"[DEBUG] Building {kind}:{name} with kwargs={kwargs}")
        return store[name](**kwargs)

    def list(self, kind: str):
        return list(self._get_store(kind).keys())

    # ── Convenience decorators for each component type ─────────
    def norm(self, name: str):
        return self.register("norm", name)

    def pos_encoding(self, name: str):
        return self.register("pos_encoding", name)

    def activation(self, name: str):
        return self.register("activation", name)

    def attention(self, name: str):
        return self.register("attention", name)

    def optimizer(self, name: str):
        return self.register("optimizer", name)

    def lr_schedule(self, name: str):
        return self.register("lr_schedule", name)

    def dataset(self, name: str):
        return self.register("dataset", name)

    # ── Build shortcuts ────────────────────────────────────────
    def build_norm(self, name: str, **kw):
        return self.build("norm", name, **kw)

    def build_pos_encoding(self, name: str, **kw):
        return self.build("pos_encoding", name, **kw)

    def build_activation(self, name: str, **kw):
        return self.build("activation", name, **kw)

    def build_attention(self, name: str, **kw):
        return self.build("attention", name, **kw)

    def build_optimizer(self, name: str, **kw):
        return self.build("optimizer", name, **kw)

    def build_dataset(self, name: str, **kw):
        return self.build("dataset", name, **kw)

    def summary(self):
        print("=" * 50)
        print("Registered Components")
        print("=" * 50)
        for kind, store in self._stores.items():
            print(f"  {kind:20s}: {list(store.keys())}")


# Global singleton
Registry = _ComponentRegistry()
