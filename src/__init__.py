"""LLMForge — Modular LLM Toolkit."""

# Register all components on import
from src.components import norms, positional, activations, attention
from src.components.registry import Registry

__all__ = ["Registry"]
