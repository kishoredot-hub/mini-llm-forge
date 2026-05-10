"""
agent/react.py — ReAct agent with pluggable tools.
"""
import re
from typing import Callable, Optional, Tuple

class MiniAgent:
    SYSTEM = """You are a helpful LLM course assistant.
Tools available:
  calculator(expr)  — math calculations
  doc_search(query) — search knowledge base
  definition(term)  — LLM term definitions

Always respond in this format:
THINK: <reasoning>
ACTION: tool(input)

When you have the final answer:
THINK: <reasoning>
ANSWER: <final answer>
"""

    DEFINITIONS = {
        "attention": "Weighted sum of values using query-key similarity scores.",
        "transformer": "Decoder-only architecture using self-attention + FFN layers.",
        "tokenization": "Splitting text into subword units (tokens) for model input.",
        "lora": "Low-Rank Adaptation: efficient fine-tuning via rank-decomposed weight updates.",
        "rag": "Retrieval-Augmented Generation: grounding LLM answers in retrieved documents.",
        "perplexity": "exp(cross-entropy loss) — measures how well a model predicts text.",
        "bpe": "Byte-Pair Encoding: iteratively merges frequent character pairs into subwords.",
        "rmsnorm": "Root Mean Square normalization without mean subtraction — used in LLaMA.",
        "rope": "Rotary Position Embedding — encodes position by rotating Q/K vectors.",
        "alibi": "Attention with Linear Biases — subtracts distance penalty from attention scores.",
        "swiglu": "Gated activation: SiLU(gate) * up projection, used in LLaMA FFN.",
        "gqa": "Grouped Query Attention — fewer KV heads than Q heads, reduces KV cache.",
        "moe": "Mixture of Experts — routes each token to a subset of specialist FFN layers.",
        "kv cache": "Cached past key/value states for efficient autoregressive generation.",
    }

    def __init__(self, generate_fn: Callable, rag=None):
        self.generate = generate_fn
        self.rag      = rag

    def calculator(self, expr: str) -> str:
        allowed = set("0123456789.+-*/()** eE")
        if any(c not in allowed for c in expr.replace(" ", "")):
            return "Invalid — only math expressions allowed"
        try:
            return str(round(eval(expr), 6))
        except Exception as e:
            return f"Error: {e}"

    def doc_search(self, query: str) -> str:
        if self.rag is None:
            return "No knowledge base loaded."
        chunks = self.rag.retrieve(query, k=2)
        return " | ".join(c[:200] for c in chunks) if chunks else "Nothing found."

    def definition(self, term: str) -> str:
        return self.DEFINITIONS.get(term.lower().strip(),
                                     f"No definition for '{term}'.")

    def run(self, query: str, max_steps: int = 5) -> Tuple[str, str]:
        ctx   = f"{self.SYSTEM}\n\nUser: {query}\n"
        steps = []

        for i in range(max_steps):
            # Simplify generation call: only use max_new_tokens
            response = self.generate(ctx, max_new_tokens=200)
            response = response.strip()
            steps.append(f"Step {i+1}:\n{response}")

            if "ACTION:" in response:
                action = response.split("ACTION:")[-1].strip().split("\n")[0]
                m = re.match(r"(\w+)\((.+)\)", action.strip())
                if m:
                    tool, inp = m.group(1), m.group(2).strip("'\"")
                    result    = getattr(self, tool, lambda x: "Unknown tool")(inp)
                    steps.append(f"  → {tool}({inp}) = {result}")
                    ctx += f"{response}\nOBSERVATION: {result}\n"
                else:
                    ctx += f"{response}\nOBSERVATION: Could not parse tool call\n"

            elif "ANSWER:" in response:
                answer = response.split("ANSWER:")[-1].strip()
                return "\n".join(steps), answer

        return "\n".join(steps), "Max steps reached — no final answer."
