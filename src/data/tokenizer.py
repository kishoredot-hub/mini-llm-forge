"""
data/tokenizer.py — Train a custom BPE tokenizer on any corpus.

Supports:
  - Train from scratch on custom text / PDFs
  - Load pre-trained HuggingFace tokenizers
  - Unified interface: .encode() / .decode() / .vocab_size

Usage:
    # Train custom BPE
    tok = CustomTokenizer.train("data/corpus.txt", vocab_size=8000)
    tok.save("data/my_tokenizer")

    # Load HuggingFace tokenizer (same interface)
    tok = CustomTokenizer.from_pretrained("gpt2")

    # Use
    ids = tok.encode("Hello world")
    txt = tok.decode(ids)
"""
from __future__ import annotations
import os
from typing import List, Union, Optional


class CustomTokenizer:
    """
    Unified tokenizer wrapper.
    Train a BPE tokenizer from scratch OR wrap a HuggingFace tokenizer —
    both expose the same encode/decode interface.
    """

    def __init__(self, _tokenizer, _is_hf: bool = False):
        self._tok   = _tokenizer
        self._is_hf = _is_hf

    # ── Core interface ────────────────────────────────────────────
    def encode(self, text: str, add_special_tokens: bool = True,
               return_tensors: Optional[str] = None):
        if self._is_hf:
            out = self._tok.encode(text,
                                   add_special_tokens=add_special_tokens,
                                   return_tensors=return_tensors)
            return out
        else:
            ids = self._tok.encode(text).ids
            if return_tensors == "pt":
                import torch
                return torch.tensor([ids], dtype=torch.long)
            return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if self._is_hf:
            import torch
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, list) and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            return self._tok.decode(token_ids,
                                     skip_special_tokens=skip_special_tokens)
        else:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            return self._tok.decode(token_ids)

    def batch_encode(self, texts: List[str], padding: bool = True,
                     max_length: int = 512, return_tensors: str = "pt"):
        if self._is_hf:
            return self._tok(texts, padding=padding, truncation=True,
                             max_length=max_length, return_tensors=return_tensors)
        else:
            import torch
            all_ids = [self._tok.encode(t).ids[:max_length] for t in texts]
            if padding:
                max_len = max(len(ids) for ids in all_ids)
                pad_id  = self.pad_token_id or 0
                all_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
            return {"input_ids": torch.tensor(all_ids, dtype=torch.long)}

    @property
    def vocab_size(self) -> int:
        if self._is_hf:
            return len(self._tok)
        return self._tok.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        if self._is_hf:
            return self._tok.eos_token_id or 0
        vocab = self._tok.get_vocab()
        return vocab.get("<eos>", vocab.get("</s>", 0))

    @property
    def pad_token_id(self) -> int:
        if self._is_hf:
            return self._tok.pad_token_id or self._tok.eos_token_id or 0
        vocab = self._tok.get_vocab()
        return vocab.get("<pad>", 0)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        if self._is_hf:
            return self._tok.convert_ids_to_tokens(ids)
        vocab_rev = {v: k for k, v in self._tok.get_vocab().items()}
        return [vocab_rev.get(i, f"[{i}]") for i in ids]

    # ── Constructors ──────────────────────────────────────────────
    @classmethod
    def train(
        cls,
        source: Union[str, List[str]],
        vocab_size:    int   = 8000,
        min_frequency: int   = 2,
        special_tokens: List[str] = None,
        save_path:     Optional[str] = None,
    ) -> "CustomTokenizer":
        """
        Train a BPE tokenizer from scratch on a text file, list of
        files, or a directory of .txt / .pdf files.

        Args:
            source:        path to text file, list of paths, or directory
            vocab_size:    target vocabulary size
            min_frequency: minimum token frequency to include
            save_path:     where to save the trained tokenizer

        Returns:
            CustomTokenizer instance
        """
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError("pip install tokenizers")

        if special_tokens is None:
            special_tokens = ["<pad>", "<eos>", "<unk>", "<mask>",
                               "<s>", "</s>"]

        # Collect file list
        if isinstance(source, str):
            if os.path.isdir(source):
                files = _collect_files(source)
            else:
                files = [source]
        else:
            files = source

        # Extract text from PDFs if needed, write to temp .txt files
        txt_files = []
        for f in files:
            if f.endswith(".pdf"):
                txt_path = f.replace(".pdf", "_extracted.txt")
                _pdf_to_txt(f, txt_path)
                txt_files.append(txt_path)
            elif f.endswith(".txt"):
                txt_files.append(f)

        if not txt_files:
            raise ValueError(f"No .txt or .pdf files found in: {source}")

        print(f"Training BPE tokenizer on {len(txt_files)} file(s) | "
              f"vocab_size={vocab_size}")

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=txt_files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            tokenizer.save_model(save_path)
            print(f"Tokenizer saved → {save_path}")

        # Report compression stats
        with open(txt_files[0], encoding="utf-8", errors="replace") as f:
            sample = f.read(10000)
        n_chars  = len(sample)
        n_tokens = len(tokenizer.encode(sample).ids)
        print(f"Compression ratio: {n_chars/max(n_tokens,1):.2f} chars/token")

        return cls(tokenizer, _is_hf=False)

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "CustomTokenizer":
        """
        Load any HuggingFace tokenizer.
        Examples: 'gpt2', 'meta-llama/Llama-3.2-1B', 'microsoft/phi-2'
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("pip install transformers")

        tok = AutoTokenizer.from_pretrained(name_or_path,
                                             trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"Loaded tokenizer: {name_or_path} | vocab={len(tok):,}")
        return cls(tok, _is_hf=True)

    @classmethod
    def load(cls, path: str) -> "CustomTokenizer":
        """Load a previously trained BPE tokenizer."""
        from tokenizers import ByteLevelBPETokenizer
        tok = ByteLevelBPETokenizer(
            os.path.join(path, "vocab.json"),
            os.path.join(path, "merges.txt"),
        )
        return cls(tok, _is_hf=False)

    def save(self, path: str):
        """Save tokenizer to directory."""
        os.makedirs(path, exist_ok=True)
        if self._is_hf:
            self._tok.save_pretrained(path)
        else:
            self._tok.save_model(path)
        print(f"Tokenizer saved → {path}")

    def __repr__(self):
        return f"CustomTokenizer(vocab_size={self.vocab_size}, is_hf={self._is_hf})"


# ── Helpers ───────────────────────────────────────────────────────
def _collect_files(directory: str) -> List[str]:
    files = []
    for fname in os.listdir(directory):
        if fname.endswith((".txt", ".pdf")):
            files.append(os.path.join(directory, fname))
    return sorted(files)


def _pdf_to_txt(pdf_path: str, txt_path: str):
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    except ImportError:
        raise ImportError("pip install PyMuPDF for PDF support")
