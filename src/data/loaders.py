"""
data/loaders.py — Modular dataset loaders.

Registered datasets:
    tinystories   — TinyStories (HuggingFace)
    wikitext      — WikiText-103 (HuggingFace)
    openwebtext   — OpenWebText subset (HuggingFace)
    alpaca        — Stanford Alpaca instruction data
    dolly         — Databricks Dolly
    custom_text   — Any plain .txt file
    custom_csv    — CSV with 'text' column or instruction/output cols
    pdf           — Extract text from PDF files (Data Recipes style)

Usage:
    loader = DatasetLoader.from_config(cfg)
    train_ds, val_ds = loader.get_splits()
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.components.registry import Registry


# ── Text block dataset ───────────────────────────────────────────
class TextBlockDataset(Dataset):
    """Chunks a flat token sequence into fixed-size blocks."""

    def __init__(self, token_ids: List[int], block_size: int):
        self.data       = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, i):
        chunk = self.data[i: i + self.block_size + 1]
        return chunk[:-1], chunk[1:]   # (x, y) for next-token prediction


# ── Instruction dataset ──────────────────────────────────────────
class InstructionDataset(Dataset):
    """
    For supervised fine-tuning.
    Formats: {"instruction": ..., "input": ..., "output": ...}
    Masks loss on the prompt — only trains on the response.
    """

    def __init__(self, samples: List[dict], tokenizer,
                 max_length: int = 512, input_col: str = "instruction",
                 output_col: str = "output"):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.input_col  = input_col
        self.output_col = output_col

    def _format_prompt(self, sample: dict) -> Tuple[str, str]:
        inp  = sample.get(self.input_col, "")
        ctx  = sample.get("input", "")
        out  = sample.get(self.output_col, "")
        prompt = (f"### Instruction:\n{inp}\n"
                  + (f"### Context:\n{ctx}\n" if ctx else "")
                  + "### Response:\n")
        return prompt, out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        prompt, response = self._format_prompt(self.samples[i])
        full_text = prompt + response

        tok    = self.tokenizer
        ids    = tok.encode(full_text)[:self.max_length]
        p_ids  = tok.encode(prompt)

        # Build labels: -1 on prompt (masked), token ids on response
        labels = [-1] * len(p_ids) + ids[len(p_ids):]
        labels = labels[:self.max_length]

        # Pad to max_length
        pad_len = self.max_length - len(ids)
        ids     = ids + [tok.eos_token_id or 0] * pad_len
        labels  = labels + [-1] * pad_len

        return (torch.tensor(ids[:self.max_length], dtype=torch.long),
                torch.tensor(labels[:self.max_length], dtype=torch.long))


# ── Base loader ───────────────────────────────────────────────────
class BaseDatasetLoader:
    def __init__(self, tokenizer, block_size: int = 512,
                 train_split: float = 0.9, max_samples: Optional[int] = None):
        self.tokenizer   = tokenizer
        self.block_size  = block_size
        self.train_split = train_split
        self.max_samples = max_samples

    def load_text(self) -> str:
        raise NotImplementedError

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError


# ── HuggingFace loaders ───────────────────────────────────────────
@Registry.dataset("tinystories")
class TinyStoriesLoader(BaseDatasetLoader):
    """TinyStories — great for small model training, ~2GB."""

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        from datasets import load_dataset
        print("Loading TinyStories...")
        ds     = load_dataset("roneneldan/TinyStories", split="train")
        texts  = ds["text"]
        if self.max_samples:
            texts = texts[:self.max_samples]
        full   = " ".join(texts)
        tokens = self.tokenizer.encode(full)
        n      = int(len(tokens) * self.train_split)
        return (TextBlockDataset(tokens[:n],     self.block_size),
                TextBlockDataset(tokens[n:],     self.block_size))


@Registry.dataset("wikitext")
class WikitextLoader(BaseDatasetLoader):
    """WikiText-103 — standard LM benchmark."""

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        from datasets import load_dataset
        print("Loading WikiText-103...")
        train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        val   = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        train_text = " ".join(t for t in train["text"] if t.strip())
        val_text   = " ".join(t for t in val["text"]   if t.strip())
        if self.max_samples:
            train_text = train_text[:self.max_samples * 100]
        return (TextBlockDataset(self.tokenizer.encode(train_text), self.block_size),
                TextBlockDataset(self.tokenizer.encode(val_text),   self.block_size))


@Registry.dataset("openwebtext")
class OpenWebTextLoader(BaseDatasetLoader):
    """OpenWebText — GPT-2 training data replica."""

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        from datasets import load_dataset
        print("Loading OpenWebText (this may take a while)...")
        ds     = load_dataset("Skylion007/openwebtext", split="train",
                              streaming=True)
        texts  = []
        for i, ex in enumerate(ds):
            texts.append(ex["text"])
            if self.max_samples and i >= self.max_samples:
                break
        full   = " ".join(texts)
        tokens = self.tokenizer.encode(full)
        n      = int(len(tokens) * self.train_split)
        return (TextBlockDataset(tokens[:n], self.block_size),
                TextBlockDataset(tokens[n:], self.block_size))


@Registry.dataset("alpaca")
class AlpacaLoader(BaseDatasetLoader):
    """Stanford Alpaca — 52K instruction pairs."""

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        from datasets import load_dataset
        print("Loading Alpaca...")
        ds      = load_dataset("tatsu-lab/alpaca", split="train")
        samples = [dict(s) for s in ds]
        if self.max_samples:
            samples = samples[:self.max_samples]
        n       = int(len(samples) * self.train_split)
        return (InstructionDataset(samples[:n], self.tokenizer,
                                   self.block_size),
                InstructionDataset(samples[n:], self.tokenizer,
                                   self.block_size))


@Registry.dataset("dolly")
class DollyLoader(BaseDatasetLoader):
    """Databricks Dolly 15K — open instruction data."""

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        from datasets import load_dataset
        print("Loading Dolly-15K...")
        ds      = load_dataset("databricks/databricks-dolly-15k", split="train")
        samples = [{"instruction": s["instruction"],
                    "input":       s.get("context", ""),
                    "output":      s["response"]} for s in ds]
        if self.max_samples:
            samples = samples[:self.max_samples]
        n = int(len(samples) * self.train_split)
        return (InstructionDataset(samples[:n], self.tokenizer, self.block_size),
                InstructionDataset(samples[n:], self.tokenizer, self.block_size))


# ── Local file loaders ────────────────────────────────────────────
@Registry.dataset("custom_text")
class CustomTextLoader(BaseDatasetLoader):
    """Load any plain .txt file for pre-training."""

    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
        with open(self.path, encoding="utf-8", errors="replace") as f:
            text   = f.read()
        print(f"Loaded {len(text):,} chars from {self.path}")
        tokens = self.tokenizer.encode(text)
        n      = int(len(tokens) * self.train_split)
        return (TextBlockDataset(tokens[:n], self.block_size),
                TextBlockDataset(tokens[n:], self.block_size))


@Registry.dataset("custom_csv")
class CustomCSVLoader(BaseDatasetLoader):
    """
    Load CSV for pre-training or fine-tuning.
    Auto-detects mode from columns:
        - Has 'text' column → pre-training
        - Has instruction/output columns → fine-tuning
    """

    def __init__(self, path: str, input_col: str = "instruction",
                 output_col: str = "output", **kwargs):
        super().__init__(**kwargs)
        self.path       = path
        self.input_col  = input_col
        self.output_col = output_col

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        import csv
        with open(self.path, encoding="utf-8") as f:
            reader  = csv.DictReader(f)
            rows    = list(reader)

        if not rows:
            raise ValueError(f"Empty CSV: {self.path}")

        print(f"Loaded {len(rows)} rows from {self.path}")

        # Auto-detect mode
        if "text" in rows[0]:
            # Pre-training mode
            text   = " ".join(r["text"] for r in rows if r.get("text"))
            tokens = self.tokenizer.encode(text)
            n      = int(len(tokens) * self.train_split)
            return (TextBlockDataset(tokens[:n], self.block_size),
                    TextBlockDataset(tokens[n:], self.block_size))
        else:
            # Fine-tuning mode
            samples = [{"instruction": r.get(self.input_col, ""),
                        "output":      r.get(self.output_col, "")}
                       for r in rows]
            n = int(len(samples) * self.train_split)
            return (InstructionDataset(samples[:n], self.tokenizer,
                                       self.block_size,
                                       self.input_col, self.output_col),
                    InstructionDataset(samples[n:], self.tokenizer,
                                       self.block_size,
                                       self.input_col, self.output_col))


@Registry.dataset("pdf")
class PDFLoader(BaseDatasetLoader):
    """
    Data Recipes style: Extract text from PDF(s) for pre-training.
    Supports single file or directory of PDFs.
    """

    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def _extract(self, pdf_path: str) -> str:
        try:
            import fitz
            doc  = fitz.open(pdf_path)
            text = " ".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            raise ImportError("Install PyMuPDF: pip install PyMuPDF")

    def get_splits(self) -> Tuple[Dataset, Dataset]:
        if os.path.isdir(self.path):
            pdfs  = [os.path.join(self.path, f)
                     for f in os.listdir(self.path)
                     if f.endswith(".pdf")]
        else:
            pdfs  = [self.path]

        print(f"Extracting text from {len(pdfs)} PDF(s)...")
        texts  = [self._extract(p) for p in pdfs]
        full   = " ".join(texts)
        # Clean common PDF artifacts
        full   = re.sub(r'\s+', ' ', full).strip()
        print(f"Extracted {len(full):,} characters")
        tokens = self.tokenizer.encode(full)
        n      = int(len(tokens) * self.train_split)
        return (TextBlockDataset(tokens[:n], self.block_size),
                TextBlockDataset(tokens[n:], self.block_size))


# ── DataLoader factory ────────────────────────────────────────────
def make_dataloaders(
    dataset_name: str,
    tokenizer,
    batch_size:   int   = 32,
    block_size:   int   = 512,
    train_split:  float = 0.9,
    num_workers:  int   = 2,
    max_samples:  Optional[int] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function — builds train/val DataLoaders from config.

    Args:
        dataset_name: registered name (tinystories, alpaca, pdf, ...)
        tokenizer:    any tokenizer with .encode() method
        **kwargs:     forwarded to dataset loader (e.g., path=...)

    Returns:
        (train_loader, val_loader)
    """
    loader_cls = Registry.build("dataset", dataset_name,
                                tokenizer=tokenizer,
                                block_size=block_size,
                                train_split=train_split,
                                max_samples=max_samples,
                                **kwargs)
    train_ds, val_ds = loader_cls.get_splits()

    print(f"Dataset: {dataset_name} | "
          f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True)
    return train_dl, val_dl
