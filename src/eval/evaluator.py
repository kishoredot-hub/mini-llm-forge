"""
eval/evaluator.py — Comprehensive evaluation suite.

Metrics:
  - Perplexity (language model quality)
  - BLEU / ROUGE (generation quality)
  - RAG faithfulness (grounded answer quality)
  - Agent success rate (tool-calling accuracy)
  - Inference speed (tok/s, latency, VRAM)
  - LoRA ablation (rank vs quality)
  - Prompt technique comparison
"""
from __future__ import annotations
import math, time
from typing import List, Dict, Optional, Callable, Tuple
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class Evaluator:
    """
    All evaluation metrics in one class.
    Works with: TinyGPT, HuggingFace models, or any model with
    a forward(input_ids, labels) method.
    """

    def __init__(self, model, tokenizer, device: str = "auto"):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = (device if device != "auto" else
                          "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ══════════════════════════════════════════════════════════════
    # 1. Language Model Quality
    # ══════════════════════════════════════════════════════════════
    def perplexity(self, texts: List[str], batch_size: int = 4) -> Dict:
        """
        Compute perplexity on a list of texts.
        PPL = exp(mean cross-entropy loss)
        Lower is better.
        """
        ppls = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            for text in batch:
                ids = self.tokenizer.encode(text, return_tensors="pt")
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                ids = ids.to(self.device)
                if ids.shape[1] < 2:
                    continue
                with torch.no_grad():
                    out  = self.model(ids, labels=ids)
                    loss = out.loss if hasattr(out, "loss") else (
                        out["loss"] if isinstance(out, dict) else out[1]
                    )
                ppls.append(math.exp(min(loss.item(), 20)))

        return {
            "mean_ppl": round(float(np.mean(ppls)), 3),
            "std_ppl":  round(float(np.std(ppls)),  3),
            "min_ppl":  round(float(np.min(ppls)),  3),
            "max_ppl":  round(float(np.max(ppls)),  3),
            "n_samples": len(ppls),
        }

    # ══════════════════════════════════════════════════════════════
    # 2. Generation Quality
    # ══════════════════════════════════════════════════════════════
    def bleu_score(self, predictions: List[str],
                   references: List[str]) -> Dict:
        """BLEU score for generation quality."""
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            import nltk
            nltk.download("punkt", quiet=True)

            refs  = [[r.lower().split()] for r in references]
            hyps  = [p.lower().split()   for p in predictions]
            sf    = SmoothingFunction().method1

            return {
                "bleu_1": round(corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=sf), 4),
                "bleu_2": round(corpus_bleu(refs, hyps, weights=(.5,.5,0,0), smoothing_function=sf), 4),
                "bleu_4": round(corpus_bleu(refs, hyps, weights=(.25,.25,.25,.25), smoothing_function=sf), 4),
            }
        except ImportError:
            return {"error": "pip install nltk"}

    def rouge_score(self, predictions: List[str],
                    references: List[str]) -> Dict:
        """ROUGE scores for summarization / generation."""
        try:
            from rouge_score import rouge_scorer
            scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
            scores  = {"rouge1": [], "rouge2": [], "rougeL": []}
            for pred, ref in zip(predictions, references):
                s = scorer.score(ref, pred)
                for k in scores:
                    scores[k].append(s[k].fmeasure)
            return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
        except ImportError:
            return {"error": "pip install rouge-score"}

    # ══════════════════════════════════════════════════════════════
    # 3. RAG Evaluation
    # ══════════════════════════════════════════════════════════════
    def rag_faithfulness(self, qa_pairs: List[Tuple[str, str]],
                          rag) -> Dict:
        """
        Faithfulness: does the answer come only from retrieved context?
        Simple overlap-based check (use NLI model for production).

        Args:
            qa_pairs: list of (question, reference_answer)
            rag:      RAGPipeline instance
        """
        faithful   = 0
        gap_count  = 0
        results    = []

        for question, ref_answer in qa_pairs:
            chunks   = rag.retrieve(question, k=3)
            context  = " ".join(chunks).lower()
            ref_words= ref_answer.lower().split()
            overlap  = sum(1 for w in ref_words if w in context)
            ratio    = overlap / max(len(ref_words), 1)
            is_faith = ratio > 0.4
            is_gap   = len(chunks) == 0

            faithful  += int(is_faith)
            gap_count += int(is_gap)
            results.append({
                "question":    question,
                "faithful":    is_faith,
                "overlap_pct": round(ratio * 100, 1),
                "gap":         is_gap,
            })

        return {
            "faithfulness":    round(faithful / max(len(qa_pairs), 1), 3),
            "gap_rate":        round(gap_count / max(len(qa_pairs), 1), 3),
            "n_evaluated":     len(qa_pairs),
            "details":         results,
        }

    def retrieval_recall(self, qa_pairs: List[Tuple[str, str, str]],
                          rag, k: int = 5) -> Dict:
        """
        Recall@k: does the correct source appear in top-k retrieved chunks?

        Args:
            qa_pairs: list of (question, answer, source_passage)
        """
        hits = 0
        for question, answer, source in qa_pairs:
            chunks = rag.retrieve(question, k=k)
            # Check if any chunk overlaps significantly with source
            source_words = set(source.lower().split())
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                iou = len(source_words & chunk_words) / max(
                    len(source_words | chunk_words), 1
                )
                if iou > 0.3:
                    hits += 1
                    break
        return {
            f"recall@{k}": round(hits / max(len(qa_pairs), 1), 3),
            "n_queries": len(qa_pairs),
        }

    # ══════════════════════════════════════════════════════════════
    # 4. Inference Speed
    # ══════════════════════════════════════════════════════════════
    def speed_benchmark(
        self,
        prompt:       str,
        n_tokens:     int   = 100,
        n_warmup:     int   = 3,
        n_timed:      int   = 5,
        batch_sizes:  List[int] = None,
        precisions:   List[str] = None,
    ) -> pd.DataFrame:
        """
        Comprehensive inference speed benchmark.
        Tests multiple batch sizes and precision modes.
        """
        if batch_sizes is None:
            batch_sizes = [1]
        if precisions is None:
            precisions = ["current"]

        rows = []
        ids  = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids

        for bs in batch_sizes:
            batch_ids = ids.repeat(bs, 1).to(self.device)

            # Warm-up
            with torch.no_grad():
                for _ in range(n_warmup):
                    self.model.generate(
                        batch_ids, max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id or 0,
                    ) if hasattr(self.model, "generate") else self.model(batch_ids)

            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            latencies = []
            for _ in range(n_timed):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    if hasattr(self.model, "generate"):
                        self.model.generate(
                            batch_ids, max_new_tokens=n_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id or 0,
                        )
                    else:
                        self.model(batch_ids)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.time() - t0)

            avg_s  = np.mean(latencies)
            vram   = (torch.cuda.max_memory_allocated() / 1e6
                      if self.device == "cuda" else 0)

            rows.append({
                "batch_size":     bs,
                "tok/sec":        round(bs * n_tokens / avg_s, 2),
                "latency_ms":     round(avg_s * 1000 / n_tokens, 2),
                "total_time_s":   round(avg_s, 3),
                "vram_mb":        round(vram, 1),
            })

        return pd.DataFrame(rows)

    def compare_precisions(self, prompt: str, n_tokens: int = 50) -> pd.DataFrame:
        """Compare FP32 vs FP16 vs 4-bit throughput and quality."""
        import copy
        results = []

        configs = {
            "FP32":  {"dtype": torch.float32},
            "FP16":  {"dtype": torch.float16},
        }

        # Try 4-bit if bitsandbytes available
        try:
            import bitsandbytes
            configs["4-bit"] = {"quantize": True}
        except ImportError:
            pass

        base_ppl = None
        for name, cfg in configs.items():
            try:
                if "quantize" in cfg:
                    # Skip if model doesn't support quantization natively
                    continue

                m = copy.deepcopy(self.model)
                if cfg.get("dtype") == torch.float16:
                    m = m.half()
                m = m.to(self.device)

                # Speed
                ids  = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    if hasattr(m, "generate"):
                        m.generate(ids, max_new_tokens=n_tokens, do_sample=False,
                                   pad_token_id=self.tokenizer.pad_token_id or 0)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - t0
                vram    = torch.cuda.max_memory_allocated() / 1e6 if self.device == "cuda" else 0

                # Perplexity
                ppl_result = Evaluator(m, self.tokenizer, self.device).perplexity([prompt])
                ppl        = ppl_result["mean_ppl"]
                if base_ppl is None:
                    base_ppl = ppl

                results.append({
                    "precision":  name,
                    "tok/sec":    round(n_tokens / elapsed, 2),
                    "vram_mb":    round(vram, 1),
                    "perplexity": ppl,
                    "ppl_delta%": round(abs(ppl - base_ppl) / max(base_ppl, 1) * 100, 3),
                })
                del m
            except Exception as e:
                results.append({
                    "precision": name, "tok/sec": "N/A",
                    "vram_mb": "N/A", "perplexity": "N/A",
                    "ppl_delta%": f"Error: {e}"
                })

        return pd.DataFrame(results)

    # ══════════════════════════════════════════════════════════════
    # 5. LoRA Ablation
    # ══════════════════════════════════════════════════════════════
    def lora_rank_ablation(
        self,
        base_model_factory: Callable,
        train_fn:           Callable,
        eval_texts:         List[str],
        ranks:              List[int] = None,
    ) -> pd.DataFrame:
        """
        Compare LoRA quality vs rank (r=4,8,16,32).
        base_model_factory: callable that returns a fresh model
        train_fn: callable(model, rank) → trained model
        """
        from src.training.lora import LoRAConfig, inject_lora

        if ranks is None:
            ranks = [4, 8, 16, 32]

        results = []
        for r in ranks:
            model = base_model_factory()
            cfg   = LoRAConfig(rank=r, alpha=r * 2)
            model = inject_lora(model, cfg)

            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())

            model     = train_fn(model, r)
            ev        = Evaluator(model, self.tokenizer, self.device)
            ppl       = ev.perplexity(eval_texts)["mean_ppl"]

            results.append({
                "rank":           r,
                "alpha":          r * 2,
                "trainable_params": trainable,
                "trainable_%":    round(100 * trainable / total, 3),
                "perplexity":     ppl,
            })

        return pd.DataFrame(results)

    # ══════════════════════════════════════════════════════════════
    # 6. Prompt Technique Comparison
    # ══════════════════════════════════════════════════════════════
    def compare_prompt_techniques(
        self,
        question:   str,
        generate_fn: Callable,
        examples:   List[Dict] = None,
    ) -> pd.DataFrame:
        """
        Run the same question through 4 prompting techniques
        and return outputs for comparison.
        """
        if examples is None:
            examples = [
                {"q": "What is tokenization?",
                 "a": "Splitting text into subword units called tokens."},
                {"q": "What is attention?",
                 "a": "Mechanism that weighs token relationships using Q, K, V."},
            ]

        shots   = "\n".join(f"Q: {e['q']}\nA: {e['a']}" for e in examples)
        results = []

        techniques = {
            "zero_shot": (
                f"Question: {question}\nAnswer:"
            ),
            "few_shot": (
                f"{shots}\nQ: {question}\nA:"
            ),
            "chain_of_thought": (
                f"Question: {question}\n"
                f"Let's think step by step:\nStep 1:"
            ),
            "system_prompt": (
                f"[SYSTEM]: You are an expert IISc professor "
                f"specializing in Large Language Models.\n"
                f"[USER]: {question}\n[ASSISTANT]:"
            ),
        }

        for name, prompt in techniques.items():
            try:
                output = generate_fn(prompt, max_new_tokens=100)
                results.append({
                    "technique": name,
                    "prompt_len": len(prompt.split()),
                    "output":    output[:200],
                })
            except Exception as e:
                results.append({
                    "technique": name,
                    "prompt_len": 0,
                    "output": f"Error: {e}",
                })

        return pd.DataFrame(results)

    # ══════════════════════════════════════════════════════════════
    # 7. Full Evaluation Report
    # ══════════════════════════════════════════════════════════════
    def full_report(
        self,
        test_texts:    List[str],
        test_prompt:   str       = "The transformer uses attention",
        qa_pairs:      List      = None,
        rag            = None,
    ) -> Dict:
        """
        Run all evaluations and return a summary dict.
        Prints a formatted report to stdout.
        """
        report = {}

        print("=" * 60)
        print("  MINI ASSISTANT — FULL EVALUATION REPORT")
        print("=" * 60)

        # Perplexity
        ppl = self.perplexity(test_texts[:20])
        report["perplexity"] = ppl
        print(f"\n📊 Language Model Quality")
        print(f"   Mean PPL : {ppl['mean_ppl']:.3f}")
        print(f"   Std  PPL : {ppl['std_ppl']:.3f}")
        print(f"   Samples  : {ppl['n_samples']}")

        # Speed
        speed = self.speed_benchmark(test_prompt, n_tokens=50, n_timed=3)
        report["speed"] = speed.to_dict()
        print(f"\n⚡ Inference Speed")
        print(speed.to_string(index=False))

        # RAG
        if rag is not None and qa_pairs is not None:
            faith = self.rag_faithfulness(qa_pairs, rag)
            report["rag_faithfulness"] = faith
            print(f"\n📚 RAG Faithfulness")
            print(f"   Faithful : {faith['faithfulness']*100:.1f}%")
            print(f"   Gap Rate : {faith['gap_rate']*100:.1f}%")

        # Model stats
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        report["model_stats"] = {
            "total_params": total,
            "trainable_params": trainable,
            "fp16_mb": total * 2 / 1e6,
        }
        print(f"\n🔧 Model Stats")
        print(f"   Total params    : {total/1e6:.2f}M")
        print(f"   Trainable       : {trainable/1e6:.3f}M "
              f"({100*trainable/max(total,1):.2f}%)")
        print(f"   FP16 memory est : {total*2/1e6:.1f} MB")

        print("\n" + "=" * 60)
        return report
