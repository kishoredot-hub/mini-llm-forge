"""
training/trainer.py — Training engine with swappable optimizers,
LR schedulers, mixed precision, and real-time observability.

Inspired by:
  - buildanllm.com: glass-box training with live metrics
  - Unsloth Studio: real-time loss/grad/GPU observability
"""
from __future__ import annotations
import os, math, time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


# ── Training Config ───────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Optimizer
    optimizer:    str   = "adamw"   # adamw | lion | sophia | adafactor
    lr:           float = 3e-4
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    eps:          float = 1e-8
    grad_clip:    float = 1.0

    # Schedule
    lr_schedule:  str   = "cosine"  # cosine | linear | wsd | constant
    warmup_steps: int   = 200
    max_steps:    int   = 5000

    # Batching
    batch_size:           int = 32
    gradient_accumulation: int = 1

    # Precision
    precision:    str   = "fp32"    # fp32 | fp16 | bf16

    # Checkpointing
    save_dir:     str   = "checkpoints"
    save_every:   int   = 500
    eval_every:   int   = 200
    log_every:    int   = 50

    # Logging
    wandb:        bool  = False
    project_name: str   = "LLMForge"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        t = raw.get("training", raw)
        return cls(**{k: v for k, v in t.items() if hasattr(cls, k)})


# ── Optimizer factory ─────────────────────────────────────────────
def build_optimizer(name: str, params, cfg: TrainConfig) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay, eps=cfg.eps,
        )
    elif name == "adam":
        return torch.optim.Adam(
            params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
        )
    elif name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    elif name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=cfg.lr / 3,
                        weight_decay=cfg.weight_decay,
                        betas=(cfg.beta1, cfg.beta2))
        except ImportError:
            print("Lion not found (pip install lion-pytorch). Falling back to AdamW.")
            return build_optimizer("adamw", params, cfg)
    elif name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=cfg.lr, relative_step=False,
                             scale_parameter=False)
        except ImportError:
            print("Adafactor not found. Falling back to AdamW.")
            return build_optimizer("adamw", params, cfg)
    else:
        raise ValueError(f"Unknown optimizer: {name}. "
                         f"Choose: adamw | adam | sgd | lion | adafactor")


# ── LR Schedule factory ───────────────────────────────────────────
def build_scheduler(name: str, optimizer, cfg: TrainConfig):
    warmup = cfg.warmup_steps
    total  = cfg.max_steps

    if name == "cosine":
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif name == "linear":
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            return max(0.0, (total - step) / max(total - warmup, 1))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif name == "wsd":
        # Warmup → Stable → Decay (MiniCPM / WSD schedule)
        stable_end = int(total * 0.8)
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            elif step < stable_end:
                return 1.0   # stable phase
            else:
                progress = (step - stable_end) / max(total - stable_end, 1)
                return 1.0 - progress   # linear decay
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0
        )
    else:
        raise ValueError(f"Unknown schedule: {name}. "
                         f"Choose: cosine | linear | wsd | constant")


# ── Metrics tracker ───────────────────────────────────────────────
class MetricsTracker:
    """Collects and stores training metrics for visualization."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps:       List[int]   = []
        self.train_loss:  List[float] = []
        self.val_loss:    List[float] = []
        self.perplexity:  List[float] = []
        self.grad_norms:  List[float] = []
        self.lr:          List[float] = []
        self.tokens_seen: List[int]   = []
        self.throughput:  List[float] = []  # tokens/sec

    def log(self, step: int, loss: float, grad_norm: float,
            lr: float, tokens: int, tps: float):
        self.steps.append(step)
        self.train_loss.append(loss)
        self.perplexity.append(math.exp(min(loss, 20)))
        self.grad_norms.append(grad_norm)
        self.lr.append(lr)
        self.tokens_seen.append(tokens)
        self.throughput.append(tps)

    def log_val(self, val_loss: float):
        self.val_loss.append(val_loss)

    def latest(self) -> Dict:
        if not self.steps:
            return {}
        return {
            "step":       self.steps[-1],
            "loss":       self.train_loss[-1],
            "ppl":        self.perplexity[-1],
            "grad_norm":  self.grad_norms[-1],
            "lr":         self.lr[-1],
            "tokens":     self.tokens_seen[-1],
            "tok/s":      self.throughput[-1],
        }


# ── Main Trainer ──────────────────────────────────────────────────
class Trainer:
    """
    Modular trainer supporting:
      - Mixed precision (fp16, bf16, fp32)
      - Gradient accumulation
      - Swappable optimizers and LR schedules
      - Live metrics via callback
      - Auto checkpointing
    """

    def __init__(
        self,
        model:       nn.Module,
        train_dl:    DataLoader,
        val_dl:      DataLoader,
        cfg:         TrainConfig,
        device:      str           = "cuda",
        on_log:      Optional[Callable] = None,   # callback for live UI
    ):
        self.model    = model.to(device)
        self.train_dl = train_dl
        self.val_dl   = val_dl
        self.cfg      = cfg
        self.device   = device
        self.on_log   = on_log   # called with metrics dict each log step

        # Build optimizer and scheduler
        self.optimizer = build_optimizer(cfg.optimizer,
                                         model.parameters(), cfg)
        self.scheduler = build_scheduler(cfg.lr_schedule,
                                         self.optimizer, cfg)

        # Mixed precision
        self.use_amp = cfg.precision in ("fp16", "bf16")
        self.dtype   = (torch.float16 if cfg.precision == "fp16"
                        else torch.bfloat16)
        self.scaler  = GradScaler() if cfg.precision == "fp16" else None

        # Metrics
        self.metrics  = MetricsTracker()
        self.step     = 0
        self.best_val = float("inf")

        os.makedirs(cfg.save_dir, exist_ok=True)
        print(f"Trainer ready | optimizer={cfg.optimizer} "
              f"| schedule={cfg.lr_schedule} | precision={cfg.precision}")

    def _compute_grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm().item() ** 2
        return total ** 0.5

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for x, y in self.val_dl:
            x, y = x.to(self.device), y.to(self.device)
            with autocast(dtype=self.dtype, enabled=self.use_amp):
                out  = self.model(x, targets=y)
                loss = out["loss"] if isinstance(out, dict) else out[1]
            total_loss += loss.item()
            n_batches  += 1
            if n_batches >= 50:   # cap at 50 batches for speed
                break
        self.model.train()
        return total_loss / max(n_batches, 1)

    def train(self) -> MetricsTracker:
        self.model.train()
        tokens_seen  = 0
        t0           = time.time()
        accum_loss   = 0.0
        accum_steps  = 0

        data_iter    = iter(self.train_dl)

        print(f"\n{'─'*60}")
        print(f"  Training: {self.cfg.max_steps} steps | "
              f"batch={self.cfg.batch_size} | "
              f"grad_accum={self.cfg.gradient_accumulation}")
        print(f"{'─'*60}\n")

        while self.step < self.cfg.max_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dl)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)

            # ── Forward pass ──────────────────────────────────────
            with autocast(dtype=self.dtype, enabled=self.use_amp):
                out  = self.model(x, targets=y)
                loss = out["loss"] if isinstance(out, dict) else out[1]
                loss = loss / self.cfg.gradient_accumulation

            # ── Backward pass ─────────────────────────────────────
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss  += loss.item() * self.cfg.gradient_accumulation
            accum_steps += 1
            tokens_seen += x.numel()

            # ── Optimizer step (after accumulation) ──────────────
            if accum_steps % self.cfg.gradient_accumulation == 0:
                # Unscale before clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = self._compute_grad_norm()

                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.step += 1
                avg_loss   = accum_loss / self.cfg.gradient_accumulation
                accum_loss = 0.0

                # ── Compute throughput ────────────────────────────
                elapsed = time.time() - t0
                tps     = tokens_seen / max(elapsed, 1e-6)

                # ── Logging ───────────────────────────────────────
                if self.step % self.cfg.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.metrics.log(
                        self.step, avg_loss, grad_norm, lr,
                        tokens_seen, tps
                    )
                    msg = (f"  step {self.step:>5} | "
                           f"loss {avg_loss:.4f} | "
                           f"ppl {math.exp(min(avg_loss,20)):.2f} | "
                           f"grad {grad_norm:.3f} | "
                           f"lr {lr:.2e} | "
                           f"{tps:.0f} tok/s")
                    print(msg)

                    # Call UI callback if provided (for live Gradio)
                    if self.on_log:
                        self.on_log(self.metrics.latest())

                # ── Validation ────────────────────────────────────
                if self.step % self.cfg.eval_every == 0:
                    val_loss = self.evaluate()
                    self.metrics.log_val(val_loss)
                    print(f"  ✓ Val loss: {val_loss:.4f} | "
                          f"Val PPL: {math.exp(min(val_loss,20)):.2f}")

                    if self.cfg.wandb:
                        try:
                            import wandb
                            wandb.log({"val_loss": val_loss,
                                       "val_ppl": math.exp(val_loss),
                                       "step": self.step})
                        except Exception:
                            pass

                    # Save best checkpoint
                    if val_loss < self.best_val:
                        self.best_val = val_loss
                        self._save("best")

                # ── Periodic checkpoint ───────────────────────────
                if self.step % self.cfg.save_every == 0:
                    self._save(f"step_{self.step}")

                if self.step >= self.cfg.max_steps:
                    break

        # Final save
        self._save("final")
        total_time = time.time() - t0
        print(f"\n{'─'*60}")
        print(f"  Training complete in {total_time/60:.1f} min")
        print(f"  Best val loss: {self.best_val:.4f}")
        print(f"{'─'*60}\n")
        return self.metrics

    def _save(self, tag: str):
        path = os.path.join(self.cfg.save_dir, f"ckpt_{tag}.pt")
        torch.save({
            "step":       self.step,
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "cfg":        self.cfg,
            "metrics":    self.metrics,
        }, path)
        print(f"  Saved checkpoint → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        print(f"Resumed from step {self.step}")
