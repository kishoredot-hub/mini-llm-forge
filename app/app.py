"""
app/app.py — LLMForge Gradio Application

Glass-box UI inspired by buildanllm.com:
  - Real-time training visualization (loss, PPL, grad norms)
  - Attention heatmaps during inference
  - Architecture explorer (swap components live)
  - RAG Q&A with source attribution
  - ReAct agent with tool calling
  - Evaluation dashboard

Run: python app/app.py
"""
import os, sys, json, math, threading, time
from typing import Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Global state ──────────────────────────────────────────────────
class AppState:
    model        = None
    tokenizer    = None
    rag          = None
    agent        = None
    trainer      = None
    metrics      = None
    train_thread = None
    is_training  = False
    device       = "cuda" if torch.cuda.is_available() else "cpu"

state = AppState()


# ── Lazy imports (avoid loading all deps on startup) ─────────────
def get_tokenizer(name: str = "gpt2"):
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    return tok


def get_model(preset: str = "gpt2", **overrides):
    from src.model import TransformerLM, ModelConfig
    cfg   = ModelConfig.from_preset(preset, **overrides)
    return TransformerLM(cfg).to(state.device)


# ══════════════════════════════════════════════════════════════════
# TAB 1: Architecture Explorer
# ══════════════════════════════════════════════════════════════════
def build_architecture_tab():
    with gr.Tab("🏗️ Architecture"):
        gr.Markdown("## Build Your Model\nSwap any component and see parameter counts instantly.")

        with gr.Row():
            with gr.Column(scale=1):
                preset_dd  = gr.Dropdown(
                    ["gpt2", "llama", "olmo", "tiny", "custom"],
                    value="gpt2", label="🚀 Architecture Preset"
                )
                pos_enc_dd = gr.Dropdown(
                    ["learned", "rope", "alibi", "none"],
                    value="learned", label="Positional Encoding"
                )
                norm_dd    = gr.Dropdown(
                    ["layernorm", "rmsnorm"],
                    value="layernorm", label="Normalization"
                )
                act_dd     = gr.Dropdown(
                    ["gelu", "swiglu", "relu", "geglu"],
                    value="gelu", label="Activation"
                )
                attn_dd    = gr.Dropdown(
                    ["standard", "grouped_query"],
                    value="standard", label="Attention Type"
                )

            with gr.Column(scale=1):
                d_model_sl = gr.Slider(64, 1024, value=256, step=64, label="d_model (embedding dim)")
                n_heads_sl = gr.Slider(1, 16, value=8, step=1, label="n_heads")
                n_layers_sl= gr.Slider(1, 24, value=6, step=1, label="n_layers")
                d_ff_sl    = gr.Slider(128, 4096, value=1024, step=128, label="d_ff (FFN dim)")
                vocab_sl   = gr.Slider(1000, 50257, value=8000, step=1000, label="vocab_size")

        build_btn  = gr.Button("⚡ Build Model", variant="primary")
        arch_out   = gr.Textbox(label="Model Summary", lines=15)

        def apply_preset(preset):
            from src.model import ModelConfig
            if preset == "custom":
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            if preset not in ModelConfig.PRESETS:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            p = ModelConfig.PRESETS[preset]
            return (
                gr.update(value=p.get("pos_encoding", "learned")),
                gr.update(value=p.get("norm_type", "layernorm")),
                gr.update(value=p.get("activation", "gelu")),
                gr.update(value=p.get("attention_type", "standard")),
                gr.update(value=p.get("d_model", 256)),
            )

        preset_dd.change(apply_preset, inputs=preset_dd,
                         outputs=[pos_enc_dd, norm_dd, act_dd, attn_dd, d_model_sl])

        def build_model(pos_enc, norm, act, attn, d_model, n_heads, n_layers, d_ff, vocab):
            try:
                from src.model import TransformerLM, ModelConfig
                cfg   = ModelConfig(
                    vocab_size=int(vocab), d_model=int(d_model),
                    n_heads=int(n_heads), n_layers=int(n_layers),
                    d_ff=int(d_ff), pos_encoding=pos_enc,
                    norm_type=norm, activation=act, attention_type=attn
                )
                model = TransformerLM(cfg)
                state.model = model

                total  = sum(p.numel() for p in model.parameters())
                fp32   = total * 4 / 1e6
                fp16   = total * 2 / 1e6
                lines  = [
                    f"✅ Model built successfully!",
                    f"",
                    f"Architecture Configuration:",
                    f"  Positional Encoding : {pos_enc}",
                    f"  Normalization       : {norm}",
                    f"  Activation (MLP)    : {act}",
                    f"  Attention Type      : {attn}",
                    f"",
                    f"Dimensions:",
                    f"  d_model    : {d_model}",
                    f"  n_heads    : {n_heads}",
                    f"  n_layers   : {n_layers}",
                    f"  d_ff       : {d_ff}",
                    f"  vocab_size : {vocab}",
                    f"",
                    f"Parameter Count: {total:,} ({total/1e6:.2f}M)",
                    f"FP32 Memory    : {fp32:.1f} MB",
                    f"FP16 Memory    : {fp16:.1f} MB",
                    f"",
                    f"Per-layer breakdown:",
                    f"  Token emb  : {int(vocab)*int(d_model):,}",
                    f"  Pos emb    : {512*int(d_model):,} (if learned)",
                    f"  Attn QKV   : {3*int(d_model)**2:,}",
                    f"  Attn out   : {int(d_model)**2:,}",
                    f"  MLP up     : {int(d_model)*int(d_ff):,}",
                    f"  MLP down   : {int(d_ff)*int(d_model):,}",
                    f"  LayerNorms : {4*int(d_model):,}",
                    f"  Per layer  : {3*int(d_model)**2+int(d_model)**2+2*int(d_model)*int(d_ff)+4*int(d_model):,}",
                ]
                return "\n".join(lines)
            except Exception as e:
                return f"❌ Error: {e}"

        build_btn.click(
            build_model,
            inputs=[pos_enc_dd, norm_dd, act_dd, attn_dd,
                    d_model_sl, n_heads_sl, n_layers_sl, d_ff_sl, vocab_sl],
            outputs=arch_out
        )


# ══════════════════════════════════════════════════════════════════
# TAB 2: Training
# ══════════════════════════════════════════════════════════════════
def build_training_tab():
    with gr.Tab("🏋️ Train"):
        gr.Markdown("## Pre-train or Fine-tune\nReal-time loss, perplexity, and gradient norms.")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dd  = gr.Dropdown(
                    ["tinystories", "wikitext", "alpaca", "dolly",
                     "custom_text", "custom_csv", "pdf"],
                    value="tinystories", label="Dataset"
                )
                data_path   = gr.Textbox(label="Custom Path (for custom_text/csv/pdf)",
                                          placeholder="data/my_text.txt")
                optimizer_dd= gr.Dropdown(
                    ["adamw", "lion", "adafactor", "sgd"],
                    value="adamw", label="Optimizer"
                )
                schedule_dd = gr.Dropdown(
                    ["cosine", "linear", "wsd", "constant"],
                    value="cosine", label="LR Schedule"
                )
                precision_dd= gr.Dropdown(
                    ["fp32", "fp16", "bf16"],
                    value="fp32", label="Precision"
                )

            with gr.Column(scale=1):
                lr_sl       = gr.Slider(1e-5, 1e-2, value=3e-4, step=1e-5, label="Learning Rate")
                bs_sl       = gr.Slider(4, 256, value=32, step=4, label="Batch Size")
                steps_sl    = gr.Slider(100, 10000, value=1000, step=100, label="Max Steps")
                warmup_sl   = gr.Slider(0, 500, value=100, step=10, label="Warmup Steps")
                accum_sl    = gr.Slider(1, 16, value=1, step=1, label="Gradient Accumulation")
                max_samples = gr.Slider(1000, 100000, value=10000, step=1000, label="Max Samples")

        with gr.Row():
            start_btn   = gr.Button("▶ Start Training", variant="primary")
            stop_btn    = gr.Button("⏹ Stop", variant="stop")

        train_log   = gr.Textbox(label="Training Log", lines=10, max_lines=20)

        with gr.Row():
            loss_plot   = gr.Plot(label="Loss Curve")
            ppl_plot    = gr.Plot(label="Perplexity")

        grad_plot   = gr.Plot(label="Gradient Norms")

        def make_loss_plot(metrics):
            if not metrics or not metrics.steps:
                return None
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(metrics.steps, metrics.train_loss,
                    label="Train", color="#4C72B0", lw=2)
            if metrics.val_loss:
                val_steps = metrics.steps[::max(1, len(metrics.steps)//len(metrics.val_loss))]
                ax.plot(val_steps[:len(metrics.val_loss)], metrics.val_loss,
                        label="Val", color="#DD8452", lw=2, ls="--")
            ax.set_xlabel("Step"); ax.set_ylabel("Loss")
            ax.set_title("Training Loss"); ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout(); return fig

        def make_ppl_plot(metrics):
            if not metrics or not metrics.steps:
                return None
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(metrics.steps, metrics.perplexity,
                    color="#55A868", lw=2)
            ax.set_xlabel("Step"); ax.set_ylabel("Perplexity")
            ax.set_title("Perplexity"); ax.grid(alpha=0.3)
            plt.tight_layout(); return fig

        def make_grad_plot(metrics):
            if not metrics or not metrics.steps:
                return None
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(metrics.steps, metrics.grad_norms,
                    color="#C44E52", lw=1.5, alpha=0.8)
            ax.set_xlabel("Step"); ax.set_ylabel("Grad Norm")
            ax.set_title("Gradient Norms"); ax.grid(alpha=0.3)
            plt.tight_layout(); return fig

        def start_training(dataset, data_path, optimizer, schedule, precision,
                            lr, bs, steps, warmup, accum, max_samp):
            if state.model is None:
                return "⚠️ Build a model first in the Architecture tab.", None, None, None

            from src.training.trainer import Trainer, TrainConfig
            from src.data.loaders import make_dataloaders

            cfg = TrainConfig(
                optimizer=optimizer, lr=lr, lr_schedule=schedule,
                precision=precision, batch_size=int(bs),
                max_steps=int(steps), warmup_steps=int(warmup),
                gradient_accumulation=int(accum),
            )

            log_lines = [f"Starting training: {dataset} | {optimizer} | {schedule} | {precision}"]

            def on_log(m):
                log_lines.append(
                    f"Step {m['step']:>5} | loss={m['loss']:.4f} | "
                    f"ppl={m['ppl']:.2f} | {m['tok/s']:.0f} tok/s"
                )

            try:
                if state.tokenizer is None:
                    state.tokenizer = get_tokenizer()

                kwargs = {}
                if data_path:
                    kwargs["path"] = data_path

                train_dl, val_dl = make_dataloaders(
                    dataset, state.tokenizer,
                    batch_size=int(bs), block_size=512,
                    max_samples=int(max_samp), **kwargs
                )
                trainer = Trainer(state.model, train_dl, val_dl, cfg,
                                  device=state.device, on_log=on_log)
                state.trainer = trainer
                state.is_training = True

                def run():
                    state.metrics = trainer.train()
                    state.is_training = False

                t = threading.Thread(target=run, daemon=True)
                t.start()
                state.train_thread = t

                return "\n".join(log_lines[-20:]), None, None, None

            except Exception as e:
                return f"❌ Error: {e}", None, None, None

        start_btn.click(
            start_training,
            inputs=[dataset_dd, data_path, optimizer_dd, schedule_dd,
                    precision_dd, lr_sl, bs_sl, steps_sl, warmup_sl,
                    accum_sl, max_samples],
            outputs=[train_log, loss_plot, ppl_plot, grad_plot]
        )

        def refresh_plots():
            if state.metrics:
                m = state.metrics
                log = "\n".join([
                    f"Step {s} | loss={l:.4f} | ppl={p:.2f}"
                    for s, l, p in zip(m.steps[-10:], m.train_loss[-10:], m.perplexity[-10:])
                ])
                return log, make_loss_plot(m), make_ppl_plot(m), make_grad_plot(m)
            return "No metrics yet.", None, None, None

        gr.Timer(value=5).tick(refresh_plots,
                               outputs=[train_log, loss_plot, ppl_plot, grad_plot])


# ══════════════════════════════════════════════════════════════════
# TAB 3: RAG Q&A
# ══════════════════════════════════════════════════════════════════
def build_rag_tab():
    with gr.Tab("📚 RAG Q&A"):
        gr.Markdown("## Document Q&A with Source Attribution\nUpload PDFs or enter text to build a knowledge base.")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_upload  = gr.File(label="Upload PDF(s)", file_types=[".pdf"],
                                      file_count="multiple")
                text_input  = gr.Textbox(label="Or paste text directly",
                                          lines=5, placeholder="Paste any text here...")
                embed_model = gr.Dropdown(
                    ["all-MiniLM-L6-v2", "all-mpnet-base-v2",
                     "paraphrase-MiniLM-L6-v2"],
                    value="all-MiniLM-L6-v2", label="Embedding Model"
                )
                chunk_sl    = gr.Slider(64, 512, value=256, step=32, label="Chunk Size")
                top_k_sl    = gr.Slider(1, 10, value=3, step=1, label="Top-K Retrieval")
                ingest_btn  = gr.Button("📥 Build Knowledge Base", variant="primary")
                ingest_status = gr.Textbox(label="Status", lines=2)

            with gr.Column(scale=2):
                rag_question = gr.Textbox(label="Ask a question", lines=2,
                                           placeholder="What is the attention mechanism?")
                ask_btn      = gr.Button("🔍 Ask", variant="primary")
                rag_answer   = gr.Textbox(label="Answer", lines=6)
                sources_out  = gr.Textbox(label="Sources Used", lines=8)
                gap_flag     = gr.Textbox(label="Gap Detection", lines=1)

        def ingest(files, text, embed_model_name, chunk_size):
            try:
                from src.rag.pipeline import RAGPipeline
                rag = RAGPipeline(embed_model=embed_model_name,
                                  chunk_size=int(chunk_size))
                n_chunks = 0
                if files:
                    for f in files:
                        n = rag.ingest_pdf(f.name)
                        n_chunks += n
                if text and text.strip():
                    n = rag.ingest_text(text)
                    n_chunks += n
                state.rag = rag
                return f"✅ Knowledge base built: {n_chunks} chunks indexed"
            except Exception as e:
                return f"❌ Error: {e}"

        ingest_btn.click(ingest, inputs=[pdf_upload, text_input, embed_model, chunk_sl],
                         outputs=ingest_status)

        def ask_rag(question, top_k):
            if state.rag is None:
                return "⚠️ Build a knowledge base first.", "", ""
            if state.model is None and state.tokenizer is None:
                return "⚠️ Load a model first.", "", ""
            try:
                chunks = state.rag.retrieve(question, k=int(top_k))
                if not chunks:
                    return "No relevant content found.", "", "⚠️ GAP: Topic not in knowledge base"

                # Format context
                context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
                prompt  = (f"Answer using ONLY the context below.\n"
                           f"If not in context, say 'I don't know.'\n\n"
                           f"Context:\n{context}\n\n"
                           f"Question: {question}\nAnswer:")

                # Generate (use HF model if our model not trained enough)
                try:
                    from transformers import pipeline
                    if not hasattr(state, "_gen_pipe") or state._gen_pipe is None:
                        state._gen_pipe = pipeline("text-generation",
                                                    model="gpt2",
                                                    device=0 if state.device=="cuda" else -1)
                    result = state._gen_pipe(prompt, max_new_tokens=150,
                                             do_sample=False)[0]["generated_text"]
                    answer = result[len(prompt):].strip()
                except Exception:
                    answer = "[Generation requires a loaded model]"

                gap = "⚠️ GAP DETECTED" if "don't know" in answer.lower() else "✅ Grounded answer"

                src_text = "\n\n".join(
                    f"Source {i+1}:\n{c[:300]}..." for i, c in enumerate(chunks)
                )
                return answer, src_text, gap
            except Exception as e:
                return f"Error: {e}", "", ""

        ask_btn.click(ask_rag, inputs=[rag_question, top_k_sl],
                      outputs=[rag_answer, sources_out, gap_flag])


# ══════════════════════════════════════════════════════════════════
# TAB 4: Agent
# ══════════════════════════════════════════════════════════════════
def build_agent_tab():
    with gr.Tab("🤖 Agent"):
        gr.Markdown("## ReAct Tool-Calling Agent\nReason → Act → Observe → Answer")

        with gr.Row():
            agent_input  = gr.Textbox(
                label="Query",
                placeholder="How many parameters does GPT-2 Medium have and how much FP16 memory?",
                lines=2
            )
            max_steps_sl = gr.Slider(1, 10, value=5, step=1, label="Max Steps")

        run_btn       = gr.Button("🚀 Run Agent", variant="primary")

        with gr.Row():
            steps_out = gr.Textbox(label="Agent Reasoning Steps", lines=15)
            final_out = gr.Textbox(label="Final Answer", lines=5)

        def run_agent(query, max_steps):
            try:
                from src.agent.react import MiniAgent

                def gen_fn(prompt, max_new_tokens=150):
                    try:
                        from transformers import pipeline
                        if not hasattr(state, "_gen_pipe") or state._gen_pipe is None:
                            state._gen_pipe = pipeline("text-generation", model="gpt2",
                                                        device=0 if state.device=="cuda" else -1)
                        out = state._gen_pipe(prompt, max_new_tokens=max_new_tokens,
                                              do_sample=False)[0]["generated_text"]
                        return out[len(prompt):]
                    except Exception as e:
                        return f"[Generation error: {e}]"

                agent  = MiniAgent(generate_fn=gen_fn, rag=state.rag)
                answer, steps = agent.run(query, max_steps=int(max_steps))
                return steps, answer
            except Exception as e:
                return f"Error: {e}", ""

        run_btn.click(run_agent, inputs=[agent_input, max_steps_sl],
                      outputs=[steps_out, final_out])


# ══════════════════════════════════════════════════════════════════
# TAB 5: Inference
# ══════════════════════════════════════════════════════════════════
def build_inference_tab():
    with gr.Tab("⚡ Inference"):
        gr.Markdown("## Generate Text + Visualize Attention\nInspect what the model attends to (Glass-box mode).")

        with gr.Row():
            with gr.Column(scale=1):
                inf_model   = gr.Dropdown(
                    ["gpt2", "gpt2-medium", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                     "microsoft/phi-2", "custom (local checkpoint)"],
                    value="gpt2", label="Model"
                )
                precision_dd= gr.Dropdown(
                    ["fp32", "fp16", "4-bit"], value="fp32", label="Precision"
                )
                load_btn    = gr.Button("📥 Load Model")
                load_status = gr.Textbox(label="Status", lines=1)

            with gr.Column(scale=2):
                prompt_in   = gr.Textbox(label="Prompt", lines=3,
                                          value="Once upon a time")
                temp_sl     = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                top_k_sl    = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                top_p_sl    = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                max_tok_sl  = gr.Slider(10, 500, value=100, step=10, label="Max New Tokens")
                gen_btn     = gr.Button("🎲 Generate", variant="primary")

        gen_out     = gr.Textbox(label="Generated Text", lines=8)
        attn_plot   = gr.Plot(label="Attention Heatmap (Last Layer)")

        def load_model(model_name, precision):
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                kwargs = {"trust_remote_code": True}
                if precision == "fp16":
                    kwargs["torch_dtype"] = torch.float16
                elif precision == "bf16":
                    kwargs["torch_dtype"] = torch.bfloat16
                elif precision == "4-bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                    )
                if model_name == "custom (local checkpoint)":
                    return "⚠️ Load via 'python -c \"from src.model import TransformerLM; ...\"'"

                state.tokenizer = AutoTokenizer.from_pretrained(model_name)
                state.tokenizer.pad_token = state.tokenizer.eos_token
                hf_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                if precision != "4-bit":
                    hf_model = hf_model.to(state.device)
                state._hf_model = hf_model
                total = sum(p.numel() for p in hf_model.parameters())
                return f"✅ Loaded {model_name} | {total/1e6:.1f}M params | {precision}"
            except Exception as e:
                return f"❌ Error: {e}"

        load_btn.click(load_model, inputs=[inf_model, precision_dd], outputs=load_status)

        def generate(prompt, temp, top_k, top_p, max_new):
            model  = getattr(state, "_hf_model", None)
            tok    = state.tokenizer
            if model is None or tok is None:
                return "⚠️ Load a model first.", None

            ids    = tok.encode(prompt, return_tensors="pt").to(state.device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=int(max_new),
                    temperature=temp, top_k=int(top_k), top_p=top_p,
                    do_sample=True,
                    output_attentions=False,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0], skip_special_tokens=True)

            # Attention heatmap (if model supports it)
            try:
                with torch.no_grad():
                    attn_out = model(ids, output_attentions=True)
                attn = attn_out.attentions[-1][0].mean(0).cpu().numpy()
                tokens = tok.convert_ids_to_tokens(ids[0])
                T = min(len(tokens), 20)
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(attn[:T, :T], cmap="Blues", aspect="auto")
                ax.set_xticks(range(T)); ax.set_xticklabels(tokens[:T], rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(T)); ax.set_yticklabels(tokens[:T], fontsize=8)
                ax.set_title("Attention Heatmap — Last Layer (avg heads)")
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                return text, fig
            except Exception:
                return text, None

        gen_btn.click(generate,
                      inputs=[prompt_in, temp_sl, top_k_sl, top_p_sl, max_tok_sl],
                      outputs=[gen_out, attn_plot])


# ══════════════════════════════════════════════════════════════════
# TAB 6: Evaluation
# ══════════════════════════════════════════════════════════════════
def build_eval_tab():
    with gr.Tab("📊 Evaluation"):
        gr.Markdown("## Evaluation Dashboard")

        eval_prompt = gr.Textbox(
            label="Test Prompt for Speed Benchmark",
            value="The transformer architecture uses self-attention"
        )
        eval_btn    = gr.Button("▶ Run Full Evaluation", variant="primary")
        eval_out    = gr.Dataframe(
            headers=["Metric", "Value"],
            label="Results"
        )

        def run_eval(prompt):
            model  = getattr(state, "_hf_model", None)
            tok    = state.tokenizer
            if model is None:
                return [["Status", "⚠️ Load a model first in Inference tab"]]

            rows = []
            try:
                # Perplexity
                import math
                ids  = tok.encode(prompt, return_tensors="pt").to(state.device)
                with torch.no_grad():
                    out  = model(ids, labels=ids)
                    ppl  = math.exp(out.loss.item())
                rows.append(["Perplexity", f"{ppl:.2f}"])

                # Speed
                times = []
                for _ in range(3):
                    if state.device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.no_grad():
                        model.generate(ids, max_new_tokens=50,
                                       do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    if state.device == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.time() - t0)
                tps = 50 / (sum(times)/len(times))
                rows.append(["Throughput (tok/s)", f"{tps:.1f}"])
                rows.append(["Latency (ms/tok)", f"{1000/tps:.1f}"])

                # VRAM
                if state.device == "cuda":
                    vram = torch.cuda.max_memory_allocated() / 1e6
                    rows.append(["Peak VRAM (MB)", f"{vram:.0f}"])

                # Params
                total = sum(p.numel() for p in model.parameters())
                rows.append(["Parameters", f"{total/1e6:.2f}M"])
                rows.append(["FP16 Memory Est.", f"{total*2/1e6:.0f} MB"])

            except Exception as e:
                rows.append(["Error", str(e)])
            return rows

        eval_btn.click(run_eval, inputs=eval_prompt, outputs=eval_out)


# ══════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════
def build_app():
    with gr.Blocks(
        title="LLMForge — LLM from Scratch",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as app:

        gr.Markdown("""
# 🤖 LLMForge — End-to-End AI Assistant System
**IISc LLM Course Project | Covers: Architecture · Tokenization · Training · LoRA · RAG · Agents · Evaluation**

Inspired by [buildanllm.com](https://buildanllm.com) and [Unsloth Studio](https://unsloth.ai)

Authors: ***Bommaji Kishore Kumar***: [GitHub](https://github.com/bommajikishore)
        """)

        # System info banner
        device_info = f"Device: **{state.device.upper()}**"
        if state.device == "cuda":
            device_info += f" | GPU: **{torch.cuda.get_device_name(0)}**"
            device_info += f" | VRAM: **{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB**"
        gr.Markdown(device_info)

        build_architecture_tab()
        build_training_tab()
        build_rag_tab()
        build_agent_tab()
        build_inference_tab()
        build_eval_tab()

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,        # set True for public URL
        show_error=True,
    )
