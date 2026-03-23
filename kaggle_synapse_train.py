#!/usr/bin/env python3
"""Synapse Architecture Training — Kaggle Notebook Version.

Designed for Kaggle's free T4 GPU (16GB VRAM).
- 9-hour session limit → auto-saves checkpoints
- Resumes from last checkpoint on next session
- Smaller batch to fit T4's 16GB VRAM
- Saves final checkpoint to /kaggle/working/ for download

Usage: Create a new Kaggle notebook, enable GPU T4x2, paste this code.
"""

import os
import sys
import time
import json
import subprocess

# ─── Install deps ──────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "safetensors"], check=True)

# ─── Clone or update the repo ─────────────────────────────────────────────
REPO_DIR = "/kaggle/working/titan-synapse"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "https://github.com/Djtony707/titan-synapse.git", REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

sys.path.insert(0, REPO_DIR)

import torch

# ─── Check GPU ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SYNAPSE — KAGGLE T4 TRAINING")
print("=" * 60)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_vram:.1f} GB)")
    n_gpus = torch.cuda.device_count()
    print(f"GPU count: {n_gpus}")
else:
    print("ERROR: No GPU available!")
    sys.exit(1)

# ─── Import Synapse ────────────────────────────────────────────────────────
from python.synapse_arch.config import SynapseModelConfig
from python.synapse_arch.trainer import SynapseTrainer, TrainerConfig
from python.synapse_arch.data import DataConfig

# ─── Find latest checkpoint to resume from ─────────────────────────────────
CHECKPOINT_DIR = "/kaggle/working/checkpoints/kaggle"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

resume_from = None
if os.path.exists(CHECKPOINT_DIR):
    steps = []
    for d in os.listdir(CHECKPOINT_DIR):
        if d.startswith("step_"):
            try:
                steps.append(int(d.split("_")[1]))
            except ValueError:
                pass
    if steps:
        latest = max(steps)
        resume_from = os.path.join(CHECKPOINT_DIR, f"step_{latest}")
        print(f"Resuming from checkpoint: {resume_from}")

# ─── Model config — 768D/12L (same as full training) ──────────────────────
model_config = SynapseModelConfig(
    d_model=768,
    n_layers=12,
    n_experts=8,
    top_k=2,
    d_expert=3072,
    d_xlstm_hidden=1536,
    n_xlstm_heads=4,
    d_memory=64,
    n_memory_heads=8,
    d_state=16,
)

# ─── Training config — tuned for T4 16GB ──────────────────────────────────
# T4 has 16GB — much tighter than 5090's 32GB
# batch=1, grad_accum=32, seq_len=256 to fit
trainer_config = TrainerConfig(
    lr=3e-4,
    min_lr=1e-5,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    warmup_steps=2000,

    batch_size=1,
    grad_accumulation=32,
    seq_len=256,  # Shorter than 5090 to fit T4 VRAM

    max_steps=100_000,
    eval_interval=1000,
    save_interval=500,  # Save frequently — 9hr session limit!
    log_interval=10,

    use_bf16=True,
    use_gradient_checkpointing=True,
    load_balance_coeff=0.01,

    output_dir=CHECKPOINT_DIR,
    resume_from=resume_from,
    use_wandb=False,
)

# ─── Data config ───────────────────────────────────────────────────────────
data_config = DataConfig(
    seq_len=256,
    batch_size=1,
    tokenizer_name="Qwen/Qwen2.5-3B",
    datasets={
        "allenai/c4:en": 0.4,
        "HuggingFaceFW/fineweb-edu": 0.3,
        "teknium/OpenHermes-2.5": 0.15,
        "glaiveai/glaive-function-calling-v2": 0.15,
    },
    max_tokens=5_000_000_000,
)

# ─── Print summary ─────────────────────────────────────────────────────────
eff = trainer_config.batch_size * trainer_config.grad_accumulation
tps = eff * trainer_config.seq_len
print(f"\nModel: {model_config.total_params() / 1e6:.0f}M params")
print(f"Batch: {trainer_config.batch_size} x {trainer_config.grad_accumulation} = {eff} effective")
print(f"Tokens/step: {tps:,}")
print(f"Save every {trainer_config.save_interval} steps (frequent for session limits)")
if resume_from:
    print(f"RESUMING from: {resume_from}")
print()

# ─── Auto-save timer (save 30 min before 9hr limit) ───────────────────────
SESSION_START = time.time()
MAX_SESSION_HOURS = 8.5  # Stop at 8.5hr to leave time for final save

# ─── Train! ────────────────────────────────────────────────────────────────
print("Starting training...")
trainer = SynapseTrainer(model_config, trainer_config, data_config)
trainer.train()

print("\nTraining complete or session limit reached!")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
print("Download checkpoints from /kaggle/working/checkpoints/kaggle/")
