#!/usr/bin/env python3
"""Block 4: Full training run — 768D/12L (~1.06B total, ~550M active/token).

This is the real training run. Only launch after Block 3 validation proves
the architecture trains stably.

Run on Titan PC:
    nohup python3 -u -m python.synapse_arch.train_full 2>&1 | tee /tmp/synapse_full.log &

Resume from checkpoint:
    python3 -u -m python.synapse_arch.train_full --resume ~/titan-synapse/checkpoints/full/step_10000
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from python.synapse_arch.config import SynapseModelConfig
from python.synapse_arch.trainer import SynapseTrainer, TrainerConfig
from python.synapse_arch.data import DataConfig


def main():
    print("=" * 60)
    print("SYNAPSE ARCHITECTURE — FULL TRAINING RUN")
    print("768D / 12 layers / 8 experts (top-2) / ~1.06B params")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name()
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_vram:.1f} GB)")

    # Resume support
    resume_from = None
    for i, arg in enumerate(sys.argv):
        if arg == "--resume" and i + 1 < len(sys.argv):
            resume_from = sys.argv[i + 1]

    # Full model config — 768D/12L
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

    # Training config — tuned for RTX 5090 with 1B model
    # At BF16 + gradient checkpointing: ~14GB VRAM estimated
    trainer_config = TrainerConfig(
        # Optimization
        lr=3e-4,
        min_lr=1e-5,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        warmup_steps=2000,

        # Batch — tight fit on RTX 5090 (32GB) with Ollama using ~4GB
        # batch=1 + grad_accum=32 = same effective batch of 32
        batch_size=1,
        grad_accumulation=32,  # effective batch = 32
        seq_len=512,  # Keep conservative — 1B model + grad ckpt

        # Training
        max_steps=100_000,  # ~1.6B tokens at batch 32 * 512 seq
        eval_interval=1000,
        save_interval=2000,
        log_interval=10,

        # Mixed precision
        use_bf16=True,
        use_gradient_checkpointing=True,

        # MoE
        load_balance_coeff=0.01,

        # Paths
        output_dir=os.path.expanduser("~/titan-synapse/checkpoints/full"),
        resume_from=resume_from,

        # Logging
        use_wandb=False,
    )

    # Data — larger mix for full run
    data_config = DataConfig(
        seq_len=512,
        batch_size=1,
        tokenizer_name="Qwen/Qwen2.5-3B",
        datasets={
            "allenai/c4:en": 0.4,            # General web text
            "HuggingFaceFW/fineweb-edu": 0.3,  # Educational
            "teknium/OpenHermes-2.5": 0.15,    # Instruction-following
            "glaiveai/glaive-function-calling-v2": 0.15,  # Tool use
        },
        max_tokens=5_000_000_000,  # 5B tokens
    )

    print(f"\nModel: {model_config.total_params() / 1e6:.0f}M total, "
          f"{model_config.active_params_per_token() / 1e6:.0f}M active/token")
    print(f"Training: {trainer_config.max_steps} steps, "
          f"effective batch {trainer_config.batch_size * trainer_config.grad_accumulation}")
    estimated_tokens = (trainer_config.max_steps * trainer_config.batch_size
                        * trainer_config.grad_accumulation * trainer_config.seq_len)
    print(f"Estimated tokens: {estimated_tokens / 1e9:.1f}B")
    print(f"Estimated VRAM: {model_config.estimated_vram_mb(bytes_per_param=2) / 1024:.1f} GB (model only, BF16)")
    print()

    if resume_from:
        print(f"Resuming from: {resume_from}")

    trainer = SynapseTrainer(model_config, trainer_config, data_config)

    print("Loading streaming datasets from HuggingFace...")
    trainer.train()

    print("\nFull training complete!")
    print(f"Checkpoints saved to: {trainer_config.output_dir}")

    # Auto-run evaluation on final checkpoint
    print("\n" + "=" * 60)
    print("Running evaluation on final checkpoint...")
    print("=" * 60)
    os.system(
        f"python3 -u python/synapse_arch/eval_benchmarks.py "
        f"--checkpoint {trainer_config.output_dir}/final "
        f"--config full"
    )


if __name__ == "__main__":
    main()
