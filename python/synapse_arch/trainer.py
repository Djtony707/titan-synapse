"""Trainer for the Synapse Architecture.

Handles gradient checkpointing, mixed precision, logging, and checkpointing.
Designed to train on a single RTX 5090 (32GB VRAM).
"""

import os
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from .config import SynapseModelConfig
from .synapse_model import SynapseModel
from .data import DataConfig, create_dataloader, DummyDataset


@dataclass
class TrainerConfig:
    """Training configuration."""
    # Optimization
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    # Batch
    batch_size: int = 4
    grad_accumulation: int = 8  # effective batch = 32
    seq_len: int = 2048

    # Training
    max_steps: int = 100_000
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 10

    # Mixed precision
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True

    # MoE
    load_balance_coeff: float = 0.01

    # Paths
    output_dir: str = "checkpoints/synapse"
    resume_from: str | None = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "synapse-training"
    wandb_run_name: str = "synapse-1b"


class SynapseTrainer:
    """Training loop for Synapse model."""

    def __init__(
        self,
        model_config: SynapseModelConfig,
        trainer_config: TrainerConfig,
        data_config: DataConfig | None = None,
    ):
        self.model_config = model_config
        self.config = trainer_config
        self.data_config = data_config or DataConfig(
            seq_len=trainer_config.seq_len,
            batch_size=trainer_config.batch_size,
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Model
        self.model = SynapseModel(model_config).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params / 1e6:.1f}M")

        # Gradient checkpointing
        if trainer_config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=trainer_config.use_bf16 and self.device.type == "cuda")
        self.autocast_dtype = torch.bfloat16 if trainer_config.use_bf16 else torch.float32

        # State
        self.step = 0
        self.tokens_processed = 0
        self.best_loss = float("inf")

        # Output dir
        os.makedirs(trainer_config.output_dir, exist_ok=True)

        # Resume
        if trainer_config.resume_from:
            self._load_checkpoint(trainer_config.resume_from)

        # WandB
        if trainer_config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=trainer_config.wandb_project,
                    name=trainer_config.wandb_run_name,
                    config={
                        "model": vars(model_config),
                        "training": vars(trainer_config),
                    },
                )
            except ImportError:
                print("wandb not installed, logging to stdout only")
                self.config.use_wandb = False

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on each SynapseLayer."""
        from torch.utils.checkpoint import checkpoint

        for layer in self.model.layers:
            original_forward = layer.forward

            def make_checkpointed(orig_fn):
                def checkpointed_forward(x, training=False):
                    # use_reentrant=True avoids tensor count/metadata verification
                    # that fails with MoE routing's conditional paths (different
                    # experts activate for different tokens → different tensor counts).
                    # autocast context is automatically preserved with reentrant mode.
                    return checkpoint(orig_fn, x, training, use_reentrant=True)
                return checkpointed_forward

            layer.forward = make_checkpointed(original_forward)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay groups."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            groups,
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
        )

    def _get_lr(self) -> float:
        """Cosine learning rate with warmup."""
        if self.step < self.config.warmup_steps:
            return self.config.lr * self.step / self.config.warmup_steps

        progress = (self.step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr + (self.config.lr - self.config.min_lr) * cosine

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train(self, dataloader: DataLoader | None = None):
        """Main training loop."""
        if dataloader is None:
            print("Creating streaming dataloader...")
            dataloader = create_dataloader(self.data_config)

        self.model.train()
        self.optimizer.zero_grad()

        running_loss = 0.0
        step_start = time.time()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Update learning rate
            lr = self._get_lr()
            self._set_lr(lr)

            # Forward pass with mixed precision
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                result = self.model(input_ids, labels=labels, training=True)
                loss = result["loss"] / self.config.grad_accumulation

            # Backward
            self.scaler.scale(loss).backward()
            running_loss += loss.item() * self.config.grad_accumulation

            # Step optimizer every grad_accumulation steps
            if (self.step + 1) % self.config.grad_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.step += 1
            self.tokens_processed += input_ids.numel()

            # Logging
            if self.step % self.config.log_interval == 0:
                avg_loss = running_loss / self.config.log_interval
                elapsed = time.time() - step_start
                tok_per_sec = (self.config.log_interval * input_ids.numel()) / elapsed

                print(
                    f"step {self.step:>6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                    f"{tok_per_sec:.0f} tok/s | {self.tokens_processed / 1e6:.1f}M tokens"
                )

                if self.config.use_wandb:
                    import wandb
                    wandb.log({
                        "loss": avg_loss,
                        "lr": lr,
                        "tokens_per_sec": tok_per_sec,
                        "tokens_processed": self.tokens_processed,
                    }, step=self.step)

                running_loss = 0.0
                step_start = time.time()

            # Save checkpoint
            if self.step % self.config.save_interval == 0:
                self._save_checkpoint()

            # Max steps
            if self.step >= self.config.max_steps:
                print(f"Reached max_steps ({self.config.max_steps})")
                break

        # Final save
        self._save_checkpoint(final=True)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint as SafeTensors + training state."""
        suffix = "final" if final else f"step_{self.step}"
        path = Path(self.config.output_dir) / suffix
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        from safetensors.torch import save_file
        state_dict = self.model.state_dict()
        # Remove non-parameter buffers (fast_weights are runtime state)
        save_dict = {k: v for k, v in state_dict.items() if "fast_weights" not in k}
        save_file(save_dict, str(path / "model.safetensors"))

        # Save training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "tokens_processed": self.tokens_processed,
            "best_loss": self.best_loss,
        }, str(path / "training_state.pt"))

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(vars(self.model_config), f, indent=2, default=str)

        print(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Load checkpoint."""
        path = Path(path)
        from safetensors.torch import load_file

        # Load model weights
        state_dict = load_file(str(path / "model.safetensors"))
        self.model.load_state_dict(state_dict, strict=False)

        # Load training state
        train_state = torch.load(str(path / "training_state.pt"), map_location=self.device)
        self.optimizer.load_state_dict(train_state["optimizer"])
        self.scaler.load_state_dict(train_state["scaler"])
        self.step = train_state["step"]
        self.tokens_processed = train_state["tokens_processed"]
        self.best_loss = train_state["best_loss"]

        print(f"Resumed from {path} at step {self.step}")


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Synapse Architecture")
    parser.add_argument("--config", choices=["small", "validation", "full"], default="validation")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints/synapse")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dummy-data", action="store_true", help="Use random data (for testing)")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # Model config
    if args.config == "small":
        model_config = SynapseModelConfig.small_test()
    elif args.config == "validation":
        model_config = SynapseModelConfig.validation()
    else:
        model_config = SynapseModelConfig()  # Full 768D/12L

    # Trainer config
    trainer_config = TrainerConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        resume_from=args.resume,
        use_wandb=args.wandb,
        max_steps=args.max_steps or (1000 if args.config == "small" else 100_000),
    )

    # Data
    data_config = DataConfig(
        batch_size=args.batch_size,
        seq_len=trainer_config.seq_len,
    )

    trainer = SynapseTrainer(model_config, trainer_config, data_config)

    if args.dummy_data:
        from torch.utils.data import DataLoader
        dummy = DummyDataset(
            vocab_size=model_config.vocab_size,
            seq_len=trainer_config.seq_len,
            size=trainer_config.max_steps * args.batch_size,
        )
        dataloader = DataLoader(dummy, batch_size=args.batch_size, num_workers=2)
        trainer.train(dataloader)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
