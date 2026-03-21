"""Data loading for Synapse pretraining.

Loads and tokenizes datasets for training the Synapse Architecture.
Includes general language + agent/tool-use data for TITAN compatibility.
"""

from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


@dataclass
class DataConfig:
    """Training data configuration."""
    seq_len: int = 2048
    batch_size: int = 4
    tokenizer_name: str = "Qwen/Qwen2.5-3B"
    num_workers: int = 2
    # Dataset mix (proportions)
    datasets: dict[str, float] = field(default_factory=lambda: {
        "cerebras/SlimPajama-627B": 0.5,      # General web text
        "HuggingFaceFW/fineweb-edu": 0.3,     # Educational content
        "teknium/OpenHermes-2.5": 0.1,        # Instruction following
        "glaiveai/glaive-function-calling-v2": 0.1,  # Tool/function calling
    })
    max_tokens: int = 2_000_000_000  # 2B tokens default


def get_tokenizer(name: str = "Qwen/Qwen2.5-3B"):
    """Load tokenizer. Uses Qwen2.5 (vocab_size=151936) to match config."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class StreamingTextDataset(IterableDataset):
    """Streaming dataset that loads and tokenizes on-the-fly.

    Avoids downloading entire datasets to disk — streams from HuggingFace.
    Packs sequences to seq_len with no padding waste.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = get_tokenizer(config.tokenizer_name)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset, interleave_datasets

        # Load each dataset as streaming
        streams = []
        probs = []
        for name, weight in self.config.datasets.items():
            try:
                ds = load_dataset(name, split="train", streaming=True, trust_remote_code=True)
                streams.append(ds)
                probs.append(weight)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
                continue

        if not streams:
            raise RuntimeError("No datasets could be loaded")

        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]

        # Interleave datasets
        combined = interleave_datasets(streams, probabilities=probs)

        # Token buffer for packing
        token_buffer = []
        tokens_yielded = 0

        for example in combined:
            # Extract text from various dataset formats
            text = self._extract_text(example)
            if not text:
                continue

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            # Yield packed sequences
            while len(token_buffer) >= self.config.seq_len + 1:
                chunk = token_buffer[:self.config.seq_len + 1]
                token_buffer = token_buffer[self.config.seq_len:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}

                tokens_yielded += self.config.seq_len
                if tokens_yielded >= self.config.max_tokens:
                    return

    def _extract_text(self, example: dict) -> str:
        """Extract text from various dataset formats."""
        # Try common field names
        for key in ["text", "content", "instruction", "conversations"]:
            if key in example:
                val = example[key]
                if isinstance(val, str):
                    return val
                elif isinstance(val, list):
                    # Chat format (list of messages)
                    return "\n".join(
                        m.get("value", m.get("content", ""))
                        for m in val if isinstance(m, dict)
                    )
        return ""


def create_dataloader(config: DataConfig) -> DataLoader:
    """Create a streaming DataLoader for training."""
    dataset = StreamingTextDataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,  # streaming doesn't support multi-worker well
        pin_memory=True,
    )


# Quick test dataset for development (no download needed)
class DummyDataset(Dataset):
    """Random token dataset for testing training loop without real data."""

    def __init__(self, vocab_size: int = 151936, seq_len: int = 2048, size: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }
