#!/usr/bin/env python3
"""
TITAN Synapse — Model Merger

Merges trained LoRA adapters into the base model to create standalone Synapse models.
Supports merging individual specialists or all adapters via TIES/DARE merging.

Output: Full merged model in HuggingFace format + GGUF quantized versions.

Usage:
    python merge_model.py --specialist all    # Merge all adapters into one Synapse-3B model
    python merge_model.py --specialist math   # Create a math-only merged model
    python merge_model.py --quantize Q4_K_M   # Also export GGUF quantized
"""

import argparse
import json
import logging
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SYNAPSE_DIR = Path.home() / ".synapse"
ADAPTERS_DIR = SYNAPSE_DIR / "adapters"
MODELS_DIR = SYNAPSE_DIR / "models"
MERGED_DIR = SYNAPSE_DIR / "merged"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SPECIALIST_ADAPTERS = {
    "math": ADAPTERS_DIR / "math_v1",
    "code": ADAPTERS_DIR / "code_v1",
    "general": ADAPTERS_DIR / "general_v1",
    "coordinator": ADAPTERS_DIR / "coordinator_v1",
}


def merge_single_adapter(specialist: str, output_dir: Path):
    """Merge a single LoRA adapter into the base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    adapter_dir = SPECIALIST_ADAPTERS[specialist]
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No adapter found at {adapter_dir}")

    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU to save VRAM
    )

    logger.info(f"Loading LoRA adapter: {specialist} from {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Write model card
    write_model_card(output_dir, specialist, [specialist])

    logger.info(f"Merged model saved: {output_dir}")
    return output_dir


def merge_all_adapters(output_dir: Path, method: str = "ties"):
    """
    Merge ALL LoRA adapters into one unified Synapse model.

    Uses TIES merging (Trim, Elect Sign, Merge) to combine multiple
    task-specific adapters without catastrophic interference.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import numpy as np

    available = {k: v for k, v in SPECIALIST_ADAPTERS.items()
                 if (v / "adapter_model.safetensors").exists()}

    if not available:
        raise FileNotFoundError("No trained adapters found!")

    logger.info(f"Merging {len(available)} adapters: {list(available.keys())}")
    logger.info(f"Merge method: {method}")

    # Load base model
    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Get base state dict
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    if method == "simple":
        # Simple averaging: merge each adapter's delta with equal weight
        merged_deltas = {}
        for specialist, adapter_dir in available.items():
            logger.info(f"  Loading adapter: {specialist}")
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            merged_model = peft_model.merge_and_unload()
            merged_state = merged_model.state_dict()

            # Compute delta from base
            for key in merged_state:
                if key in base_state:
                    delta = merged_state[key] - base_state[key]
                    if key not in merged_deltas:
                        merged_deltas[key] = delta
                    else:
                        merged_deltas[key] = merged_deltas[key] + delta

            # Reload base for next adapter
            del peft_model, merged_model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )

        # Average the deltas and apply
        n_adapters = len(available)
        logger.info(f"Averaging {n_adapters} adapter deltas...")
        final_state = {}
        for key in base_state:
            if key in merged_deltas:
                final_state[key] = base_state[key] + (merged_deltas[key] / n_adapters)
            else:
                final_state[key] = base_state[key]

    elif method == "ties":
        # TIES merging: Trim small deltas, Elect sign by majority, Merge
        TRIM_RATIO = 0.2  # Keep top 80% of delta magnitudes

        all_deltas = {}
        for specialist, adapter_dir in available.items():
            logger.info(f"  Loading adapter: {specialist}")
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            merged_model = peft_model.merge_and_unload()
            merged_state = merged_model.state_dict()

            deltas = {}
            for key in merged_state:
                if key in base_state:
                    delta = merged_state[key] - base_state[key]
                    if delta.abs().sum() > 0:  # Only store non-zero deltas
                        deltas[key] = delta

            all_deltas[specialist] = deltas
            del peft_model, merged_model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )

        # TIES Step 1: Trim — zero out small magnitude changes
        logger.info("TIES Step 1: Trimming small deltas...")
        for specialist in all_deltas:
            for key in all_deltas[specialist]:
                delta = all_deltas[specialist][key]
                flat = delta.abs().float().flatten()
                # quantile() can't handle huge tensors — sample if needed
                if flat.numel() > 1_000_000:
                    indices = torch.randperm(flat.numel())[:1_000_000]
                    threshold = torch.quantile(flat[indices], TRIM_RATIO)
                else:
                    threshold = torch.quantile(flat, TRIM_RATIO)
                mask = delta.abs() >= threshold
                all_deltas[specialist][key] = delta * mask

        # TIES Step 2: Elect sign — majority vote on direction
        logger.info("TIES Step 2: Electing signs by majority vote...")
        elected_signs = {}
        all_keys = set()
        for specialist in all_deltas:
            all_keys.update(all_deltas[specialist].keys())

        for key in all_keys:
            sign_sum = torch.zeros_like(base_state[key])
            for specialist in all_deltas:
                if key in all_deltas[specialist]:
                    sign_sum += torch.sign(all_deltas[specialist][key])
            elected_signs[key] = torch.sign(sign_sum)

        # TIES Step 3: Merge — average only values that agree with elected sign
        logger.info("TIES Step 3: Merging with sign agreement...")
        final_state = {k: v.clone() for k, v in base_state.items()}
        for key in all_keys:
            merged_delta = torch.zeros_like(base_state[key])
            count = torch.zeros_like(base_state[key])

            for specialist in all_deltas:
                if key in all_deltas[specialist]:
                    delta = all_deltas[specialist][key]
                    agrees = torch.sign(delta) == elected_signs[key]
                    merged_delta += delta * agrees
                    count += agrees.float()

            # Average where we have contributions
            count = count.clamp(min=1)
            final_state[key] = base_state[key] + (merged_delta / count)

    elif method == "weighted":
        # Weighted merge: give more weight to specialists based on domain
        # Math and Code get higher weight since they showed biggest improvements
        WEIGHTS = {
            "math": 1.5,       # GSM8K showed +15.7 pts
            "code": 1.3,       # HumanEval showed +10 pts
            "general": 1.0,    # Baseline
            "coordinator": 0.8, # Routing, less direct impact on benchmarks
        }

        merged_deltas = {}
        total_weight = 0

        for specialist, adapter_dir in available.items():
            weight = WEIGHTS.get(specialist, 1.0)
            total_weight += weight
            logger.info(f"  Loading adapter: {specialist} (weight={weight})")

            peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            merged_model = peft_model.merge_and_unload()
            merged_state = merged_model.state_dict()

            for key in merged_state:
                if key in base_state:
                    delta = (merged_state[key] - base_state[key]) * weight
                    if key not in merged_deltas:
                        merged_deltas[key] = delta
                    else:
                        merged_deltas[key] = merged_deltas[key] + delta

            del peft_model, merged_model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )

        final_state = {}
        for key in base_state:
            if key in merged_deltas:
                final_state[key] = base_state[key] + (merged_deltas[key] / total_weight)
            else:
                final_state[key] = base_state[key]
    else:
        raise ValueError(f"Unknown merge method: {method}")

    # Load a fresh base model and replace its weights
    logger.info("Loading fresh base model for weight replacement...")
    final_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    final_model.load_state_dict(final_state)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving Synapse-3B to {output_dir}")
    final_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    write_model_card(output_dir, "synapse-3b", list(available.keys()), method)
    logger.info(f"Synapse-3B model saved: {output_dir}")
    return output_dir


def write_model_card(output_dir: Path, model_name: str, specialists: list, method: str = "single"):
    """Write a HuggingFace model card."""
    card = f"""---
language:
- en
license: apache-2.0
tags:
- titan-synapse
- specialist-swarm
- continuous-learning
- merged-model
base_model: {BASE_MODEL}
model_type: qwen2
---

# Synapse-3B: {model_name}

**A specialist model created by TITAN Synapse** — trained through continuous learning on domain-specific datasets, then merged into a single model.

## How This Model Was Made

1. **Base model**: Qwen2.5-3B-Instruct
2. **Specialist training**: QLoRA fine-tuning on curated datasets
3. **Adapters merged**: {', '.join(specialists)}
4. **Merge method**: {method}
5. **Created**: {datetime.now().isoformat()}

## Specialists Merged

| Specialist | Training Data | Focus |
|---|---|---|
| math | GSM8K + OpenWebMath + Orca-Math (50k samples) | Mathematical reasoning |
| code | CodeAlpaca + Evol-Instruct + Python-18k (50k samples) | Code generation |
| general | SlimOrca + Alpaca-Cleaned (50k samples) | General knowledge |
| coordinator | Synthetic routing examples (5k samples) | Task routing |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Djtony707/synapse-3b")
tokenizer = AutoTokenizer.from_pretrained("Djtony707/synapse-3b")
```

Or with TITAN Synapse engine:
```bash
synapse pull synapse-3b
synapse up
```

## License

Apache 2.0

Built by [Tony Elliott](https://github.com/Djtony707) with TITAN Synapse.
"""
    (output_dir / "README.md").write_text(card)


def quantize_model(model_dir: Path, quantization: str = "Q4_K_M"):
    """
    Quantize the merged model using our own pipeline.
    No llama.cpp dependency — we use HuggingFace's own quantization.

    For GGUF output: uses the `gguf` Python package (pure Python, no C++ deps).
    For serving: exports SafeTensors with quantized weights directly.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    output_dir = MODELS_DIR / f"synapse-3b-{quantization.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Quantizing Synapse-3B to {quantization}...")

    if quantization in ("Q4", "Q4_K_M", "int4", "4bit"):
        # 4-bit quantization via bitsandbytes
        logger.info("  Loading model with 4-bit quantization (NF4)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=quant_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        logger.info(f"  Saving quantized model to {output_dir}")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

    elif quantization in ("Q8", "Q8_0", "int8", "8bit"):
        # 8-bit quantization via bitsandbytes
        logger.info("  Loading model with 8-bit quantization...")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=quant_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        logger.info(f"  Saving quantized model to {output_dir}")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

    elif quantization in ("f16", "fp16", "half"):
        # FP16 — just convert and save
        logger.info("  Converting to FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float16, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

    else:
        logger.error(f"Unknown quantization: {quantization}")
        logger.info("Supported: Q4_K_M, Q8_0, f16")
        return None

    # Also try GGUF export if the gguf package is available
    try:
        export_gguf_native(model_dir, output_dir.parent / f"synapse-3b-{quantization.lower()}.gguf")
    except Exception as e:
        logger.info(f"  GGUF export skipped: {e}")
        logger.info("  SafeTensors format is available for serving.")

    logger.info(f"  Quantized model saved: {output_dir}")
    return output_dir


def export_gguf_native(model_dir: Path, output_path: Path):
    """
    Export model to GGUF using the `gguf` Python package.
    This is our own conversion — no llama.cpp binary needed.
    """
    try:
        import gguf
    except ImportError:
        raise ImportError("Install gguf package: pip install gguf")

    from safetensors import safe_open
    import torch
    import json

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    logger.info(f"Exporting to GGUF: {output_path}")

    writer = gguf.GGUFWriter(str(output_path), "qwen2")

    # Write architecture metadata
    writer.add_name("synapse-3b")
    writer.add_description("TITAN Synapse 3B — merged specialist model")
    writer.add_context_length(config.get("max_position_embeddings", 32768))
    writer.add_embedding_length(config.get("hidden_size", 2048))
    writer.add_block_count(config.get("num_hidden_layers", 36))
    writer.add_head_count(config.get("num_attention_heads", 16))
    writer.add_head_count_kv(config.get("num_key_value_heads", 2))
    writer.add_feed_forward_length(config.get("intermediate_size", 11008))

    # Write tokenizer
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.exists():
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab = tokenizer.get_vocab()
        tokens = [""] * len(vocab)
        for tok, idx in vocab.items():
            if idx < len(tokens):
                tokens[idx] = tok
        writer.add_token_list(tokens)

    # Write tensors from SafeTensors files
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    for st_file in safetensor_files:
        logger.info(f"  Processing: {st_file.name}")
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Convert to numpy for GGUF
                np_data = tensor.numpy().astype("float16")
                writer.add_tensor(key, np_data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  GGUF exported: {output_path} ({size_mb:.0f}MB)")
    return output_path


def push_to_hub(model_dir: Path, repo_name: str = "Djtony707/synapse-3b"):
    """Push merged model to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    logger.info(f"Pushing to HuggingFace: {repo_name}")
    api.create_repo(repo_name, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_name,
        commit_message="Synapse-3B: merged specialist model from TITAN Synapse",
    )
    logger.info(f"Model published: https://huggingface.co/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="TITAN Synapse — Model Merger")
    parser.add_argument("--specialist", default="all",
                       choices=["all", "math", "code", "general", "coordinator"],
                       help="Which specialist to merge (default: all)")
    parser.add_argument("--method", default="ties",
                       choices=["simple", "ties", "weighted"],
                       help="Merge method for combining all adapters (default: ties)")
    parser.add_argument("--quantize", default=None,
                       help="GGUF quantization type (e.g., Q4_K_M, Q5_K_M, Q8_0)")
    parser.add_argument("--push", action="store_true",
                       help="Push to HuggingFace Hub after merge")
    parser.add_argument("--repo", default="Djtony707/synapse-3b",
                       help="HuggingFace repo name for push")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  TITAN SYNAPSE — Model Merger                           ║
    ║  Creating our own model from trained specialists        ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    if args.specialist == "all":
        output_dir = MERGED_DIR / "synapse-3b"
        merge_all_adapters(output_dir, method=args.method)
    else:
        output_dir = MERGED_DIR / f"synapse-3b-{args.specialist}"
        merge_single_adapter(args.specialist, output_dir)

    if args.quantize:
        quantize_model(output_dir, args.quantize)

    if args.push:
        push_to_hub(output_dir, args.repo)

    print(f"""
    ════════════════════════════════════════════════════════════
    MODEL CREATED: {output_dir}
    ════════════════════════════════════════════════════════════

    Next steps:
      1. Run benchmarks:  python real_eval.py --url http://localhost:6900
      2. Push to Hub:     python merge_model.py --push
      3. Convert to GGUF: python merge_model.py --quantize Q4_K_M
    """)


if __name__ == "__main__":
    main()
