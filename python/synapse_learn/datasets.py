"""Public dataset downloader for specialist training.

Uses clean, factual, publicly available datasets from HuggingFace.
No proprietary data. No scraping. Just high-quality open datasets.

Available datasets:
- OpenWebMath: Mathematical reasoning
- The Stack v2: Code (Python, SQL, Rust, JS, etc.)
- SlimPajama: General knowledge
- FLAN: Instruction following
- MedQA: Medical knowledge
- Alpaca-Cleaned: General instructions
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("synapse-datasets")

DATA_DIR = Path(os.environ.get("SYNAPSE_DATA_DIR", os.path.expanduser("~/.synapse")))
DATASETS_DIR = DATA_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Registry of curated public datasets for specialist training
DATASET_REGISTRY = {
    "code_python": {
        "hf_name": "bigcode/starcoderdata",
        "subset": "python",
        "description": "Python code from The Stack — for python_expert specialist",
        "format": "code",
        "specialist": "python_expert",
    },
    "code_sql": {
        "hf_name": "b-mc2/sql-create-context",
        "subset": None,
        "description": "SQL queries with context — for sql_expert specialist",
        "format": "instruction",
        "specialist": "sql_expert",
    },
    "math": {
        "hf_name": "open-web-math/open-web-math",
        "subset": None,
        "description": "OpenWebMath — mathematical reasoning and proofs",
        "format": "text",
        "specialist": "math_expert",
    },
    "general_instruct": {
        "hf_name": "yahma/alpaca-cleaned",
        "subset": None,
        "description": "Cleaned Alpaca — general instruction following",
        "format": "instruction",
        "specialist": "general",
    },
    "science": {
        "hf_name": "camel-ai/physics",
        "subset": None,
        "description": "Physics Q&A — for science_expert specialist",
        "format": "qa",
        "specialist": "science_expert",
    },
    "writing": {
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "description": "FineWeb-Edu — high-quality educational text",
        "format": "text",
        "specialist": "writing_expert",
    },
}


def list_datasets() -> list:
    """List all available datasets in the registry."""
    return [
        {
            "id": k,
            "hf_name": v["hf_name"],
            "description": v["description"],
            "specialist": v["specialist"],
            "downloaded": (DATASETS_DIR / k).exists(),
        }
        for k, v in DATASET_REGISTRY.items()
    ]


def download_dataset(dataset_id: str, max_samples: int = 10000) -> dict:
    """Download a dataset from HuggingFace and prepare for training.

    Args:
        dataset_id: Key from DATASET_REGISTRY
        max_samples: Max number of samples to download (for VRAM-constrained training)

    Returns:
        Dict with path, sample count, and format info
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list(DATASET_REGISTRY.keys())}")

    info = DATASET_REGISTRY[dataset_id]
    output_dir = DATASETS_DIR / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        logger.info(f"Downloading {info['hf_name']}...")

        kwargs = {"split": f"train[:{max_samples}]", "trust_remote_code": True}
        if info["subset"]:
            kwargs["name"] = info["subset"]

        dataset = load_dataset(info["hf_name"], **kwargs)

        # Convert to training format
        training_data = []
        for item in dataset:
            formatted = format_for_training(item, info["format"])
            if formatted:
                training_data.append(formatted)

        # Save as JSONL
        output_file = output_dir / "train.jsonl"
        with open(output_file, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        # Save metadata
        meta = {
            "dataset_id": dataset_id,
            "hf_name": info["hf_name"],
            "samples": len(training_data),
            "format": info["format"],
            "specialist": info["specialist"],
            "downloaded_at": str(Path(output_file).stat().st_mtime),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Downloaded {len(training_data)} samples for {dataset_id}")

        return {
            "path": str(output_file),
            "samples": len(training_data),
            "specialist": info["specialist"],
        }

    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return {"error": "datasets library not installed"}
    except Exception as e:
        logger.error(f"Failed to download {dataset_id}: {e}")
        return {"error": str(e)}


def format_for_training(item: dict, fmt: str) -> Optional[dict]:
    """Convert a dataset item to Synapse training format.

    All training data is stored as chat-template formatted text:
    <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    """
    try:
        if fmt == "instruction":
            # Alpaca-style: instruction + input → output
            instruction = item.get("instruction") or item.get("question") or item.get("prompt", "")
            inp = item.get("input", "")
            output = item.get("output") or item.get("answer") or item.get("response", "")

            if not instruction or not output:
                return None

            prompt = f"{instruction}\n{inp}".strip() if inp else instruction
            return {
                "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
            }

        elif fmt == "code":
            # Code: use the content directly as training text
            content = item.get("content") or item.get("code") or item.get("text", "")
            if not content or len(content) < 50:
                return None
            # Truncate very long code
            content = content[:4096]
            return {
                "text": f"<|im_start|>user\nWrite the following code:<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>",
            }

        elif fmt == "qa":
            # Q&A format
            question = item.get("message_1") or item.get("question") or item.get("prompt", "")
            answer = item.get("message_2") or item.get("answer") or item.get("response", "")
            if not question or not answer:
                return None
            return {
                "text": f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>",
            }

        elif fmt == "text":
            # Raw text — used for continued pretraining
            text = item.get("text", "")
            if not text or len(text) < 100:
                return None
            text = text[:4096]
            return {"text": text}

        return None

    except Exception:
        return None


def prepare_specialist_dataset(specialist: str, max_samples: int = 5000) -> dict:
    """Download and prepare all relevant datasets for a specialist."""
    results = []
    for dataset_id, info in DATASET_REGISTRY.items():
        if info["specialist"] == specialist:
            result = download_dataset(dataset_id, max_samples)
            results.append(result)

    return {
        "specialist": specialist,
        "datasets": results,
        "total_samples": sum(r.get("samples", 0) for r in results),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Available datasets:")
    for ds in list_datasets():
        status = "✓" if ds["downloaded"] else "✗"
        print(f"  {status} {ds['id']}: {ds['description']} (for {ds['specialist']})")
