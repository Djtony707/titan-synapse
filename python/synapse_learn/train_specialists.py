"""TITAN Synapse — Real Specialist Training Pipeline

Downloads actual high-quality datasets and trains QLoRA specialist adapters
that will measurably improve benchmark scores.

Target improvements:
- HumanEval: 65.2% → 75%+ (code specialist trained on CodeAlpaca + Evol-Instruct)
- GSM8K: 83.7% → 90%+ (math specialist trained on MetaMathQA + Orca-Math)
- MMLU: 61.9% → 65%+ (general specialist trained on SlimOrca + OpenHermes)
- TruthfulQA: 89.1% → maintain (honesty/refusal training)

Hardware: RTX 5090 32GB VRAM
Training time: ~2-4 hours per specialist (10-20k samples, 2 epochs)

Usage:
    python train_specialists.py --specialist all
    python train_specialists.py --specialist code
    python train_specialists.py --specialist math
    python train_specialists.py --specialist general
    python train_specialists.py --specialist coordinator
"""

import os
import sys
import json
import logging
import argparse
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional

# Fix import path (same as real_eval.py)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger("synapse-train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DATA_DIR = Path(os.environ.get("SYNAPSE_DATA_DIR", os.path.expanduser("~/.synapse")))
ADAPTERS_DIR = DATA_DIR / "adapters"
TRAINING_DIR = DATA_DIR / "training"
CACHE_DIR = DATA_DIR / "hf_cache"

for d in [ADAPTERS_DIR, TRAINING_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ============================================================
# Dataset Definitions — what we train each specialist on
# ============================================================

SPECIALIST_DATASETS = {
    "code": {
        "description": "Code generation specialist — targets HumanEval improvement",
        "datasets": [
            {
                "name": "sahil2801/CodeAlpaca-20k",
                "split": "train",
                "samples": 20000,
                "format": "alpaca",  # instruction/input/output
            },
            {
                "name": "nickrosh/Evol-Instruct-Code-80k-v1",
                "split": "train",
                "samples": 20000,
                "format": "evol",  # instruction/output
            },
            {
                "name": "iamtarun/python_code_instructions_18k_alpaca",
                "split": "train",
                "samples": 18000,
                "format": "alpaca",
            },
        ],
        "system_prompt": "You are an expert programmer. Write clean, correct, well-tested code. Always include proper error handling.",
        "lora_rank": 64,
        "epochs": 2,
        "max_seq_length": 2048,
        "learning_rate": 2e-4,
    },
    "math": {
        "description": "Math reasoning specialist — targets GSM8K improvement",
        "datasets": [
            {
                "name": "meta-math/MetaMathQA",
                "split": "train",
                "samples": 30000,
                "format": "metamath",  # query/response
            },
            {
                "name": "microsoft/orca-math-word-problems-200k",
                "split": "train",
                "samples": 20000,
                "format": "orca_math",  # question/answer
            },
        ],
        "system_prompt": "You are a math expert. Solve problems step by step, showing your work clearly. Always verify your answer at the end.",
        "lora_rank": 64,
        "epochs": 2,
        "max_seq_length": 1024,
        "learning_rate": 2e-4,
    },
    "general": {
        "description": "General knowledge specialist — targets MMLU improvement",
        "datasets": [
            {
                "name": "Open-Orca/SlimOrca",
                "split": "train",
                "samples": 25000,
                "format": "sharegpt",  # conversations list
            },
            {
                "name": "yahma/alpaca-cleaned",
                "split": "train",
                "samples": 25000,
                "format": "alpaca",
            },
        ],
        "system_prompt": "You are a knowledgeable assistant. Give accurate, well-structured answers. If you're unsure, say so.",
        "lora_rank": 64,
        "epochs": 2,
        "max_seq_length": 2048,
        "learning_rate": 1.5e-4,
    },
    "coordinator": {
        "description": "Swarm coordinator — routes queries to the right specialist",
        "datasets": [],  # Generated synthetically below
        "system_prompt": "You are a routing coordinator. Analyze each query and decide which specialist should handle it. Output JSON with your routing decision.",
        "lora_rank": 32,
        "epochs": 3,
        "max_seq_length": 512,
        "learning_rate": 2e-4,
    },
}


# ============================================================
# Data Formatting — convert each dataset to chat template
# ============================================================

def format_chat(system: str, user: str, assistant: str) -> str:
    """Format a conversation in Qwen2.5 chat template."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
    return "\n".join(parts)


def format_dataset_item(item: dict, fmt: str, system_prompt: str) -> Optional[str]:
    """Convert a dataset item to chat-template formatted text."""
    try:
        if fmt == "alpaca":
            instruction = item.get("instruction", "") or item.get("prompt", "")
            inp = item.get("input", "")
            output = item.get("output", "")
            if not instruction or not output:
                return None
            user = f"{instruction}\n{inp}".strip() if inp else instruction
            return format_chat(system_prompt, user, output)

        elif fmt == "evol":
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            if not instruction or not output:
                return None
            return format_chat(system_prompt, instruction, output)

        elif fmt == "metamath":
            query = item.get("query", "")
            response = item.get("response", "")
            if not query or not response:
                return None
            return format_chat(system_prompt, query, response)

        elif fmt == "orca_math":
            question = item.get("question", "")
            answer = item.get("answer", "")
            if not question or not answer:
                return None
            return format_chat(system_prompt, question, answer)

        elif fmt == "sharegpt":
            # SlimOrca uses conversations format
            convos = item.get("conversations", [])
            if not convos or len(convos) < 2:
                return None
            # Find system, human, gpt messages
            system = ""
            user = ""
            assistant = ""
            for msg in convos:
                role = msg.get("from", "")
                value = msg.get("value", "")
                if role == "system":
                    system = value
                elif role == "human":
                    user = value
                elif role == "gpt":
                    assistant = value
            if not user or not assistant:
                return None
            return format_chat(system or system_prompt, user, assistant)

        elif fmt == "routing":
            # Pre-formatted routing examples
            return item.get("text", None)

        return None
    except Exception:
        return None


# ============================================================
# Coordinator Routing Data Generation
# ============================================================

def generate_routing_data(count: int = 5000) -> list:
    """Generate synthetic routing training data for the coordinator.

    The coordinator needs to learn:
    1. Which specialist handles which query
    2. When to use swarm mode (multi-specialist)
    3. Confidence scoring
    """
    import random

    templates = {
        "python_expert": [
            "Write a Python function to {task}",
            "Debug this Python code: {code}",
            "How do I {task} in Python?",
            "Create a Python class that {task}",
            "What's the best way to {task} with Python?",
            "Optimize this Python code for performance",
            "Write unit tests for this Python function",
            "Explain this Python error: {error}",
            "Convert this code to async Python",
            "Build a FastAPI endpoint that {task}",
        ],
        "sql_expert": [
            "Write a SQL query to {task}",
            "Optimize this SQL query for performance",
            "How do I join these tables: {tables}",
            "Create a database schema for {domain}",
            "Explain this SQL execution plan",
            "Write a stored procedure that {task}",
            "Migrate this schema from MySQL to Postgres",
            "Add an index to improve this query",
            "Write a complex GROUP BY query for {task}",
            "Design a normalized database for {domain}",
        ],
        "math_expert": [
            "Solve this equation: {equation}",
            "Calculate the probability of {event}",
            "Prove that {theorem}",
            "What is the derivative of {function}?",
            "How many ways can you {task}?",
            "Solve this word problem: {problem}",
            "Find the integral of {function}",
            "What is the expected value of {event}?",
            "Simplify this expression: {expression}",
            "Solve this system of equations",
        ],
        "general": [
            "Explain {topic} in simple terms",
            "What is {concept}?",
            "Compare {a} and {b}",
            "Summarize the key points of {topic}",
            "What are the pros and cons of {topic}?",
            "How does {concept} work?",
            "Give me an overview of {topic}",
            "What should I know about {topic}?",
            "Tell me about {topic}",
            "What's the difference between {a} and {b}?",
        ],
    }

    # Fillers for templates
    python_tasks = [
        "sort a list of dictionaries", "parse JSON", "handle file uploads",
        "connect to a database", "build a web scraper", "implement a binary tree",
        "create a REST API", "process CSV files", "implement caching",
        "handle concurrent requests", "validate email addresses", "parse dates",
        "implement rate limiting", "build a CLI tool", "create a decorator",
        "manage environment variables", "implement retry logic", "stream large files",
        "build a state machine", "implement pagination",
    ]

    sql_tasks = [
        "find duplicate records", "aggregate sales by month", "rank customers",
        "calculate running totals", "find gaps in sequences", "pivot data",
        "merge overlapping ranges", "find nth highest salary",
        "recursive CTE for hierarchies", "window functions for analytics",
    ]

    math_problems = [
        "a train traveling at 60mph", "probability of rolling dice",
        "compound interest calculation", "geometric sequence sum",
        "optimization of area", "combinatorics problem",
        "linear regression coefficients", "matrix multiplication",
    ]

    topics = [
        "machine learning", "quantum computing", "blockchain", "neural networks",
        "cloud computing", "cybersecurity", "distributed systems", "microservices",
        "DevOps practices", "agile methodology", "functional programming",
        "graph databases", "event-driven architecture", "container orchestration",
        "data warehousing", "API design", "load balancing", "caching strategies",
    ]

    # Swarm (multi-specialist) examples
    swarm_queries = [
        "Build a REST API with a database, authentication, and tests",
        "Create a data pipeline that processes CSV, stores in PostgreSQL, and generates reports",
        "Refactor this legacy codebase and add comprehensive tests",
        "Design a microservice architecture with database schemas and API contracts",
        "Build a machine learning pipeline with data preprocessing, model training, and evaluation",
        "Create a web application with user auth, database, API, and deployment scripts",
        "Analyze this dataset statistically and create visualizations with Python",
        "Optimize both the SQL queries and the Python code calling them",
        "Build a recommendation system with a database backend and API layer",
        "Create a monitoring dashboard with alerting, database queries, and Python scripts",
    ]

    data = []
    specialist_names = list(templates.keys())

    for i in range(count):
        if random.random() < 0.15:  # 15% swarm examples
            query = random.choice(swarm_queries)
            specialists = random.sample(specialist_names, k=random.randint(2, 3))
            subtasks = []
            for s in specialists:
                subtasks.append({"specialist": s, "task": f"Handle the {s.replace('_expert', '')} aspect"})
            response = json.dumps({
                "mode": "swarm",
                "confidence": round(random.uniform(0.80, 0.95), 2),
                "subtasks": subtasks,
            })
        else:  # Single specialist
            specialist = random.choice(specialist_names)
            template = random.choice(templates[specialist])

            # Fill in template
            if specialist == "python_expert":
                query = template.format(
                    task=random.choice(python_tasks),
                    code="...",
                    error="TypeError: unsupported operand type",
                )
            elif specialist == "sql_expert":
                query = template.format(
                    task=random.choice(sql_tasks),
                    tables="users, orders, products",
                    domain=random.choice(["e-commerce", "healthcare", "finance", "social media"]),
                )
            elif specialist == "math_expert":
                query = template.format(
                    task=random.choice(math_problems),
                    equation="2x² + 3x - 5 = 0",
                    theorem="the sum of angles in a triangle is 180°",
                    function="x³ + 2x",
                    event="getting at least one head in 3 coin flips",
                    problem=random.choice(math_problems),
                    expression="(3x² + 2x) / x",
                )
            else:  # general
                t1, t2 = random.sample(topics, 2)
                query = template.format(
                    topic=t1, concept=t1, a=t1, b=t2,
                )

            response = json.dumps({
                "mode": "single",
                "specialist": specialist,
                "confidence": round(random.uniform(0.82, 0.98), 2),
                "reasoning": f"Query matches {specialist} domain",
            })

        text = format_chat(
            "You are a routing coordinator for a specialist AI swarm. Analyze the query and decide which specialist(s) should handle it. Respond with JSON.",
            f"Route this query: {query}",
            response,
        )
        data.append({"text": text})

    return data


# ============================================================
# Training Core
# ============================================================

def load_and_prepare_data(specialist: str, max_total: int = 50000) -> list:
    """Download and format training data for a specialist."""
    config = SPECIALIST_DATASETS[specialist]
    all_texts = []

    if specialist == "coordinator":
        logger.info("Generating synthetic routing data...")
        data = generate_routing_data(count=5000)
        all_texts = [d["text"] for d in data]
        logger.info(f"Generated {len(all_texts)} routing examples")
        return all_texts

    system_prompt = config["system_prompt"]

    for ds_info in config["datasets"]:
        name = ds_info["name"]
        split = ds_info["split"]
        samples = min(ds_info["samples"], max_total - len(all_texts))
        fmt = ds_info["format"]

        if samples <= 0:
            break

        logger.info(f"Loading {name} ({samples} samples)...")
        try:
            # Use streaming for large datasets to avoid downloading everything
            if samples < 50000:
                dataset = load_dataset(
                    name,
                    split=f"{split}[:{samples}]",
                    cache_dir=str(CACHE_DIR),
                )
            else:
                dataset = load_dataset(
                    name,
                    split=split,
                    streaming=True,
                    cache_dir=str(CACHE_DIR),
                )

            count = 0
            for item in dataset:
                if count >= samples:
                    break
                text = format_dataset_item(item, fmt, system_prompt)
                if text and len(text) > 50:
                    all_texts.append(text)
                    count += 1

                if count % 5000 == 0 and count > 0:
                    logger.info(f"  Processed {count}/{samples} from {name}")

            logger.info(f"  Got {count} valid samples from {name}")

        except Exception as e:
            logger.error(f"  Failed to load {name}: {e}")
            continue

    logger.info(f"Total training samples for {specialist}: {len(all_texts)}")
    return all_texts


def train_specialist(
    specialist: str,
    base_model: str = BASE_MODEL,
    resume_from: Optional[str] = None,
) -> dict:
    """Train a QLoRA adapter for a specialist."""
    config = SPECIALIST_DATASETS[specialist]
    logger.info("=" * 70)
    logger.info(f"TRAINING SPECIALIST: {specialist}")
    logger.info(f"  {config['description']}")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  LoRA rank: {config['lora_rank']}")
    logger.info(f"  Epochs: {config['epochs']}")
    logger.info(f"  Max seq length: {config['max_seq_length']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info("=" * 70)

    start_time = time.time()

    # Step 1: Load and prepare data
    logger.info("\n[1/4] Loading training data...")
    texts = load_and_prepare_data(specialist)

    if not texts:
        logger.error("No training data available!")
        return {"error": "no_data", "specialist": specialist}

    # Save training data for reproducibility
    data_file = TRAINING_DIR / f"{specialist}_train.jsonl"
    with open(data_file, "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
    logger.info(f"  Saved {len(texts)} samples to {data_file}")

    dataset = Dataset.from_dict({"text": texts})

    # Step 2: Load base model with QLoRA
    logger.info("\n[2/4] Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager",
    )

    model = prepare_model_for_kbit_training(model)

    # Step 3: Configure LoRA
    logger.info("\n[3/4] Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"] * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Print GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # Step 4: Train
    logger.info("\n[4/4] Starting training...")
    output_dir = str(ADAPTERS_DIR / f"{specialist}_v1")

    training_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        packing=True,  # Pack short examples for efficiency
        gradient_checkpointing=True,  # Save VRAM
        optim="paged_adamw_8bit",  # 8-bit optimizer saves VRAM
        report_to="none",  # No wandb/tensorboard
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train(resume_from_checkpoint=resume_from)
    duration = time.time() - start_time

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    meta = {
        "specialist": specialist,
        "description": config["description"],
        "base_model": base_model,
        "lora_rank": config["lora_rank"],
        "training_loss": result.training_loss,
        "epochs": config["epochs"],
        "samples": len(texts),
        "duration_seconds": round(duration),
        "duration_human": f"{duration/3600:.1f}h",
        "trainable_params": trainable_params,
        "total_params": total_params,
        "max_seq_length": config["max_seq_length"],
        "learning_rate": config["learning_rate"],
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "created_by": "titan-synapse",
    }
    with open(Path(output_dir) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info(f"TRAINING COMPLETE: {specialist}")
    logger.info(f"  Loss: {result.training_loss:.4f}")
    logger.info(f"  Duration: {duration/3600:.1f}h ({duration:.0f}s)")
    logger.info(f"  Samples: {len(texts)}")
    logger.info(f"  Adapter saved: {output_dir}")
    logger.info("=" * 70)

    # Cleanup GPU memory
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return meta


def merge_and_export(specialist: str, quantize: str = "Q4_K_M") -> dict:
    """Merge LoRA adapter into base model and export as GGUF."""
    adapter_dir = ADAPTERS_DIR / f"{specialist}_v1"

    if not adapter_dir.exists():
        logger.error(f"Adapter not found: {adapter_dir}")
        return {"error": "adapter_not_found"}

    logger.info(f"Merging {specialist} adapter into base model...")

    # Load base + adapter
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()

    # Save merged model
    merge_dir = TRAINING_DIR / f"{specialist}_merged"
    merged.save_pretrained(str(merge_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(merge_dir))

    logger.info(f"Merged model saved to {merge_dir}")
    logger.info(f"To convert to GGUF, run:")
    logger.info(f"  python llama.cpp/convert_hf_to_gguf.py {merge_dir} --outtype {quantize.lower()}")

    del model, merged
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "specialist": specialist,
        "merged_path": str(merge_dir),
        "next_step": f"Convert to GGUF with llama.cpp",
    }


# ============================================================
# Main Pipeline
# ============================================================

def train_all_specialists():
    """Train all specialist adapters sequentially."""
    results = {}
    specialists = ["math", "code", "general", "coordinator"]

    logger.info("=" * 70)
    logger.info("TITAN SYNAPSE — Full Specialist Training Pipeline")
    logger.info(f"Training {len(specialists)} specialists: {', '.join(specialists)}")
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        logger.info(f"VRAM: {total_vram:.0f}GB")
    logger.info("=" * 70)

    total_start = time.time()

    for specialist in specialists:
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting training for: {specialist}")
        logger.info(f"{'='*70}\n")

        try:
            result = train_specialist(specialist)
            results[specialist] = result
        except Exception as e:
            logger.error(f"Failed to train {specialist}: {e}", exc_info=True)
            results[specialist] = {"error": str(e)}

    total_duration = time.time() - total_start

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Total duration: {total_duration/3600:.1f}h")
    logger.info("")
    for specialist, result in results.items():
        if "error" in result:
            logger.info(f"  ✗ {specialist}: FAILED — {result['error']}")
        else:
            logger.info(f"  ✓ {specialist}: loss={result['training_loss']:.4f}, "
                       f"samples={result['samples']}, "
                       f"time={result['duration_human']}")
    logger.info("=" * 70)

    # Save pipeline results
    with open(TRAINING_DIR / "pipeline_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "results": results,
        }, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Synapse specialist adapters")
    parser.add_argument(
        "--specialist",
        choices=["all", "code", "math", "general", "coordinator"],
        default="all",
        help="Which specialist to train (default: all)",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help=f"Base model (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Merge and export trained adapter after training",
    )

    args = parser.parse_args()

    if args.base_model != BASE_MODEL:
        BASE_MODEL = args.base_model

    if args.specialist == "all":
        results = train_all_specialists()
    else:
        result = train_specialist(args.specialist, base_model=BASE_MODEL)
        if args.export and "error" not in result:
            merge_and_export(args.specialist)
