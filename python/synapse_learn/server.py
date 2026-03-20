"""TITAN Synapse Learning Sidecar — FastAPI server for QLoRA training + self-evaluation."""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
import os
from pathlib import Path
from datetime import datetime

app = FastAPI(title="Synapse Learning Engine", version="0.1.0")

DATA_DIR = Path(os.environ.get("SYNAPSE_DATA_DIR", os.path.expanduser("~/.synapse")))
PREFERENCES_DIR = DATA_DIR / "preferences"
ADAPTERS_DIR = DATA_DIR / "adapters"
PREFERENCES_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)


class EvalRequest(BaseModel):
    specialist: str
    prompt: str
    response: str


class EvalResponse(BaseModel):
    score: float
    improved_response: Optional[str] = None
    feedback: str


class TrainRequest(BaseModel):
    specialist: str
    base_model: str


class TrainResponse(BaseModel):
    adapter_path: str
    loss: float
    pairs_used: int


class LearnStatus(BaseModel):
    pairs_collected: int
    training_queue: int
    last_trained: Optional[str] = None
    adapters_created: int


def count_preferences(specialist: Optional[str] = None) -> int:
    """Count preference pairs on disk."""
    total = 0
    for f in PREFERENCES_DIR.glob("*.jsonl"):
        if specialist and specialist not in f.name:
            continue
        with open(f) as fh:
            total += sum(1 for _ in fh)
    return total


def count_adapters() -> int:
    """Count created adapters."""
    return len(list(ADAPTERS_DIR.glob("*.safetensors")))


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "synapse-learn"}


@app.get("/status")
async def status():
    return LearnStatus(
        pairs_collected=count_preferences(),
        training_queue=count_preferences(),  # All uncollected pairs are queued
        last_trained=None,
        adapters_created=count_adapters(),
    )


@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    """Self-evaluate a response. Score 1-5, generate improved version if low."""
    # Simple heuristic scoring (will be replaced by model-based evaluation)
    score = 3.0
    feedback = "Baseline evaluation"

    response_len = len(req.response)
    if response_len < 50:
        score = 2.0
        feedback = "Response too short"
    elif response_len > 200:
        score = 4.0
        feedback = "Detailed response"

    # Check for common quality signals
    if "error" in req.response.lower() or "placeholder" in req.response.lower():
        score = 1.5
        feedback = "Response contains error/placeholder text"

    # Store preference pair if score is low
    if score < 3.0:
        pair = {
            "specialist": req.specialist,
            "prompt": req.prompt,
            "rejected": req.response,
            "chosen": None,  # Will be filled by improved generation
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }
        pref_file = PREFERENCES_DIR / f"{req.specialist}.jsonl"
        with open(pref_file, "a") as f:
            f.write(json.dumps(pair) + "\n")

    return EvalResponse(
        score=score,
        improved_response=None,
        feedback=feedback,
    )


@app.post("/train")
async def train(req: TrainRequest):
    """Trigger QLoRA training for a specialist."""
    pairs = count_preferences(req.specialist)

    if pairs == 0:
        return TrainResponse(
            adapter_path="",
            loss=0.0,
            pairs_used=0,
        )

    # TODO: Implement actual QLoRA training via Unsloth
    # from unsloth import FastLanguageModel
    # model, tokenizer = FastLanguageModel.from_pretrained(req.base_model)
    # ... training loop ...

    adapter_name = f"{req.specialist}_v1.safetensors"
    adapter_path = str(ADAPTERS_DIR / adapter_name)

    return TrainResponse(
        adapter_path=adapter_path,
        loss=0.0,
        pairs_used=pairs,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
