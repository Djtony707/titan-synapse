#!/usr/bin/env python3
"""Benchmark Synapse-3B merged model on standard evaluations."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, os, time

MODEL_DIR = os.path.expanduser("~/.synapse/merged/synapse-3b")
print("Loading Synapse-3B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Try GPU first, fall back to CPU if Blackwell kernels not available
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    DEVICE = "cuda"
    print("Model loaded on GPU")
except RuntimeError:
    print("GPU failed (likely Blackwell kernel issue), using CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, device_map="cpu"
    )
    DEVICE = "cpu"
    print("Model loaded on CPU")

results = {}

# 1. GSM8K (math) - 50 samples
print("\n=== GSM8K (Math Reasoning) ===")
from datasets import load_dataset
gsm = load_dataset("openai/gsm8k", "main", split="test")
correct = 0
total = 20 if DEVICE == "cpu" else 50
for i in range(total):
    q = gsm[i]["question"]
    answer = gsm[i]["answer"].split("####")[-1].strip()
    messages = [{"role": "user", "content": f"Solve step by step: {q}\nGive your final answer after ####"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if answer in response:
        correct += 1
    if (i+1) % 10 == 0:
        print(f"  GSM8K: {i+1}/{total} done, {correct}/{i+1} correct")
gsm_score = correct / total * 100
results["GSM8K"] = gsm_score
print(f"GSM8K: {gsm_score:.1f}%")

# 2. HumanEval (code) - 20 samples
print("\n=== HumanEval (Code) ===")
he = load_dataset("openai/openai_humaneval", split="test")
code_correct = 0
code_total = 10 if DEVICE == "cpu" else 20
for i in range(code_total):
    prompt = he[i]["prompt"]
    test_code = he[i]["test"]
    messages = [{"role": "user", "content": f"Complete this Python function. Return ONLY the code, no explanation:\n\n{prompt}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Extract code from response
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        code = parts[1] if len(parts) > 1 else response
    else:
        code = response
    full_code = prompt + code
    try:
        exec(full_code + "\n" + test_code, {})
        code_correct += 1
    except Exception:
        pass
    if (i+1) % 10 == 0:
        print(f"  HumanEval: {i+1}/{code_total} done, {code_correct}/{i+1} correct")
he_score = code_correct / code_total * 100
results["HumanEval"] = he_score
print(f"HumanEval: {he_score:.1f}%")

# 3. Speed test
print("\n=== Speed Test ===")
messages = [{"role": "user", "content": "Write a detailed explanation of how neural networks learn."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
# Warmup
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=32, do_sample=False)
# Actual test
start = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
elapsed = time.time() - start
tokens = out.shape[1] - inputs["input_ids"].shape[1]
tok_s = tokens / elapsed
results["tok_per_sec"] = round(tok_s, 1)
results["tokens_generated"] = int(tokens)
results["time_seconds"] = round(elapsed, 2)
print(f"Speed: {tok_s:.1f} tok/s ({tokens} tokens in {elapsed:.1f}s)")

print("\n=== FINAL RESULTS ===")
print(json.dumps(results, indent=2))
with open("/tmp/synapse-bench-results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to /tmp/synapse-bench-results.json")
