#!/usr/bin/env python3
"""Full benchmark suite for Synapse-3B merged model.

Runs standard evaluations that match published benchmarks for Qwen2-3B class models:
- GSM8K (math reasoning) — full test set or configurable N
- HumanEval (code generation) — full 164 problems
- MMLU (general knowledge) — 5-shot, standard benchmark
- Speed test — tok/s on RTX 5090

Apples-to-apples comparison against published Qwen2-3B scores.
"""

import os
import sys
import torch
import json
import time
import re
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.expanduser("~/.synapse/merged/synapse-3b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\nLoading Synapse-3B from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
print(f"Model loaded: {type(model).__name__} on {DEVICE}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

results = {}

# ============================================================
# 1. GSM8K — Math Reasoning (standard: 8-shot CoT)
# Published Qwen2-3B: ~50-55% on GSM8K
# ============================================================
print("\n" + "="*60)
print("BENCHMARK 1: GSM8K (Math Reasoning)")
print("="*60)

from datasets import load_dataset

gsm = load_dataset("openai/gsm8k", "main", split="test")
GSM_N = int(os.environ.get("GSM_N", len(gsm)))  # default: full test set (1319)
print(f"Running {GSM_N} / {len(gsm)} problems")

# 8-shot examples (standard for GSM8K benchmark)
FEW_SHOT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 4 * 5 = 20 computers were added. 9 + 20 = 29. #### 29

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. #### 33

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each = 5 * 3 = 15 dollars. 23 - 15 = 8. #### 8"""

def extract_gsm_answer(text):
    """Extract the number after #### in GSM8K format."""
    matches = re.findall(r'####\s*([\-\d,\.]+)', text)
    if matches:
        return matches[-1].replace(",", "").strip()
    # Fallback: last number in the text
    numbers = re.findall(r'[\-]?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else ""

correct = 0
for i in range(GSM_N):
    q = gsm[i]["question"]
    gold = gsm[i]["answer"].split("####")[-1].strip().replace(",", "")

    prompt = f"{FEW_SHOT_EXAMPLES}\n\nQ: {q}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_gsm_answer(response)

    if pred == gold:
        correct += 1

    if (i + 1) % 50 == 0 or i == GSM_N - 1:
        print(f"  GSM8K: {i+1}/{GSM_N} done — {correct}/{i+1} correct ({correct/(i+1)*100:.1f}%)")

gsm_score = correct / GSM_N * 100
results["GSM8K"] = {"score": round(gsm_score, 1), "correct": correct, "total": GSM_N}
print(f"\nGSM8K Final: {gsm_score:.1f}% ({correct}/{GSM_N})")

# ============================================================
# 2. HumanEval — Code Generation (pass@1)
# Published Qwen2-3B: ~30-40% HumanEval
# ============================================================
print("\n" + "="*60)
print("BENCHMARK 2: HumanEval (Code Generation, pass@1)")
print("="*60)

try:
    he = load_dataset("openai/openai_humaneval", split="test")
    HE_N = int(os.environ.get("HE_N", len(he)))  # default: full 164 problems
    print(f"Running {HE_N} / {len(he)} problems")

    code_correct = 0
    code_errors = 0
    for i in range(HE_N):
        prompt = he[i]["prompt"]
        test_code = he[i]["test"]
        entry_point = he[i]["entry_point"]

        messages = [{"role": "user", "content": f"Complete this Python function. Return ONLY the function body, no explanation:\n\n{prompt}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Extract code
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            parts = response.split("```")
            code = parts[1] if len(parts) > 1 else response
        else:
            code = response

        full_code = prompt + code
        try:
            exec_globals = {}
            exec(full_code + "\n" + test_code, exec_globals)
            code_correct += 1
        except Exception:
            code_errors += 1

        if (i + 1) % 20 == 0 or i == HE_N - 1:
            print(f"  HumanEval: {i+1}/{HE_N} done — {code_correct}/{i+1} pass ({code_correct/(i+1)*100:.1f}%)")

    he_score = code_correct / HE_N * 100
    results["HumanEval"] = {"score": round(he_score, 1), "correct": code_correct, "total": HE_N}
    print(f"\nHumanEval Final: {he_score:.1f}% ({code_correct}/{HE_N})")
except Exception as e:
    print(f"HumanEval skipped: {e}")
    traceback.print_exc()

# ============================================================
# 3. MMLU — General Knowledge (5-shot)
# Published Qwen2-3B: ~53-55% on MMLU
# ============================================================
print("\n" + "="*60)
print("BENCHMARK 3: MMLU (General Knowledge, 5-shot)")
print("="*60)

try:
    # Use cais/mmlu which has all subjects
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu_dev = load_dataset("cais/mmlu", "all", split="dev")
    MMLU_N = int(os.environ.get("MMLU_N", len(mmlu)))
    print(f"Running {MMLU_N} / {len(mmlu)} problems")

    CHOICES = ["A", "B", "C", "D"]

    def format_mmlu_question(item, few_shot_items=None):
        """Format an MMLU question with optional few-shot examples."""
        subject = item.get("subject", "general knowledge").replace("_", " ")
        prompt = f"The following are multiple choice questions about {subject}.\n\n"

        if few_shot_items:
            for fs in few_shot_items[:5]:
                prompt += f"Question: {fs['question']}\n"
                for j, choice in enumerate(fs["choices"]):
                    prompt += f"{CHOICES[j]}. {choice}\n"
                prompt += f"Answer: {CHOICES[fs['answer']]}\n\n"

        prompt += f"Question: {item['question']}\n"
        for j, choice in enumerate(item["choices"]):
            prompt += f"{CHOICES[j]}. {choice}\n"
        prompt += "Answer:"
        return prompt

    # Group dev set by subject for few-shot
    dev_by_subject = {}
    for item in mmlu_dev:
        subj = item.get("subject", "unknown")
        if subj not in dev_by_subject:
            dev_by_subject[subj] = []
        dev_by_subject[subj].append(item)

    mmlu_correct = 0
    subject_results = {}

    for i in range(MMLU_N):
        item = mmlu[i]
        subj = item.get("subject", "unknown")
        few_shot = dev_by_subject.get(subj, [])[:5]

        prompt = format_mmlu_question(item, few_shot)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Extract first letter answer
        pred_letter = response[0].upper() if response and response[0].upper() in CHOICES else ""
        gold_letter = CHOICES[item["answer"]]

        is_correct = pred_letter == gold_letter
        if is_correct:
            mmlu_correct += 1

        if subj not in subject_results:
            subject_results[subj] = {"correct": 0, "total": 0}
        subject_results[subj]["total"] += 1
        if is_correct:
            subject_results[subj]["correct"] += 1

        if (i + 1) % 500 == 0 or i == MMLU_N - 1:
            print(f"  MMLU: {i+1}/{MMLU_N} done — {mmlu_correct}/{i+1} correct ({mmlu_correct/(i+1)*100:.1f}%)")

    mmlu_score = mmlu_correct / MMLU_N * 100
    results["MMLU"] = {"score": round(mmlu_score, 1), "correct": mmlu_correct, "total": MMLU_N}

    # Top and bottom subjects
    subject_scores = {}
    for subj, data in subject_results.items():
        if data["total"] >= 5:
            subject_scores[subj] = data["correct"] / data["total"] * 100
    top_subjects = sorted(subject_scores.items(), key=lambda x: -x[1])[:5]
    bottom_subjects = sorted(subject_scores.items(), key=lambda x: x[1])[:5]

    results["MMLU_top_subjects"] = {s: round(v, 1) for s, v in top_subjects}
    results["MMLU_bottom_subjects"] = {s: round(v, 1) for s, v in bottom_subjects}

    print(f"\nMMLU Final: {mmlu_score:.1f}% ({mmlu_correct}/{MMLU_N})")
    print(f"Top subjects: {top_subjects[:3]}")
    print(f"Bottom subjects: {bottom_subjects[:3]}")

except Exception as e:
    print(f"MMLU skipped: {e}")
    traceback.print_exc()

# ============================================================
# 4. Speed Benchmark
# ============================================================
print("\n" + "="*60)
print("BENCHMARK 4: Inference Speed")
print("="*60)

messages = [{"role": "user", "content": "Write a detailed explanation of how neural networks learn through backpropagation."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

# Warmup (3 runs)
print("Warming up...")
for _ in range(3):
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32, do_sample=False)

# Actual speed tests
speeds = []
for run in range(5):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    tokens = out.shape[1] - inputs["input_ids"].shape[1]
    tok_s = tokens / elapsed
    speeds.append(tok_s)
    print(f"  Run {run+1}: {tok_s:.1f} tok/s ({tokens} tokens in {elapsed:.2f}s)")

avg_speed = sum(speeds) / len(speeds)
max_speed = max(speeds)
min_speed = min(speeds)

results["speed"] = {
    "avg_tok_s": round(avg_speed, 1),
    "max_tok_s": round(max_speed, 1),
    "min_tok_s": round(min_speed, 1),
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "N/A",
    "dtype": "bfloat16" if DEVICE == "cuda" else "float32",
}
print(f"\nSpeed: avg {avg_speed:.1f} tok/s (min {min_speed:.1f}, max {max_speed:.1f})")

# ============================================================
# 5. TTFT (Time to First Token)
# ============================================================
print("\n" + "="*60)
print("BENCHMARK 5: Time to First Token (TTFT)")
print("="*60)

ttft_times = []
for run in range(10):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ttft = (time.time() - start) * 1000  # ms
    ttft_times.append(ttft)

avg_ttft = sum(ttft_times) / len(ttft_times)
p50_ttft = sorted(ttft_times)[5]
p99_ttft = sorted(ttft_times)[9]

results["ttft"] = {
    "avg_ms": round(avg_ttft, 1),
    "p50_ms": round(p50_ttft, 1),
    "p99_ms": round(p99_ttft, 1),
}
print(f"TTFT: avg {avg_ttft:.1f}ms, p50 {p50_ttft:.1f}ms, p99 {p99_ttft:.1f}ms")

# ============================================================
# VRAM Usage
# ============================================================
if DEVICE == "cuda":
    vram_used = torch.cuda.max_memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    results["vram"] = {
        "used_gb": round(vram_used, 2),
        "total_gb": round(vram_total, 2),
        "utilization_pct": round(vram_used / vram_total * 100, 1),
    }
    print(f"\nVRAM: {vram_used:.2f} GB / {vram_total:.2f} GB ({vram_used/vram_total*100:.1f}%)")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS — Synapse-3B (TIES Merged)")
print("="*60)

# Reference scores for Qwen2-3B (published)
print(f"\n{'Benchmark':<20} {'Synapse-3B':>12} {'Qwen2-3B (ref)':>15}")
print("-" * 50)
if "GSM8K" in results:
    print(f"{'GSM8K':<20} {results['GSM8K']['score']:>11.1f}% {'~54%':>15}")
if "HumanEval" in results:
    print(f"{'HumanEval':<20} {results['HumanEval']['score']:>11.1f}% {'~36%':>15}")
if "MMLU" in results:
    print(f"{'MMLU (5-shot)':<20} {results['MMLU']['score']:>11.1f}% {'~53%':>15}")
if "speed" in results:
    print(f"{'Tok/s (avg)':<20} {results['speed']['avg_tok_s']:>11.1f}  {'N/A':>15}")
if "ttft" in results:
    print(f"{'TTFT (avg)':<20} {results['ttft']['avg_ms']:>10.1f}ms {'N/A':>15}")

print(json.dumps(results, indent=2))

# Save
out_path = "/tmp/synapse-bench-results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
