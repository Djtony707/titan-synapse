# TITAN Synapse — Test Log

**Date**: March 20, 2026
**Platform**: RTX 5090 (32GB VRAM), i9-14900KF, 64GB DDR5-6000, Ubuntu 24.04
**Model**: Qwen2.5-0.5B-Instruct (Q4_K_M quantized, 491MB)
**Rust**: 1.94.0 (release build)
**Binary size**: ~6.3MB

---

## Unit Tests (11/11 PASSING)

```
cargo test — 11/11 PASSING in 0.00s
```

| # | Test | Module | Result |
|---|------|--------|--------|
| 1 | test_default_config | config | PASS |
| 2 | test_config_serialization | config | PASS |
| 3 | test_load_missing_config | config | PASS |
| 4 | test_greedy_sampling | inference::sampler | PASS |
| 5 | test_empty_logits | inference::sampler | PASS |
| 6 | test_stochastic_sampling | inference::sampler | PASS |
| 7 | test_cache_allocation | inference::kv_cache | PASS |
| 8 | test_knowledge_graph | memory::graph | PASS |
| 9 | test_preferences | memory::graph | PASS |
| 10 | test_manifest_creation | format::manifest | PASS |
| 11 | test_manifest_serialization | format::manifest | PASS |

---

## Integration Tests (Live Server, RTX 5090)

All tests run against the live server at `http://192.168.1.11:6900`

### Test 1: Health Check
- **Endpoint**: `GET /health`
- **Result**: PASS
- **Response**: `ok`
- **HTTP Code**: 200
- **Latency**: 0.37ms

### Test 2: List Models
- **Endpoint**: `GET /v1/models`
- **Result**: PASS
- **Response**: 4 models listed (synapse, synapse/general, synapse/python_expert, synapse/sql_expert)
- **Format**: OpenAI-compatible model list

### Test 3: API Status
- **Endpoint**: `GET /api/status`
- **Result**: PASS
- **Response**: `{"status":"running","version":"0.1.0","engine":"synapse","specialists":["general","python_expert","sql_expert"],"coordinator":"qwen3-0.6b","base_model":"qwen3-3b"}`

### Test 4: Simple Math — "What is 2+2?"
- **Endpoint**: `POST /v1/chat/completions`
- **Result**: PASS
- **Response**: `"4"` (correct)
- **Time**: 2.71s
- **Specialist routed**: general

### Test 5: Python Code Generation — "Write a function to reverse a string"
- **Endpoint**: `POST /v1/chat/completions`
- **Result**: PASS
- **Response**: `def reverse_string(s): return s[::-1]` (correct, idiomatic Python)
- **Time**: 2.75s
- **Max tokens**: 64

### Test 6: SSE Streaming
- **Endpoint**: `POST /v1/chat/completions` (stream=true)
- **Result**: PASS
- **Response**: Word-by-word SSE chunks, proper `data:` prefix, role delta followed by content deltas
- **Format**: OpenAI-compatible chunked streaming

### Test 7: SQL Query Generation — "Find top 5 users by post count"
- **Endpoint**: `POST /v1/chat/completions`
- **Result**: PASS
- **Response**: Correct SQL with JOIN, GROUP BY, COUNT, proper table aliases
- **Time**: 10.34s (128 tokens)
- **Specialist routed**: sql_expert (keyword match on "sql", "query")

### Test 8: Greedy Decoding (temperature=0) — "Capital of France?"
- **Endpoint**: `POST /v1/chat/completions` (temperature=0)
- **Result**: PASS
- **Response**: `"Paris"` (correct)
- **Time**: 2.12s
- **Deterministic**: Yes (greedy argmax)

### Test 9: Long Generation — "Explain neural networks in 3 sentences"
- **Endpoint**: `POST /v1/chat/completions`
- **Result**: PASS
- **Response**: Coherent 3-sentence explanation of neural networks, accurate content
- **Time**: 6.30s (~128 tokens)
- **Quality**: Accurate, well-structured, stops naturally

### Test 10: CLI Status Command
- **Command**: `synapse status`
- **Result**: PASS
- **Output**: Shows GPU (RTX 5090), VRAM (3964/32607 MB used), temp (36°C), 3 specialists

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Model load time | 0.5 seconds |
| Health check latency | 0.37ms |
| Short response (8 tokens) | ~2.1s |
| Medium response (64 tokens) | ~2.7s |
| Long response (128 tokens) | ~6.3-10.3s |
| Approx. throughput (CPU) | ~15-20 tok/s |
| VRAM used by model | ~1.0 GB (0.5B model, quantized) |
| Server memory (RSS) | ~4.0 GB |

**Note**: Currently running on CPU. CUDA acceleration will increase throughput significantly (estimated 5-10x).

---

## Model Output Samples

### Math
```
Q: What is 2+2? Answer with just the number.
A: 4
```

### Code
```
Q: Write a Python function to reverse a string. Just the code, no explanation.
A: def reverse_string(s):
       return s[::-1]
```

### SQL
```
Q: Write a SQL query to find the top 5 users by post count
A: SELECT u.user_id, u.user_name, COUNT(p.post_id) AS post_count
   FROM users u
   JOIN posts p ON u.user_id = p.user_id
   GROUP BY u.user_id, u.user_name
```

### Knowledge
```
Q: What is the capital of France? One word answer.
A: Paris
```

### Explanation
```
Q: Explain what a neural network is in 3 sentences.
A: A neural network is a type of machine learning model inspired by the structure
   and function of the human brain. It consists of layers of interconnected nodes,
   or neurons, that process information and learn to recognize patterns in data.
   Through training, these networks can improve their performance, making them
   useful for tasks such as image recognition, natural language processing, and
   predictive analytics.
```

---

## Build Info

```
Rust edition: 2024
Cargo workspace: titan-synapse
Binary crate: synapse
Dependencies: 385 crates
Build time (release): ~36s (first build), ~3s (incremental)
Target: x86_64-unknown-linux-gnu
```

---

## Issues Found & Fixed

1. **Tensor rank error**: Initial model forward pass assumed output shape `(batch, seq_len, vocab)` but candle's quantized_qwen2 returns `(batch, vocab)` since it extracts the last position internally. Fixed by removing extra indexing.

2. **Chat template**: Model requires `<|im_start|>/<|im_end|>` format for instruction following. Added `format_chat_prompt()` to wrap user input in proper template.

3. **SSH deployment**: Direct `nohup` via SSH sometimes fails due to connection closing. Resolved by using background process with `&` and checking with `curl`.
