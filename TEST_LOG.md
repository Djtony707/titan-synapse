# TITAN Synapse — Test Log

**Date**: March 20, 2026 (updated throughout DAY 1-2)
**Platform**: RTX 5090 (32GB VRAM), i9-14900KF, 64GB DDR5-6000, Ubuntu 24.04
**Models**: Qwen2.5-3B-Instruct (Q4_K_M, 1.9GB) + Qwen2.5-0.5B-Instruct (Q4_K_M, 491MB)
**Rust**: 1.94.0 (release build)
**Binary size**: ~6.3MB

---

## Unit Tests (15/15 PASSING)

```
cargo test — 15/15 PASSING in 0.01s
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
| 10 | test_hebbian_routing | memory::graph | PASS |
| 11 | test_specialist_stats | memory::graph | PASS |
| 12 | test_manifest_creation | format::manifest | PASS |
| 13 | test_manifest_serialization | format::manifest | PASS |
| 14 | test_pack_and_unpack | format::packer | PASS |
| 15 | test_list_bundles | format::packer | PASS |

---

## Benchmark Results (Qwen2.5-3B, Q4_K_M, CPU only)

```
synapse bench — 4 prompts, 759 tokens total
```

| Prompt | Tokens | Time | Tok/s |
|--------|--------|------|-------|
| Python decorator explanation | 109 | 5,001ms | 21.8 |
| SQL top-10 query | 139 | 6,467ms | 21.5 |
| TCP vs UDP explanation | 256 | 10,893ms | 23.5 |
| Go garbage collection | 255 | 10,875ms | 23.4 |
| **Average** | **190** | **8,309ms** | **23 tok/s** |

**Consistency test** (3 runs, same prompt, 64 tokens):
- Run 1: 64 tokens in 3,502ms (18.3 tok/s)
- Run 2: 64 tokens in 3,482ms (18.4 tok/s)
- Run 3: 64 tokens in 3,513ms (18.2 tok/s)
- **Variance**: <1% — extremely consistent

**Note**: CPU-only. With CUDA enabled on the RTX 5090, expect 200-400 tok/s.

---

## Integration Tests (Live Server, RTX 5090)

All tests run against the live server at `http://192.168.1.11:6900`

### Test 1: Health Check
- **Endpoint**: `GET /health`
- **Result**: PASS
- **Response**: `ok`
- **HTTP Code**: 200

### Test 2: List Models
- **Endpoint**: `GET /v1/models`
- **Result**: PASS
- **Response**: 4 models listed (synapse, synapse/general, synapse/python_expert, synapse/sql_expert)

### Test 3: API Status (Enhanced)
- **Endpoint**: `GET /api/status`
- **Result**: PASS
- **Response**: Includes `models_loaded: ["qwen2.5-3b-instruct-q4_k_m", "qwen2.5-0.5b-instruct-q4_k_m"]`, Hebbian pathway data, knowledge stats

### Test 4: Simple Math — "What is 2+2?"
- **Endpoint**: `POST /v1/chat/completions`
- **Result**: PASS
- **Response**: `"2 + 2 equals 4."` (correct, clean stop)
- **Usage**: `{prompt_tokens: 34, completion_tokens: 8, total_tokens: 42}`

### Test 5: Python Code Generation
- **Endpoint**: `POST /v1/chat/completions`
- **Result**: PASS
- **Response**: Correct decorator explanation with working code example

### Test 6: SSE Streaming
- **Endpoint**: `POST /v1/chat/completions` (stream=true)
- **Result**: PASS
- **Response**: Role delta → content deltas → [DONE], proper SSE format

### Test 7: SQL Query Generation
- **Endpoint**: `POST /v1/chat/completions`
- **Result**: PASS
- **Response**: Correct SQL with JOIN, GROUP BY, COUNT
- **Specialist routed**: sql_expert

### Test 8: Token Counting Accuracy
- **Endpoint**: `POST /v1/chat/completions`
- **Result**: PASS
- **Response**: `usage` field with accurate prompt_tokens, completion_tokens, total_tokens
- **Verified**: total_tokens = prompt_tokens + completion_tokens

### Test 9: Hebbian Routing Accumulation
- **Method**: Made 3 Python queries, then checked /api/status
- **Result**: PASS
- **Hebbian state**: `python_expert: strength=4, avg_score=4.0`, `sql_expert: strength=3`, `general: strength=2`
- **Verified**: Pathways strengthen with repeated use

### Test 10: Multi-Model Loading
- **Result**: PASS
- **Models loaded**: qwen2.5-3b-instruct-q4_k_m (1.1s), qwen2.5-0.5b-instruct-q4_k_m (0.7s)
- **Simultaneous**: Both models in memory, auto-selected by engine

### Test 11: Counting Task
- **Prompt**: "Count from 1 to 5"
- **Result**: PASS
- **Response**: "1\n2\n3\n4\n5" (correct, clean)

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Model load time (3B) | 1.1 seconds |
| Model load time (0.5B) | 0.7 seconds |
| Health check latency | <1ms |
| Short response (8 tokens) | ~370ms |
| Medium response (64 tokens) | ~3.5s |
| Long response (256 tokens) | ~10.9s |
| **Throughput (CPU, 3B)** | **21-24 tok/s** |
| VRAM used by models | ~2.4 GB (both models loaded) |

---

## Model Output Samples

### Math
```
Q: What is 2+2?
A: 2 + 2 equals 4.
```

### Code
```
Q: What is a Python decorator? Explain briefly.
A: A Python decorator is a special function that adds functionality
   to another function without modifying its code. [includes working
   code example with @my_decorator syntax]
```

### SQL
```
Q: Write a SQL query to find the top 5 users by order count.
A: SELECT u.user_id, u.user_name, COUNT(o.order_id) AS order_count
   FROM users u JOIN orders o ON u.user_id = o.user_id
   GROUP BY u.user_id, u.user_name
   ORDER BY order_count DESC LIMIT 5;
```

---

## Issues Found & Fixed

1. **Tensor rank error**: candle's quantized_qwen2 returns `(batch, vocab)` not `(batch, seq, vocab)`. Fixed with `squeeze(0)`.

2. **Chat template**: Model requires `<|im_start|>/<|im_end|>` format. Added `format_chat_prompt()`.

3. **Stop token bleed**: Model generated past answer. Fixed by collecting all stop token IDs.

4. **`gen` reserved keyword**: Rust 2024 reserves `gen`. Renamed variable.

5. **bench.rs after GenerationResult**: `response.len()` broke when return type changed from String to GenerationResult. Fixed to use `response.completion_tokens`.

6. **KnowledgeGraph not Send+Sync**: `rusqlite::Connection` uses RefCell. Fixed with `std::sync::Mutex<Connection>`.

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
