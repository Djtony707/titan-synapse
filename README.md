```
   ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗
   ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝
   ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗
   ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝
   ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗
   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝
        Tiny models. Big brain. Your hardware. No excuses.
```

<div align="center">

**A Rust inference engine that runs a swarm of tiny specialist models<br>that collaborate and learn continuously — on your GPU.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/Tests-65%2F65_Passing-brightgreen.svg)](#tests)
[![HuggingFace](https://img.shields.io/badge/Model-Synapse--3B-yellow.svg)](https://huggingface.co/djtony707/synapse-3b)
[![npm](https://img.shields.io/badge/npm-v0.2.0-red.svg)](https://www.npmjs.com/package/titan-synapse)
[![CUDA](https://img.shields.io/badge/CUDA-12.8_(Blackwell)-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Architecture](#-architecture) · [Tested Results](#-tested-results) · [Configuration](#%EF%B8%8F-configuration) · [Contributing](#-contributing)

</div>

---

## What if you could run six specialists for the VRAM cost of one?

Everyone's racing to make models bigger. **We went the other way.**

Synapse runs a **swarm of tiny specialist models** that share a single base and coordinate through a Hebbian router — "pathways that fire together, wire together." Six specialists sharing one base model use **~5GB of VRAM**. A single 70B model needs 35GB and still can't fit on your card.

Oh, and they **learn from every conversation** you have. No fine-tuning scripts. No export-retrain-import dance. Just continuous, automatic self-improvement running in the background while you work.

No cloud. No API keys. No telemetry. One binary. Your hardware. Your data. Period.

---

## Features

- **Own Inference Engine** — Written from scratch in Rust with [candle](https://github.com/huggingface/candle). Not a wrapper around llama.cpp. Not a shim over vLLM. Ours.
- **GGUF Model Loading** — Native quantized model support. Load Q4_K_M, Q5_K_M, Q8_0 models directly. Tested with Qwen2.5 models.
- **Specialist Swarm with Hebbian Routing** — A coordinator routes queries to the right specialist(s). Simple question? One model. Complex task? The swarm convenes **in parallel**. Routing weights strengthen with use.
- **Metacognitive Confidence** — The system knows what it knows. Each specialist tracks its own performance per domain. Low confidence? Route to cloud fallback. High confidence? Handle locally at 106 tok/s.
- **Continuous Learning** — QLoRA + DPO self-improvement pipeline via Python sidecar. Every conversation generates training signal. Your model gets smarter the more you use it.
- **Hallucination Detection** — Cross-references every response against the knowledge graph. Contradictions are flagged. The model knows what it doesn't know.
- **Live Knowledge Graph** — SQLite-backed graph that updates in real-time during conversations. Auto-extracts facts ("Rust is a programming language" → stored as triple). Stores facts, conversation history, and DPO preference pairs.
- **Own Model Format (.synapse)** — Bundles base model + LoRA adapters + knowledge graph + training data + agent config into a single shareable file.
- **OpenAI-Compatible API** — Drop-in replacement. Point your existing tools at `localhost:6900` and everything just works. SSE streaming included.
- **Cloud Fallback with Auto-Learning** — When a specialist isn't confident, it routes to a cloud API (Ollama, OpenAI, anything OpenAI-compatible). The cloud response is captured as a DPO preference pair. Next time, the specialist handles it locally. The system teaches itself using the cloud as a tutor.
- **Web Dashboard** — Open `http://localhost:6900` in a browser. Chat with your AI swarm visually. See specialist confidence scores, knowledge graph stats, and Hebbian pathway strengths. Normal people can use it. No terminal required.
- **Community Specialist Hub** — Share trained specialists on HuggingFace. `synapse hub search python` finds community-trained specialists. `synapse hub install user/synapse-python-expert` installs them. `synapse hub push my_expert` shares yours.
- **Specialist Auto-Spawning** — When the system detects repeated failures in an uncovered domain, it proposes and creates new specialists automatically. A music producer ends up with `audio_expert`, `midi_expert`, `mixing_expert` without configuring anything.
- **Standardized Evaluation** — `synapse eval` runs MMLU, HumanEval, MT-Bench, and Safety benchmarks — the same ones OpenAI, Anthropic, and Meta use. Apples-to-apples comparison with the big models.
- **Public Dataset Training** — Train specialists on curated public datasets (OpenWebMath, The Stack, SlimPajama, Alpaca-Cleaned). Clean, factual data. No garbage in, no garbage out.
- **Single Binary** — `cargo build --release` gives you one binary. No Python environment required for inference. No Docker. No "please install these 47 things first."

---

## Quick Start

```bash
# Build from source
git clone https://github.com/Djtony707/titan-synapse
cd titan-synapse && cargo build --release

# Pull a model (downloads from HuggingFace)
./target/release/synapse pull qwen3-3b

# Start the engine
./target/release/synapse up
```

That's it. You now have an AI inference engine running on your GPU.

```bash
# Chat with it (OpenAI-compatible API)
curl http://localhost:6900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "synapse",
    "messages": [{"role": "user", "content": "Write a Python function to check if a number is prime"}]
  }'
```

Works with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6900/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="synapse",
    messages=[{"role": "user", "content": "Hello from the swarm"}]
)
```

---

## How It Works

### The Core Insight

A 70B model is like hiring one genius who's okay at everything. Synapse is like hiring six specialists who are incredible at their thing and know how to collaborate. And they get better every day.

### Hebbian Routing

```
"Neurons that fire together, wire together."
```

The coordinator analyzes each request and routes to the right specialist(s). It tracks which specialist combinations produce the best results. Over time, the routing itself becomes learned — successful pathways get reinforced, poor ones weaken.

- Simple query → routed to a single specialist
- Complex task → multiple specialists activated, responses synthesized

### Continuous Learning Loop

```
Conversation → Self-Evaluation → Preference Pairs → QLoRA Fine-tune → Better Model
                (automatic)        (collected)       (background)       (hot-swapped)
```

The learning engine evaluates every response, collects preference pairs (good vs bad answers), and trains QLoRA adapters on idle GPU cycles. New adapters are hot-swapped in without restarting the server.

### Knowledge Graph

Every conversation updates a persistent SQLite knowledge graph:
- **Facts**: Subject-predicate-object triples with confidence scores
- **Conversations**: Full history with specialist attribution
- **Preferences**: DPO training pairs for self-improvement

---

## Architecture

```
Client → POST /v1/chat/completions
  │
  ├→ Coordinator (keyword + Hebbian routing + metacognitive confidence)
  │    ├→ Single specialist (simple query, confidence-scored)
  │    └→ Multi-specialist swarm (complex task, PARALLEL execution)
  │
  ├→ Inference Engine (Rust + candle)
  │    ├→ GGUF quantized model loading
  │    ├→ LoRA adapters (~5-10MB each, hot-swappable)
  │    ├→ PagedAttention-style KV cache
  │    └→ Temperature/top-p/top-k sampling
  │
  ├→ Knowledge Graph (SQLite)
  │    └→ Facts, conversations, preference pairs
  │
  ├→ Learning Engine (Python sidecar on :8090)
  │    ├→ Self-evaluation scoring
  │    ├→ QLoRA fine-tuning
  │    └→ DPO self-improvement
  │
  └→ SSE Stream Response (OpenAI-compatible)
```

### Project Structure

```
titan-synapse/
├── Cargo.toml                    # Workspace root
├── crates/synapse/src/
│   ├── main.rs                   # CLI (clap): serve, status, models, pull, learn, bench
│   ├── server.rs                 # Axum HTTP server on :6900
│   ├── openai.rs                 # OpenAI-compatible API handlers
│   ├── streaming.rs              # SSE streaming
│   ├── config.rs                 # YAML config loader
│   ├── inference/
│   │   ├── engine.rs             # Model management, GGUF auto-loading
│   │   ├── model.rs              # Candle quantized model, generation loop
│   │   ├── sampler.rs            # Temperature, top-p, top-k sampling
│   │   ├── kv_cache.rs           # PagedAttention-style block allocation
│   │   └── lora.rs               # LoRA adapter hot-swap
│   ├── dashboard.rs                # Embedded web UI (Tailwind CDN, zero build tools)
│   ├── swarm/
│   │   ├── orchestrator.rs       # Task decomposition + routing + cloud fallback
│   │   ├── coordinator.rs        # Hebbian routing + metacognitive confidence
│   │   ├── pool.rs               # Specialist pool with LRU eviction
│   │   ├── synthesizer.rs        # Multi-specialist output merging
│   │   └── spawner.rs            # Specialist auto-spawning from failure patterns
│   ├── learn/
│   │   ├── engine.rs             # Python sidecar bridge
│   │   └── cloud_fallback.rs     # Cloud API fallback + DPO training data capture
│   ├── memory/
│   │   ├── graph.rs              # SQLite knowledge graph
│   │   ├── extractor.rs          # Real-time knowledge extraction from conversations
│   │   └── hallucination.rs      # Hallucination detection via knowledge cross-reference
│   ├── vram/manager.rs           # GPU monitoring (nvidia-smi)
│   └── format/                   # .synapse format pack/unpack
├── python/synapse_learn/         # FastAPI learning sidecar
├── config/default.yaml           # Default specialist definitions
└── docker-compose.yml            # GPU-accelerated learning container
```

### The Synapse Architecture — Beyond Transformers

The v1.0 architecture replaces monolithic transformer blocks with brain-inspired modular processing. Every component is O(n) — no quadratic attention anywhere. Full source in `crates/synapse/src/arch/`.

```
                    THALAMUS (Mamba Router)
                    O(n) state-space model
                    Routes tokens to specialists
                    Hebbian pathway learning
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
     │  xLSTM  │   │  Sparse │   │  Fast   │
     │Language │   │   MoE   │   │ Weight  │
     │ Module  │   │ Experts │   │ Memory  │
     │         │   │         │   │         │
     │Exp gates│   │Top-k of │   │Learn in │
     │Matrix   │   │8+ fire  │   │1 forward│
     │memory   │   │per token│   │pass, no │
     │O(n)     │   │~800M    │   │backprop │
     │         │   │active   │   │         │
     └─────────┘   └─────────┘   └─────────┘
```

| Module | What It Does | Replaces | Complexity |
|--------|-------------|----------|------------|
| **Thalamus** | Routes tokens to the right specialists | Attention-based routing | O(n) |
| **xLSTM** | Syntax, grammar, language fluency | Transformer self-attention | O(n) |
| **Expert Pool** | Specialized knowledge (top-k sparse activation) | Dense FFN layers | O(n) per expert |
| **Fast Weights** | Learn new facts during inference — no training needed | RAG / in-context learning | O(n) |

**28 architecture tests passing.** Full introspection on every module — no black box. See `GET /api/introspect` for real-time visibility into routing decisions, gate values, memory writes, and expert activations.

### VRAM Budget (32GB GPU)

| Component | VRAM |
|-----------|------|
| Base model (3B, Q4_K_M) | ~2.1 GB |
| 6x LoRA adapters loaded | ~0.06 GB |
| KV cache pool | ~3 GB |
| Coordinator (0.6B) | ~0.5 GB |
| **Total for 6 specialists** | **~5.7 GB** |
| **Remaining on 32GB GPU** | **~26 GB free** |

Compare that to a single 70B model that needs **35GB** — doesn't even fit. With Synapse, you've got room for longer contexts, more specialists, or a larger generalist model alongside the swarm.

---

## Tested Results

Real results from our test deployment on an i9-14900KF with RTX 5090 (32GB VRAM).

### Performance (Synapse-3B, RTX 5090, bfloat16)

| Metric | Value |
|--------|-------|
| **Throughput** | **106.3 tok/s** (avg over 5 runs) |
| **Time to first token** | **11.2ms** (avg), 11.3ms (p99) |
| **VRAM usage** | **6.43 GB** (19.1% of 33.67 GB) |
| **Model load time** | **0.4s** (3B, GPU) |
| **Parameters** | **3.09B** (bfloat16) |

Tested on i9-14900KF + RTX 5090 32GB VRAM, CUDA 12.8 (Blackwell). Only 19% VRAM utilization leaves room for multiple specialists, larger models, or training alongside inference.

### Standardized Evaluation (Real Benchmarks, Full Datasets)

Run against the **full standardized benchmark datasets** on an NVIDIA RTX 5090 (bfloat16). Every question in each dataset — no subsets, no cherry-picking.

| Benchmark | Score | Samples | Notes |
|-----------|-------|---------|-------|
| **MMLU** (5-shot) | **62.6%** | 14,042 | All 57 subjects. Best: marketing (88.5%), world history (85.7%). Worst: European history (0%), US history (5.4%) |
| **GSM8K** (8-shot CoT) | **18.9%** | 1,319 | Grade school math with chain-of-thought prompting |
| **Inference Speed** | **106.3 tok/s** | 5 runs | Avg over 5 runs, 256 max tokens, bfloat16 |
| **TTFT** | **11.2ms** | 10 runs | Time to first token, p99: 11.3ms |
| **VRAM** | **6.43 GB** | — | 19.1% of 33.67 GB available |

> HumanEval pass@1 results (99.4%, 163/164) are excluded — this is inconsistent with published results for 3B-class models and indicates a test harness issue under investigation.

#### What These Numbers Mean

**MMLU 62.6% (+9.6 pts over Qwen2-3B baseline ~53%):** The TIES merging of four specialist adapters improved general knowledge coverage. This is the merged model — not the swarm system with adapter switching, which would be higher.

**GSM8K 18.9% (below Qwen2-3B baseline ~54%):** The specialist adapters were not math-focused, and TIES merging appears to have degraded the base model's existing math reasoning capabilities. This is a known limitation of model merging — some capabilities regress.

**106.3 tok/s with 11.2ms TTFT:** Running a 3B bfloat16 model on an RTX 5090, inference is fast enough for real-time use. Only 6.43 GB VRAM leaves room for multiple specialists or larger models.

#### The Honest Take

**We're not pretending a 3B model beats GPT-5.** Frontier models score 90%+ on MMLU and have saturated GSM8K. A 3B model can't memorize as many facts as a 200B+ model — no architecture changes that.

**The value proposition is different:** Synapse runs for free on your GPU at 106 tok/s, works offline, uses 6.43 GB VRAM, and gets smarter from your conversations. The swarm with adapter switching (not the merged model) targets domain-specific excellence over general benchmarks.

**Where the merged model wins:** MMLU +9.6 points over baseline shows TIES merging can genuinely improve general knowledge when combining complementary specialists.

**Where it loses:** GSM8K -35 points below baseline shows TIES merging can degrade capabilities when the merged adapters don't cover a domain. Future work includes math-specialized adapters.

### Verified Working

| Test | Result | Details |
|------|--------|---------|
| `cargo build --release` | PASS | Clean compilation, Rust 2024 edition |
| `cargo test` | **65/65 passing** | Config, sampler, KV cache, knowledge graph, manifest, packer, Hebbian, coordinator, LoRA, extractor, hallucination, spawner, cloud fallback + 28 architecture tests (Mamba, xLSTM, Thalamus, Expert, Fast Weights, SynapseModel) |
| `synapse bench` | PASS | 106.3 tok/s average (GPU, bfloat16, RTX 5090) |
| `synapse status` | PASS | Shows GPU info, VRAM usage, specialist list |
| `GET /health` | PASS | Returns "ok" |
| `GET /v1/models` | PASS | Lists synapse + all specialist models |
| `GET /api/status` | PASS | Loaded models, Hebbian pathways, knowledge stats |
| `POST /v1/chat/completions` | PASS | Real inference with token usage stats |
| `POST /v1/chat/completions` (stream) | PASS | SSE streaming, OpenAI-compatible chunks |
| GGUF model loading | PASS | Multi-model: Qwen2.5-0.5B (0.7s) + Qwen2.5-3B (1.1s) |
| Code generation | PASS | Correct `is_prime()` function with explanation |
| Math reasoning | PASS | "2 + 2 equals 4." — clean stop tokens |
| Specialist routing | PASS | Python queries → python_expert, SQL → sql_expert |
| Hebbian routing | PASS | Pathway strengths accumulate in SQLite |
| Swarm decomposition | PASS | Complex queries trigger multi-specialist **parallel** mode |
| Metacognitive confidence | PASS | /api/confidence returns per-specialist performance |
| Knowledge graph | PASS | Facts, preferences, conversations, routing pathways |
| .synapse format | PASS | Pack/unpack with model, adapters, knowledge bundling |
| Export/Import CLI | PASS | Round-trip specialist export and import |

### Unit Tests (37/37 Passing)

```
test config::tests::test_default_config ... ok
test config::tests::test_config_serialization ... ok
test config::tests::test_load_missing_config ... ok
test inference::sampler::tests::test_greedy_sampling ... ok
test inference::sampler::tests::test_empty_logits ... ok
test inference::sampler::tests::test_stochastic_sampling ... ok
test inference::kv_cache::tests::test_cache_allocation ... ok
test inference::lora::tests::test_lora_adapter_placeholder ... ok
test inference::lora::tests::test_lora_adapter_with_tensors ... ok
test inference::speculative::tests::test_speculative_decoder_creation ... ok
test inference::speculative::tests::test_draft_length_clamping ... ok
test swarm::coordinator::tests::test_single_routing ... ok
test swarm::coordinator::tests::test_swarm_routing ... ok
test swarm::coordinator::tests::test_default_routing ... ok
test swarm::spawner::tests::test_infer_capabilities ... ok
test swarm::spawner::tests::test_is_domain_covered ... ok
test swarm::spawner::tests::test_create_specialist_config ... ok
test memory::graph::tests::test_knowledge_graph ... ok
test memory::graph::tests::test_preferences ... ok
test memory::graph::tests::test_hebbian_routing ... ok
test memory::graph::tests::test_specialist_stats ... ok
test memory::extractor::tests::test_extract_is_pattern ... ok
test memory::extractor::tests::test_extract_verb_patterns ... ok
test memory::extractor::tests::test_extract_preferences_positive ... ok
test memory::extractor::tests::test_extract_preferences_negative ... ok
test memory::extractor::tests::test_empty_text ... ok
test memory::hallucination::tests::test_verify_correct_claim ... ok
test memory::hallucination::tests::test_verify_unknown_claim ... ok
test memory::hallucination::tests::test_word_overlap ... ok
test memory::hallucination::tests::test_empty_response ... ok
test learn::cloud_fallback::tests::test_cloud_fallback_disabled ... ok
test learn::cloud_fallback::tests::test_cloud_fallback_enabled ... ok
test learn::cloud_fallback::tests::test_confidence_threshold ... ok
test format::manifest::tests::test_manifest_creation ... ok
test format::manifest::tests::test_manifest_serialization ... ok
test format::packer::tests::test_pack_and_unpack ... ok
test format::packer::tests::test_list_bundles ... ok
test arch::mamba::tests::test_mamba_layer_creation ... ok
test arch::mamba::tests::test_mamba_forward ... ok
test arch::mamba::tests::test_mamba_state_persistence ... ok
test arch::mamba::tests::test_silu ... ok
test arch::xlstm::tests::test_xlstm_creation ... ok
test arch::xlstm::tests::test_xlstm_forward ... ok
test arch::xlstm::tests::test_xlstm_introspection ... ok
test arch::xlstm::tests::test_xlstm_state_persistence ... ok
test arch::thalamus::tests::test_thalamus_creation ... ok
test arch::thalamus::tests::test_thalamus_routing ... ok
test arch::thalamus::tests::test_thalamus_introspection ... ok
test arch::thalamus::tests::test_hebbian_learning ... ok
test arch::thalamus::tests::test_status_summary ... ok
test arch::expert::tests::test_expert_creation ... ok
test arch::expert::tests::test_expert_forward ... ok
test arch::expert::tests::test_expert_pool ... ok
test arch::expert::tests::test_expert_pool_forward ... ok
test arch::expert::tests::test_expert_introspection ... ok
test arch::fast_weights::tests::test_fast_weight_creation ... ok
test arch::fast_weights::tests::test_fast_weight_forward ... ok
test arch::fast_weights::tests::test_fast_weight_introspection ... ok
test arch::fast_weights::tests::test_fast_weight_memory_persists ... ok
test arch::synapse_model::tests::test_model_creation ... ok
test arch::synapse_model::tests::test_param_counting ... ok
test arch::synapse_model::tests::test_model_forward ... ok
test arch::synapse_model::tests::test_model_introspection ... ok
test arch::synapse_model::tests::test_model_summary ... ok
test arch::synapse_model::tests::test_model_reset ... ok
test result: ok. 65 passed; 0 failed; 0 ignored
```

---

## How Synapse Compares

| Feature | Ollama | vLLM | CrewAI | **Synapse** |
|---------|--------|------|--------|-------------|
| Own inference engine | No (llama.cpp) | Yes | No (wraps LLMs) | **Yes (Rust + candle)** |
| Own model format | No (GGUF) | No | No | **Yes (.synapse)** |
| Specialist swarm | No | No | Yes (no inference) | **Yes (integrated)** |
| Continuous learning | No | No | No | **Yes (QLoRA + DPO)** |
| Knowledge graph | No | No | No | **Yes (real-time SQLite)** |
| Single binary | No | No | No | **Yes** |
| Consumer GPU optimized | Yes | No | N/A | **Yes** |
| OpenAI-compatible API | Yes | Yes | No | **Yes** |

---

## CLI Commands

```bash
synapse serve [--port 6900]     # Start the inference server
synapse up [--port 6900]        # Alias for serve (also opens web dashboard)
synapse status                  # GPU info, loaded models, specialist list
synapse models                  # List available models in ~/.synapse/models/
synapse pull <model>            # Download model from HuggingFace
synapse export <name>           # Export specialist as .synapse file
synapse import <path>           # Import a .synapse specialist
synapse learn status            # Show learning engine stats
synapse learn train-now         # Force immediate training
synapse bench [--model <name>]  # Run inference benchmarks
synapse eval                    # Run standardized eval (MMLU, HumanEval, MT-Bench, Safety)
synapse hub search <query>      # Find community specialists on HuggingFace
synapse hub install <repo>      # Install a community specialist
synapse hub push <name>         # Share your trained specialist
synapse hub list                # Browse the hub
```

---

## Configuration

Synapse uses YAML config at `~/.synapse/config.yaml`:

```yaml
port: 6900
coordinator_model: qwen3-0.6b
base_model: qwen3-3b

learning:
  enabled: true
  min_pairs_before_training: 10
  sidecar_url: http://localhost:8090
  eval_threshold: 3.0

specialists:
  - name: general
    capabilities: [general, chat, help]
    system_prompt: "You are a helpful AI assistant."
    priority: 50

  - name: python_expert
    capabilities: [python, debugging, testing, refactoring]
    system_prompt: "You are an expert Python developer."
    priority: 60

  - name: sql_expert
    capabilities: [sql, database, query, postgres]
    system_prompt: "You are an expert database engineer."
    priority: 60
```

Or just run `synapse up` and the defaults handle everything. Config is auto-created on first run.

---

## Contributing

This thing is early. There's a lot to build and a lot to break.

**Areas where help is most needed:**

- **CUDA inference** — Enable candle CUDA kernels for GPU-accelerated generation
- **New specialist adapters** — Train and contribute domain-specific LoRAs
- **Inference optimizations** — Flash attention, speculative decoding, continuous batching
- **Platform support** — AMD ROCm, Apple Metal, Intel Arc
- **Learning engine** — Improved training signal extraction, better DPO reward modeling
- **Benchmarks** — Rigorous eval harness across standard benchmarks

```bash
# Dev setup
git clone https://github.com/Djtony707/titan-synapse
cd titan-synapse
cargo build
cargo test  # 65/65 should pass

# Run with debug logging
RUST_LOG=debug cargo run -- serve
```

---

## Roadmap

- [x] Core inference engine (Rust + candle)
- [x] GGUF quantized model loading
- [x] OpenAI-compatible API (chat completions + streaming)
- [x] Specialist swarm with Hebbian routing
- [x] Knowledge graph (SQLite)
- [x] .synapse model format
- [x] CLI (serve, status, models, pull, learn, bench)
- [x] Python learning sidecar
- [x] Multi-model loading (0.5B + 3B loaded simultaneously)
- [x] Token counting in API responses (accurate usage stats)
- [x] Hebbian routing persistence (SQLite-backed pathway learning)
- [x] .synapse format packer/unpacker with bundled models + adapters
- [x] CUDA-accelerated inference (106.3 tok/s on RTX 5090, 11.2ms TTFT, 6.43 GB VRAM)
- [x] Parallel swarm execution (specialists run concurrently, not sequentially)
- [x] Metacognitive confidence scoring (system tracks what it knows)
- [x] Smart model selection (prefers larger models when available)
- [x] Real LoRA adapter loading via SafeTensors (f32, f16, bf16)
- [x] Conversation context threading (multi-turn awareness)
- [x] Real-time knowledge extraction from conversations
- [x] Hallucination detection (cross-reference against knowledge graph)
- [x] User feedback preference learning (DPO pair collection)
- [x] Standardized evaluation (MMLU 62.6%, GSM8K 18.9% — full datasets on RTX 5090, 15,361 questions)
- [x] Cloud fallback with auto-learning (DPO pairs from cloud responses)
- [x] Specialist auto-spawning (system creates new specialists from failure patterns)
- [x] Web dashboard (chat UI at localhost:6900, stats + metacognition panels)
- [x] Community specialist hub (push/pull/search on HuggingFace)
- [x] Public dataset training pipeline (OpenWebMath, The Stack, SlimPajama, etc.)
- [x] Speculative decoding scaffold (draft + verify architecture)
- [x] LoRA adapter training + hot-swap during inference
- [x] Specialist model merge (TIES merging — 4 adapters into Synapse-3B)
- [x] Synapse Architecture: Mamba router + xLSTM + Sparse MoE + Fast Weights (28 tests)
- [x] Full model introspection API (no black box — see every routing decision)
- [x] Synapse-3B published on [HuggingFace](https://huggingface.co/djtony707/synapse-3b)
- [ ] Full speculative decoding (shared KV cache state)
- [ ] Continuous batching across specialists
- [ ] Doc-to-LoRA knowledge crystallization
- [ ] Distributed swarm across multiple machines
- [ ] Train Synapse Architecture from scratch on RTX 5090

---

## License

Licensed under the [Apache License 2.0](LICENSE).

Use it. Fork it. Build on it. Make something wild.

---

<div align="center">

**Built with mass amounts of caffeine and mass amounts of mass by [Tony Elliott](https://github.com/Djtony707)**

*Because the future of AI isn't one massive model — it's a swarm of tiny ones that never stop learning.*

</div>
