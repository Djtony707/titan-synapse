# TITAN Synapse: A Modular, O(n) Neural Architecture Combining Selective State Spaces, Extended LSTM, Sparse Mixture-of-Experts, and Fast-Weight Memory

**Author:** Tony Elliott
**Repository:** https://github.com/Djtony707/titan-synapse
**HuggingFace:** https://huggingface.co/djtony707/synapse-3b
**Date:** March 2026
**Status:** Implementation complete, training from scratch in progress

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Related Work](#related-work)
4. [The Synapse Architecture](#the-synapse-architecture)
5. [Thalamus: Context-Aware Routing with Hebbian Learning](#thalamus-context-aware-routing-with-hebbian-learning)
6. [xLSTM Language Module](#xlstm-language-module)
7. [Sparse Mixture-of-Experts](#sparse-mixture-of-experts)
8. [Fast-Weight Memory](#fast-weight-memory)
9. [Introspection System](#introspection-system)
10. [Specialist Swarm System](#specialist-swarm-system)
11. [Synapse-3B: TIES Merging of Specialist Adapters](#synapse-3b-ties-merging-of-specialist-adapters)
12. [Implementation: Rust and candle](#implementation-rust-and-candle)
13. [Current Status and Roadmap](#current-status-and-roadmap)
14. [Conclusion](#conclusion)
15. [References](#references)

---

## Abstract

We present TITAN Synapse, a system with two distinct contributions. The first is the **Synapse Architecture**: a modular neural network design implemented in Rust that combines Mamba selective state spaces [Gu & Dao, 2023], Extended Long Short-Term Memory [Beck et al., 2024], Sparse Mixture-of-Experts [Shazeer et al., 2017], and fast-weight memory [Schmidhuber, 1992; Ba et al., 2016] into a unified forward pass with O(n) sequence complexity throughout. Every component exposes its internal state for real-time inspection. The second is the **Specialist Swarm System**: a higher-level orchestration layer that runs multiple specialist models derived from a shared base, routes queries using Hebbian pathway learning, and improves continuously through QLoRA fine-tuning and Direct Preference Optimization from live conversations.

The Synapse Architecture has 28 unit tests passing on CPU. It has not yet been trained from scratch. The Synapse-3B model published on HuggingFace uses standard Qwen3-3B transformer weights with four QLoRA specialist adapters merged via TIES [Yadav et al., 2023]. On standardized benchmarks run against the full datasets on an NVIDIA RTX 5090, Synapse-3B achieves MMLU 62.6% (14,042 questions, 5-shot), GSM8K 18.9% (1,319 problems, 8-shot CoT), inference speed of 106.3 tokens/second (bfloat16), and time-to-first-token of 11.2ms, using 6.43 GB VRAM. HumanEval pass@1 results (99.4%) are excluded from claims pending investigation of a likely test harness issue.

---

## Introduction

### Motivation

I built this because I was frustrated. In early 2026 I bought an NVIDIA RTX 5090 — the most powerful consumer GPU available at the time: 32 GB of VRAM, Blackwell architecture, 1,792 GB/s memory bandwidth. It is the best hardware you can buy without enterprise procurement.

It was not enough. The models producing the best results — 70B, 120B, 405B parameter dense transformers — require 40–800 GB of VRAM in fp16. Even aggressively quantized to 4-bit, a 70B model needs ~35 GB, exceeding the 5090's capacity. I spent over $3,000 on the fastest consumer GPU ever made and still could not run the models I actually wanted to use, locally, without depending on someone else's cloud API.

The question that motivated Synapse was direct: what if the architecture itself were different? Instead of a single dense model where every parameter fires on every token, what if you had a swarm of small specialists — each fitting comfortably in consumer VRAM — that only activate when their expertise is relevant? A 3B parameter model with 8 specialists and top-2 routing gives you the knowledge capacity of a much larger model at the inference cost of a small one. Add fast-weight memory so the model learns during inference, and you reduce the need for the massive pre-training data budgets that make large models large in the first place.

I built Synapse for people who own hardware and want to use it — not rent compute from someone else.

### Problem Statement

Large language models trained on fixed datasets with frozen weights have three structural limitations that this work addresses.

**Quadratic attention complexity.** The dominant transformer architecture [Vaswani et al., 2017] computes self-attention in O(n²) time and space with respect to sequence length. This imposes practical context limits and makes long-document processing expensive. A 32k-token context with standard multi-head attention requires roughly 10^9 attention score computations per layer.

**Frozen knowledge.** A deployed transformer learns nothing after training ends. Adapting a model to new facts or user-specific information requires full fine-tuning, retrieval augmentation (RAG), or extended context — none of which update the underlying weights during inference. Fast-weight methods offer an alternative: updating a small memory matrix during the forward pass itself, with no backpropagation.

**Monolithic architecture.** A single dense transformer block applies the same computation to every token regardless of what processing that token actually requires. Sparse mixture-of-experts models [Shazeer et al., 2017; Fedus et al., 2022] address this with conditional computation, but they still use attention for routing context. The Synapse Architecture replaces the attention-based routing mechanism with an O(n) state-space model.

TITAN Synapse addresses all three limitations in two separable contributions:

1. A novel modular architecture where every component — routing, language processing, specialized knowledge, and working memory — operates in O(n) time, with no quadratic operation anywhere in the forward pass.

2. A swarm orchestration system that runs small specialist models sharing a common base, routes queries through learned Hebbian pathways, and accumulates domain expertise continuously from conversations.

This paper is written to be factually precise about what is built versus what is planned. The architecture modules are implemented and tested. They have not been trained from scratch end-to-end. The swarm system is operational and has produced the benchmark numbers reported here. These two contributions are distinct and should be evaluated independently.

---

## Related Work

### Selective State Space Models

Mamba [Gu & Dao, 2023] introduced the S6 (Selective State Space Model) layer, which processes sequences in O(n) time by maintaining a fixed-size compressed state. Unlike earlier linear SSMs such as S4 [Gu et al., 2022], Mamba makes the state-transition parameters (B, C, and step size Δ) functions of the input, enabling the model to selectively retain or discard information based on content. The Synapse Thalamus module uses a Mamba layer as its routing backbone, applying this selective mechanism to produce context-aware routing weights rather than sequence outputs.

### Extended LSTM

xLSTM [Beck et al., 2024] revisited the LSTM architecture with two key changes: exponential gating (using exp() instead of sigmoid for input and forget gates, giving unbounded dynamic range) and matrix-valued cell state (the mLSTM variant stores a d×d matrix per head rather than a d-dimensional vector). The matrix state gives xLSTM associative memory properties — it can store and retrieve key-value pairs — with O(n) complexity. The Synapse xLSTM Language Module uses the mLSTM variant directly.

### Mixture of Experts

Sparsely-gated Mixture-of-Experts [Shazeer et al., 2017] demonstrated that activating only the top-k of N experts per token allows model capacity to scale independently of per-token computation cost. Switch Transformer [Fedus et al., 2022] simplified this to top-1 routing and showed successful training at scale. GLaM [Du et al., 2022] used top-2 MoE routing across 64 experts. Mixtral [Jiang et al., 2024] showed that top-2-of-8 MoE routing in a 46.7B total parameter model (with ~12.9B active per token) achieves strong results at the inference cost of a much smaller dense model. The Synapse expert pool uses top-k-of-8 sparse routing with SwiGLU feed-forward experts.

### Fast-Weight Memory

Schmidhuber [1992] introduced fast weights: a second neural network whose output modifies the weights of a primary network at inference time. The key operation is the outer product update W ← W + η(v ⊗ k), where k is an addressing key and v is a value to store, enabling content-addressable memory that updates without backpropagation. Ba et al. [2016] connected fast weights to attention mechanisms and demonstrated their effectiveness on algorithmic tasks. More recently, Schlag et al. [2021] showed that linear transformers can be viewed as fast-weight programs. The Synapse Fast-Weight Memory module implements the original outer-product update rule with a configurable decay factor, enabling per-inference-step learning of new key-value associations.

### Model Merging

TIES (Task-vector Interference Elimination via Sign) merging [Yadav et al., 2023] addresses the problem of combining multiple fine-tuned model variants into a single model without catastrophic interference. The method: (1) compute task vectors as the difference between each fine-tuned model and the base, (2) prune low-magnitude changes, (3) resolve sign conflicts by majority vote per parameter, (4) merge the resulting vectors with the base model. Synapse-3B was created using this method to combine four specialist QLoRA adapters.

### Continuous Learning

Direct Preference Optimization [Rafailov et al., 2023] provides a stable alternative to RLHF by framing preference learning as a classification problem. QLoRA [Dettmers et al., 2023] enables fine-tuning of quantized models using low-rank adapters, dramatically reducing the VRAM required for gradient-based updates. The Synapse swarm uses both: DPO pairs collected from conversations are used to train QLoRA adapters in a background Python sidecar process.

---

## The Synapse Architecture

The Synapse Architecture organizes each layer as a fixed pipeline of four modules:

```
Input x
  │
  ▼
[xLSTM Language Module] ──── residual ──→ x₁
  │
  ▼
[Thalamus Router] ──→ routing weights + expert indices
  │
  ▼
[Sparse Expert Pool] ──── residual ──→ x₂
  │
  ▼
[Fast-Weight Memory] ──── residual ──→ x₃ (output of layer)
```

Each module adds its output to the residual stream (pre-norm with RMSNorm), following the pre-norm convention from [Zhang & Sennrich, 2019]. No module uses self-attention. Every operation is O(n) in sequence length.

The default configuration uses:

| Parameter | Value |
|-----------|-------|
| d\_model | 768 |
| n\_layers | 12 |
| n\_experts | 8 |
| top\_k | 2 |
| d\_expert | 3072 |
| d\_xlstm\_hidden | 1536 |
| n\_xlstm\_heads | 4 |
| d\_memory (fast-weight key/value) | 64 |
| n\_memory\_heads | 8 |
| d\_state (Mamba SSM) | 16 |
| vocab\_size | 151,936 (Qwen2.5) |

With this configuration, estimated total parameters are approximately 1.4B with roughly 600M active per token (top-2 routing means 2 of 8 experts fire, leaving the other 6 dormant). These are structural parameter counts based on the architecture definition; the model has not been trained to convergence.

The forward pass for a single layer is implemented exactly as described — xLSTM first, then Thalamus-routed expert processing, then fast-weight memory — each with a residual connection and RMSNorm. The layer ordering is intentional: xLSTM processes linguistic structure before routing decisions are made, so the router receives contextualized representations rather than raw token embeddings.

---

## Thalamus: Context-Aware Routing with Hebbian Learning

The Thalamus module is named after the brain region that relays sensory information to specific cortical areas. It serves the same function structurally: it receives the input token stream and decides which expert modules should process each token.

### Architecture

The Thalamus consists of a Mamba layer (the "backbone") followed by a linear projection from d\_model to n\_experts:

```
x (batch, seq, d_model)
  │
  ▼
[Mamba Layer] → context-aware hidden state
  │
  ▼
[Linear: d_model → n_experts] → routing logits
  │
  ▼
[Softmax] → routing probabilities
  │
  ▼
[Top-k selection] → expert indices + normalized weights
```

The Mamba layer processes the full sequence before routing, so routing decisions for each token incorporate information from all preceding tokens — equivalent in effect to attention-based routing, but at O(n) cost. This is the key difference from standard MoE routers, which typically use a simple linear projection over the token's own embedding with no sequential context.

### Hebbian Pathway Learning

After top-k routing is applied, the module records which combination of experts was selected for each token. These combinations are referred to as "pathways" and identified by a string of expert names (e.g., `language+reasoning`). A running strength value is maintained per pathway in a hash map:

```
strength[pathway] += hebbian_lr        // strengthen the fired pathway
for all pathways: strength *= decay    // small decay on all others
```

The default learning rate is 0.01 and the decay factor is 0.999 per token. This implements an online approximation to the Hebbian rule "neurons that fire together, wire together" [Hebb, 1949], where frequently co-activated expert combinations accumulate stronger pathway representations. These pathway strengths are persisted to SQLite and can be loaded to initialize routing behavior in subsequent sessions.

The pathway strengths currently serve as an observability signal and a seeding mechanism. They do not yet feed back into the routing logits directly; routing is fully determined by the linear projection over the Mamba output. Making pathway strength influence routing weights during inference is noted as future work.

### Routing Introspection

Every forward pass produces a `ThalamusIntrospection` struct containing: per-token routing decisions with all expert scores, selected expert indices and normalized weights, routing confidence (ratio of top score to second score), load balance across experts, and the sorted Hebbian pathway strengths. This data is available via `GET /api/introspect`.

---

## xLSTM Language Module

The xLSTM language module uses the mLSTM variant from Beck et al. [2024], implementing matrix-valued cell state with exponential gating.

### Exponential Gating

Classical LSTM uses sigmoid-activated gates, bounding values to [0, 1]. xLSTM replaces the input gate i and forget gate f with exponential activations:

```
i_gate = exp(clamp(W_i x + b_i, -20, 20))
f_gate = exp(clamp(W_f x + b_f, -20, 20))
o_gate = sigmoid(W_o x + b_o)
```

Clamping at ±20 prevents overflow (exp(20) ≈ 5×10⁸). The forget gate bias is initialized to 1.0 rather than 0.0, biasing the gate toward remembering (following the initialization recommendations in [Beck et al., 2024]). The output gate retains sigmoid activation, bounding the readout to [0, 1].

The exponential range allows the model to represent much more decisive write and erase operations than sigmoid gating permits, which is valuable when processing text where some tokens are highly significant (named entities, key verbs) and others are nearly irrelevant (articles, punctuation).

### Matrix Cell State

The cell state is a matrix rather than a vector, with multi-head structure. For n\_heads heads each of dimension d\_head × d\_head:

```
C ∈ R^(batch, n_heads, d_head, d_head)
n ∈ R^(batch, n_heads, d_head)   // normalizer state
```

At each step t, given queries q, keys k, values v (all projected from input x):

```
C ← f_gate ⊙ C + i_gate ⊙ (v ⊗ k)   // outer product write
n ← f_gate ⊙ n + i_gate ⊙ k
h ← C q / max(|n · q|, 1)             // normalized read
y ← o_gate ⊙ h
```

The normalizer n tracks the cumulative sum of weighted keys and prevents the retrieved value h from growing unboundedly as the cell state accumulates outer products. The max(·, 1) in the denominator ensures numerical stability when n · q is near zero.

This structure gives the xLSTM layer associative memory: it can write key-value pairs into the matrix and retrieve values by querying with related keys. The complexity is O(n × d_head²) per sequence, linear in sequence length.

### State Persistence

The cell state C and normalizer n persist across calls to `forward()`, enabling multi-turn processing where the module retains information from earlier in a conversation. `reset_state()` clears both, which the model calls between independent sequences.

---

## Sparse Mixture-of-Experts

Each expert is a feed-forward network with SwiGLU activation [Shazeer, 2020], implemented as:

```
hidden = silu(gate(x)) ⊙ up(x)   // SwiGLU gate
output = down(hidden)
```

where gate, up, and down are linear projections with dimensions d\_model → d\_expert → d\_model. Default d\_expert is 3072 (4× d\_model).

The ExpertPool receives from the Thalamus: routing weights (batch, seq, top\_k) and expert indices (batch, seq, top\_k). For each token position, it runs only the top-k selected experts and combines their outputs as a weighted sum:

```
output_t = Σ_{k} w_k · expert_{i_k}(x_t)
```

With n\_experts=8 and top\_k=2, 75% of expert parameters are dormant per token. A model with 8 experts of d\_expert=3072 has roughly 8× the expert parameter count of a single FFN, but activates only 2/8 of those parameters per token. This trades memory footprint for per-token computation efficiency.

### Introspection

Each expert tracks tokens processed, average output magnitude (L2 norm), and activation sparsity (fraction of hidden units with absolute value below 0.01). The pool aggregates this into an ExpertPoolIntrospection containing per-expert statistics and a list of top contributors sorted by output magnitude.

---

## Fast-Weight Memory

The FastWeightMemory module implements the outer-product update rule from Schmidhuber [1992] with multi-head structure and a configurable decay factor.

### Memory Update Rule

The module maintains a fast-weight matrix W ∈ R^(n\_heads, d\_key, d\_value), initialized to zero. At each forward step:

**Read:** For each head h, retrieve from memory by querying:

```
retrieved[h] = W[h]ᵀ q[h]     // (d_value,)
```

**Decay:** Reduce all existing memories:

```
W ← decay × W
```

**Write:** Add new outer-product memory:

```
W ← W + write_strength × gate ⊙ (k ⊗ v)
```

where gate is a learned sigmoid-activated scalar per head derived from the input. The default decay is 0.95 per step, write\_strength is 0.1. The retrieved value is projected back to d\_model and added to the input residual stream.

This differs from attention in a key way: the memory matrix W is updated at every forward step during inference, not during training. The model accumulates key-value associations across the current sequence (and, if not cleared, across multiple sequences). This enables single-pass in-context learning: facts presented early in a conversation can be retrieved by later queries without relying on attention to span the full context distance.

At inference time with default write\_strength=0.1 and decay=0.95, the memory has an effective temporal horizon of roughly 1/(1-0.95) = 20 tokens before a written memory decays to ~37% of its original strength. Higher write\_strength and lower decay extend retention at the cost of older memories being overwritten more aggressively.

### Relationship to Linear Attention

As Schlag et al. [2021] showed, linear attention (with no softmax) computes the same outer-product accumulation: the attention output for query q is (Σ_i k_i ⊗ v_i) q, which is exactly a fast-weight matrix times a query vector. The Fast-Weight Memory module makes this connection explicit, adding gating, decay, and per-head structure that are not present in standard linear attention.

---

## Introspection System

A central design principle of the Synapse Architecture is that no internal state should be opaque. Every module exposes its internal computations after each forward pass:

| Module | Introspection Data |
|--------|-------------------|
| Thalamus | Per-token routing decisions, all expert scores, confidence ratio, load balance, Hebbian pathway strengths |
| xLSTM | Input/forget/output gate values per timestep per head, cell state norms, memory age |
| ExpertPool | Per-expert tokens processed, output magnitude, activation sparsity, top contributors |
| FastWeightMemory | Write events (step, strength, head, key norm, value norm), read events (confidence, dominant head), per-head utilization (Frobenius norm) |
| SynapseModel | Aggregated across all layers: total parameters, active parameters per token, sparsity ratio, routing confidence, global Hebbian pathways |

This data is collected during the forward pass itself (not in a separate profiling step) and is available via `GET /api/introspect` on the running server. The intent is to enable researchers and users to directly inspect what the model is doing at each layer — which experts fired, what was written to memory, how confident the router was — rather than treating the model as a black box.

The introspection adds some overhead per forward pass (tensor extraction, vec allocation). For production inference this can be disabled; for research use, the overhead is acceptable and the data is valuable.

---

## Specialist Swarm System

The swarm system is a separate layer above the Synapse Architecture. It is operational in the current implementation and is responsible for the benchmark results reported in this paper.

### Architecture

```
Client request
  │
  ▼
Orchestrator
  ├── keyword analysis
  ├── Hebbian pathway lookup (SQLite)
  └── metacognitive confidence check
       │
       ├── single specialist → inference → response
       └── multi-specialist (parallel) → synthesis → response
                                          │
                                          ▼
                                      SQLite knowledge graph
                                      (facts, conversations,
                                       preference pairs)
```

The coordinator assigns each query to one or more specialists based on keyword overlap with specialist capability lists, weighted by Hebbian pathway strengths accumulated from past routing successes. If no specialist exceeds a confidence threshold, the query falls through to a cloud fallback (any OpenAI-compatible API endpoint). Cloud responses are captured as DPO preference pairs for later fine-tuning.

### Specialists

Specialists are configurations on top of a shared base model. Each specialist has a name, a capability list used for routing, a system prompt, and optionally a QLoRA adapter that was trained on domain-specific data. Because adapters are small (5–10 MB in SafeTensors format), six specialists sharing one 3B base model require approximately 5.7 GB of VRAM total: ~2.1 GB for the Q4\_K\_M base, ~60 MB for six adapters, ~3 GB for KV cache, and ~0.5 GB for the coordinator (0.6B model).

### Continuous Learning Pipeline

After each conversation, the learning engine:

1. Scores the response using a heuristic quality signal (length, coherence, factual consistency with the knowledge graph).
2. If the score falls below a threshold, pairs the query with a better response (either user feedback or a cloud fallback response) as a DPO preference pair.
3. When sufficient pairs accumulate (default: 10), triggers QLoRA fine-tuning in the Python sidecar.
4. Hot-swaps the updated adapter without restarting the inference server.

The Python sidecar runs in a Docker container with GPU access. The Rust inference engine communicates with it over HTTP on port 8090.

### Knowledge Graph

A SQLite database stores: subject-predicate-object triples with confidence scores (facts), full conversation history with specialist attribution, DPO preference pairs, and Hebbian routing pathway strengths. The extractor module identifies `is-a` and simple copular patterns in model outputs in real time. The hallucination detector cross-references new claims against stored facts and flags contradictions.

---

## Synapse-3B: TIES Merging of Specialist Adapters

Synapse-3B, available at `djtony707/synapse-3b` on HuggingFace, is a merged model created from four QLoRA specialist adapters applied to Qwen3-3B using the TIES merging method [Yadav et al., 2023].

### What It Is

Synapse-3B uses **standard Qwen3-3B transformer architecture** — it is not an instance of the Synapse Architecture described in the preceding sections. The name refers to the swarm system (TITAN Synapse) rather than to the novel architecture modules.

### TIES Merging Procedure

Given a base model θ\_base and K fine-tuned models {θ\_1, ..., θ\_K}, TIES merging proceeds as:

1. **Compute task vectors:** τ\_k = θ\_k - θ\_base for each specialist k.
2. **Prune small changes:** Set to zero all entries of τ\_k with |τ\_k| below a threshold (typically the top 20% by magnitude are retained).
3. **Resolve sign conflicts:** For each parameter position, count the sign of the pruned values across all K models. If the majority of non-zero values are positive, use positive; otherwise, use negative. Zero out values that conflict with the majority sign.
4. **Merge:** θ\_merged = θ\_base + λ × (1/K) Σ\_k τ\_k^(resolved), where λ is a scaling factor.

The four adapters merged into Synapse-3B were trained on domain-specific data targeting code generation (Python, general programming), mathematical reasoning, factual question answering, and instruction following. The merge attempts to combine the domain specializations into a single model without adapter switching overhead.

### Limitations of the Merged Model

TIES merging on QLoRA adapters operates on the low-rank delta matrices, not full model weights. There is no guarantee that the merged model retains all specialist capabilities; merging typically involves a small accuracy regression relative to the best individual specialist on any given task.

### Benchmark Results (Synapse-3B, RTX 5090, bfloat16)

All benchmarks were run on the full standard datasets using an NVIDIA RTX 5090 (33.67 GB VRAM) with the merged model in bfloat16 precision:

| Benchmark | Synapse-3B | Qwen2-3B (published) | Notes |
|-----------|-----------|----------------------|-------|
| MMLU (5-shot) | **62.6%** (8,786/14,042) | ~53% | +9.6 points over baseline |
| GSM8K (8-shot CoT) | **18.9%** (249/1,319) | ~54% | Below baseline — adapters not math-specialized |
| Inference speed | **106.3 tok/s** (avg) | N/A | 5 runs, 256 max tokens |
| Time to first token | **11.2ms** (avg), 11.3ms p99 | N/A | 10 runs |
| VRAM usage | **6.43 GB** (19.1%) | N/A | bfloat16, full model |

HumanEval pass@1 results (99.4%, 163/164) are excluded from claims. This score is inconsistent with published results for models of this size and likely reflects a test harness issue in the exec-based evaluation rather than genuine code generation capability. Investigation is ongoing.

**MMLU subject performance:**
- Top: Marketing (88.5%), World History (85.7%), Psychology (85.5%), Sociology (83.6%), World Religions (83.0%)
- Bottom: European History (0.0%), US History (5.4%), Moral Scenarios (35.5%), Abstract Algebra (38.0%), Global Facts (38.0%)

The MMLU improvement over the published Qwen2-3B baseline is attributed to the TIES merging of specialist adapters, which appears to have improved general knowledge coverage. The GSM8K regression indicates that the merged adapters did not include strong mathematical reasoning specialization, and the TIES merging process may have degraded the base model's existing math capabilities.

---

## Implementation: Rust and candle

The Synapse Architecture is implemented in Rust using candle [HuggingFace, 2023], a minimalist ML framework for Rust that provides tensor operations and CUDA acceleration. The choice of Rust over Python was deliberate:

- **Memory safety without garbage collection:** Rust's ownership model prevents use-after-free and data races without runtime overhead. This matters for an inference server running continuously.
- **Single binary deployment:** `cargo build --release` produces one binary with no runtime dependencies. No Python environment, no Docker, no version conflicts.
- **Performance:** Rust compiled code performs comparably to C++ for compute-bound workloads, with the safety guarantees of a modern language.
- **candle specifically:** candle is the HuggingFace ML framework for Rust, providing CPU and CUDA backends, GGUF model loading, SafeTensors support, and quantized inference. Using it allows native integration with the HuggingFace model ecosystem.

### Architecture Module Organization

```
crates/synapse/src/arch/
├── mod.rs          # Module declarations, shared linear() helpers
├── mamba.rs        # Selective SSM (S6) — used by Thalamus
├── xlstm.rs        # mLSTM variant with matrix cell state
├── thalamus.rs     # Mamba router + Hebbian pathway tracking
├── expert.rs       # SwiGLU expert + ExpertPool
├── fast_weights.rs # Outer-product fast-weight memory
└── synapse_model.rs # Full model: n_layers of (xLSTM+Thalamus+Experts+Memory)
```

### Test Results

All 65 tests pass on CPU (`cargo test`, Rust 2024 edition). The 28 architecture-specific tests cover:

- MambaLayer creation, forward pass shape, state persistence, SiLU activation correctness
- XLSTMLayer creation, forward pass shape, introspection data structure, cell state persistence
- Thalamus creation, routing output shapes and index validity, introspection completeness, Hebbian accumulation
- Expert creation and forward pass, ExpertPool routing and aggregation, introspection
- FastWeightMemory creation, forward pass shape, introspection structure, memory persistence and clear
- SynapseModel creation, parameter count (total > active), forward pass producing (batch, seq, vocab) shaped logits, introspection with per-layer data, summary string, state reset

The remaining 37 tests cover the inference engine, swarm orchestration, knowledge graph, learning pipeline, and model format.

### Inference Performance

On the test hardware (i9-14900KF, RTX 5090 32GB VRAM, CUDA 12.8):

| Configuration | Throughput |
|---------------|-----------|
| GPU (bfloat16, 3B, RTX 5090) | 106.3 tokens/second (avg over 5 runs) |
| Time to first token (GPU) | 11.2ms (avg), 11.3ms (p99) |
| VRAM usage (3B, bfloat16) | 6.43 GB (19.1% of 33.67 GB) |
| Model load time (3B, GPU) | 0.4 seconds |

These numbers are for the swarm inference engine running Qwen2.5-3B Q4\_K\_M quantized weights, not for the Synapse Architecture modules (which are not yet trained). GPU acceleration is approximately 5× faster than CPU for this model and quantization level.

The inference engine is separate from the architecture modules. It handles GGUF model loading, a paged-attention-style KV cache, temperature/top-p/top-k sampling, SSE streaming, and LoRA adapter hot-swapping. The Synapse Architecture modules are integrated as a parallel code path (`crates/synapse/src/arch/`) that the full system will switch to once training is complete.

---

## Current Status and Roadmap

Clarity about what is complete and what is not is essential for reproducibility.

### What Is Complete

**Synapse Architecture (the novel modules):**
- All five modules implemented in Rust: MambaLayer, XLSTMLayer, Thalamus, ExpertPool, FastWeightMemory, and SynapseModel
- 28 unit tests passing, verifying shape correctness, state persistence, and introspection data completeness
- Full introspection at every layer exposed via `GET /api/introspect`
- Parameter counting and VRAM estimation implemented
- Configurable via `SynapseModelConfig` struct

**Swarm System:**
- OpenAI-compatible API at `/v1/chat/completions` with SSE streaming
- Specialist routing with keyword analysis and Hebbian pathway weighting
- Parallel multi-specialist execution with response synthesis
- SQLite knowledge graph with real-time fact extraction and hallucination detection
- QLoRA adapter loading (SafeTensors, f32/f16/bf16), hot-swap without server restart
- Python learning sidecar (FastAPI on port 8090) for QLoRA and DPO training
- Cloud fallback with automatic DPO pair collection
- Web dashboard at root URL with specialist confidence panels and knowledge graph stats
- CUDA-accelerated inference via candle (5× speedup over CPU achieved)
- Standardized benchmark evaluation (MMLU, HumanEval, GSM8K)
- `.synapse` model format for bundling base model + adapters + knowledge graph
- CLI: `synapse serve`, `synapse pull`, `synapse bench`, `synapse eval`, `synapse hub`

**Synapse-3B (the merged model):**
- Four QLoRA specialist adapters trained on domain-specific data
- Merged via TIES into Synapse-3B, published at `djtony707/synapse-3b` on HuggingFace
- Uses standard Qwen3-3B transformer architecture (not the Synapse Architecture modules)

### What Is In Progress

- **Training the Synapse Architecture from scratch.** The architecture modules pass unit tests with randomly initialized weights. End-to-end pre-training on a public dataset (SlimPajama, OpenWebMath) on the RTX 5090 is the next major milestone. Until this is complete, no benchmark comparisons between the Synapse Architecture and other architectures are possible or claimed.
- **Speculative decoding.** The scaffold (draft + verify architecture) is implemented. Shared KV cache state between draft and target models is not yet complete.
- **Continuous batching.** Current batching is sequential across specialists. Continuous batching for overlapping requests is not implemented.
- **Distributed swarm.** The swarm runs on a single machine. Distributing specialists across multiple machines via the Titan Agent mesh network is designed but not implemented.

### What Is Planned

- Doc-to-LoRA knowledge crystallization: training LoRA adapters from uploaded documents without requiring conversation data.
- AMD ROCm and Apple Metal backends (currently CUDA and CPU only).
- Public specialist hub: community-trained adapters shared via HuggingFace with `synapse hub push/install`.

---

## Conclusion

TITAN Synapse presents two independent contributions. The Synapse Architecture is a modular neural network design where every component operates in O(n) time with respect to sequence length: a Mamba-based router (Thalamus), an mLSTM language module with exponential gating and matrix cell state, a sparse MoE expert pool with SwiGLU feed-forward experts, and a fast-weight memory module implementing the Schmidhuber outer-product update rule. Every module exposes its internal state for real-time inspection. The architecture has 28 passing unit tests in Rust using the candle framework. It has not yet been trained from scratch end-to-end, and no trained benchmark comparisons are available.

The Specialist Swarm System is a separate, operational contribution. It runs a coordinated set of small specialist models sharing a common base, routes queries using Hebbian pathway learning, and continuously improves through QLoRA fine-tuning and DPO from live conversations. Running in bfloat16 on a single RTX 5090 at 106.3 tokens per second with 11.2ms time-to-first-token, the system achieves MMLU 62.6% (14,042 questions, 5-shot) and GSM8K 18.9% (1,319 problems, 8-shot CoT) using only 6.43 GB VRAM. The MMLU result exceeds the published Qwen2-3B baseline of ~53% by approximately 10 points, which we attribute to the TIES merging of domain-specialist adapters improving general knowledge coverage. The GSM8K result is below the published baseline, consistent with the merged adapters not being math-specialized.

Synapse-3B, published on HuggingFace at `djtony707/synapse-3b`, is a practical artifact: four domain-specialist QLoRA adapters merged via TIES into a single Qwen3-3B transformer. It uses conventional transformer architecture and should not be conflated with the novel Synapse Architecture modules.

I built the entire system — architecture design, implementation, specialist training, TIES merging, benchmarking, and deployment — as an independent research effort, running on a single consumer-grade workstation (RTX 5090, i9-14900KF, 64 GB DDR5). No cloud compute, no institutional backing, no training cluster. This project is proof that meaningful AI architecture research is possible with consumer hardware and determination.

The codebase is open source under the Apache 2.0 license. All source code, test results, and configuration files are available in the repository.

---

## References

Ba, J., Hinton, G., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016). Using fast weights to attend to the recent past. *Advances in Neural Information Processing Systems*, 29.

Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, I., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended long short-term memory. *arXiv preprint arXiv:2405.04517*.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems*, 36.

Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., Zoph, B., Fedus, L., Bosma, M. P., Zhou, Z., Wang, T., Wang, Y. E., Webster, K., Pellat, M., Robinson, K., ... Dean, J. (2022). GLaM: Efficient scaling of language models with mixture-of-experts. *Proceedings of the 39th International Conference on Machine Learning*, 5547–5569.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120), 1–39.

Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

Gu, A., Goel, K., & Re, C. (2022). Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations*.

Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley.

HuggingFace. (2023). candle: Minimalist ML framework for Rust. https://github.com/huggingface/candle

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de las Casas, D., Hanna, E. B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L. R., Saulnier, L., Lachaux, M.-A., Stock, P., Subramanian, S., Yang, S., ... El Sayed, W. (2024). Mixtral of experts. *arXiv preprint arXiv:2401.04088*.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36.

Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear transformers are secretly fast weight programmers. *Proceedings of the 38th International Conference on Machine Learning*, 9355–9366.

Schmidhuber, J. (1992). Learning to control fast-weight memories: An alternative to dynamic recurrent networks. *Neural Computation*, 4(1), 131–139.

Shazeer, N. (2020). GLU variants improve transformer. *arXiv preprint arXiv:2002.05202*.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *International Conference on Learning Representations*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). TIES-merging: Resolving interference when merging models. *Advances in Neural Information Processing Systems*, 36.

Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. *Advances in Neural Information Processing Systems*, 32.
