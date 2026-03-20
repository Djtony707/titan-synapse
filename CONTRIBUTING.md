# Contributing to Titan Synapse

First off — thanks for considering a contribution. Synapse is a passion project, but it has real teeth and real ambitions. If you're here, you probably believe small models can outperform big ones when they work together. Good. You're in the right place.

---

## Building from Source

```bash
# Clone it
git clone https://github.com/Djtony707/titan-synapse.git
cd titan-synapse

# Build (CPU only)
cargo build --release

# Build with CUDA (Linux, NVIDIA GPU)
cargo build --release --features cuda

# Build with Metal (macOS, Apple Silicon)
cargo build --release --features metal

# The binary lands at:
./target/release/synapse
```

**Requirements:**
- Rust 2024 edition (install via [rustup.rs](https://rustup.rs))
- For CUDA builds: CUDA toolkit 12.x+ and `nvcc` in PATH
- For the learning sidecar: Python 3.10+ with `torch`, `peft`, `trl` (see `python/requirements.txt`)
- Patience during first build. Candle compiles a lot of tensor ops. Go make coffee.

---

## Running Tests

```bash
# Run the full suite
cargo test

# Run tests for a specific module
cargo test --package synapse -- knowledge
cargo test --package synapse -- inference

# Run with output (for debugging)
cargo test -- --nocapture

# Run only ignored (slow) integration tests
cargo test -- --ignored
```

Tests should pass on both Linux and macOS. If they don't, that's a bug — file it.

---

## Project Structure

```
titan-synapse/
├── Cargo.toml                    # Workspace root
├── crates/synapse/src/
│   ├── main.rs                   # CLI entry point (clap): serve, status, models, pull, learn, bench
│   ├── server.rs                 # Axum HTTP server on :6900
│   ├── openai.rs                 # OpenAI-compatible API handlers
│   ├── inference/
│   │   ├── engine.rs             # Core inference engine (candle)
│   │   ├── gguf.rs               # GGUF model loader
│   │   ├── sampler.rs            # Temperature/top-p/top-k sampling
│   │   └── kv_cache.rs           # PagedAttention-style KV cache
│   ├── swarm/
│   │   ├── coordinator.rs        # Routes queries to specialists
│   │   ├── specialist.rs         # Specialist agent definition
│   │   └── hebbian.rs            # Hebbian routing weights
│   ├── knowledge/
│   │   ├── graph.rs              # SQLite knowledge graph
│   │   └── facts.rs              # Fact extraction + storage
│   ├── learning/
│   │   ├── pipeline.rs           # QLoRA + DPO training orchestration
│   │   ├── eval.rs               # Self-evaluation scoring
│   │   └── sidecar.rs            # Python process management
│   └── format/
│       └── synapse_file.rs       # .synapse bundle format
├── config/                       # Default configuration files
├── python/                       # Learning sidecar (QLoRA/DPO)
├── docker-compose.yml            # For the learning sidecar
└── target/                       # Build artifacts (gitignored)
```

The crate is a single binary (`synapse`) built from `crates/synapse/`. We may add more workspace members later (e.g., `crates/synapse-python` for the FFI bridge), but for now, simplicity wins.

---

## Code Style

**Edition:** Rust 2024. We use the latest stable features.

**General rules:**

- `cargo fmt` before every commit. Non-negotiable.
- `cargo clippy` should produce zero warnings. If Clippy is wrong (rare), add an `#[allow()]` with a comment explaining why.
- Error handling: use `anyhow::Result` for application code, `thiserror` for library-facing errors. No `.unwrap()` in production paths. Tests can `.unwrap()` — they're supposed to panic on failure, that's the point.
- Naming: structs are `PascalCase`, functions are `snake_case`, constants are `SCREAMING_SNAKE`. Standard Rust. Nothing weird.
- Comments: explain **why**, not **what**. The code should explain what. If it can't, refactor it until it does.
- Keep functions under ~50 lines when possible. If a function needs a scroll wheel, it needs a refactor.
- Async: we use Tokio. If you block the async runtime, you buy the team coffee. Metaphorically. We don't have a coffee fund yet.

**Commit messages:**

```
feat: add Metal acceleration for Apple Silicon
fix: prevent KV cache overflow on long contexts
refactor: split coordinator routing into separate module
test: add integration tests for .synapse bundle format
docs: update CUDA build instructions
```

Conventional commits. Keep the subject under 72 characters. Body is optional but appreciated for non-obvious changes.

---

## PR Process

1. **Fork and branch.** Branch names: `feat/thing`, `fix/thing`, `refactor/thing`.
2. **Make your changes.** Keep PRs focused — one feature or fix per PR. Mega-PRs get mega-delayed.
3. **Run the checks:**
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```
4. **Open a PR** against `main`. Fill in the template (what, why, how to test).
5. **Review.** We'll review within a few days. Don't take feedback personally — we're all trying to make this thing great.
6. **Merge.** Squash-merge is preferred for clean history.

If your PR adds a new feature, include tests. If it fixes a bug, include a test that would have caught it. "Works on my machine" is not a test plan.

---

## Where Help Is Needed

These are the areas where contributions would have the most impact:

### CUDA Optimization
The candle CUDA backend works but there's room for custom kernels — especially for the Hebbian routing matrix updates and batched specialist inference. If you've written CUDA kernels before and enjoy staring at memory access patterns, we want you.

### New Specialists
The swarm currently has a handful of specialists. We want more: math reasoning, code generation, creative writing, structured output, tool calling. Each specialist is relatively self-contained — it's a great first contribution.

### Inference Performance
KV cache management, speculative decoding, continuous batching, quantization-aware optimizations. If you can make the inference engine faster without sacrificing quality, that's a direct win for every user.

### Model Format (.synapse)
The `.synapse` bundle format is new and could use: compression, integrity verification, streaming loads, and a proper spec document. If you like file formats (we don't judge), this one's for you.

### Testing
More integration tests, benchmarks, and edge case coverage. The test suite is solid but not exhaustive. Especially needed: stress tests for concurrent inference, memory leak detection, and cross-platform CI.

### Documentation
API docs, tutorials, example configs, deployment guides. If you can explain things clearly, that's a superpower we need.

---

## Setting Up for Development

```bash
# Recommended: install rust-analyzer for your editor
rustup component add rust-analyzer

# Watch mode (rebuild on save)
cargo install cargo-watch
cargo watch -x "test" -x "clippy"

# Generate docs locally
cargo doc --open --no-deps
```

---

## Ground Rules

- Be respectful. This is a technical project, not a battlefield.
- If you're stuck, open an issue or discussion before spending three days on something that might get rejected.
- No AI-generated PRs without understanding the code. We can tell. We literally build AI.

---

Built by Tony Elliott. Contributions make it better.
