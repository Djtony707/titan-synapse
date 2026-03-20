#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Titan Synapse — Install Script
# Small models that think together. And learn.
# https://github.com/Djtony707/titan-synapse
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
step()    { echo -e "\n${CYAN}${BOLD}>>> $*${NC}"; }

# ── ASCII Header ──────────────────────────────────────────────────────
echo -e "${MAGENTA}"
cat << 'BANNER'

   ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗
   ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝
   ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗
   ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝
   ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗
   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝

BANNER
echo -e "${NC}"
echo -e "   ${DIM}Tiny models. Big brain. Your hardware. No excuses.${NC}"
echo -e "   ${DIM}────────────────────────────────────────────────${NC}"
echo ""

# ── Constants ─────────────────────────────────────────────────────────
REPO_URL="https://github.com/Djtony707/titan-synapse.git"
SYNAPSE_DIR="${HOME}/.synapse"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
BINARY_NAME="synapse"

# ── OS Detection ──────────────────────────────────────────────────────
step "Detecting operating system"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        PLATFORM="linux"
        success "Linux detected (${ARCH})"
        ;;
    Darwin)
        PLATFORM="macos"
        success "macOS detected (${ARCH})"
        ;;
    *)
        fail "Unsupported operating system: ${OS}. Synapse supports Linux and macOS."
        ;;
esac

# ── Dependency Checks ─────────────────────────────────────────────────
step "Checking dependencies"

# git
if command -v git &>/dev/null; then
    success "git found: $(git --version | head -1)"
else
    fail "git is not installed. Please install git first."
fi

# curl or wget
DOWNLOADER=""
if command -v curl &>/dev/null; then
    DOWNLOADER="curl"
    success "curl found"
elif command -v wget &>/dev/null; then
    DOWNLOADER="wget"
    success "wget found"
else
    fail "Neither curl nor wget found. Please install one of them."
fi

# ── Rust Toolchain ────────────────────────────────────────────────────
step "Checking Rust toolchain"

if command -v rustc &>/dev/null && command -v cargo &>/dev/null; then
    RUST_VER="$(rustc --version)"
    success "Rust already installed: ${RUST_VER}"
else
    warn "Rust not found. Installing via rustup..."
    if [ "$DOWNLOADER" = "curl" ]; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    else
        wget -qO- https://sh.rustup.rs | sh -s -- -y
    fi
    # Source cargo env for this session
    # shellcheck source=/dev/null
    source "${HOME}/.cargo/env" 2>/dev/null || true
    if command -v rustc &>/dev/null; then
        success "Rust installed: $(rustc --version)"
    else
        fail "Rust installation failed. Try manually: https://rustup.rs"
    fi
fi

# ── CUDA Detection ────────────────────────────────────────────────────
step "Checking for CUDA toolkit"

CARGO_FEATURES=""
if command -v nvcc &>/dev/null; then
    CUDA_VER="$(nvcc --version | grep -oP 'release \K[0-9.]+' 2>/dev/null || nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    success "CUDA toolkit detected: ${CUDA_VER}"
    CARGO_FEATURES="--features cuda"
    info "Build will include CUDA acceleration"
elif [ -d "/usr/local/cuda" ] || [ -d "/opt/cuda" ]; then
    warn "CUDA directory found but nvcc not in PATH. Building without CUDA."
    info "To enable CUDA: export PATH=/usr/local/cuda/bin:\$PATH and re-run"
else
    info "No CUDA toolkit found. Building CPU-only (this is fine for starters)."
    # Check for Metal on macOS
    if [ "$PLATFORM" = "macos" ] && [ "$ARCH" = "arm64" ]; then
        CARGO_FEATURES="--features metal"
        info "Apple Silicon detected — building with Metal acceleration"
    fi
fi

# ── Clone or Use Current Directory ────────────────────────────────────
step "Setting up source code"

BUILD_DIR=""
if [ -f "Cargo.toml" ] && grep -q "titan-synapse\|synapse" Cargo.toml 2>/dev/null; then
    BUILD_DIR="$(pwd)"
    success "Already in titan-synapse repo: ${BUILD_DIR}"
elif [ -d "titan-synapse" ]; then
    BUILD_DIR="$(pwd)/titan-synapse"
    success "Found existing clone: ${BUILD_DIR}"
else
    info "Cloning titan-synapse..."
    git clone "$REPO_URL" titan-synapse
    BUILD_DIR="$(pwd)/titan-synapse"
    success "Cloned to ${BUILD_DIR}"
fi

cd "$BUILD_DIR"

# ── Build ─────────────────────────────────────────────────────────────
step "Building Synapse (release mode)"

info "This may take a few minutes on first build..."
if [ -n "$CARGO_FEATURES" ]; then
    info "Build flags: ${CARGO_FEATURES}"
    cargo build --release ${CARGO_FEATURES}
else
    cargo build --release
fi

if [ -f "target/release/${BINARY_NAME}" ]; then
    success "Build complete!"
else
    fail "Build failed — binary not found at target/release/${BINARY_NAME}"
fi

# ── Create ~/.synapse Directory ───────────────────────────────────────
step "Setting up Synapse home directory"

mkdir -p "${SYNAPSE_DIR}"
mkdir -p "${SYNAPSE_DIR}/models"
mkdir -p "${SYNAPSE_DIR}/knowledge"
mkdir -p "${SYNAPSE_DIR}/adapters"
mkdir -p "${SYNAPSE_DIR}/logs"

success "Created ${SYNAPSE_DIR}/"
info "  models/     — GGUF model files"
info "  knowledge/  — SQLite knowledge graphs"
info "  adapters/   — QLoRA adapter weights"
info "  logs/       — Runtime logs"

# ── Install Binary ────────────────────────────────────────────────────
step "Installing binary"

INSTALL_DIR=""
if [ -d "${HOME}/.local/bin" ] || mkdir -p "${HOME}/.local/bin" 2>/dev/null; then
    INSTALL_DIR="${HOME}/.local/bin"
elif [ -w "/usr/local/bin" ]; then
    INSTALL_DIR="/usr/local/bin"
else
    warn "Cannot write to ~/.local/bin or /usr/local/bin"
    info "Attempting /usr/local/bin with sudo..."
    sudo mkdir -p /usr/local/bin 2>/dev/null || true
    if [ -w "/usr/local/bin" ] || sudo test -w "/usr/local/bin" 2>/dev/null; then
        INSTALL_DIR="/usr/local/bin"
        sudo cp "target/release/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
        sudo chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
        success "Installed to ${INSTALL_DIR}/${BINARY_NAME} (via sudo)"
        INSTALL_DIR="" # skip the normal copy below
    else
        fail "No writable install directory. Copy target/release/synapse to your PATH manually."
    fi
fi

if [ -n "$INSTALL_DIR" ]; then
    cp "target/release/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
    chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
    success "Installed to ${INSTALL_DIR}/${BINARY_NAME}"

    # Check if install dir is in PATH
    if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
        warn "${INSTALL_DIR} is not in your PATH"
        info "Add this to your shell profile:"
        echo -e "  ${BOLD}export PATH=\"${INSTALL_DIR}:\$PATH\"${NC}"
    fi
fi

# ── Download Default Model ────────────────────────────────────────────
step "Downloading default model (Qwen2.5-0.5B Q4_K_M)"

MODEL_PATH="${SYNAPSE_DIR}/models/${MODEL_FILE}"

if [ -f "$MODEL_PATH" ]; then
    success "Model already exists: ${MODEL_PATH}"
else
    info "Downloading from HuggingFace (~400MB)..."
    if [ "$DOWNLOADER" = "curl" ]; then
        curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
    else
        wget --show-progress -O "$MODEL_PATH" "$MODEL_URL"
    fi

    if [ -f "$MODEL_PATH" ] && [ -s "$MODEL_PATH" ]; then
        MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
        success "Model downloaded: ${MODEL_PATH} (${MODEL_SIZE})"
    else
        warn "Model download may have failed. You can retry manually:"
        info "  synapse pull qwen2.5-0.5b"
    fi
fi

# ── Write Default Config ─────────────────────────────────────────────
CONFIG_PATH="${SYNAPSE_DIR}/config.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    cat > "$CONFIG_PATH" << 'YAML'
# Titan Synapse Configuration
# Docs: https://github.com/Djtony707/titan-synapse

server:
  host: "127.0.0.1"
  port: 6900

model:
  path: "~/.synapse/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  context_length: 4096

learning:
  enabled: true
  min_conversations: 5
  eval_threshold: 0.7

knowledge:
  database: "~/.synapse/knowledge/graph.db"

logging:
  level: "info"
YAML
    success "Default config written to ${CONFIG_PATH}"
fi

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}"
cat << 'DONE'
   ╔══════════════════════════════════════════════════════╗
   ║          Installation complete!                     ║
   ╚══════════════════════════════════════════════════════╝
DONE
echo -e "${NC}"

echo -e "  ${BOLD}Next steps:${NC}"
echo ""
echo -e "    ${CYAN}1.${NC} Start the engine:"
echo -e "       ${BOLD}synapse up${NC}"
echo ""
echo -e "    ${CYAN}2.${NC} Chat with it:"
echo -e "       ${BOLD}curl http://localhost:6900/v1/chat/completions \\${NC}"
echo -e "       ${BOLD}  -H 'Content-Type: application/json' \\${NC}"
echo -e "       ${BOLD}  -d '{\"model\":\"synapse\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'${NC}"
echo ""
echo -e "    ${CYAN}3.${NC} Check status:"
echo -e "       ${BOLD}synapse status${NC}"
echo ""
echo -e "    ${CYAN}4.${NC} Pull more models:"
echo -e "       ${BOLD}synapse pull qwen3-3b${NC}"
echo ""
echo -e "  ${DIM}Config:  ${SYNAPSE_DIR}/config.yaml${NC}"
echo -e "  ${DIM}Models:  ${SYNAPSE_DIR}/models/${NC}"
echo -e "  ${DIM}Docs:    https://github.com/Djtony707/titan-synapse${NC}"
echo ""
echo -e "  ${DIM}────────────────────────────────────────────────${NC}"
echo -e "  ${DIM}Created by Tony Elliott${NC}"
echo -e "  ${DIM}https://github.com/Djtony707${NC}"
echo ""
