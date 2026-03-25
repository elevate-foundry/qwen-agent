#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------
# Qwen Agent — One-command bootstrap
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/<repo>/setup.sh | bash
#   -- or --
#   chmod +x setup.sh && ./setup.sh
#
# What this does:
#   1. Detects OS (macOS / Linux)
#   2. Installs Ollama if not present
#   3. Starts Ollama server
#   4. Pulls qwen3.5 model
#   5. Sets up Python venv + dependencies
#   6. Seeds memory.json if not present
#   7. Launches the agent
# -----------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-qwen3.5}"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }
step() { echo -e "\n${CYAN}───${NC} $* ${CYAN}───${NC}"; }

# -----------------------------------------------------------
# 1) Detect OS
# -----------------------------------------------------------
step "Detecting platform"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin) PLATFORM="macOS" ;;
    Linux)  PLATFORM="Linux" ;;
    *)      err "Unsupported OS: $OS (use setup.ps1 for Windows)"; exit 1 ;;
esac

log "Platform: ${PLATFORM} (${ARCH})"

# -----------------------------------------------------------
# 2) Install Ollama
# -----------------------------------------------------------
step "Checking Ollama"

if command -v ollama &>/dev/null; then
    OLLAMA_VERSION="$(ollama --version 2>/dev/null || echo 'unknown')"
    log "Ollama already installed: ${OLLAMA_VERSION}"
else
    warn "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    if command -v ollama &>/dev/null; then
        log "Ollama installed successfully."
    else
        err "Ollama installation failed."
        exit 1
    fi
fi

# -----------------------------------------------------------
# 3) Start Ollama server
# -----------------------------------------------------------
step "Starting Ollama server"

if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    log "Ollama server already running."
else
    warn "Starting Ollama server in background..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!

    # Wait up to 30s for server
    for i in $(seq 1 30); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            log "Ollama server started (PID: ${OLLAMA_PID})."
            break
        fi
        sleep 1
    done

    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        err "Ollama server failed to start within 30s."
        exit 1
    fi
fi

# -----------------------------------------------------------
# 4) Pull model
# -----------------------------------------------------------
step "Pulling model: ${MODEL}"

# Check if already pulled
if ollama list 2>/dev/null | grep -q "${MODEL}"; then
    log "Model ${MODEL} already available."
else
    warn "Downloading ${MODEL} (this may take a few minutes)..."
    ollama pull "${MODEL}"
    log "Model ${MODEL} pulled successfully."
fi

# -----------------------------------------------------------
# 5) Python environment
# -----------------------------------------------------------
step "Setting up Python environment"

# Check Python version
if ! command -v "${PYTHON}" &>/dev/null; then
    err "Python 3 not found. Install Python 3.9+ and try again."
    exit 1
fi

PY_VERSION="$(${PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="$(echo "$PY_VERSION" | cut -d. -f1)"
PY_MINOR="$(echo "$PY_VERSION" | cut -d. -f2)"

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    err "Python 3.9+ required, found ${PY_VERSION}."
    exit 1
fi

log "Python ${PY_VERSION} found."

# Create venv if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    warn "Creating virtual environment..."
    "${PYTHON}" -m venv "${VENV_DIR}"
    log "Virtual environment created at ${VENV_DIR}"
fi

# Activate
source "${VENV_DIR}/bin/activate"
log "Virtual environment activated."

# Install deps
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    warn "Installing Python dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"
    log "Dependencies installed."
else
    warn "No requirements.txt found. Installing minimal deps..."
    pip install --quiet ollama pydantic requests
    log "Minimal dependencies installed."
fi

# -----------------------------------------------------------
# 6) Seed memory
# -----------------------------------------------------------
step "Checking memory.json"

MEMORY_FILE="${SCRIPT_DIR}/memory.json"

if [ -f "${MEMORY_FILE}" ]; then
    MEMORY_COUNT="$(${PYTHON} -c "import json; d=json.load(open('${MEMORY_FILE}')); print(len(d.get('memories', [])))")"
    AUDIT_COUNT="$(${PYTHON} -c "import json; d=json.load(open('${MEMORY_FILE}')); print(len(d.get('audit_log', [])))")"
    log "memory.json exists: ${MEMORY_COUNT} memories, ${AUDIT_COUNT} audit entries."
else
    warn "No memory.json found. Creating seed..."
    cat > "${MEMORY_FILE}" << 'SEED'
{
  "metadata": {
    "created": null,
    "version": 3,
    "owner": "model",
    "last_modified": null,
    "memory_count": 0,
    "audit_entries": 1
  },
  "memories": [],
  "audit_log": [
    {
      "timestamp": null,
      "event": "created",
      "actor": "executor",
      "description": "Initialized empty memory store via setup script."
    }
  ]
}
SEED
    # Stamp real timestamps
    NOW="$(${PYTHON} -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')"
    ${PYTHON} -c "
import json
with open('${MEMORY_FILE}') as f: d = json.load(f)
d['metadata']['created'] = '${NOW}'
d['metadata']['last_modified'] = '${NOW}'
d['audit_log'][0]['timestamp'] = '${NOW}'
with open('${MEMORY_FILE}', 'w') as f: json.dump(d, f, indent=2)
"
    log "memory.json seeded with real timestamps."
fi

# -----------------------------------------------------------
# 7) Verify setup
# -----------------------------------------------------------
step "Verifying setup"

# Quick import check
"${PYTHON}" -c "
import ollama, pydantic, requests
print('  ollama:', ollama.__version__ if hasattr(ollama, '__version__') else 'ok')
print('  pydantic:', pydantic.__version__)
print('  requests:', requests.__version__)
" 2>/dev/null && log "All imports verified." || { err "Import check failed."; exit 1; }

# Verify agent loads
"${PYTHON}" -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from qwen3_5 import Op, ProofPlan, PersistentMemory, Trace
from training_algebra import TrainingOp, TrainingStep
print('  Tool ops:', len(Op))
print('  Training ops:', len(TrainingOp))
" 2>/dev/null && log "Agent modules verified." || { err "Agent module check failed."; exit 1; }

# -----------------------------------------------------------
# Done
# -----------------------------------------------------------
step "Setup complete"

echo ""
echo -e "${GREEN}Everything is ready.${NC}"
echo ""
echo "  Run the agent:"
echo -e "    ${CYAN}source .venv/bin/activate${NC}"
echo -e "    ${CYAN}python qwen3_5.py 'your request here'${NC}"
echo ""
echo "  Deploy to Modal:"
echo -e "    ${CYAN}modal deploy modal_app.py${NC}"
echo ""
echo "  Restore memory from backup:"
echo -e "    ${CYAN}python qwen3_5.py --restore${NC}"
echo ""
