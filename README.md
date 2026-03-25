# Qwen Agent — Provable Finite Tool Algebra

A self-improving AI agent built on a **finite tool algebra** with provable safety guarantees, model-owned persistent memory, and real GPU training via Modal.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Web UI (Chat / Memory / Audit Log)         │
├─────────────────────────────────────────────┤
│  FastAPI (Modal ASGI)                       │
├──────────────────┬──────────────────────────┤
│  Agent (GPU)     │  Trainer (GPU)           │
│  Ollama + Qwen   │  transformers/peft/trl   │
├──────────────────┴──────────────────────────┤
│  Tool Algebra        │  Training Algebra     │
│  search_web          │  lora / qlora / dora  │
│  fetch_url           │  full_finetune        │
│  list_dir            │  chinchilla           │
│  read_text           │  dpo                  │
│  train ─────────────►│  distillation         │
│  answer_direct       │  merging / pruning    │
│                      │  evaluate             │
├──────────────────────┴──────────────────────┤
│  Persistent Memory (Modal Volume)           │
│  memory.json + memory_backup.json           │
│  Immutable audit log (SOC 2 / ISO style)    │
└─────────────────────────────────────────────┘
```

## What's provable

- Only enum-defined tools can execute
- Arguments are schema-validated before execution
- Filesystem access is sandboxed
- No arbitrary shell execution
- Every memory mutation is audit-logged with timestamps + state hashes
- Training configs are validated (e.g., QLoRA enforces 4-bit, distillation weights must sum to 1.0)

## Files

| File | Purpose |
|---|---|
| `qwen3_5.py` | Main agent: tool algebra, trace algebra, memory, model interface |
| `training_algebra.py` | Training tool algebra: LoRA, QLoRA, DoRA, DPO, etc. |
| `modal_app.py` | Modal deployment: GPU runtime, web UI, real training |
| `memory.json` | Model-owned persistent memory with audit log |
| `bootstrap.sh` | One-command bootstrap (macOS / Linux) |
| `bootstrap.ps1` | One-command bootstrap (Windows) |
| `qwen3_5_backup.py` | Original agent code (pre-enhancement backup) |

## Quick Start

```bash
# macOS / Linux — one command from zero:
./bootstrap.sh

# Windows (PowerShell):
.\bootstrap.ps1

# Or remote:
curl -fsSL https://raw.githubusercontent.com/<repo>/bootstrap.sh | bash
```

This installs Ollama, pulls qwen3.5, sets up a Python venv, seeds memory, and verifies everything.

## Manual Usage

```bash
source .venv/bin/activate
python qwen3_5.py 'what files are in this directory?'

# Restore memory from backup
python qwen3_5.py --restore
```

## Deploy to Modal

```bash
# Install Modal CLI
pip install modal
modal setup  # one-time auth

# Deploy (creates GPU-backed app with web UI)
modal deploy modal_app.py

# Your app will be live at:
# https://your-workspace--qwen-agent-web.modal.run
```

## Memory System

Qwen owns her own memory. After each interaction, she decides:
- **What to remember** — `key_insight` + `relevance`
- **What to forget** — drop stale memories by index
- **TTL** — optional `forget_after` for auto-expiring memories

Every mutation is logged to an **immutable audit trail**:
- ISO 8601 timestamps (UTC)
- Event type: `created`, `added`, `dropped`, `expired`, `restored`, `session_start`
- Actor: `model`, `executor`, `ryan`, `cascade`
- SHA-256 state hash after each change

## Training (via Modal GPU)

The agent can request training operations through the `train` tool op. On Modal, these execute on real A10G GPUs:

| Op | Backend | What it does |
|---|---|---|
| `lora` | peft | Low-rank adaptation |
| `qlora` | peft + bitsandbytes | 4-bit quantized LoRA |
| `dora` | peft | Weight-decomposed LoRA |
| `dpo` | trl | Direct preference optimization |

Training artifacts are saved to the Modal volume at `/data/training_output/`.
