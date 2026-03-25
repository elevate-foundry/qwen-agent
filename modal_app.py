#!/usr/bin/env python3
"""
Modal deployment for the Qwen provable-agent with:
- Ollama serving Qwen 3.5 on GPU
- Persistent memory.json on a Modal Volume
- Web UI for chat
- Real GPU training via the training algebra
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import modal

# ----------------------------
# Modal infrastructure
# ----------------------------

app = modal.App("qwen-agent")

# Persistent volume for memory.json and model artifacts
volume = modal.Volume.from_name("qwen-agent-data", create_if_missing=True)
VOLUME_PATH = "/data"
MEMORY_PATH = f"{VOLUME_PATH}/memory.json"
MEMORY_BACKUP_PATH = f"{VOLUME_PATH}/memory_backup.json"
MODELS_PATH = f"{VOLUME_PATH}/models"
TRAINING_OUTPUT_PATH = f"{VOLUME_PATH}/training_output"

# Base image with Ollama + Python deps
ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "procps")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
    )
    .pip_install(
        "ollama>=0.4.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
    )
    .copy_local_file("qwen3_5.py", "/app/qwen3_5.py")
    .copy_local_file("training_algebra.py", "/app/training_algebra.py")
    .copy_local_file("memory.json", "/app/memory_seed.json")
)

# Training image with HuggingFace stack (heavier, used only for train ops)
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
        "datasets>=3.0.0",
        "bitsandbytes>=0.44.0",
        "accelerate>=1.0.0",
        "pydantic>=2.0.0",
    )
    .copy_local_file("training_algebra.py", "/app/training_algebra.py")
)


# ----------------------------
# Ollama lifecycle helpers
# ----------------------------

def _start_ollama():
    """Start Ollama server in background and wait until ready."""
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = MODELS_PATH
    subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for Ollama to be ready
    for _ in range(60):
        try:
            import requests as req
            resp = req.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Ollama failed to start within 60s")


def _ensure_model(model_name: str = "qwen3.5"):
    """Pull model if not already present."""
    import requests as req
    resp = req.get("http://localhost:11434/api/tags", timeout=10)
    models = [m["name"] for m in resp.json().get("models", [])]
    if not any(model_name in m for m in models):
        print(f"Pulling {model_name}...")
        subprocess.run(["ollama", "pull", model_name], check=True,
                        env={**os.environ, "OLLAMA_MODELS": MODELS_PATH})
        print(f"{model_name} pulled successfully.")


def _ensure_memory():
    """Seed memory.json from bundled seed if not present on volume."""
    if not Path(MEMORY_PATH).exists():
        seed = Path("/app/memory_seed.json")
        if seed.exists():
            shutil.copy2(seed, MEMORY_PATH)
            print("Seeded memory.json from bundled seed.")
        else:
            # Create minimal memory
            data = {
                "metadata": {"version": 3, "owner": "model"},
                "memories": [],
                "audit_log": [],
            }
            Path(MEMORY_PATH).write_text(json.dumps(data, indent=2))


# ----------------------------
# Agent runner (GPU)
# ----------------------------

@app.cls(
    image=ollama_image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=300,
    container_idle_timeout=120,
)
class Agent:
    @modal.enter()
    def startup(self):
        os.makedirs(MODELS_PATH, exist_ok=True)
        os.makedirs(TRAINING_OUTPUT_PATH, exist_ok=True)
        _ensure_memory()
        _start_ollama()
        _ensure_model()

    @modal.method()
    def run(self, user_text: str, model: str = "qwen3.5") -> Dict[str, Any]:
        """Run the agent and return the answer + updated memory state."""
        import sys
        sys.path.insert(0, "/app")

        # Override memory paths to use the volume
        import qwen3_5
        qwen3_5.MEMORY_FILE = Path(MEMORY_PATH)
        qwen3_5.MEMORY_BACKUP = Path(MEMORY_BACKUP_PATH)
        qwen3_5.ALLOWED_ROOT = Path(VOLUME_PATH)

        try:
            answer = qwen3_5.run_agent(model, user_text)
            # Read back memory for the response
            mem_data = json.loads(Path(MEMORY_PATH).read_text())
            volume.commit()
            return {
                "ok": True,
                "answer": answer,
                "memory_count": len(mem_data.get("memories", [])),
                "audit_count": len(mem_data.get("audit_log", [])),
            }
        except Exception as e:
            return {
                "ok": False,
                "answer": f"Error: {type(e).__name__}: {e}",
                "memory_count": 0,
                "audit_count": 0,
            }

    @modal.method()
    def get_memory(self) -> Dict[str, Any]:
        """Return the current memory state."""
        if Path(MEMORY_PATH).exists():
            data = json.loads(Path(MEMORY_PATH).read_text())
            volume.commit()
            return data
        return {"memories": [], "audit_log": [], "metadata": {}}

    @modal.method()
    def restore_memory(self) -> str:
        """Restore memory from backup."""
        if Path(MEMORY_BACKUP_PATH).exists():
            shutil.copy2(MEMORY_BACKUP_PATH, MEMORY_PATH)
            volume.commit()
            return "Restored from backup."
        return "No backup found."


# ----------------------------
# Real GPU training (Modal)
# ----------------------------

@app.cls(
    image=training_image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
class Trainer:
    @modal.method()
    def run_lora(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run real LoRA/QLoRA/DoRA fine-tuning on Modal GPU."""
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer as HFTrainer
        from datasets import load_dataset

        base_model = config.get("base_model", "Qwen/Qwen2.5-0.5B")
        dataset_name = config.get("dataset", "tatsu-lab/alpaca")
        lora = config.get("lora_config", {})
        output_dir = f"{TRAINING_OUTPUT_PATH}/lora-{int(time.time())}"

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantize_bits = lora.get("quantize_bits")
        load_kwargs = {"trust_remote_code": True}
        if quantize_bits == 4:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
            )
        elif quantize_bits == 8:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

        peft_config = PeftLoraConfig(
            r=lora.get("rank", 16),
            lora_alpha=lora.get("alpha", 32),
            target_modules=lora.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora.get("dropout", 0.05),
            task_type=TaskType.CAUSAL_LM,
            use_dora=lora.get("use_dora", False),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        ds = load_dataset(dataset_name, split="train[:1000]")

        def tokenize(example):
            text = example.get("text", example.get("instruction", ""))
            return tokenizer(text, truncation=True, max_length=512, padding="max_length")

        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        ds.set_format("torch")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
        )

        trainer = HFTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
        )
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        volume.commit()

        return {
            "ok": True,
            "output_dir": output_dir,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
        }

    @modal.method()
    def run_dpo(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run real DPO training on Modal GPU."""
        from trl import DPOTrainer, DPOConfig as TRLDPOConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        base_model = config.get("base_model", "Qwen/Qwen2.5-0.5B")
        dataset_name = config.get("dataset", "Anthropic/hh-rlhf")
        dpo = config.get("dpo_config", {})
        output_dir = f"{TRAINING_OUTPUT_PATH}/dpo-{int(time.time())}"

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)

        ds = load_dataset(dataset_name, split="train[:500]")

        training_args = TRLDPOConfig(
            output_dir=output_dir,
            beta=dpo.get("beta", 0.1),
            learning_rate=dpo.get("learning_rate", 5e-7),
            num_train_epochs=dpo.get("epochs", 1),
            per_device_train_batch_size=dpo.get("batch_size", 2),
            logging_steps=10,
            fp16=True,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(output_dir)
        volume.commit()

        return {
            "ok": True,
            "output_dir": output_dir,
        }


# ----------------------------
# Web UI (FastAPI)
# ----------------------------

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.115.0", "uvicorn>=0.30.0")
)

@app.function(
    image=web_image,
    volumes={VOLUME_PATH: volume},
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel as PydanticBaseModel

    api = FastAPI(title="Qwen Agent")

    class ChatRequest(PydanticBaseModel):
        message: str
        model: str = "qwen3.5"

    @api.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_UI

    @api.post("/api/chat")
    async def chat(req: ChatRequest):
        agent = Agent()
        result = agent.run.remote(req.message, req.model)
        return JSONResponse(result)

    @api.get("/api/memory")
    async def memory():
        agent = Agent()
        data = agent.get_memory.remote()
        return JSONResponse(data)

    @api.post("/api/memory/restore")
    async def restore():
        agent = Agent()
        msg = agent.restore_memory.remote()
        return JSONResponse({"message": msg})

    @api.post("/api/train")
    async def train(req: dict):
        trainer = Trainer()
        op = req.get("op", "lora")
        if op in ("lora", "qlora", "dora"):
            result = trainer.run_lora.remote(req)
        elif op == "dpo":
            result = trainer.run_dpo.remote(req)
        else:
            return JSONResponse({"ok": False, "error": f"Training op '{op}' not yet wired to Modal GPU."})
        return JSONResponse(result)

    return api


# ----------------------------
# HTML UI
# ----------------------------

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen Agent</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --text: #e0e0e8;
    --text-dim: #8888a0;
    --accent: #7c6ff0;
    --accent-glow: rgba(124, 111, 240, 0.15);
    --user-bg: #1a1a2e;
    --agent-bg: #0f1a1a;
    --agent-border: #1a3a3a;
    --red: #f05050;
    --green: #50f080;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
  }
  header h1 {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  header .status {
    font-size: 11px;
    color: var(--text-dim);
  }
  header .status .dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    margin-right: 6px;
  }
  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .tab {
    padding: 10px 20px;
    font-size: 12px;
    color: var(--text-dim);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }
  .tab:hover { color: var(--text); }
  .tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .panel { display: none; flex: 1; overflow: hidden; flex-direction: column; }
  .panel.active { display: flex; }

  /* Chat */
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
  }
  .msg {
    max-width: 720px;
    margin: 0 auto 16px;
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    background: var(--user-bg);
    border: 1px solid var(--border);
  }
  .msg.agent {
    background: var(--agent-bg);
    border: 1px solid var(--agent-border);
  }
  .msg .label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 8px;
  }
  .msg.agent .label { color: var(--accent); }
  .msg.error { border-color: var(--red); }

  #input-bar {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    gap: 12px;
    align-items: center;
  }
  #input-bar input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
    outline: none;
  }
  #input-bar input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
  #input-bar button {
    padding: 12px 24px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: inherit;
    font-size: 13px;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  #input-bar button:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Memory panel */
  #memory-view {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    font-size: 12px;
  }
  .mem-entry {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    margin-bottom: 12px;
    max-width: 720px;
    margin-left: auto;
    margin-right: auto;
  }
  .mem-entry .idx { color: var(--accent); font-weight: bold; }
  .mem-entry .insight { margin-top: 6px; }
  .mem-entry .meta { color: var(--text-dim); font-size: 11px; margin-top: 6px; }

  /* Audit panel */
  #audit-view {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    font-size: 11px;
  }
  .audit-entry {
    max-width: 720px;
    margin: 0 auto 8px;
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    display: flex;
    gap: 16px;
    align-items: baseline;
  }
  .audit-entry .ts { color: var(--text-dim); min-width: 200px; }
  .audit-entry .ev { color: var(--accent); min-width: 100px; font-weight: 600; }
  .audit-entry .actor { color: var(--green); min-width: 80px; }
  .audit-entry .desc { color: var(--text); }

  .spinner { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>QWEN AGENT <span style="color:var(--text-dim);font-weight:400">// provable finite tool algebra</span></h1>
  <div class="status"><span class="dot"></span>connected</div>
</header>

<div class="tabs">
  <div class="tab active" onclick="switchTab('chat')">Chat</div>
  <div class="tab" onclick="switchTab('memory')">Memory</div>
  <div class="tab" onclick="switchTab('audit')">Audit Log</div>
</div>

<div id="chat-panel" class="panel active">
  <div id="messages"></div>
  <div id="input-bar">
    <input id="user-input" type="text" placeholder="Talk to Qwen..." autocomplete="off" />
    <button id="send-btn" onclick="sendMessage()">Send</button>
  </div>
</div>

<div id="memory-panel" class="panel">
  <div id="memory-view">Loading...</div>
</div>

<div id="audit-panel" class="panel">
  <div id="audit-view">Loading...</div>
</div>

<script>
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(name + '-panel').classList.add('active');
  if (name === 'memory' || name === 'audit') loadMemory();
}

function addMsg(role, text, isError) {
  const d = document.createElement('div');
  d.className = 'msg ' + role + (isError ? ' error' : '');
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = role === 'user' ? 'You' : 'Qwen';
  d.appendChild(label);
  const body = document.createElement('div');
  body.textContent = text;
  d.appendChild(body);
  document.getElementById('messages').appendChild(d);
  d.scrollIntoView({ behavior: 'smooth' });
}

async function sendMessage() {
  const input = document.getElementById('user-input');
  const btn = document.getElementById('send-btn');
  const msg = input.value.trim();
  if (!msg) return;

  addMsg('user', msg);
  input.value = '';
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner">&#9697;</span>';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg}),
    });
    const data = await res.json();
    addMsg('agent', data.answer, !data.ok);
  } catch (e) {
    addMsg('agent', 'Network error: ' + e.message, true);
  }
  btn.disabled = false;
  btn.textContent = 'Send';
}

document.getElementById('user-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function loadMemory() {
  try {
    const res = await fetch('/api/memory');
    const data = await res.json();

    // Render memories
    const memView = document.getElementById('memory-view');
    if (data.memories && data.memories.length) {
      memView.innerHTML = data.memories.map((m, i) =>
        `<div class="mem-entry">
          <span class="idx">[${i}]</span>
          <span style="color:var(--text-dim);margin-left:8px;">${m.source || 'unknown'}</span>
          <div class="insight">${m.key_insight}</div>
          <div class="meta">relevance: ${m.relevance} | age: ${m.session_age} sessions | ttl: ${m.forget_after ?? '∞'} | created: ${m.created}</div>
        </div>`
      ).join('');
    } else {
      memView.innerHTML = '<div style="color:var(--text-dim);text-align:center;padding:40px;">No memories yet.</div>';
    }

    // Render audit
    const auditView = document.getElementById('audit-view');
    if (data.audit_log && data.audit_log.length) {
      auditView.innerHTML = data.audit_log.map(a =>
        `<div class="audit-entry">
          <span class="ts">${a.timestamp}</span>
          <span class="ev">${a.event}</span>
          <span class="actor">${a.actor}</span>
          <span class="desc">${a.description}</span>
        </div>`
      ).join('');
    } else {
      auditView.innerHTML = '<div style="color:var(--text-dim);text-align:center;padding:40px;">No audit entries.</div>';
    }
  } catch (e) {
    document.getElementById('memory-view').innerHTML = 'Error loading memory: ' + e.message;
  }
}
</script>
</body>
</html>"""
