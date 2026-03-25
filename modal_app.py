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
from typing import Any, Dict, List

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
CONVERSATION_LOG_PATH = f"{VOLUME_PATH}/conversations.json"

# Base image with Ollama + Python deps + pre-pulled model
MODEL_NAME = "qwen3:1.7b"

ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "procps", "zstd")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
    )
    .pip_install(
        "ollama>=0.4.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "starlette>=0.40.0",
    )
    .env({"OLLAMA_MODELS": "/ollama_models"})
    .run_commands(
        # Pre-pull the model at image build time so cold starts are instant
        "ollama serve & sleep 5 && ollama pull qwen3:1.7b; kill %1 2>/dev/null; true",
    )
    # Local files MUST be last (Modal requirement)
    .add_local_file("qwen3_5.py", "/app/qwen3_5.py")
    .add_local_file("training_algebra.py", "/app/training_algebra.py")
    .add_local_file("memory.json", "/app/memory_seed.json")
    .add_local_file("braille_stream.py", "/app/braille_stream.py")
    .add_local_file("infinite_algebra.py", "/app/infinite_algebra.py")
    .add_local_file("braille_algebra.py", "/app/braille_algebra.py")
)

# Training image with HuggingFace stack (heavier, used only for train ops)
# Built lazily — only instantiated when a training op is actually called
DEFAULT_TRAIN_MODEL = "Qwen/Qwen2.5-0.5B"

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.45.2",
        "peft==0.13.2",
        "trl==0.12.2",
        "datasets==3.0.1",
        "bitsandbytes==0.44.1",
        "accelerate==1.0.1",
        "pydantic>=2.0.0",
        "starlette>=0.40.0",
        "requests>=2.31.0",
        "ollama>=0.4.0",
    )
    .add_local_file("training_algebra.py", "/app/training_algebra.py", copy=True)
    .add_local_file("metric_algebra.py", "/app/metric_algebra.py", copy=True)
    .add_local_file("qwen3_5.py", "/app/qwen3_5.py", copy=True)
    .add_local_file("braille_algebra.py", "/app/braille_algebra.py", copy=True)
    .add_local_file("braille_stream.py", "/app/braille_stream.py", copy=True)
    .add_local_file("infinite_algebra.py", "/app/infinite_algebra.py", copy=True)
    .add_local_file("memory.json", "/app/memory_seed.json", copy=True)
    # Pre-cache model weights at image build time (eliminates ~60s cold-start download)
    .run_commands(
        f"python3 -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
        f"AutoTokenizer.from_pretrained('{DEFAULT_TRAIN_MODEL}', trust_remote_code=True); "
        f"AutoModelForCausalLM.from_pretrained('{DEFAULT_TRAIN_MODEL}', trust_remote_code=True)\""
    )
)


# ----------------------------
# Ollama lifecycle helpers
# ----------------------------

def _start_ollama():
    """Start Ollama server in background and wait until ready."""
    env = os.environ.copy()
    # Use baked-in models from /ollama_models, fall back to volume
    env["OLLAMA_MODELS"] = "/ollama_models"
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


def _ensure_model(model_name: str = MODEL_NAME):
    """Check model is available — should already be baked into the image."""
    import requests as req
    resp = req.get("http://localhost:11434/api/tags", timeout=10)
    models = [m["name"] for m in resp.json().get("models", [])]
    if not any(model_name in m for m in models):
        print(f"Model {model_name} not baked in, pulling...")
        subprocess.run(["ollama", "pull", model_name], check=True,
                        env={**os.environ, "OLLAMA_MODELS": "/ollama_models"})
        print(f"{model_name} pulled successfully.")
    else:
        print(f"Model {model_name} ready (baked into image).")


def _ensure_memory():
    """Seed memory.json from bundled seed if not present or empty on volume."""
    needs_seed = False
    if not Path(MEMORY_PATH).exists():
        needs_seed = True
    else:
        try:
            existing = json.loads(Path(MEMORY_PATH).read_text())
            if not existing.get("memories"):
                needs_seed = True
                print("Existing memory.json has no memories, re-seeding.")
        except Exception:
            needs_seed = True

    if needs_seed:
        seed = Path("/app/memory_seed.json")
        if seed.exists():
            shutil.copy2(seed, MEMORY_PATH)
            print("Seeded memory.json from bundled seed.")
            volume.commit()
        else:
            data = {
                "metadata": {"version": 3, "owner": "model"},
                "memories": [],
                "audit_log": [],
            }
            Path(MEMORY_PATH).write_text(json.dumps(data, indent=2))
            volume.commit()


def _append_conversation(timestamp: str, user_text: str, model: str, result: Dict[str, Any]):
    """Append a request/response pair to conversations.json on the volume."""
    log_path = Path(CONVERSATION_LOG_PATH)
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text())
        except Exception:
            log = []
    else:
        log = []

    log.append({
        "timestamp": timestamp,
        "request": user_text,
        "model": model,
        "ok": result.get("ok", False),
        "response": result.get("answer", ""),
        "memory_count": result.get("memory_count", 0),
        "audit_count": result.get("audit_count", 0),
    })
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))


# ----------------------------
# Agent runner (GPU)
# ----------------------------

@app.cls(
    image=ollama_image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("openrouter-key")],
    timeout=300,
    scaledown_window=300,
    min_containers=1,
)
class Agent:
    @modal.enter()
    def startup(self):
        os.makedirs(MODELS_PATH, exist_ok=True)
        os.makedirs(TRAINING_OUTPUT_PATH, exist_ok=True)
        _ensure_memory()
        _start_ollama()
        _ensure_model()

    def _init_qwen(self):
        """Common setup for agent methods."""
        import sys
        sys.path.insert(0, "/app")
        import qwen3_5
        qwen3_5.MEMORY_FILE = Path(MEMORY_PATH)
        qwen3_5.MEMORY_BACKUP = Path(MEMORY_BACKUP_PATH)
        qwen3_5.ALLOWED_ROOT = Path(VOLUME_PATH)
        return qwen3_5

    @modal.method()
    def run(self, user_text: str, model: str = MODEL_NAME) -> Dict[str, Any]:
        """Run the agent and return the answer + updated memory state."""
        from datetime import datetime, timezone
        qwen3_5 = self._init_qwen()

        ts = datetime.now(timezone.utc).isoformat()
        try:
            answer = qwen3_5.run_agent(model, user_text)
            mem_data = json.loads(Path(MEMORY_PATH).read_text())
            result = {
                "ok": True,
                "answer": answer,
                "memory_count": len(mem_data.get("memories", [])),
                "audit_count": len(mem_data.get("audit_log", [])),
            }
        except Exception as e:
            result = {
                "ok": False,
                "answer": f"Error: {type(e).__name__}: {e}",
                "memory_count": 0,
                "audit_count": 0,
            }

        _append_conversation(ts, user_text, model, result)
        volume.commit()
        return result

    @modal.method()
    def run_stream(self, user_text: str, model: str = MODEL_NAME):
        """Streaming agent — planner on local Qwen, answerer streams via OpenRouter."""
        from datetime import datetime, timezone
        qwen3_5 = self._init_qwen()
        volume.reload()

        ts = datetime.now(timezone.utc).isoformat()
        full_answer = ""
        try:
            # Phase 1: Planner (local Qwen, fast JSON routing)
            persistent = qwen3_5.PersistentMemory.load()
            print(f"[stream] Loaded {len(persistent.memories)} memories from {qwen3_5.MEMORY_FILE}", flush=True)
            persistent.age_memories()
            model_memories = persistent.to_context()
            trace = qwen3_5.Trace.identity()

            plan = None
            result = None
            for attempt in range(3):
                try:
                    plan = qwen3_5.plan_once(model, user_text, model_memories)
                    break
                except qwen3_5.ValidationError:
                    if attempt == 2:
                        raise

            # Phase 2: Execute or stream answer
            if plan.step.op == qwen3_5.Op.ANSWER_DIRECT:
                # Stream the answer via OpenRouter (or Ollama fallback)
                for chunk in qwen3_5.swarm_answer_stream(model, user_text, persistent):
                    full_answer += chunk
                    yield chunk
            else:
                # Tool execution — not streamed, return result at once
                result = qwen3_5.execute_step(plan.step)
                entry = qwen3_5.TraceEntry(
                    step=plan.step.model_dump(),
                    result=qwen3_5.asdict(result),
                )
                trace = trace.compose(qwen3_5.Trace(entries=[entry]))
                full_answer = qwen3_5.final_answer(model, user_text, plan, result)
                yield full_answer

            # Phase 3: Memory update (local Qwen, best-effort, not streamed)
            try:
                mem_update = qwen3_5.generate_memory_update(
                    model, user_text, full_answer, persistent
                )
                persistent.apply_update(mem_update)
            except Exception as e:
                print(f"[memory] {e}", file=__import__('sys').stderr)
            persistent.save()

            # Log conversation
            mem_data = json.loads(Path(MEMORY_PATH).read_text())
            log_result = {
                "ok": True, "answer": full_answer,
                "memory_count": len(mem_data.get("memories", [])),
                "audit_count": len(mem_data.get("audit_log", [])),
            }
        except Exception as e:
            err_msg = f"Error: {type(e).__name__}: {e}"
            if not full_answer:
                yield err_msg
                full_answer = err_msg
            log_result = {"ok": False, "answer": full_answer, "memory_count": 0, "audit_count": 0}

        _append_conversation(ts, user_text, model, log_result)
        volume.commit()

    @modal.method()
    def get_memory(self) -> Dict[str, Any]:
        """Return the current memory state."""
        volume.reload()
        if Path(MEMORY_PATH).exists():
            data = json.loads(Path(MEMORY_PATH).read_text())
            return data
        return {"memories": [], "audit_log": [], "metadata": {}}

    @modal.method()
    def get_conversations(self) -> List:
        """Return the conversation log."""
        volume.reload()
        log_path = Path(CONVERSATION_LOG_PATH)
        if log_path.exists():
            try:
                return json.loads(log_path.read_text())
            except Exception:
                return []
        return []

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
    min_containers=0,  # Scale to zero when idle — saves ~$26/day on A10G
)
class Trainer:
    @modal.enter()
    def preload(self):
        """Load default model + tokenizer once at container startup."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._default_tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_TRAIN_MODEL, trust_remote_code=True
        )
        if self._default_tokenizer.pad_token is None:
            self._default_tokenizer.pad_token = self._default_tokenizer.eos_token
        self._default_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_TRAIN_MODEL, trust_remote_code=True
        )

    def _get_model_and_tokenizer(self, base_model: str, quantize_bits=None):
        """Return (model, tokenizer) — uses preloaded cache if base_model matches default."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if base_model == DEFAULT_TRAIN_MODEL and quantize_bits is None:
            # Clone from preloaded (avoids re-download)
            import copy
            return copy.deepcopy(self._default_model), self._default_tokenizer
        # Different model or quantized — load fresh
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = {"trust_remote_code": True}
        if quantize_bits == 4:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_quant_type="nf4",
            )
        elif quantize_bits == 8:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        return model, tokenizer

    @modal.method()
    def run_lora(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run real LoRA/QLoRA/DoRA fine-tuning on Modal GPU."""
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        from transformers import TrainingArguments, Trainer as HFTrainer
        from datasets import load_dataset

        base_model = config.get("base_model", DEFAULT_TRAIN_MODEL)
        dataset_name = config.get("dataset", "tatsu-lab/alpaca")
        lora = config.get("lora_config", {})
        output_dir = f"{TRAINING_OUTPUT_PATH}/lora-{int(time.time())}"

        model, tokenizer = self._get_model_and_tokenizer(
            base_model, quantize_bits=lora.get("quantize_bits")
        )

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
            out = tokenizer(text, truncation=True, max_length=512, padding="max_length")
            out["labels"] = out["input_ids"].copy()
            return out

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
        from datasets import load_dataset

        base_model = config.get("base_model", DEFAULT_TRAIN_MODEL)
        dataset_name = config.get("dataset", "Anthropic/hh-rlhf")
        dpo = config.get("dpo_config", {})
        output_dir = f"{TRAINING_OUTPUT_PATH}/dpo-{int(time.time())}"

        model, tokenizer = self._get_model_and_tokenizer(base_model)

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
# Metric Algebra Evaluator (GPU)
# ----------------------------

@app.cls(
    image=training_image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
class MetricEvaluator:
    @modal.method()
    def behavioral_distance(
        self, model_a: str, model_b: str, prompts: List[str] = None
    ) -> Dict[str, Any]:
        """Compute d(x,y) — behavioral pseudometric between two model states."""
        import sys
        sys.path.insert(0, "/app")
        from metric_algebra import compute_behavioral_distance, DEFAULT_CALIBRATION_PROMPTS
        prompts = prompts or DEFAULT_CALIBRATION_PROMPTS
        dist = compute_behavioral_distance(model_a, model_b, prompts)
        return {
            "kl_divergence": dist.kl_divergence,
            "symmetric_kl": dist.symmetric_kl,
            "cosine_distance": dist.cosine_distance,
            "l2_output_distance": dist.l2_output_distance,
            "num_samples": dist.num_samples,
            "primary": dist.primary,
        }

    @modal.method()
    def commutator_defect(
        self,
        base_model: str,
        op_a_name: str,
        op_a_config: Dict[str, Any],
        op_b_name: str,
        op_b_config: Dict[str, Any],
        prompts: List[str] = None,
    ) -> Dict[str, Any]:
        """Measure \U0001d520(A, B) = d(A\u2218B, B\u2218A) — commutator defect.

        Applies operator A and B in both orders to the base model,
        then measures behavioral distance between the two outcomes.
        """
        import sys, os, time as _t
        sys.path.insert(0, "/app")
        from metric_algebra import compute_behavioral_distance, DEFAULT_CALIBRATION_PROMPTS
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        prompts = prompts or DEFAULT_CALIBRATION_PROMPTS
        output_base = f"{TRAINING_OUTPUT_PATH}/metric-{int(_t.time())}"
        os.makedirs(output_base, exist_ok=True)

        def apply_op(model_path: str, op_name: str, config: dict, tag: str) -> str:
            """Apply a training operator and save the result."""
            out_dir = f"{output_base}/{tag}"
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if op_name in ("lora", "qlora", "dora"):
                load_kw = {"trust_remote_code": True, "torch_dtype": torch.float16}
                if op_name == "qlora":
                    from transformers import BitsAndBytesConfig
                    load_kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype="float16",
                        bnb_4bit_quant_type="nf4",
                    )
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
                rank = config.get("rank", 16)
                peft_cfg = PeftLoraConfig(
                    r=rank, lora_alpha=config.get("alpha", rank * 2),
                    target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
                    task_type=TaskType.CAUSAL_LM,
                    use_dora=(op_name == "dora"),
                )
                model = get_peft_model(model, peft_cfg)
                # Merge back for comparison (so output is a dense model)
                model = model.merge_and_unload()
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)

            elif op_name == "quantize":
                from transformers import BitsAndBytesConfig
                bits = config.get("bits", 4)
                qconfig = BitsAndBytesConfig(
                    load_in_4bit=(bits == 4), load_in_8bit=(bits == 8),
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, quantization_config=qconfig, trust_remote_code=True,
                )
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)

            elif op_name == "identity":
                # Just copy — the identity operator
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True, torch_dtype=torch.float16,
                )
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)

            else:
                raise ValueError(f"Metric evaluator: unsupported op '{op_name}'")

            del model
            torch.cuda.empty_cache()
            return out_dir

        # Path 1: A then B
        mid_a = apply_op(base_model, op_a_name, op_a_config, "path1_a")
        after_ab = apply_op(mid_a, op_b_name, op_b_config, "path1_ab")

        # Path 2: B then A
        mid_b = apply_op(base_model, op_b_name, op_b_config, "path2_b")
        after_ba = apply_op(mid_b, op_a_name, op_a_config, "path2_ba")

        # Measure d(AB, BA)
        dist = compute_behavioral_distance(after_ab, after_ba, prompts)

        volume.commit()
        return {
            "operator_a": op_a_name,
            "operator_b": op_b_name,
            "defect": dist.primary,
            "kl_divergence": dist.kl_divergence,
            "symmetric_kl": dist.symmetric_kl,
            "cosine_distance": dist.cosine_distance,
            "num_samples": dist.num_samples,
            "path_ab": after_ab,
            "path_ba": after_ba,
        }

    @modal.method()
    def idempotence_defect(
        self,
        base_model: str,
        op_name: str,
        op_config: Dict[str, Any],
        prompts: List[str] = None,
    ) -> Dict[str, Any]:
        """Measure \u03b4_idem(T) = d(T\u2218T, T) — idempotence defect."""
        import sys, os, time as _t
        sys.path.insert(0, "/app")
        from metric_algebra import compute_behavioral_distance, DEFAULT_CALIBRATION_PROMPTS
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        prompts = prompts or DEFAULT_CALIBRATION_PROMPTS
        output_base = f"{TRAINING_OUTPUT_PATH}/idem-{int(_t.time())}"
        os.makedirs(output_base, exist_ok=True)

        # This reuses the apply_op logic from commutator_defect
        # For brevity, we inline a simpler version for quantize (the key idempotence test)
        if op_name == "quantize":
            from transformers import BitsAndBytesConfig
            bits = op_config.get("bits", 4)
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Apply once
            qconfig = BitsAndBytesConfig(
                load_in_4bit=(bits == 4), load_in_8bit=(bits == 8),
            )
            model_once = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=qconfig, trust_remote_code=True,
            )
            once_dir = f"{output_base}/once"
            model_once.save_pretrained(once_dir)
            tokenizer.save_pretrained(once_dir)
            del model_once
            torch.cuda.empty_cache()

            # Apply twice (quantize the already-quantized)
            model_twice = AutoModelForCausalLM.from_pretrained(
                once_dir, quantization_config=qconfig, trust_remote_code=True,
            )
            twice_dir = f"{output_base}/twice"
            model_twice.save_pretrained(twice_dir)
            tokenizer.save_pretrained(twice_dir)
            del model_twice
            torch.cuda.empty_cache()

            dist = compute_behavioral_distance(once_dir, twice_dir, prompts)
        else:
            return {"error": f"Idempotence test not yet implemented for '{op_name}'"}

        volume.commit()
        return {
            "operator": op_name,
            "defect": dist.primary,
            "symmetric_kl": dist.symmetric_kl,
            "cosine_distance": dist.cosine_distance,
            "num_samples": dist.num_samples,
        }


# ----------------------------
# HTML UI (must be defined before web function)
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
  .braille-toggle {
    padding: 6px 14px;
    font-size: 11px;
    font-family: inherit;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.2s;
  }
  .braille-toggle.active {
    background: var(--accent-glow);
    border-color: var(--accent);
    color: var(--accent);
  }
  .braille-stats {
    font-size: 10px;
    color: var(--text-dim);
    text-align: center;
    padding: 4px;
    opacity: 0.7;
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
  <button class="braille-toggle" id="braille-toggle" onclick="toggleBraille()" title="Use Aria braille stream encoding (faster)">⠃ Braille Stream</button>
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
// ============================================================================
// Aria Braille Stream Decoder (ported from ai-native-ide/src/braille.js)
// ============================================================================
const BRAILLE_BASE = 0x2800;
let useBraille = false;

function toggleBraille() {
  useBraille = !useBraille;
  const btn = document.getElementById('braille-toggle');
  btn.classList.toggle('active', useBraille);
  btn.textContent = useBraille ? '\u2803 Braille ON' : '\u2803 Braille Stream';
}

function fromBraille(braille) {
  const bytes = [];
  for (const char of braille) {
    const cp = char.codePointAt(0);
    if (cp >= BRAILLE_BASE && cp <= BRAILLE_BASE + 255) {
      bytes.push(cp - BRAILLE_BASE);
    }
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

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

  // Create a placeholder message for streaming
  const d = document.createElement('div');
  d.className = 'msg agent';
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = 'Qwen';
  d.appendChild(label);
  const body = document.createElement('div');
  body.textContent = '';
  d.appendChild(body);
  document.getElementById('messages').appendChild(d);

  try {
    const res = await fetch('/api/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg, braille: useBraille}),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let brailleStats = null;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const payload = trimmed.slice(6);
        if (payload === '[DONE]') break;
        try {
          const chunk = JSON.parse(payload);
          if (chunk && chunk._stats) {
            brailleStats = chunk._stats;
            continue;
          }
          if (chunk && chunk.b) {
            // Braille-encoded chunk — decode to text
            body.textContent += fromBraille(chunk.b);
          } else {
            body.textContent += chunk;
          }
          d.scrollIntoView({behavior: 'smooth'});
        } catch (e) {}
      }
    }
    if (!body.textContent) {
      body.textContent = '(no response)';
      d.classList.add('error');
    }
    if (brailleStats) {
      const statsDiv = document.createElement('div');
      statsDiv.className = 'braille-stats';
      const ratio = (brailleStats.compression_ratio * 100).toFixed(0);
      statsDiv.textContent = `\u2803 braille stream: ${brailleStats.chunks} chunks, ${brailleStats.braille_out} cells, ${ratio}% ratio`;
      d.appendChild(statsDiv);
    }
  } catch (e) {
    body.textContent = 'Network error: ' + e.message;
    d.classList.add('error');
  }
  btn.disabled = false;
  btn.textContent = 'Send';
}

document.getElementById('user-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// Load conversation history on page load
async function loadConversations() {
  try {
    const res = await fetch('/api/conversations');
    const data = await res.json();
    if (data && data.length) {
      data.forEach(c => {
        addMsg('user', c.request);
        addMsg('agent', c.response, !c.ok);
      });
    }
  } catch (e) {
    console.log('Could not load conversation history:', e);
  }
}
loadConversations();

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


# ----------------------------
# Web UI — uses ollama_image so it has all code in scope
# Starlette imports are guarded so Trainer/MetricEvaluator containers
# (which use training_image) don't crash if starlette isn't installed.
# ----------------------------

try:
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse as StarletteHTML, JSONResponse as StarletteJSON, StreamingResponse
    from starlette.routing import Route
    _HAS_STARLETTE = True
except ImportError:
    _HAS_STARLETTE = False


async def _index(request):
    return StarletteHTML(HTML_UI)


async def _chat(request):
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        message = body.get("message", request.query_params.get("message", ""))
        model_name = body.get("model", request.query_params.get("model", MODEL_NAME))
    else:
        message = request.query_params.get("message", "")
        model_name = request.query_params.get("model", MODEL_NAME)
    agent = Agent()
    result = agent.run.remote(message, model_name)
    return StarletteJSON(result)


async def _chat_stream(request):
    """SSE streaming endpoint — planner on local Qwen, answerer streams via OpenRouter."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = body.get("message", "")
    model_name = body.get("model", MODEL_NAME)
    use_braille = body.get("braille", False)

    async def event_generator():
        import sys
        sys.path.insert(0, "/app")
        from braille_stream import BrailleStreamProcessor
        processor = BrailleStreamProcessor(use_contractions=True) if use_braille else None

        agent = Agent()
        for chunk in agent.run_stream.remote_gen(message, model_name):
            if processor:
                braille_chunk = processor.process_chunk(chunk)
                payload = json.dumps({"b": braille_chunk})
            else:
                payload = json.dumps(chunk)
            yield f"data: {payload}\n\n"

        if processor:
            stats = processor.get_stats()
            yield f"data: {json.dumps({'_stats': stats})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _get_memory(request):
    agent = Agent()
    data = agent.get_memory.remote()
    return StarletteJSON(data)


async def _restore_memory(request):
    agent = Agent()
    msg = agent.restore_memory.remote()
    return StarletteJSON({"message": msg})


async def _train(request):
    try:
        req = await request.json()
    except Exception:
        req = {}
    trainer = Trainer()
    op = req.get("op", "lora")
    try:
        if op in ("lora", "qlora", "dora"):
            result = await trainer.run_lora.remote.aio(req)
        elif op == "dpo":
            result = await trainer.run_dpo.remote.aio(req)
        else:
            return StarletteJSON({"ok": False, "error": f"Training op '{op}' not yet wired."})
        return StarletteJSON(result)
    except Exception as e:
        return StarletteJSON({"ok": False, "error": str(e)})


async def _get_conversations(request):
    agent = Agent()
    data = agent.get_conversations.remote()
    return StarletteJSON(data)


async def _algebra(request):
    """Braille algebra endpoint — returns the N×N grid as JSON or HTML.

    GET /api/algebra          → JSON grid data
    GET /api/algebra?html=1   → standalone HTML visualization
    POST /api/algebra         → add operator or set defect, return updated grid
    """
    import sys
    sys.path.insert(0, "/app")
    from braille_algebra import BrailleAlgebra, OperatorInfo, render_html, from_relation_table

    if request.method == "POST":
        try:
            req = await request.json()
        except Exception:
            req = {}

        action = req.get("action", "get")

        if action == "add_operator":
            alg = BrailleAlgebra()
            op = req.get("operator", {})
            alg.add_operator(OperatorInfo(**op))
            alg.save()
            return StarletteJSON(alg.to_dict())

        elif action == "set_defect":
            alg = BrailleAlgebra()
            alg.set_commutator_defect(req["a"], req["b"], req["defect"])
            alg.save()
            return StarletteJSON(alg.to_dict())

        elif action == "from_relation_table":
            alg = from_relation_table(
                operator_names=req["operator_names"],
                defects=req["defects"],
            )
            return StarletteJSON(alg.to_dict())

        return StarletteJSON({"error": f"Unknown action: {action}"})

    # GET
    params = dict(request.query_params)
    alg = BrailleAlgebra()

    if params.get("html") == "1":
        html = render_html(alg)
        from starlette.responses import HTMLResponse
        return HTMLResponse(html)

    return StarletteJSON(alg.to_dict())


async def _metric(request):
    """Metric algebra endpoint — compute behavioral distances and algebraic quantities.

    POST /api/metric with JSON body:
      {"experiment": "distance"|"commutator"|"idempotence",
       "model_a": "...", "model_b": "...",  // for distance
       "base_model": "...", "op_a": "...", "op_a_config": {...}, ...  // for commutator
      }
    """
    try:
        req = await request.json()
    except Exception:
        req = {}
    experiment = req.get("experiment", "distance")
    evaluator = MetricEvaluator()

    if experiment == "distance":
        result = evaluator.behavioral_distance.remote(
            model_a=req.get("model_a", "Qwen/Qwen2.5-0.5B"),
            model_b=req.get("model_b", "Qwen/Qwen2.5-0.5B"),
            prompts=req.get("prompts"),
        )
    elif experiment == "commutator":
        result = evaluator.commutator_defect.remote(
            base_model=req.get("base_model", "Qwen/Qwen2.5-0.5B"),
            op_a_name=req.get("op_a", "lora"),
            op_a_config=req.get("op_a_config", {}),
            op_b_name=req.get("op_b", "lora"),
            op_b_config=req.get("op_b_config", {}),
            prompts=req.get("prompts"),
        )
    elif experiment == "idempotence":
        result = evaluator.idempotence_defect.remote(
            base_model=req.get("base_model", "Qwen/Qwen2.5-0.5B"),
            op_name=req.get("op", "quantize"),
            op_config=req.get("op_config", {}),
            prompts=req.get("prompts"),
        )
    else:
        return StarletteJSON({"error": f"Unknown experiment: {experiment}"})

    return StarletteJSON(result)


if _HAS_STARLETTE:
    web_app = Starlette(routes=[
        Route("/", _index),
        Route("/api/chat", _chat, methods=["POST"]),
        Route("/api/stream", _chat_stream, methods=["POST"]),
        Route("/api/memory", _get_memory),
        Route("/api/memory/restore", _restore_memory, methods=["POST"]),
        Route("/api/conversations", _get_conversations),
        Route("/api/train", _train, methods=["POST"]),
        Route("/api/metric", _metric, methods=["POST"]),
        Route("/api/algebra", _algebra, methods=["GET", "POST"]),
    ])

    @app.function(
        image=ollama_image,
        volumes={VOLUME_PATH: volume},
    )
    @modal.asgi_app()
    def web():
        return web_app
