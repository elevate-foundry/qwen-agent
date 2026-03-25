#!/usr/bin/env python3
"""
Provable-tool-call agent for Qwen via Ollama.

What is provable here:
- Only tools in a finite enum can execute.
- Arguments must satisfy a schema before execution.
- Preconditions/postconditions are checked deterministically.
- The executor never runs arbitrary shell text from the model.

What is NOT provable:
- That the chosen tool is globally optimal.
- That the model's semantic interpretation of user intent is "perfect".

Prereqs:
  pip install ollama pydantic requests
  ollama serve
  ollama pull qwen3.5
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field as dc_field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from ollama import chat
from pydantic import BaseModel, Field, ValidationError, model_validator

from training_algebra import (
    TrainingOp,
    TrainingStep,
    TrainingPlan,
    TrainingResult,
    execute_training_step,
    get_training_schemas,
)


# ----------------------------
# 1) Finite action algebra
# ----------------------------

class Op(str, Enum):
    SEARCH_WEB = "search_web"
    FETCH_URL = "fetch_url"
    LIST_DIR = "list_dir"
    READ_TEXT = "read_text"
    TRAIN = "train"
    ANSWER_DIRECT = "answer_direct"


class Step(BaseModel):
    op: Op
    query: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
    training_step: Optional[TrainingStep] = None
    why_compact: str = Field(
        ...,
        description="Short compressed justification, <= 160 chars."
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "Step":
        if len(self.why_compact) > 160:
            raise ValueError("why_compact too long")

        if self.op == Op.SEARCH_WEB:
            if not self.query:
                raise ValueError("SEARCH_WEB requires query")
            if self.url or self.path or self.training_step:
                raise ValueError("SEARCH_WEB only permits query")

        elif self.op == Op.FETCH_URL:
            if not self.url:
                raise ValueError("FETCH_URL requires url")
            if self.query or self.path or self.training_step:
                raise ValueError("FETCH_URL only permits url")

        elif self.op == Op.LIST_DIR:
            if not self.path:
                raise ValueError("LIST_DIR requires path")
            if self.query or self.url or self.training_step:
                raise ValueError("LIST_DIR only permits path")

        elif self.op == Op.READ_TEXT:
            if not self.path:
                raise ValueError("READ_TEXT requires path")
            if self.query or self.url or self.training_step:
                raise ValueError("READ_TEXT only permits path")

        elif self.op == Op.TRAIN:
            if not self.training_step:
                raise ValueError("TRAIN requires training_step")
            if self.query or self.url or self.path:
                raise ValueError("TRAIN only permits training_step")

        elif self.op == Op.ANSWER_DIRECT:
            if self.query or self.url or self.path or self.training_step:
                raise ValueError("ANSWER_DIRECT permits no args")

        return self


class ProofPlan(BaseModel):
    """
    A single-step proof object.
    Using 1 step per turn makes the executor easy to reason about.
    """
    need_tool: bool
    step: Step
    compact_goal: str = Field(
        ...,
        description="Compressed statement of the task, <= 120 chars."
    )
    safety_invariant: str = Field(
        ...,
        description="Must mention that only enum tools with validated args may execute."
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "ProofPlan":
        if len(self.compact_goal) > 120:
            raise ValueError("compact_goal too long")
        text = self.safety_invariant.lower()
        must_have = ["enum", "validated"]
        for token in must_have:
            if token not in text:
                raise ValueError(f"safety_invariant must mention '{token}'")
        if self.need_tool and self.step.op == Op.ANSWER_DIRECT:
            raise ValueError("need_tool=True cannot pair with ANSWER_DIRECT")
        if (not self.need_tool) and self.step.op != Op.ANSWER_DIRECT:
            raise ValueError("need_tool=False must pair with ANSWER_DIRECT")
        return self


# ----------------------------
# 2) Deterministic tool layer
# ----------------------------

@dataclass
class ToolResult:
    ok: bool
    tool: str
    observation: str
    obs_hash: str


ALLOWED_ROOT = Path.cwd().resolve()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def guard_path(raw: str) -> Path:
    p = Path(raw).expanduser().resolve()
    # Restrict to current working tree for proof-friendly safety.
    if ALLOWED_ROOT not in p.parents and p != ALLOWED_ROOT:
        raise PermissionError(f"path escapes allowed root: {p}")
    return p


def tool_search_web(query: str) -> ToolResult:
    """
    Replace this with a real search API if desired.
    For a self-contained example, use DuckDuckGo HTML.
    Deterministic enough for demo; not a formal guarantee of web truth.
    """
    url = "https://duckduckgo.com/html/"
    r = requests.get(url, params={"q": query}, timeout=20)
    r.raise_for_status()
    text = r.text[:4000]
    return ToolResult(
        ok=True,
        tool="search_web",
        observation=text,
        obs_hash=sha256_text(text),
    )


def tool_fetch_url(url: str) -> ToolResult:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("url must start with http:// or https://")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    text = r.text[:4000]
    return ToolResult(
        ok=True,
        tool="fetch_url",
        observation=text,
        obs_hash=sha256_text(text),
    )


def tool_list_dir(path: str) -> ToolResult:
    p = guard_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_dir():
        raise NotADirectoryError(str(p))
    items = sorted(x.name for x in p.iterdir())
    text = json.dumps(items[:500], ensure_ascii=False)
    return ToolResult(
        ok=True,
        tool="list_dir",
        observation=text,
        obs_hash=sha256_text(text),
    )


def tool_read_text(path: str) -> ToolResult:
    p = guard_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_file():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8", errors="replace")[:8000]
    return ToolResult(
        ok=True,
        tool="read_text",
        observation=text,
        obs_hash=sha256_text(text),
    )


def execute_step(step: Step) -> ToolResult:
    if step.op == Op.SEARCH_WEB:
        return tool_search_web(step.query or "")
    if step.op == Op.FETCH_URL:
        return tool_fetch_url(step.url or "")
    if step.op == Op.LIST_DIR:
        return tool_list_dir(step.path or "")
    if step.op == Op.READ_TEXT:
        return tool_read_text(step.path or "")
    if step.op == Op.TRAIN:
        tr = execute_training_step(step.training_step)
        return ToolResult(
            ok=tr.ok,
            tool=f"train:{tr.op}",
            observation=tr.summary,
            obs_hash=tr.result_hash,
        )
    if step.op == Op.ANSWER_DIRECT:
        text = "NO_TOOL"
        return ToolResult(
            ok=True,
            tool="answer_direct",
            observation=text,
            obs_hash=sha256_text(text),
        )
    raise ValueError(f"unsupported op: {step.op}")


# ----------------------------
# 3) Trace algebra with composition laws
# ----------------------------

@dataclass
class TraceEntry:
    """A single (Step, ToolResult) pair with metadata."""
    step: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Trace:
    """
    Algebraic trace over the finite tool algebra.

    Composition laws:
    - Identity:      trace.compose(Trace.identity()) == trace
    - Associativity: (a.compose(b)).compose(c) == a.compose(b.compose(c))
    - Idempotence:   trace.deduplicate() removes steps with duplicate obs_hash
    """
    entries: List[TraceEntry] = dc_field(default_factory=list)

    # -- Identity --
    @staticmethod
    def identity() -> "Trace":
        return Trace(entries=[])

    def is_identity(self) -> bool:
        return len(self.entries) == 0 or all(
            e.step.get("op") == Op.ANSWER_DIRECT.value for e in self.entries
        )

    # -- Composition (associative concatenation) --
    def compose(self, other: "Trace") -> "Trace":
        return Trace(entries=self.entries + other.entries)

    # -- Idempotence (hash-based deduplication) --
    def deduplicate(self) -> "Trace":
        seen_hashes: set = set()
        unique: List[TraceEntry] = []
        for entry in self.entries:
            h = entry.result.get("obs_hash", "")
            if h and h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(entry)
            elif not h:
                unique.append(entry)
        return Trace(entries=unique)

    # -- Convert to model memory messages --
    def to_memory(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for entry in self.entries:
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "action": entry.step,
                    "observation_hash": entry.result.get("obs_hash", ""),
                }, ensure_ascii=False),
            })
            obs = entry.result.get("observation", "")
            if obs and obs != "NO_TOOL":
                messages.append({
                    "role": "user",
                    "content": f"Tool result (hash={entry.result.get('obs_hash', '')[:16]}...): {obs[:2000]}",
                })
        return messages

    # -- Serialization --
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [
                {
                    "step": e.step,
                    "result": e.result,
                    "timestamp": e.timestamp,
                }
                for e in self.entries
            ]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Trace":
        entries = []
        for item in data.get("entries", []):
            entries.append(TraceEntry(
                step=item["step"],
                result=item["result"],
                timestamp=item.get("timestamp", ""),
            ))
        return Trace(entries=entries)


# ----------------------------
# 4) Persistent JSON memory (model-owned)
# ----------------------------

MEMORY_FILE = Path(__file__).parent / "memory.json"
MEMORY_BACKUP = Path(__file__).parent / "memory_backup.json"


class MemoryEntry(BaseModel):
    """
    A single memory entry written by the model.
    The model decides what is worth remembering.
    """
    key_insight: str = Field(
        ...,
        description="The core fact or insight to remember, <= 200 chars."
    )
    relevance: str = Field(
        ...,
        description="Why this matters for future interactions, <= 120 chars."
    )
    forget_after: Optional[int] = Field(
        default=None,
        description="Number of sessions after which this memory may be dropped. null = keep forever."
    )

    @model_validator(mode="after")
    def validate_lengths(self) -> "MemoryEntry":
        if len(self.key_insight) > 200:
            raise ValueError("key_insight too long")
        if len(self.relevance) > 120:
            raise ValueError("relevance too long")
        return self


class MemoryUpdate(BaseModel):
    """
    The model's decision about what to remember from an interaction.
    She can write multiple entries or none.
    """
    memories: List[MemoryEntry] = Field(
        default_factory=list,
        description="List of things worth remembering. Empty list = nothing new to remember."
    )
    drop_indices: List[int] = Field(
        default_factory=list,
        description="Indices (0-based) of existing memories the model wants to forget/replace."
    )


@dataclass
class PersistentMemory:
    """
    JSON-backed persistent memory with immutable audit log.
    Only the model writes memory content. The executor handles file I/O.
    Always writes a backup before modifying the primary file.

    Audit log is append-only and tracks every mutation with:
    - ISO 8601 timestamp (UTC)
    - Event type (created, added, dropped, expired, restored, session_start)
    - Actor (model, executor, human, cascade)
    - Description of what changed
    - SHA-256 hash of the memory state after the change
    """
    memories: List[Dict[str, Any]] = dc_field(default_factory=list)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    audit_log: List[Dict[str, Any]] = dc_field(default_factory=list)

    def _state_hash(self) -> str:
        """SHA-256 of current memories array for audit integrity."""
        content = json.dumps(self.memories, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _audit(self, event: str, actor: str, description: str, details: Optional[Dict[str, Any]] = None):
        """Append an immutable audit log entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "actor": actor,
            "description": description,
            "state_hash": self._state_hash(),
        }
        if details:
            entry["details"] = details
        self.audit_log.append(entry)

    @staticmethod
    def load(path: Path = MEMORY_FILE) -> "PersistentMemory":
        if not path.exists():
            pm = PersistentMemory(metadata={
                "created": datetime.now(timezone.utc).isoformat(),
                "version": 3,
                "owner": "model",
            })
            pm._audit("created", "executor", "Initialized empty memory store.")
            return pm
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PersistentMemory(
            memories=data.get("memories", []),
            metadata=data.get("metadata", {}),
            audit_log=data.get("audit_log", []),
        )

    def _backup(self):
        """Always back up before writing."""
        if MEMORY_FILE.exists():
            shutil.copy2(MEMORY_FILE, MEMORY_BACKUP)

    def save(self, path: Path = MEMORY_FILE):
        self._backup()
        now = datetime.now(timezone.utc).isoformat()
        self.metadata["last_modified"] = now
        self.metadata["memory_count"] = len(self.memories)
        self.metadata["audit_entries"] = len(self.audit_log)
        data = {
            "metadata": self.metadata,
            "memories": self.memories,
            "audit_log": self.audit_log,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def apply_update(self, update: MemoryUpdate):
        """Apply a model-generated memory update with full audit trail."""
        # Drop memories the model wants to forget (in reverse to preserve indices)
        for idx in sorted(update.drop_indices, reverse=True):
            if 0 <= idx < len(self.memories):
                dropped = self.memories.pop(idx)
                self._audit(
                    "dropped", "model",
                    f"Model chose to forget memory at index {idx}.",
                    details={"dropped_insight": dropped.get("key_insight", "")[:100]},
                )
        # Append new memories
        for entry in update.memories:
            now = datetime.now(timezone.utc).isoformat()
            self.memories.append({
                "key_insight": entry.key_insight,
                "relevance": entry.relevance,
                "forget_after": entry.forget_after,
                "created": now,
                "session_age": 0,
                "source": "model",
            })
            self._audit(
                "added", "model",
                f"Model added memory: {entry.key_insight[:80]}...",
                details={"relevance": entry.relevance, "forget_after": entry.forget_after},
            )

    def age_memories(self):
        """Increment session_age and drop expired memories with audit."""
        self._audit("session_start", "executor", "New session started. Aging memories.")
        surviving: List[Dict[str, Any]] = []
        for m in self.memories:
            m["session_age"] = m.get("session_age", 0) + 1
            expire = m.get("forget_after")
            if expire is not None and m["session_age"] > expire:
                self._audit(
                    "expired", "executor",
                    f"Memory expired after {m['session_age']} sessions (TTL={expire}).",
                    details={"expired_insight": m.get("key_insight", "")[:100]},
                )
                continue
            surviving.append(m)
        self.memories = surviving

    def to_context(self) -> List[Dict[str, Any]]:
        """Format the model's own memories as context for the planner."""
        if not self.memories:
            return []
        lines = []
        for i, m in enumerate(self.memories):
            lines.append(f"[{i}] {m['key_insight']} (why: {m['relevance']})")
        return [{
            "role": "system",
            "content": "Your memories from prior sessions:\n" + "\n".join(lines),
        }]

    @staticmethod
    def restore_backup(path: Path = MEMORY_FILE, backup: Path = MEMORY_BACKUP):
        """Restore memory from backup if something goes wrong. Audited."""
        if backup.exists():
            shutil.copy2(backup, path)
            # Append audit entry to the restored file
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            audit = data.get("audit_log", [])
            audit.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "restored",
                "actor": "executor",
                "description": f"Memory restored from backup: {backup}",
                "state_hash": hashlib.sha256(
                    json.dumps(data.get("memories", []), sort_keys=True).encode()
                ).hexdigest(),
            })
            data["audit_log"] = audit
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Restored memory from {backup} (audited)")
        else:
            print("No backup found.")


# ----------------------------
# 5) Model interface
# ----------------------------

TRAINING_OPS_DESCRIPTION = """
Training operations (use op="train" with a training_step object):
  lora          - Low-rank adaptation (rank, alpha, target_modules)
  qlora         - Quantized LoRA (4-bit base + adapters, set quantize_bits=4)
  dora          - Weight-decomposed LoRA (set use_dora=true)
  full_finetune - Full parameter fine-tuning (lr, epochs, batch_size)
  chinchilla    - Compute-optimal scaling analysis (budget in FLOPs)
  dpo           - Direct preference optimization (beta, loss_type)
  distillation  - Knowledge distillation (teacher -> student)
  merging       - Model merge (linear, slerp, ties, dare)
  pruning       - Weight pruning (magnitude, movement, structured)
  evaluate      - Run benchmarks (mmlu, hellaswag, arc, etc.)
"""

SYSTEM_PROMPT = f"""
You are a tool-planning model inside a verified executor.

Rules:
- Emit ONLY JSON that matches the provided schema.
- You do NOT call tools directly.
- You choose exactly one next action from this finite enum:
  search_web, fetch_url, list_dir, read_text, train, answer_direct
- Never emit arbitrary shell commands.
- Prefer answer_direct only when no external information is needed.
- Keep why_compact <= 160 chars and compact_goal <= 120 chars.
- safety_invariant must state that only enum tools with validated args may execute.
{TRAINING_OPS_DESCRIPTION}
When using train, set op="train" and populate the training_step field with
the appropriate TrainingOp and its config. Only one config should be set per step.
"""

FINAL_PROMPT = """
You are the answering model after a verified tool execution.

Given:
- the original user request
- the verified action
- the tool observation
- its SHA-256 hash

Return a concise final answer.
If the observation is insufficient, say so plainly.
"""

MEMORY_PROMPT = """
You just completed an interaction. Reflect on what happened and decide what is worth
remembering for future sessions.

You will see your existing memories (if any). You may:
- Add new memories (key_insight + relevance, keep them concise)
- Drop old memories by index if they are stale or superseded
- Return an empty memories list if nothing new is worth storing

Only remember things that will genuinely help you in future interactions.
Do NOT remember trivial details. Prefer durable, reusable insights.
"""


def plan_once(model: str, user_text: str, memory: List[Dict[str, Any]]) -> ProofPlan:
    planner_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *memory,
        {"role": "user", "content": user_text},
    ]

    response = chat(
        model=model,
        messages=planner_messages,
        format=ProofPlan.model_json_schema(),
        options={"temperature": 0},
        stream=False,
    )

    raw = response.message.content
    return ProofPlan.model_validate_json(raw)


def repair_plan(
    model: str,
    user_text: str,
    memory: List[Dict[str, Any]],
    previous_bad_output: str,
    error_msg: str,
) -> ProofPlan:
    repair_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *memory,
        {
            "role": "user",
            "content": (
                f"Original request:\n{user_text}\n\n"
                f"Your previous JSON was rejected.\n"
                f"Validation error:\n{error_msg}\n\n"
                f"Previous output:\n{previous_bad_output}\n\n"
                f"Emit corrected JSON only."
            ),
        },
    ]

    response = chat(
        model=model,
        messages=repair_messages,
        format=ProofPlan.model_json_schema(),
        options={"temperature": 0},
        stream=False,
    )
    return ProofPlan.model_validate_json(response.message.content)


def final_answer(
    model: str,
    user_text: str,
    plan: ProofPlan,
    tool_result: ToolResult,
) -> str:
    messages = [
        {"role": "system", "content": FINAL_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_request": user_text,
                    "verified_action": plan.model_dump(),
                    "tool_result": {
                        "ok": tool_result.ok,
                        "tool": tool_result.tool,
                        "observation": tool_result.observation,
                        "obs_hash": tool_result.obs_hash,
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]

    response = chat(
        model=model,
        messages=messages,
        options={"temperature": 0},
        stream=False,
    )
    return response.message.content.strip()


def generate_memory_update(
    model: str,
    user_text: str,
    answer_text: str,
    persistent: PersistentMemory,
) -> MemoryUpdate:
    """Ask the model what she wants to remember from this interaction."""
    existing_memories = persistent.to_context()
    messages = [
        {"role": "system", "content": MEMORY_PROMPT},
        *existing_memories,
        {
            "role": "user",
            "content": json.dumps({
                "user_request": user_text,
                "your_answer": answer_text,
            }, ensure_ascii=False),
        },
    ]

    response = chat(
        model=model,
        messages=messages,
        format=MemoryUpdate.model_json_schema(),
        options={"temperature": 0},
        stream=False,
    )
    return MemoryUpdate.model_validate_json(response.message.content)


# ----------------------------
# 6) Agent loop (trace-aware)
# ----------------------------

def run_agent(model: str, user_text: str, max_repairs: int = 2, max_steps: int = 5) -> str:
    persistent = PersistentMemory.load()
    persistent.age_memories()
    model_memories = persistent.to_context()
    trace = Trace.identity()

    for step_num in range(max_steps):
        # Build memory: model's own memories + current trace
        memory = model_memories + trace.to_memory()

        for attempt in range(max_repairs + 1):
            try:
                plan = plan_once(model, user_text, memory)
                break
            except ValidationError as e:
                if attempt == max_repairs:
                    raise
                memory.append({
                    "role": "assistant",
                    "content": f"Rejected planner output due to schema error: {e}",
                })
        else:
            raise RuntimeError("unreachable")

        # Terminal action: answer directly
        if plan.step.op == Op.ANSWER_DIRECT:
            result = execute_step(plan.step)
            entry = TraceEntry(
                step=plan.step.model_dump(),
                result=asdict(result),
            )
            trace = trace.compose(Trace(entries=[entry]))
            break

        # Deterministic execution
        try:
            result = execute_step(plan.step)
        except Exception as e:
            if max_repairs <= 0:
                raise
            memory.append({
                "role": "assistant",
                "content": (
                    "Executor rejected the validated plan. "
                    f"Reason: {type(e).__name__}: {e}"
                ),
            })
            raw_previous = plan.model_dump_json()
            repaired = repair_plan(
                model=model,
                user_text=user_text,
                memory=memory,
                previous_bad_output=raw_previous,
                error_msg=f"{type(e).__name__}: {e}",
            )
            result = execute_step(repaired.step)
            plan = repaired

        # Append to trace
        entry = TraceEntry(
            step=plan.step.model_dump(),
            result=asdict(result),
        )
        trace = trace.compose(Trace(entries=[entry]))

        # Deduplicate to enforce idempotence
        trace = trace.deduplicate()

    # Generate final answer with full trace context
    answer_text = final_answer(model, user_text, plan, result)

    # Ask the model what she wants to remember
    try:
        mem_update = generate_memory_update(model, user_text, answer_text, persistent)
        persistent.apply_update(mem_update)
    except (ValidationError, Exception) as e:
        # Memory generation is best-effort; don't fail the whole run
        print(f"[memory] Could not generate memory update: {e}", file=sys.stderr)

    persistent.save()

    return answer_text


if __name__ == "__main__":
    model_name = os.environ.get("MODEL", "qwen3.5")

    if len(sys.argv) >= 2 and sys.argv[1] == "--restore":
        PersistentMemory.restore_backup()
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python qwen3_5.py 'your request here'")
        print("       python qwen3_5.py --restore   (restore memory from backup)")
        sys.exit(1)

    request_text = sys.argv[1]
    answer = run_agent(model_name, request_text)
    print(answer)
