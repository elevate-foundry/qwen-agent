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
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from ollama import chat
from pydantic import BaseModel, Field, ValidationError, model_validator


# ----------------------------
# 1) Finite action algebra
# ----------------------------

class Op(str, Enum):
    SEARCH_WEB = "search_web"
    FETCH_URL = "fetch_url"
    LIST_DIR = "list_dir"
    READ_TEXT = "read_text"
    ANSWER_DIRECT = "answer_direct"


class Step(BaseModel):
    op: Op
    query: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
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
            if self.url or self.path:
                raise ValueError("SEARCH_WEB only permits query")

        elif self.op == Op.FETCH_URL:
            if not self.url:
                raise ValueError("FETCH_URL requires url")
            if self.query or self.path:
                raise ValueError("FETCH_URL only permits url")

        elif self.op == Op.LIST_DIR:
            if not self.path:
                raise ValueError("LIST_DIR requires path")
            if self.query or self.url:
                raise ValueError("LIST_DIR only permits path")

        elif self.op == Op.READ_TEXT:
            if not self.path:
                raise ValueError("READ_TEXT requires path")
            if self.query or self.url:
                raise ValueError("READ_TEXT only permits path")

        elif self.op == Op.ANSWER_DIRECT:
            if self.query or self.url or self.path:
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
# 3) Model interface
# ----------------------------

SYSTEM_PROMPT = """
You are a tool-planning model inside a verified executor.

Rules:
- Emit ONLY JSON that matches the provided schema.
- You do NOT call tools directly.
- You choose exactly one next action from this finite enum:
  search_web, fetch_url, list_dir, read_text, answer_direct
- Never emit arbitrary shell commands.
- Prefer answer_direct only when no external information is needed.
- Keep why_compact <= 160 chars and compact_goal <= 120 chars.
- safety_invariant must state that only enum tools with validated args may execute.
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


# ----------------------------
# 4) Agent loop
# ----------------------------

def run_agent(model: str, user_text: str, max_repairs: int = 2) -> str:
    memory: List[Dict[str, Any]] = []
    last_raw = ""

    for attempt in range(max_repairs + 1):
        try:
            plan = plan_once(model, user_text, memory)
            break
        except ValidationError as e:
            if attempt == max_repairs:
                raise
            last_raw = str(e)
            memory.append({
                "role": "assistant",
                "content": f"Rejected planner output due to schema error: {e}",
            })
    else:
        raise RuntimeError("unreachable")

    # Deterministic execution
    try:
        result = execute_step(plan.step)
    except Exception as e:
        # One repair round on executor failure
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

    return final_answer(model, user_text, plan, result)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL", "qwen3.5")
    if len(sys.argv) < 2:
        print("Usage: python provable_agent.py 'your request here'")
        sys.exit(1)

    request_text = sys.argv[1]
    answer = run_agent(model_name, request_text)
    print(answer)
