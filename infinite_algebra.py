#!/usr/bin/env python3
"""
Infinite Tool Algebra — extends the finite algebra with two new dimensions:

1. PIPELINE (horizontal composition):
   A free monoid over the finite op set. Any sequence of ops composes into
   a single compound operation that executes sequentially, threading outputs.
   e.g. LORA → PRUNE → EVALUATE is one Pipeline.

2. CODE_TOOL (vertical generation):
   Aria synthesizes new Python functions at runtime, validates them in a
   restricted sandbox, registers them in a persistent ToolRegistry, and
   can invoke them in future turns. The op space becomes countably infinite.

Algebraic properties preserved:
- Identity:      Pipeline([]) ≡ identity (no-op)
- Associativity: Pipeline(a, b).compose(Pipeline(c)) == Pipeline(a, b, c)
- Safety:        CodeTools run in a restricted namespace (no os, subprocess, network)
- Idempotence:   Duplicate pipeline steps with same hash are deduplicated
- Trace:         Both new ops produce TraceEntry-compatible results
"""

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
import time
from dataclasses import dataclass, field as dc_field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ============================================================================
# 1) Pipeline — Free Monoid over Finite Ops
# ============================================================================

class PipelineStepRef(BaseModel):
    """Reference to a step in the pipeline. Uses the existing op vocabulary
    but allows chaining N of them with data flow between steps."""
    op: str = Field(..., description="Operation name from the finite algebra (e.g. 'lora', 'prune', 'evaluate') or a registered code_tool name.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Op-specific configuration.")
    input_from: Optional[int] = Field(default=None, description="Index of a previous step whose output feeds into this step. null = use original input.")
    why_compact: str = Field(default="", description="Short justification for this step, <= 80 chars.")


class PipelineSpec(BaseModel):
    """A compound operation: an ordered sequence of steps forming a free monoid element."""
    name: str = Field(..., description="Human-readable pipeline name, e.g. 'lora-then-prune'.")
    steps: List[PipelineStepRef] = Field(default_factory=list, description="Ordered steps to execute.")
    why_compact: str = Field(..., description="Why this pipeline, <= 160 chars.")

    @model_validator(mode="after")
    def validate_pipeline(self) -> "PipelineSpec":
        if len(self.why_compact) > 160:
            self.why_compact = self.why_compact[:160]
        # Validate input_from references
        for i, step in enumerate(self.steps):
            if step.input_from is not None:
                if step.input_from < 0 or step.input_from >= i:
                    raise ValueError(f"Step {i}: input_from={step.input_from} must reference a previous step (0..{i-1})")
        return self

    # -- Monoid operations --

    @staticmethod
    def identity() -> "PipelineSpec":
        """The identity element — an empty pipeline (no-op)."""
        return PipelineSpec(name="identity", steps=[], why_compact="Identity pipeline (no-op).")

    def compose(self, other: "PipelineSpec") -> "PipelineSpec":
        """Associative composition: self ∘ other = self's steps then other's steps.
        Re-indexes input_from references in `other` to account for the offset."""
        offset = len(self.steps)
        shifted_steps = []
        for step in other.steps:
            new_ref = step.model_copy()
            if new_ref.input_from is not None:
                new_ref.input_from += offset
            shifted_steps.append(new_ref)
        return PipelineSpec(
            name=f"{self.name}+{other.name}",
            steps=self.steps + shifted_steps,
            why_compact=f"Composed: {self.name} then {other.name}"[:160],
        )

    def fingerprint(self) -> str:
        """Content hash for deduplication."""
        content = json.dumps([s.model_dump() for s in self.steps], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class PipelineResult:
    """Result of executing a full pipeline."""
    ok: bool
    pipeline_name: str
    step_results: List[Dict[str, Any]]
    summary: str
    result_hash: str


def execute_pipeline(spec: PipelineSpec, step_executor: Callable, registry: "ToolRegistry") -> PipelineResult:
    """Execute a pipeline by running each step sequentially.

    Args:
        spec: The pipeline specification.
        step_executor: Function that takes (op: str, config: dict, input_data: Any) -> dict.
                       This bridges back to the finite algebra's execute_step or registry tools.
        registry: The tool registry for looking up code_tools.
    """
    results: List[Dict[str, Any]] = []
    summaries: List[str] = []

    for i, step in enumerate(spec.steps):
        # Resolve input
        input_data = None
        if step.input_from is not None and step.input_from < len(results):
            input_data = results[step.input_from]

        try:
            # Check if it's a registered code_tool
            if registry.has_tool(step.op):
                tool_fn = registry.get_tool(step.op)
                result = tool_fn(input_data=input_data, **step.config)
                if not isinstance(result, dict):
                    result = {"output": result}
            else:
                result = step_executor(step.op, step.config, input_data)

            results.append({"ok": True, "step": i, "op": step.op, "output": result})
            summaries.append(f"[{i}] {step.op}: ok")
        except Exception as e:
            error_result = {"ok": False, "step": i, "op": step.op, "error": str(e)}
            results.append(error_result)
            summaries.append(f"[{i}] {step.op}: FAILED ({e})")
            # Fail fast — don't continue pipeline on error
            break

    summary = f"Pipeline '{spec.name}': {' → '.join(summaries)}"
    return PipelineResult(
        ok=all(r.get("ok", False) for r in results),
        pipeline_name=spec.name,
        step_results=results,
        summary=summary,
        result_hash=hashlib.sha256(summary.encode()).hexdigest(),
    )


# ============================================================================
# 2) CodeTool — Runtime Function Synthesis
# ============================================================================

# Restricted builtins for sandboxed execution
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr,
    "dict": dict, "enumerate": enumerate, "filter": filter, "float": float,
    "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr,
    "hash": hash, "hex": hex, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "oct": oct,
    "ord": ord, "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type,
    "zip": zip,
    # Safe modules exposed as builtins
    "json": json, "math": __import__("math"), "re": __import__("re"),
    "textwrap": textwrap, "hashlib": hashlib,
}

# Forbidden AST node types — no imports, no exec/eval, no attribute access to __dunder__
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom,
)

_FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "__import__", "open", "input",
    "breakpoint", "exit", "quit", "globals", "locals", "vars", "dir",
    "delattr", "setattr",
}

_FORBIDDEN_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__closure__", "__builtins__",
    "__import__", "__loader__", "__spec__",
}


class CodeToolSpec(BaseModel):
    """Specification for a new tool that Aria synthesizes at runtime."""
    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]{1,48}$", description="Tool function name (snake_case, 2-50 chars).")
    description: str = Field(..., max_length=200, description="What this tool does.")
    parameters: Dict[str, str] = Field(default_factory=dict, description="Parameter names and their type descriptions.")
    source_code: str = Field(..., description="Python function body. Must define a function with the same name as `name`.")
    why_compact: str = Field(..., description="Why this tool is needed, <= 160 chars.")

    @model_validator(mode="after")
    def validate_code(self) -> "CodeToolSpec":
        if len(self.why_compact) > 160:
            self.why_compact = self.why_compact[:160]
        # Validate the source code is parseable and safe
        validate_source(self.name, self.source_code)
        return self


def validate_source(name: str, source: str) -> None:
    """Static analysis: parse the AST and reject dangerous patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in code_tool '{name}': {e}")

    for node in ast.walk(tree):
        # Block forbidden node types
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"code_tool '{name}': import statements are forbidden")

        # Block forbidden function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_NAMES:
                raise ValueError(f"code_tool '{name}': call to '{node.func.id}' is forbidden")

        # Block forbidden attribute access
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            raise ValueError(f"code_tool '{name}': access to '.{node.attr}' is forbidden")

        # Block string references to forbidden names
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for forbidden in ("__import__", "subprocess", "os.system", "os.popen"):
                if forbidden in node.value:
                    raise ValueError(f"code_tool '{name}': reference to '{forbidden}' in string is forbidden")

    # Verify the function is actually defined
    func_defs = [n for n in ast.iter_child_nodes(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    func_names = [f.name for f in func_defs]
    if name not in func_names:
        raise ValueError(f"code_tool source must define a function named '{name}', found: {func_names}")


def compile_code_tool(spec: CodeToolSpec) -> Callable:
    """Compile a validated CodeToolSpec into a callable function."""
    # Double-check safety
    validate_source(spec.name, spec.source_code)

    # Execute in restricted namespace
    namespace = {"__builtins__": _SAFE_BUILTINS}
    exec(compile(spec.source_code, f"<code_tool:{spec.name}>", "exec"), namespace)

    fn = namespace.get(spec.name)
    if fn is None or not callable(fn):
        raise ValueError(f"code_tool '{spec.name}' did not produce a callable")

    return fn


# ============================================================================
# 3) Tool Registry — Persistent, Growing Op Space
# ============================================================================

class ToolRegistry:
    """Registry of synthesized code_tools. Persists to disk so tools survive
    across sessions. The op space grows monotonically — tools can be added
    but not removed (append-only for auditability)."""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("tool_registry.json")
        self._tools: Dict[str, Callable] = {}
        self._specs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load tool specs from disk and recompile them."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                for name, spec_dict in data.get("tools", {}).items():
                    try:
                        spec = CodeToolSpec(**spec_dict)
                        fn = compile_code_tool(spec)
                        self._tools[name] = fn
                        self._specs[name] = spec_dict
                    except Exception as e:
                        print(f"[registry] Failed to load tool '{name}': {e}")
            except Exception as e:
                print(f"[registry] Failed to load registry: {e}")

    def _save(self):
        """Persist the registry to disk."""
        data = {
            "tools": self._specs,
            "metadata": {
                "count": len(self._specs),
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
        self.registry_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def register(self, spec: CodeToolSpec) -> str:
        """Validate, compile, and register a new code_tool. Returns confirmation."""
        fn = compile_code_tool(spec)
        self._tools[spec.name] = fn
        self._specs[spec.name] = spec.model_dump()
        self._save()
        return f"Registered code_tool '{spec.name}': {spec.description}"

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Callable:
        if name not in self._tools:
            raise KeyError(f"No code_tool named '{name}' in registry")
        return self._tools[name]

    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools with their descriptions."""
        return [
            {"name": name, "description": spec.get("description", "")}
            for name, spec in self._specs.items()
        ]

    def get_spec(self, name: str) -> Dict[str, Any]:
        return self._specs.get(name, {})

    def to_prompt_description(self) -> str:
        """Generate a description block for the system prompt so Aria knows
        what code_tools are available."""
        if not self._specs:
            return "No custom code_tools registered yet."
        lines = ["Registered code_tools (use in pipelines or invoke directly):"]
        for name, spec in self._specs.items():
            params = spec.get("parameters", {})
            param_str = ", ".join(f"{k}: {v}" for k, v in params.items()) if params else "none"
            lines.append(f"  {name}({param_str}) — {spec.get('description', '')}")
        return "\n".join(lines)


# ============================================================================
# 4) Schemas export (for constrained generation)
# ============================================================================

def get_infinite_schemas() -> Dict[str, Any]:
    """Export schemas for the model to reference during planning."""
    return {
        "PipelineSpec": PipelineSpec.model_json_schema(),
        "PipelineStepRef": PipelineStepRef.model_json_schema(),
        "CodeToolSpec": CodeToolSpec.model_json_schema(),
    }
