#!/usr/bin/env python3
"""Quick sanity test for the infinite algebra."""

import tempfile
import pathlib
from infinite_algebra import (
    PipelineSpec, PipelineStepRef, CodeToolSpec,
    ToolRegistry, execute_pipeline, validate_source,
)

# Test 1: Pipeline monoid identity and composition
p1 = PipelineSpec(
    name="step-a",
    steps=[PipelineStepRef(op="lora", config={"base_model": "test"}, why_compact="adapt")],
    why_compact="first",
)
p2 = PipelineSpec(
    name="step-b",
    steps=[PipelineStepRef(op="evaluate", config={"benchmarks": ["mmlu"]}, why_compact="eval")],
    why_compact="second",
)
composed = p1.compose(p2)
print(f"Composed: {composed.name}, {len(composed.steps)} steps")
assert len(composed.steps) == 2

identity = PipelineSpec.identity()
assert len(identity.steps) == 0
print("Monoid identity OK")

# Test 2: CodeTool validation — safe code
validate_source("add_nums", "def add_nums(a, b, input_data=None):\n    return a + b")
print("Safe code validated OK")

# Test 3: CodeTool validation — reject imports
try:
    validate_source("bad", "import os\ndef bad(): pass")
    print("FAIL: should have rejected import")
except ValueError as e:
    print(f"Correctly rejected import: {e}")

# Test 4: CodeTool validation — reject exec
try:
    validate_source("bad2", "def bad2():\n    exec('print(1)')")
    print("FAIL: should have rejected exec")
except ValueError as e:
    print(f"Correctly rejected exec: {e}")

# Test 5: ToolRegistry register + invoke
reg = ToolRegistry(pathlib.Path(tempfile.mktemp(suffix=".json")))
spec = CodeToolSpec(
    name="double_it",
    description="Doubles a number",
    parameters={"x": "int"},
    source_code="def double_it(x, input_data=None):\n    return x * 2",
    why_compact="Need a doubler tool",
)
msg = reg.register(spec)
print(msg)
fn = reg.get_tool("double_it")
assert fn(x=5) == 10
print(f"double_it(5) = {fn(x=5)}")

# Test 6: Registry persistence
reg2 = ToolRegistry(reg.registry_path)
assert reg2.has_tool("double_it")
print("Registry persists across loads")

# Test 7: Pipeline with code_tool
def mock_executor(op, config, input_data):
    return {"echo": op, "config": config}

pr = execute_pipeline(composed, mock_executor, reg)
print(f"Pipeline result: ok={pr.ok}, summary={pr.summary}")

print("\nALL TESTS PASSED")
