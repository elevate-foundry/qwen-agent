#!/usr/bin/env python3
"""Sanity test for metric_algebra.py — tests everything that doesn't need GPU."""

from metric_algebra import (
    ModelType, Precision, AdapterMethod,
    TypedOperator, compose_operators,
    make_update_op, make_lora_attach_op, make_merge_op, make_quantize_op,
    make_prune_op, make_identity_op,
    DENSE_FP16, QUANT4, LORA_TYPE,
    OperatorCost,
    RelationTable,
    training_op_to_typed_operator,
    get_metric_schemas,
)

# ------- 1. Type system -------
print("--- Type System ---")
assert DENSE_FP16 == ModelType(precision=Precision.FP16)
assert DENSE_FP16 != QUANT4
lora16 = LORA_TYPE(16)
assert lora16.adapter == AdapterMethod.LORA
assert lora16.adapter_rank == 16
print(f"  DenseFP16: {DENSE_FP16}")
print(f"  Quant4:    {QUANT4}")
print(f"  LoRA(16):  {lora16}")

# ------- 2. Operator construction -------
print("\n--- Operators ---")
U = make_update_op(learning_rate=2e-4, steps=100)
L = make_lora_attach_op(rank=16)
M = make_merge_op(rank=16)
Q = make_quantize_op(bits=4)
P = make_prune_op(sparsity=0.5)
I = make_identity_op()
print(f"  {U}")
print(f"  {L}")
print(f"  {M}")
print(f"  {Q}")
print(f"  {P}")
print(f"  {I}")

# ------- 3. Type-checked composition -------
print("\n--- Composition ---")

# Valid: M ∘ L (LoRA attach then merge)
ML = compose_operators(M, L)
print(f"  M ∘ L = {ML}")
assert ML.domain == DENSE_FP16
assert ML.codomain == DENSE_FP16

# Valid: Q ∘ U (update then quantize)
QU = compose_operators(Q, U)
print(f"  Q ∘ U = {QU}")

# Invalid: L ∘ Q (can't attach LoRA to quantized model)
try:
    compose_operators(L, Q)
    print("  FAIL: should have rejected L ∘ Q")
except TypeError as e:
    print(f"  Correctly rejected L ∘ Q: {e}")

# ------- 4. Cost algebra -------
print("\n--- Cost Algebra (tropical + max-plus) ---")
c1 = OperatorCost(compute_flops=1e12, peak_memory_bytes=4e9, wall_seconds=60)
c2 = OperatorCost(compute_flops=2e12, peak_memory_bytes=8e9, wall_seconds=120)
composed_cost = c1.compose(c2)
print(f"  c1 = {c1}")
print(f"  c2 = {c2}")
print(f"  c1∘c2 = {composed_cost}")
assert composed_cost.compute_flops == 3e12   # additive
assert composed_cost.peak_memory_bytes == 8e9  # max (not sum!)
assert composed_cost.wall_seconds == 180       # additive

identity_cost = OperatorCost.identity()
assert identity_cost.compute_flops == 0
print(f"  Identity cost: {identity_cost}")
print(f"  Identity dominates c1: {identity_cost.dominates(c1)}")

# ------- 5. Relation table -------
print("\n--- Relation Table ---")
rt = RelationTable(
    operator_names=["U", "L", "Q", "M", "P"],
    defects=[
        [0.0, 0.12, 0.45, 0.08, 0.15],
        [0.12, 0.0, 0.67, 0.03, 0.22],
        [0.45, 0.67, 0.0, 0.55, 0.38],
        [0.08, 0.03, 0.55, 0.0, 0.11],
        [0.15, 0.22, 0.38, 0.11, 0.0],
    ]
)
print(rt.to_markdown())
most_a, most_b, most_d = rt.most_noncommutative()
print(f"\n  Most noncommutative: ({most_a}, {most_b}) with 𝔠 = {most_d:.4f}")
commuting = rt.approximately_commuting(threshold=0.05)
print(f"  Approximately commuting (ε<0.05): {commuting}")

# ------- 6. Training algebra bridge -------
print("\n--- Training Algebra Bridge ---")
lora_op = training_op_to_typed_operator("lora", {"rank": 32})
print(f"  LoRA(32): {lora_op}")
dpo_op = training_op_to_typed_operator("dpo", {"beta": 0.1})
print(f"  DPO: {dpo_op}")
prune_op = training_op_to_typed_operator("pruning", {"sparsity": 0.3})
print(f"  Prune(0.3): {prune_op}")

# ------- 7. Schema export -------
print("\n--- Schemas ---")
schemas = get_metric_schemas()
print(f"  Operators: {list(schemas['operators'].keys())}")
print(f"  Measurements: {list(schemas['measurements'].keys())}")

print("\n" + "=" * 50)
print("ALL METRIC ALGEBRA TESTS PASSED")
print("=" * 50)
