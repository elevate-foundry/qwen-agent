#!/usr/bin/env python3
"""
Metric Algebra — Behavioral distance and algebraic measurements for training operators.

Implements the evaluation harness from:
  "A Typed Metric Calculus for Composable Training Transformations"

Core quantities:
  d(x, y)     — behavioral pseudometric between two model states
  d(T, S)     — operator distance (sup over calibration states)
  𝔠(A, B)     — commutator defect: d(A∘B, B∘A)
  κ(A, B)     — training curvature: d(A⁻¹B⁻¹AB, I)
  δ_idem(T)   — idempotence defect: d(T∘T, T)
  δ_inv(T,T⁻) — inverse defect: d(T⁻∘T, I)

The behavioral pseudometric uses expected KL divergence over model outputs
on a calibration dataset, which is invariant to reparameterization.

Type system:
  Each operator has declared domain and codomain types.
  Composition is only valid when cod(T) = dom(S).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


# ============================================================================
# 1) Model Type System
# ============================================================================

class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class AdapterMethod(str, Enum):
    NONE = "none"
    LORA = "lora"
    DORA = "dora"


class ModelType(BaseModel):
    """Type descriptor for a model state in the training category.

    Objects in the category. A model's type is determined by its precision,
    adapter state, and structural properties.
    """
    precision: Precision = Precision.FP16
    adapter: AdapterMethod = AdapterMethod.NONE
    adapter_rank: Optional[int] = None
    is_merged: bool = False
    sparsity: float = Field(default=0.0, ge=0.0, lt=1.0)
    is_student: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)

    def __eq__(self, other):
        if not isinstance(other, ModelType):
            return False
        return (self.precision == other.precision and
                self.adapter == other.adapter and
                self.adapter_rank == other.adapter_rank and
                self.is_merged == other.is_merged and
                abs(self.sparsity - other.sparsity) < 1e-6 and
                self.is_student == other.is_student)

    def __hash__(self):
        return hash((self.precision, self.adapter, self.adapter_rank,
                      self.is_merged, round(self.sparsity, 4), self.is_student))

    def __repr__(self):
        parts = [self.precision.value]
        if self.adapter != AdapterMethod.NONE:
            parts.append(f"+{self.adapter.value}(r={self.adapter_rank})")
        if self.is_merged:
            parts.append("+merged")
        if self.sparsity > 0:
            parts.append(f"+sparse({self.sparsity:.0%})")
        if self.is_student:
            parts.append("+student")
        return f"Type({''.join(parts)})"

    def compatible_with(self, other: "ModelType") -> bool:
        """Check if this type is compatible with (assignable to) another."""
        return self == other


# ============================================================================
# 2) Typed Training Operators
# ============================================================================

@dataclass
class TypedOperator:
    """A morphism in the training category: a typed function on model states.

    Each operator has:
    - name: human-readable identifier
    - domain: input ModelType (precondition)
    - codomain: output ModelType (postcondition)
    - apply: the actual function (model_path, config) -> model_path
    - config: operator-specific parameters
    - cost: (compute_flops, peak_memory_bytes, wall_seconds)
    """
    name: str
    domain: ModelType
    codomain: ModelType
    config: Dict[str, Any] = dc_field(default_factory=dict)
    cost: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (compute, memory, latency)

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}"

    def composable_with(self, other: "TypedOperator") -> bool:
        """Check if self ∘ other is valid (other runs first, then self)."""
        return self.domain.compatible_with(other.codomain)


def compose_operators(outer: TypedOperator, inner: TypedOperator) -> TypedOperator:
    """Compose two operators: (outer ∘ inner)(x) = outer(inner(x)).

    Type-checks that cod(inner) = dom(outer).
    Cost uses correct algebra: compute adds, memory maxes, latency adds.
    """
    if not outer.composable_with(inner):
        raise TypeError(
            f"Cannot compose {outer.name}: {outer.domain} → {outer.codomain} "
            f"with {inner.name}: {inner.domain} → {inner.codomain}. "
            f"cod(inner) = {inner.codomain} ≠ dom(outer) = {outer.domain}"
        )
    return TypedOperator(
        name=f"({outer.name} ∘ {inner.name})",
        domain=inner.domain,
        codomain=outer.codomain,
        config={"outer": outer.config, "inner": inner.config},
        cost=(
            inner.cost[0] + outer.cost[0],             # compute: additive
            max(inner.cost[1], outer.cost[1]),           # memory: max (tropical)
            inner.cost[2] + outer.cost[2],               # latency: additive
        ),
    )


# ============================================================================
# 3) Standard Operator Library (the generating set 𝒢)
# ============================================================================

# Type constants
DENSE_FP16 = ModelType(precision=Precision.FP16)
DENSE_FP32 = ModelType(precision=Precision.FP32)
DENSE_BF16 = ModelType(precision=Precision.BF16)
QUANT4 = ModelType(precision=Precision.INT4)
QUANT8 = ModelType(precision=Precision.INT8)


def LORA_TYPE(rank: int) -> ModelType:
    return ModelType(precision=Precision.FP16, adapter=AdapterMethod.LORA, adapter_rank=rank)


def DORA_TYPE(rank: int) -> ModelType:
    return ModelType(precision=Precision.FP16, adapter=AdapterMethod.DORA, adapter_rank=rank)


def SPARSE_TYPE(sparsity: float) -> ModelType:
    return ModelType(precision=Precision.FP16, sparsity=sparsity)


STUDENT_TYPE = ModelType(precision=Precision.FP16, is_student=True)


def make_update_op(learning_rate: float = 2e-4, steps: int = 100) -> TypedOperator:
    """U_η : DenseFP16 → DenseFP16 — gradient update (fine-tuning)."""
    return TypedOperator(
        name=f"U(lr={learning_rate},steps={steps})",
        domain=DENSE_FP16,
        codomain=DENSE_FP16,
        config={"learning_rate": learning_rate, "steps": steps},
        cost=(6.0 * steps * 5e8, 4e9, steps * 0.5),  # rough estimates
    )


def make_lora_attach_op(rank: int = 16) -> TypedOperator:
    """L_r : DenseFP16 → DenseFP16+LoRA(r) — attach LoRA adapters."""
    return TypedOperator(
        name=f"L(r={rank})",
        domain=DENSE_FP16,
        codomain=LORA_TYPE(rank),
        config={"rank": rank, "alpha": rank * 2, "target_modules": ["q_proj", "v_proj"]},
        cost=(1e6, 1e8, 1.0),
    )


def make_lora_train_op(rank: int = 16, steps: int = 100, lr: float = 2e-4) -> TypedOperator:
    """L_r^train : DenseFP16+LoRA(r) → DenseFP16+LoRA(r) — train the LoRA adapters."""
    lora_type = LORA_TYPE(rank)
    return TypedOperator(
        name=f"L_train(r={rank},steps={steps})",
        domain=lora_type,
        codomain=lora_type,
        config={"rank": rank, "steps": steps, "learning_rate": lr},
        cost=(6.0 * steps * rank * 1e5, 2e9, steps * 0.3),
    )


def make_merge_op(rank: int = 16) -> TypedOperator:
    """M : DenseFP16+LoRA(r) → DenseFP16 — merge adapters back into base weights."""
    return TypedOperator(
        name=f"M(r={rank})",
        domain=LORA_TYPE(rank),
        codomain=DENSE_FP16,
        config={"rank": rank},
        cost=(1e7, 2e9, 2.0),
    )


def make_quantize_op(bits: int = 4) -> TypedOperator:
    """Q_b : DenseFP16 → Quant(b) — post-training quantization."""
    codomain = QUANT4 if bits == 4 else QUANT8
    return TypedOperator(
        name=f"Q({bits}bit)",
        domain=DENSE_FP16,
        codomain=codomain,
        config={"bits": bits, "method": "gptq"},
        cost=(1e9, 4e9, 60.0),
    )


def make_prune_op(sparsity: float = 0.5) -> TypedOperator:
    """P_s : DenseFP16 → Sparse(s) — weight pruning."""
    return TypedOperator(
        name=f"P(s={sparsity})",
        domain=DENSE_FP16,
        codomain=SPARSE_TYPE(sparsity),
        config={"sparsity": sparsity, "method": "magnitude"},
        cost=(5e8, 2e9, 30.0),
    )


def make_identity_op(model_type: ModelType = DENSE_FP16) -> TypedOperator:
    """I : T → T — identity operator (no-op)."""
    return TypedOperator(
        name="I",
        domain=model_type,
        codomain=model_type,
        config={},
        cost=(0.0, 0.0, 0.0),
    )


# ============================================================================
# 4) Behavioral Pseudometric d(x, y)
# ============================================================================

@dataclass
class BehavioralDistance:
    """Result of computing behavioral distance between two model states."""
    kl_divergence: float         # E_z[KL(f_x(z) || f_y(z))]
    symmetric_kl: float          # (KL(p||q) + KL(q||p)) / 2
    cosine_distance: float       # 1 - cos(logits_x, logits_y) averaged
    l2_output_distance: float    # E_z[||f_x(z) - f_y(z)||_2]
    num_samples: int
    calibration_dataset: str

    @property
    def primary(self) -> float:
        """Primary distance metric — symmetric KL."""
        return self.symmetric_kl

    def __repr__(self):
        return (f"d = {self.symmetric_kl:.4f} "
                f"(KL={self.kl_divergence:.4f}, cos={self.cosine_distance:.4f}, "
                f"L2={self.l2_output_distance:.4f}, n={self.num_samples})")


def compute_behavioral_distance(
    model_a_path: str,
    model_b_path: str,
    calibration_prompts: List[str],
    max_tokens: int = 32,
    device: str = "cuda",
) -> BehavioralDistance:
    """Compute d(x, y) — the behavioral pseudometric between two model states.

    Uses expected KL divergence over output distributions on calibration data.
    This is the fundamental measurement from which all algebraic quantities derive.

    Args:
        model_a_path: Path or HF name of first model.
        model_b_path: Path or HF name of second model.
        calibration_prompts: List of input prompts for calibration.
        max_tokens: Number of tokens to compare per prompt.
        device: Device to run on.

    Returns:
        BehavioralDistance with multiple distance metrics.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load both models
    tokenizer_a = AutoTokenizer.from_pretrained(model_a_path, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_path, trust_remote_code=True)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_path, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()
    model_b = AutoModelForCausalLM.from_pretrained(
        model_b_path, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()

    kl_sum = 0.0
    reverse_kl_sum = 0.0
    cosine_sum = 0.0
    l2_sum = 0.0
    n = 0

    with torch.no_grad():
        for prompt in calibration_prompts:
            # Tokenize with both tokenizers (should be same for same arch)
            inputs_a = tokenizer_a(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs_b = tokenizer_b(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            # Get logits
            logits_a = model_a(**inputs_a).logits[:, -1, :]  # last token logits
            logits_b = model_b(**inputs_b).logits[:, -1, :]

            # Ensure same vocab size (trim to min)
            min_vocab = min(logits_a.shape[-1], logits_b.shape[-1])
            logits_a = logits_a[:, :min_vocab].float()
            logits_b = logits_b[:, :min_vocab].float()

            # Softmax distributions
            p = F.softmax(logits_a, dim=-1)
            q = F.softmax(logits_b, dim=-1)

            # KL(p || q) — clamp for numerical stability
            log_p = torch.log(p.clamp(min=1e-10))
            log_q = torch.log(q.clamp(min=1e-10))

            kl_pq = F.kl_div(log_q, p, reduction="batchmean", log_target=False)
            kl_qp = F.kl_div(log_p, q, reduction="batchmean", log_target=False)

            kl_sum += kl_pq.item()
            reverse_kl_sum += kl_qp.item()

            # Cosine distance
            cos_sim = F.cosine_similarity(logits_a, logits_b, dim=-1).mean()
            cosine_sum += (1.0 - cos_sim.item())

            # L2 distance on log-probs
            l2 = torch.norm(log_p - log_q, p=2, dim=-1).mean()
            l2_sum += l2.item()

            n += 1

    # Clean up GPU memory
    del model_a, model_b
    if device == "cuda":
        torch.cuda.empty_cache()

    if n == 0:
        return BehavioralDistance(0, 0, 0, 0, 0, "empty")

    return BehavioralDistance(
        kl_divergence=kl_sum / n,
        symmetric_kl=(kl_sum + reverse_kl_sum) / (2 * n),
        cosine_distance=cosine_sum / n,
        l2_output_distance=l2_sum / n,
        num_samples=n,
        calibration_dataset="custom",
    )


# ============================================================================
# 5) Default Calibration Dataset
# ============================================================================

DEFAULT_CALIBRATION_PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent",
    "def fibonacci(n):",
    "The theory of relativity states that",
    "To train a neural network, you need",
    "The difference between LoRA and full fine-tuning is",
    "When quantizing a model to 4-bit precision,",
    "The Chinchilla scaling law suggests that",
    "Knowledge distillation works by",
    "Model merging techniques like TIES and DARE",
    "Pruning a neural network involves",
    "The attention mechanism in transformers",
    "Backpropagation computes gradients by",
    "The loss function measures",
    "Batch normalization helps training by",
    "Dropout regularization prevents",
    "The transformer architecture consists of",
    "Transfer learning is useful because",
    "The softmax function converts logits to",
    "Weight decay is a form of regularization that",
]


# ============================================================================
# 6) Algebraic Measurements
# ============================================================================

@dataclass
class CommutatorDefect:
    """𝔠(A, B) = d(A∘B, B∘A) — measures how much operator order matters."""
    operator_a: str
    operator_b: str
    distance: BehavioralDistance
    defect: float  # primary distance value

    def __repr__(self):
        return f"𝔠({self.operator_a}, {self.operator_b}) = {self.defect:.6f}"


@dataclass
class InverseDefect:
    """δ_inv(T, T⁻) = d(T⁻∘T, I) — measures how well T⁻ undoes T."""
    operator: str
    inverse_operator: str
    distance: BehavioralDistance
    defect: float

    def __repr__(self):
        return f"δ_inv({self.operator}, {self.inverse_operator}) = {self.defect:.6f}"


@dataclass
class IdempotenceDefect:
    """δ_idem(T) = d(T∘T, T) — measures whether applying T twice equals once."""
    operator: str
    distance: BehavioralDistance
    defect: float

    def __repr__(self):
        return f"δ_idem({self.operator}) = {self.defect:.6f}"


@dataclass
class TrainingCurvature:
    """κ(A, B) = d(A⁻¹B⁻¹AB, I) — the holonomy / path-dependence measure."""
    operator_a: str
    operator_b: str
    distance: BehavioralDistance
    curvature: float

    def __repr__(self):
        return f"κ({self.operator_a}, {self.operator_b}) = {self.curvature:.6f}"


@dataclass
class AlgebraicRelation:
    """A single measured algebraic relation between operators."""
    relation_type: str  # "commutator", "inverse", "idempotence", "curvature"
    operators: List[str]
    defect: float
    distance: BehavioralDistance
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================================
# 7) Experiment Harness
# ============================================================================

@dataclass
class ExperimentResult:
    """Full results from a metric algebra experiment."""
    commutators: List[CommutatorDefect] = dc_field(default_factory=list)
    inverse_defects: List[InverseDefect] = dc_field(default_factory=list)
    idempotence_defects: List[IdempotenceDefect] = dc_field(default_factory=list)
    curvatures: List[TrainingCurvature] = dc_field(default_factory=list)
    relations: List[AlgebraicRelation] = dc_field(default_factory=list)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "commutators": [
                {"a": c.operator_a, "b": c.operator_b, "defect": c.defect}
                for c in self.commutators
            ],
            "inverse_defects": [
                {"op": d.operator, "inv": d.inverse_operator, "defect": d.defect}
                for d in self.inverse_defects
            ],
            "idempotence_defects": [
                {"op": d.operator, "defect": d.defect}
                for d in self.idempotence_defects
            ],
            "curvatures": [
                {"a": c.operator_a, "b": c.operator_b, "curvature": c.curvature}
                for c in self.curvatures
            ],
            "metadata": self.metadata,
        }

    def summary_table(self) -> str:
        """Pretty-print a summary table of all measurements."""
        lines = ["=" * 60, "METRIC ALGEBRA — EXPERIMENT RESULTS", "=" * 60]

        if self.commutators:
            lines.append("\n--- Commutator Defects 𝔠(A,B) = d(A∘B, B∘A) ---")
            for c in self.commutators:
                lines.append(f"  {c}")

        if self.inverse_defects:
            lines.append("\n--- Inverse Defects δ_inv(T,T⁻) = d(T⁻∘T, I) ---")
            for d in self.inverse_defects:
                lines.append(f"  {d}")

        if self.idempotence_defects:
            lines.append("\n--- Idempotence Defects δ_idem(T) = d(T∘T, T) ---")
            for d in self.idempotence_defects:
                lines.append(f"  {d}")

        if self.curvatures:
            lines.append("\n--- Training Curvature κ(A,B) = d(A⁻¹B⁻¹AB, I) ---")
            for c in self.curvatures:
                lines.append(f"  {c}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def measure_commutator(
    base_model: str,
    apply_a: Callable[[str], str],
    apply_b: Callable[[str], str],
    name_a: str,
    name_b: str,
    calibration_prompts: List[str] = None,
    device: str = "cuda",
) -> CommutatorDefect:
    """Measure 𝔠(A, B) = d(A∘B, B∘A).

    Args:
        base_model: Starting model path.
        apply_a: Function that takes model_path, returns new model_path after applying A.
        apply_b: Function that takes model_path, returns new model_path after applying B.
        name_a, name_b: Operator names for reporting.
        calibration_prompts: Prompts for behavioral distance.
        device: Compute device.

    Returns:
        CommutatorDefect with the measured distance.
    """
    prompts = calibration_prompts or DEFAULT_CALIBRATION_PROMPTS

    # Path 1: A∘B (apply B first, then A)
    after_b = apply_b(base_model)
    after_ab = apply_a(after_b)

    # Path 2: B∘A (apply A first, then B)
    after_a = apply_a(base_model)
    after_ba = apply_b(after_a)

    # Measure d(A∘B, B∘A)
    dist = compute_behavioral_distance(after_ab, after_ba, prompts, device=device)

    return CommutatorDefect(
        operator_a=name_a,
        operator_b=name_b,
        distance=dist,
        defect=dist.primary,
    )


def measure_inverse_defect(
    base_model: str,
    apply_forward: Callable[[str], str],
    apply_inverse: Callable[[str], str],
    name_forward: str,
    name_inverse: str,
    calibration_prompts: List[str] = None,
    device: str = "cuda",
) -> InverseDefect:
    """Measure δ_inv(T, T⁻) = d(T⁻∘T(x), x).

    Compares the model after applying T then T⁻ against the original.
    """
    prompts = calibration_prompts or DEFAULT_CALIBRATION_PROMPTS

    # Apply T then T⁻
    after_forward = apply_forward(base_model)
    after_roundtrip = apply_inverse(after_forward)

    # Compare roundtrip to original
    dist = compute_behavioral_distance(base_model, after_roundtrip, prompts, device=device)

    return InverseDefect(
        operator=name_forward,
        inverse_operator=name_inverse,
        distance=dist,
        defect=dist.primary,
    )


def measure_idempotence(
    base_model: str,
    apply_op: Callable[[str], str],
    name: str,
    calibration_prompts: List[str] = None,
    device: str = "cuda",
) -> IdempotenceDefect:
    """Measure δ_idem(T) = d(T∘T(x), T(x)).

    Compares applying the operator once vs twice.
    """
    prompts = calibration_prompts or DEFAULT_CALIBRATION_PROMPTS

    after_once = apply_op(base_model)
    after_twice = apply_op(after_once)

    dist = compute_behavioral_distance(after_once, after_twice, prompts, device=device)

    return IdempotenceDefect(
        operator=name,
        distance=dist,
        defect=dist.primary,
    )


# ============================================================================
# 8) Cost Algebra (tropical semiring + max-plus)
# ============================================================================

@dataclass
class OperatorCost:
    """Cost triple for an operator.

    Composition rules:
    - compute: additive (total FLOPs)
    - memory: max-plus (peak memory)
    - latency: additive (wall-clock time)
    """
    compute_flops: float = 0.0
    peak_memory_bytes: float = 0.0
    wall_seconds: float = 0.0

    def compose(self, other: "OperatorCost") -> "OperatorCost":
        """Cost of (self after other) — the correct composition law."""
        return OperatorCost(
            compute_flops=self.compute_flops + other.compute_flops,
            peak_memory_bytes=max(self.peak_memory_bytes, other.peak_memory_bytes),
            wall_seconds=self.wall_seconds + other.wall_seconds,
        )

    @staticmethod
    def identity() -> "OperatorCost":
        return OperatorCost(0.0, 0.0, 0.0)

    def dominates(self, other: "OperatorCost") -> bool:
        """Pareto dominance: self is cheaper in all dimensions."""
        return (self.compute_flops <= other.compute_flops and
                self.peak_memory_bytes <= other.peak_memory_bytes and
                self.wall_seconds <= other.wall_seconds)

    def __repr__(self):
        compute_str = f"{self.compute_flops:.1e}" if self.compute_flops > 1e6 else f"{self.compute_flops:.0f}"
        mem_gb = self.peak_memory_bytes / 1e9
        return f"Cost(compute={compute_str} FLOPs, memory={mem_gb:.1f}GB, time={self.wall_seconds:.1f}s)"


# ============================================================================
# 9) Relation Table Builder
# ============================================================================

@dataclass
class RelationTable:
    """An N×N table of commutator defects for a set of operators.

    This is the core empirical artifact — the 'multiplication table' of the algebra.
    """
    operator_names: List[str]
    defects: List[List[float]]  # defects[i][j] = 𝔠(op_i, op_j)

    def to_markdown(self) -> str:
        """Render as a markdown table."""
        n = len(self.operator_names)
        header = "| | " + " | ".join(self.operator_names) + " |"
        sep = "|---" * (n + 1) + "|"
        rows = [header, sep]
        for i in range(n):
            vals = " | ".join(
                f"{self.defects[i][j]:.4f}" if i != j else "—"
                for j in range(n)
            )
            rows.append(f"| **{self.operator_names[i]}** | {vals} |")
        return "\n".join(rows)

    def most_noncommutative(self) -> Tuple[str, str, float]:
        """Find the pair with largest commutator defect."""
        best = (0, 0, 0.0)
        for i in range(len(self.operator_names)):
            for j in range(i + 1, len(self.operator_names)):
                if self.defects[i][j] > best[2]:
                    best = (i, j, self.defects[i][j])
        i, j, d = best
        return self.operator_names[i], self.operator_names[j], d

    def approximately_commuting(self, threshold: float = 0.01) -> List[Tuple[str, str]]:
        """Find pairs that approximately commute."""
        pairs = []
        for i in range(len(self.operator_names)):
            for j in range(i + 1, len(self.operator_names)):
                if self.defects[i][j] < threshold:
                    pairs.append((self.operator_names[i], self.operator_names[j]))
        return pairs


# ============================================================================
# 10) Integration with Training Algebra
# ============================================================================

def training_op_to_typed_operator(op_name: str, config: Dict[str, Any]) -> TypedOperator:
    """Map a TrainingOp string to a TypedOperator with correct types.

    This bridges the finite algebra (training_algebra.py) to the metric algebra.
    """
    op = op_name.lower()

    if op == "lora":
        rank = config.get("rank", config.get("lora_config", {}).get("rank", 16))
        return make_lora_attach_op(rank)

    elif op == "qlora":
        rank = config.get("rank", 16)
        return TypedOperator(
            name=f"QL(r={rank})",
            domain=DENSE_FP16,
            codomain=ModelType(precision=Precision.INT4, adapter=AdapterMethod.LORA, adapter_rank=rank),
            config=config,
            cost=(1e9, 2e9, 60.0),
        )

    elif op == "dora":
        rank = config.get("rank", 16)
        return TypedOperator(
            name=f"DoRA(r={rank})",
            domain=DENSE_FP16,
            codomain=DORA_TYPE(rank),
            config=config,
            cost=(1e9, 3e9, 45.0),
        )

    elif op == "full_finetune":
        lr = config.get("learning_rate", 2e-4)
        epochs = config.get("epochs", 1)
        return TypedOperator(
            name=f"FT(lr={lr},ep={epochs})",
            domain=DENSE_FP16,
            codomain=DENSE_FP16,
            config=config,
            cost=(6e12, 8e9, 3600.0),
        )

    elif op == "dpo":
        beta = config.get("beta", 0.1)
        return TypedOperator(
            name=f"DPO(β={beta})",
            domain=DENSE_FP16,
            codomain=DENSE_FP16,
            config=config,
            cost=(3e12, 6e9, 1800.0),
        )

    elif op == "distillation":
        return TypedOperator(
            name="Distill",
            domain=ModelType(precision=Precision.FP16),  # teacher-student pair
            codomain=STUDENT_TYPE,
            config=config,
            cost=(5e12, 10e9, 7200.0),
        )

    elif op == "merging":
        method = config.get("method", "linear")
        return TypedOperator(
            name=f"Merge({method})",
            domain=DENSE_FP16,
            codomain=ModelType(precision=Precision.FP16, is_merged=True),
            config=config,
            cost=(5e8, 4e9, 30.0),
        )

    elif op == "pruning":
        sparsity = config.get("sparsity", 0.5)
        return make_prune_op(sparsity)

    elif op == "evaluate":
        return TypedOperator(
            name="Eval",
            domain=DENSE_FP16,
            codomain=DENSE_FP16,  # evaluation doesn't change the model
            config=config,
            cost=(1e10, 4e9, 120.0),
        )

    else:
        raise ValueError(f"Unknown training op: {op_name}")


# ============================================================================
# 11) Schema Export
# ============================================================================

def get_metric_schemas() -> Dict[str, Any]:
    """Export schemas for reference."""
    return {
        "ModelType": ModelType.model_json_schema(),
        "Precision": [e.value for e in Precision],
        "AdapterMethod": [e.value for e in AdapterMethod],
        "operators": {
            "U": "Gradient update: DenseFP16 → DenseFP16",
            "L": "LoRA attach: DenseFP16 → DenseFP16+LoRA(r)",
            "M": "Merge: DenseFP16+LoRA(r) → DenseFP16",
            "Q": "Quantize: DenseFP16 → Quant(b)",
            "P": "Prune: DenseFP16 → Sparse(s)",
            "D": "Distill: TeacherStudent → Student",
            "I": "Identity: T → T",
        },
        "measurements": {
            "d(x,y)": "Behavioral pseudometric (symmetric KL over outputs)",
            "𝔠(A,B)": "Commutator defect: d(A∘B, B∘A)",
            "δ_inv": "Inverse defect: d(T⁻∘T, I)",
            "δ_idem": "Idempotence defect: d(T∘T, T)",
            "κ(A,B)": "Training curvature: d(A⁻¹B⁻¹AB, I)",
        },
    }
