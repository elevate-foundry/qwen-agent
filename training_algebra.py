#!/usr/bin/env python3
"""
Finite tool algebra for model training and fine-tuning strategies.

Each training operation is a first-class element of a finite enum with:
- Validated argument schemas (preconditions)
- Deterministic executor stubs (postconditions)
- Composition via the Trace algebra from the main agent

Operations:
  LORA          - Low-rank adaptation
  QLORA         - Quantized LoRA (4-bit base + adapters)
  DORA          - Weight-decomposed low-rank adaptation
  FULL_FINETUNE - Full parameter fine-tuning
  CHINCHILLA    - Compute-optimal scaling (tokens-to-params ratio planning)
  DPO           - Direct preference optimization
  DISTILLATION  - Knowledge distillation from a teacher model
  MERGING       - Model merge (TIES, DARE, SLERP, linear)
  PRUNING       - Structured/unstructured weight pruning
  EVALUATE      - Run evaluation benchmarks (terminal op, like ANSWER_DIRECT)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ----------------------------
# 1) Training operation enum
# ----------------------------

class TrainingOp(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    FULL_FINETUNE = "full_finetune"
    CHINCHILLA = "chinchilla"
    DPO = "dpo"
    DISTILLATION = "distillation"
    MERGING = "merging"
    PRUNING = "pruning"
    EVALUATE = "evaluate"


# ----------------------------
# 2) Per-op argument schemas
# ----------------------------

class LoraConfig(BaseModel):
    """Arguments for LoRA / QLoRA / DoRA."""
    rank: int = Field(..., ge=1, le=512, description="LoRA rank (r). Typical: 8-64.")
    alpha: float = Field(..., gt=0, description="LoRA alpha scaling. Often 2*rank.")
    target_modules: List[str] = Field(
        ...,
        min_length=1,
        description="Module names to apply adapters to, e.g. ['q_proj', 'v_proj']."
    )
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    quantize_bits: Optional[int] = Field(
        default=None,
        description="Set to 4 for QLoRA, 8 for 8-bit, null for standard LoRA."
    )
    use_dora: bool = Field(
        default=False,
        description="Use weight-decomposed adaptation (DoRA)."
    )


class FullFinetuneConfig(BaseModel):
    """Arguments for full parameter fine-tuning."""
    learning_rate: float = Field(..., gt=0, le=1.0, description="Peak learning rate.")
    epochs: int = Field(..., ge=1, le=100)
    batch_size: int = Field(..., ge=1, le=4096)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)


class ChinchillaConfig(BaseModel):
    """Compute-optimal scaling law planning."""
    compute_budget_flops: float = Field(
        ...,
        gt=0,
        description="Total compute budget in FLOPs (e.g. 1e21)."
    )
    model_params: Optional[float] = Field(
        default=None,
        description="Current model size in params. If provided, computes optimal tokens."
    )
    target_tokens: Optional[float] = Field(
        default=None,
        description="Target training tokens. If provided, computes optimal model size."
    )

    @model_validator(mode="after")
    def at_least_one(self) -> "ChinchillaConfig":
        if self.model_params is None and self.target_tokens is None:
            raise ValueError("Provide model_params or target_tokens (or both) for scaling analysis.")
        return self


class DpoConfig(BaseModel):
    """Arguments for direct preference optimization."""
    beta: float = Field(default=0.1, gt=0, le=10.0, description="DPO beta (KL penalty).")
    learning_rate: float = Field(..., gt=0, le=1.0)
    epochs: int = Field(..., ge=1, le=50)
    batch_size: int = Field(..., ge=1, le=1024)
    reference_model: Optional[str] = Field(
        default=None,
        description="Reference model name. null = use base model copy."
    )
    loss_type: str = Field(
        default="sigmoid",
        description="DPO loss variant: sigmoid, hinge, ipo."
    )

    @model_validator(mode="after")
    def validate_loss(self) -> "DpoConfig":
        if self.loss_type not in ("sigmoid", "hinge", "ipo"):
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        return self


class DistillationConfig(BaseModel):
    """Arguments for knowledge distillation."""
    teacher_model: str = Field(..., description="Name/path of the teacher model.")
    student_model: str = Field(..., description="Name/path of the student model.")
    temperature: float = Field(default=2.0, gt=0, le=100.0)
    alpha_ce: float = Field(default=0.5, ge=0, le=1.0, description="Weight for hard-label cross-entropy.")
    alpha_kd: float = Field(default=0.5, ge=0, le=1.0, description="Weight for KL-divergence distillation loss.")
    epochs: int = Field(..., ge=1, le=100)
    batch_size: int = Field(..., ge=1, le=4096)

    @model_validator(mode="after")
    def weights_sum(self) -> "DistillationConfig":
        total = self.alpha_ce + self.alpha_kd
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"alpha_ce + alpha_kd must sum to ~1.0, got {total}")
        return self


class MergingConfig(BaseModel):
    """Arguments for model merging."""
    models: List[str] = Field(
        ...,
        min_length=2,
        description="List of model names/paths to merge."
    )
    method: str = Field(
        default="linear",
        description="Merge method: linear, slerp, ties, dare."
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Per-model weights for linear/ties merge. Must sum to ~1.0."
    )
    density: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="For TIES/DARE: density of parameters to keep."
    )

    @model_validator(mode="after")
    def validate_merge(self) -> "MergingConfig":
        valid_methods = ("linear", "slerp", "ties", "dare")
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError("weights length must match models length")
            total = sum(self.weights)
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"weights must sum to ~1.0, got {total}")
        return self


class PruningConfig(BaseModel):
    """Arguments for weight pruning."""
    sparsity: float = Field(..., gt=0, lt=1.0, description="Target sparsity ratio (e.g. 0.5 = 50% zeros).")
    method: str = Field(
        default="magnitude",
        description="Pruning method: magnitude, movement, structured."
    )
    granularity: str = Field(
        default="unstructured",
        description="Granularity: unstructured, row, column, block."
    )

    @model_validator(mode="after")
    def validate_pruning(self) -> "PruningConfig":
        valid_methods = ("magnitude", "movement", "structured")
        valid_granularity = ("unstructured", "row", "column", "block")
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        if self.granularity not in valid_granularity:
            raise ValueError(f"granularity must be one of {valid_granularity}")
        return self


class EvalConfig(BaseModel):
    """Arguments for evaluation (terminal operation)."""
    benchmarks: List[str] = Field(
        ...,
        min_length=1,
        description="Benchmark names to run, e.g. ['mmlu', 'hellaswag', 'arc']."
    )
    num_few_shot: int = Field(default=5, ge=0, le=25)


# ----------------------------
# 3) Unified training step
# ----------------------------

class TrainingStep(BaseModel):
    """A single training operation with validated, op-specific arguments."""
    op: TrainingOp
    base_model: str = Field(..., description="Base model name/path to operate on.")
    dataset: Optional[str] = Field(default=None, description="Dataset name/path (not required for merging/pruning/evaluate).")
    output_path: Optional[str] = Field(default=None, description="Where to save results.")
    why_compact: str = Field(..., description="Short justification, <= 160 chars.")

    # Op-specific configs (only the relevant one should be populated)
    lora_config: Optional[LoraConfig] = None
    full_finetune_config: Optional[FullFinetuneConfig] = None
    chinchilla_config: Optional[ChinchillaConfig] = None
    dpo_config: Optional[DpoConfig] = None
    distillation_config: Optional[DistillationConfig] = None
    merging_config: Optional[MergingConfig] = None
    pruning_config: Optional[PruningConfig] = None
    eval_config: Optional[EvalConfig] = None

    @model_validator(mode="after")
    def validate_step(self) -> "TrainingStep":
        if len(self.why_compact) > 160:
            raise ValueError("why_compact too long")

        # Map each op to its required config field
        op_config_map = {
            TrainingOp.LORA: ("lora_config", True),
            TrainingOp.QLORA: ("lora_config", True),       # QLoRA uses LoraConfig with quantize_bits=4
            TrainingOp.DORA: ("lora_config", True),         # DoRA uses LoraConfig with use_dora=True
            TrainingOp.FULL_FINETUNE: ("full_finetune_config", True),
            TrainingOp.CHINCHILLA: ("chinchilla_config", True),
            TrainingOp.DPO: ("dpo_config", True),
            TrainingOp.DISTILLATION: ("distillation_config", True),
            TrainingOp.MERGING: ("merging_config", True),
            TrainingOp.PRUNING: ("pruning_config", True),
            TrainingOp.EVALUATE: ("eval_config", True),
        }

        required_field, needs_it = op_config_map[self.op]
        config_val = getattr(self, required_field)

        if needs_it and config_val is None:
            raise ValueError(f"{self.op.value} requires {required_field}")

        # Enforce QLoRA must have quantize_bits=4
        if self.op == TrainingOp.QLORA:
            if config_val.quantize_bits != 4:
                raise ValueError("QLoRA requires quantize_bits=4")

        # Enforce DoRA must have use_dora=True
        if self.op == TrainingOp.DORA:
            if not config_val.use_dora:
                raise ValueError("DoRA requires use_dora=True")

        # Dataset required for training ops (not merging/pruning/evaluate/chinchilla)
        needs_dataset = {
            TrainingOp.LORA, TrainingOp.QLORA, TrainingOp.DORA,
            TrainingOp.FULL_FINETUNE, TrainingOp.DPO, TrainingOp.DISTILLATION,
        }
        if self.op in needs_dataset and not self.dataset:
            raise ValueError(f"{self.op.value} requires a dataset")

        # Ensure only the relevant config is populated
        all_configs = [
            "lora_config", "full_finetune_config", "chinchilla_config",
            "dpo_config", "distillation_config", "merging_config",
            "pruning_config", "eval_config",
        ]
        populated = [c for c in all_configs if getattr(self, c) is not None]
        if len(populated) > 1:
            raise ValueError(f"Only one config should be set, got: {populated}")

        return self


# ----------------------------
# 4) Training proof plan
# ----------------------------

class TrainingPlan(BaseModel):
    """
    A single-step training proof object.
    Mirrors ProofPlan from the main agent algebra.
    """
    step: TrainingStep
    compact_goal: str = Field(..., description="What this training step achieves, <= 120 chars.")
    safety_invariant: str = Field(
        ...,
        description="Must mention that only enum training ops with validated configs may execute."
    )
    estimated_compute: Optional[str] = Field(
        default=None,
        description="Optional: estimated GPU-hours or FLOPs."
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "TrainingPlan":
        if len(self.compact_goal) > 120:
            raise ValueError("compact_goal too long")
        text = self.safety_invariant.lower()
        for token in ["enum", "validated"]:
            if token not in text:
                raise ValueError(f"safety_invariant must mention '{token}'")
        return self


# ----------------------------
# 5) Deterministic executor
# ----------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class TrainingResult:
    ok: bool
    op: str
    summary: str
    result_hash: str
    artifacts: Dict[str, Any]


def _chinchilla_compute(config: ChinchillaConfig) -> Dict[str, Any]:
    """
    Chinchilla scaling law: C ≈ 6 * N * D
    where C = FLOPs, N = params, D = tokens.
    Optimal ratio: D/N ≈ 20 (Hoffmann et al., 2022).
    """
    C = config.compute_budget_flops
    results = {}

    if config.model_params is not None:
        N = config.model_params
        optimal_tokens = C / (6 * N)
        results["given_params"] = N
        results["optimal_tokens"] = optimal_tokens
        results["tokens_per_param_ratio"] = optimal_tokens / N

    if config.target_tokens is not None:
        D = config.target_tokens
        optimal_params = C / (6 * D)
        results["given_tokens"] = D
        results["optimal_params"] = optimal_params
        results["params_per_token_ratio"] = optimal_params / D

    # Compute-optimal point (balanced)
    import math
    optimal_N = math.sqrt(C / (6 * 20))  # D/N = 20
    optimal_D = 20 * optimal_N
    results["chinchilla_optimal"] = {
        "params": optimal_N,
        "tokens": optimal_D,
        "tokens_per_param": 20.0,
    }

    return results


def execute_training_step(step: TrainingStep) -> TrainingResult:
    """
    Execute a validated training step.
    Currently returns structured plans/configs rather than actually training.
    This is the executor stub — replace internals with real training calls.
    """
    op = step.op

    if op in (TrainingOp.LORA, TrainingOp.QLORA, TrainingOp.DORA):
        config = step.lora_config
        label = op.value.upper()
        summary = (
            f"{label}: rank={config.rank}, alpha={config.alpha}, "
            f"targets={config.target_modules}, dropout={config.dropout}"
        )
        if config.quantize_bits:
            summary += f", {config.quantize_bits}-bit quantized"
        if config.use_dora:
            summary += ", weight-decomposed (DoRA)"
        summary += f" on {step.base_model} with dataset {step.dataset}"
        artifacts = {"config": config.model_dump(), "base_model": step.base_model}

    elif op == TrainingOp.FULL_FINETUNE:
        config = step.full_finetune_config
        summary = (
            f"Full finetune: lr={config.learning_rate}, epochs={config.epochs}, "
            f"batch={config.batch_size} on {step.base_model} with {step.dataset}"
        )
        artifacts = {"config": config.model_dump(), "base_model": step.base_model}

    elif op == TrainingOp.CHINCHILLA:
        config = step.chinchilla_config
        results = _chinchilla_compute(config)
        summary = f"Chinchilla scaling analysis for budget={config.compute_budget_flops:.2e} FLOPs"
        artifacts = {"scaling_results": results}

    elif op == TrainingOp.DPO:
        config = step.dpo_config
        summary = (
            f"DPO: beta={config.beta}, lr={config.learning_rate}, "
            f"epochs={config.epochs}, loss={config.loss_type} on {step.base_model}"
        )
        artifacts = {"config": config.model_dump(), "base_model": step.base_model}

    elif op == TrainingOp.DISTILLATION:
        config = step.distillation_config
        summary = (
            f"Distill {config.teacher_model} -> {config.student_model}, "
            f"T={config.temperature}, alpha_ce={config.alpha_ce}, alpha_kd={config.alpha_kd}"
        )
        artifacts = {"config": config.model_dump()}

    elif op == TrainingOp.MERGING:
        config = step.merging_config
        summary = (
            f"Merge {len(config.models)} models via {config.method}, "
            f"density={config.density}"
        )
        artifacts = {"config": config.model_dump()}

    elif op == TrainingOp.PRUNING:
        config = step.pruning_config
        summary = (
            f"Prune {step.base_model}: sparsity={config.sparsity}, "
            f"method={config.method}, granularity={config.granularity}"
        )
        artifacts = {"config": config.model_dump(), "base_model": step.base_model}

    elif op == TrainingOp.EVALUATE:
        config = step.eval_config
        summary = (
            f"Evaluate {step.base_model} on {config.benchmarks}, "
            f"{config.num_few_shot}-shot"
        )
        artifacts = {"config": config.model_dump(), "base_model": step.base_model}

    else:
        raise ValueError(f"Unsupported training op: {op}")

    return TrainingResult(
        ok=True,
        op=op.value,
        summary=summary,
        result_hash=sha256_text(summary),
        artifacts=artifacts,
    )


# ----------------------------
# 6) Schema export (for model)
# ----------------------------

def get_training_schemas() -> Dict[str, Any]:
    """Export all schemas for the model to reference."""
    return {
        "TrainingOp": [e.value for e in TrainingOp],
        "TrainingStep": TrainingStep.model_json_schema(),
        "TrainingPlan": TrainingPlan.model_json_schema(),
    }
