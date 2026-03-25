#!/usr/bin/env python3
# ⠠⠃⠗⠁⠊⠇⠇⠑ ⠠⠁⠇⠛⠑⠃⠗⠁  —  ⠺⠗⠊⠞⠞⠑⠝ ⠊⠝ ⠃⠗⠁⠊⠇⠇⠑
"""
⠠⠑⠠⠇⠠⠊⠼⠑ ⠠⠃⠗⠁⠊⠇⠇⠑ ⠠⠁⠇⠛⠑⠃⠗⠁  —  ⠠⠁⠭⠊⠕⠍⠁⠞⠊⠉ ⠠⠋⠕⠥⠝⠙⠁⠞⠊⠕⠝

[decoded: ELI5 Braille Algebra — Axiomatic Foundation]

=======================================================================
Axiomatic Basis
=======================================================================

Let **Train** be a category whose objects are model types (ModelType) and
whose morphisms are training operators A : X → Y.  Equip Train with:

  (i)   a behavioral pseudometric  d(x, y) = E_z[KL(f_x(z) || f_y(z))]
  (ii)  a tropical cost functor    c : Train → (ℝ₊, min, +)

From these structures we derive exactly 8 independent boolean predicates
on any morphism pair (A, B).  Each predicate maps to one dot of an 8-dot
braille cell (U+2800 – U+28FF).

  Axiom 1 — Composition (directional):
    P₁(A,B) := dom(A) = cod(B)          → Dot 1  (composable A∘B)
    P₅(A,B) := dom(B) = cod(A)          → Dot 5  (composable B∘A)

  Axiom 2 — Type structure:
    P₂(A,B) := P₁ ∨ dom(A)=dom(B) ∨ cod(A)=cod(B)  → Dot 2  (types align)
    P₆(A,B) := cod(A) = cod(B)          → Dot 6  (same output type)

  Axiom 3 — Commutativity (metric):
    P₃(A,B) := 𝔠(A,B) < ε              → Dot 3  (approximately commuting)
    where 𝔠(A,B) = d(A∘B, B∘A) is the commutator defect.
    Requires P₁ ∧ P₅; otherwise P₃ := False.

  Axiom 4 — Idempotence (metric):
    P₄(A,B) := δ_idem(A) < δ  ∨  δ_idem(B) < δ     → Dot 4
    where δ_idem(T) = d(T∘T, T).  Requires T to be an endomorphism;
    otherwise δ_idem(T) := ∞.

  Axiom 5 — Invertibility (metric):
    P₇(A,B) := δ_inv(A) < δ  ∨  δ_inv(B) < δ        → Dot 7
    where δ_inv(T) = d(T⁻¹∘T, I).  If no candidate T⁻¹ exists,
    δ_inv(T) := ∞.

  Axiom 6 — Resource-boundedness (tropical):
    P₈(A,B) := c(A) ⊕ c(B) ≤ θ         → Dot 8  (cost-efficient)
    where ⊕ is addition in the tropical semiring on compute,
    max on memory.

Completeness Theorem
--------------------
The classifying map

    χ : Mor(Train) × Mor(Train) → 𝔹⁸ ≅ {U+2800, …, U+28FF}

sending (A, B) ↦ (P₁, P₂, …, P₈) is a **complete boolean invariant** of
the pair up to compositional, type-theoretic, behavioral, structural,
and resource equivalence.  Any further boolean predicate on (A, B) is
either derivable from {P₁…P₈} or requires more than 1 bit of
measurement data.

The 256-element codomain is isomorphic to the Unicode 8-dot braille
block.  The braille cell IS the classifying object.

=======================================================================

  ⣿ (⠁⠇⠇ ⠼⠓ ⠙⠕⠞⠎) = ⠏⠑⠗⠋⠑⠉⠞ ⠏⠁⠊⠗
  ⠀ (⠃⠇⠁⠝⠅)       = ⠊⠝⠉⠕⠍⠏⠁⠞⠊⠃⠇⠑
  ⡇ (⠇⠑⠋⠞ ⠉⠕⠇)    = ⠕⠝⠑-⠺⠁⠽

⠠⠮ ⠛⠗⠊⠙ ⠛⠗⠕⠺⠎ ⠊⠝⠋⠊⠝⠊⠞⠑⠇⠽ ⠁⠎ ⠠⠁⠗⠊⠁ ⠎⠽⠝⠮⠎⠊⠵⠑⠎ ⠝⠑⠺ ⠞⠕⠕⠇⠎⠲
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from braille_stream import to_braille, from_braille, braid, unbraid
from metric_algebra import (
    ModelType, Precision, AdapterMethod,
    DENSE_FP16, QUANT4, STUDENT_TYPE, LORA_TYPE, SPARSE_TYPE,
    OperatorCost,
)

BRAILLE_BASE = 0x2800
INF = float("inf")


# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
# ⠠⠃⠗⠁⠊⠇⠇⠑ ⠠⠙⠥⠁⠇ — ⠮ ⠋⠥⠝⠙⠁⠍⠑⠝⠞⠁⠇ ⠞⠽⠏⠑
# [decoded: Braille Dual — the fundamental type]
# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶


@dataclass(frozen=True)
class BrailleDual:
    """⠠⠁ ⠎⠞⠗⠊⠝⠛ ⠞⠓⠁⠞ ⠑⠭⠊⠎⠞⠎ ⠊⠝ ⠃⠕⠞⠓ ⠃⠗⠁⠊⠇⠇⠑ ⠯ ⠏⠇⠁⠊⠝⠞⠑⠭⠞⠲
    [decoded: A string that exists in both braille & plaintext.
     The braille form is canonical; the plaintext is derived.]"""
    text: str
    braided: str

    def __str__(self):
        return self.braided

    def decode(self) -> str:
        return self.text

    def __repr__(self):
        return f"⠃⠗({self.braided!r})"


def br(text: str) -> BrailleDual:
    """⠠⠉⠗⠑⠁⠞⠑ ⠁ ⠃⠗⠁⠊⠇⠇⠑ ⠙⠥⠁⠇ ⠋⠗⠕⠍ ⠏⠇⠁⠊⠝⠞⠑⠭⠞⠲
    [decoded: Create a braille dual from plaintext.]"""
    return BrailleDual(text=text, braided=braid(text))


# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
# ⠼⠁) ⠠⠙⠕⠞ ⠠⠎⠑⠍⠁⠝⠞⠊⠉⠎ — ⠺⠓⠁⠞ ⠑⠁⠉⠓ ⠕⠋ ⠮ ⠼⠓ ⠙⠕⠞⠎ ⠍⠑⠁⠝⠎
# [decoded: 1) Dot Semantics — what each of the 8 dots means]
# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶

class Dot:
    """⠠⠮ ⠼⠓ ⠙⠕⠞⠎ ⠕⠋ ⠁ ⠃⠗⠁⠊⠇⠇⠑ ⠉⠑⠇⠇⠂ ⠑⠁⠉⠓ ⠑⠝⠉⠕⠙⠊⠝⠛ ⠕⠝⠑ ⠁⠇⠛⠑⠃⠗⠁⠊⠉ ⠏⠗⠕⠏⠑⠗⠞⠽⠲
    [decoded: The 8 dots of a braille cell, each encoding one algebraic property.]"""
    COMPOSABLE_AB  = 0  # ⠙⠕⠞ ⠁: ⠉⠁⠝ ⠺⠑ ⠙⠕ ⠠⠁ ⠮⠝ ⠠⠃?
    TYPES_MATCH    = 1  # ⠙⠕⠞ ⠃: ⠙⠕ ⠊⠝⠞⠑⠗⠍⠑⠙⠊⠁⠞⠑ ⠞⠽⠏⠑⠎ ⠁⠇⠊⠛⠝?
    COMMUTATOR_LOW = 2  # ⠙⠕⠞ ⠉: ⠙⠕⠑⠎ ⠕⠗⠙⠑⠗ ⠠⠝⠠⠕⠠⠞ ⠍⠁⠞⠞⠑⠗?
    IDEMPOTENT     = 3  # ⠙⠕⠞ ⠛: ⠊⠎ ⠠⠁∘⠠⠁ ≈ ⠠⠁?
    COMPOSABLE_BA  = 4  # ⠙⠕⠞ ⠙: ⠉⠁⠝ ⠺⠑ ⠙⠕ ⠠⠃ ⠮⠝ ⠠⠁?
    SAME_OUTPUT    = 5  # ⠙⠕⠞ ⠑: ⠎⠁⠍⠑ ⠕⠥⠞⠏⠥⠞ ⠞⠽⠏⠑?
    INVERSE_EXISTS = 6  # ⠙⠕⠞ ⠋: ⠓⠁⠎ ⠊⠝⠧⠑⠗⠎⠑?
    COST_EFFICIENT = 7  # ⠙⠕⠞ ⠓: ⠉⠕⠎⠞-⠑⠋⠋⠊⠉⠊⠑⠝⠞?

    # ⠠⠃⠗⠁⠊⠇⠇⠑ ⠙⠕⠞ ⠝⠥⠍⠃⠑⠗⠊⠝⠛ → ⠃⠊⠞ ⠏⠕⠎⠊⠞⠊⠕⠝
    # [decoded: Braille dot numbering → bit position]
    NAMES = {
        0: "⠉⠕⠍⠏⠕⠎⠁⠃⠇⠑ ⠠⠁→⠠⠃",
        1: "⠞⠽⠏⠑⠎ ⠁⠇⠊⠛⠝",
        2: "⠕⠗⠙⠑⠗ ⠙⠕⠑⠎⠝⠔⠞ ⠍⠁⠞⠞⠑⠗",
        3: "⠊⠙⠑⠍⠏⠕⠞⠑⠝⠞",
        4: "⠉⠕⠍⠏⠕⠎⠁⠃⠇⠑ ⠠⠃→⠠⠁",
        5: "⠎⠁⠍⠑ ⠕⠥⠞⠏⠥⠞ ⠞⠽⠏⠑",
        6: "⠓⠁⠎ ⠊⠝⠧⠑⠗⠎⠑",
        7: "⠉⠕⠎⠞-⠑⠋⠋⠊⠉⠊⠑⠝⠞",
    }

    # [decoded names: composable A→B, types align, order doesn't matter,
    #  idempotent, composable B→A, same output type, has inverse, cost-efficient]
    NAMES_DECODED = {
        0: "composable A→B",
        1: "types align",
        2: "order doesn't matter",
        3: "idempotent",
        4: "composable B→A",
        5: "same output type",
        6: "has inverse",
        7: "cost-efficient",
    }


def dots_to_braille(dots: List[int]) -> str:
    """⠙⠕⠞⠎ → ⠃⠗⠁⠊⠇⠇⠑ ⠉⠓⠁⠗⠁⠉⠞⠑⠗⠲
    [decoded: Convert active dot numbers (0-7) to a braille character.]"""
    byte_val = 0
    for d in dots:
        byte_val |= (1 << d)
    return chr(BRAILLE_BASE + byte_val)


def braille_to_dots(ch: str) -> List[int]:
    """⠃⠗⠁⠊⠇⠇⠑ ⠉⠓⠁⠗⠁⠉⠞⠑⠗ → ⠙⠕⠞⠎⠲
    [decoded: Decode a braille character back to its active dot numbers.]"""
    byte_val = ord(ch) - BRAILLE_BASE
    return [i for i in range(8) if byte_val & (1 << i)]


def dots_to_byte(dots: List[int]) -> int:
    """⠙⠕⠞⠎ → ⠃⠽⠞⠑ ⠧⠁⠇⠥⠑⠲
    [decoded: Convert active dots to the byte value.]"""
    return sum(1 << d for d in dots)


# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
# ⠼⠃) ⠠⠕⠏⠑⠗⠁⠞⠕⠗ ⠠⠑⠝⠞⠗⠽ — ⠺⠓⠁⠞ ⠺⠑ ⠅⠝⠕⠺ ⠁⠃⠕⠥⠞ ⠑⠁⠉⠓ ⠕⠏⠑⠗⠁⠞⠕⠗
# [decoded: 2) Operator Entry — what we know about each operator]
# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶

@dataclass
class OperatorInfo:
    """⠠⠑⠧⠑⠗⠽⠮⠊⠝⠛ ⠺⠑ ⠅⠝⠕⠺ ⠁⠃⠕⠥⠞ ⠁ ⠞⠗⠁⠊⠝⠊⠝⠛ ⠕⠏⠑⠗⠁⠞⠕⠗ ⠿ ⠮ ⠃⠗⠁⠊⠇⠇⠑ ⠁⠇⠛⠑⠃⠗⠁⠲
    [decoded: Everything we know about a training operator for the braille algebra.
     domain/codomain are now typed ModelType objects (category objects).
     has_inverse and is_idempotent are *priors* — overridden by measurements
     when available (δ_inv < δ and δ_idem < δ respectively).]"""
    name: str                          # ⠎⠓⠕⠗⠞ ⠝⠁⠍⠑
    full_name: str                     # ⠓⠥⠍⠁⠝ ⠝⠁⠍⠑
    emoji: str                         # ⠿ ⠠⠑⠠⠇⠠⠊⠼⠑ ⠙⠊⠎⠏⠇⠁⠽
    domain: ModelType                  # ⠊⠝⠏⠥⠞ ⠞⠽⠏⠑ (⠉⠁⠞⠑⠛⠕⠗⠽ ⠕⠃⠚⠑⠉⠞)
    codomain: ModelType                # ⠕⠥⠞⠏⠥⠞ ⠞⠽⠏⠑ (⠉⠁⠞⠑⠛⠕⠗⠽ ⠕⠃⠚⠑⠉⠞)
    has_inverse: bool = False          # ⠏⠗⠊⠕⠗: ⠓⠁⠎ ⠊⠝⠧⠑⠗⠎⠑?  (⠕⠧⠑⠗⠗⠊⠙⠙⠑⠝ ⠃⠽ δ_inv)
    is_idempotent: bool = False        # ⠏⠗⠊⠕⠗: ⠊⠎ ⠠⠞∘⠠⠞ ≈ ⠠⠞? (⠕⠧⠑⠗⠗⠊⠙⠙⠑⠝ ⠃⠽ δ_idem)
    cost_budget: float = 1.0           # ⠝⠕⠗⠍⠁⠇⠊⠵⠑⠙ ⠉⠕⠎⠞ c(T) ∈ [0, 1]
    eli5: str = ""                     # ⠕⠝⠑-⠎⠑⠝⠞⠑⠝⠉⠑ ⠑⠭⠏⠇⠁⠝⠁⠞⠊⠕⠝

    @property
    def is_endomorphism(self) -> bool:
        """⠠⠞ : ⠠⠭ → ⠠⠭ (⠙⠕⠍⠁⠊⠝ = ⠉⠕⠙⠕⠍⠁⠊⠝)⠲
        [decoded: T is an endomorphism iff domain == codomain.]"""
        return self.domain == self.codomain

    @property
    def domain_str(self) -> str:
        """⠠⠃⠁⠉⠅⠺⠁⠗⠙-⠉⠕⠍⠏⠁⠞ ⠎⠞⠗⠊⠝⠛ ⠋⠕⠗ ⠎⠑⠗⠊⠁⠇⠊⠵⠁⠞⠊⠕⠝⠲
        [decoded: Backward-compat string for serialization.]"""
        return repr(self.domain)

    @property
    def codomain_str(self) -> str:
        return repr(self.codomain)


@dataclass
class MeasurementCache:
    """⠠⠍⠑⠁⠎⠥⠗⠑⠙ ⠙⠑⠋⠑⠉⠞⠎ ⠿ ⠕⠏⠑⠗⠁⠞⠕⠗⠎ — ⠕⠧⠑⠗⠗⠊⠙⠑ ⠏⠗⠊⠕⠗ ⠙⠑⠉⠇⠁⠗⠁⠞⠊⠕⠝⠎⠲
    [decoded: Measured defects for operators — override prior declarations.
     INF means "not yet measured"; the axioms fall back to declared priors.]"""
    # Unary defects per operator name
    idempotence_defects: Dict[str, float] = dc_field(default_factory=dict)  # δ_idem(T)
    inverse_defects: Dict[str, float] = dc_field(default_factory=dict)      # δ_inv(T)
    # Pairwise defects (name_a, name_b) → 𝔠(A,B)
    commutator_defects: Dict[Tuple[str, str], float] = dc_field(default_factory=dict)

    def get_idem(self, name: str) -> float:
        """⠠⠛⠑⠞ δ_idem(T) ⠕⠗ ⠊⠝⠋⠊⠝⠊⠞⠽ ⠊⠋ ⠝⠕⠞ ⠍⠑⠁⠎⠥⠗⠑⠙⠲
        [decoded: Get δ_idem(T) or INF if not measured.]"""
        return self.idempotence_defects.get(name, INF)

    def get_inv(self, name: str) -> float:
        """⠠⠛⠑⠞ δ_inv(T) ⠕⠗ ⠊⠝⠋⠊⠝⠊⠞⠽ ⠊⠋ ⠝⠕⠞ ⠍⠑⠁⠎⠥⠗⠑⠙⠲
        [decoded: Get δ_inv(T) or INF if not measured.]"""
        return self.inverse_defects.get(name, INF)

    def get_comm(self, name_a: str, name_b: str) -> float:
        """⠠⠛⠑⠞ 𝔠(A,B) ⠕⠗ ⠊⠝⠋⠊⠝⠊⠞⠽ ⠊⠋ ⠝⠕⠞ ⠍⠑⠁⠎⠥⠗⠑⠙⠲
        [decoded: Get 𝔠(A,B) or INF if not measured.]"""
        return self.commutator_defects.get((name_a, name_b), INF)


# ⠠⠎⠞⠁⠝⠙⠁⠗⠙ ⠛⠑⠝⠑⠗⠁⠞⠕⠗ ⠎⠑⠞ 𝒢 — ⠝⠕⠺ ⠾ ⠞⠽⠏⠑⠙ ⠠⠍⠕⠙⠑⠇⠠⠞⠽⠏⠑ ⠕⠃⠚⠑⠉⠞⠎
# [decoded: Standard generator set — now with typed ModelType objects]
GENERATORS: List[OperatorInfo] = [
    OperatorInfo("U", "⠠⠛⠗⠁⠙⠊⠑⠝⠞ ⠠⠥⠏⠙⠁⠞⠑", "📚",
                 DENSE_FP16, DENSE_FP16,
                 has_inverse=False, is_idempotent=False, cost_budget=0.3,
                 eli5="⠞⠑⠁⠉⠓⠊⠝⠛ ⠮ ⠍⠕⠙⠑⠇ ⠝⠑⠺ ⠮⠊⠝⠛⠎⠂ ⠇⠊⠅⠑ ⠎⠞⠥⠙⠽⠊⠝⠛ ⠋⠇⠁⠎⠓⠉⠁⠗⠙⠎"),
    OperatorInfo("L", "⠠⠇⠕⠠⠗⠠⠁ ⠠⠁⠞⠞⠁⠉⠓", "🧩",
                 DENSE_FP16, LORA_TYPE(16),
                 has_inverse=True, is_idempotent=False, cost_budget=0.1,
                 eli5="⠎⠝⠁⠏⠏⠊⠝⠛ ⠕⠝ ⠁ ⠎⠍⠁⠇⠇ ⠓⠑⠇⠏⠑⠗ ⠃⠗⠁⠊⠝ ⠞⠓⠁⠞ ⠇⠑⠁⠗⠝⠎ ⠮ ⠝⠑⠺ ⠎⠞⠥⠋⠋"),
    OperatorInfo("M", "⠠⠍⠑⠗⠛⠑", "🔗",
                 LORA_TYPE(16), DENSE_FP16,
                 has_inverse=False, is_idempotent=True, cost_budget=0.05,
                 eli5="⠛⠇⠥⠊⠝⠛ ⠮ ⠓⠑⠇⠏⠑⠗ ⠃⠗⠁⠊⠝ ⠃⠁⠉⠅ ⠊⠝⠞⠕ ⠮ ⠍⠁⠊⠝ ⠃⠗⠁⠊⠝"),
    OperatorInfo("Q", "⠠⠡⠥⠁⠝⠞⠊⠵⠑", "📦",
                 DENSE_FP16, QUANT4,
                 has_inverse=False, is_idempotent=True, cost_budget=0.2,
                 eli5="⠎⠓⠗⠊⠝⠅⠊⠝⠛ ⠮ ⠍⠕⠙⠑⠇ ⠞⠕ ⠋⠊⠞ ⠊⠝ ⠁ ⠎⠍⠁⠇⠇⠑⠗ ⠃⠕⠭"),
    OperatorInfo("P", "⠠⠏⠗⠥⠝⠑", "✂️",
                 DENSE_FP16, SPARSE_TYPE(0.5),
                 has_inverse=False, is_idempotent=False, cost_budget=0.15,
                 eli5="⠉⠥⠞⠞⠊⠝⠛ ⠁⠺⠁⠽ ⠏⠁⠗⠞⠎ ⠮ ⠍⠕⠙⠑⠇ ⠙⠕⠑⠎⠝⠔⠞ ⠗⠑⠁⠇⠇⠽ ⠝⠑⠑⠙"),
    OperatorInfo("D", "⠠⠙⠊⠎⠞⠊⠇⠇", "🍯",
                 DENSE_FP16, STUDENT_TYPE,
                 has_inverse=False, is_idempotent=False, cost_budget=0.8,
                 eli5="⠁ ⠃⠊⠛ ⠍⠕⠙⠑⠇ ⠞⠑⠁⠉⠓⠊⠝⠛ ⠁ ⠇⠊⠞⠞⠇⠑ ⠍⠕⠙⠑⠇ ⠊⠞⠎ ⠎⠑⠉⠗⠑⠞⠎"),
    OperatorInfo("I", "⠠⠊⠙⠑⠝⠞⠊⠞⠽", "🪞",
                 DENSE_FP16, DENSE_FP16,
                 has_inverse=True, is_idempotent=True, cost_budget=0.0,
                 eli5="⠙⠕⠊⠝⠛ ⠝⠕⠮⠊⠝⠛ — ⠮ ⠍⠕⠙⠑⠇ ⠎⠞⠁⠽⠎ ⠑⠭⠁⠉⠞⠇⠽ ⠮ ⠎⠁⠍⠑"),
]
# [decoded full_names: Gradient Update, LoRA Attach, Merge, Quantize, Prune, Distill, Identity]
# [decoded eli5s: teaching the model new things like studying flashcards,
#  snapping on a small helper brain that learns the new stuff,
#  gluing the helper brain back into the main brain,
#  shrinking the model to fit in a smaller box,
#  cutting away parts the model doesn't really need,
#  a big model teaching a little model its secrets,
#  doing nothing - the model stays exactly the same]


# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
# ⠼⠉) ⠠⠉⠑⠇⠇ ⠠⠉⠕⠍⠏⠥⠞⠁⠞⠊⠕⠝ — ⠙⠑⠞⠑⠗⠍⠊⠝⠑ ⠮ ⠃⠗⠁⠊⠇⠇⠑ ⠉⠓⠁⠗⠁⠉⠞⠑⠗ ⠿ (⠠⠁⠂ ⠠⠃)
# [decoded: 3) Cell Computation — determine the braille character for (A, B)]
# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶

@dataclass
class CellAnalysis:
    """⠠⠋⠥⠇⠇ ⠁⠝⠁⠇⠽⠎⠊⠎ ⠕⠋ ⠕⠝⠑ ⠉⠑⠇⠇ ⠊⠝ ⠮ ⠠⠝×⠠⠝ ⠃⠗⠁⠊⠇⠇⠑ ⠁⠇⠛⠑⠃⠗⠁⠲
    [decoded: Full analysis of one cell in the N×N braille algebra.]"""
    op_a: OperatorInfo
    op_b: OperatorInfo
    dots: List[int]
    braille: str
    properties: Dict[str, bool]
    commutator_defect: float = 0.0      # measured or estimated 𝔠(A,B)
    eli5_explanation: str = ""

    @property
    def dot_count(self) -> int:
        return len(self.dots)

    @property
    def compatibility_score(self) -> float:
        """⠼⠁⠲⠼⠁ (⠊⠝⠉⠕⠍⠏⠁⠞⠊⠃⠇⠑) ⠞⠕ ⠼⠁⠲⠼⠁ (⠏⠑⠗⠋⠑⠉⠞ ⠏⠁⠊⠗)⠲
        [decoded: 0.0 (incompatible) to 1.0 (perfect pair).]"""
        return self.dot_count / 8.0


def _resolve_idempotent(op: OperatorInfo, cache: MeasurementCache, threshold: float) -> bool:
    """⠠⠁⠭⠊⠕⠍ ⠼⠙: P₄ — δ_idem(T) < δ.
    [decoded: Axiom 4 resolution. Measurement overrides prior.
     Non-endomorphisms get δ_idem = ∞ (cannot self-compose).]"""
    measured = cache.get_idem(op.name)
    if measured < INF:
        return measured < threshold
    # No measurement — fall back to prior, but only if endomorphism
    if not op.is_endomorphism:
        return False
    return op.is_idempotent


def _resolve_invertible(op: OperatorInfo, cache: MeasurementCache, threshold: float) -> bool:
    """⠠⠁⠭⠊⠕⠍ ⠼⠑: P₇ — δ_inv(T) < δ.
    [decoded: Axiom 5 resolution. Measurement overrides prior.
     If no candidate inverse exists, δ_inv = ∞.]"""
    measured = cache.get_inv(op.name)
    if measured < INF:
        return measured < threshold
    # No measurement — fall back to declared prior
    return op.has_inverse


def compute_cell(
    a: OperatorInfo,
    b: OperatorInfo,
    cache: MeasurementCache = None,
    commutator_threshold: float = 0.1,
    defect_threshold: float = 0.1,
    cost_threshold: float = 0.7,
    # Backward compat: old callers may pass commutator_defect directly
    commutator_defect: float = -1.0,
) -> CellAnalysis:
    """⠠⠉⠕⠍⠏⠥⠞⠑ ⠮ ⠃⠗⠁⠊⠇⠇⠑ ⠉⠑⠇⠇ ⠿ ⠕⠏⠑⠗⠁⠞⠕⠗ ⠏⠁⠊⠗ (⠠⠁⠂ ⠠⠃)⠲
    ⠠⠁⠭⠊⠕⠍⠁⠞⠊⠉⠁⠇⠇⠽ ⠛⠗⠕⠥⠝⠙⠑⠙: ⠑⠁⠉⠓ ⠙⠕⠞ ⠊⠎ ⠁ ⠏⠗⠑⠙⠊⠉⠁⠞⠑ P_i⠲
    [decoded: Compute the braille cell for operator pair (A, B).
     Axiomatically grounded: each dot is a predicate Pᵢ.

     Predicates:
       P₁: dom(A) = cod(B)              → Dot 1 (composable A∘B)
       P₂: P₁ ∨ dom=dom ∨ cod=cod      → Dot 2 (types align)
       P₃: 𝔠(A,B) < ε  (requires P₁∧P₅) → Dot 3 (commuting)
       P₄: δ_idem(A)<δ ∨ δ_idem(B)<δ   → Dot 4 (idempotent)
       P₅: dom(B) = cod(A)              → Dot 5 (composable B∘A)
       P₆: cod(A) = cod(B)              → Dot 6 (same output)
       P₇: δ_inv(A)<δ ∨ δ_inv(B)<δ     → Dot 7 (invertible)
       P₈: c(A)+c(B) ≤ θ               → Dot 8 (cost-efficient)
    ]"""
    if cache is None:
        cache = MeasurementCache()

    # Migrate legacy commutator_defect parameter into cache
    if commutator_defect >= 0 and cache.get_comm(a.name, b.name) == INF:
        cache.commutator_defects[(a.name, b.name)] = commutator_defect

    dots = []
    props = {}

    # ── Axiom 1: Composition (directional) ──
    # P₁(A,B) := dom(A) = cod(B)  →  A∘B exists
    composable_ab = (a.domain == b.codomain)
    props["composable_ab"] = composable_ab
    if composable_ab:
        dots.append(Dot.COMPOSABLE_AB)

    # P₅(A,B) := dom(B) = cod(A)  →  B∘A exists
    composable_ba = (b.domain == a.codomain)
    props["composable_ba"] = composable_ba
    if composable_ba:
        dots.append(Dot.COMPOSABLE_BA)

    # ── Axiom 2: Type structure ──
    # P₂(A,B) := P₁ ∨ dom(A)=dom(B) ∨ cod(A)=cod(B)
    types_match = (composable_ab or a.domain == b.domain or a.codomain == b.codomain)
    props["types_align"] = types_match
    if types_match:
        dots.append(Dot.TYPES_MATCH)

    # P₆(A,B) := cod(A) = cod(B)
    same_output = (a.codomain == b.codomain)
    props["same_output"] = same_output
    if same_output:
        dots.append(Dot.SAME_OUTPUT)

    # ── Axiom 3: Commutativity (metric) ──
    # P₃(A,B) := 𝔠(A,B) < ε.  Requires P₁ ∧ P₅; otherwise P₃ := False.
    comm = cache.get_comm(a.name, b.name)
    if composable_ab and composable_ba:
        if comm < INF:
            commutes = comm < commutator_threshold
        else:
            # No measurement — estimate: both endomorphisms on same type → likely low
            commutes = (a.is_endomorphism and b.is_endomorphism
                        and a.domain == b.domain)
    else:
        commutes = False
    props["commutator_low"] = commutes
    if commutes:
        dots.append(Dot.COMMUTATOR_LOW)

    # ── Axiom 4: Idempotence (metric) ──
    # P₄(A,B) := δ_idem(A) < δ  ∨  δ_idem(B) < δ
    either_idempotent = (_resolve_idempotent(a, cache, defect_threshold)
                         or _resolve_idempotent(b, cache, defect_threshold))
    props["idempotent"] = either_idempotent
    if either_idempotent:
        dots.append(Dot.IDEMPOTENT)

    # ── Axiom 5: Invertibility (metric) ──
    # P₇(A,B) := δ_inv(A) < δ  ∨  δ_inv(B) < δ
    either_invertible = (_resolve_invertible(a, cache, defect_threshold)
                         or _resolve_invertible(b, cache, defect_threshold))
    props["inverse_exists"] = either_invertible
    if either_invertible:
        dots.append(Dot.INVERSE_EXISTS)

    # ── Axiom 6: Resource-boundedness (tropical) ──
    # P₈(A,B) := c(A) ⊕ c(B) ≤ θ
    combined_cost = a.cost_budget + b.cost_budget
    cost_ok = combined_cost <= cost_threshold
    props["cost_efficient"] = cost_ok
    if cost_ok:
        dots.append(Dot.COST_EFFICIENT)

    braille_char = dots_to_braille(dots)

    # The resolved commutator for the CellAnalysis
    resolved_comm = comm if comm < INF else commutator_defect

    # Generate ELI5 explanation
    eli5 = _make_eli5(a, b, props, resolved_comm)

    return CellAnalysis(
        op_a=a, op_b=b,
        dots=dots, braille=braille_char,
        properties=props,
        commutator_defect=resolved_comm,
        eli5_explanation=eli5,
    )


def _make_eli5(a: OperatorInfo, b: OperatorInfo, props: Dict[str, bool], defect: float) -> str:
    """⠠⠛⠑⠝⠑⠗⠁⠞⠑ ⠁⠝ ⠠⠑⠠⠇⠠⠊⠼⠑ ⠑⠭⠏⠇⠁⠝⠁⠞⠊⠕⠝ ⠊⠝ ⠃⠗⠁⠊⠇⠇⠑⠲
    [decoded: Generate an ELI5 explanation in braille.]"""
    parts = []

    if props["composable_ab"] and props["composable_ba"]:
        parts.append(braid(f"{a.emoji}{a.name} and {b.emoji}{b.name} can go in either order"))
    elif props["composable_ab"]:
        parts.append(braid(f"{b.emoji}{b.name} first, then {a.emoji}{a.name} works"))
    elif props["composable_ba"]:
        parts.append(braid(f"{a.emoji}{a.name} first, then {b.emoji}{b.name} works"))
    else:
        parts.append(braid(f"{a.emoji}{a.name} and {b.emoji}{b.name} don't connect"))

    if props["commutator_low"]:
        parts.append(braid("the order doesn't matter much"))
    elif defect > 0:
        parts.append(braid(f"order matters a lot (defect={defect:.3f})"))

    if props["cost_efficient"]:
        parts.append(braid("and it's cheap to do both"))

    return " — ".join(parts) + "⠲"


# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
# ⠼⠙) ⠠⠮ ⠠⠝×⠠⠝ ⠠⠃⠗⠁⠊⠇⠇⠑ ⠠⠛⠗⠊⠙ — ⠮ ⠁⠇⠛⠑⠃⠗⠁ ⠊⠞⠎⠑⠇⠋
# [decoded: 4) The N×N Braille Grid — the algebra itself]
# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶

class BrailleAlgebra:
    """⠠⠁⠝ ⠠⠝×⠠⠝ ⠃⠗⠁⠊⠇⠇⠑ ⠁⠇⠛⠑⠃⠗⠁ ⠕⠧⠑⠗ ⠞⠗⠁⠊⠝⠊⠝⠛ ⠕⠏⠑⠗⠁⠞⠕⠗⠎⠲

    ⠠⠮ ⠛⠗⠊⠙ ⠛⠗⠕⠺⠎ ⠊⠝⠋⠊⠝⠊⠞⠑⠇⠽ ⠁⠎ ⠝⠑⠺ ⠕⠏⠑⠗⠁⠞⠕⠗⠎ ⠁⠗⠑ ⠗⠑⠛⠊⠎⠞⠑⠗⠑⠙⠲
    ⠠⠑⠁⠉⠓ ⠉⠑⠇⠇ ⠊⠎ ⠁ ⠃⠗⠁⠊⠇⠇⠑ ⠉⠓⠁⠗⠁⠉⠞⠑⠗ ⠑⠝⠉⠕⠙⠊⠝⠛ ⠮ ⠁⠇⠛⠑⠃⠗⠁⠊⠉ ⠗⠑⠇⠁⠞⠊⠕⠝⠩⠊⠏
    ⠃⠑⠞⠺⠑⠑⠝ ⠮ ⠗⠕⠺ ⠕⠏⠑⠗⠁⠞⠕⠗ ⠯ ⠮ ⠉⠕⠇⠥⠍⠝ ⠕⠏⠑⠗⠁⠞⠕⠗⠲

    ⠠⠑⠠⠇⠠⠊⠼⠑: ⠊⠞⠔⠎ ⠁ ⠍⠥⠇⠞⠊⠏⠇⠊⠉⠁⠞⠊⠕⠝ ⠞⠁⠃⠇⠑ ⠺⠓⠑⠗⠑ ⠑⠁⠉⠓ ⠁⠝⠎⠺⠑⠗ ⠊⠎ ⠁
    ⠃⠗⠁⠊⠇⠇⠑ ⠏⠁⠞⠞⠑⠗⠝ ⠩⠁⠞ ⠎⠓⠕⠺⠎ ⠓⠕⠺ ⠺⠑⠇⠇ ⠞⠺⠕ ⠞⠗⠁⠊⠝⠊⠝⠛ ⠗⠑⠉⠊⠏⠑⠎
    ⠺⠕⠗⠅ ⠞⠕⠛⠑⠮⠑⠗⠲

    [decoded: An N×N braille algebra over training operators.
     The grid grows infinitely as new operators are registered.
     Each cell is a braille character encoding the algebraic relationship
     between the row operator and the column operator.
     ELI5: it's a multiplication table where each answer is a braille
     pattern that shows how well two training recipes work together.]
    """

    def __init__(self, operators: List[OperatorInfo] = None,
                 measurements: MeasurementCache = None):
        self._operators: List[OperatorInfo] = list(operators or GENERATORS)
        self._cells: Dict[Tuple[str, str], CellAnalysis] = {}
        self._measurements: MeasurementCache = measurements or MeasurementCache()
        # Backward compat: keep _defects as a view into the cache
        self._defects = self._measurements.commutator_defects
        self._recompute()

    @property
    def n(self) -> int:
        return len(self._operators)

    @property
    def operators(self) -> List[OperatorInfo]:
        return list(self._operators)

    @property
    def measurements(self) -> MeasurementCache:
        return self._measurements

    def _recompute(self):
        """⠗⠑⠉⠕⠍⠏⠥⠞⠑ ⠁⠇⠇ ⠉⠑⠇⠇⠎ ⠊⠝ ⠮ ⠛⠗⠊⠙⠲
        [decoded: Recompute all cells in the grid.]"""
        self._cells.clear()
        for a in self._operators:
            for b in self._operators:
                key = (a.name, b.name)
                self._cells[key] = compute_cell(a, b, cache=self._measurements)

    # -- Infinite expansion --

    def add_operator(self, op: OperatorInfo):
        """⠠⠁⠙⠙ ⠁ ⠝⠑⠺ ⠕⠏⠑⠗⠁⠞⠕⠗ — ⠛⠗⠕⠺⠎ ⠮ ⠛⠗⠊⠙ ⠃⠽ ⠕⠝⠑ ⠗⠕⠺ ⠯ ⠕⠝⠑ ⠉⠕⠇⠥⠍⠝⠲
        [decoded: Add a new operator — grows the grid by one row and one column.]"""
        if any(o.name == op.name for o in self._operators):
            raise ValueError(f"Operator '{op.name}' already exists")
        self._operators.append(op)
        self._recompute()

    # -- Measurement injection (hardened axioms) --

    def set_commutator_defect(self, name_a: str, name_b: str, defect: float):
        """⠠⠎⠑⠞ ⠁ ⠍⠑⠁⠎⠥⠗⠑⠙ ⠉⠕⠍⠍⠥⠞⠁⠞⠕⠗ ⠙⠑⠋⠑⠉⠞ 𝔠(A,B) ⠿ ⠁ ⠏⠁⠊⠗⠲
        [decoded: Set a measured commutator defect 𝔠(A,B) for a pair.
         Axiom 3: P₃ uses this to determine if order matters.]"""
        self._measurements.commutator_defects[(name_a, name_b)] = defect
        self._measurements.commutator_defects[(name_b, name_a)] = defect
        self._recompute()

    def set_idempotence_defect(self, name: str, defect: float):
        """⠠⠎⠑⠞ ⠁ ⠍⠑⠁⠎⠥⠗⠑⠙ ⠊⠙⠑⠍⠏⠕⠞⠑⠝⠉⠑ ⠙⠑⠋⠑⠉⠞ δ_idem(T) ⠿ ⠁⠝ ⠕⠏⠑⠗⠁⠞⠕⠗⠲
        [decoded: Set a measured idempotence defect δ_idem(T) for an operator.
         Axiom 4: P₄ uses this to determine if T∘T ≈ T.]"""
        self._measurements.idempotence_defects[name] = defect
        self._recompute()

    def set_inverse_defect(self, name: str, defect: float):
        """⠠⠎⠑⠞ ⠁ ⠍⠑⠁⠎⠥⠗⠑⠙ ⠊⠝⠧⠑⠗⠎⠑ ⠙⠑⠋⠑⠉⠞ δ_inv(T) ⠿ ⠁⠝ ⠕⠏⠑⠗⠁⠞⠕⠗⠲
        [decoded: Set a measured inverse defect δ_inv(T) for an operator.
         Axiom 5: P₇ uses this to determine if T⁻¹∘T ≈ I.]"""
        self._measurements.inverse_defects[name] = defect
        self._recompute()

    # -- Grid rendering --

    def get_cell(self, name_a: str, name_b: str) -> CellAnalysis:
        return self._cells[(name_a, name_b)]

    def to_braille_grid(self) -> str:
        """⠠⠗⠑⠝⠙⠑⠗ ⠮ ⠋⠥⠇⠇ ⠠⠝×⠠⠝ ⠛⠗⠊⠙ ⠁⠎ ⠁ ⠃⠗⠁⠊⠇⠇⠑ ⠎⠞⠗⠊⠝⠛ ⠃⠇⠕⠉⠅⠲
        [decoded: Render the full N×N grid as a braille string block.]"""
        names = [op.name for op in self._operators]
        # Header
        header = "  " + " ".join(f"{n:>2}" for n in names)
        lines = [header]
        for a in self._operators:
            row_chars = []
            for b in self._operators:
                cell = self._cells[(a.name, b.name)]
                row_chars.append(f" {cell.braille}")
            lines.append(f"{a.name:>2}" + "".join(row_chars))
        return "\n".join(lines)

    def to_braille_string(self) -> str:
        """⠠⠮ ⠛⠗⠊⠙ ⠁⠎ ⠁ ⠎⠊⠝⠛⠇⠑ ⠃⠗⠁⠊⠇⠇⠑ ⠎⠞⠗⠊⠝⠛ (⠠⠝² ⠉⠓⠁⠗⠎)⠲
        [decoded: The grid as a single braille string (N² chars).]"""
        chars = []
        for a in self._operators:
            for b in self._operators:
                chars.append(self._cells[(a.name, b.name)].braille)
        return "".join(chars)

    def to_emoji_grid(self) -> str:
        """⠠⠑⠠⠇⠠⠊⠼⠑: ⠑⠍⠕⠨⠊ + ⠃⠗⠁⠊⠇⠇⠑ ⠎⠊⠙⠑ ⠃⠽ ⠎⠊⠙⠑⠲
        [decoded: ELI5 version: emoji + braille side by side.]"""
        names = [op.name for op in self._operators]
        emojis = [op.emoji for op in self._operators]
        header = "     " + "  ".join(f"{e}" for e in emojis)
        lines = [header]
        for a in self._operators:
            row = []
            for b in self._operators:
                cell = self._cells[(a.name, b.name)]
                row.append(cell.braille)
            lines.append(f" {a.emoji}   " + "  ".join(row))
        return "\n".join(lines)

    def to_markdown_table(self) -> str:
        """⠠⠋⠥⠇⠇ ⠍⠁⠗⠅⠙⠕⠺⠝ ⠞⠁⠃⠇⠑ ⠾ ⠃⠗⠁⠊⠇⠇⠑ ⠉⠓⠁⠗⠁⠉⠞⠑⠗⠎⠲
        [decoded: Full markdown table with braille characters.]"""
        names = [op.name for op in self._operators]
        header = "| | " + " | ".join(f"**{n}**" for n in names) + " |"
        sep = "|---" * (len(names) + 1) + "|"
        rows = [header, sep]
        for a in self._operators:
            cells = []
            for b in self._operators:
                cell = self._cells[(a.name, b.name)]
                score = cell.dot_count
                cells.append(f"{cell.braille} ({score}/8)")
            rows.append(f"| **{a.name}** | " + " | ".join(cells) + " |")
        return "\n".join(rows)

    def eli5_explain(self, name_a: str, name_b: str) -> str:
        """⠠⠛⠑⠞ ⠮ ⠠⠑⠠⠇⠠⠊⠼⠑ ⠑⠭⠏⠇⠁⠝⠁⠞⠊⠕⠝ ⠿ ⠁ ⠎⠏⠑⠉⠊⠋⠊⠉ ⠉⠑⠇⠇⠲
        [decoded: Get the ELI5 explanation for a specific cell.]"""
        cell = self._cells[(name_a, name_b)]
        a = cell.op_a
        b = cell.op_b
        dots = cell.dots
        lines = [
            f"## {a.emoji} {a.full_name} × {b.emoji} {b.full_name}",
            f"**Braille:** {cell.braille}  ({cell.dot_count}/8 dots raised)",
            f"**Score:** {cell.compatibility_score:.0%} compatible",
            "",
            f"### What does {a.emoji} {a.name} do?",
            f"{a.eli5}",
            "",
            f"### What does {b.emoji} {b.name} do?",
            f"{b.eli5}",
            "",
            f"### Together?",
            f"{cell.eli5_explanation}",
            "",
            "### Dot-by-dot breakdown:",
        ]
        for i in range(8):
            raised = "⬤" if i in dots else "○"
            lines.append(f"  {raised} Dot {i+1}: {Dot.NAMES[i]}")
        return "\n".join(lines)

    def eli5_summary(self) -> str:
        """⠠⠑⠠⠇⠠⠊⠼⠑ ⠎⠥⠍⠍⠁⠗⠽ ⠕⠋ ⠮ ⠺⠓⠕⠇⠑ ⠁⠇⠛⠑⠃⠗⠁⠲
        [decoded: ELI5 summary of the whole algebra.]"""
        n = self.n
        total_cells = n * n
        total_dots = sum(c.dot_count for c in self._cells.values())
        max_dots = total_cells * 8
        density = total_dots / max_dots if max_dots > 0 else 0

        best_pair = max(self._cells.values(), key=lambda c: c.dot_count)
        worst_pair = min(self._cells.values(), key=lambda c: c.dot_count)

        return "\n".join([
            f"# 🧮 Braille Algebra — {n}×{n} grid ({n} operators)",
            f"",
            f"**What is this?** A multiplication table for AI training recipes.",
            f"Each cell is a braille character — more bumps means the recipes",
            f"work better together.",
            f"",
            f"**Size:** {n}×{n} = {total_cells} cells",
            f"**Density:** {density:.0%} ({total_dots}/{max_dots} dots raised)",
            f"**Best pair:** {best_pair.op_a.emoji}{best_pair.op_a.name} × "
            f"{best_pair.op_b.emoji}{best_pair.op_b.name} = "
            f"{best_pair.braille} ({best_pair.dot_count}/8)",
            f"**Worst pair:** {worst_pair.op_a.emoji}{worst_pair.op_a.name} × "
            f"{worst_pair.op_b.emoji}{worst_pair.op_b.name} = "
            f"{worst_pair.braille} ({worst_pair.dot_count}/8)",
            f"",
            f"The grid grows every time a new tool is created.",
        ])

    # -- Serialization --

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "operators": [
                {"name": op.name, "full_name": op.full_name, "emoji": op.emoji,
                 "domain": op.domain.model_dump(), "codomain": op.codomain.model_dump(),
                 "has_inverse": op.has_inverse, "is_idempotent": op.is_idempotent,
                 "cost_budget": op.cost_budget, "eli5": op.eli5}
                for op in self._operators
            ],
            "grid": {
                f"{a.name},{b.name}": {
                    "braille": self._cells[(a.name, b.name)].braille,
                    "dots": self._cells[(a.name, b.name)].dots,
                    "score": self._cells[(a.name, b.name)].dot_count,
                    "properties": self._cells[(a.name, b.name)].properties,
                }
                for a in self._operators for b in self._operators
            },
            "measurements": {
                "commutator_defects": {
                    f"{k[0]},{k[1]}": v
                    for k, v in self._measurements.commutator_defects.items()
                },
                "idempotence_defects": dict(self._measurements.idempotence_defects),
                "inverse_defects": dict(self._measurements.inverse_defects),
            },
            # Backward compat key
            "defects": {
                f"{k[0]},{k[1]}": v
                for k, v in self._measurements.commutator_defects.items()
            },
            "braille_string": self.to_braille_string(),
        }

    def save(self, path: Path = None):
        """⠠⠎⠁⠧⠑ ⠮ ⠁⠇⠛⠑⠃⠗⠁ ⠞⠕ ⠙⠊⠎⠅⠲  [decoded: Save the algebra to disk.]"""
        path = path or Path("braille_algebra.json")
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: Path = None) -> "BrailleAlgebra":
        """⠠⠇⠕⠁⠙ ⠮ ⠁⠇⠛⠑⠃⠗⠁ ⠋⠗⠕⠍ ⠙⠊⠎⠅⠲  [decoded: Load the algebra from disk.]"""
        path = path or Path("braille_algebra.json")
        data = json.loads(path.read_text())

        # Deserialize operators — domain/codomain may be dict (new) or str (old)
        ops = []
        for op_data in data["operators"]:
            d = dict(op_data)
            if isinstance(d.get("domain"), dict):
                d["domain"] = ModelType(**d["domain"])
            elif isinstance(d.get("domain"), str):
                d["domain"] = DENSE_FP16  # best-effort fallback for old files
            if isinstance(d.get("codomain"), dict):
                d["codomain"] = ModelType(**d["codomain"])
            elif isinstance(d.get("codomain"), str):
                d["codomain"] = DENSE_FP16
            ops.append(OperatorInfo(**d))

        # Rebuild MeasurementCache
        cache = MeasurementCache()
        meas = data.get("measurements", {})
        for key_str, val in meas.get("commutator_defects", {}).items():
            a, b = key_str.split(",")
            cache.commutator_defects[(a, b)] = val
        cache.idempotence_defects.update(meas.get("idempotence_defects", {}))
        cache.inverse_defects.update(meas.get("inverse_defects", {}))

        # Backward compat: old files only have "defects" (commutator only)
        if not meas and "defects" in data:
            for key_str, val in data["defects"].items():
                a, b = key_str.split(",")
                cache.commutator_defects[(a, b)] = val

        alg = BrailleAlgebra(ops, measurements=cache)
        return alg


# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
# ⠼⠑) ⠠⠃⠗⠊⠙⠛⠑: ⠍⠑⠞⠗⠊⠉_⠁⠇⠛⠑⠃⠗⠁⠲⠠⠗⠑⠇⠁⠞⠊⠕⠝⠠⠞⠁⠃⠇⠑ → ⠠⠃⠗⠁⠊⠇⠇⠑⠠⠁⠇⠛⠑⠃⠗⠁
# [decoded: 5) Bridge: metric_algebra.RelationTable → BrailleAlgebra]
# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶

def from_relation_table(
    operator_names: List[str],
    defects: List[List[float]],
    operator_infos: Dict[str, OperatorInfo] = None,
    idempotence_defects: Dict[str, float] = None,
    inverse_defects: Dict[str, float] = None,
) -> BrailleAlgebra:
    """⠠⠃⠥⠊⠇⠙ ⠁ ⠠⠃⠗⠁⠊⠇⠇⠑⠠⠁⠇⠛⠑⠃⠗⠁ ⠋⠗⠕⠍ ⠁ ⠍⠑⠞⠗⠊⠉_⠁⠇⠛⠑⠃⠗⠁ ⠠⠗⠑⠇⠁⠞⠊⠕⠝⠠⠞⠁⠃⠇⠑⠲
    ⠠⠍⠁⠏⠎ ⠍⠑⠁⠎⠥⠗⠑⠙ ⠙⠑⠋⠑⠉⠞⠎ ⠊⠝⠞⠕ ⠮ ⠃⠗⠁⠊⠇⠇⠑ ⠛⠗⠊⠙⠲
    [decoded: Build a BrailleAlgebra from a metric_algebra RelationTable.
     Maps measured defects into the braille grid.
     Now accepts all three measurement types for full axiomatic grounding.]"""
    # Build OperatorInfo lookup from generators
    known = {op.name: op for op in GENERATORS}
    if operator_infos:
        known.update(operator_infos)

    ops = []
    for name in operator_names:
        if name in known:
            ops.append(known[name])
        else:
            ops.append(OperatorInfo(
                name=name, full_name=name, emoji="🔧",
                domain=DENSE_FP16, codomain=DENSE_FP16,
                eli5=f"Operator {name}",
            ))

    # Build MeasurementCache upfront (avoids N² recomputes)
    cache = MeasurementCache()
    for i, name_a in enumerate(operator_names):
        for j, name_b in enumerate(operator_names):
            if i != j and defects[i][j] > 0:
                cache.commutator_defects[(name_a, name_b)] = defects[i][j]
    if idempotence_defects:
        cache.idempotence_defects.update(idempotence_defects)
    if inverse_defects:
        cache.inverse_defects.update(inverse_defects)

    return BrailleAlgebra(ops, measurements=cache)


# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
# ⠼⠋) ⠠⠊⠝⠞⠑⠗⠁⠉⠞⠊⠧⠑ ⠠⠓⠠⠞⠠⠍⠠⠇ ⠠⠗⠑⠝⠙⠑⠗⠑⠗
# [decoded: 6) Interactive HTML Renderer]
# ⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶

def render_html(algebra: BrailleAlgebra) -> str:
    """⠠⠛⠑⠝⠑⠗⠁⠞⠑ ⠁ ⠎⠞⠁⠝⠙⠁⠇⠕⠝⠑ ⠠⠓⠠⠞⠠⠍⠠⠇ ⠏⠁⠛⠑ ⠧⠊⠎⠥⠁⠇⠊⠵⠊⠝⠛ ⠮ ⠃⠗⠁⠊⠇⠇⠑ ⠁⠇⠛⠑⠃⠗⠁⠲
    [decoded: Generate a standalone HTML page visualizing the braille algebra.]"""
    data = algebra.to_dict()
    ops = algebra.operators
    n = algebra.n

    # Build the cell data as JSON for the JS
    cells_json = json.dumps(data["grid"], ensure_ascii=False)
    ops_json = json.dumps([
        {"name": op.name, "full_name": op.full_name, "emoji": op.emoji, "eli5": op.eli5}
        for op in ops
    ], ensure_ascii=False)

    # Build the braille grid rows
    grid_rows = []
    for a in ops:
        cells_html = []
        for b in ops:
            cell = algebra.get_cell(a.name, b.name)
            score = cell.dot_count
            # Color: green (high score) → red (low score)
            hue = int(score / 8 * 120)  # 0=red, 120=green
            color = f"hsl({hue}, 70%, 45%)"
            cells_html.append(
                f'<td class="cell" data-a="{a.name}" data-b="{b.name}" '
                f'style="color:{color}" title="{a.name}×{b.name}: {score}/8">'
                f'{cell.braille}</td>'
            )
        grid_rows.append(
            f'<tr><th class="row-hdr">{a.emoji} {a.name}</th>'
            + "".join(cells_html) + '</tr>'
        )

    header_cells = "".join(
        f'<th class="col-hdr">{op.emoji}<br>{op.name}</th>' for op in ops
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Braille Algebra — {n}×{n}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'SF Pro', -apple-system, system-ui, sans-serif;
    background: #0a0a0f; color: #e0e0e0;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 2rem;
  }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .subtitle {{ color: #888; margin-bottom: 1.5rem; font-size: 0.95rem; }}
  .grid-container {{
    overflow-x: auto; max-width: 95vw;
    border-radius: 12px; background: #111118;
    padding: 1rem; box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  }}
  table {{ border-collapse: collapse; }}
  th, td {{ padding: 0.5rem 0.7rem; text-align: center; }}
  .col-hdr {{ font-size: 0.8rem; color: #aaa; padding-bottom: 0.8rem; }}
  .row-hdr {{ font-size: 0.9rem; text-align: right; padding-right: 1rem; white-space: nowrap; }}
  .cell {{
    font-size: 1.8rem; cursor: pointer; transition: all 0.15s;
    border-radius: 6px; position: relative;
  }}
  .cell:hover {{
    background: #1a1a2e; transform: scale(1.3);
    box-shadow: 0 0 12px rgba(100,100,255,0.3);
  }}
  .cell.selected {{ background: #1e1e3a; box-shadow: 0 0 16px rgba(100,200,255,0.4); }}
  #detail {{
    margin-top: 1.5rem; padding: 1.5rem; background: #111118;
    border-radius: 12px; max-width: 600px; width: 100%;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    display: none; line-height: 1.6;
  }}
  #detail.visible {{ display: block; }}
  #detail h2 {{ font-size: 1.3rem; margin-bottom: 0.5rem; }}
  #detail .braille-big {{ font-size: 4rem; text-align: center; margin: 0.5rem 0; }}
  #detail .score {{ text-align: center; color: #aaa; margin-bottom: 1rem; }}
  .dot-row {{ display: flex; gap: 0.5rem; align-items: center; padding: 0.15rem 0; font-size: 0.9rem; }}
  .dot-on {{ color: #4caf50; }}
  .dot-off {{ color: #333; }}
  .eli5-box {{
    background: #1a1a2e; border-radius: 8px; padding: 1rem;
    margin-top: 1rem; font-size: 0.95rem; color: #ccc;
  }}
  .stats {{
    margin-top: 1rem; display: flex; gap: 1.5rem; justify-content: center;
    font-size: 0.85rem; color: #888;
  }}
  .stat-val {{ font-size: 1.1rem; color: #e0e0e0; font-weight: 600; }}
  .legend {{
    margin-top: 1.5rem; display: flex; gap: 1rem; flex-wrap: wrap;
    justify-content: center; font-size: 0.8rem; color: #888;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3rem; }}
  .legend-swatch {{
    width: 14px; height: 14px; border-radius: 3px; display: inline-block;
  }}
</style>
</head>
<body>
<h1>🧮 Braille Algebra</h1>
<p class="subtitle">{n}×{n} infinite operator grid — tap any cell to explore</p>

<div class="grid-container">
<table>
  <thead><tr><th></th>{header_cells}</tr></thead>
  <tbody>{"".join(grid_rows)}</tbody>
</table>
</div>

<div class="stats">
  <div><span class="stat-val">{n}</span> operators</div>
  <div><span class="stat-val">{n*n}</span> cells</div>
  <div><span class="stat-val">{sum(c.dot_count for c in algebra._cells.values())}</span> dots raised</div>
  <div><span class="stat-val">{sum(c.dot_count for c in algebra._cells.values()) / (n*n*8) * 100:.0f}%</span> density</div>
</div>

<div class="legend">
  <div class="legend-item"><span class="legend-swatch" style="background:hsl(120,70%,45%)"></span> 8/8 perfect</div>
  <div class="legend-item"><span class="legend-swatch" style="background:hsl(80,70%,45%)"></span> 6/8 good</div>
  <div class="legend-item"><span class="legend-swatch" style="background:hsl(40,70%,45%)"></span> 4/8 partial</div>
  <div class="legend-item"><span class="legend-swatch" style="background:hsl(0,70%,45%)"></span> 0/8 incompatible</div>
</div>

<div id="detail">
  <h2 id="detail-title"></h2>
  <div class="braille-big" id="detail-braille"></div>
  <div class="score" id="detail-score"></div>
  <div id="detail-dots"></div>
  <div class="eli5-box" id="detail-eli5"></div>
</div>

<script>
const cells = {cells_json};
const ops = {ops_json};
const dotNames = [
  "composable A→B", "types align", "order doesn't matter", "idempotent",
  "composable B→A", "same output type", "has inverse", "cost-efficient"
];

document.querySelectorAll('.cell').forEach(td => {{
  td.addEventListener('click', () => {{
    document.querySelectorAll('.cell.selected').forEach(c => c.classList.remove('selected'));
    td.classList.add('selected');
    const a = td.dataset.a, b = td.dataset.b;
    const key = a + ',' + b;
    const cell = cells[key];
    const opA = ops.find(o => o.name === a);
    const opB = ops.find(o => o.name === b);
    const detail = document.getElementById('detail');
    detail.classList.add('visible');
    document.getElementById('detail-title').textContent =
      opA.emoji + ' ' + opA.full_name + '  ×  ' + opB.emoji + ' ' + opB.full_name;
    document.getElementById('detail-braille').textContent = cell.braille;
    document.getElementById('detail-score').textContent =
      cell.score + '/8 dots raised — ' + Math.round(cell.score/8*100) + '% compatible';
    const dotsDiv = document.getElementById('detail-dots');
    dotsDiv.innerHTML = '';
    for (let i = 0; i < 8; i++) {{
      const on = cell.dots.includes(i);
      const row = document.createElement('div');
      row.className = 'dot-row';
      row.innerHTML = '<span class="' + (on ? 'dot-on' : 'dot-off') + '">' +
        (on ? '⬤' : '○') + '</span> Dot ' + (i+1) + ': ' + dotNames[i];
      dotsDiv.appendChild(row);
    }}
    document.getElementById('detail-eli5').innerHTML =
      '<strong>ELI5:</strong> ' + opA.eli5 + '<br><br>' +
      '<strong>Together:</strong> ' +
      (cell.properties.composable_ab && cell.properties.composable_ba
        ? 'They can go in either order!'
        : cell.properties.composable_ab
        ? opB.emoji + ' ' + opB.name + ' first, then ' + opA.emoji + ' ' + opA.name
        : cell.properties.composable_ba
        ? opA.emoji + ' ' + opA.name + ' first, then ' + opB.emoji + ' ' + opB.name
        : 'These two don\\'t connect directly.');
  }});
}});
</script>
</body>
</html>"""
