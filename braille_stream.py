"""
Braille Stream Encoding — Python port of Aria's braille encoding layer.

Ported from: github.com/elevate-foundry/ai-native-ide/src/braille.js
             github.com/elevate-foundry/ai-native-ide/src/braille-harness.js

Two encoding modes:
1. Byte-level: maps UTF-8 bytes 1:1 to braille codepoints U+2800-U+28FF (lossless)
2. UEB Grade-2: compresses common English words to 1-2 braille cells before byte encoding

The byte-level encoding is strictly denser on the wire because each braille codepoint
(3 bytes in UTF-8) carries 8 bits of payload, vs plain ASCII (1 byte = 7 useful bits).
The real win is that SSE frames contain fewer *characters*, so JSON-escaping overhead
and browser TextDecoder passes are reduced for the same semantic content.
"""

from __future__ import annotations

BRAILLE_BASE = 0x2800

# ============================================================================
# Byte-level encoding (lossless UTF-8 ↔ Braille)
# ============================================================================

def to_braille(text: str) -> str:
    """Encode UTF-8 text to braille codepoints (one per byte)."""
    return "".join(chr(BRAILLE_BASE + b) for b in text.encode("utf-8"))


def from_braille(braille: str) -> str:
    """Decode braille codepoints back to UTF-8 text."""
    raw = bytes(
        ord(ch) - BRAILLE_BASE
        for ch in braille
        if BRAILLE_BASE <= ord(ch) <= BRAILLE_BASE + 255
    )
    return raw.decode("utf-8", errors="replace")


# ============================================================================
# UEB Grade-2 contractions (semantic compression)
# ============================================================================

UEB_CONTRACTIONS: dict[str, str] = {
    # Single-cell whole word contractions
    "but": "⠃", "can": "⠉", "do": "⠙", "every": "⠑",
    "from": "⠋", "go": "⠛", "have": "⠓", "just": "⠚",
    "knowledge": "⠅", "like": "⠇", "more": "⠍", "not": "⠝",
    "people": "⠏", "quite": "⠟", "rather": "⠗", "so": "⠎",
    "that": "⠞", "us": "⠥", "very": "⠧", "will": "⠺",
    "it": "⠭", "you": "⠽", "as": "⠵", "and": "⠯",
    "for": "⠿", "of": "⠷", "the": "⠮", "with": "⠾",
    "child": "⠡", "shall": "⠩", "this": "⠹", "which": "⠱",
    "out": "⠳", "still": "⠌",
    # Programming-specific contractions (custom extension from Aria)
    "function": "⣋⣥", "return": "⣗⣞", "const": "⣉⣎",
    "let": "⣇⣞", "var": "⣧⣗", "if": "⣊⣋", "else": "⣑⣇",
    "while": "⣺⣓", "class": "⣉⣇", "import": "⣊⣍",
    "export": "⣑⣭", "async": "⣁⣎", "await": "⣁⣺",
    "true": "⣞⣗", "false": "⣋⣇", "null": "⣝⣥",
    "undefined": "⣥⣙", "console": "⣉⣎⣇", "error": "⣑⣗⣗",
    "string": "⣎⣞⣗", "number": "⣝⣥⣍", "boolean": "⣃⣕⣇",
    "array": "⣁⣗⣗", "object": "⣕⣃⣚", "promise": "⣏⣗⣍",
}

# Reverse map for decoding contractions
UEB_REVERSE: dict[str, str] = {v: k for k, v in UEB_CONTRACTIONS.items()}

import re

# Pre-compile regex for contraction matching (longest first)
_CONTRACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), braille)
    for word, braille in sorted(UEB_CONTRACTIONS.items(), key=lambda x: -len(x[0]))
]


def braid(text: str, use_contractions: bool = True) -> str:
    """Encode text with optional UEB contractions, then byte-level braille.

    With contractions enabled, common words become 1-2 braille cells instead
    of N byte-level cells, giving ~15-30% compression on English prose.
    """
    if not text:
        return ""

    if not use_contractions:
        return to_braille(text)

    # Mark contraction boundaries with a sentinel
    SENTINEL = "\x00"
    result = text
    for pattern, braille_cell in _CONTRACTION_PATTERNS:
        result = pattern.sub(f"{SENTINEL}{braille_cell}{SENTINEL}", result)

    # Encode non-contracted segments as raw byte-level braille
    parts = result.split(SENTINEL)
    output: list[str] = []
    for i, part in enumerate(parts):
        if not part:
            continue
        # Odd-index parts are already braille contraction cells
        if i % 2 == 1:
            output.append(part)
        else:
            output.append(to_braille(part))
    return "".join(output)


def unbraid(braille: str) -> str:
    """Decode braided braille back to text.

    Handles both UEB contractions and byte-level encoded segments.
    """
    if not braille:
        return ""

    chars = list(braille)
    result: list[str] = []
    i = 0

    while i < len(chars):
        # Try 3-cell contraction
        three = "".join(chars[i:i+3])
        two = "".join(chars[i:i+2])
        one = chars[i]

        if three in UEB_REVERSE:
            result.append(UEB_REVERSE[three])
            i += 3
        elif two in UEB_REVERSE:
            result.append(UEB_REVERSE[two])
            i += 2
        elif one in UEB_REVERSE:
            result.append(UEB_REVERSE[one])
            i += 1
        else:
            # Byte-level braille cell — accumulate for UTF-8 decoding
            cp = ord(one)
            if BRAILLE_BASE <= cp <= BRAILLE_BASE + 255:
                byte_buf = [cp - BRAILLE_BASE]
                j = i + 1
                while j < len(chars):
                    ncp = ord(chars[j])
                    if BRAILLE_BASE <= ncp <= BRAILLE_BASE + 255:
                        nb = ncp - BRAILLE_BASE
                        # UTF-8 continuation byte
                        if 0x80 <= nb < 0xC0:
                            byte_buf.append(nb)
                            j += 1
                        else:
                            break
                    else:
                        break
                result.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                i = j
            else:
                result.append(one)
                i += 1

    return "".join(result)


# ============================================================================
# Streaming helpers
# ============================================================================

class BrailleStreamProcessor:
    """Stateful processor that converts text chunks to braille on the fly."""

    def __init__(self, use_contractions: bool = True):
        self.use_contractions = use_contractions
        self.text_buffer = ""
        self.braille_buffer = ""
        self.stats = {"bytes_in": 0, "braille_out": 0, "chunks": 0}

    def process_chunk(self, text_chunk: str) -> str:
        """Convert a text chunk to braille and return the braille."""
        self.text_buffer += text_chunk
        self.stats["bytes_in"] += len(text_chunk.encode("utf-8"))

        braille_chunk = braid(text_chunk, self.use_contractions)
        self.braille_buffer += braille_chunk
        self.stats["braille_out"] += len(braille_chunk)
        self.stats["chunks"] += 1

        return braille_chunk

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "compression_ratio": (
                self.stats["braille_out"] / self.stats["bytes_in"]
                if self.stats["bytes_in"] > 0 else 1.0
            ),
        }

    def reset(self):
        self.text_buffer = ""
        self.braille_buffer = ""
        self.stats = {"bytes_in": 0, "braille_out": 0, "chunks": 0}
