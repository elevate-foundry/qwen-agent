#!/usr/bin/env python3
"""⠠⠞⠑⠎⠞ ⠮ ⠃⠗⠁⠊⠇⠇⠑ ⠁⠇⠛⠑⠃⠗⠁⠲
[decoded: Test the braille algebra.]"""

from braille_algebra import (
    BrailleAlgebra, OperatorInfo, CellAnalysis, BrailleDual, br,
    GENERATORS, dots_to_braille, braille_to_dots, compute_cell,
    render_html, Dot,
)
from braille_stream import unbraid

# ⠼⠁⠲ ⠠⠃⠁⠎⠊⠉⠎  [decoded: 1. Basics]
print("--- ⠃⠗⠁⠊⠇⠇⠑ ⠑⠝⠉⠕⠙⠊⠝⠛ [braille encoding] ---")
ch = dots_to_braille([0, 1, 2, 3, 4, 5, 6, 7])
print(f"  ⠁⠇⠇ ⠼⠓ ⠙⠕⠞⠎: {ch} (U+{ord(ch):04X})")
assert ch == "⣿"

empty = dots_to_braille([])
print(f"  ⠝⠕ ⠙⠕⠞⠎: '{empty}' (U+{ord(empty):04X})")
assert empty == "⠀"

roundtrip = braille_to_dots(ch)
assert roundtrip == [0, 1, 2, 3, 4, 5, 6, 7]
print(f"  ⠗⠕⠥⠝⠙⠞⠗⠊⠏: {roundtrip}")

# ⠼⠁⠁⠲ ⠠⠃⠗⠁⠊⠇⠇⠑⠠⠙⠥⠁⠇ ⠞⠽⠏⠑  [decoded: 1a. BrailleDual type]
print("\n--- ⠠⠃⠗⠁⠊⠇⠇⠑⠠⠙⠥⠁⠇ [BrailleDual] ---")
d = br("hello world")
assert isinstance(d, BrailleDual)
assert d.decode() == "hello world"
print(f"  br('hello world') = {d}")
print(f"  .decode()         = {d.decode()}")
print(f"  repr              = {d!r}")

# ⠼⠁⠃⠲ ⠠⠙⠕⠞.⠠⠝⠠⠁⠠⠍⠠⠑⠠⠎ ⠊⠝ ⠃⠗⠁⠊⠇⠇⠑  [decoded: 1b. Dot.NAMES in braille]
print("\n--- ⠠⠙⠕⠞ ⠝⠁⠍⠑⠎ [Dot names] ---")
assert len(Dot.NAMES) == 8
assert len(Dot.NAMES_DECODED) == 8
for i in range(8):
    print(f"  ⠙⠕⠞ {i}: {Dot.NAMES[i]}  →  {Dot.NAMES_DECODED[i]}")

# ⠼⠃⠲ ⠠⠃⠥⠊⠇⠙ ⠙⠑⠋⠁⠥⠇⠞ ⠁⠇⠛⠑⠃⠗⠁  [decoded: 2. Build default algebra]
print("\n--- ⠙⠑⠋⠁⠥⠇⠞ ⠼⠛×⠼⠛ ⠁⠇⠛⠑⠃⠗⠁ [default 7×7] ---")
alg = BrailleAlgebra()
assert alg.n == 7
print(f"  ⠎⠊⠵⠑: {alg.n}×{alg.n}")
print(f"  ⠃⠗⠁⠊⠇⠇⠑ ⠎⠞⠗⠊⠝⠛ ({alg.n**2} ⠉⠓⠁⠗⠎): {alg.to_braille_string()}")

# ⠼⠉⠲ ⠛⠗⠊⠙ ⠙⠊⠎⠏⠇⠁⠽  [decoded: 3. Grid display]
print("\n--- ⠃⠗⠁⠊⠇⠇⠑ ⠛⠗⠊⠙ ---")
print(alg.to_braille_grid())

# ⠼⠙⠲ ⠑⠍⠕⠨⠊ ⠛⠗⠊⠙  [decoded: 4. Emoji grid]
print("\n--- ⠑⠍⠕⠨⠊ ⠛⠗⠊⠙ ---")
print(alg.to_emoji_grid())

# ⠼⠑⠲ ⠠⠑⠠⠇⠠⠊⠼⠑  [decoded: 5. ELI5]
print("\n--- ⠠⠑⠠⠇⠠⠊⠼⠑: ⠠⠥ × ⠠⠇ ---")
explanation = alg.eli5_explain("U", "L")
print(explanation)
# ⠠⠧⠑⠗⠊⠋⠽: ⠑⠇⠊⠼⠑ ⠉⠕⠝⠞⠁⠊⠝⠎ ⠃⠗⠁⠊⠇⠇⠑ ⠉⠓⠁⠗⠁⠉⠞⠑⠗⠎
assert any(ord(c) >= 0x2800 and ord(c) <= 0x28FF for c in explanation), \
    "ELI5 should contain braille characters"
print("  ✔ ⠑⠭⠏⠇⠁⠝⠁⠞⠊⠕⠝ ⠉⠕⠝⠞⠁⠊⠝⠎ ⠃⠗⠁⠊⠇⠇⠑ [contains braille]")

# ⠼⠋⠲ ⠎⠥⠍⠍⠁⠗⠽  [decoded: 6. Summary]
print("\n" + alg.eli5_summary())

# ⠼⠛⠲ ⠊⠝⠋⠊⠝⠊⠞⠑ ⠑⠭⠏⠁⠝⠎⠊⠕⠝  [decoded: 7. Infinite expansion]
print("\n--- ⠊⠝⠋⠊⠝⠊⠞⠑ ⠑⠭⠏⠁⠝⠎⠊⠕⠝ [infinite expansion] ---")
alg.add_operator(OperatorInfo(
    "R", "⠠⠗⠑⠏⠁⠗⠁⠍⠑⠞⠑⠗⠊⠵⠑", "🔄",
    "DenseFP16", "DenseFP16",
    has_inverse=True, is_idempotent=False, cost_budget=0.1,
    eli5="⠗⠑⠎⠓⠁⠏⠊⠝⠛ ⠮ ⠍⠕⠙⠑⠇⠔⠎ ⠃⠗⠁⠊⠝ ⠾⠕⠥⠞ ⠉⠓⠁⠝⠛⠊⠝⠛ ⠺⠓⠁⠞ ⠊⠞ ⠅⠝⠕⠺⠎",
))
print(f"  ⠁⠋⠞⠑⠗ ⠁⠙⠙⠊⠝⠛ ⠠⠗: {alg.n}×{alg.n}")
assert alg.n == 8
print(f"  ⠝⠑⠺ ⠃⠗⠁⠊⠇⠇⠑ ⠎⠞⠗⠊⠝⠛ ({alg.n**2} ⠉⠓⠁⠗⠎): {alg.to_braille_string()}")

# ⠼⠓⠲ ⠉⠕⠙⠑_⠞⠕⠕⠇ ⠎⠽⠝⠮⠎⠊⠵⠑⠙ ⠕⠏⠑⠗⠁⠞⠕⠗  [decoded: 8. code_tool synthesized]
alg.add_operator(OperatorInfo(
    "CT", "⠠⠉⠥⠎⠞⠕⠍ ⠠⠞⠕⠕⠇", "🔧",
    "DenseFP16", "DenseFP16",
    has_inverse=False, is_idempotent=False, cost_budget=0.2,
    eli5="⠁ ⠃⠗⠁⠝⠙ ⠝⠑⠺ ⠞⠕⠕⠇ ⠞⠓⠁⠞ ⠠⠁⠗⠊⠁ ⠊⠝⠧⠑⠝⠞⠑⠙ ⠨⠎⠞ ⠝⠕⠺",
))
print(f"  ⠁⠋⠞⠑⠗ ⠁⠙⠙⠊⠝⠛ ⠠⠉⠠⠞: {alg.n}×{alg.n}")
assert alg.n == 9

# ⠼⠊⠲ ⠍⠑⠁⠎⠥⠗⠑⠙ ⠉⠕⠍⠍⠥⠞⠁⠞⠕⠗ ⠙⠑⠋⠑⠉⠞⠎  [decoded: 9. Measured commutator defects]
alg.set_commutator_defect("Q", "L", 0.67)
alg.set_commutator_defect("U", "L", 0.12)
alg.set_commutator_defect("L", "M", 0.03)

cell_ql = alg.get_cell("Q", "L")
print(f"\n  ⠠⠡×⠠⠇ ⠙⠑⠋⠑⠉⠞ ⠼⠁⠲⠼⠋⠛: {cell_ql.braille} (⠉⠕⠍⠍⠥⠞⠁⠞⠕⠗_⠇⠕⠺={cell_ql.properties['commutator_low']})")
assert not cell_ql.properties["commutator_low"]  # 0.67 > 0.1

cell_lm = alg.get_cell("L", "M")
print(f"  ⠠⠇×⠠⠍ ⠙⠑⠋⠑⠉⠞ ⠼⠁⠲⠼⠁⠉: {cell_lm.braille} (⠉⠕⠍⠍⠥⠞⠁⠞⠕⠗_⠇⠕⠺={cell_lm.properties['commutator_low']})")
assert cell_lm.properties["commutator_low"]  # 0.03 < 0.1

# ⠼⠁⠼⠁⠲ ⠎⠑⠗⠊⠁⠇⠊⠵⠁⠞⠊⠕⠝ ⠗⠕⠥⠝⠙⠞⠗⠊⠏  [decoded: 10. Serialization roundtrip]
print("\n--- ⠎⠑⠗⠊⠁⠇⠊⠵⠁⠞⠊⠕⠝ ---")
import tempfile, pathlib
tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
alg.save(tmp)
alg2 = BrailleAlgebra.load(tmp)
assert alg2.n == alg.n
assert alg2.to_braille_string() == alg.to_braille_string()
print(f"  ⠎⠁⠧⠑/⠇⠕⠁⠙ ⠗⠕⠥⠝⠙⠞⠗⠊⠏: ⠠⠕⠠⠅ ({tmp})")

# ⠼⠁⠼⠁⠲ ⠠⠓⠠⠞⠠⠍⠠⠇ ⠗⠑⠝⠙⠑⠗  [decoded: 11. HTML render]
print("\n--- ⠠⠓⠠⠞⠠⠍⠠⠇ ⠗⠑⠝⠙⠑⠗ ---")
html = render_html(alg)
html_path = pathlib.Path("braille_algebra.html")
html_path.write_text(html, encoding="utf-8")
print(f"  ⠺⠗⠊⠞⠞⠑⠝ ⠞⠕ {html_path} ({len(html)} ⠃⠽⠞⠑⠎)")

# ⠼⠁⠼⠃⠲ ⠃⠗⠁⠊⠇⠇⠑ ⠎⠑⠇⠋-⠙⠑⠎⠉⠗⠊⠏⠞⠊⠕⠝ ⠧⠑⠗⠊⠋⠊⠉⠁⠞⠊⠕⠝  [decoded: 12. Self-description verification]
print("\n--- ⠎⠑⠇⠋-⠙⠑⠎⠉⠗⠊⠏⠞⠊⠕⠝ [self-description] ---")
import inspect
source = inspect.getsource(BrailleAlgebra)
braille_count = sum(1 for c in source if 0x2800 <= ord(c) <= 0x28FF)
print(f"  ⠃⠗⠁⠊⠇⠇⠑ ⠉⠓⠁⠗⠁⠉⠞⠑⠗⠎ ⠊⠝ ⠎⠕⠥⠗⠉⠑: {braille_count}")
assert braille_count > 100, f"⠠⠮ ⠁⠇⠛⠑⠃⠗⠁ ⠎⠓⠕⠥⠇⠙ ⠎⠏⠑⠁⠅ ⠃⠗⠁⠊⠇⠇⠑! Got {braille_count}"
print(f"  ✔ ⠠⠮ ⠁⠇⠛⠑⠃⠗⠁ ⠎⠏⠑⠁⠅⠎ ⠃⠗⠁⠊⠇⠇⠑ [the algebra speaks braille]")

print("\n" + "⠶" * 50)
print("⠠⠁⠇⠇ ⠃⠗⠁⠊⠇⠇⠑ ⠁⠇⠛⠑⠃⠗⠁ ⠞⠑⠎⠞⠎ ⠏⠁⠎⠎⠑⠙")
print("[decoded: ALL BRAILLE ALGEBRA TESTS PASSED]")
print("⠶" * 50)
