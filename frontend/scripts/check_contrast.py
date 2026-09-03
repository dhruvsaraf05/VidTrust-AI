"""WCAG AA contrast check for both palettes (light and dark).

Parses the `@theme` block (light, the default) and the
`:root[data-theme='dark']` block out of src/index.css -- both palettes live in
that one file -- then checks each text colour against the background it is
actually rendered on.

Checking by eye does not work: a colour that looks fine on a laptop can fall
under 4.5:1 and disappear on a projector at the back of a room.

    python scripts/check_contrast.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

AA_NORMAL = 4.5
AA_LARGE = 3.0

# (foreground token, background token, label, is_large_text)
PAIRS = [
    ("ink", "paper", "body text on page ground", False),
    ("ink", "panel", "body text on panel", False),
    ("ink-2", "paper", "secondary text on ground", False),
    ("ink-2", "panel", "secondary text on panel", False),
    ("ink-3", "paper", "faint label on ground", False),
    ("ink-3", "panel", "faint label on panel", False),
    ("accent", "paper", "link/accent on ground", False),
    ("accent", "panel", "accent on panel", False),
    ("panel", "accent", "button text on accent fill", False),
    ("panel", "ink", "active toggle text on ink fill", False),
    ("crimson", "panel", "AI verdict on panel", False),
    ("crimson", "crimson-tint", "AI verdict on its tint", False),
    ("teal", "panel", "REAL verdict on panel", False),
    ("teal", "teal-tint", "REAL verdict on its tint", False),
    ("ochre", "panel", "UNCERTAIN verdict on panel", False),
    ("ochre", "ochre-tint", "UNCERTAIN verdict on its tint", False),
    ("crimson", "paper", "verdict number (large) on ground", True),
    ("teal", "paper", "verdict number (large) on ground", True),
    ("ochre", "paper", "verdict number (large) on ground", True),
]


def channel(value: float) -> float:
    value /= 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def ratio(fg: str, bg: str) -> float:
    a, b = luminance(fg), luminance(bg)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def tokens_from(text: str) -> dict[str, str]:
    return {
        name: value
        for name, value in re.findall(r"--color-([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", text)
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    index_css = (root / "src" / "index.css").read_text(encoding="utf-8")

    light_block = re.search(r"@theme\s*\{(.*?)\n\}", index_css, re.S)
    dark_block = re.search(
        r"data-theme='dark'\]\s*\{(.*?)\n\}", index_css, re.S
    )
    if not light_block or not dark_block:
        print("Could not find both palette blocks in index.css")
        return 1

    palettes = {
        "Light (default)": tokens_from(light_block.group(1)),
        "Dark": tokens_from(dark_block.group(1)),
    }

    failures = 0
    for palette, tokens in palettes.items():
        print(f"\n{palette}")
        print("-" * 74)
        for fg, bg, label, large in PAIRS:
            if fg not in tokens or bg not in tokens:
                print(f"  ?  {label:<40} missing token {fg} or {bg}")
                continue
            value = ratio(tokens[fg], tokens[bg])
            need = AA_LARGE if large else AA_NORMAL
            ok = value >= need
            if not ok:
                failures += 1
            print(f"  {'PASS' if ok else 'FAIL'} {label:<40} "
                  f"{value:5.2f}:1  (needs {need})")

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
