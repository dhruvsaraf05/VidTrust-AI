"""Reorder the \\bibitem entries so references are numbered by first citation.

IEEE style numbers references in the order they are first cited in the text.
The bibliography here is a hand-written thebibliography, whose numbering is
simply the order of its \\bibitem entries, so adding a reference at the end, or
citing an old one earlier, silently breaks the rule. This derives the order from
the text instead of trusting it:

    cd paper && python order_references.py          # rewrite in place
    cd paper && python order_references.py --check  # exit 1 if out of order

It refuses to write if any citation has no entry or any entry is never cited.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent / "vidtrust_ieee.tex"

source = io.open(TEX, encoding="utf-8").read()
start = source.index("\\begin{thebibliography}")
end = source.index("\\end{thebibliography}")
header_end = source.index("\n", start) + 1

# Citations in the body, in reading order, ignoring comment lines.
body = "\n".join(line for line in source[:start].splitlines()
                 if not line.lstrip().startswith("%"))
order: list[str] = []
for group in re.findall(r"\\cite\{([^}]*)\}", body):
    for key in (k.strip() for k in group.split(",")):
        if key and key not in order:
            order.append(key)

# Split the bibliography into one block per entry, keyed by its cite key.
bib = source[header_end:end]
blocks = {}
for match in re.finditer(r"\\bibitem\{([^}]+)\}.*?(?=\\bibitem\{|\Z)", bib, re.S):
    blocks[match.group(1)] = match.group(0).strip() + "\n"

missing = [k for k in order if k not in blocks]
uncited = [k for k in blocks if k not in order]
if missing or uncited:
    print("missing entries :", missing or "none")
    print("never cited     :", uncited or "none")
    sys.exit(1)

current = list(blocks)
if "--check" in sys.argv:
    if current != order:
        print("references are NOT in order of first citation")
        sys.exit(1)
    print(f"{len(order)} references, numbered in order of first citation")
    sys.exit(0)

rebuilt = "\n" + "\n".join(blocks[k] for k in order) + "\n"
io.open(TEX, "w", encoding="utf-8").write(source[:header_end] + rebuilt + source[end:])
for number, key in enumerate(order, 1):
    moved = "" if current.index(key) + 1 == number else f"   (was [{current.index(key) + 1}])"
    print(f"[{number:>2}] {key}{moved}")
