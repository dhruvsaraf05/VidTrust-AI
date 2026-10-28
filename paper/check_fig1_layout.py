"""Fail if Fig. 1's labels overlap, spill out of their boxes, or sit off-centre.

Fig. 1 is drawn in data coordinates on an axes whose aspect is auto, so its
boxes stretch with the figure height while the text stays at a fixed point
size. Shrinking the figure to fit a page limit therefore crowds the text
without any error, which is how the fusion box's equation ended up colliding
with both its own title and the subtext beneath it. This measures the rendered
bounding box of every label instead of trusting the eye at preview zoom.

It also checks that each box's contents are centred: the margin above the
contents must equal the margin below, and left must equal right. The same drift
once left the signal boxes with 11pt of space above their text and 0.05pt
below it.

    cd paper && python check_fig1_layout.py
"""
from __future__ import annotations

import sys
import tempfile
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

import make_figures

# Labels closer than this many pixels (at the 300 dpi the PNG is written at)
# count as touching. Zero would pass glyphs that sit flush against each other.
CLEARANCE_PX = 2.0

# Largest allowed difference between opposite margins of a box, in points.
# Half a point is below what the eye can see at print size.
BALANCE_PT = 0.5

captured = []
real_close = plt.close
plt.close = lambda fig=None: captured.append(fig)
try:
    with tempfile.TemporaryDirectory() as scratch:
        make_figures.architecture(Path(scratch) / "fig1.pdf")
finally:
    plt.close = real_close

fig = captured[0]
fig.set_dpi(300)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
ax = fig.axes[0]

labels = [(t.get_text().replace("\n", " / "), t.get_window_extent(renderer))
          for t in ax.texts if t.get_text().strip()]
boxes = [p.get_window_extent(renderer) for p in ax.patches
         if isinstance(p, FancyBboxPatch) and p.get_linewidth() > 0]
# The colour swatches in the Verdict box are content, not frames.
swatches = [p.get_window_extent(renderer) for p in ax.patches
            if isinstance(p, FancyBboxPatch) and p.get_linewidth() == 0]

problems = []
for (name_a, a), (name_b, b) in combinations(labels, 2):
    grown = a.expanded(1.0, 1.0)
    grown.x0 -= CLEARANCE_PX
    grown.x1 += CLEARANCE_PX
    grown.y0 -= CLEARANCE_PX
    grown.y1 += CLEARANCE_PX
    if grown.overlaps(b):
        problems.append(f"overlap: {name_a!r} and {name_b!r}")

for name, extent in labels:
    centre = ((extent.x0 + extent.x1) / 2, (extent.y0 + extent.y1) / 2)
    for frame in boxes:
        if frame.contains(*centre):
            if (extent.x0 < frame.x0 or extent.x1 > frame.x1
                    or extent.y0 < frame.y0 or extent.y1 > frame.y1):
                problems.append(f"spills out of its box: {name!r}")

px_per_pt = fig.dpi / 72.0
for frame in boxes:
    contents = [extent for _, extent in labels] + swatches
    inside = [e for e in contents
              if frame.contains((e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2)]
    if not inside:
        continue
    block = Bbox.union(inside)
    top, bottom = frame.y1 - block.y1, block.y0 - frame.y0
    left, right = block.x0 - frame.x0, frame.x1 - block.x1
    name = next(n for n, e in labels
                if frame.contains((e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2))
    for side, a, b in (("vertically", top, bottom), ("horizontally", left, right)):
        if abs(a - b) / px_per_pt > BALANCE_PT:
            problems.append(f"not centred {side} in its box: {name!r} "
                            f"({a / px_per_pt:.2f}pt against {b / px_per_pt:.2f}pt)")

real_close(fig)
print(f"{len(labels)} labels, {len(boxes)} boxes checked")
if problems:
    print("FAILED:")
    for line in problems:
        print("  -", line)
    sys.exit(1)
print("no overlaps, nothing outside its box")
