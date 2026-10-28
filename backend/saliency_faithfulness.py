"""Does the Grad-CAM heatmap agree with where the model is actually sensitive?

Offline only, never on the request path. For each image this compares two
attributions over the same 8x8 grid of 32x32 cells on the processor's 256x256
input:

  * Grad-CAM, exactly as detectors/saliency.py computes it (stage 2, 16x16,
    pooled 2x2 to 8x8 so the two maps share a grid), and
  * occlusion: zero one cell of the normalised input -- the processor's mean
    colour -- and record how far the machine-generated logit drops.

Occlusion is the ground truth in the narrow sense that it measures what the
model's output actually depends on. Agreement is Spearman rank correlation
over the 64 cells. A heatmap is only worth drawing if this is clearly
positive, so QUOTE THE MEDIAN WHEREVER THE HEATMAP IS SHOWN.

It also records the statistic the gate in saliency.py uses (p90-p10 span) and
the share of the logit the single most damaging cell removes, which is the
direct measure of whether the evidence is localised at all.

This reports. It does not set saliency.MIN_SPAN: these images are the
evaluation set, and fitting a constant to them would break invariant 10.

    cd backend
    .venv/Scripts/python saliency_faithfulness.py              # 10 AI + 10 REAL
    .venv/Scripts/python saliency_faithfulness.py --per-class 3   # quick check

Cost: 1 gradient pass plus 64 occluded forward passes per image, about a
minute per image on the 4-core CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from detectors import model_detector, saliency
from evaluate import CROP_SIZE, _crop_native

HERE = Path(__file__).resolve().parent
EVAL = HERE / "evaluation_set"
OUTPUT = HERE / "saliency_faithfulness.json"
SEED = 0
OCCLUSION_GRID = 8
BATCH = 16


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman's rho with average ranks for ties. None if either is constant."""
    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x), dtype=float)
        for value in np.unique(x):
            tied = x == value
            if tied.sum() > 1:
                r[tied] = r[tied].mean()
        return r
    ra, rb = ranks(a.ravel()), ranks(b.ravel())
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def pool(grid: np.ndarray, side: int) -> np.ndarray:
    factor = grid.shape[0] // side
    return grid.reshape(side, factor, side, factor).mean(axis=(1, 3))


def occlusion(model, pixel_values, ai_index: int) -> tuple[np.ndarray, float]:
    import torch

    _, _, height, width = pixel_values.shape
    cell_h, cell_w = height // OCCLUSION_GRID, width // OCCLUSION_GRID
    variants = []
    for row in range(OCCLUSION_GRID):
        for col in range(OCCLUSION_GRID):
            masked = pixel_values.clone()
            masked[..., row * cell_h:(row + 1) * cell_h,
                   col * cell_w:(col + 1) * cell_w] = 0.0
            variants.append(masked)

    with torch.no_grad():
        baseline = float(model(pixel_values=pixel_values).logits[0, ai_index])
        logits = []
        for start in range(0, len(variants), BATCH):
            batch = torch.cat(variants[start:start + BATCH])
            logits.extend(model(pixel_values=batch).logits[:, ai_index].tolist())

    drops = baseline - np.asarray(logits)
    return drops.reshape(OCCLUSION_GRID, OCCLUSION_GRID), baseline


def choose(per_class: int) -> list[dict]:
    with (EVAL / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(SEED)
    chosen = []
    for label in ("AI", "REAL"):
        pool_rows = [r for r in rows if r["expected"] == label]
        chosen += rng.sample(pool_rows, per_class)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--resummarise", action="store_true",
                        help="recompute the summary from the saved rows; no inference")
    args = parser.parse_args()

    if args.resummarise:
        saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
        rows = [{k: v for k, v in r.items() if k != "max_cell_share"} for r in saved["rows"]]
        header = {k: v for k, v in saved.items() if k not in ("overall", "by_class", "rows")}
        write_report(rows, header)
        return

    if not model_detector.load_model():
        raise SystemExit(f"classifier did not load: {model_detector.load_error()}")
    pipe = model_detector.pipe()
    model, processor = pipe.model, pipe.image_processor
    model.eval()
    ai_index = model_detector.ai_label_index()
    if ai_index is None:
        raise SystemExit("unrecognised label vocabulary; refusing to guess (invariant 2)")

    # Centre-cropped with the evaluation harness's own crop, so both classes
    # enter at the same size and the resolution confound (REPORT.md section 6)
    # is not part of what is measured.
    folder = EVAL / "images"
    rows = []
    for number, entry in enumerate(choose(args.per_class), 1):
        started = time.perf_counter()
        with Image.open(folder / entry["filename"]) as image:
            rgb, _ = _crop_native(image.convert("RGB"), CROP_SIZE)
        pixel_values = processor(images=rgb, return_tensors="pt")["pixel_values"]

        cam, logit = saliency.cam_grid(model, pixel_values, ai_index)
        drops, baseline = occlusion(model, pixel_values, ai_index)
        cam8 = pool(cam, OCCLUSION_GRID)

        ceiling = float(cam.max())
        span = saliency.gate_span(cam / ceiling) if ceiling > 0 else 0.0
        rho = spearman(cam8, drops)
        rows.append({
            "filename": entry["filename"],
            "expected": entry["expected"],
            "architecture": entry["architecture"],
            "ai_logit": round(baseline, 4),
            "spearman_rho": None if rho is None else round(rho, 4),
            "gate_span": round(span, 4),
            "gate_draws": bool(ceiling > 0 and span >= saliency.MIN_SPAN),
            "max_cell_drop": round(float(drops.max()), 4),
            "cam_nonzero_cells": int((cam > 0).sum()),
        })
        print(f"[{number:>2}] {entry['expected']:<4} rho={rows[-1]['spearman_rho']!s:>7} "
              f"span={span:.3f} max_drop={rows[-1]['max_cell_drop']} "
              f"({time.perf_counter() - started:.0f}s) {entry['filename'][:40]}",
              flush=True)

    write_report(rows)


def summary(subset: list[dict]) -> dict:
    """Aggregate per-image rows. Pure: rerunnable from saved rows (--resummarise).

    The single-cell drop is reported in raw logit units and deliberately NOT as
    a share of the logit: the 'artificial' logit alone is often near zero or
    negative (the softmax depends on the DIFFERENCE between the two logits), so
    dividing by it gave meaningless shares above 5 and below 0 in the first
    run of this study.
    """
    rhos = [r["spearman_rho"] for r in subset if r["spearman_rho"] is not None]
    drops = [r["max_cell_drop"] for r in subset]
    drawn = [r for r in subset if r["gate_draws"]]
    return {
        "n": len(subset),
        "rho_median": round(float(np.median(rhos)), 4) if rhos else None,
        "rho_min": round(min(rhos), 4) if rhos else None,
        "rho_max": round(max(rhos), 4) if rhos else None,
        "rho_positive": sum(1 for r in rhos if r > 0),
        "gate_draws": len(drawn),
        # Of the maps the flatness gate WOULD draw, how many agree with
        # occlusion at all. This is the number that says whether the gate
        # protects anyone.
        "gate_draws_rho_positive": sum(1 for r in drawn
                                       if r["spearman_rho"] is not None
                                       and r["spearman_rho"] > 0),
        "max_cell_drop_median": round(float(np.median(drops)), 4) if drops else None,
        "max_cell_drop_max": round(float(max(drops)), 4) if drops else None,
    }


def write_report(rows: list[dict], header: dict | None = None) -> None:
    report = {**(header or {
        "model": model_detector.model_name(),
        "cam_stage": saliency.CAM_STAGE,
        "min_span": saliency.MIN_SPAN,
        "occlusion_grid": OCCLUSION_GRID,
        "seed": SEED,
        "source": f"evaluation_set/images, centre-cropped to {CROP_SIZE}",
    }),
        "overall": summary(rows),
        "by_class": {label: summary([r for r in rows if r["expected"] == label])
                     for label in ("AI", "REAL")},
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("overall", "by_class")}, indent=2))
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
