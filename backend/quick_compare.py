"""Fast head-to-head of cached classifiers on the labelled samples.

Time-boxed replacement for the full benchmark: 13 images x N models, using
weights already in the HuggingFace cache. Inference only -- nothing trains.
"""
from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import config

CANDIDATES = [
    "Organika/sdxl-detector",
    "Ateeqq/ai-vs-human-image-detector",
    "dima806/ai_vs_real_image_detection",
    "dima806/ai_vs_human_generated_image_detection",
    "haywoodsloan/ai-image-detector-deploy",
]

AI_WORDS = {"artificial", "ai", "ai-generated", "ai_generated", "fake",
            "generated", "machine", "sdxl", "synthetic", "deepfake"}
REAL_WORDS = {"human", "hum", "real", "natural", "photo", "photograph",
              "authentic", "nature"}


def ai_probability(preds):
    for p in preds:
        if str(p["label"]).strip().lower() in AI_WORDS:
            return float(p["score"])
    for p in preds:
        if str(p["label"]).strip().lower() in REAL_WORDS:
            return 1.0 - float(p["score"])
    return None


def main() -> int:
    from PIL import Image
    from transformers import pipeline

    labels = {}
    with open("samples/MANIFEST.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n = (row.get("filename") or "").strip()
            if n and not n.startswith("#"):
                labels[n] = (row.get("expected") or "UNKNOWN").strip().upper()

    items = []
    for p in sorted(Path("samples").iterdir()):
        if p.suffix.lower() in config.IMAGE_EXTENSIONS and labels.get(p.name) in ("AI", "REAL"):
            items.append((p, labels[p.name]))
    print(f"{len(items)} labelled images "
          f"({sum(1 for _, t in items if t == 'AI')} AI / "
          f"{sum(1 for _, t in items if t == 'REAL')} REAL)\n")

    images = []
    for p, _ in items:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))

    results = {}
    for name in CANDIDATES:
        t0 = time.perf_counter()
        try:
            clf = pipeline("image-classification", model=name, device="cpu")
        except Exception as exc:
            print(f"SKIP {name}: {type(exc).__name__}")
            continue
        raw = clf(images, top_k=None)
        if raw and isinstance(raw[0], dict):
            raw = [raw]
        scores = [ai_probability(r) for r in raw]
        if any(s is None for s in scores):
            print(f"SKIP {name}: labels did not resolve")
            del clf
            continue
        results[name] = scores
        print(f"scored {name}  ({time.perf_counter() - t0:.0f}s)")
        del clf

    truth = [1 if t == "AI" else 0 for _, t in items]
    print("\n" + "=" * 118)
    print(f"{'file':<42}{'truth':>6}" + "".join(f"{n.split('/')[-1][:16]:>18}"
                                               for n in results))
    print("-" * 118)
    for i, (p, t) in enumerate(items):
        row = "".join(f"{results[n][i]:>18.3f}" for n in results)
        print(f"{p.name[:41]:<42}{t:>6}{row}")

    print("-" * 118)
    # A model is only useful here if AI images score high AND real ones low.
    for n, sc in results.items():
        ai = [s for s, y in zip(sc, truth) if y == 1]
        re_ = [s for s, y in zip(sc, truth) if y == 0]
        correct = sum(1 for s, y in zip(sc, truth) if (s >= 0.5) == bool(y))
        ai_caught = sum(1 for s in ai if s >= 0.5)
        print(f"{n:<50} meanAI={sum(ai)/len(ai):.3f}  meanREAL={sum(re_)/len(re_):.3f}  "
              f"AI caught {ai_caught}/{len(ai)}  correct {correct}/{len(sc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
