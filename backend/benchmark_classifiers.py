"""Score several pretrained detectors over the labelled sets, for model choice.

Inference only -- nothing here trains or fine-tunes anything. It exists
because the shipped classifier (Organika/sdxl-detector) is an SDXL-specific
detector, and on real-world files it is close to anti-correlated: it scored
genuine iPhone photographs at 0.998 "artificial" and Gemini-generated images
at 0.003.

Writes one row per image per model to classifier_benchmark.csv so the choice
can be made from measured AUC rather than from a model card's claims.

    .venv/Scripts/python benchmark_classifiers.py
"""

from __future__ import annotations

import csv
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import config

CANDIDATES = [
    "Organika/sdxl-detector",                        # current, the baseline
    "Ateeqq/ai-vs-human-image-detector",
    "dima806/ai_vs_real_image_detection",
    "dima806/ai_vs_human_generated_image_detection",
    "haywoodsloan/ai-image-detector-deploy",
]

# Label vocabularies, taken from each model's own id2label -- never guessed.
# A model whose labels don't resolve here is skipped rather than assigned a
# direction by assumption (see invariant 2 in CLAUDE.md).
AI_WORDS = {"artificial", "ai", "ai-generated", "ai_generated", "fake",
            "generated", "machine", "sdxl", "synthetic", "deepfake"}
REAL_WORDS = {"human", "hum", "real", "natural", "photo", "photograph",
              "authentic", "nature"}

BATCH = 16


def ai_probability(preds) -> float | None:
    for p in preds:
        if str(p["label"]).strip().lower() in AI_WORDS:
            return float(p["score"])
    for p in preds:
        if str(p["label"]).strip().lower() in REAL_WORDS:
            return 1.0 - float(p["score"])
    return None


def load_set(images_dir: Path, manifest: Path) -> list[tuple[Path, str]]:
    labels = {}
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("filename") or "").strip()
                if name and not name.startswith("#"):
                    labels[name] = (row.get("expected") or "UNKNOWN").strip().upper()
    out = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in config.IMAGE_EXTENSIONS:
            continue
        truth = labels.get(p.name, "UNKNOWN")
        if truth in ("AI", "REAL"):
            out.append((p, truth))
    return out


def main() -> int:
    from PIL import Image
    from transformers import pipeline

    sets = {
        "public": load_set(Path("evaluation_set/images"),
                           Path("evaluation_set/MANIFEST.csv")),
        "samples": load_set(Path("samples"), Path("samples/MANIFEST.csv")),
    }
    for name, items in sets.items():
        n_ai = sum(1 for _, t in items if t == "AI")
        print(f"{name:<9} {len(items):>4} labelled  ({n_ai} AI / {len(items)-n_ai} REAL)")
    total = sum(len(v) for v in sets.values())
    print(f"\n{len(CANDIDATES)} models x {total} images = {len(CANDIDATES)*total} "
          f"inferences; roughly {len(CANDIDATES)*total*0.35/60:.0f} min on CPU\n")

    rows = []
    for model_name in CANDIDATES:
        print(f"--- {model_name}")
        started = time.perf_counter()
        try:
            clf = pipeline("image-classification", model=model_name, device="cpu")
        except Exception as exc:
            print(f"    could not load: {type(exc).__name__}: {exc}")
            continue
        labels = list(clf.model.config.id2label.values())
        probe = ai_probability([{"label": l, "score": 0.0} for l in labels])
        if probe is None:
            print(f"    labels {labels} do not resolve to an AI/real direction "
                  f"-- skipped rather than guessed")
            continue
        print(f"    labels={labels}")

        for set_name, items in sets.items():
            for i in range(0, len(items), BATCH):
                chunk = items[i:i + BATCH]
                images, keep = [], []
                for path, truth in chunk:
                    try:
                        with Image.open(path) as im:
                            images.append(im.convert("RGB"))
                        keep.append((path, truth))
                    except Exception:
                        pass
                if not images:
                    continue
                try:
                    raw = clf(images, top_k=None)
                except Exception as exc:
                    print(f"    inference failed on a batch: {type(exc).__name__}")
                    continue
                if raw and isinstance(raw[0], dict):
                    raw = [raw]
                for (path, truth), preds in zip(keep, raw):
                    score = ai_probability(preds)
                    if score is None:
                        continue
                    rows.append({"model": model_name, "set": set_name,
                                 "filename": path.name, "expected": truth,
                                 "score": round(score, 6)})
            print(f"    {set_name}: done")
        del clf
        print(f"    {time.perf_counter() - started:.0f}s")

    out = Path("classifier_benchmark.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "set", "filename",
                                           "expected", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
