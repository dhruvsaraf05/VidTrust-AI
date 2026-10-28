"""Evaluate the detector against a labelled set. PRD deliverable D5.

Two tracks, deliberately never merged into one headline number
--------------------------------------------------------------
    --track public    the Community Forensics subset in evaluation_set/
                      (~400 images). Public datasets are redistributed
                      re-encoded, so EXIF is stripped across the whole set and
                      the provenance signal cannot fire. This track measures
                      the CLASSIFIER and FREQUENCY signals at scale.

    --track samples   the hand-collected set in samples/ (~20 files). Small,
                      but the only set carrying intact provenance metadata, so
                      it is the only one that exercises the metadata signal.

Averaging those together would produce a number that describes neither: the
public track would drown the samples, and the ensemble's headline would silently
become a two-signal result. They get separate outputs and separate reporting.

Note on the samples track: those files were used for development and are the
demo material. Its figures are a sanity check, not a held-out estimate, and the
report should say so.

UNCERTAIN
---------
Recorded as neither correct nor incorrect. It is the system declining to
answer, and forcing it into the confusion matrix would either invent errors or
inflate accuracy. It is counted separately and reported as coverage.

Outputs
-------
    evaluation_report_<track>.csv    one row per file, every signal's raw score
                                     AND its availability flag
    evaluation_metrics_<track>.json  accuracy, precision, recall, F1, ROC-AUC,
                                     confusion matrix, abstention and degraded
                                     counts

The per-signal dump exists so the ablation study can be recomputed from the CSV
without re-running inference. Do not remove those columns.

Usage
-----
    .venv/Scripts/python evaluate.py --track public
    .venv/Scripts/python evaluate.py --track samples
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import config
import perturbations
from detectors import model_detector
from pipeline import aggregator

SIGNALS = ("model", "metadata", "frequency")

TRACKS = {
    "public": {
        "images": Path("evaluation_set/images"),
        "manifest": Path("evaluation_set/MANIFEST.csv"),
        "carry": ["model_name", "architecture", "real_source", "subset"],
        "note": "Community Forensics (small), CVPR 2025. EXIF stripped by redistribution.",
    },
    "samples": {
        "images": Path("samples"),
        "manifest": Path("samples/MANIFEST.csv"),
        "carry": ["source", "notes"],
        "note": "Hand-collected demo/development set. Not a held-out estimate.",
    },
}

# Per-file cost on CPU. Used only for the up-front estimate; the live ETA
# replaces it after a few files.
#
# Measured from the runs' own elapsed_seconds, not from API_NOTES.md: the naive
# public pass did 400 files in 517.6 s (1.29 s/file) and the normalised pass in
# 850.7 s (2.13 s/file). The old value of 0.4 predated the classifier swap and
# was 4x optimistic, which made a 40-minute sweep look like it had hung.
SECONDS_PER_IMAGE = 1.6
SECONDS_PER_VIDEO = 3.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def roc_auc(scores: list[float], positives: list[int]) -> float | None:
    """AUC via the rank (Mann-Whitney U) identity, ties averaged.

    Computed over every labelled file including the abstentions: AUC ranks the
    continuous confidence and never consults a threshold, so excluding
    UNCERTAIN rows would discard real information about the ranking.
    """
    pairs = sorted(zip(scores, positives), key=lambda pair: pair[0])
    total = len(pairs)
    n_pos = sum(positives)
    n_neg = total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    rank_sum_pos = 0.0
    index = 0
    while index < total:
        end = index
        while end + 1 < total and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            if pairs[position][1] == 1:
                rank_sum_pos += average_rank
        index = end + 1

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


# ---------------------------------------------------------------------------
# Resolution normalisation
# ---------------------------------------------------------------------------
CROP_SIZE = 512
NORMALISED_DIRNAME = "_normalised"
PERTURBED_DIRNAME = "_perturbed"


def condition_name(normalise: str, perturb: str) -> str:
    """Stable name for a (normalisation, perturbation) pair.

    The two legacy conditions keep the names they already have on disk, because
    ablation.json, paper/audit_numbers.py and every committed figure refer to
    them:

        none / none   -> ""            evaluation_report_public.csv
        crop / none   -> "_normalised" evaluation_report_public_normalised.csv

    Anything with a perturbation gets an explicit arm prefix, so a filename says
    which arm it came from and the two can never be confused in a table:

        crop / jpeg50 -> "_crop_jpeg50"
        none / jpeg50 -> "_native_jpeg50"
    """
    if perturb == "none":
        return "_normalised" if normalise == "crop" else ""
    return f"_{'crop' if normalise == 'crop' else 'native'}_{perturb}"


def _crop_native(image: Image.Image, size: int) -> tuple[Image.Image, str]:
    """The centre crop itself, split out so a perturbation can follow it."""
    note = ""
    width, height = image.size

    if width < size or height < size:
        scale = size / min(width, height)
        image = image.resize(
            (max(size, round(width * scale)), max(size, round(height * scale))),
            Image.LANCZOS,
        )
        width, height = image.size
        note = "upscaled"

    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size)), note


def prepare_image(source: Path, out_dir: Path, *, normalise: str = "none",
                  perturb: str = "none") -> tuple[Path, str]:
    """Crop, then perturb, then write a lossless PNG. Returns (path, note).

    ORDER IS CROP THEN PERTURB, and it matters: JPEG-of-crop and crop-of-JPEG
    differ, for JPEG by whether the 8x8 block grid aligns to the crop origin.
    The order is recorded in the metrics JSON so a run can be reproduced.

    The output is always PNG -- lossless, so the container adds nothing on top
    of whatever the perturbation already did in memory.

    ---- why the crop exists (this is the control, not the perturbation) ----

    The confound this removes
    -------------------------
    In the public set the real images are natively 1024x1024 and the generated
    ones 512x512. Every detector resizes to 512 internally, so the real images
    arrive having been downsampled -- which destroys high-frequency content --
    while the generated images arrive untouched. The frequency signal then
    separates the classes by measuring who got downsampled, not who was
    synthesised.

    Why cropping rather than resizing
    ---------------------------------
    Resizing everything to a common target does NOT fix this. A 1024 image
    resized to 512 is still downsampled 2x while a native 512 image is not, so
    the differential survives the "identical" operation -- the operation is the
    same but its EFFECT depends on the input size, which is exactly what
    differs by class.

    A centre crop takes CROP_SIZE native pixels out of whatever it is given.
    Nothing is interpolated, no frequency content is created or destroyed, and
    every image reaches the detectors as an unresampled 512x512 block. That
    makes the processing history genuinely identical across classes, which is
    the requirement.

    What it does not fix: the crop shows a sub-region of a large image and the
    whole of a small one, so framing differs. That is a content difference, not
    a resampling one, and it is reported separately.

    Images smaller than CROP_SIZE on a side must be scaled up first -- that
    reintroduces resampling for those files, so the count is reported.

    Saved as PNG: lossless, so the crop itself adds no compression artefacts.
    EXIF is not carried across, which is why this mode is refused on tracks
    where provenance matters.
    """
    note = ""
    with Image.open(source) as handle:
        image = handle.convert("RGB")

    if normalise == "crop":
        image, note = _crop_native(image, CROP_SIZE)

    if perturb != "none":
        image = perturbations.apply(perturb, image)

    target = out_dir / f"{source.stem}.png"
    image.save(target, format="PNG")
    return target, note


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def load_manifest(path: Path, carry: list[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    entries: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("filename") or "").strip()
            if not name or name.startswith("#"):
                continue
            label = (row.get("expected") or "UNKNOWN").strip().upper()
            entries[name] = {
                "expected": label if label in {"AI", "REAL"} else "UNKNOWN",
                **{key: (row.get(key) or "").strip() for key in carry},
            }
    return entries


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N files (for a quick smoke run)")
    parser.add_argument("--normalise", choices=["none", "crop"], default="none",
                        help="crop: centre-crop every image to 512x512 native "
                             "pixels before analysis, removing the resampling "
                             "confound. See prepare_image().")
    parser.add_argument("--perturb", choices=sorted(perturbations.PERTURBATIONS),
                        default="none",
                        help="degrade every image before analysis, for the "
                             "operating-envelope study. Orthogonal to "
                             "--normalise, and applied AFTER it: pair it with "
                             "--normalise crop for the measurement, or use it "
                             "alone to demonstrate the resolution confound.")
    args = parser.parse_args()

    track = TRACKS[args.track]
    folder = track["images"]
    if not folder.is_dir():
        print(f"No such directory: {folder}")
        if args.track == "public":
            print("Build it first: .venv/Scripts/python build_evaluation_set.py")
        return 1

    manifest = load_manifest(track["manifest"], track["carry"])
    files = sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in config.ALLOWED_EXTENSIONS
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No media files in {folder}/")
        return 1

    videos = [p for p in files if p.suffix.lower() in config.VIDEO_EXTENSIONS]
    images = [p for p in files if p not in videos]
    labelled = sum(
        1 for p in files if manifest.get(p.name, {}).get("expected") in ("AI", "REAL")
    )

    estimate = len(images) * SECONDS_PER_IMAGE + len(videos) * SECONDS_PER_VIDEO
    print(f"track     : {args.track} -- {track['note']}")
    print(f"files     : {len(files)} ({len(images)} images, {len(videos)} video)")
    print(f"labelled  : {labelled} of {len(files)} usable for metrics")
    print(f"estimate  : ~{estimate / 60:.1f} min "
          f"({estimate:.0f}s at {SECONDS_PER_IMAGE}s/image on CPU)")
    if labelled == 0:
        print("\nNothing is labelled AI or REAL, so no metrics can be computed.")
        print(f"Fill in {track['manifest']} first.")
        return 1
    print()

    # Both the crop and every perturbation re-save the pixels, which discards
    # EXIF. Neither may ever run on a track whose whole purpose is provenance.
    prepared_dir = None
    upscaled_count = 0
    preprocessing = args.normalise == "crop" or args.perturb != "none"

    if preprocessing:
        if args.track == "samples":
            reason = ("--normalise crop" if args.normalise == "crop"
                      else f"--perturb {args.perturb}")
            print(f"{reason} re-saves images, which discards EXIF.")
            print("The samples track exists to exercise provenance, so refusing.")
            return 1

        if args.perturb == "none":
            # Legacy path: keep writing to _normalised/ so the existing
            # condition reproduces byte-for-byte.
            prepared_dir = folder.parent / NORMALISED_DIRNAME
        else:
            prepared_dir = (folder.parent / PERTURBED_DIRNAME
                            / condition_name(args.normalise, args.perturb).lstrip("_"))
        prepared_dir.mkdir(parents=True, exist_ok=True)
        for stale in prepared_dir.glob("*"):
            stale.unlink()

        if args.normalise == "crop":
            print(f"normalise : centre-crop {CROP_SIZE}x{CROP_SIZE} native "
                  f"pixels, no resampling")
        if args.perturb != "none":
            print(f"perturb   : {args.perturb} -- "
                  f"{perturbations.describe(args.perturb)}")
            if args.normalise != "crop":
                print("            ! NATIVE arm: the 1024-vs-512 resolution "
                      "split is uncontrolled here.")
                print("            ! This arm demonstrates the confound; it "
                      "does not measure robustness.")
        print(f"            writing to {prepared_dir}/\n")

    print("Loading classifier...")
    if not model_detector.load_model():
        print(f"  ! classifier unavailable: {model_detector.load_error()}")
        print("  ! continuing -- the model signal will report unavailable "
              "throughout, which the report will show")
    print()

    records = []
    started = time.perf_counter()

    for position, path in enumerate(files, start=1):
        entry = manifest.get(path.name, {})
        expected = entry.get("expected", "UNKNOWN")
        is_video = path.suffix.lower() in config.VIDEO_EXTENSIONS

        file_started = time.perf_counter()
        try:
            analysis_path = path
            if prepared_dir is not None and not is_video:
                analysis_path, note = prepare_image(
                    path, prepared_dir,
                    normalise=args.normalise, perturb=args.perturb,
                )
                if note == "upscaled":
                    upscaled_count += 1

            result = (
                aggregator.analyze_video(str(analysis_path))
                if is_video
                else aggregator.analyze_image(str(analysis_path))
            )
            error = ""
        except Exception as exc:  # a corrupt file is a result, not a crash
            result = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = int((time.perf_counter() - file_started) * 1000)

        record = {
            "filename": path.name,
            "expected": expected,
            **{key: entry.get(key, "") for key in track["carry"]},
            "media_type": "video" if is_video else "image",
            "processing_ms": elapsed_ms,
            "error": error,
        }

        if result is None:
            record["verdict"] = ""
            record["confidence"] = ""
            record["decision"] = "ERROR"
            record["degraded"] = ""
            for key in SIGNALS:
                record[f"{key}_score"] = ""
                record[f"{key}_available"] = ""
        else:
            signals = result["signals"]
            verdict = result["verdict"]
            record["verdict"] = verdict
            record["confidence"] = round(result["confidence"], 6)
            record["decision"] = {
                "AI_GENERATED": "AI",
                "LIKELY_REAL": "REAL",
                "UNCERTAIN": "ABSTAIN",
            }[verdict]
            record["degraded"] = (
                "yes" if any(not signals[k]["available"] for k in SIGNALS) else "no"
            )
            # The per-signal dump. Ablation is recomputed from these columns
            # without re-running inference.
            for key in SIGNALS:
                record[f"{key}_score"] = round(signals[key]["score"], 6)
                record[f"{key}_available"] = (
                    "yes" if signals[key]["available"] else "no"
                )

        # Correctness is only defined for a labelled, decided file.
        if expected in ("AI", "REAL") and record["decision"] in ("AI", "REAL"):
            record["correct"] = "yes" if record["decision"] == expected else "no"
        else:
            record["correct"] = ""

        records.append(record)

        if position % 10 == 0 or position == len(files):
            rate = (time.perf_counter() - started) / position
            remaining = rate * (len(files) - position)
            print(f"  {position:>4}/{len(files)}  "
                  f"{rate:.2f}s/file  ~{remaining / 60:.1f} min left")

    total_elapsed = time.perf_counter() - started

    # --- metrics ------------------------------------------------------------
    scored = [r for r in records if r["expected"] in ("AI", "REAL") and r["decision"] != "ERROR"]
    decided = [r for r in scored if r["decision"] in ("AI", "REAL")]
    abstained = [r for r in scored if r["decision"] == "ABSTAIN"]

    tp = sum(1 for r in decided if r["expected"] == "AI" and r["decision"] == "AI")
    fp = sum(1 for r in decided if r["expected"] == "REAL" and r["decision"] == "AI")
    tn = sum(1 for r in decided if r["expected"] == "REAL" and r["decision"] == "REAL")
    fn = sum(1 for r in decided if r["expected"] == "AI" and r["decision"] == "REAL")

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        safe_divide(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    auc = roc_auc(
        [float(r["confidence"]) for r in scored],
        [1 if r["expected"] == "AI" else 0 for r in scored],
    )

    availability = {
        key: sum(1 for r in records if r.get(f"{key}_available") == "yes")
        for key in SIGNALS
    }

    metrics = {
        "track": args.track,
        "note": track["note"],
        "normalisation": (
            f"centre-crop {CROP_SIZE}x{CROP_SIZE} native pixels, no resampling"
            if args.normalise == "crop" else "none (images analysed as-is)"
        ),
        "upscaled_before_crop": upscaled_count,
        # The operating-envelope study. `arm` is the one field that says whether
        # a number means anything: on the native arm the 1024-vs-512 resolution
        # split is uncontrolled, so those rows demonstrate the confound rather
        # than measuring robustness. See perturbations.py.
        "perturbation": perturbations.describe(args.perturb),
        "perturbation_name": args.perturb,
        "preprocessing_order": ("crop then perturb, saved lossless as PNG"
                                if preprocessing else "none"),
        "arm": ("matched" if args.normalise == "crop" else "native"),
        "condition": condition_name(args.normalise, args.perturb).lstrip("_") or "clean",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(total_elapsed, 1),
        "model": {
            "name": model_detector.model_name(),
            "available": model_detector.is_available(),
        },
        "weights": config.WEIGHTS,
        "thresholds": {
            "ai_generated": config.THRESHOLD_AI_GENERATED,
            "likely_real": config.THRESHOLD_LIKELY_REAL,
        },
        "counts": {
            "files": len(records),
            "errored": sum(1 for r in records if r["decision"] == "ERROR"),
            "labelled": len(scored),
            "decided": len(decided),
            "uncertain": len(abstained),
            "degraded": sum(1 for r in records if r.get("degraded") == "yes"),
            "signal_available": availability,
        },
        # Coverage is the honest companion to accuracy: a detector that
        # abstains on half the set and is right on the rest is not "accurate".
        "coverage": safe_divide(len(decided), len(scored)),
        "confusion_matrix": {
            "positive_class": "AI",
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "excluded_uncertain": len(abstained),
        },
        "metrics_on_decided": {
            "accuracy": safe_divide(tp + tn, len(decided)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "roc_auc_all_labelled": auc,
    }

    # --- write --------------------------------------------------------------
    columns = (
        ["filename", "expected", *track["carry"], "media_type", "verdict",
         "confidence", "decision", "correct", "degraded"]
        + [f"{key}_{suffix}" for key in SIGNALS for suffix in ("score", "available")]
        + ["processing_ms", "error"]
    )
    suffix = condition_name(args.normalise, args.perturb)
    report_path = Path(f"evaluation_report_{args.track}{suffix}.csv")
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    metrics_path = Path(f"evaluation_metrics_{args.track}{suffix}.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # --- report -------------------------------------------------------------
    def show(label: str, value) -> str:
        return f"{label:<22}{'n/a' if value is None else f'{value:.4f}'}"

    print(f"\n{'=' * 62}")
    condition_label = (
        f"{'NORMALISED' if args.normalise == 'crop' else 'naive'}"
        + (f", {args.perturb.upper()}" if args.perturb != "none" else "")
    )
    print(f"TRACK {args.track.upper()} ({condition_label}) "
          f"-- {len(records)} files in {total_elapsed:.0f}s")
    print("=" * 62)
    print(f"labelled              {len(scored)}")
    print(f"decided               {len(decided)}")
    print(f"uncertain (abstained) {len(abstained)}   "
          f"-- neither correct nor incorrect")
    print(f"degraded runs         {metrics['counts']['degraded']}   "
          f"-- at least one signal unavailable")
    print(f"signal availability   {availability}")
    if args.normalise == "crop" and upscaled_count:
        print(f"upscaled before crop  {upscaled_count}  "
              f"-- these DID get resampled; the rest did not")
    print()
    print(f"confusion matrix (positive = AI, {len(abstained)} abstentions excluded)")
    print(f"                   predicted AI   predicted REAL")
    print(f"  actual AI        {tp:>12}   {fn:>14}")
    print(f"  actual REAL      {fp:>12}   {tn:>14}")
    print()
    print(show("accuracy", metrics["metrics_on_decided"]["accuracy"]))
    print(show("precision", precision))
    print(show("recall", recall))
    print(show("f1", f1))
    print(show("roc-auc", auc))
    print(show("coverage", metrics["coverage"]))
    print()
    print(f"wrote {report_path} and {metrics_path}")
    print("\nThese figures describe this track only. Do not average the two "
          "tracks into a single headline number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
