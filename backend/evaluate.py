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

import config
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

# Per-file cost on CPU, from the measured timings in API_NOTES.md. Used only
# for the up-front estimate; the live ETA replaces it after a few files.
SECONDS_PER_IMAGE = 0.4
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
            result = (
                aggregator.analyze_video(str(path))
                if is_video
                else aggregator.analyze_image(str(path))
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
    report_path = Path(f"evaluation_report_{args.track}.csv")
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    metrics_path = Path(f"evaluation_metrics_{args.track}.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # --- report -------------------------------------------------------------
    def show(label: str, value) -> str:
        return f"{label:<22}{'n/a' if value is None else f'{value:.4f}'}"

    print(f"\n{'=' * 62}")
    print(f"TRACK {args.track.upper()} -- {len(records)} files in {total_elapsed:.0f}s")
    print("=" * 62)
    print(f"labelled              {len(scored)}")
    print(f"decided               {len(decided)}")
    print(f"uncertain (abstained) {len(abstained)}   "
          f"-- neither correct nor incorrect")
    print(f"degraded runs         {metrics['counts']['degraded']}   "
          f"-- at least one signal unavailable")
    print(f"signal availability   {availability}")
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
