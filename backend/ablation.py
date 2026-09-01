"""Ablation study and threshold selection. PRD deliverables D6 and D7.

Everything here is recomputed from the per-signal columns already written by
evaluate.py. No inference is re-run: each configuration just re-fuses the
scores that were dumped per image, which is the whole reason those columns
exist.

D6 -- ablation
--------------
Six configurations, each fused by the same rule the live pipeline uses:

    confidence = sum(score x weight) / sum(weight)   over AVAILABLE signals

Dropping a signal from a configuration is exactly the same operation as that
signal being unavailable at runtime, so the ablation exercises the real fusion
path rather than a parallel reimplementation of it.

A configuration in which no signal is available for a given image yields 0.5
and UNCERTAIN, matching the backend.

D7 -- threshold selection
-------------------------
Thresholds are chosen on one half of the data and reported on the other. A
threshold picked on the same rows it is scored against is fitted to those rows,
and the number it produces is not an estimate of anything. The split is
stratified by class and seeded so it reproduces.

Usage
-----
    .venv/Scripts/python ablation.py --report evaluation_report_public.csv
    .venv/Scripts/python ablation.py --report evaluation_report_public.csv \\
        --compare evaluation_report_public_normalised.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import config
from evaluate import roc_auc, safe_divide

SIGNALS = ("model", "metadata", "frequency")

CONFIGURATIONS = [
    ("full ensemble", ("model", "metadata", "frequency")),
    ("model only", ("model",)),
    ("metadata only", ("metadata",)),
    ("frequency only", ("frequency",)),
    ("ensemble - metadata", ("model", "frequency")),
    ("ensemble - frequency", ("model", "metadata")),
]


# ---------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["expected"] in ("AI", "REAL")]
    for row in rows:
        for key in SIGNALS:
            raw = row.get(f"{key}_score", "")
            row[f"{key}_score"] = float(raw) if raw not in ("", None) else 0.0
            row[f"{key}_available"] = row.get(f"{key}_available") == "yes"
    return rows


def fuse(row: dict, signals: tuple[str, ...]) -> tuple[float, int]:
    """Re-fuse one row over a subset. Returns (confidence, signals_used)."""
    total_weight = sum(
        config.WEIGHTS[key] for key in signals if row[f"{key}_available"]
    )
    if total_weight <= 0:
        # Same rule as the backend: total signal failure is maximum ignorance,
        # not a reading of "real".
        return 0.5, 0
    weighted = sum(
        row[f"{key}_score"] * config.WEIGHTS[key]
        for key in signals if row[f"{key}_available"]
    )
    used = sum(1 for key in signals if row[f"{key}_available"])
    return weighted / total_weight, used


def verdict_for(confidence: float, low: float, high: float) -> str:
    if confidence >= high:
        return "AI"
    if confidence <= low:
        return "REAL"
    return "ABSTAIN"


def score_configuration(rows, signals, low, high) -> dict:
    confidences, truth, decisions = [], [], []
    never_available = 0

    for row in rows:
        confidence, used = fuse(row, signals)
        if used == 0:
            never_available += 1
        confidences.append(confidence)
        truth.append(1 if row["expected"] == "AI" else 0)
        decisions.append(verdict_for(confidence, low, high))

    tp = sum(1 for d, t in zip(decisions, truth) if d == "AI" and t == 1)
    fp = sum(1 for d, t in zip(decisions, truth) if d == "AI" and t == 0)
    tn = sum(1 for d, t in zip(decisions, truth) if d == "REAL" and t == 0)
    fn = sum(1 for d, t in zip(decisions, truth) if d == "REAL" and t == 1)
    abstained = sum(1 for d in decisions if d == "ABSTAIN")
    decided = tp + fp + tn + fn

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        safe_divide(2 * precision * recall, precision + recall)
        if precision and recall else None
    )

    return {
        "n": len(rows),
        "decided": decided,
        "abstained": abstained,
        "no_signal": never_available,
        "accuracy": safe_divide(tp + tn, decided),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc(confidences, truth),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def cell(value) -> str:
    return "  --  " if value is None else f"{value:.4f}"


def print_table(title: str, results: dict) -> None:
    print(f"\n{title}")
    print(f"{'configuration':<22}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}"
          f"{'AUC':>8}{'decided':>9}{'abstain':>9}")
    print("-" * 80)
    for name, _ in CONFIGURATIONS:
        r = results[name]
        flag = "   <- no signal available" if r["no_signal"] == r["n"] else ""
        print(f"{name:<22}{cell(r['accuracy']):>8}{cell(r['precision']):>8}"
              f"{cell(r['recall']):>8}{cell(r['f1']):>8}{cell(r['auc']):>8}"
              f"{r['decided']:>9}{r['abstained']:>9}{flag}")


# ---------------------------------------------------------------------------
# D7
# ---------------------------------------------------------------------------
def stratified_halves(rows: list[dict], seed: int = 20260903):
    """Split by class so both halves keep the 50/50 balance."""
    rng = random.Random(seed)
    selection, reporting = [], []
    for label in ("AI", "REAL"):
        group = [r for r in rows if r["expected"] == label]
        rng.shuffle(group)
        middle = len(group) // 2
        selection.extend(group[:middle])
        reporting.extend(group[middle:])
    return selection, reporting


def best_single_cut(rows: list[dict], signals: tuple[str, ...]) -> tuple[float, float]:
    """Threshold maximising accuracy with no abstention band."""
    scored = [(fuse(r, signals)[0], 1 if r["expected"] == "AI" else 0) for r in rows]
    best_accuracy, best_threshold = 0.0, 0.5
    for step in range(1001):
        threshold = step / 1000
        correct = sum(1 for c, t in scored if (c >= threshold) == bool(t))
        accuracy = correct / len(scored)
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, threshold
    return best_threshold, best_accuracy


def best_band(rows: list[dict], signals: tuple[str, ...]) -> tuple[float, float, float, float]:
    """Best (low, high) band, scored on decided rows, requiring >=70% coverage.

    Without a coverage floor the search degenerates: a band covering almost
    everything abstains on every hard case and reports near-perfect accuracy on
    the handful it still answers. Coverage is reported alongside so the
    trade-off is visible rather than hidden inside one number.
    """
    scored = [(fuse(r, signals)[0], 1 if r["expected"] == "AI" else 0) for r in rows]
    best = (config.THRESHOLD_LIKELY_REAL, config.THRESHOLD_AI_GENERATED, 0.0, 0.0)
    for low_step in range(0, 101, 2):
        low = low_step / 100
        for high_step in range(low_step, 101, 2):
            high = high_step / 100
            decided = [(c, t) for c, t in scored if c >= high or c <= low]
            if len(decided) < 0.70 * len(scored):
                continue
            correct = sum(1 for c, t in decided if (c >= high) == bool(t))
            accuracy = correct / len(decided)
            if accuracy > best[2]:
                best = (low, high, accuracy, len(decided) / len(scored))
    return best


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="naive evaluation_report csv")
    parser.add_argument("--compare", help="normalised evaluation_report csv")
    parser.add_argument("--out", default="ablation.json")
    args = parser.parse_args()

    low = config.THRESHOLD_LIKELY_REAL
    high = config.THRESHOLD_AI_GENERATED

    conditions = [("naive", Path(args.report))]
    if args.compare:
        conditions.append(("normalised", Path(args.compare)))

    payload = {"thresholds_used": {"likely_real": low, "ai_generated": high},
               "weights": config.WEIGHTS, "conditions": {}}
    loaded = {}

    print("=" * 80)
    print("D6 -- ABLATION   (recomputed from the per-signal dump, no re-inference)")
    print("=" * 80)
    print(f"fusion rule: sum(score x weight) / sum(weight) over available signals")
    print(f"thresholds : REAL <= {low}  |  AI >= {high}")

    for name, path in conditions:
        if not path.exists():
            print(f"\n! {path} not found, skipping the {name} condition")
            continue
        rows = load_rows(path)
        loaded[name] = rows
        results = {
            label: score_configuration(rows, signals, low, high)
            for label, signals in CONFIGURATIONS
        }
        payload["conditions"][name] = results
        print_table(f"{name.upper()}  (n={len(rows)})", results)

    # --- D7 -----------------------------------------------------------------
    print()
    print("=" * 80)
    print("D7 -- THRESHOLD SELECTION   (chosen on one half, reported on the other)")
    print("=" * 80)

    payload["threshold_selection"] = {}
    for name, rows in loaded.items():
        selection, reporting = stratified_halves(rows)
        full = ("model", "metadata", "frequency")

        cut, cut_accuracy = best_single_cut(selection, full)
        band_low, band_high, band_accuracy, band_coverage = best_band(selection, full)

        held_cut = [(fuse(r, full)[0], 1 if r["expected"] == "AI" else 0)
                    for r in reporting]
        held_cut_accuracy = sum(
            1 for c, t in held_cut if (c >= cut) == bool(t)
        ) / len(held_cut)

        held_band = score_configuration(reporting, full, band_low, band_high)
        current = score_configuration(reporting, full, low, high)

        print(f"\n{name.upper()}  selection n={len(selection)}, "
              f"reporting n={len(reporting)}")
        print(f"  single cut chosen on selection : {cut:.3f} "
              f"(accuracy {cut_accuracy:.4f} on selection)")
        print(f"  ... same cut on HELD-OUT half  : {held_cut_accuracy:.4f}")
        print(f"  band chosen on selection       : "
              f"REAL <= {band_low:.2f} | AI >= {band_high:.2f} "
              f"(accuracy {band_accuracy:.4f}, coverage {band_coverage:.2f})")
        print(f"  ... same band on HELD-OUT half : "
              f"accuracy {cell(held_band['accuracy'])}, "
              f"coverage {held_band['decided'] / held_band['n']:.2f}")
        print(f"  current band {low}/{high} on HELD-OUT half : "
              f"accuracy {cell(current['accuracy'])}, "
              f"coverage {current['decided'] / current['n']:.2f}")

        payload["threshold_selection"][name] = {
            "selection_n": len(selection),
            "reporting_n": len(reporting),
            "single_cut": {"threshold": cut,
                           "accuracy_on_selection": cut_accuracy,
                           "accuracy_on_heldout": held_cut_accuracy},
            "band": {"low": band_low, "high": band_high,
                     "accuracy_on_selection": band_accuracy,
                     "coverage_on_selection": band_coverage,
                     "accuracy_on_heldout": held_band["accuracy"],
                     "coverage_on_heldout": held_band["decided"] / held_band["n"]},
            "current_band_on_heldout": {
                "low": low, "high": high,
                "accuracy": current["accuracy"],
                "coverage": current["decided"] / current["n"]},
        }

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
