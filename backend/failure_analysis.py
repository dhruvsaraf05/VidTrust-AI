"""Failure analysis. PRD deliverable D8.

Answers "which generators slip through", not "some images were wrong". The
evaluation set records the `model_name` and `architecture` behind every
generated image, so a miss can be attributed rather than merely counted.

Two views:

  1. The ten worst misclassifications by MARGIN -- how far past the decision
     threshold the fused confidence sat, in the wrong direction. A file that
     missed by 0.01 is a threshold problem; one that missed by 0.40 is a
     detector problem, and they deserve different sentences in the report.

  2. A per-generator table. A generator is "missed" when its image was decided
     REAL, and "ducked" when the system abstained. Abstention is counted
     separately because declining to answer is not the same as being wrong.

Run against the NORMALISED report: on the naive one the resolution confound is
doing part of the work, so the attribution would be partly fictional.

Usage
-----
    .venv/Scripts/python failure_analysis.py \\
        --report evaluation_report_public_normalised.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", default="failure_analysis.json")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    low = config.THRESHOLD_LIKELY_REAL
    high = config.THRESHOLD_AI_GENERATED

    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["expected"] in ("AI", "REAL")]

    for row in rows:
        row["confidence"] = float(row["confidence"]) if row["confidence"] else 0.0

    # --- 1. worst misclassifications by margin ------------------------------
    wrong = []
    for row in rows:
        if row["decision"] == "REAL" and row["expected"] == "AI":
            margin = low - row["confidence"]          # how deep into REAL
            direction = "AI called REAL"
        elif row["decision"] == "AI" and row["expected"] == "REAL":
            margin = row["confidence"] - high         # how deep into AI
            direction = "REAL called AI"
        else:
            continue
        wrong.append({**row, "margin": margin, "direction": direction})

    wrong.sort(key=lambda r: -r["margin"])
    worst = wrong[: args.top]

    print("=" * 92)
    print(f"D8 -- {args.top} WORST MISCLASSIFICATIONS BY MARGIN")
    print(f"source: {path.name}   thresholds: REAL <= {low} | AI >= {high}")
    print("=" * 92)
    print(f"{'margin':>7}  {'direction':<16}{'conf':>6}{'model':>7}{'freq':>7}  "
          f"{'generator / architecture':<44}")
    print("-" * 92)
    for row in worst:
        generator = row.get("model_name") or "(real photo)"
        architecture = row.get("architecture") or "-"
        model_score = row.get("model_score") or "-"
        freq_score = row.get("frequency_score") or "-"
        print(f"{row['margin']:>7.3f}  {row['direction']:<16}"
              f"{row['confidence']:>6.3f}"
              f"{float(model_score):>7.3f}{float(freq_score):>7.3f}  "
              f"{(generator + ' / ' + architecture)[:44]:<44}")

    if not worst:
        print("  (no misclassifications)")

    # --- 2. per-generator attribution ---------------------------------------
    stats = defaultdict(lambda: {"n": 0, "missed": 0, "abstained": 0,
                                 "caught": 0, "architecture": "",
                                 "model_sum": 0.0})
    for row in rows:
        if row["expected"] != "AI":
            continue
        key = row.get("model_name") or "unknown"
        entry = stats[key]
        entry["n"] += 1
        entry["architecture"] = row.get("architecture") or ""
        try:
            entry["model_sum"] += float(row.get("model_score") or 0.0)
        except ValueError:
            pass
        if row["decision"] == "REAL":
            entry["missed"] += 1
        elif row["decision"] == "ABSTAIN":
            entry["abstained"] += 1
        else:
            entry["caught"] += 1

    table = []
    for generator, entry in stats.items():
        slipped = entry["missed"] + entry["abstained"]
        table.append({
            "generator": generator,
            "architecture": entry["architecture"],
            "n": entry["n"],
            "caught": entry["caught"],
            "missed": entry["missed"],
            "abstained": entry["abstained"],
            "slip_rate": slipped / entry["n"] if entry["n"] else None,
            "mean_model_score": entry["model_sum"] / entry["n"] if entry["n"] else None,
        })
    table.sort(key=lambda r: (-r["slip_rate"], -r["n"]))

    fully_missed = [r for r in table if r["caught"] == 0]
    fully_caught = [r for r in table if r["slip_rate"] == 0]

    print()
    print("=" * 92)
    print("GENERATORS RANKED BY SLIP RATE   (missed + abstained, out of that "
          "generator's images)")
    print("=" * 92)
    print(f"{'generator':<44}{'arch':<10}{'n':>4}{'caught':>8}{'missed':>8}"
          f"{'absten':>8}{'slip':>7}{'meanMdl':>9}")
    print("-" * 92)
    for entry in table[:20]:
        print(f"{entry['generator'][:43]:<44}{entry['architecture'][:9]:<10}"
              f"{entry['n']:>4}{entry['caught']:>8}{entry['missed']:>8}"
              f"{entry['abstained']:>8}{entry['slip_rate']:>7.2f}"
              f"{entry['mean_model_score']:>9.3f}")
    if len(table) > 20:
        print(f"  ... {len(table) - 20} more generators")

    print()
    print(f"generators evaluated        : {len(table)}")
    print(f"never caught once           : {len(fully_missed)}")
    print(f"caught every time           : {len(fully_caught)}")

    Path(args.out).write_text(json.dumps({
        "source": path.name,
        "thresholds": {"likely_real": low, "ai_generated": high},
        "worst_by_margin": [
            {k: v for k, v in r.items() if k in
             ("filename", "expected", "decision", "confidence", "margin",
              "direction", "model_name", "architecture", "model_score",
              "frequency_score")}
            for r in worst
        ],
        "per_generator": table,
        "generators_never_caught": [r["generator"] for r in fully_missed],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
