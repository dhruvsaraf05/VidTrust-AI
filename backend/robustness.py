"""The operating envelope: how far each signal degrades, and where it breaks.

    .venv/Scripts/python robustness.py

Reads every `evaluation_report_public_*.csv` the sweep produced and builds the
degradation matrix. No inference is re-run -- each condition's per-signal scores
were already dumped by evaluate.py, exactly as ablation.py works.

WHY ΔAUC IS NOT ENOUGH, AND WHAT TO READ INSTEAD
------------------------------------------------
AUC is threshold-free, so it measures whether the *ranking* survives a
perturbation. It is blind to the thing that actually breaks the product: the
frozen 0.35/0.65 band staying put while the whole score distribution slides
underneath it.

Measured on 24 matched crops, the classifier's mean score on generated images
falls from 0.779 clean to 0.505 at JPEG quality 30 -- below
THRESHOLD_AI_GENERATED (0.65). Every one of those files silently stops being
called AI_GENERATED. AUC drops by 0.25 and never names that cliff.

So every row here carries BOTH: AUC (did the ranking survive) and the verdict
distribution at the frozen band (did the product survive). They answer
different questions and the second one is the one a user feels.

TWO ARMS
--------
`matched` rows had both classes centre-cropped to 512 native pixels first and
are the measurement. `native` rows did not, and in this dataset resolution is
the label (real 1024, generated 512), so they are reported separately and only
as a demonstration that skipping the control inverts a conclusion. Never read a
native row as a robustness result. See perturbations.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import config
from ablation import fuse, load_rows, score_configuration
from evaluate import roc_auc

SIGNALS = ("model", "metadata", "frequency")

# Conditions in the order they belong in a table: baseline, then increasing
# severity within each family.
ORDER = ["clean", "jpeg95", "jpeg75", "jpeg50", "jpeg30",
         "downscale75", "downscale50", "crop256",
         "noise2", "noise5", "noise10", "messaging"]


def discover() -> dict[str, dict[str, Path]]:
    """Find every condition CSV. Returns {arm: {condition: path}}."""
    found: dict[str, dict[str, Path]] = {"matched": {}, "native": {}}

    # The two clean baselines already exist under their legacy names.
    if Path("evaluation_report_public_normalised.csv").exists():
        found["matched"]["clean"] = Path("evaluation_report_public_normalised.csv")
    if Path("evaluation_report_public.csv").exists():
        found["native"]["clean"] = Path("evaluation_report_public.csv")

    for path in sorted(Path(".").glob("evaluation_report_public_*.csv")):
        match = re.fullmatch(r"evaluation_report_public_(crop|native)_(\w+)\.csv",
                             path.name)
        if match:
            arm = "matched" if match.group(1) == "crop" else "native"
            found[arm][match.group(2)] = path
    return found


def signal_auc(rows: list[dict], key: str) -> float | None:
    """AUC of one raw signal, over the files where it actually ran.

    A signal that never ran has no ranking to measure, so this returns None
    rather than 0.5 -- 0.5 would look like a measurement of a useless signal,
    when the truth is that there is no measurement at all. Same distinction the
    fusion rule makes between unavailable and zero.
    """
    usable = [r for r in rows if r[f"{key}_available"]]
    if len(usable) < len(rows) * 0.5:
        return None
    return roc_auc([r[f"{key}_score"] for r in usable],
                   [1 if r["expected"] == "AI" else 0 for r in usable])


def describe_condition(rows: list[dict]) -> dict:
    """Everything worth knowing about one condition."""
    low = config.THRESHOLD_LIKELY_REAL
    high = config.THRESHOLD_AI_GENERATED
    truth = [1 if r["expected"] == "AI" else 0 for r in rows]

    fused = [fuse(r, SIGNALS)[0] for r in rows]
    ensemble = score_configuration(rows, SIGNALS, low, high)

    ai_scores = [r["model_score"] for r in rows if r["expected"] == "AI"]
    real_scores = [r["model_score"] for r in rows if r["expected"] == "REAL"]

    return {
        "n": len(rows),
        "auc": {
            "ensemble": roc_auc(fused, truth),
            **{key: signal_auc(rows, key) for key in SIGNALS},
        },
        # The verdict distribution at the FROZEN band -- the half AUC hides.
        "at_frozen_band": {
            "accuracy": ensemble["accuracy"],
            "precision": ensemble["precision"],
            "recall": ensemble["recall"],
            "decided": ensemble["decided"],
            "abstained": ensemble["abstained"],
            "coverage": ensemble["decided"] / ensemble["n"] if ensemble["n"] else None,
            "confusion": ensemble["confusion"],
        },
        # Where the distribution sits relative to the threshold it must cross.
        "classifier_mean": {
            "generated": sum(ai_scores) / len(ai_scores) if ai_scores else None,
            "real": sum(real_scores) / len(real_scores) if real_scores else None,
            "generated_above_threshold": sum(1 for s in ai_scores if s >= high),
            "generated_total": len(ai_scores),
        },
    }


def cell(value, places: int = 4) -> str:
    return "  --  " if value is None else f"{value:.{places}f}"


def delta(value, baseline) -> str:
    if value is None or baseline is None:
        return "  --  "
    return f"{value - baseline:+.4f}"


def main() -> int:
    found = discover()
    if not found["matched"]:
        print("No matched-arm conditions found. Run run_robustness_sweep.py first.")
        return 1

    payload = {
        "thresholds": {"likely_real": config.THRESHOLD_LIKELY_REAL,
                       "ai_generated": config.THRESHOLD_AI_GENERATED},
        "weights": config.WEIGHTS,
        "note": ("matched = both classes centre-cropped to 512 native pixels "
                 "before the perturbation; native = uncontrolled, retained "
                 "only to demonstrate the resolution confound"),
        "arms": {},
    }

    for arm in ("matched", "native"):
        conditions = found[arm]
        if not conditions:
            continue

        ordered = ([c for c in ORDER if c in conditions]
                   + sorted(set(conditions) - set(ORDER)))
        results = {name: describe_condition(load_rows(conditions[name]))
                   for name in ordered}
        payload["arms"][arm] = results

        baseline = results.get("clean", {}).get("auc", {})

        banner = ("THE MEASUREMENT" if arm == "matched"
                  else "CONFOUND DEMONSTRATION -- not a robustness result")
        print(f"\n{'=' * 92}")
        print(f"{arm.upper()} ARM -- {banner}")
        print("=" * 92)
        print(f"{'condition':<13}{'ens AUC':>9}{'d':>9}{'clf AUC':>9}{'d':>9}"
              f"{'freq AUC':>10}{'d':>9}{'acc':>8}{'recall':>8}{'absten':>8}")
        print("-" * 92)

        for name in ordered:
            r = results[name]
            auc = r["auc"]
            band = r["at_frozen_band"]
            print(
                f"{name:<13}"
                f"{cell(auc['ensemble']):>9}{delta(auc['ensemble'], baseline.get('ensemble')):>9}"
                f"{cell(auc['model']):>9}{delta(auc['model'], baseline.get('model')):>9}"
                f"{cell(auc['frequency']):>10}{delta(auc['frequency'], baseline.get('frequency')):>9}"
                f"{cell(band['accuracy']):>8}{cell(band['recall']):>8}"
                f"{band['abstained']:>8}"
            )

        # The cliff. This is the sentence the report needs.
        print()
        print(f"{'condition':<13}{'mean score, generated':>24}"
              f"{'still >= ' + str(config.THRESHOLD_AI_GENERATED):>18}")
        print("-" * 92)
        for name in ordered:
            m = results[name]["classifier_mean"]
            above = f"{m['generated_above_threshold']}/{m['generated_total']}"
            print(f"{name:<13}{cell(m['generated'], 3):>24}{above:>18}")

    Path("robustness.json").write_text(json.dumps(payload, indent=2),
                                       encoding="utf-8")
    print("\nwrote robustness.json")
    print("\nmetadata is unavailable on all 400 files in every condition, so "
          "this is a two-signal\nrobustness result. The provenance signal "
          "cannot be measured at scale on this set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
