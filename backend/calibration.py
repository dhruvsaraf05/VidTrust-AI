"""Does the confidence number mean anything? And are the weights defensible?

    .venv/Scripts/python calibration.py --self-check
    .venv/Scripts/python calibration.py

Two questions, both answered on the matched (centre-cropped) public condition,
both under a selection/reporting split so no number is scored on the rows that
chose it.

1. CALIBRATION. `confidence` is a weighted average of three incommensurate
   scores. It is reported to the user as a percentage, which invites reading it
   as a probability -- but nothing has ever checked that of the files scoring
   0.80, about 80% are actually generated. This measures that (expected
   calibration error, plus a reliability table), fits a Platt scaling on the
   selection half, and reports the improvement on the held-out half.

2. FITTED FUSION. The weights 0.60/0.25/0.15 were chosen by inspection, and
   "why 0.6?" has never had an answer that points at a measurement. This fits a
   logistic regression over the same per-signal scores and compares it to the
   hand-chosen weights under leave-one-generator-out cross-validation -- i.e.
   every prediction is made by a model that never saw that generator.

   A fitted weight is NOT adopted here. REPORT.md section 4.3 already declines
   a measured threshold band because it is fitted to this set's confounds, and
   the same reasoning applies with more force to the weights. The point is to
   be able to say what fitting buys, and on what evidence.

WHY THERE IS NO SCIKIT-LEARN HERE
---------------------------------
This fits four coefficients. Hand-rolling it is ~40 lines of numpy against a
~70 MB dependency, it keeps "every number comes from a script in this repo"
literally true, and it is explainable in a viva. `--self-check` validates the
fitter against cases with known answers, which is what makes that defensible
rather than reckless.

A NOTE ON WHAT CAN ACTUALLY BE FITTED HERE
-------------------------------------------
The metadata signal is unavailable on all 400 files of this set, so its score
is a constant and its coefficient is unidentifiable. Any "fitted weight" for
provenance would be an artefact of the optimiser, not a measurement. It is
therefore excluded and the result is reported as a two-signal fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
from ablation import fuse, load_rows, stratified_halves
from evaluate import roc_auc

SIGNALS = ("model", "metadata", "frequency")


# ---------------------------------------------------------------------------
# Logistic regression, by hand
# ---------------------------------------------------------------------------
def fit_logistic(features: np.ndarray, labels: np.ndarray, *,
                 l2: float = 1e-4, iterations: int = 200) -> np.ndarray:
    """Fit by IRLS (Newton-Raphson). Returns coefficients with intercept last.

    IRLS rather than gradient descent because with four parameters the Hessian
    is 4x4 -- inverting it is free and converges in a handful of steps, with no
    learning rate to tune and no question of whether it had finished.

    The small L2 term is not regularisation for its own sake: on a separable
    split the likelihood has no finite maximum and the coefficients run away to
    infinity. It keeps the fit finite and the result reproducible.
    """
    design = np.column_stack([features, np.ones(len(features))])
    weights = np.zeros(design.shape[1])

    for _ in range(iterations):
        logits = design @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        # Floor the variance so a saturated point cannot zero out the Hessian.
        variance = np.clip(probabilities * (1 - probabilities), 1e-9, None)

        gradient = design.T @ (labels - probabilities) - l2 * weights
        hessian = -(design.T * variance) @ design - l2 * np.eye(design.shape[1])

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break

        weights = weights - step
        if np.max(np.abs(step)) < 1e-10:
            break

    return weights


def predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    return 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -500, 500)))


# ---------------------------------------------------------------------------
# Calibration measures
# ---------------------------------------------------------------------------
def reliability(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10):
    """Expected calibration error, plus the per-bin table behind it.

    ECE = sum over bins of (bin share) x |empirical rate - mean predicted|.
    A perfectly calibrated score has ECE 0: among files it scores 0.8, exactly
    80% are generated.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, error = [], 0.0

    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = ((probabilities >= low) & (probabilities < high)
                if index < bins - 1
                else (probabilities >= low) & (probabilities <= high))
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": f"{low:.1f}-{high:.1f}", "n": 0,
                         "mean_predicted": None, "empirical": None})
            continue

        mean_predicted = float(probabilities[mask].mean())
        empirical = float(labels[mask].mean())
        error += (count / len(probabilities)) * abs(empirical - mean_predicted)
        rows.append({"bin": f"{low:.1f}-{high:.1f}", "n": count,
                     "mean_predicted": mean_predicted, "empirical": empirical})

    return error, rows


def brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def self_check() -> int:
    """Validate the fitter on cases whose answers are known independently."""
    rng = np.random.default_rng(0)
    failures = []

    # 1. Recover known coefficients from data generated by the model itself.
    truth = np.array([2.5, -1.5, 0.3])
    features = rng.normal(size=(4000, 2))
    probabilities = 1 / (1 + np.exp(-(features @ truth[:2] + truth[2])))
    labels = (rng.uniform(size=4000) < probabilities).astype(float)
    fitted = fit_logistic(features, labels)
    error = np.max(np.abs(fitted - truth))
    print(f"  recover known coefficients   max error {error:.3f}  "
          f"{'ok' if error < 0.2 else 'FAIL'}")
    if error >= 0.2:
        failures.append("coefficient recovery")

    # 2. A perfectly calibrated score must have ECE ~ 0.
    probabilities = rng.uniform(size=20000)
    labels = (rng.uniform(size=20000) < probabilities).astype(float)
    ece, _ = reliability(probabilities, labels)
    print(f"  ECE of a calibrated score    {ece:.4f}  "
          f"{'ok' if ece < 0.02 else 'FAIL'}")
    if ece >= 0.02:
        failures.append("ECE on calibrated input")

    # 3. A deliberately overconfident score must have clearly nonzero ECE.
    skewed = np.clip(probabilities ** 0.4, 0, 1)
    ece_bad, _ = reliability(skewed, labels)
    print(f"  ECE of an overconfident score {ece_bad:.4f}  "
          f"{'ok' if ece_bad > 0.05 else 'FAIL'}")
    if ece_bad <= 0.05:
        failures.append("ECE on miscalibrated input")

    # 4. Platt scaling must repair exactly that miscalibration.
    platt = fit_logistic(skewed.reshape(-1, 1), labels)
    repaired, _ = reliability(predict(platt, skewed.reshape(-1, 1)), labels)
    print(f"  ECE after Platt scaling      {repaired:.4f}  "
          f"{'ok' if repaired < ece_bad / 2 else 'FAIL'}")
    if repaired >= ece_bad / 2:
        failures.append("Platt repair")

    print("\nself-check: " + ("PASSED" if not failures
                              else "FAILED -- " + ", ".join(failures)))
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",
                        default="evaluation_report_public_normalised.csv",
                        help="which condition to calibrate on (default: the "
                             "matched clean baseline)")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out", default="calibration.json")
    args = parser.parse_args()

    if args.self_check:
        print("Validating the fitter against known answers:\n")
        return self_check()

    path = Path(args.report)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    rows = load_rows(path)
    labels_all = np.array([1.0 if r["expected"] == "AI" else 0.0 for r in rows])
    fused_all = np.array([fuse(r, SIGNALS)[0] for r in rows])

    available = {key: sum(1 for r in rows if r[f"{key}_available"])
                 for key in SIGNALS}
    usable = [key for key in SIGNALS if available[key] == len(rows)]

    print("=" * 78)
    print(f"CALIBRATION AND FITTED FUSION -- {path.name}, n={len(rows)}")
    print("=" * 78)
    print(f"signal availability : {available}")
    print(f"fittable signals    : {usable}"
          + ("" if len(usable) == len(SIGNALS)
             else "   (the rest are constant here and cannot be fitted)"))

    payload = {"source": path.name, "n": len(rows),
               "signal_availability": available, "fittable_signals": usable}

    # --- 1. calibration ----------------------------------------------------
    selection, reporting = stratified_halves(rows)
    select_fused = np.array([fuse(r, SIGNALS)[0] for r in selection])
    select_labels = np.array([1.0 if r["expected"] == "AI" else 0.0
                              for r in selection])
    report_fused = np.array([fuse(r, SIGNALS)[0] for r in reporting])
    report_labels = np.array([1.0 if r["expected"] == "AI" else 0.0
                              for r in reporting])

    platt = fit_logistic(select_fused.reshape(-1, 1), select_labels)
    report_calibrated = predict(platt, report_fused.reshape(-1, 1))

    ece_raw, bins_raw = reliability(report_fused, report_labels)
    ece_cal, bins_cal = reliability(report_calibrated, report_labels)

    print(f"\n{'-' * 78}")
    print("1. IS THE CONFIDENCE A PROBABILITY?   (fitted on one half, "
          "reported on the other)")
    print("-" * 78)
    print(f"{'':22}{'raw confidence':>18}{'Platt-scaled':>18}")
    print(f"{'expected cal. error':22}{ece_raw:>18.4f}{ece_cal:>18.4f}")
    print(f"{'Brier score':22}{brier(report_fused, report_labels):>18.4f}"
          f"{brier(report_calibrated, report_labels):>18.4f}")
    print(f"{'ROC-AUC':22}"
          f"{roc_auc(list(report_fused), [int(v) for v in report_labels]):>18.4f}"
          f"{roc_auc(list(report_calibrated), [int(v) for v in report_labels]):>18.4f}"
          "   (unchanged: Platt is monotone)")

    # Reliability over ALL rows, and whether it is even monotone. This is the
    # diagnostic that explains the result above: Platt scaling is a monotone
    # map, so if the empirical rate is NOT monotone in the score, no Platt fit
    # can repair it and a worse ECE is the expected outcome, not a bug.
    ece_full, bins_full = reliability(fused_all, labels_all)
    populated = [b for b in bins_full if b["n"] > 0]
    violations = [(a, b) for a, b in zip(populated, populated[1:])
                  if b["empirical"] < a["empirical"] - 1e-9]

    print(f"\n   reliability over all {len(rows)} rows")
    print(f"   {'bin':<12}{'n':>5}{'mean predicted':>17}"
          f"{'actually generated':>21}")
    previous = None
    for row in populated:
        drop = ("   <-- empirical rate DROPS as the score rises"
                if previous is not None
                and row["empirical"] < previous - 1e-9 else "")
        print(f"   {row['bin']:<12}{row['n']:>5}{row['mean_predicted']:>17.3f}"
              f"{row['empirical']:>21.3f}{drop}")
        previous = row["empirical"]

    if violations:
        print(f"\n   The empirical rate is NOT monotone in the score "
              f"({len(violations)} inversions).")
        print("   Platt scaling is monotone, so it cannot repair this, and the "
              "held-out ECE")
        print("   getting worse is the expected consequence rather than a "
              "failure of the fit.")

    payload["monotonicity"] = {
        "ece_all_rows": ece_full,
        "monotone": not violations,
        "inversions": [{"from": a["bin"], "to": b["bin"],
                        "empirical_from": a["empirical"],
                        "empirical_to": b["empirical"]}
                       for a, b in violations],
        "reliability_all_rows": bins_full,
    }

    payload["calibration"] = {
        "platt_coefficients": {"slope": float(platt[0]),
                               "intercept": float(platt[1])},
        "selection_n": len(selection), "reporting_n": len(reporting),
        "held_out": {
            "ece_raw": ece_raw, "ece_calibrated": ece_cal,
            "brier_raw": brier(report_fused, report_labels),
            "brier_calibrated": brier(report_calibrated, report_labels),
            "reliability_raw": bins_raw,
            "reliability_calibrated": bins_cal,
        },
    }

    # --- 2. fitted fusion, leave-one-generator-out -------------------------
    print(f"\n{'-' * 78}")
    print("2. WHY 0.6?   (leave-one-generator-out: every prediction comes "
          "from a fit\n   that never saw that generator)")
    print("-" * 78)

    generators = sorted({r.get("model_name", "") for r in rows
                         if r["expected"] == "AI" and r.get("model_name")})
    features_all = np.column_stack([[r[f"{key}_score"] for r in rows]
                                    for key in usable])

    if len(generators) < 2:
        print("   Not enough generator labels in this report to run LOGO.")
        oof = None
    else:
        # Generated images go to their generator's fold. Real images carry no
        # generator, so they are dealt round-robin over the same folds -- which
        # keeps every fold's test set two-class, so a fold-level AUC exists.
        fold_of = {}
        for index, row in enumerate(rows):
            if row["expected"] == "AI" and row.get("model_name") in generators:
                fold_of[index] = generators.index(row["model_name"])
        real_indices = [i for i, r in enumerate(rows) if r["expected"] == "REAL"]
        for position, index in enumerate(real_indices):
            fold_of[index] = position % len(generators)

        oof = np.zeros(len(rows))
        for fold in range(len(generators)):
            test = [i for i in range(len(rows)) if fold_of.get(i) == fold]
            train = [i for i in range(len(rows)) if fold_of.get(i) != fold]
            if not test or not train:
                continue
            weights = fit_logistic(features_all[train], labels_all[train])
            oof[test] = predict(weights, features_all[test])

        auc_fitted = roc_auc(list(oof), [int(v) for v in labels_all])
        auc_hand = roc_auc(list(fused_all), [int(v) for v in labels_all])

        full = fit_logistic(features_all, labels_all)
        # Convert the fitted coefficients to comparable positive weights so
        # they can be read against 0.60/0.25/0.15 at all.
        magnitudes = np.abs(full[:-1])
        implied = (magnitudes / magnitudes.sum()) if magnitudes.sum() else magnitudes

        print(f"   folds                      {len(generators)} generators")
        print(f"   hand-chosen weights, AUC   {auc_hand:.4f}   "
              f"({', '.join(f'{k} {config.WEIGHTS[k]}' for k in usable)})")
        print(f"   fitted, out-of-fold AUC    {auc_fitted:.4f}")
        print(f"   difference                 {auc_fitted - auc_hand:+.4f}")
        print(f"\n   coefficients fitted on all {len(rows)} rows "
              f"(for interpretation only, never adopted):")
        for key, coefficient, share in zip(usable, full[:-1], implied):
            print(f"     {key:<12} coefficient {coefficient:>8.3f}   "
                  f"implied relative weight {share:.2f}   "
                  f"(hand-chosen {config.WEIGHTS[key]})")
        print(f"     {'intercept':<12} {full[-1]:>19.3f}")

        payload["fitted_fusion"] = {
            "folds": len(generators),
            "auc_hand_chosen": auc_hand,
            "auc_fitted_out_of_fold": auc_fitted,
            "delta": auc_fitted - auc_hand,
            "coefficients_full_fit": {k: float(c)
                                      for k, c in zip(usable, full[:-1])},
            "intercept_full_fit": float(full[-1]),
            "implied_relative_weights": {k: float(s)
                                         for k, s in zip(usable, implied)},
            "hand_chosen_weights": {k: config.WEIGHTS[k] for k in usable},
            "adopted": False,
            "note": ("Reported, not adopted. Fitted on a set whose real images "
                     "are all FFHQ and whose classes differ in native "
                     "resolution; see REPORT.md section 5."),
        }

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("\nNeither the Platt scaling nor the fitted weights are adopted in "
          "config.py.\nThey are measurements of what fitting would buy, on a "
          "set with known confounds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
