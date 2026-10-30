"""Cross-check every number in the paper against the evaluation outputs.

The paper's tables are typed LaTeX, so they can drift from the scripts that
produced them. This asserts they have not. Run it after any re-evaluation:

    cd paper && python audit_numbers.py

Historic values (the replaced classifier, and the pre-fix provenance counts)
are listed separately: they are quoted in the paper as history and cannot be
regenerated from the current code, so they are checked against the prose only
for presence, not recomputed.
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
TEX = (Path(__file__).resolve().parent / "vidtrust_ieee.tex").read_text(encoding="utf-8")

sys.path.insert(0, str(BACKEND))
import config  # noqa: E402

abl = json.loads((BACKEND / "ablation.json").read_text(encoding="utf-8"))
naive, norm = abl["conditions"]["naive"], abl["conditions"]["normalised"]
d7 = abl["threshold_selection"]
m_naive = json.loads((BACKEND / "evaluation_metrics_public.json").read_text(encoding="utf-8"))
m_norm = json.loads((BACKEND / "evaluation_metrics_public_normalised.json").read_text(encoding="utf-8"))
fa = json.loads((BACKEND / "failure_analysis.json").read_text(encoding="utf-8"))


def rows(name):
    with (BACKEND / name).open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["expected"] in ("AI", "REAL")]


R_NAIVE, R_NORM = rows("evaluation_report_public.csv"), rows("evaluation_report_public_normalised.csv")

failures: list[str] = []
checks = 0


def claim(label, value, fmt="{:.4f}"):
    """Assert that `value`, formatted as the paper prints it, appears in the paper."""
    global checks
    checks += 1
    text = fmt.format(value) if not isinstance(value, str) else value
    if text not in TEX:
        failures.append(f"{label}: computed {text!r}, not found in the paper")


# --- headline -------------------------------------------------------------
# AUC note: evaluate.py ranks the fused confidence as the API rounds it (4 dp),
# which leaves ties; ablation.py re-fuses from the 6 dp per-signal dump and
# breaks them. The two therefore differ by ~2.5e-5 in the naive condition and
# agree exactly in the normalised one. The paper quotes ablation.py throughout
# so one source is used consistently; the assertion below fails if that gap
# ever grows past the fourth decimal, where it would start to matter.
for tag, m, cond in (("naive", m_naive, naive), ("norm", m_norm, norm)):
    d = m["metrics_on_decided"]
    claim(f"{tag} accuracy", d["accuracy"])
    claim(f"{tag} precision", d["precision"])
    claim(f"{tag} recall", d["recall"])
    claim(f"{tag} f1", d["f1"])
    claim(f"{tag} auc", cond["full ensemble"]["auc"])
    checks += 1
    if abs(cond["full ensemble"]["auc"] - m["roc_auc_all_labelled"]) > 1e-4:
        failures.append(
            f"{tag}: evaluate.py AUC {m['roc_auc_all_labelled']} and ablation.py "
            f"AUC {cond['full ensemble']['auc']} now disagree above 1e-4")
    cm = m["confusion_matrix"]
    for k in ("true_positive", "false_positive", "true_negative", "false_negative"):
        claim(f"{tag} {k}", cm[k], "{:d}")
    claim(f"{tag} abstentions", cm["excluded_uncertain"], "{:d}")

# --- ablation -------------------------------------------------------------
# The paper's ablation block reports AUC per configuration, plus accuracy for
# the full ensemble only (it shares the headline rows above). The per-signal
# accuracies were cut for the 6-page limit and are deliberately not asserted.
for tag, cond in (("naive", naive), ("norm", norm)):
    for name in ("full ensemble", "model only", "frequency only"):
        claim(f"{tag} {name} auc", cond[name]["auc"])
    claim(f"{tag} full ensemble acc", cond["full ensemble"]["accuracy"])
    claim(f"{tag} metadata auc", cond["metadata only"]["auc"])
    # the two identity rows must equal the rows they duplicate
    checks += 2
    if cond["ensemble - metadata"]["auc"] != cond["full ensemble"]["auc"]:
        failures.append(f"{tag}: 'ensemble - metadata' is no longer an identity")
    if cond["ensemble - frequency"]["auc"] != cond["model only"]["auc"]:
        failures.append(f"{tag}: 'ensemble - frequency' is no longer an identity")
claim("frequency precision", naive["frequency only"]["precision"])
claim("frequency recall", naive["frequency only"]["recall"])

# --- threshold selection --------------------------------------------------
# The threshold-selection subsection was cut for the six-page limit, so none of
# its numbers appear in the paper any more and none are asserted here. What
# survives is the claim the paper still makes in the introduction and in
# section VI-D: a measured operating point that beats the deployed one exists
# and was not adopted. That is checked against the data rather than the prose,
# because the prose no longer quotes it.
checks += 1
if (d7["normalised"]["single_cut"]["accuracy_on_heldout"]
        <= d7["normalised"]["current_band_on_heldout"]["accuracy"]):
    failures.append("the searched single cut no longer beats the deployed band "
                    "on held-out data, so the introduction's 'a fitted operating "
                    "point, measurably better, that we did not adopt' is stale")

# --- derived claims made in the prose -------------------------------------
over_half = {tag: sum(1 for r in rs if r["expected"] == "REAL" and float(r["model_score"]) >= 0.5)
             for tag, rs in (("naive", R_NAIVE), ("norm", R_NORM))}
claim("real images over 0.5, naive", f"{over_half['naive']}/200")
claim("real images over 0.5, normalised", f"{over_half['norm']}/200")

fp_drop = m_naive["confusion_matrix"]["false_positive"] - m_norm["confusion_matrix"]["false_positive"]
claim("false-positive reduction", fp_drop, "{:d}")

# recall must be identical across conditions, which the paper asserts
checks += 1
if m_naive["metrics_on_decided"]["recall"] != m_norm["metrics_on_decided"]["recall"]:
    failures.append("recall is no longer identical across conditions")

# generated-image scores must be untouched by the crop, which is why recall holds
checks += 1
by_name = {r["filename"]: r for r in R_NORM}
drift = max(abs(float(r["model_score"]) - float(by_name[r["filename"]]["model_score"]))
            for r in R_NAIVE if r["expected"] == "AI")
if drift > 1e-9:
    failures.append(f"generated scores changed under the crop (max delta {drift})")

# --- failure analysis -----------------------------------------------------
# The paper now states the error STRUCTURE rather than the per-generator
# counts: no generator missed on all four images, and a worst slip of 2 of 4.
claim("generators evaluated", len(fa["per_generator"]), "{:d}")
claim("never caught", len(fa["generators_never_caught"]), "{:d}")
checks += 2
if fa["generators_never_caught"]:
    failures.append("a generator is now missed on all four images -- 'misses no "
                    "generator on all four' in section VII is stale")
worst_slip = max(g["slip_rate"] for g in fa["per_generator"])
if abs(worst_slip - 0.5) > 1e-9:
    failures.append(f"worst slip rate is now {worst_slip}, not the 2 of 4 the paper states")

# --- dataset ---------------------------------------------------------------
counts = m_naive["counts"]
claim("files", counts["files"], "{:d}")
claim("metadata availability", counts["signal_available"]["metadata"], "{:d}")
checks += 1
if counts["signal_available"]["metadata"] != 0:
    failures.append("provenance is no longer dead on Track A -- section IV-A is stale")

# --- timing ----------------------------------------------------------------
claim("in-process seconds per image",
      st.mean(int(r["processing_ms"]) for r in R_NAIVE) / 1000, "{:.2f}")

# --- model selection table -------------------------------------------------
for model, correct, gen, real in [
        ("haywoodsloan/ai-image-detector-deploy", "12/12", "0.987", "0.002"),
        ("Ateeqq/ai-vs-human-image-detector", "11/12", "0.752", "0.001"),
        ("Organika/sdxl-detector", "5/12", "0.438", "0.574")]:
    claim(f"{model} correct", correct)
    claim(f"{model} mean generated", gen)
    claim(f"{model} mean real", real)

# --- operating envelope (section VI, Table IV, Fig. 1) ---------------------
rob = json.loads((BACKEND / "robustness.json").read_text(encoding="utf-8"))
MATCHED, NATIVE = rob["arms"]["matched"], rob["arms"]["native"]

# Table IV, row by row. Provenance is omitted from the table because it is
# unavailable in every condition, which is itself asserted below.
for condition in ("clean", "jpeg95", "jpeg75", "jpeg50", "jpeg30",
                  "downscale75", "downscale50", "crop256",
                  "noise2", "noise5", "noise10", "messaging"):
    entry = MATCHED[condition]
    claim(f"{condition} ensemble auc", entry["auc"]["ensemble"])
    claim(f"{condition} classifier auc", entry["auc"]["model"])
    claim(f"{condition} frequency auc", entry["auc"]["frequency"])
    claim(f"{condition} accuracy", entry["at_frozen_band"]["accuracy"])
    claim(f"{condition} detected", entry["at_frozen_band"]["confusion"]["tp"], "{:d}")
    checks += 1
    if entry["auc"]["metadata"] is not None:
        failures.append(f"{condition}: provenance now has an AUC, so Table IV may "
                        f"no longer omit it")

# The headline of section VI-A: AUC barely moves while detections collapse.
clean_auc = MATCHED["clean"]["auc"]["ensemble"]
q30_auc = MATCHED["jpeg30"]["auc"]["ensemble"]
clean_tp = MATCHED["clean"]["at_frozen_band"]["confusion"]["tp"]
q30_tp = MATCHED["jpeg30"]["at_frozen_band"]["confusion"]["tp"]
claim("auc movement clean to q30", abs(clean_auc - q30_auc), "{:.3f}")
claim("detection loss clean to q30",
      f"{round(100 * (clean_tp - q30_tp) / clean_tp):d}\\%")
claim("q30 precision", MATCHED["jpeg30"]["at_frozen_band"]["precision"])
for condition, tag in (("clean", "clean"), ("jpeg30", "q30")):
    means = MATCHED[condition]["classifier_mean"]
    claim(f"{tag} mean score generated", means["generated"], "{:.3f}")
    claim(f"{tag} mean score real", means["real"], "{:.3f}")
checks += 1
if MATCHED["jpeg30"]["classifier_mean"]["generated"] >= config.THRESHOLD_AI_GENERATED:
    failures.append("the q30 mean generated score no longer falls below the "
                    "AI_GENERATED threshold -- section VI-A's mechanism is stale")

# Section VI-A, last sentence: light degradation raises false alarms. This
# replaced "rescaling costs nothing", which was read off the detections column
# alone and ignored the real images flagged.
claim("noise2 mean score real", MATCHED["noise2"]["classifier_mean"]["real"], "{:.3f}")
claim("noise2 real flagged", MATCHED["noise2"]["at_frozen_band"]["confusion"]["fp"], "{:d} of 200")
claim("downscale50 real flagged", MATCHED["downscale50"]["at_frozen_band"]["confusion"]["fp"],
      "flags {:d}")
checks += 1
if not (MATCHED["noise2"]["at_frozen_band"]["confusion"]["fp"]
        > MATCHED["clean"]["at_frozen_band"]["confusion"]["fp"]
        < MATCHED["downscale50"]["at_frozen_band"]["confusion"]["fp"]):
    failures.append("light noise / halving no longer raise false alarms over clean "
                    "-- section VI-A's last sentence is stale")

# Section VI-B: the uncontrolled arm inverts the sign on two of three.
flips = 0
for condition in ("jpeg50", "noise10", "messaging"):
    matched_delta = MATCHED[condition]["auc"]["model"] - MATCHED["clean"]["auc"]["model"]
    native_delta = NATIVE[condition]["auc"]["model"] - NATIVE["clean"]["auc"]["model"]
    claim(f"{condition} matched delta", matched_delta, "{:+.3f}")
    claim(f"{condition} native delta", native_delta, "{:+.3f}")
    if (matched_delta < 0) != (native_delta < 0):
        flips += 1
checks += 1
if flips != 2:
    failures.append(f"{flips} of three conditions now change sign, not the two "
                    f"the abstract, Fig. 1 and the conclusion all state")

# Section VI-C: frequency is flat across the JPEG sweep, which is the argument
# that it is not reading an up-sampling artefact.
jpeg_freq = [MATCHED[c]["auc"]["frequency"] for c in ("jpeg95", "jpeg75", "jpeg50", "jpeg30")]
claim("frequency jpeg min", min(jpeg_freq))
claim("frequency jpeg max", max(jpeg_freq))
claim("frequency jpeg range", max(jpeg_freq) - min(jpeg_freq))

# --- calibration (section VI-D) --------------------------------------------
cal = json.loads((BACKEND / "calibration.json").read_text(encoding="utf-8"))
held = cal["calibration"]["held_out"]
claim("ece raw", held["ece_raw"])
claim("ece calibrated", held["ece_calibrated"])
claim("brier raw", held["brier_raw"])
claim("brier calibrated", held["brier_calibrated"])
checks += 2
if held["ece_calibrated"] <= held["ece_raw"]:
    failures.append("Platt scaling now improves ECE -- section VI-D is stale")
if cal["monotonicity"]["monotone"]:
    failures.append("the empirical rate is now monotone -- section VI-D is stale")
for name, label in (("0.7-0.8", "upper"), ("0.1-0.2", "lower")):
    binned = next(b for b in cal["monotonicity"]["reliability_all_rows"] if b["bin"] == name)
    claim(f"{label} bin n", binned["n"], "{:d}")
    claim(f"{label} bin rate", 100 * binned["empirical"], "{:.1f}\\%")

fitted = cal["fitted_fusion"]
claim("fitted fusion auc", fitted["auc_fitted_out_of_fold"])
claim("fitted fusion folds", fitted["folds"], "{:d}")
claim("fitted weight model", fitted["implied_relative_weights"]["model"], "{:.2f}")
claim("fitted weight frequency", fitted["implied_relative_weights"]["frequency"], "{:.2f}")
checks += 1
if fitted["adopted"]:
    failures.append("the fitted fusion is now adopted, which contradicts the paper")

# --- Section IX: Grad-CAM faithfulness ---------------------------------------
faith = json.loads((BACKEND / "saliency_faithfulness.json").read_text(encoding="utf-8"))["overall"]
claim("faithfulness images", faith["n"], "occlusion on {:d} images")
claim("faithfulness rho", faith["rho_median"], "\\rho = {:.2f}")
checks += 1
if faith["rho_median"] >= 0.3:
    failures.append("Grad-CAM now passes the 0.3 faithfulness floor -- section IX "
                    "says the system draws no saliency map")

# the live config must still match what the paper describes
claim("weight model", config.WEIGHTS["model"], "{:.2f}")
claim("weight metadata", config.WEIGHTS["metadata"], "{:.2f}")
claim("weight frequency", config.WEIGHTS["frequency"], "{:.2f}")
claim("threshold high", config.THRESHOLD_AI_GENERATED, "{:.2f}")
claim("threshold low", config.THRESHOLD_LIKELY_REAL, "{:.2f}")
checks += 1
if m_naive["model"]["name"] not in ("haywoodsloan/ai-image-detector-deploy",):
    failures.append(f"results were produced by {m_naive['model']['name']}, not the model the paper names")

print(f"{checks} checks")
if failures:
    print("\nFAILED:")
    for f in failures:
        print("  -", f)
    raise SystemExit(1)
print("all numbers in the paper match the evaluation outputs")
