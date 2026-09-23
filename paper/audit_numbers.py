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
for tag, cond in (("naive", naive), ("norm", norm)):
    for name in ("full ensemble", "model only", "frequency only"):
        claim(f"{tag} {name} auc", cond[name]["auc"])
        claim(f"{tag} {name} acc", cond[name]["accuracy"])
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
for tag, key in (("naive", "naive"), ("norm", "normalised")):
    t = d7[key]
    claim(f"{tag} cut", t["single_cut"]["threshold"], "{:.3f}")
    claim(f"{tag} cut sel", t["single_cut"]["accuracy_on_selection"])
    claim(f"{tag} cut held", t["single_cut"]["accuracy_on_heldout"])
    claim(f"{tag} band low", t["band"]["low"], "{:.2f}")
    claim(f"{tag} band high", t["band"]["high"], "{:.2f}")
    claim(f"{tag} band sel", t["band"]["accuracy_on_selection"])
    claim(f"{tag} band held", t["band"]["accuracy_on_heldout"])
    claim(f"{tag} current held", t["current_band_on_heldout"]["accuracy"])

# --- derived claims made in the prose -------------------------------------
claim("AUC margin, ensemble over classifier, naive",
      naive["full ensemble"]["auc"] - naive["model only"]["auc"], "{:.3f}")
claim("AUC margin, ensemble over classifier, normalised",
      norm["full ensemble"]["auc"] - norm["model only"]["auc"], "{:.3f}")

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
claim("generators evaluated", len(fa["per_generator"]), "{:d}")
claim("never caught", len(fa["generators_never_caught"]), "{:d}")
claim("caught every time", sum(1 for g in fa["per_generator"] if g["slip_rate"] == 0), "{:d}")
worst = [w["direction"] for w in fa["worst_by_margin"]]
claim("worst errors that are misses", worst.count("AI called REAL"), "{:d}")
slipped = [g for g in fa["per_generator"] if g["slip_rate"] >= 0.5]
claim("generators slipping half", len(slipped), "{:d}")
claim("worst slip mean score, min", min(g["mean_model_score"] for g in slipped), "{:.3f}")
claim("worst slip mean score, max", max(g["mean_model_score"] for g in slipped), "{:.3f}")

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

# the live config must still match what the paper describes
sys.path.insert(0, str(BACKEND))
import config  # noqa: E402
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
