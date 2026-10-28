"""The Reliability Card: how far to trust THIS verdict, from measurement.

A detection score means something only inside a measured operating envelope.
This reads the input's condition cheaply and with no model -- container format,
dimensions, estimated JPEG quality, whether EXIF survived, whether it came
through a platform -- maps it to the nearest condition the robustness study
measured, and reports what the system achieved under that condition next to
what it achieved on clean originals.

WHAT IT LEADS WITH, AND WHY
---------------------------
Detections at the deployed 0.35/0.65 band, not AUC. AUC is threshold-free and
barely moves under recompression (0.949 clean, 0.931 at q75), while the share
of generated images the deployed system actually flags falls from 172/200 to
129/200. A card that led with AUC would report a robustness the user does not
get.

WHAT IT DOES NOT CLAIM
----------------------
* The figures are Track A figures: 400 public still images, centre-cropped to
  remove the resolution confound, with the three dataset confounds of
  REPORT.md section 5 still present. They are an UPPER BOUND, and the card
  says so.
* "Nearest measured condition" is a lookup, not an equivalence. An estimated
  JPEG quality of 72 is reported against the q75 run; a platform re-encode is
  reported against the synthetic "messaging" pipeline, which approximates it.
* Video is not covered by the robustness study at all, which measured stills.
  For video the card says that, and quotes the still-image figure for the
  frame condition only as the closest thing that was measured.
* Provenance was unavailable on all 400 Track A files, so every figure here
  is for the two-signal (classifier + frequency) system.

No model, no fitting, no network: cost is one file open.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from PIL import Image

import config
import perturbations

logger = logging.getLogger(__name__)

ROBUSTNESS = Path(__file__).resolve().parent / "robustness.json"

# IJG standard luminance quantisation table (ITU-T T.81 Annex K), the table
# every libjpeg-family encoder scales by quality.
_IJG_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)

# The JPEG qualities the robustness study ran.
_MEASURED_JPEG = {95: "jpeg95", 75: "jpeg75", 50: "jpeg50", 30: "jpeg30"}


@lru_cache(maxsize=1)
def _study() -> dict | None:
    try:
        return json.loads(ROBUSTNESS.read_text(encoding="utf-8"))["arms"]["matched"]
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Reliability card unavailable: %s", exc)
        return None


def estimate_jpeg_quality(image: Image.Image) -> int | None:
    """IJG quality estimate from the luminance table, or None if not JPEG.

    Uses the ratio of table SUMS, which does not depend on whether the library
    hands the table over in natural or zigzag order. Encoders that do not
    scale the IJG table (some phone ISPs, Photoshop) give an approximate
    figure, which is all a nearest-condition lookup needs.

    Measured against libjpeg: within 2 of the true quality from 30 to 95.
    Below about 15, table entries clip at 255 and the estimate reads high
    (q10 reads 15) -- far below every measured condition either way.
    """
    tables = getattr(image, "quantization", None)
    if not tables or 0 not in tables:
        return None
    luma = list(tables[0])
    if len(luma) != 64:
        return None
    scale = 100.0 * sum(luma) / sum(_IJG_LUMA)
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return int(round(min(100.0, max(1.0, quality))))


def observe(path: str, media_type: str, source: dict | None = None) -> dict:
    """What can be read off the file without analysing it."""
    observed = {"media_type": media_type, "via": "url" if source else "upload",
                "platform": (source or {}).get("platform"),
                "format": None, "width": None, "height": None,
                "jpeg_quality": None, "exif": None}
    if media_type != "image":
        return observed
    with Image.open(path) as image:
        observed["format"] = image.format
        observed["width"], observed["height"] = image.size
        observed["jpeg_quality"] = estimate_jpeg_quality(image)
        observed["exif"] = bool(image.getexif())
    return observed


def nearest_condition(observed: dict) -> tuple[str, str]:
    """(condition name, why it was chosen)."""
    if observed["via"] == "url":
        return "messaging", (
            f"fetched from {observed['platform'] or 'a platform'}, which "
            "re-encodes on upload; the nearest measured condition is the "
            "synthetic messaging-app pipeline")
    if observed["media_type"] != "image":
        return "messaging", (
            "video frames are decoded from a lossy codec; the robustness study "
            "measured stills only, so the nearest measured condition is the "
            "messaging-app pipeline")
    quality = observed["jpeg_quality"]
    if quality is None:
        return "clean", f"{observed['format'] or 'lossless'} file, no JPEG recompression detected"
    nearest = min(_MEASURED_JPEG, key=lambda q: abs(q - quality))
    return _MEASURED_JPEG[nearest], (
        f"estimated JPEG quality {quality}, nearest the measured q{nearest}")


def _figures(entry: dict) -> dict:
    band = entry["at_frozen_band"]
    confusion = band["confusion"]
    generated = entry["classifier_mean"]["generated_total"]
    real = entry["n"] - generated
    return {
        "generated_flagged": confusion["tp"],
        "generated_total": generated,
        "real_flagged": confusion["fp"],
        "real_total": real,
        "abstained": band["abstained"],
        "auc_ensemble": round(entry["auc"]["ensemble"], 4),
    }


def card(path: str, media_type: str, source: dict | None = None) -> dict:
    """The Reliability Card for one input. Never raises."""
    study = _study()
    if study is None:
        return {"available": False, "detail": "robustness.json not found; run robustness.py"}
    try:
        observed = observe(path, media_type, source)
    except Exception as exc:
        logger.warning("Could not read the input's condition: %s", exc)
        return {"available": False,
                "detail": f"Could not read the input's condition ({type(exc).__name__})"}

    condition, reason = nearest_condition(observed)
    measured, clean = _figures(study[condition]), _figures(study["clean"])

    caveats = [
        "Measured on 400 public still images (Track A), centre-cropped; an upper "
        "bound, since that set's dataset confounds remain.",
        "Provenance was unavailable throughout the study, so these are "
        "two-signal (classifier + frequency) figures.",
        f"Verdict thresholds {config.THRESHOLD_LIKELY_REAL}/"
        f"{config.THRESHOLD_AI_GENERATED} are hand-chosen and unfitted.",
    ]
    if media_type != "image":
        caveats.insert(0, "Video is not covered by the robustness study; this is "
                          "the closest still-image measurement.")

    label = "clean originals" if condition == "clean" else perturbations.describe(condition)
    detail = (
        f"Nearest measured condition: {label} ({reason}). There the system "
        f"flagged {measured['generated_flagged']} of {measured['generated_total']} "
        f"generated images and {measured['real_flagged']} of "
        f"{measured['real_total']} real ones"
    )
    if condition != "clean":
        detail += (f", against {clean['generated_flagged']} and "
                   f"{clean['real_flagged']} on clean originals")
    detail += "."

    return {
        "available": True,
        "observed": observed,
        "condition": condition,
        "condition_label": label,
        "reason": reason,
        "measured": measured,
        "clean": clean,
        "caveats": caveats,
        "detail": detail,
    }
