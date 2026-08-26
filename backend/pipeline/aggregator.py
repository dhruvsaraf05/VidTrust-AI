"""Signal fusion and the per-media-type orchestration.

Core property this module protects: a signal that could not run is reported as
unavailable and removed from the weighted average, with the remaining weights
renormalised. A missing signal must never be able to drag the confidence
toward "real" -- that failure mode would make the system quietly unsafe.
"""

from __future__ import annotations

import logging

from PIL import Image

import config
from detectors import frequency_detector, metadata_detector, model_detector
from pipeline.video_sampler import sample_frames

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def build_signal(key: str, result: dict) -> dict:
    """Attach the display name and configured weight to a detector result."""
    return {
        "name": config.SIGNAL_DISPLAY_NAMES[key],
        "score": float(result["score"]) if result["available"] else 0.0,
        "weight": config.WEIGHTS[key],
        "detail": result["detail"],
        "available": bool(result["available"]),
    }


def fuse(signals: dict) -> float:
    """Weighted average over available signals only, weights renormalised.

    With every signal present this is the plain weighted average. With, say,
    metadata unavailable, the remaining 0.60/0.15 are rescaled to 0.80/0.20 --
    so the model and frequency signals decide the verdict between themselves
    instead of being diluted by a zero.

    If nothing is available at all we return 0.5: maximum ignorance, which
    lands inside the UNCERTAIN band rather than falsely reading as real.
    """
    total_weight = sum(s["weight"] for s in signals.values() if s["available"])

    if total_weight <= 0.0:
        logger.warning("No signals available; returning neutral confidence")
        return 0.5

    weighted = sum(
        s["score"] * s["weight"] for s in signals.values() if s["available"]
    )
    return weighted / total_weight


def verdict_for(confidence: float) -> str:
    if confidence >= config.THRESHOLD_AI_GENERATED:
        return "AI_GENERATED"
    if confidence <= config.THRESHOLD_LIKELY_REAL:
        return "LIKELY_REAL"
    return "UNCERTAIN"


# ---------------------------------------------------------------------------
# Image path
# ---------------------------------------------------------------------------
def analyze_image(path: str) -> dict:
    with Image.open(path) as img:
        img = img.convert("RGB")

        model_result = model_detector.analyze(img)
        frequency_result = frequency_detector.analyze(img)

    metadata_result = metadata_detector.analyze(path, "image")

    signals = {
        "model": build_signal("model", model_result),
        "metadata": build_signal("metadata", metadata_result),
        "frequency": build_signal("frequency", frequency_result),
    }

    confidence = fuse(signals)
    return {
        "media_type": "image",
        "verdict": verdict_for(confidence),
        "confidence": round(confidence, 4),
        "signals": signals,
        "frames": None,
    }


# ---------------------------------------------------------------------------
# Video path
# ---------------------------------------------------------------------------
def analyze_video(path: str) -> dict:
    """Sample frames, score each, aggregate to a single video-level verdict.

    Cost split, per the design:
      * model     -> every sampled frame (batched)
      * metadata  -> once, on the container
      * frequency -> every FREQUENCY_FRAME_STRIDE-th sampled frame
    """
    frames, info = sample_frames(path)
    images = [f["image"] for f in frames]

    # --- model signal, per frame -------------------------------------------
    frame_scores: list[float | None] = []
    if model_detector.is_available():
        try:
            frame_scores = model_detector.score_images(images)
        except Exception as exc:
            logger.exception("Per-frame classification failed")
            frame_scores = []
            model_result = {
                "score": 0.0,
                "available": False,
                "detail": f"Per-frame inference failed: {exc}",
            }

    usable = [s for s in frame_scores if s is not None]

    if usable:
        mean_score = sum(usable) / len(usable)
        flagged = sum(1 for s in usable if s > 0.5)
        fraction = flagged / len(usable)
        model_result = {
            "score": round(mean_score, 4),
            "available": True,
            "detail": (
                f"{model_detector.model_name()} | mean over {len(usable)} frames"
                f" | {flagged}/{len(usable)} frames ({fraction:.0%}) scored > 0.5"
            ),
        }
    elif not frame_scores:
        model_result = {
            "score": 0.0,
            "available": False,
            "detail": f"Classifier unavailable ({model_detector.load_error() or 'not loaded'})",
        }
    else:
        model_result = {
            "score": 0.0,
            "available": False,
            "detail": f"Unrecognised label vocabulary from {model_detector.model_name()}",
        }

    # --- frequency signal, on a subset --------------------------------------
    subset = images[:: config.FREQUENCY_FRAME_STRIDE]
    freq_scores: list[float] = []
    for image in subset:
        result = frequency_detector.analyze(image)
        if result["available"]:
            freq_scores.append(result["score"])

    if freq_scores:
        mean_freq = sum(freq_scores) / len(freq_scores)
        frequency_result = {
            "score": round(mean_freq, 4),
            "available": True,
            "detail": (
                f"Mean high-frequency score over {len(freq_scores)} of "
                f"{len(images)} sampled frames (provisional scale)"
            ),
        }
    else:
        frequency_result = {
            "score": 0.0,
            "available": False,
            "detail": "Frequency analysis failed on every sampled frame",
        }

    # --- metadata signal, once on the container -----------------------------
    metadata_result = metadata_detector.analyze(path, "video")

    signals = {
        "model": build_signal("model", model_result),
        "metadata": build_signal("metadata", metadata_result),
        "frequency": build_signal("frequency", frequency_result),
    }

    # Note the sampling coverage in the response so a truncated long video is
    # visible rather than silently partial.
    if info["truncated"]:
        signals["model"]["detail"] += (
            f" | capped at {config.VIDEO_MAX_FRAMES} frames"
            f" (first {info['covered_s']:.0f}s of {info['duration_s']:.0f}s)"
        )

    confidence = fuse(signals)

    # frames[].score is the CLASSIFIER score for that frame. When the model is
    # unavailable there is no per-frame number to report, so the scores are
    # 0.0 and the model signal is flagged available: false.
    frame_payload = [
        {
            "index": f["index"],
            "timestamp": f["timestamp"],
            "score": round(float(s), 4) if s is not None else 0.0,
        }
        for f, s in zip(frames, frame_scores or [None] * len(frames))
    ]

    return {
        "media_type": "video",
        "verdict": verdict_for(confidence),
        "confidence": round(confidence, 4),
        "signals": signals,
        "frames": frame_payload,
    }
