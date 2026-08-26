"""Frequency-domain signal (FFT high-frequency energy ratio).

HONEST FRAMING -- read before defending this in a viva:

This is a weak, hand-tuned heuristic, not a detector. It computes the fraction
of an image's spectral energy that lies outside a central low-frequency disc,
then maps that fraction linearly onto 0..1 using constants chosen by
inspection rather than fitted to a labelled dataset.

Why it is weak:
  * JPEG compression, resizing and denoising all move the ratio more than the
    generator does.
  * Image content dominates: a photo of foliage or fabric is high-frequency;
    a photo of a clear sky is not.
  * Modern generators do not share one consistent spectral fingerprint, and
    several deliberately add sensor-like noise.

It is included because it is cheap, fully explainable, and independent of the
neural classifier -- so it can disagree with the model in a visible way. It
carries the smallest weight (0.15) for exactly these reasons. Do not present
it as evidence on its own.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


def high_frequency_ratio(image: Image.Image) -> float:
    """Fraction of total spectral energy outside the central low-freq disc."""
    grey = image.convert("L").resize(
        (config.FFT_IMAGE_SIZE, config.FFT_IMAGE_SIZE), Image.BILINEAR
    )
    arr = np.asarray(grey, dtype=np.float64)

    # Remove the DC offset so a uniformly bright image does not swamp the
    # spectrum with a single enormous centre bin.
    arr -= arr.mean()

    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    energy = np.abs(spectrum) ** 2

    n = config.FFT_IMAGE_SIZE
    centre = n / 2.0
    yy, xx = np.ogrid[:n, :n]
    radius = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2)

    low_freq_mask = radius <= (config.FFT_LOW_FREQ_RADIUS * n)

    total = float(energy.sum())
    if total <= 0.0:
        return 0.0

    high = float(energy[~low_freq_mask].sum())
    return high / total


def _to_score(ratio: float) -> float:
    """Linear map of the raw ratio onto 0..1 using PROVISIONAL constants."""
    lo, hi = config.FFT_RATIO_MIN, config.FFT_RATIO_MAX
    if hi <= lo:  # guard against a bad config edit
        return 0.0
    return float(np.clip((ratio - lo) / (hi - lo), 0.0, 1.0))


def analyze(image: Image.Image) -> dict:
    """Score one PIL image. Returns the signal dict used by the aggregator."""
    try:
        ratio = high_frequency_ratio(image)
    except Exception as exc:
        logger.exception("Frequency analysis failed")
        return {
            "score": 0.0,
            "available": False,
            "detail": f"FFT failed: {exc}",
        }

    score = _to_score(ratio)

    if score >= 0.6:
        wording = "Elevated high-frequency energy"
    elif score <= 0.25:
        wording = "Low high-frequency energy"
    else:
        wording = "High-frequency energy within typical range"

    return {
        "score": round(score, 4),
        "available": True,
        "detail": f"{wording} (ratio {ratio:.3f}, provisional scale)",
    }
