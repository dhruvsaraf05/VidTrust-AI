"""Pretrained-classifier signal.

Wraps a HuggingFace image-classification pipeline and converts its label
distribution into a single "probability this image is machine-generated".

The model is loaded exactly once, at application startup, and cached in module
state. Nothing here loads a model per request. Nothing here trains or
fine-tunes anything -- this is inference against published weights only.

If neither the primary nor the fallback checkpoint loads, the module stays in
an unavailable state and the endpoint still answers using the other signals.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import config

logger = logging.getLogger(__name__)

# --- module-level cache -----------------------------------------------------
_pipe: Any = None
_model_id: str | None = None
_load_error: str | None = None
_load_attempted: bool = False


def load_model() -> bool:
    """Load the classifier once. Returns True on success.

    Called from the FastAPI startup hook. Safe to call again; it will not
    reload an already-loaded pipeline.
    """
    global _pipe, _model_id, _load_error, _load_attempted

    if _pipe is not None:
        return True
    if _load_attempted:
        # Already failed once -- do not re-attempt a slow network load on the
        # request path.
        return False

    _load_attempted = True

    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except Exception as exc:  # pragma: no cover - environment problem
        _load_error = f"torch/transformers import failed: {exc}"
        logger.error(_load_error)
        return False

    # CPU inference is the target. Cap threads so a demo laptop stays usable.
    try:
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4) // 2))
    except Exception:
        pass

    for candidate in (config.MODEL_PRIMARY, config.MODEL_FALLBACK):
        try:
            logger.info("Loading image classifier: %s", candidate)
            _pipe = hf_pipeline(
                "image-classification",
                model=candidate,
                device="cpu",
            )
            _model_id = candidate
            _load_error = None
            logger.info("Classifier ready: %s (labels=%s)",
                        candidate, list(_pipe.model.config.id2label.values()))
            return True
        except Exception as exc:
            logger.warning("Could not load %s: %s", candidate, exc)
            _load_error = f"{candidate}: {exc}"

    logger.error("No classifier could be loaded. Model signal is unavailable.")
    _pipe = None
    return False


def is_available() -> bool:
    return _pipe is not None


def model_name() -> str | None:
    return _model_id


def load_error() -> str | None:
    return _load_error


def _ai_probability(predictions: Sequence[dict]) -> float | None:
    """Collapse a label distribution into P(machine-generated).

    Returns None if the head's label vocabulary is not recognised, so the
    caller can report the signal as unavailable rather than inventing a number.
    """
    for pred in predictions:
        if str(pred["label"]).strip().lower() in config.AI_LABELS:
            return float(pred["score"])

    # No explicit "AI" label -- try to invert a "real" label instead.
    for pred in predictions:
        if str(pred["label"]).strip().lower() in config.REAL_LABELS:
            return 1.0 - float(pred["score"])

    return None


def score_images(images: list) -> list[float | None]:
    """Score a batch of PIL images. Returns one value per image.

    A None entry means the classifier ran but its labels could not be
    interpreted.
    """
    if _pipe is None:
        return [None] * len(images)
    if not images:
        return []

    raw = _pipe(images, top_k=None)

    # A single-image call returns a flat list of predictions; a batch returns a
    # list of such lists. Normalise both shapes.
    if raw and isinstance(raw[0], dict):
        raw = [raw]

    return [_ai_probability(preds) for preds in raw]


def analyze(image) -> dict:
    """Score one PIL image. Returns the signal dict used by the aggregator."""
    if _pipe is None:
        return {
            "score": 0.0,
            "available": False,
            "detail": f"Classifier unavailable ({_load_error or 'not loaded'})",
        }

    try:
        score = score_images([image])[0]
    except Exception as exc:
        logger.exception("Classifier inference failed")
        return {
            "score": 0.0,
            "available": False,
            "detail": f"Inference failed: {exc}",
        }

    if score is None:
        return {
            "score": 0.0,
            "available": False,
            "detail": f"Unrecognised label vocabulary from {_model_id}",
        }

    return {
        "score": round(float(score), 4),
        "available": True,
        "detail": _model_id or "unknown",
    }
