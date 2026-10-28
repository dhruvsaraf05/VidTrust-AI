"""Where did the classifier look? Grad-CAM over the SwinV2 backbone.

HONEST FRAMING -- read this before putting a heatmap in front of anyone
-----------------------------------------------------------------------
A saliency overlay is the most persuasive thing this project can draw and the
easiest to overclaim with. So it was measured before being shown.

saliency_faithfulness.py compares this module's stage-2 Grad-CAM against
occlusion (blank each 32x32 cell, record the drop in the 'artificial' logit)
on 20 evaluation images, 10 per class:

    median Spearman rho     -0.02   (range -0.47 to +0.76)
    rho > 0                  9 of 20
    flatness gate would draw 16 of 20, of which 8 have rho > 0

**The map does not locate what the model responds to.** It is no better than
chance, and the flatness gate below does not help: it rejects flat maps, not
wrong ones, and passes half-and-half.

An earlier note here, from a single image, said the evidence was "spatially
diffuse". Twenty images do not support that: on 3 of 10 generated images one
cell moves the logit by 1.5 to 3.7. Whether that is evidence or the model
reacting to an unnatural grey square is not something occlusion can settle.
What IS settled is that Grad-CAM does not find those cells.

So explain() runs a FAITHFULNESS GATE first: if the measured median rho is
below MIN_FAITHFUL_RHO, or the study has not been run, it declines to draw
and says why, without spending the gradient pass. It is the same rule the
fusion obeys -- an unavailable signal is declared, never faked. The code
below the gate is kept because saliency_faithfulness.py validates it, and a
different checkpoint or stage may pass.

WHY STAGE 2
-----------
Measured hidden-state resolutions for a 256x256 input (the processor's size):

    hidden_states[0]   64x64    embeddings
    hidden_states[1]   32x32
    hidden_states[2]   16x16    <- this module
    hidden_states[3]    8x8
    hidden_states[4]    8x8     final

The last stage is 8x8: each cell covers 32x32 input pixels, so a "blob" is an
eighth of the frame. Stage 2 is 16x16, four times the resolution, at no extra
cost. It is also the last point before locality is destroyed: this checkpoint
clamps its window to the input resolution, so stages 3 and 4 run PLAIN GLOBAL
self-attention (18 of the 24 blocks), after which every position has already
attended to every other.

Targeting stage 3 was measured to produce a blank map -- 0 of 64 cells nonzero
after the ReLU -- which is the gate firing for a different reason.

TWO IMPLEMENTATION TRAPS, both measured
---------------------------------------
1. `output.reshaped_hidden_states` CANNOT be the Grad-CAM target. Those are
   .view().permute() branches off the graph, not ancestors of the logits, and
   autograd.grad raises "appears to not have been used in the graph". Use
   `output.hidden_states[k]`, shape (1, L, C), and reshape the gradient here.
2. Use the RAW LOGIT, not the softmax probability. The two give rank-identical
   maps, but the probability map spans ~5e-4 on a confident image, which
   renders as a uniform grey rectangle.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

import config
from detectors import model_detector
from detectors.redact import redact

logger = logging.getLogger(__name__)

# Which hidden state to attribute against. See WHY STAGE 2 above.
CAM_STAGE = 2

# If the normalised CAM's p90-p10 span falls below this, the map carries
# no usable structure and we decline to draw it. PROVISIONAL: chosen by
# inspection of a handful of images, not fitted. Measured to be a weak
# filter (see the docstring); the faithfulness gate is the one that matters.
MIN_SPAN = 0.15

# Below this measured median agreement with occlusion, no map is drawn.
# PROVISIONAL: a judgement of "clearly better than chance", not fitted. It is
# not tuned to the study's result (-0.02 fails any positive floor), and it
# must not be lowered to make a map appear.
MIN_FAITHFUL_RHO = 0.3

# Single-hue sequential ramp, light -> dark, with alpha carrying intensity so
# the unattributed majority of the frame stays transparent. One hue because
# this encodes magnitude, not category.
RAMP_RGB = (176, 42, 42)


FAITHFULNESS = Path(__file__).resolve().parent.parent / "saliency_faithfulness.json"


@lru_cache(maxsize=1)
def faithfulness() -> dict | None:
    """The measured agreement with occlusion, attached to every response.

    The heatmap must never travel without this number. None when the study
    has not been run on this machine, which the interface states as "not
    measured" rather than leaving out.
    """
    try:
        report = json.loads(FAITHFULNESS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    overall = report.get("overall", {})
    return {"n": overall.get("n"), "rho_median": overall.get("rho_median"),
            "rho_positive": overall.get("rho_positive"),
            "gate_draws": overall.get("gate_draws"),
            "gate_draws_rho_positive": overall.get("gate_draws_rho_positive"),
            "min_faithful_rho": MIN_FAITHFUL_RHO}


def _unavailable(detail: str) -> dict:
    return {"available": False, "detail": detail, "heatmap_png_b64": None,
            "grid": None, "target_logit": None, "span": None}


def _colourise(grid: np.ndarray, size: tuple[int, int]) -> str:
    """Render the CAM as an RGBA PNG at the original image's dimensions.

    The processor squashes any input to a 256x256 square rather than preserving
    aspect ratio, so the grid corresponds to a squashed view. Resizing the grid
    straight to (width, height) is exactly the inverse of that squash, which is
    why no letterboxing correction is needed here.
    """
    normalised = np.clip(grid, 0.0, 1.0)
    heat = Image.fromarray((normalised * 255).astype(np.uint8), mode="L")
    heat = heat.resize(size, Image.BICUBIC)

    alpha = np.asarray(heat, dtype=np.float64) / 255.0
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    for channel, value in enumerate(RAMP_RGB):
        rgba[..., channel] = value
    # Cap the alpha so the underlying image stays visible through the hottest
    # region -- an overlay nobody can see through invites reading the heat as
    # the evidence itself.
    rgba[..., 3] = (np.clip(alpha, 0, 1) * 200).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class CamError(ValueError):
    """The model's structure does not support this CAM; the message says why."""


def cam_grid(model, pixel_values, ai_index: int, stage: int = CAM_STAGE):
    """Raw (unnormalised, un-gated) Grad-CAM grid and the target logit.

    Split out of explain() so saliency_faithfulness.py validates THIS code,
    not a reimplementation of it. Raises CamError on a structural mismatch.
    """
    import torch

    # Grad-CAM needs gradients, so this forward pass is deliberately NOT
    # under no_grad: activations have to be retained.
    output = model(pixel_values=pixel_values, output_hidden_states=True)
    states = output.hidden_states
    if states is None or len(states) <= stage:
        raise CamError("Model did not expose the expected hidden states")

    activations = states[stage]                    # (1, L, C)
    target = output.logits[0, ai_index]

    gradients = torch.autograd.grad(target, activations, retain_graph=False)[0]

    # Grad-CAM: channel weights are the spatially averaged gradients, then
    # a ReLU over the weighted sum keeps only evidence FOR the class.
    channel_weights = gradients.mean(dim=1, keepdim=True)      # (1, 1, C)
    cam = torch.relu((channel_weights * activations).sum(dim=-1))[0]  # (L,)

    length = cam.shape[0]
    side = int(math.isqrt(length))
    if side * side != length:
        raise CamError(f"Hidden state at stage {stage} is not square ({length} positions)")

    return cam.detach().numpy().reshape(side, side), float(target.detach())


def gate_span(normalised: np.ndarray) -> float:
    """The statistic MIN_SPAN gates on: p90 - p10 of the max-normalised map."""
    return float(np.percentile(normalised, 90) - np.percentile(normalised, 10))


def explain(image: Image.Image) -> dict:
    """Grad-CAM for the 'machine-generated' logit. Never raises."""
    measured = faithfulness()
    if measured is None or measured.get("rho_median") is None:
        return _unavailable(
            "Not drawn: this heatmap's agreement with occlusion has not been "
            "measured on this machine (run saliency_faithfulness.py)."
        )
    if measured["rho_median"] < MIN_FAITHFUL_RHO:
        return _unavailable(
            f"Not drawn. Measured on {measured['n']} evaluation images, this "
            f"heatmap agrees with where the classifier is actually sensitive "
            f"at a median Spearman rho of {measured['rho_median']:.2f}, which "
            "is no better than chance. A map would look like an explanation "
            "without being one."
        )

    pipeline = model_detector.pipe()
    if pipeline is None:
        return _unavailable(
            f"Classifier unavailable ({model_detector.load_error() or 'not loaded'})"
        )

    ai_index = model_detector.ai_label_index()
    if ai_index is None:
        return _unavailable(
            f"Unrecognised label vocabulary from {model_detector.model_name()}; "
            "cannot tell which class means machine-generated"
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment problem
        return _unavailable(f"torch unavailable: {redact(exc)}")

    try:
        model = pipeline.model
        model.eval()
        rgb = image.convert("RGB")
        pixel_values = pipeline.image_processor(
            images=rgb, return_tensors="pt")["pixel_values"]
        grid, logit = cam_grid(model, pixel_values, ai_index)
        side = grid.shape[0]
    except CamError as exc:
        return _unavailable(str(exc))
    except Exception as exc:
        logger.exception("Saliency failed")
        return _unavailable(f"Saliency computation failed: {type(exc).__name__}")

    ceiling = float(grid.max())
    if ceiling <= 0.0:
        return _unavailable(
            f"Grad-CAM at stage {CAM_STAGE} is empty for this image: no region "
            "contributes positively to the machine-generated logit"
        )

    normalised = grid / ceiling
    # p90-p10 span rather than max-min: a single hot cell against an
    # otherwise flat map is noise, and max-min would not notice.
    span = gate_span(normalised)

    if span < MIN_SPAN:
        return _unavailable(
            f"The attribution map is nearly flat (span {span:.3f} < {MIN_SPAN}), "
            "so it marks no region. Reported rather than drawn."
        )

    return {
        "available": True,
        "detail": (
            f"Grad-CAM over {model_detector.model_name()} stage {CAM_STAGE} "
            f"({side}x{side} cells), attributing the "
            f"'{config.SIGNAL_DISPLAY_NAMES['model'].lower()}' logit "
            f"({logit:+.2f}). PROVISIONAL: see saliency_faithfulness.py for "
            "how well this agrees with occlusion."
        ),
        "heatmap_png_b64": _colourise(normalised, rgb.size),
        "grid": [[round(float(v), 4) for v in row] for row in normalised],
        "target_logit": logit,
        "span": span,
    }
