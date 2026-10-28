"""Where did the classifier look? Grad-CAM over the SwinV2 backbone.

HONEST FRAMING -- read this before putting a heatmap in front of anyone
-----------------------------------------------------------------------
A saliency overlay is the most persuasive thing this project can draw and the
easiest to overclaim with. Measured on this checkpoint, against occlusion as
ground truth, Grad-CAM at the final stage agreed with where the model is
actually sensitive at Spearman rho = 0.083 on a generated image. That is
essentially no agreement.

There is a structural reason, and it is itself the most interesting thing this
module produces. Against a baseline logit margin of 7.66, the single most
damaging 32x32 occlusion patch cost only 0.76 -- about 10%. **No single region
drives the verdict.** The model's evidence is spatially diffuse, which is what
you would expect if it keys on global texture statistics rather than on an
object. So the correct finding to report is "no localised region explains this
score", not "here is the region that did".

That is why this module has a dynamic-range gate and why it returns
available: False rather than rendering a flat map. It is the same rule the
fusion obeys -- an unavailable signal is declared, never faked.

Quote the faithfulness number from saliency_faithfulness.py wherever the
heatmap is shown. A heatmap without it is a picture of nothing, presented
confidently.

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
import logging
import math

import numpy as np
from PIL import Image

import config
from detectors import model_detector

logger = logging.getLogger(__name__)

# Which hidden state to attribute against. See WHY STAGE 2 above.
CAM_STAGE = 2

# If the normalised CAM's interquartile span falls below this, the map carries
# no usable structure and we decline to draw it. PROVISIONAL: chosen by
# inspection of a handful of images, not fitted. saliency_faithfulness.py is
# what should set it.
MIN_SPAN = 0.15

# Single-hue sequential ramp, light -> dark, with alpha carrying intensity so
# the unattributed majority of the frame stays transparent. One hue because
# this encodes magnitude, not category.
RAMP_RGB = (176, 42, 42)


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


def explain(image: Image.Image) -> dict:
    """Grad-CAM for the 'machine-generated' logit. Never raises."""
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
        return _unavailable(f"torch unavailable: {exc}")

    try:
        model = pipeline.model
        processor = pipeline.image_processor
        model.eval()

        rgb = image.convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt")

        # Grad-CAM needs gradients, so this forward pass is deliberately NOT
        # under no_grad: activations have to be retained.
        output = model(**inputs, output_hidden_states=True)
        states = output.hidden_states
        if states is None or len(states) <= CAM_STAGE:
            return _unavailable("Model did not expose the expected hidden states")

        activations = states[CAM_STAGE]            # (1, L, C)
        target = output.logits[0, ai_index]

        gradients = torch.autograd.grad(target, activations, retain_graph=False)[0]

        # Grad-CAM: channel weights are the spatially averaged gradients, then
        # a ReLU over the weighted sum keeps only evidence FOR the class.
        channel_weights = gradients.mean(dim=1, keepdim=True)      # (1, 1, C)
        cam = torch.relu((channel_weights * activations).sum(dim=-1))[0]  # (L,)

        length = cam.shape[0]
        side = int(math.isqrt(length))
        if side * side != length:
            return _unavailable(
                f"Hidden state at stage {CAM_STAGE} is not square ({length} positions)"
            )

        grid = cam.detach().numpy().reshape(side, side)
        logit = float(target.detach())
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
    # Interquartile span rather than max-min: a single hot cell against an
    # otherwise flat map is noise, and max-min would not notice.
    span = float(np.percentile(normalised, 90) - np.percentile(normalised, 10))

    if span < MIN_SPAN:
        return _unavailable(
            f"The attribution map is nearly flat (span {span:.3f} < {MIN_SPAN}). "
            "The model's evidence for this image is spatially diffuse, so no "
            "region explains the score. Reported rather than drawn."
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
